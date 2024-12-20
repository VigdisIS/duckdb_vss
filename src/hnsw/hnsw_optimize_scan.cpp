#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/optimizer/column_lifetime_analyzer.hpp"
#include "duckdb/optimizer/matcher/expression_matcher.hpp"
#include "duckdb/optimizer/optimizer_extension.hpp"
#include "duckdb/optimizer/remove_unused_columns.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_top_n.hpp"
#include "duckdb/storage/data_table.hpp"
#include "hnsw/hnsw.hpp"
#include "hnsw/hnsw_index.hpp"
#include "hnsw/hnsw_index_scan.hpp"

// #include <chrono>
// #include <fstream>
// #include <nlohmann/json.hpp>
// #include <iostream>

namespace duckdb {

//-----------------------------------------------------------------------------
// Plan rewriter
//-----------------------------------------------------------------------------
class HNSWIndexScanOptimizer : public OptimizerExtension {
public:
	HNSWIndexScanOptimizer() {
		optimize_function = Optimize;
	}

	static bool TryOptimize(ClientContext &context, unique_ptr<LogicalOperator> &plan) {
		// auto total_start = std::chrono::high_resolution_clock::now();
		
		// Look for a TopN operator
		auto &op = *plan;

		if (op.type != LogicalOperatorType::LOGICAL_TOP_N) {
			return false;
		}

		auto &top_n = op.Cast<LogicalTopN>();

		if (top_n.orders.size() != 1) {
			// We can only optimize if there is a single order by expression right now
			return false;
		}

		const auto &order = top_n.orders[0];

		if (order.type != OrderType::ASCENDING) {
			// We can only optimize if the order by expression is ascending
			return false;
		}

		if (order.expression->type != ExpressionType::BOUND_COLUMN_REF) {
			// The expression has to reference the child operator (a projection with the distance function)
			return false;
		}
		const auto &bound_column_ref = order.expression->Cast<BoundColumnRefExpression>();

		// find the expression that is referenced
		if (top_n.children.size() != 1 || top_n.children.front()->type != LogicalOperatorType::LOGICAL_PROJECTION) {
			// The child has to be a projection
			return false;
		}

		auto &projection = top_n.children.front()->Cast<LogicalProjection>();

		// This the expression that is referenced by the order by expression
		const auto projection_index = bound_column_ref.binding.column_index;
		const auto &projection_expr = projection.expressions[projection_index];

		// The projection must sit on top of a get
		if (projection.children.size() != 1 || projection.children.front()->type != LogicalOperatorType::LOGICAL_GET) {
			return false;
		}

		auto &get_ptr = projection.children.front();
		auto &get = get_ptr->Cast<LogicalGet>();
		// Check if the get is a table scan
		if (get.function.name != "seq_scan") {
			return false;
		}

		if (get.dynamic_filters && get.dynamic_filters->HasFilters()) {
			// Cant push down!
			return false;
		}

		// We have a top-n operator on top of a table scan
		// We can replace the function with a custom index scan (if the table has a custom index)

		// Get the table
		auto &table = *get.GetTable();
		if (!table.IsDuckTable()) {
			// We can only replace the scan if the table is a duck table
			return false;
		}

		auto &duck_table = table.Cast<DuckTableEntry>();
		auto &table_info = *table.GetStorage().GetDataTableInfo();

		// Find the index
		unique_ptr<HNSWIndexScanBindData> bind_data = nullptr;
		vector<reference<Expression>> bindings;

		// first do lambda func to return key from centroid index
		// Then do another lambda func to find the matching index
		// Then set the bind_data to the matching index

		unum::usearch::misaligned_ref_gt<const duckdb::row_t> key = nullptr;

		// auto chrono_starts = std::chrono::high_resolution_clock::now();

		table_info.GetIndexes().BindAndScan<HNSWIndex>(context, table_info, [&](HNSWIndex &hnsw_index) {
			// From this, first search 'centroid_index' where k = 1
			// then output which cluster to perform search in
			// This becomes the new hnsw_index variable

			// Seems to by default pick first HNSW index available, make sure this is the centroid index
			if (hnsw_index.GetIndexName() != "centroid_index") {
				return false;
			}

			// Reset the bindings
			bindings.clear();

			// Check that the projection expression is a distance function that matches the index
			if (!hnsw_index.TryMatchDistanceFunction(projection_expr, bindings)) {
				return false;
			}
			// Check that the HNSW index actually indexes the expression
			unique_ptr<Expression> index_expr;
			if (!hnsw_index.TryBindIndexExpression(get, index_expr)) {
				return false;
			}

			// Now, ensure that one of the bindings is a constant vector, and the other our index expression
			auto &const_expr_ref = bindings[1];
			auto &index_expr_ref = bindings[2];

			if (const_expr_ref.get().type != ExpressionType::VALUE_CONSTANT || !index_expr->Equals(index_expr_ref)) {
				// Swap the bindings and try again
				std::swap(const_expr_ref, index_expr_ref);
				if (const_expr_ref.get().type != ExpressionType::VALUE_CONSTANT ||
				    !index_expr->Equals(index_expr_ref)) {
					// Nope, not a match, we can't optimize.
					return false;
				}
			}

			const auto vector_size = hnsw_index.GetVectorSize();
			const auto &matched_vector = const_expr_ref.get().Cast<BoundConstantExpression>().value;
			auto query_vector = make_unsafe_uniq_array<float>(vector_size);
			auto vector_elements = ArrayValue::GetChildren(matched_vector);
			for (idx_t i = 0; i < vector_size; i++) {
				query_vector[i] = vector_elements[i].GetValue<float>();
			}

			// auto ef_search = hnsw_index.index.expansion_search();
			auto search_result = hnsw_index.index.search(query_vector.get(), 1);

			auto centroid = search_result[0];

			key = centroid.member.key;

			return true;
		});

		// auto chrono_ends = std::chrono::high_resolution_clock::now();
		// auto chrono_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(chrono_ends - chrono_starts);

		// std::cout << "Time spent searching centroid index: " << chrono_duration.count() << " nanoseconds" << std::endl;

		// nlohmann::json idx_searches = nlohmann::json::array();

		// idx_searches.push_back({
		// 					{"index", "centroid_index"},
		// 					{"duration (ns)", chrono_duration.count()}
		// 				});

		// auto chrono_start_second = std::chrono::high_resolution_clock::now();

		// std::string last_index;

		table_info.GetIndexes().BindAndScan<HNSWIndex>(context, table_info, [&](HNSWIndex &inner_index) {
			if (inner_index.GetIndexName() != std::to_string(key)) {
				return false;
			}

			// last_index = inner_index.GetIndexName();
 						// Reset the bindings
 						bindings.clear();
 						// Check that the projection expression is a distance function that matches the index
 						if (!inner_index.TryMatchDistanceFunction(projection_expr, bindings)) {
 							return false;
 						}
 						// Check that the HNSW index actually indexes the expression
 						unique_ptr<Expression> index_expr;
 						if (!inner_index.TryBindIndexExpression(get, index_expr)) {
 							return false;
 						}
 
 						auto &const_expr_ref = bindings[1];
 						auto &index_expr_ref = bindings[2];
 
 						if (const_expr_ref.get().type != ExpressionType::VALUE_CONSTANT ||
 						    !index_expr->Equals(index_expr_ref)) {
 							// Swap the bindings and try again
 							std::swap(const_expr_ref, index_expr_ref);
 							if (const_expr_ref.get().type != ExpressionType::VALUE_CONSTANT ||
 							    !index_expr->Equals(index_expr_ref)) {
 								// Nope, not a match, we can't optimize.
 								return false;
 							}
 						}
 
 						const auto vector_size = inner_index.GetVectorSize();
 						const auto &matched_vector = const_expr_ref.get().Cast<BoundConstantExpression>().value;
 						auto query_vector = make_unsafe_uniq_array<float>(vector_size);
 						auto vector_elements = ArrayValue::GetChildren(matched_vector);
 						for (idx_t i = 0; i < vector_size; i++) {
 							query_vector[i] = vector_elements[i].GetValue<float>();
 						}
 
 						bind_data = make_uniq<HNSWIndexScanBindData>(duck_table, inner_index, top_n.limit,
 						                                             std::move(query_vector));
			return true;
		});

		// auto chrono_end_second = std::chrono::high_resolution_clock::now();
		// auto chrono_duration_second = std::chrono::duration_cast<std::chrono::nanoseconds>(chrono_end_second - chrono_start_second);

		// std::cout << "Time spent searching cluster index " << last_index << ": " << chrono_duration_second.count() << " nanoseconds" << std::endl;

		// idx_searches.push_back({
		// 					{"index", last_index},
		// 					{"duration (ns)", chrono_duration_second.count()}
		// 				});

	
			// // Open a file in append mode.
    		// std::ofstream outFile("cluster_indexes.txt");

			// if (outFile.is_open()) {
			// 	// Write the string_val to the file
			// 	outFile << last_index;

			// 	// Close the file after writing
			// 	outFile.close();
			// } else {
			// 	std::cerr << "Failed to open file for writing." << std::endl;
			// }

		if (!bind_data) {
			// No index found
			return false;
		}

		// If there are no table filters pushed down into the get, we can just replace the get with the index scan
		const auto cardinality = get.function.cardinality(context, bind_data.get());
		get.function = HNSWIndexScanFunction::GetFunction();
		get.has_estimated_cardinality = cardinality->has_estimated_cardinality;
		get.estimated_cardinality = cardinality->estimated_cardinality;
		get.bind_data = std::move(bind_data);
		if (get.table_filters.filters.empty()) {

			// Remove the TopN operator
			plan = std::move(top_n.children[0]);

			// auto total_end = std::chrono::high_resolution_clock::now();
			// auto total_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(total_end - total_start);

			// std::cout << "Total time spent TryOptimize: " << total_duration.count() << " nanoseconds" << std::endl;

			// // Full try optimize
			// nlohmann::json jsonOutputS;
			
			// std::ifstream inputFileTimeS("clustering_time_operations_search_bm.json");
			
			// if (inputFileTimeS.good()) {
			// 	inputFileTimeS >> jsonOutputS;
			// 	inputFileTimeS.close();
			// }

			// nlohmann::json newSearch;

			// newSearch["dataset"] = table_info.GetTableName();
			// newSearch["cluster_amount"] = table_info.GetIndexes().Count() - 1;
			// newSearch["total_duration (ns)"] = total_duration.count();
			// newSearch["searches"] = idx_searches;

			// jsonOutputS.push_back(newSearch);

			// std::ofstream outputFile("clustering_time_operations_search_bm.json");
			// outputFile << jsonOutputS.dump(4); // Pretty-printing with 4 spaces indent
			// outputFile.close();
			
			return true;
		}

		// Otherwise, things get more complicated. We need to pullup the filters from the table scan as our index scan
		// does not support regular filter pushdown.
		get.projection_ids.clear();
		get.types.clear();

		auto new_filter = make_uniq<LogicalFilter>();
		auto &column_ids = get.GetColumnIds();
		for (const auto &entry : get.table_filters.filters) {
			idx_t column_id = entry.first;
			auto &type = get.returned_types[column_id];
			bool found = false;
			for (idx_t i = 0; i < column_ids.size(); i++) {
				if (column_ids[i] == column_id) {
					column_id = i;
					found = true;
					break;
				}
			}
			if (!found) {
				throw InternalException("Could not find column id for filter");
			}
			auto column = make_uniq<BoundColumnRefExpression>(type, ColumnBinding(get.table_index, column_id));
			new_filter->expressions.push_back(entry.second->ToExpression(*column));
		}
		new_filter->children.push_back(std::move(get_ptr));
		new_filter->ResolveOperatorTypes();
		get_ptr = std::move(new_filter);

		// Remove the TopN operator
		plan = std::move(top_n.children[0]);
		return true;
	}

	static bool OptimizeChildren(ClientContext &context, unique_ptr<LogicalOperator> &plan) {

		auto ok = TryOptimize(context, plan);
		// Recursively optimize the children
		for (auto &child : plan->children) {
			ok |= OptimizeChildren(context, child);
		}
		return ok;
	}

	static void MergeProjections(unique_ptr<LogicalOperator> &plan) {
		if (plan->type == LogicalOperatorType::LOGICAL_PROJECTION) {
			if (plan->children[0]->type == LogicalOperatorType::LOGICAL_PROJECTION) {
				auto &child = plan->children[0];

				if (child->children[0]->type == LogicalOperatorType::LOGICAL_GET &&
				    child->children[0]->Cast<LogicalGet>().function.name == "hnsw_index_scan") {
					auto &parent_projection = plan->Cast<LogicalProjection>();
					auto &child_projection = child->Cast<LogicalProjection>();

					column_binding_set_t referenced_bindings;
					for (auto &expr : parent_projection.expressions) {
						ExpressionIterator::EnumerateExpression(expr, [&](Expression &expr_ref) {
							if (expr_ref.type == ExpressionType::BOUND_COLUMN_REF) {
								auto &bound_column_ref = expr_ref.Cast<BoundColumnRefExpression>();
								referenced_bindings.insert(bound_column_ref.binding);
							}
						});
					}

					auto child_bindings = child_projection.GetColumnBindings();
					for (idx_t i = 0; i < child_projection.expressions.size(); i++) {
						auto &expr = child_projection.expressions[i];
						auto &outgoing_binding = child_bindings[i];

						if (referenced_bindings.find(outgoing_binding) == referenced_bindings.end()) {
							// The binding is not referenced
							// We can remove this expression. But positionality matters so just replace with int.
							expr = make_uniq_base<Expression, BoundConstantExpression>(Value(LogicalType::TINYINT));
						}
					}
					return;
				}
			}
		}
		for (auto &child : plan->children) {
			MergeProjections(child);
		}
	}

	static void Optimize(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
		auto did_use_hnsw_scan = OptimizeChildren(input.context, plan);
		if (did_use_hnsw_scan) {
			MergeProjections(plan);
		}
	}
};

//-----------------------------------------------------------------------------
// Register
//-----------------------------------------------------------------------------
void HNSWModule::RegisterScanOptimizer(DatabaseInstance &db) {
	// Register the optimizer extension
	db.config.optimizer_extensions.push_back(HNSWIndexScanOptimizer());
}

} // namespace duckdb
