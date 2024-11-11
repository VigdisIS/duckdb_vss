#include "hnsw/hnsw_index_physical_create.hpp"

#include "duckdb/catalog/catalog_entry/duck_index_entry.hpp"
#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/exception/transaction_exception.hpp"
#include "duckdb/main/attached_database.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/storage/storage_manager.hpp"
#include "duckdb/storage/table_io_manager.hpp"
#include "hnsw/hnsw_index.hpp"
#include <iostream>
#include "usearch/duckdb_usearch.hpp"
#include <vector>
#include <string>
#include <unordered_map>

#include "duckdb/parallel/base_pipeline_event.hpp"

using some_scalar_t = float;

namespace duckdb {

PhysicalCreateHNSWIndex::PhysicalCreateHNSWIndex(LogicalOperator &op, TableCatalogEntry &table_p,
                                                 const vector<column_t> &column_ids, unique_ptr<CreateIndexInfo> info,
                                                 vector<unique_ptr<Expression>> unbound_expressions,
                                                 idx_t estimated_cardinality)
    // Declare this operators as a EXTENSION operator
    : PhysicalOperator(PhysicalOperatorType::EXTENSION, op.types, estimated_cardinality),
      table(table_p.Cast<DuckTableEntry>()), info(std::move(info)), unbound_expressions(std::move(unbound_expressions)),
      sorted(false) {

	// convert virtual column ids to storage column ids
	for (auto &column_id : column_ids) {
		storage_ids.push_back(table.GetColumns().LogicalToPhysical(LogicalIndex(column_id)).index);
	};
}

// Define a custom smart iterator template
template<typename T>
class SmartIterator {
public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using reference = T&;

    // Constructor: Accepts an iterator of std::vector<std::vector<T>>
    SmartIterator(typename std::vector<std::vector<T>>::iterator ptr, std::size_t level)
        : ptr_(ptr), cluster_level_(level) {}

    // Dereference operator: Returns a raw pointer to the underlying data of the inner vector
    pointer operator*() const { return ptr_->data(); }

    // Arrow operator: Returns a raw pointer to the underlying data of the inner vector
    pointer operator->() { return ptr_->data(); }

    // Pre-increment operator
    SmartIterator& operator++() {
        deepen_into_cluster();
        ++ptr_;
        return *this;
    }

    // Post-increment operator
    SmartIterator operator++(int) {
        SmartIterator temp = *this;
        ++(*this);
        return temp;
    }

    // Pre-decrement operator
    SmartIterator& operator--() {
        --ptr_;
        return *this;
    }

    // Post-decrement operator
    SmartIterator operator--(int) {
        SmartIterator temp = *this;
        --(*this);
        return temp;
    }

    // Difference operator (calculate the distance between two iterators)
    difference_type operator-(const SmartIterator& other) const {
        return ptr_ - other.ptr_;
    }

    // Random access operator (it + n): returns a raw pointer to the data of the nth inner vector
    pointer operator+(difference_type n) const {
        return (ptr_ + n)->data();  // Return raw pointer to nth inner vector's data
    }

    // Random access operator (it[n]): return raw pointer to data of nth vector
    pointer operator[](difference_type n) const {
        return (ptr_ + n)->data();  // Return raw pointer to nth inner vector's data
    }

    // Equality comparison
    bool operator==(const SmartIterator& other) const { return ptr_ == other.ptr_; }

    // Inequality comparison
    bool operator!=(const SmartIterator& other) const { return !(*this == other); }

    // Custom logic for deepening into a cluster
    void deepen_into_cluster() {
        if (cluster_level_ > 0) {
            std::cout << "Deepening into cluster at level: " << cluster_level_ << std::endl;
            --cluster_level_;
        }
    }

private:
    typename std::vector<std::vector<T>>::iterator ptr_;  // Iterator for std::vector<std::vector<float>>
    std::size_t cluster_level_;  // Simulating the level for deepening into clusters
};

// Custom container class to manage dataset and iterators
template<typename T>
class SmartIterable {
public:
    using iterator = SmartIterator<T>;

    // Constructor takes in the dataset and the depth level
    SmartIterable(std::vector<std::vector<T>>& data, std::size_t depth_level)
        : data_(data), depth_level_(depth_level) {}

    // Return a smart iterator for the beginning of the data
    iterator begin() { return iterator(data_.begin(), depth_level_); }

    // Return a smart iterator for the end of the data
    iterator end() { return iterator(data_.end(), depth_level_); }

private:
    std::vector<std::vector<T>>& data_;   // Reference to the dataset (vector of vectors)
    std::size_t depth_level_; // Depth level to manage cluster traversal
};

//-------------------------------------------------------------
// Global State
//-------------------------------------------------------------
class CreateHNSWIndexGlobalState final : public GlobalSinkState {
public:
	CreateHNSWIndexGlobalState(const PhysicalOperator &op_p) : op(op_p) {
	}

	const PhysicalOperator &op;
	//! Centroid index to be added to the table
	unique_ptr<HNSWIndex> centroid_index;
	unique_ptr<HNSWIndex> dummy_index;

	// Cluster indexes
	std::unordered_map<unum::usearch::default_key_t, std::unique_ptr<HNSWIndex>> cluster_indexes;
	// Cluster indexes + centroid index
	std::unordered_map<std::string, std::unique_ptr<HNSWIndex>> all_indexes;

	mutex glock;
	unique_ptr<ColumnDataCollection> collection;
	shared_ptr<ClientContext> context;

	// Table manager
	TableIOManager *table_manager;
	
	duckdb::vector<duckdb::column_t> storage_ids;

	// Storage
	DataTable *storage;

	// Estimated cardinality
	idx_t estimated_cardinality;

	//
	duckdb::vector<duckdb::unique_ptr<duckdb::Expression, std::__1::default_delete<duckdb::Expression>, true>, true> unbound_expressions;

	// Info
	CreateIndexInfo *info;

	vector<LogicalType> data_types;

	std::vector<std::vector<float>> train_data;

	// Parallel scan state
	ColumnDataParallelScanState scan_state;

	// Track which phase we're in
	atomic<bool> is_building = {false};
	atomic<idx_t> loaded_count = {0};
	atomic<idx_t> built_count = {0};
};

unique_ptr<GlobalSinkState> PhysicalCreateHNSWIndex::GetGlobalSinkState(ClientContext &context) const {
	auto gstate = make_uniq<CreateHNSWIndexGlobalState>(*this);
	vector<LogicalType> data_types = {unbound_expressions[0]->return_type, LogicalType::ROW_TYPE};
	gstate->collection = make_uniq<ColumnDataCollection>(BufferManager::GetBufferManager(context), data_types);
	gstate->context = context.shared_from_this();

	// Create the index
	auto &storage = table.GetStorage();
	auto &table_manager = TableIOManager::Get(storage);
	auto &constraint_type = info->constraint_type;
	auto &db = storage.db;

	// Input to HNSW constructor class which we move from this step to the next,
	// so we bring with us the correct input which is available at this step
	gstate->table_manager = &table_manager;
	gstate->storage = &storage;
	gstate->estimated_cardinality = estimated_cardinality;
	gstate->info = info.get();
	gstate->data_types = data_types;
	gstate->storage_ids = storage_ids;
	
	std::vector<std::vector<float>> train_data;

	// TODO:
	// For some reason the tasks after this global state will not be executed
	// unless we add a dummy index to the all_indexes map. This dummy index
	// is removed from the map in the ExecuteTask step
	gstate->dummy_index =
	    make_uniq<HNSWIndex>("dummy_index", constraint_type, storage_ids, table_manager, unbound_expressions, db,
	                         info->options, IndexStorageInfo(), estimated_cardinality);

	// Initialize centroid index
	gstate->centroid_index =
	    make_uniq<HNSWIndex>("centroid_index", constraint_type, storage_ids, table_manager, unbound_expressions, db,
	                         info->options, IndexStorageInfo(), estimated_cardinality);

	gstate->all_indexes.emplace("dummy_index", std::move(gstate->dummy_index));

	// duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> destination;
    for (const auto& expr : unbound_expressions) {
        // Use the Copy() method to clone each Expression
        gstate->unbound_expressions.push_back(expr->Copy());
    }

	return std::move(gstate);
}

//-------------------------------------------------------------
// Local State
//-------------------------------------------------------------
class CreateHNSWIndexLocalState final : public LocalSinkState {
public:
	unique_ptr<ColumnDataCollection> collection;
	ColumnDataAppendState append_state;
};

unique_ptr<LocalSinkState> PhysicalCreateHNSWIndex::GetLocalSinkState(ExecutionContext &context) const {
	auto state = make_uniq<CreateHNSWIndexLocalState>();

	vector<LogicalType> data_types = {unbound_expressions[0]->return_type, LogicalType::ROW_TYPE};
	state->collection = make_uniq<ColumnDataCollection>(BufferManager::GetBufferManager(context.client), data_types);
	state->collection->InitializeAppend(state->append_state);
	return std::move(state);
}

//-------------------------------------------------------------
// Sink
//-------------------------------------------------------------

SinkResultType PhysicalCreateHNSWIndex::Sink(ExecutionContext &context, DataChunk &chunk,
                                             OperatorSinkInput &input) const {

	auto &lstate = input.local_state.Cast<CreateHNSWIndexLocalState>();
	auto &gstate = input.global_state.Cast<CreateHNSWIndexGlobalState>();
	lstate.collection->Append(lstate.append_state, chunk);
	gstate.loaded_count += chunk.size();
	return SinkResultType::NEED_MORE_INPUT;
}

//-------------------------------------------------------------
// Combine
//-------------------------------------------------------------
SinkCombineResultType PhysicalCreateHNSWIndex::Combine(ExecutionContext &context,
                                                       OperatorSinkCombineInput &input) const {
	auto &gstate = input.global_state.Cast<CreateHNSWIndexGlobalState>();
	auto &lstate = input.local_state.Cast<CreateHNSWIndexLocalState>();

	if (lstate.collection->Count() == 0) {
		return SinkCombineResultType::FINISHED;
	}

	lock_guard<mutex> l(gstate.glock);
	if (!gstate.collection) {
		gstate.collection = std::move(lstate.collection);
	} else {
		gstate.collection->Combine(*lstate.collection);
	}

	return SinkCombineResultType::FINISHED;
}

//-------------------------------------------------------------
// Finalize
//-------------------------------------------------------------

class HNSWIndexConstructTask final : public ExecutorTask {

public:
    HNSWIndexConstructTask(shared_ptr<Event> event_p, ClientContext &context, CreateHNSWIndexGlobalState &gstate_p,
                           size_t thread_id_p, const PhysicalCreateHNSWIndex &op_p)
        : ExecutorTask(context, std::move(event_p), op_p), gstate(gstate_p), thread_id(thread_id_p),
          local_scan_state() {
		// Initialize the scan chunk
		gstate.collection->InitializeScanChunk(scan_chunk);
	}

	TaskExecutionResult ExecuteTask(TaskExecutionMode mode) override {

	// Remove dummy index
	gstate.all_indexes.erase("dummy_index");

	auto &scan_state = gstate.scan_state;
    auto &collection = gstate.collection;

    const auto array_size = ArrayType::GetSize(scan_chunk.data[0].GetType());

	// Iterate over the data and construct the index
    while (collection->Scan(scan_state, local_scan_state, scan_chunk)) {
        const auto count = scan_chunk.size();
        auto &vec_vec = scan_chunk.data[0];
        auto &data_vec = ArrayVector::GetEntry(vec_vec);
        auto &rowid_vec = scan_chunk.data[1];

        UnifiedVectorFormat vec_format;
        UnifiedVectorFormat data_format;
        UnifiedVectorFormat rowid_format;

        vec_vec.ToUnifiedFormat(count, vec_format);
        data_vec.ToUnifiedFormat(count * array_size, data_format);
        rowid_vec.ToUnifiedFormat(count, rowid_format);

        const auto row_ptr = UnifiedVectorFormat::GetData<row_t>(rowid_format);
        const auto data_ptr = UnifiedVectorFormat::GetData<float>(data_format);

        for (idx_t i = 0; i < count; i++) {
            const auto vec_idx = vec_format.sel->get_index(i);
            const auto row_idx = rowid_format.sel->get_index(i);

            // Check for NULL values
            const auto vec_valid = vec_format.validity.RowIsValid(vec_idx);
            const auto rowid_valid = rowid_format.validity.RowIsValid(row_idx);
            if (!vec_valid || !rowid_valid) {
                executor.PushError(
                    ErrorData("Invalid data in HNSW index construction: Cannot construct index with NULL values."));
                return TaskExecutionResult::TASK_ERROR;
            }

			// Create a vector from the data pointed to by data_ptr for this row
			std::vector<float> current_vector;
			for (size_t j = 0; j < array_size; ++j) {
				float value = *(data_ptr + (vec_idx * array_size) + j); // Access the correct element
				current_vector.push_back(value);
			}

			// Insert the constructed vector into train_data
			gstate.train_data.push_back(current_vector);
        }
		std::cout << "Added all vectors to train_data. Size is " << gstate.train_data.size() << std::endl;
		// Update the built count
		gstate.built_count += count;

			if (mode == TaskExecutionMode::PROCESS_PARTIAL) {
				// yield!
				return TaskExecutionResult::TASK_NOT_FINISHED;
			}
	}


	std::cout << "Size of train_data: " << gstate.train_data.size() << std::endl;

	auto data_types = gstate.data_types;

	std::vector<std::vector<float>> train_data = gstate.train_data;

	// Create the index
	auto &table_manager = gstate.table_manager;
	auto &constraint_type = gstate.info->constraint_type;
	auto &db = gstate.storage->db;
	auto &unbound_expressions = gstate.unbound_expressions;

	std::cout << "Define clustering configuration...\n";
    // Define clustering configuration
	// TODO: Make this an input to the CREATE INDEX statement
    unum::usearch::index_dense_clustering_config_t config;
    config.min_clusters = 5;   // Minimum number of clusters
    config.max_clusters = 100;   // Maximum number of clusters
    config.mode = unum::usearch::index_dense_clustering_config_t::merge_closest_k;

    std::cout << "Arrays to hold clustering results...\n";

    std::size_t depth_level = 10;    // Cluster deepening level

    SmartIterable<some_scalar_t> iterable(train_data, depth_level);

    std::size_t vector_dimension = train_data[0].size();
    std::size_t dataset_size = train_data.size();

	// Arrays to hold clustering results
    unum::usearch::index_dense_t::vector_key_t* cluster_centroids_keys = new unum::usearch::index_dense_t::vector_key_t[MinValue(static_cast<idx_t>(dataset_size), gstate.estimated_cardinality)];
    unum::usearch::default_distance_t* distances_to_cluster_centroids = new unum::usearch::default_distance_t[MinValue(static_cast<idx_t>(dataset_size), gstate.estimated_cardinality)];

    // Define the metric
    unum::usearch::metric_punned_t metric(vector_dimension, unum::usearch::metric_kind_t::l2sq_k, unum::usearch::scalar_kind_t::f32_k);

    // Create the index
    unum::usearch::index_dense_t index = unum::usearch::index_dense_t::make(metric);
    index.reserve(MinValue(static_cast<idx_t>(dataset_size), gstate.estimated_cardinality));

	for (std::size_t i = 0; i < train_data.size(); ++i) {
        index.add(i, train_data[i].data());
    }

	index.reserve(MinValue(static_cast<idx_t>(dataset_size), gstate.estimated_cardinality));

    std::cout << "Performing clustering...\n";

    // Perform clustering
    unum::usearch::index_dense_t::clustering_result_t result = index.cluster(
        iterable.begin(), iterable.end(),
        config,
        cluster_centroids_keys,  distances_to_cluster_centroids
    );

	std::cout << "Finished clustering.\n";

	int cluster_centroid_size = 0;

    for (int i = 0; i < gstate.estimated_cardinality; ++i) {
        if (cluster_centroids_keys[i] != -1) {
            cluster_centroid_size++;
        }
    }

    std::cout << "Cluster centroid size: " << cluster_centroid_size << std::endl;

    std::cout << "Adding cluster indexes\n";
    for (int index_position = 0; index_position < gstate.estimated_cardinality; ++index_position) {

         // Ensure index_position is within bounds
        if (index_position >= train_data.size() || index_position >= cluster_centroid_size) {
            std::cerr << "Index out of bounds: index_position = " << index_position << std::endl;
            continue; // Skip this iteration if out of bounds
        }

        auto centroid_key = cluster_centroids_keys[index_position];

        if (centroid_key >= train_data.size()) {
            std::cerr << "Centroid key out of bounds: centroid_key = " << centroid_key << std::endl;
            continue;
        }

        auto it = gstate.cluster_indexes.find(centroid_key);
        if (it != gstate.cluster_indexes.end()) {
        	// If the key exists, add the data to the existing index
			std::cout << "Adding index to centroid key " << centroid_key << std::endl;
			it->second->AddToIndex(index_position, train_data[index_position].data());
			std::string keyAsString = std::to_string(centroid_key);
			auto all_it = gstate.all_indexes.find(keyAsString);
			all_it->second->AddToIndex(index_position, train_data[index_position].data());
        } else {
            // If not, create a centroid index for each cluster_centroid
            std::cout << "Creating new centroid index for centroid key: " << centroid_key << std::endl;
			gstate.centroid_index->AddToIndex(centroid_key, train_data[centroid_key].data());
			std::string numberAsString = std::to_string(centroid_key);
			// TODO: Make shared ptr instead of two instances of unique lol
			auto cluster_index_for_cluster =
				make_uniq<HNSWIndex>(numberAsString, constraint_type, gstate.storage_ids, *table_manager, unbound_expressions, db,
									gstate.info->options, IndexStorageInfo(), gstate.estimated_cardinality);
			auto cluster_index_for_all =
				make_uniq<HNSWIndex>(numberAsString, constraint_type, gstate.storage_ids, *table_manager, unbound_expressions, db,
									gstate.info->options, IndexStorageInfo(), gstate.estimated_cardinality);

			cluster_index_for_cluster->AddToIndex(index_position, train_data[index_position].data());
			cluster_index_for_all->AddToIndex(index_position, train_data[index_position].data());

            // Insert the new cluster index into the map
            gstate.cluster_indexes.emplace(centroid_key, std::move(cluster_index_for_cluster));
			gstate.all_indexes.emplace(numberAsString, std::move(cluster_index_for_all));
        }
    }

	std::cout << "Finished adding cluster indexes\n";

	gstate.all_indexes.emplace("centroid_index", std::move(gstate.centroid_index));

    // Finish task!
    event->FinishTask();
    return TaskExecutionResult::TASK_FINISHED;
	}


private:
	CreateHNSWIndexGlobalState &gstate;
	size_t thread_id;

	DataChunk scan_chunk;
	ColumnDataLocalScanState local_scan_state;
};

class HNSWIndexConstructionEvent final : public BasePipelineEvent {
public:
	HNSWIndexConstructionEvent(const PhysicalCreateHNSWIndex &op_p, CreateHNSWIndexGlobalState &gstate_p,
	                           Pipeline &pipeline_p, CreateIndexInfo &info_p, const vector<column_t> &storage_ids_p,
	                           DuckTableEntry &table_p)
	    : BasePipelineEvent(pipeline_p), op(op_p), gstate(gstate_p), info(info_p), storage_ids(storage_ids_p),
	      table(table_p) {
	}

	const PhysicalCreateHNSWIndex &op;
	CreateHNSWIndexGlobalState &gstate;
	CreateIndexInfo &info;
	const vector<column_t> &storage_ids;
	DuckTableEntry &table;

public:
	void Schedule() override {
		auto &context = pipeline->GetClientContext();

		// Schedule tasks equal to the number of threads, which will construct the index
		auto &ts = TaskScheduler::GetScheduler(context);
		const auto num_threads = NumericCast<size_t>(ts.NumberOfThreads());

		vector<shared_ptr<Task>> construct_tasks;
		for (size_t tnum = 0; tnum < num_threads; tnum++) {
			construct_tasks.push_back(make_uniq<HNSWIndexConstructTask>(shared_from_this(), context, gstate, tnum, op));
		}
		SetTasks(std::move(construct_tasks));
	}

	void FinishEvent() override {

		// Add all cluster indexes to table
		for (auto &cluster_index : gstate.all_indexes) {
			auto &index = cluster_index.second;
			if(index == nullptr) {
				continue;
			}
				auto &index_name = cluster_index.first;

				// Mark the index as dirty, update its count
				index->SetDirty();
				index->SyncSize();

				auto &storage = table.GetStorage();

				// If not in memory, persist the index to disk
				if (!storage.db.GetStorageManager().InMemory()) {
					// Finalize the index
					index->PersistToDisk();
				}

				if (!storage.IsRoot()) {
					throw TransactionException("Cannot create index on non-root transaction");
				}

				// Create the index entry in the catalog

				auto &schema = table.schema;
				info.column_ids = storage_ids;
				info.index_name = index_name;

				if (schema.GetEntry(schema.GetCatalogTransaction(*gstate.context), CatalogType::INDEX_ENTRY, info.index_name)) {
					if (info.on_conflict != OnCreateConflict::IGNORE_ON_CONFLICT) {
						std::cout << "Index w name alr exists" << std::endl;
						throw CatalogException("Index with name \"%s\" already exists", info.index_name.c_str());
					}
				}

				const auto index_entry = schema.CreateIndex(schema.GetCatalogTransaction(*gstate.context), info, table).get();
				D_ASSERT(index_entry);
				auto &duck_index = index_entry->Cast<DuckIndexEntry>();
				duck_index.initial_index_size = index->Cast<BoundIndex>().GetInMemorySize();

				// Finally add it to storage
				storage.AddIndex(std::move(index));
		}

	}
};

// SinkFinalizeType PhysicalCreateHNSWIndex::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
//                                                    OperatorSinkFinalizeInput &input) const {

// 	// Get the global collection we've been appending to
// 	auto &gstate = input.global_state.Cast<CreateHNSWIndexGlobalState>();
// 	auto &collection = gstate.collection;

// 	// Move on to the next phase
// 	gstate.is_building = true;

// 	// Reserve the index size
// 	auto &ts = TaskScheduler::GetScheduler(context);
// 	auto &index = gstate.global_index->index;
// 	index.reserve({static_cast<size_t>(collection->Count()), static_cast<size_t>(ts.NumberOfThreads())});

// 	// Initialize a parallel scan for the index construction
// 	collection->InitializeScan(gstate.scan_state, ColumnDataScanProperties::ALLOW_ZERO_COPY);

// 	// Create a new event that will construct the index
// 	auto new_event = make_shared_ptr<HNSWIndexConstructionEvent>(*this, gstate, pipeline, *info, storage_ids, table);
// 	event.InsertEvent(std::move(new_event));

// 	return SinkFinalizeType::READY;
// }

SinkFinalizeType PhysicalCreateHNSWIndex::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                                   OperatorSinkFinalizeInput &input) const {
    // Get the global collection we've been appending to
    auto &gstate = input.global_state.Cast<CreateHNSWIndexGlobalState>();
    auto &collection = gstate.collection;

    // Move on to the next phase
    gstate.is_building = true;

	// // Centroid_index

	// // Reserve the index size
	// auto &ts = TaskScheduler::GetScheduler(context);
	// auto &index = gstate.centroid_index->index;
	// index.reserve({static_cast<size_t>(collection->Count()), static_cast<size_t>(ts.NumberOfThreads())});

	// // Initialize a parallel scan for the index construction
	// collection->InitializeScan(gstate.scan_state, ColumnDataScanProperties::ALLOW_ZERO_COPY);

	// // Create a new event that will construct the index
	// auto new_event = make_shared_ptr<HNSWIndexConstructionEvent>(*this, gstate, pipeline, *info, storage_ids, table);
	// event.InsertEvent(std::move(new_event));

    // Loop through each cluster index
    for (auto &cluster_index : gstate.all_indexes) {
        auto &index = cluster_index.second->index; // Assuming HNSWIndex has a member named 'index'

        // Reserve the index size
        auto &ts = TaskScheduler::GetScheduler(context);
        index.reserve({static_cast<size_t>(collection->Count()), static_cast<size_t>(ts.NumberOfThreads())});

        // Initialize a parallel scan for the index construction
        collection->InitializeScan(gstate.scan_state, ColumnDataScanProperties::ALLOW_ZERO_COPY);

        // Create a new event that will construct the index for the current cluster_index
        auto new_event = make_shared_ptr<HNSWIndexConstructionEvent>(*this, gstate, pipeline, *info, storage_ids, table);
        event.InsertEvent(std::move(new_event));
    }

    return SinkFinalizeType::READY;
}

double PhysicalCreateHNSWIndex::GetSinkProgress(ClientContext &context, GlobalSinkState &gstate,
                                                double source_progress) const {
	// The "source_progress" is not relevant for CREATE INDEX statements
	const auto &state = gstate.Cast<CreateHNSWIndexGlobalState>();
	// First half of the progress is appending to the collection
	if (!state.is_building) {
		return 50.0 *
		       MinValue(1.0, static_cast<double>(state.loaded_count) / static_cast<double>(estimated_cardinality));
	}
	// Second half is actually building the index
	return 50.0 + (50.0 * static_cast<double>(state.built_count) / static_cast<double>(state.loaded_count));
}

} // namespace duckdb
