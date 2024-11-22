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
            // std::cout << "Deepening into cluster at level: " << cluster_level_ << std::endl;
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

	unum::usearch::index_dense_t base_index;

	// Cluster indexes
	std::unordered_map<unum::usearch::default_key_t, std::unique_ptr<HNSWIndex>> cluster_indexes;
	// Cluster indexes + centroid index
	std::unordered_map<std::string, std::unique_ptr<HNSWIndex>> all_indexes;

	// Cluster centroid keys
	unum::usearch::index_dense_t::vector_key_t* cluster_centroids_keys;
	// Distances to cluster centroids
	unum::usearch::default_distance_t* distances_to_cluster_centroids;

	int cluster_centroid_size;

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

	// Unbound expressions
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
	
	gstate->train_data.resize(estimated_cardinality);

	// Arrays to hold clustering results
	gstate->cluster_centroids_keys = new unum::usearch::index_dense_t::vector_key_t[MinValue(static_cast<idx_t>(estimated_cardinality), estimated_cardinality)];
    gstate->distances_to_cluster_centroids = new unum::usearch::default_distance_t[MinValue(static_cast<idx_t>(estimated_cardinality), estimated_cardinality)];

	auto &vector_type = data_types[0];

	// Get the size of the vector
	auto vector_size = ArrayType::GetSize(vector_type);
	// auto vector_child_type = ArrayType::GetChildType(vector_type);

	// Get the scalar kind from the array child type. This parameter should be verified during binding.
	auto scalar_kind = unum::usearch::scalar_kind_t::f32_k;

	auto metric_kind = unum::usearch::metric_kind_t::l2sq_k;

	// Create the usearch index
	unum::usearch::metric_punned_t metric(vector_size, metric_kind, scalar_kind);

	gstate->base_index = unum::usearch::index_dense_t::make(metric);

	// Initialize centroid index
	gstate->centroid_index =
	    make_uniq<HNSWIndex>("centroid_index", constraint_type, storage_ids, table_manager, unbound_expressions, db,
	                         info->options, IndexStorageInfo(), estimated_cardinality);

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

class HNSWIndexClusteringTask final : public ExecutorTask {

public:
    HNSWIndexClusteringTask(shared_ptr<Event> event_p, ClientContext &context, CreateHNSWIndexGlobalState &gstate_p,
                           size_t thread_id_p, const PhysicalCreateHNSWIndex &op_p)
        : ExecutorTask(context, std::move(event_p), op_p), gstate(gstate_p), thread_id(thread_id_p),
          local_scan_state() {
		// Initialize the scan chunk
		gstate.collection->InitializeScanChunk(scan_chunk);
	}

	TaskExecutionResult ExecuteTask(TaskExecutionMode mode) override {

	auto &index = gstate.base_index;
	auto &scan_state = gstate.scan_state;
    auto &collection = gstate.collection;

    const auto array_size = ArrayType::GetSize(scan_chunk.data[0].GetType());

	// Iterate over the data and construct the initial base index
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

				// Add the vector to the index
				const auto result = index.add(row_ptr[row_idx], data_ptr + (vec_idx * array_size), thread_id);
				
				// Create a vector from the data pointed to by data_ptr for this row
				std::vector<float> current_vector;
				for (size_t j = 0; j < array_size; ++j) {
					float value = *(data_ptr + (vec_idx * array_size) + j); // Access the correct element
					current_vector.push_back(value);
				}

				// Insert the constructed vector into train_data
				gstate.train_data[row_ptr[row_idx]] = current_vector;

				// Check for errors
				if (!result) {
					executor.PushError(ErrorData(result.error.what()));
					return TaskExecutionResult::TASK_ERROR;
				}
		}

		// Update the built count
		gstate.built_count += count;

			if (mode == TaskExecutionMode::PROCESS_PARTIAL) {
				// yield!
				return TaskExecutionResult::TASK_NOT_FINISHED;
			}

	}

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

	auto &scan_state = gstate.scan_state;
    auto &collection = gstate.collection;
	// Input for creation of index
	auto &table_manager = gstate.table_manager;
	auto &constraint_type = gstate.info->constraint_type;
	auto &db = gstate.storage->db;
	auto &unbound_expressions = gstate.unbound_expressions;
	//auto data_types = gstate.data_types;

	auto &train_data = gstate.train_data;
	auto cluster_centroid_size = gstate.cluster_centroid_size;
	auto &cluster_centroids_keys = gstate.cluster_centroids_keys;

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

			auto index_position = row_ptr[row_idx];
	
			// Ensure index_position is within bounds
			if (index_position >= gstate.estimated_cardinality || index_position >= cluster_centroid_size) {
				std::cerr << "Index out of bounds: index_position = " << index_position << std::endl;
				continue; // Skip this iteration if out of bounds
			}

			auto centroid_key = cluster_centroids_keys[index_position];

			if (centroid_key >= gstate.estimated_cardinality) {
				std::cerr << "Centroid key out of bounds: centroid_key = " << centroid_key << std::endl;
				continue;
			}

			// std::unique_lock<std::mutex> lock(gstate.cluster_indexes_mutex);
			{
			lock_guard<mutex> l(gstate.glock);
			auto it = gstate.cluster_indexes.find(centroid_key);
			if (it != gstate.cluster_indexes.end()) {
				//lock.unlock();
				it->second->index.add(row_ptr[row_idx], data_ptr + (vec_idx * array_size), thread_id);
			} else {
				//lock.unlock(); 
				std::string numberAsString = std::to_string(centroid_key);
					auto cluster_index_for_cluster =
					make_uniq<HNSWIndex>(numberAsString, constraint_type, gstate.storage_ids, *table_manager, unbound_expressions, db,
										gstate.info->options, IndexStorageInfo(), gstate.estimated_cardinality);
				cluster_index_for_cluster->index.add(row_ptr[row_idx], data_ptr + (vec_idx * array_size), thread_id);

				gstate.centroid_index->index.add(centroid_key, train_data[centroid_key].data(), thread_id);

				// Lock again before modifying the shared map
				//lock.lock();
				gstate.cluster_indexes.emplace(centroid_key, std::move(cluster_index_for_cluster));
			}
			}
			//lock.unlock();
        }

		// Update the built count
		gstate.built_count += count;

		if (mode == TaskExecutionMode::PROCESS_PARTIAL) {
				// yield!
				return TaskExecutionResult::TASK_NOT_FINISHED;
		}
	}

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

class HNSWIndexClusteringEvent final : public BasePipelineEvent {
public:
	HNSWIndexClusteringEvent(const PhysicalCreateHNSWIndex &op_p, CreateHNSWIndexGlobalState &gstate_p,
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

		// Schedule tasks equal to the number of threads, which will construct 
		// the initial base index and perform clustering
		auto &ts = TaskScheduler::GetScheduler(context);
		const auto num_threads = NumericCast<size_t>(ts.NumberOfThreads());

		vector<shared_ptr<Task>> construct_tasks;
		for (size_t tnum = 0; tnum < num_threads; tnum++) {
			construct_tasks.push_back(make_uniq<HNSWIndexClusteringTask>(shared_from_this(), context, gstate, tnum, op));
		}
		SetTasks(std::move(construct_tasks));
	}

	void FinishEvent() override {

	auto &context = pipeline->GetClientContext();

		auto &ts = TaskScheduler::GetScheduler(context);
		const auto num_threads = NumericCast<size_t>(ts.NumberOfThreads());
	

		auto &index = gstate.base_index;

		// Define clustering configuration
		unum::usearch::index_dense_clustering_config_t config;
		config.min_clusters = 20;   // Minimum number of clusters
		config.max_clusters = 20;   // Maximum number of clusters
		auto cluster_amount = gstate.info->options.find("cluster_amount");
		if (cluster_amount != gstate.info->options.end()) {
			config.min_clusters = cluster_amount->second.GetValue<int32_t>();
			config.max_clusters = cluster_amount->second.GetValue<int32_t>();
		}

		config.mode = unum::usearch::index_dense_clustering_config_t::merge_closest_k;

		std::size_t depth_level = 10;    // Cluster deepening level

		SmartIterable<some_scalar_t> iterable(gstate.train_data, depth_level);

		unum::usearch::executor_stl_t executor(ts.NumberOfThreads() - 1);

		// index.reserve({static_cast<size_t>(gstate.estimated_cardinality), static_cast<size_t>(executor.size())});

		unum::usearch::index_dense_t::clustering_result_t result = index.cluster(
			iterable.begin(), iterable.end(),
			config,
			gstate.cluster_centroids_keys,  gstate.distances_to_cluster_centroids,
			executor
		);

	gstate.cluster_centroid_size = 0;

    for (int i = 0; i < gstate.estimated_cardinality; ++i) {
        if (gstate.cluster_centroids_keys[i] != -1) {
            gstate.cluster_centroid_size++;
        }
    }
	}

	void FinalizeFinish() override {
		auto &collection = gstate.collection;

		// Initialize a new parallel scan for the index construction
		collection->InitializeScan(gstate.scan_state, ColumnDataScanProperties::ALLOW_ZERO_COPY);

	}
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
		auto &storage = table.GetStorage();
		if (!storage.IsRoot()) {
			throw TransactionException("Cannot create index on non-root transaction");
		}

		auto &schema = table.schema;
		info.column_ids = storage_ids;

		// Add all cluster indexes to table
		for (auto &cluster_index : gstate.cluster_indexes) {
			auto &index = cluster_index.second;
			if(index == nullptr) {
				continue;
			}
				auto key_to_stirng = std::to_string(cluster_index.first);
				auto &index_name = key_to_stirng;

				// Mark the index as dirty, update its count
				index->SetDirty();
				index->SyncSize();

				// If not in memory, persist the index to disk
				if (!storage.db.GetStorageManager().InMemory()) {
					// Finalize the index
					index->PersistToDisk();
				}

				// Create the index entry in the catalog
				
				info.index_name = index_name;

				if (schema.GetEntry(schema.GetCatalogTransaction(*gstate.context), CatalogType::INDEX_ENTRY, info.index_name)) {
					if (info.on_conflict != OnCreateConflict::IGNORE_ON_CONFLICT) {
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

		// And the centroid index
		auto &index = gstate.centroid_index;
		if(index == nullptr) {
			throw TransactionException("Centroid index is null");
		}

				// Mark the index as dirty, update its count
				index->SetDirty();
				index->SyncSize();

				// If not in memory, persist the index to disk
				if (!storage.db.GetStorageManager().InMemory()) {
					// Finalize the index
					index->PersistToDisk();
				}

				// Create the index entry in the catalog
				
				info.index_name = "centroid_index";

				if (schema.GetEntry(schema.GetCatalogTransaction(*gstate.context), CatalogType::INDEX_ENTRY, info.index_name)) {
					if (info.on_conflict != OnCreateConflict::IGNORE_ON_CONFLICT) {
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
};

SinkFinalizeType PhysicalCreateHNSWIndex::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                                   OperatorSinkFinalizeInput &input) const {
    // Get the global collection we've been appending to
	auto &gstate = input.global_state.Cast<CreateHNSWIndexGlobalState>();
	auto &collection = gstate.collection;

	// Move on to the next phase
	gstate.is_building = true;

	// Reserve the index size
	auto &ts = TaskScheduler::GetScheduler(context);
	auto &index = gstate.base_index;
	index.reserve({static_cast<size_t>(collection->Count()), static_cast<size_t>(ts.NumberOfThreads())});

	// Initialize a parallel scan for the index construction
	collection->InitializeScan(gstate.scan_state, ColumnDataScanProperties::ALLOW_ZERO_COPY);

	// Create a new event that will populate initial index
    auto clustering_event = make_shared_ptr<HNSWIndexClusteringEvent>(*this, gstate, pipeline, *info, storage_ids, table);
	// Create a new event that will construct the indexes
	auto index_creation_event = make_shared_ptr<HNSWIndexConstructionEvent>(*this, gstate, pipeline, *info, storage_ids, table);

    event.InsertEvent(std::move(index_creation_event));
	// Insert the event into the pipeline. The new event becomes dependant on 
	// the current event.
	event.InsertEvent(std::move(clustering_event));

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
