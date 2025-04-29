#include "duckdb.hpp"
#include "usearch/helpers/database_setup.h"
#include "usearch/helpers/index_operations.h"
#include "usearch/helpers/query_runner.h"
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>
#include <cmath>




using namespace duckdb;




// ==================== Ground Truth Generator ====================
class GroundTruthGenerator {
private:
  DuckDB db;
  Connection con;
  std::vector<DatasetConfig> datasets;
  int numThreads;




  // Structure to hold vector data
  struct VectorData {
      int id;
      std::vector<float> vec;
  };


  struct DistanceResult {
   int query_id;
   int neighbor_id;
   float distance;
};




  // Compute L2 (Euclidean) distance between two vectors
  float computeDistance(const std::vector<float>& vec1, const std::vector<float>& vec2) {
      float sum = 0.0f;
      for (size_t i = 0; i < vec1.size(); i++) {
          float diff = vec1[i] - vec2[i];
          sum += diff * diff;
      }
      return std::sqrt(sum);
  }




  // Worker function for multi-threaded processing
  void processTestVectorBatch(
      const std::vector<VectorData>& testVectors,
      const std::vector<VectorData>& trainVectors,
      size_t start_idx,
      size_t end_idx,
      Appender& appender,
      std::mutex& appenderMutex,
      std::atomic<size_t>& vectorsProcessed,
      size_t totalTestVectors
  ) {




   for (size_t i = start_idx; i < end_idx && i < testVectors.size(); i++) {
       const auto& testVec = testVectors[i];


       std::vector<DistanceResult> localResults;


     
       for (const auto& trainVec : trainVectors) {
           float dist = computeDistance(testVec.vec, trainVec.vec);


           DistanceResult result;
           result.query_id = testVec.id;
           result.neighbor_id = trainVec.id;
           result.distance = computeDistance(testVec.vec, trainVec.vec);
           localResults.push_back(result);
       }


       std::lock_guard<std::mutex> lock(appenderMutex);


       for (const auto& result : localResults) {
            appender.AppendRow(result.query_id,
                            result.neighbor_id,
                            result.distance);
        }
  
          size_t processed = ++vectorsProcessed;
  
          if (processed % 1000 == 0 || processed == totalTestVectors){
           float progress = (processed * 100.0f) / totalTestVectors;
           std::cout << "\rProgress: " << processed << "/" << totalTestVectors
           << " test vectors processed (" << progress << "%)" <<std::flush;
          }
      }
  


   }


 


public:
  GroundTruthGenerator(int numThreads) : db("raw.db"), con(db), numThreads(numThreads) {
      con.Query("SET THREADS TO " + std::to_string(numThreads) + ";");
      datasets = DatabaseSetup::getDatasetConfigs();
  }




  void runGenerator(int datasetIdx) {
      try {
          auto startTime = std::chrono::high_resolution_clock::now();




          if (datasetIdx < 0 || datasetIdx >= (int)datasets.size()) {
              std::cerr << "Invalid dataset index: " << datasetIdx << std::endl;
              return;
          }




          const auto& dataset = datasets[datasetIdx];
          std::cout << "Generating ground truth table for dataset: " << dataset.name << std::endl;




          auto test_res = con.Query("SELECT * FROM raw." + dataset.name + "_test");
          auto train_res = con.Query("SELECT * FROM raw." + dataset.name + "_train");




          std::vector<VectorData> testVectors;
          testVectors.reserve(test_res->RowCount());
          for (idx_t i = 0; i < test_res->RowCount(); i++) {
              testVectors.push_back(VectorData{
                  test_res->GetValue<int>(0, i),
                  ExtractFloatVector(test_res->GetValue(1, i))
              });
          }




          std::vector<VectorData> trainVectors;
          trainVectors.reserve(train_res->RowCount());
          for (idx_t i = 0; i < train_res->RowCount(); i++) {
              trainVectors.push_back(VectorData{
                  train_res->GetValue<int>(0, i),
                  ExtractFloatVector(train_res->GetValue(1, i))
              });
          }




          std::cout << "Loaded " << testVectors.size() << " test vectors and "
                    << trainVectors.size() << " train vectors" << std::endl;




          std::cout << "Starting brute force KNN calculation using " << numThreads << " threads" << std::endl;




          // Create ground truth table
          con.Query("CREATE OR REPLACE TABLE raw." + dataset.name + "_ground_truth (" +
                    "query_id INTEGER, " +
                    "neighbor_id INTEGER, " +
                    "distance FLOAT)");




          Appender appender(con, dataset.name + "_ground_truth");
          std::mutex appenderMutex;




          size_t batchSize = (testVectors.size() + numThreads - 1) / numThreads;


          std::atomic<size_t> vectorsProcessed(0);




          std::vector<std::thread> threads;
          for (unsigned int t = 0; t < numThreads; t++) {
              size_t start_idx = t * batchSize;
              size_t end_idx = (t + 1) * batchSize;




              threads.emplace_back(
                  [this](const std::vector<VectorData>& testVectors,
                         const std::vector<VectorData>& trainVectors,
                         size_t start_idx, size_t end_idx,
                         Appender& appender,
                         std::mutex& appenderMutex,
                         std::atomic<size_t>& vectorsProcessed,
                         size_t totalTestVectors 
                       ) {
                      this->processTestVectorBatch(testVectors, trainVectors, start_idx, end_idx, appender, appenderMutex, vectorsProcessed, totalTestVectors);
                  },
                  std::ref(testVectors),
                  std::ref(trainVectors),
                  start_idx,
                  end_idx,
                  std::ref(appender),
                  std::ref(appenderMutex),
                  std::ref(vectorsProcessed),
                  testVectors.size()
              );
          }




          // Wait for all threads to finish
          for (auto& thread : threads) {
              thread.join();
          }




          appender.Close();




          // Move table into raw schema
//           con.Query("CREATE OR REPLACE TABLE raw." + dataset.name + "_ground_truth AS SELECT * FROM " + dataset.name + "_ground_truth");




          auto ground_truth_count = con.Query("SELECT COUNT(*) FROM raw." + dataset.name + "_ground_truth");
          std::cout << "Ground truth vectors: " << ground_truth_count->GetValue<int64_t>(0, 0) << std::endl;




          auto expected_rows = train_res->RowCount() * test_res->RowCount();
          std::cout << "Expected rows: " << expected_rows << std::endl;




          if (ground_truth_count->GetValue<int64_t>(0, 0) != expected_rows) {
              throw std::runtime_error("Setup failed: Incorrect number of rows inserted in ground truth table");
          } else {
              std::cout << "Successfully inserted " << expected_rows << " rows into ground truth table" << std::endl;
          }




          auto endTime = std::chrono::high_resolution_clock::now();
          auto duration = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
          std::cout << "Ground truth table generated for dataset: " << dataset.name
                    << " in " << duration << " seconds" << std::endl;




      } catch (std::exception& e) {
          std::cerr << "Error generating ground truth table: " << e.what() << std::endl;
      }
  }
};




// ==================== Main Function ====================
int main() {
  std::size_t executor_threads = std::thread::hardware_concurrency();




  try {
      GroundTruthGenerator fm_runner(executor_threads);
      fm_runner.runGenerator(0); // fashion_mnist


      GroundTruthGenerator m_runner(executor_threads);
      m_runner.runGenerator(1); // mnist




      GroundTruthGenerator s_runner(executor_threads);
      s_runner.runGenerator(2); // sift




      GroundTruthGenerator g_runner(executor_threads);
      g_runner.runGenerator(3); // gist




      return 0;
  } catch (std::exception& e) {
      std::cerr << "Fatal error: " << e.what() << std::endl;
      return 1;
  }
}











