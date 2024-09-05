#pragma once

#include "duckdb.hpp"

namespace duckdb {

struct HNSWModule {
public:
	static void Register(DatabaseInstance &db) {
		RegisterIndex(db);
		RegisterIndexScan(db);
		RegisterIndexPragmas(db);
		RegisterPlanIndexCreate(db);
		RegisterMacros(db);

		// Optimizers
		RegisterExprOptimizer(db);
		RegisterScanOptimizer(db);
		RegisterTopKOptimizer(db);
	}

private:
	static void RegisterIndex(DatabaseInstance &db);
	static void RegisterIndexScan(DatabaseInstance &db);
	static void RegisterIndexPragmas(DatabaseInstance &db);
	static void RegisterPlanIndexCreate(DatabaseInstance &db);
	static void RegisterMacros(DatabaseInstance &db);
	static void RegisterTopKOptimizer(DatabaseInstance &db);

	static void RegisterExprOptimizer(DatabaseInstance &db);
	static void RegisterTopKOperator(DatabaseInstance &db);
	static void RegisterScanOptimizer(DatabaseInstance &db);
};

} // namespace duckdb