//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "helpers.hpp"
#include "xpti/xpti_data_types.h"
#include "xpti_object_table.hpp"

#include "benchmark/benchmark.h"

#include <string>

constexpr uint64_t NUM_ITERATIONS = 100'000;

static std::vector<uint32_t> *GIDs;
static xpti::ObjectTable<uint32_t> *GObjTable = nullptr;

static void ObjectTable_Insert(benchmark::State &State) {
  if (State.thread_index == 0) {
    GObjTable = new xpti::ObjectTable<uint32_t>();
  }
  for (auto _ : State) {
    State.PauseTiming();
    const std::string Str = getRandomString();
    State.ResumeTiming();

    benchmark::DoNotOptimize(GObjTable->insert(
        Str, static_cast<uint8_t>(xpti::metadata_type_t::string)));
  }
  if (State.thread_index == 0) {
#ifdef XPTI_STATISTICS
    State.counters["Cache hits"] = GObjTable->getCacheHits();
    State.counters["Small objects"] = GObjTable->getSmallObjectsCount();
    State.counters["Large objects"] = GObjTable->getLargeObjectsCount();
#endif
    delete GObjTable;
    GObjTable = nullptr;
  }
}

BENCHMARK(ObjectTable_Insert)->Threads(1)->Iterations(NUM_ITERATIONS);
BENCHMARK(ObjectTable_Insert)->Threads(2)->Iterations(NUM_ITERATIONS);
BENCHMARK(ObjectTable_Insert)->Threads(4)->Iterations(NUM_ITERATIONS);
BENCHMARK(ObjectTable_Insert)->Threads(8)->Iterations(NUM_ITERATIONS);
BENCHMARK(ObjectTable_Insert)->Threads(16)->Iterations(NUM_ITERATIONS);
BENCHMARK(ObjectTable_Insert)->Threads(24)->Iterations(NUM_ITERATIONS);
BENCHMARK(ObjectTable_Insert)->Threads(32)->Iterations(NUM_ITERATIONS);

static void ObjectTable_Lookup(benchmark::State &State) {
  if (State.thread_index == 0) {
    GObjTable = new xpti::ObjectTable<uint32_t>(100'000);
    GIDs = new std::vector<uint32_t>();
    GIDs->resize(100'000);
    for (int I = 0; I < 100'000; I++) {
      const std::string Rand = getRandomString();
      (*GIDs)[I] = GObjTable->insert(
          Rand, static_cast<uint8_t>(xpti::metadata_type_t::string));
    }
  }

  for (auto _ : State) {
    State.PauseTiming();
    std::random_device Dev;
    std::mt19937 Range(Dev());
    std::uniform_int_distribution<std::mt19937::result_type> Dist(0, 99'999);
    size_t ID = Dist(Range);
    State.ResumeTiming();

    benchmark::DoNotOptimize(GObjTable->lookup((*GIDs)[ID]));
  }

  if (State.thread_index == 0) {
    delete GObjTable;
    delete GIDs;
    GObjTable = nullptr;
    GIDs = nullptr;
  }
}

BENCHMARK(ObjectTable_Lookup)->Threads(1)->Iterations(NUM_ITERATIONS);
BENCHMARK(ObjectTable_Lookup)->Threads(2)->Iterations(NUM_ITERATIONS);
BENCHMARK(ObjectTable_Lookup)->Threads(4)->Iterations(NUM_ITERATIONS);
BENCHMARK(ObjectTable_Lookup)->Threads(8)->Iterations(NUM_ITERATIONS);
BENCHMARK(ObjectTable_Lookup)->Threads(16)->Iterations(NUM_ITERATIONS);
BENCHMARK(ObjectTable_Lookup)->Threads(24)->Iterations(NUM_ITERATIONS);
BENCHMARK(ObjectTable_Lookup)->Threads(32)->Iterations(NUM_ITERATIONS);
