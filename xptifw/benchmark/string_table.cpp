//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
#include "helpers.hpp"
#include "xpti_string_table.hpp"

#include "benchmark/benchmark.h"

#include <random>
#include <string>

constexpr uint64_t NUM_ITERATIONS = 100'000;

static std::vector<xpti::string_id_t> *GIDs;
static xpti::StringTable *GStringTable = nullptr;

static void StringTable_Insert(benchmark::State &State) {
  if (State.thread_index == 0) {
    GStringTable = new xpti::StringTable();
  }
  for (auto _ : State) {
    State.PauseTiming();
    const std::string Str = getRandomString();
    const char *Ref = nullptr;
    State.ResumeTiming();

    benchmark::DoNotOptimize(GStringTable->add(Str.c_str(), &Ref));
  }
  if (State.thread_index == 0) {
#ifdef XPTI_STATISTICS
    State.counters["Retrievals"] = GStringTable->getRetrievals();
    State.counters["Insertions"] = GStringTable->getInsertions();
#endif
    delete GStringTable;
    GStringTable = nullptr;
  }
}

BENCHMARK(StringTable_Insert)->Threads(1)->Iterations(NUM_ITERATIONS);
BENCHMARK(StringTable_Insert)->Threads(2)->Iterations(NUM_ITERATIONS);
BENCHMARK(StringTable_Insert)->Threads(4)->Iterations(NUM_ITERATIONS);
BENCHMARK(StringTable_Insert)->Threads(8)->Iterations(NUM_ITERATIONS);
BENCHMARK(StringTable_Insert)->Threads(16)->Iterations(NUM_ITERATIONS);
BENCHMARK(StringTable_Insert)->Threads(24)->Iterations(NUM_ITERATIONS);
BENCHMARK(StringTable_Insert)->Threads(32)->Iterations(NUM_ITERATIONS);

static void StringTable_Lookup(benchmark::State &State) {
  if (State.thread_index == 0) {
    GStringTable = new xpti::StringTable(100'000);
    GIDs = new std::vector<xpti::string_id_t>();
    GIDs->resize(100'000);
    for (int I = 0; I < 100'000; I++) {
      const char *Ref;
      const std::string Rand = getRandomString();
      (*GIDs)[I] = GStringTable->add(Rand.c_str(), &Ref);
    }
  }

  for (auto _ : State) {
    State.PauseTiming();
    std::random_device Dev;
    std::mt19937 Range(Dev());
    std::uniform_int_distribution<std::mt19937::result_type> Dist(0, 99'999);
    size_t ID = Dist(Range);
    State.ResumeTiming();

    benchmark::DoNotOptimize(GStringTable->query((*GIDs)[ID]));
  }

  if (State.thread_index == 0) {
#ifdef XPTI_STATISTICS
    State.counters["Retrievals"] = GStringTable->getRetrievals();
    State.counters["Insertions"] = GStringTable->getInsertions();
#endif
    delete GStringTable;
    delete GIDs;
    GStringTable = nullptr;
    GIDs = nullptr;
  }
}

BENCHMARK(StringTable_Lookup)->Threads(1)->Iterations(NUM_ITERATIONS);
BENCHMARK(StringTable_Lookup)->Threads(2)->Iterations(NUM_ITERATIONS);
BENCHMARK(StringTable_Lookup)->Threads(4)->Iterations(NUM_ITERATIONS);
BENCHMARK(StringTable_Lookup)->Threads(8)->Iterations(NUM_ITERATIONS);
BENCHMARK(StringTable_Lookup)->Threads(16)->Iterations(NUM_ITERATIONS);
BENCHMARK(StringTable_Lookup)->Threads(24)->Iterations(NUM_ITERATIONS);
BENCHMARK(StringTable_Lookup)->Threads(32)->Iterations(NUM_ITERATIONS);
