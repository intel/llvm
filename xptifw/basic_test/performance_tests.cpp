//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
//----------------------- performance_tests.cpp -----------------------------
// Tests the performance of the API and framework by running real world
// scenarios and computing the average costs and maximum Events/sec that can
// be serviced by the framework at a given max. overhead constraint.
//---------------------------------------------------------------------------

// The performance tests rely on a thread-pool implementation from the project
// https://github.com/bshoshany/thread-pool.git
//
// Copyright (c) 2023 Barak Shoshany. Licensed under the MIT license.
#include "cl_processor.hpp"
#include "parallel_test.h"
#include "xpti/xpti_trace_framework.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <random>

namespace test {
void registerCallbacks(uint8_t sid);
namespace performance {
enum class DSColumns {
  Threads,          ///< Slot used to record the number of threads
  STInsert,         ///< Used to capture the average string insert costs
  STLookup,         ///< Used to capture the average string lookup costs
  STInsertLookup,   ///< Avg. string insert+2 lookups cost
  TPCreate,         ///< Average trace point creation costs
  TPUncachedLookup, ///< Average trace point recration costs using payload
  TPFWCache,        ///< Average trace event lookup using unique_id
  TPLocalCache,     ///< Average costs to look up locally cached event (0)
  Notify            ///< Average notification costs
};

enum class FWColumns {
  Threads,           ///< Slot used to record the number of threads
  TPLookupAndNotify, ///< Average cost to create a trace event and notify based
                     ///< on the average frequency of a new tracepoint being
                     ///< created as a function of total number of trace point
                     ///< lookup/notifications
  TPCreate,          ///< Average trace point event creation cost
  EPS10,   ///< Events/sec @ given overhead with CB handler cost of 10ns
  EPS100,  ///< Events/sec @ given overhead with CB handler cost of 100ns
  EPS500,  ///< Events/sec @ given overhead with CB handler cost of 500ns
  EPS1000, ///< Events/sec @ given overhead with CB handler cost of 1000ns
  EPS2000  ///< Events/sec @ given overhead with CB handler cost of 2000ns
};

void TestPerformance::runDataStructureTestsThreads(
    int RunNo, int NumThreads, xpti::utils::TableModel &Model) {
  xptiReset();
  uint64_t TimeInNS;
  double ElapsedTime;

  // If the num-threads specification includes 0, then a true serial version
  // outside of TBB is run
  if (!NumThreads) {
    auto &ModelRow = Model.addRow(RunNo, "Serial");
    ModelRow[(int)DSColumns::Threads] = NumThreads;
    // Hold the string IDs for measuring lookup later
    std::vector<xpti::string_id_t> IDs;
    IDs.resize(MTracepoints);
    // Columns 1, 2: Insert, 2 Lookups
    // Perform measurement tests to determine the cost of insertions into the
    // string table, the lookup costs and a composite measurement of insertion
    // and 2 lookups for strings added to the string table
    {
      // Create 'm_tracepoint' number of strings and measure the cost of serial
      // insertions into a concurrent container. Here, using an unordered_map
      // will be faster, but we rely on TBB concurrent containers to ensure they
      // are thread safe
      {
        test::utils::ScopedTimer Timer(TimeInNS, ElapsedTime, MTracepoints);
        for (int i = 0; i < MTracepoints; ++i) {
          char *TableStrRef = nullptr;
          // Assume that the string has already been created as it is normally
          // provided to the Payload constructors
          std::string &FuncName = MFunctions[i];
          IDs[i] = xptiRegisterString(FuncName.c_str(), &TableStrRef);
        }
      }
      ModelRow[(int)DSColumns::STInsert] = ElapsedTime;

      { // lookup the created strings "MTracepoints" randomly
        test::utils::ScopedTimer Timer(TimeInNS, ElapsedTime, MTracepoints * 2);
        for (int i = 0; i < MTracepoints * 2; ++i) {
          int LookupIndex = MRndmTPIndex[i % MStringTableEntries];
          const char *LUTStrRef = xptiLookupString(IDs[LookupIndex]);
        }
      }
      ModelRow[(int)DSColumns::STLookup] = ElapsedTime;
    }

    // Column 3: Insert+ 2 Lookups
    // Perform measurement tests to determine the cost of insertion and 2
    // lookups for strings added to the string table
    { // Create NEW "m_tracepoint" strings
      std::vector<xpti::string_id_t> NewIDs;
      NewIDs.resize(MTracepoints);
      long NoOfOperations = MTracepoints * 3;
      test::utils::ScopedTimer Timer(TimeInNS, ElapsedTime, NoOfOperations);
      for (int i = 0; i < MTracepoints; ++i) {
        char *TableStrRef = nullptr;
        std::string &FuncName = MFunctions2[i];
        NewIDs.push_back(xptiRegisterString(FuncName.c_str(), &TableStrRef));
      }
      for (int i = 0; i < MTracepoints * 2; ++i) {
        int LookupIndex =
            MRndmTPIndex[i % MStringTableEntries]; // Generates a value between
                                                   // 0-MTracepoints-1
        const char *LUTStrRef = xptiLookupString(IDs[LookupIndex]);
      }
    }
    ModelRow[(int)DSColumns::STInsertLookup] = ElapsedTime;

    std::vector<uint64_t> UIds;
    std::vector<xpti::trace_event_data_t *> Events;
    UIds.resize(MTracepoints);
    Events.resize(MTracepoints);
    // Column 4: Measure the cost of trace point creation and cache the returned
    // event and event IDs
    {
      // Create "MTracepoints" number of trace point events
      test::utils::ScopedTimer Timer(TimeInNS, ElapsedTime, MTracepoints);
      for (int i = 0; i < MTracepoints; ++i) {
        record &r = MRecords[i];
        int LookupIndex = r.lookup;
        std::string &fn = r.fn;
        xpti::payload_t P = xpti::payload_t(fn.c_str(), MSource, LookupIndex,
                                            LookupIndex % 80, (void *)r.lookup);
        xpti::trace_event_data_t *Ev = xptiMakeEvent(
            fn.c_str(), &P, (uint16_t)xpti::trace_event_type_t::algorithm,
            xpti::trace_activity_type_t::active, &MInstanceID);
        if (Ev) {
          UIds[LookupIndex] = Ev->unique_id;
          Events[LookupIndex] = Ev;
        }
      }
    }
    ModelRow[(int)DSColumns::TPCreate] = ElapsedTime;

    // Column 5: Measure the cost of trace point creation of previously created
    // trace points in an un-cached manner
    { // Lookup "MTracepoints" instances, uncached where we create the payload
      // each time
      test::utils::ScopedTimer Timer(TimeInNS, ElapsedTime, MTracepoints);
      for (int i = 0; i < MTracepoints; ++i) {
        record &r = MRecords[i];
        uint64_t LookupIndex = r.lookup;
        std::string &fn = r.fn;
        xpti::payload_t P =
            xpti::payload_t(fn.c_str(), MSource, (int)LookupIndex,
                            (int)LookupIndex % 80, (void *)LookupIndex);
        xpti::trace_event_data_t *Ev = xptiMakeEvent(
            fn.c_str(), &P, (uint16_t)xpti::trace_event_type_t::algorithm,
            xpti::trace_activity_type_t::active, &MInstanceID);
      }
    }
    ModelRow[(int)DSColumns::TPUncachedLookup] = ElapsedTime;

    // Column 6: Measure the cost of trace point creation of previously created
    // trace points in an framework-cached manner
    { // Lookup "MTracepoints" instances, framework-cached
      test::utils::ScopedTimer Timer(TimeInNS, ElapsedTime, MTracepoints);
      for (int i = 0; i < MTracepoints; ++i) {
        record &r = MRecords[i];
        uint64_t LookupIndex = r.lookup;
        xpti::trace_event_data_t *Ev = const_cast<xpti::trace_event_data_t *>(
            xptiFindEvent(UIds[LookupIndex]));
      }
    }
    ModelRow[(int)DSColumns::TPFWCache] = ElapsedTime;

    // Column 7: Measure the cost of trace point creation of previously created
    // and cached trace points
    { // Lookup "MTracepoints" instances, locally-cached or locally visible
      test::utils::ScopedTimer Timer(TimeInNS, ElapsedTime,
                                     MTracepointInstances);
      for (int i = 0; i < MTracepointInstances; ++i) {
        record &r = MRecords[i % MTracepoints];
        uint64_t LookupIndex = r.lookup; // get the random id to lookup
        xpti::trace_event_data_t *Ev = Events[LookupIndex];
      }
    }
    ModelRow[(int)DSColumns::TPLocalCache] = ElapsedTime;

    { // Notify "MTracepoints" number tps, locally cached
      test::utils::ScopedTimer Timer(TimeInNS, ElapsedTime,
                                     MTracepointInstances);
      for (int i = 0; i < MTracepointInstances; ++i) {
        record &r = MRecords[i % MTracepoints];
        uint64_t LookupIndex = r.lookup;
        xpti::trace_event_data_t *Ev = Events[LookupIndex];
        xpti::framework::scoped_notify ev(
            "xpti", (uint16_t)xpti::trace_point_type_t::region_begin, nullptr,
            Ev, MInstanceID, nullptr);
      }
    }
    ModelRow[(int)DSColumns::Notify] = ElapsedTime;

  } else {
    // Now run the same performance tests in multi-threaded mode to accommodate
    // lock contention costs
    BS::thread_pool tp(NumThreads);

    std::string RowTitle = "Threads " + std::to_string(NumThreads);
    auto &ModelRow = Model.addRow(RunNo, RowTitle);
    ModelRow[(int)DSColumns::Threads] = NumThreads;

    {
      std::vector<int64_t> UIds;
      std::vector<xpti::trace_event_data_t *> Events;
      UIds.resize(MTracepoints);
      Events.resize(MTracepoints);
      std::vector<xpti::string_id_t> IDs;
      IDs.resize(MTracepoints);
      // Columns 1, 2: Insert, 2 Lookups
      // Perform measurement tests to determine the cost of insertions into the
      // string table, the lookup costs and a composite measurement of insertion
      // and 2 lookups for strings added to the string table
      {
        auto ParStrInsertionsTask = [&](int min, int max) {
          for (int i = min; i < max; ++i) {
            char *TableStrRef = nullptr;
            std::string &FuncName = MFunctions[i];
            IDs[i] = xptiRegisterString(FuncName.c_str(), &TableStrRef);
          }
        };

        { // Create "MTracepoints" strings
          test::utils::ScopedTimer Timer(TimeInNS, ElapsedTime, MTracepoints);
          PARALLEL_FOR(tp, ParStrInsertionsTask, 0, MTracepoints);
        }
        ModelRow[(int)DSColumns::STInsert] = ElapsedTime;

        auto ParStrLookupTask = [&](int min, int max) {
          for (int i = min; i < max; ++i) {
            char *TableStrRef = nullptr;
            int LookupIndex = MRndmTPIndex[i % MStringTableEntries];
            const char *LUTStrRef = xptiLookupString(IDs[LookupIndex]);
          }
        };

        { // Lookup "MTracepoints*2" strings
          test::utils::ScopedTimer Timer(TimeInNS, ElapsedTime, MTracepoints);
          PARALLEL_FOR(tp, ParStrLookupTask, 0, MTracepoints);
        }
        ModelRow[(int)DSColumns::STLookup] = ElapsedTime;
      }
      // Column 3: Insert+ 2 Lookups
      // Perform measurement tests to determine the cost of insertion and 2
      // lookups for strings added to the string table
      { // insert and lookup at the same time "MStringTableEntries*10"
        std::vector<xpti::string_id_t> NewIDs;
        NewIDs.resize(MTracepoints);

        auto ParStrInsertLookupTask = [&](int min, int max) {
          for (int i = min; i < max; ++i) {
            char *TableStrRef = nullptr;
            std::string &FuncName = MFunctions[i];
            NewIDs[i] = xptiRegisterString(FuncName.c_str(), &TableStrRef);
          }
        };
        auto ParStrLookupTwiceTask = [&](int min, int max) {
          for (int i = min; i < max; ++i) {
            // Generates a value between 0 - (MTracepoints - 1)
            int LookupIndex = MRndmTPIndex[i % MStringTableEntries];
            // Read from previously added strings by looking
            // up the old IDs stored in 'IDs'
            const char *LUTStrRef = xptiLookupString(IDs[LookupIndex]);
          }
        };
        { // Lookup "MTracepoints*2" strings
          // 2 lookups + 1 insert of MTracepoints elements that occurs
          // simultaneously
          long NoOfOperations = MTracepoints * 3;
          test::utils::ScopedTimer Timer(TimeInNS, ElapsedTime, NoOfOperations);
          PARALLEL_FOR(tp, ParStrInsertLookupTask, 0, MTracepoints);
          PARALLEL_FOR(tp, ParStrLookupTwiceTask, 0, MStringTableEntries);
        }
        ModelRow[(int)DSColumns::STInsertLookup] = ElapsedTime;
      }
      // Column 4: Measure the cost of trace point creation and cache the
      // returned event and event IDs
      {
        auto ParTracepointTask = [&](int min, int max) {
          for (int i = min; i < max; ++i) {
            record &r = MRecords[i];
            int LookupIndex = r.lookup;
            std::string &fn = r.fn;
            xpti::payload_t P =
                xpti::payload_t(fn.c_str(), MSource, LookupIndex,
                                LookupIndex % 80, (void *)r.lookup);
            xpti::trace_event_data_t *Ev = xptiMakeEvent(
                fn.c_str(), &P, (uint16_t)xpti::trace_event_type_t::algorithm,
                xpti::trace_activity_type_t::active, &MInstanceID);
            if (Ev) {
              UIds[LookupIndex] = Ev->unique_id;
              Events[LookupIndex] = Ev;
            }
          }
        };
        {
          test::utils::ScopedTimer Timer(TimeInNS, ElapsedTime, MTracepoints);
          PARALLEL_FOR(tp, ParTracepointTask, 0, MTracepoints);
        }
        ModelRow[(int)DSColumns::TPCreate] = ElapsedTime;
      }
      // Column 5: Measure the cost of trace point creation of previously
      // created trace points in an un-cached manner
      { // Lookup "MTracepoints" number of trace point Events, uncached
        auto ParTracepointLookupTask = [&](int min, int max) {
          for (int i = min; i < max; ++i) {
            record &r = MRecords[i];
            int LookupIndex = r.lookup;
            std::string &fn = r.fn;
            xpti::payload_t P =
                xpti::payload_t(fn.c_str(), MSource, LookupIndex,
                                LookupIndex % 80, (void *)r.lookup);
            xpti::trace_event_data_t *Ev = xptiMakeEvent(
                fn.c_str(), &P, (uint16_t)xpti::trace_event_type_t::algorithm,
                xpti::trace_activity_type_t::active, &MInstanceID);
          }
        };

        {
          test::utils::ScopedTimer Timer(TimeInNS, ElapsedTime, MTracepoints);
          PARALLEL_FOR(tp, ParTracepointLookupTask, 0, MTracepoints);
        }
        ModelRow[(int)DSColumns::TPUncachedLookup] = ElapsedTime;
      }

      // Column 6: Measure the cost of trace point creation of previously
      // created trace points in an framework-cached manner
      { // Lookup "MTracepointInstances" number of trace point Events,
        // framework-cached
        auto ParTracepointFWLookupTask = [&](int min, int max) {
          for (int i = min; i < max; ++i) {
            record &r = MRecords[i];
            uint64_t LookupIndex = r.lookup;
            xpti::trace_event_data_t *Ev =
                const_cast<xpti::trace_event_data_t *>(
                    xptiFindEvent(UIds[LookupIndex]));
          }
        };

        {
          test::utils::ScopedTimer Timer(TimeInNS, ElapsedTime, MTracepoints);
          PARALLEL_FOR(tp, ParTracepointFWLookupTask, 0, MTracepoints);
        }
        ModelRow[(int)DSColumns::TPFWCache] = ElapsedTime;
      }

      // Column 7: Measure the cost of trace point creation of previously
      // created and cached trace points
      { // Lookup "MTracepoints" number of trace point Events, locally-cached
        auto ParTracepointLocalLookupTask = [&](int min, int max) {
          for (int i = min; i < max; ++i) {
            record &r = MRecords[i];
            uint64_t LookupIndex = r.lookup;
            xpti::trace_event_data_t *Ev =
                const_cast<xpti::trace_event_data_t *>(
                    xptiFindEvent(UIds[LookupIndex]));
          }
        };
        {
          test::utils::ScopedTimer Timer(TimeInNS, ElapsedTime,
                                         MTracepointInstances);
          PARALLEL_FOR(tp, ParTracepointLocalLookupTask, 0,
                       MTracepointInstances);
        }
        ModelRow[(int)DSColumns::TPLocalCache] = ElapsedTime;
      }

      { // Notify "MTracepoints" number tps, locally cached
        auto ParTracepointNotifyTask = [&](int min, int max) {
          for (int i = min; i < max; ++i) {
            record &r = MRecords[i % MTracepoints];
            uint64_t LookupIndex = r.lookup;
            xpti::trace_event_data_t *Ev = Events[LookupIndex];
            xpti::framework::scoped_notify ev(
                "xpti", (uint16_t)xpti::trace_point_type_t::region_begin,
                nullptr, Ev, MInstanceID, nullptr);
          }
        };
        {
          test::utils::ScopedTimer Timer(TimeInNS, ElapsedTime,
                                         MTracepointInstances);
          PARALLEL_FOR(tp, ParTracepointNotifyTask, 0, MTracepointInstances);
        }
        ModelRow[(int)DSColumns::Notify] = ElapsedTime;
      }
    }
  }
}

void TestPerformance::runDataStructureTests() {
  xpti::utils::TableModel Model;

  xpti::utils::titles_t Columns{"Threads",      "Str.Insert", "Str.Lookup",
                                "St.Ins/Lu",    "TP Create",  "TP Un-Cached",
                                "TP FW-Cached", "TP Local",   "Notify"};
  std::cout << std::setw(Columns.size() * 15 / 2)
            << "Data Structure Tests [FW=framework, Lu=lookup, "
               "TP=Tracepoint, Time=ns\n";
  Model.setHeaders(Columns);

  uint8_t sid = xptiRegisterStream("xpti");
  test::registerCallbacks(sid);

  if (MThreads.size()) {
    int RunNo = 0;
    for (auto Thread : MThreads) {
      runDataStructureTestsThreads(RunNo++, Thread, Model);
    }
  }

  Model.print();
}

void TestPerformance::runInstrumentationTestsThreads(
    int RunNo, int NumThreads, xpti::utils::TableModel &Model) {
  xptiReset();
  uint64_t TimeInNS;
  double ElapsedTime;

  std::vector<int64_t> tp_ids;
  tp_ids.resize(MTracepoints);
  std::vector<xpti::trace_event_data_t *> Events;
  Events.resize(MTracepoints);
  // Variables used to compute Events/sec
  uint64_t events_per_sec, overhead_based_cost;
  std::vector<std::pair<FWColumns, int>> cb_handler_cost = {
      {FWColumns::EPS10, 10},
      {FWColumns::EPS100, 100},
      {FWColumns::EPS500, 500},
      {FWColumns::EPS1000, 1000},
      {FWColumns::EPS2000, 2000}};

  if (!NumThreads) {
    auto &ModelRow = Model.addRow(RunNo, "Serial");
    ModelRow[(int)FWColumns::Threads] = NumThreads;
    {
      test::utils::ScopedTimer Timer(TimeInNS, ElapsedTime,
                                     MTracepointInstances * 2);
      {
        test::utils::ScopedTimer Timer(TimeInNS, ElapsedTime, MTracepoints);
        for (size_t i = 0; i < MTracepoints; ++i) {
          std::string &fn = MFunctions[i];
          xpti::payload_t P(fn.c_str(), MSource, i, i % 80, (void *)i);
          xpti::trace_event_data_t *Ev = xptiMakeEvent(
              fn.c_str(), &P, (uint16_t)xpti::trace_event_type_t::algorithm,
              xpti::trace_activity_type_t::active, &MInstanceID);
          if (Ev) {
            tp_ids[i] = Ev->unique_id;
            Events[i] = Ev;
          }
        }
      }
      ModelRow[(int)FWColumns::TPCreate] = ElapsedTime;
      for (int i = 0; i < MTracepointInstances; ++i) {
        int LookupIndex = MRndmTPIndex[i % MStringTableEntries];
        xpti::trace_event_data_t *Ev = Events[LookupIndex];
        xpti::framework::scoped_notify ev(
            "xpti", (uint16_t)xpti::trace_point_type_t::region_begin, nullptr,
            Ev, MInstanceID, nullptr);
      }
    }
    ModelRow[(int)FWColumns::TPLookupAndNotify] = ElapsedTime;
    for (auto cost : cb_handler_cost) {
      // Amount of non-instrumentation based work that needs to be present for
      // it to meet the overhead constraints requested
      overhead_based_cost = (ElapsedTime + cost.second) * (100.0 / MOverhead);
      ModelRow[(int)cost.first] = 1000000000 / overhead_based_cost;
    }

  } else {
    BS::thread_pool tp(NumThreads);

    std::string RowTitle = "Threads " + std::to_string(NumThreads);
    auto &ModelRow = Model.addRow(RunNo, RowTitle);
    ModelRow[(int)FWColumns::Threads] = NumThreads;
    {
      auto ParMakeTraceEventTask = [&](int min, int max) {
        for (size_t i = min; i < max; ++i) {
          std::string &fn = MFunctions[i];
          xpti::payload_t P(fn.c_str(), MSource, i, i % 80, (void *)i);
          xpti::trace_event_data_t *Ev = xptiMakeEvent(
              fn.c_str(), &P, (uint16_t)xpti::trace_event_type_t::algorithm,
              xpti::trace_activity_type_t::active, &MInstanceID);
          if (Ev) {
            tp_ids[i] = Ev->unique_id;
            Events[i] = Ev;
          }
        }
      };

      auto ParNotifyTask = [&](int min, int max) {
        for (int i = min; i < max; ++i) {
          record &r = MRecords[i % MTracepoints];
          uint64_t LookupIndex = r.lookup;
          xpti::trace_event_data_t *Ev = Events[LookupIndex];
          xpti::framework::scoped_notify ev(
              "xpti", (uint16_t)xpti::trace_point_type_t::region_begin, nullptr,
              Ev, MInstanceID, nullptr);
        }
      };

      {
        test::utils::ScopedTimer Timer(TimeInNS, ElapsedTime,
                                       MTracepointInstances * 2);
        {
          test::utils::ScopedTimer Timer(TimeInNS, ElapsedTime, MTracepoints);
          PARALLEL_FOR(tp, ParMakeTraceEventTask, 0, MTracepoints);
        }
        ModelRow[(int)FWColumns::TPCreate] = ElapsedTime;
        PARALLEL_FOR(tp, ParMakeTraceEventTask, 0, MTracepoints);
      }
      ModelRow[(int)FWColumns::TPLookupAndNotify] = ElapsedTime;
      for (auto cost : cb_handler_cost) {
        // Amount of non-instrumentation based work that needs to be present for
        // it to meet the overhead constraints requested
        overhead_based_cost = (ElapsedTime + cost.second) * (100.0 / MOverhead);
        ModelRow[(int)cost.first] = 1000000000 / overhead_based_cost;
      }
    }
  }
}

void TestPerformance::runInstrumentationTests() {
  xpti::utils::TableModel Model;

  xpti::utils::titles_t Columns{
      "Threads",     "TP LU+Notify(ns)", "TP Create(ns)", "Ev/s,cb=10",
      "Ev/s,cb=100", "Ev/s,cb=500",      "Ev/s,cb=1000",  "Ev/s,cb=2000"};
  std::cout << std::setw(Columns.size() * 15 / 2) << "Framework Tests\n";
  Model.setHeaders(Columns);
  uint8_t sid = xptiRegisterStream("xpti");
  test::registerCallbacks(sid);

  if (MThreads.size()) {
    int RunNo = 0;
    for (auto Thread : MThreads) {
      runInstrumentationTestsThreads(RunNo++, Thread, Model);
    }
  }

  Model.print();
}

} // namespace performance
} // namespace test
