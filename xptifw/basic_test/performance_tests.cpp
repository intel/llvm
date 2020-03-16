//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
//----------------------- performance_tests.cpp -----------------------------
// Tests the performance of the API and framework by running real world
// scenarios and computing the average costs and maximum events/sec that can
// be serviced by the framework at a given max. overhead constraint.
//---------------------------------------------------------------------------
#include <atomic>
#include <chrono>
#include <random>

#include "tbb/concurrent_vector.h"
#include "tbb/parallel_for.h"
#include "tbb/spin_mutex.h"
#include "tbb/task_arena.h"
#include "tbb/task_group.h"

#include "cl_processor.hpp"
#include "xpti_trace_framework.h"

namespace test {
void register_callbacks(uint8_t sid);
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

void test_performance::run_data_structure_tests_threads(
    int run_no, int num_threads, test::utils::table_model &t) {
  xptiReset();
  uint64_t ns;
  double ratio;

  // If the num-threads specification includes 0, then a true serial version
  // outside of TBB is run
  if (!num_threads) {
    auto &row = t.add_row(run_no, "Serial");
    row[(int)DSColumns::Threads] = num_threads;
    // Hold the string ids for measuring lookup later
    std::vector<xpti::string_id_t> ids;
    ids.resize(m_tracepoints);
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
        test::utils::scoped_timer timer(ns, ratio, m_tracepoints);
        for (int i = 0; i < m_tracepoints; ++i) {
          char *table_str = nullptr;
          // Assume that the string has already been created as it is normally
          // provided to the Payload constructors
          std::string &str = m_functions[i];
          ids[i] = xptiRegisterString(str.c_str(), &table_str);
        }
      }
      row[(int)DSColumns::STInsert] = ratio;

      { // lookup the created strings "m_tracepoints" randomly
        test::utils::scoped_timer timer(ns, ratio, m_tracepoints * 2);
        for (int i = 0; i < m_tracepoints * 2; ++i) {
          int lookup = m_rnd_tp[i % m_st_entries];
          const char *lut_string = xptiLookupString(ids[lookup]);
        }
      }
      row[(int)DSColumns::STLookup] = ratio;
    }

    // Column 3: Insert+ 2 Lookups
    // Perform measurement tests to determine the cost of insertion and 2
    // lookups for strings added to the string table
    { // Create NEW "m_tracepoint" strings
      std::vector<xpti::string_id_t> new_ids;
      new_ids.resize(m_tracepoints);
      long no_of_operations = m_tracepoints * 3;
      test::utils::scoped_timer timer(ns, ratio, no_of_operations);
      for (int i = 0; i < m_tracepoints; ++i) {
        char *table_str = nullptr;
        std::string &str = m_functions2[i];
        new_ids.push_back(xptiRegisterString(str.c_str(), &table_str));
      }
      for (int i = 0; i < m_tracepoints * 2; ++i) {
        int lookup = m_rnd_tp[i % m_st_entries]; // Generates a value between
                                                 // 0-m_tracepoints-1
        const char *lut_string = xptiLookupString(ids[lookup]);
      }
    }
    row[(int)DSColumns::STInsertLookup] = ratio;

    std::vector<int64_t> uids;
    std::vector<xpti::trace_event_data_t *> events;
    uids.resize(m_tracepoints);
    events.resize(m_tracepoints);
    // Column 4: Measure the cost of trace point creation and cache the returned
    // event and event ids
    {
      // Create "m_tracepoints" number of trace point events
      test::utils::scoped_timer timer(ns, ratio, m_tracepoints);
      for (int i = 0; i < m_tracepoints; ++i) {
        record &r = m_records[i];
        int lookup = r.lookup;
        std::string &fn = r.fn;
        xpti::payload_t p = xpti::payload_t(fn.c_str(), m_source, lookup,
                                            lookup % 80, (void *)r.lookup);
        xpti::trace_event_data_t *e = xptiMakeEvent(
            fn.c_str(), &p, (uint16_t)xpti::trace_event_type_t::algorithm,
            xpti::trace_activity_type_t::active, &m_instance_id);
        if (e) {
          uids[lookup] = e->unique_id;
          events[lookup] = e;
        }
      }
    }
    row[(int)DSColumns::TPCreate] = ratio;

    // Column 5: Measure the cost of trace point creation of previously created
    // trace points in an un-cached manner
    { // Lookup "m_tracepoints" instances, uncached where we create the payload
      // each time
      test::utils::scoped_timer timer(ns, ratio, m_tracepoints);
      for (int i = 0; i < m_tracepoints; ++i) {
        record &r = m_records[i];
        uint64_t lookup = r.lookup;
        std::string &fn = r.fn;
        xpti::payload_t p = xpti::payload_t(fn.c_str(), m_source, (int)lookup,
                                            (int)lookup % 80, (void *)lookup);
        xpti::trace_event_data_t *e = xptiMakeEvent(
            fn.c_str(), &p, (uint16_t)xpti::trace_event_type_t::algorithm,
            xpti::trace_activity_type_t::active, &m_instance_id);
      }
    }
    row[(int)DSColumns::TPUncachedLookup] = ratio;

    // Column 6: Measure the cost of trace point creation of previously created
    // trace points in an framework-cached manner
    { // Lookup "m_tracepoints" instances, framework-cached
      test::utils::scoped_timer timer(ns, ratio, m_tracepoints);
      for (int i = 0; i < m_tracepoints; ++i) {
        record &r = m_records[i];
        uint64_t lookup = r.lookup;
        xpti::trace_event_data_t *e =
            const_cast<xpti::trace_event_data_t *>(xptiFindEvent(uids[lookup]));
      }
    }
    row[(int)DSColumns::TPFWCache] = ratio;

    // Column 7: Measure the cost of trace point creation of previously created
    // and cached trace points
    { // Lookup "m_tracepoints" instances, locally-cached or locally visible
      test::utils::scoped_timer timer(ns, ratio, m_tp_instances);
      for (int i = 0; i < m_tp_instances; ++i) {
        record &r = m_records[i % m_tracepoints];
        uint64_t lookup = r.lookup; // get the random id to lookup
        xpti::trace_event_data_t *e = events[lookup];
      }
    }
    row[(int)DSColumns::TPLocalCache] = ratio;

    { // Notify "m_tracepoints" number tps, locally cached
      test::utils::scoped_timer timer(ns, ratio, m_tp_instances);
      for (int i = 0; i < m_tp_instances; ++i) {
        record &r = m_records[i % m_tracepoints];
        uint64_t lookup = r.lookup;
        xpti::trace_event_data_t *e = events[lookup];
        xpti::framework::scoped_notify ev(
            "xpti", (uint16_t)xpti::trace_point_type_t::region_begin, nullptr,
            e, m_instance_id, nullptr);
      }
    }
    row[(int)DSColumns::Notify] = ratio;

  } else {
    // Now run the same performance tests in multi-threaded mode to accommodate
    // lock contention costs

    std::string row_title = "Threads " + std::to_string(num_threads);
    auto &row = t.add_row(run_no, row_title);
    row[(int)DSColumns::Threads] = num_threads;

    // Limit TBB to use the number of threads for this run
    tbb::task_arena a(num_threads);
    a.execute([&]() {
      std::vector<xpti::string_id_t> ids;
      ids.resize(m_tracepoints);
      // Columns 1, 2: Insert, 2 Lookups
      // Perform measurement tests to determine the cost of insertions into the
      // string table, the lookup costs and a composite measurement of insertion
      // and 2 lookups for strings added to the string table
      {
        { // Create "m_tracepoints" strings
          test::utils::scoped_timer timer(ns, ratio, m_tracepoints);
          tbb::parallel_for(tbb::blocked_range<int>(0, m_tracepoints),
                            [&](tbb::blocked_range<int> &r) {
                              for (int i = r.begin(); i != r.end(); ++i) {
                                char *table_str = nullptr;
                                std::string &str = m_functions[i];
                                ids[i] =
                                    xptiRegisterString(str.c_str(), &table_str);
                              }
                            });
        }
        row[(int)DSColumns::STInsert] = ratio;

        { // lookup the created strings "m_tracepoints*2" linearly
          test::utils::scoped_timer timer(ns, ratio, m_tracepoints * 2);
          tbb::parallel_for(tbb::blocked_range<int>(0, m_tracepoints * 2),
                            [&](tbb::blocked_range<int> &r) {
                              for (int i = r.begin(); i != r.end(); ++i) {
                                int lookup = m_rnd_tp[i % m_st_entries];
                                const char *lut_string =
                                    xptiLookupString(ids[lookup]);
                              }
                            });
        }
        row[(int)DSColumns::STLookup] = ratio;
      }
      // Column 3: Insert+ 2 Lookups
      // Perform measurement tests to determine the cost of insertion and 2
      // lookups for strings added to the string table
      { // insert and lookup at the same time "m_st_entries*10"
        std::vector<xpti::string_id_t> new_ids;
        new_ids.resize(m_tracepoints);
        tbb::task_group g;
        // 2 lookups + 1 insert of m_tracepoints elements that occurs
        // simultaneously
        long no_of_operations = m_tracepoints * 3;
        test::utils::scoped_timer timer(ns, ratio, no_of_operations);
        g.run([&] {
          // Add new strings
          tbb::parallel_for(tbb::blocked_range<int>(0, m_tracepoints),
                            [&](tbb::blocked_range<int> &r) {
                              for (int i = r.begin(); i != r.end(); ++i) {
                                char *table_str = nullptr;
                                std::string &str = m_functions2[i];
                                new_ids[i] =
                                    xptiRegisterString(str.c_str(), &table_str);
                              }
                            });
        });
        g.run([&] {
          // And read previously added strings
          tbb::parallel_for(
              tbb::blocked_range<int>(0, m_st_entries),
              [&](tbb::blocked_range<int> &r) {
                for (int i = r.begin(); i != r.end(); ++i) {
                  int lookup =
                      m_rnd_tp[i % m_st_entries]; // Generates a value between
                                                  // 0-m_tracepoints-1
                  // Read from previously added strings by looking
                  // up the old IDs stored in 'ids'
                  const char *lut_string = xptiLookupString(ids[lookup]);
                }
              });
        });
        g.wait();
      }
      row[(int)DSColumns::STInsertLookup] = ratio;

      std::vector<int64_t> uids;
      std::vector<xpti::trace_event_data_t *> events;
      uids.resize(m_tracepoints);
      events.resize(m_tracepoints);
      // Column 4: Measure the cost of trace point creation and cache the
      // returned event and event ids
      { // Create "m_tracepoints" number of trace point events
        test::utils::scoped_timer timer(ns, ratio, m_tracepoints);
        tbb::parallel_for(
            tbb::blocked_range<int>(0, m_tracepoints),
            [&](tbb::blocked_range<int> &r) {
              for (int i = r.begin(); i != r.end(); ++i) {
                record &r = m_records[i];
                int lookup = r.lookup;
                std::string &fn = r.fn;
                xpti::payload_t p =
                    xpti::payload_t(fn.c_str(), m_source, lookup, lookup % 80,
                                    (void *)r.lookup);
                xpti::trace_event_data_t *e = xptiMakeEvent(
                    fn.c_str(), &p,
                    (uint16_t)xpti::trace_event_type_t::algorithm,
                    xpti::trace_activity_type_t::active, &m_instance_id);
                if (e) {
                  uids[lookup] = e->unique_id;
                  events[lookup] = e;
                }
              }
            });
      }
      row[(int)DSColumns::TPCreate] = ratio;

      // Column 5: Measure the cost of trace point creation of previously
      // created trace points in an un-cached manner
      { // Lookup "m_tracepoints" number of trace point events, uncached
        test::utils::scoped_timer timer(ns, ratio, m_tracepoints);
        tbb::parallel_for(
            tbb::blocked_range<int>(0, m_tracepoints),
            [&](tbb::blocked_range<int> &r) {
              for (int i = r.begin(); i != r.end(); ++i) {
                record &r = m_records[i];
                int lookup = r.lookup;
                std::string &fn = r.fn;
                xpti::payload_t p =
                    xpti::payload_t(fn.c_str(), m_source, lookup, lookup % 80,
                                    (void *)r.lookup);
                xpti::trace_event_data_t *e = xptiMakeEvent(
                    fn.c_str(), &p,
                    (uint16_t)xpti::trace_event_type_t::algorithm,
                    xpti::trace_activity_type_t::active, &m_instance_id);
              }
            });
      }
      row[(int)DSColumns::TPUncachedLookup] = ratio;

      // Column 6: Measure the cost of trace point creation of previously
      // created trace points in an framework-cached manner
      { // Lookup "m_tp_instances" number of trace point events,
        // framework-cached
        test::utils::scoped_timer timer(ns, ratio, m_tracepoints);
        tbb::parallel_for(tbb::blocked_range<int>(0, m_tracepoints),
                          [&](tbb::blocked_range<int> &r) {
                            for (int i = r.begin(); i != r.end(); ++i) {
                              record &r = m_records[i];
                              uint64_t lookup = r.lookup;
                              xpti::trace_event_data_t *e =
                                  const_cast<xpti::trace_event_data_t *>(
                                      xptiFindEvent(uids[lookup]));
                            }
                          });
      }
      row[(int)DSColumns::TPFWCache] = ratio;

      // Column 7: Measure the cost of trace point creation of previously
      // created and cached trace points
      { // Lookup "m_tracepoints" number of trace point events, locally-cached
        test::utils::scoped_timer timer(ns, ratio, m_tp_instances);
        tbb::parallel_for(tbb::blocked_range<int>(0, m_tp_instances),
                          [&](tbb::blocked_range<int> &r) {
                            for (int i = r.begin(); i != r.end(); ++i) {
                              record &r = m_records[i % m_tracepoints];
                              uint64_t lookup =
                                  r.lookup; // get the random id to lookup
                              xpti::trace_event_data_t *e = events[lookup];
                            }
                          });
      }
      row[(int)DSColumns::TPLocalCache] = ratio;

      { // Notify "m_tracepoints" number tps, locally cached
        test::utils::scoped_timer timer(ns, ratio, m_tp_instances);
        tbb::parallel_for(
            tbb::blocked_range<int>(0, m_tp_instances),
            [&](tbb::blocked_range<int> &r) {
              for (int i = r.begin(); i != r.end(); ++i) {
                record &r = m_records[i % m_tracepoints];
                uint64_t lookup = r.lookup;
                xpti::trace_event_data_t *e = events[lookup];
                xpti::framework::scoped_notify ev(
                    "xpti", (uint16_t)xpti::trace_point_type_t::region_begin,
                    nullptr, e, m_instance_id, nullptr);
              }
            });
      }
      row[(int)DSColumns::Notify] = ratio;
    });
  }
}

void test_performance::run_data_structure_tests() {
  test::utils::table_model table;

  test::utils::titles_t columns{"Threads",      "Str.Insert", "Str.Lookup",
                                "St.Ins/Lu",    "TP Create",  "TP Un-Cached",
                                "TP FW-Cached", "TP Local",   "Notify"};
  std::cout << std::setw(columns.size() * 15 / 2)
            << "Data Structure Tests [FW=framework, Lu=lookup, "
               "TP=Tracepoint, Time=ns\n";
  table.set_headers(columns);

  uint8_t sid = xptiRegisterStream("xpti");
  test::register_callbacks(sid);

  if (m_threads.size()) {
    int run_no = 0;
    for (auto thread : m_threads) {
      run_data_structure_tests_threads(run_no++, thread, table);
    }
  }

  table.print();
}

void test_performance::run_instrumentation_tests_threads(
    int run_no, int num_threads, test::utils::table_model &t) {
  xptiReset();
  uint64_t ns;
  double ratio;

  std::vector<int64_t> tp_ids;
  tp_ids.resize(m_tracepoints);
  std::vector<xpti::trace_event_data_t *> events;
  events.resize(m_tracepoints);
  // Variables used to compute events/sec
  uint64_t events_per_sec, overhead_based_cost;
  std::vector<std::pair<FWColumns, int>> cb_handler_cost = {
      {FWColumns::EPS10, 10},
      {FWColumns::EPS100, 100},
      {FWColumns::EPS500, 500},
      {FWColumns::EPS1000, 1000},
      {FWColumns::EPS2000, 2000}};

  if (!num_threads) {
    auto &row = t.add_row(run_no, "Serial");
    row[(int)FWColumns::Threads] = num_threads;
    {
      test::utils::scoped_timer timer(ns, ratio, m_tp_instances * 2);
      {
        test::utils::scoped_timer timer(ns, ratio, m_tracepoints);
        for (int i = 0; i < m_tracepoints; ++i) {
          std::string &fn = m_functions[i];
          xpti::payload_t p(fn.c_str(), m_source, i, i % 80, (void *)i);
          xpti::trace_event_data_t *e = xptiMakeEvent(
              fn.c_str(), &p, (uint16_t)xpti::trace_event_type_t::algorithm,
              xpti::trace_activity_type_t::active, &m_instance_id);
          if (e) {
            tp_ids[i] = e->unique_id;
            events[i] = e;
          }
        }
      }
      row[(int)FWColumns::TPCreate] = ratio;
      for (int i = 0; i < m_tp_instances; ++i) {
        int lookup = m_rnd_tp[i % m_st_entries];
        xpti::trace_event_data_t *e = events[lookup];
        xpti::framework::scoped_notify ev(
            "xpti", (uint16_t)xpti::trace_point_type_t::region_begin, nullptr,
            e, m_instance_id, nullptr);
      }
    }
    row[(int)FWColumns::TPLookupAndNotify] = ratio;
    for (auto cost : cb_handler_cost) {
      // Amount of non-instrumentation based work that needs to be present for
      // it to meet the overhead constraints requested
      overhead_based_cost = (ratio + cost.second) * (100.0 / m_overhead);
      row[(int)cost.first] = 1000000000 / overhead_based_cost;
    }

  } else {
    tbb::task_arena a(num_threads);

    std::string row_title = "Threads " + std::to_string(num_threads);
    auto &row = t.add_row(run_no, row_title);
    row[(int)FWColumns::Threads] = num_threads;
    {
      test::utils::scoped_timer timer(ns, ratio, m_tp_instances * 2);
      a.execute([&]() {
        {
          test::utils::scoped_timer timer(ns, ratio, m_tracepoints);
          tbb::parallel_for(
              tbb::blocked_range<int>(0, m_tracepoints),
              [&](tbb::blocked_range<int> &r) {
                for (int i = r.begin(); i != r.end(); ++i) {
                  std::string &fn = m_functions[i];
                  xpti::payload_t p(fn.c_str(), m_source, i, i % 80, (void *)i);
                  xpti::trace_event_data_t *e = xptiMakeEvent(
                      fn.c_str(), &p,
                      (uint16_t)xpti::trace_event_type_t::algorithm,
                      xpti::trace_activity_type_t::active, &m_instance_id);
                  if (e) {
                    tp_ids[i] = e->unique_id;
                    events[i] = e;
                  }
                }
              });
        }
        row[(int)FWColumns::TPCreate] = ratio;
        tbb::parallel_for(
            tbb::blocked_range<int>(0, m_tp_instances),
            [&](tbb::blocked_range<int> &r) {
              for (int i = r.begin(); i != r.end(); ++i) {
                record &r = m_records[i % m_tracepoints];
                uint64_t lookup = r.lookup;
                xpti::trace_event_data_t *e = events[lookup];
                xpti::framework::scoped_notify ev(
                    "xpti", (uint16_t)xpti::trace_point_type_t::region_begin,
                    nullptr, e, m_instance_id, nullptr);
              }
            });
      });
    }
    row[(int)FWColumns::TPLookupAndNotify] = ratio;
    for (auto cost : cb_handler_cost) {
      // Amount of non-instrumentation based work that needs to be present for
      // it to meet the overhead constraints requested
      overhead_based_cost = (ratio + cost.second) * (100.0 / m_overhead);
      row[(int)cost.first] = 1000000000 / overhead_based_cost;
    }
  }
}

void test_performance::run_instrumentation_tests() {
  test::utils::table_model table;

  test::utils::titles_t columns{
      "Threads",     "TP LU+Notify(ns)", "TP Create(ns)", "Ev/s,cb=10",
      "Ev/s,cb=100", "Ev/s,cb=500",      "Ev/s,cb=1000",  "Ev/s,cb=2000"};
  std::cout << std::setw(columns.size() * 15 / 2) << "Framework Tests\n";
  table.set_headers(columns);
  uint8_t sid = xptiRegisterStream("xpti");
  test::register_callbacks(sid);

  if (m_threads.size()) {
    int run_no = 0;
    for (auto thread : m_threads) {
      run_instrumentation_tests_threads(run_no++, thread, table);
    }
  }

  table.print();
}

} // namespace performance
} // namespace test