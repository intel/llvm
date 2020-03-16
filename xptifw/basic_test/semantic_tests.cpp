//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
//----------------------- semantic_tests.cpp -------------------------------
// Tests the correctness of the API by comparing it agains the spec and
// expected results.
//--------------------------------------------------------------------------
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

static void tp_cb(uint16_t trace_type, xpti::trace_event_data_t *parent,
                  xpti::trace_event_data_t *event, uint64_t instance,
                  const void *ud) {}

namespace test {
void register_callbacks(uint8_t sid) {
  xptiRegisterCallback(sid, (uint16_t)xpti::trace_point_type_t::graph_create,
                       tp_cb);
  xptiRegisterCallback(sid, (uint16_t)xpti::trace_point_type_t::node_create,
                       tp_cb);
  xptiRegisterCallback(sid, (uint16_t)xpti::trace_point_type_t::edge_create,
                       tp_cb);
  xptiRegisterCallback(sid, (uint16_t)xpti::trace_point_type_t::region_begin,
                       tp_cb);
  xptiRegisterCallback(sid, (uint16_t)xpti::trace_point_type_t::region_end,
                       tp_cb);
  xptiRegisterCallback(sid, (uint16_t)xpti::trace_point_type_t::task_begin,
                       tp_cb);
  xptiRegisterCallback(sid, (uint16_t)xpti::trace_point_type_t::task_end,
                       tp_cb);
  xptiRegisterCallback(sid, (uint16_t)xpti::trace_point_type_t::barrier_begin,
                       tp_cb);
  xptiRegisterCallback(sid, (uint16_t)xpti::trace_point_type_t::barrier_end,
                       tp_cb);
  xptiRegisterCallback(sid, (uint16_t)xpti::trace_point_type_t::lock_begin,
                       tp_cb);
  xptiRegisterCallback(sid, (uint16_t)xpti::trace_point_type_t::lock_end,
                       tp_cb);
  xptiRegisterCallback(sid, (uint16_t)xpti::trace_point_type_t::transfer_begin,
                       tp_cb);
  xptiRegisterCallback(sid, (uint16_t)xpti::trace_point_type_t::transfer_end,
                       tp_cb);
  xptiRegisterCallback(sid, (uint16_t)xpti::trace_point_type_t::thread_begin,
                       tp_cb);
  xptiRegisterCallback(sid, (uint16_t)xpti::trace_point_type_t::thread_end,
                       tp_cb);
  xptiRegisterCallback(sid, (uint16_t)xpti::trace_point_type_t::wait_begin,
                       tp_cb);
  xptiRegisterCallback(sid, (uint16_t)xpti::trace_point_type_t::wait_end,
                       tp_cb);
  xptiRegisterCallback(sid, (uint16_t)xpti::trace_point_type_t::signal, tp_cb);
}
// The semantic namespace contains tests to determine the correctness of the
// implementation. The test ensure that the framework is robust under serial and
// multi-threaded conditions.
namespace semantic {
enum class STColumns {
  Threads,
  Insertions,
  Lookups,
  DuplicateInserts,
  PassRate
};

enum class TPColumns {
  Threads,
  Insertions,
  Lookups,
  DuplicateInserts,
  PayloadLookup,
  PassRate
};

enum class NColumns { Threads, Notifications, PassRate };

void test_correctness::run_string_table_test_threads(
    int run_no, int num_threads, test::utils::table_model &t) {
  xptiReset();
  constexpr long num_strings = 1000;

  if (!num_threads) {
    std::vector<char *> strings;
    std::vector<xpti::string_id_t> ids;
    ids.resize(num_strings);
    strings.resize(num_strings);
    for (int i = 0; i < num_strings; ++i) {
      char *table_str = nullptr;
      std::string str = "Function" + std::to_string(i);
      ids[i] = xptiRegisterString(str.c_str(), &table_str);
      strings[i] = table_str;
    }
    auto &row = t.add_row(run_no, "Serial");
    row[(int)STColumns::Threads] = num_threads;
    row[(int)STColumns::Insertions] = (long double)strings.size();
    int lookup_count = 0;
    for (int i = 0; i < strings.size(); ++i) {
      const char *table_str = xptiLookupString(ids[i]);
      if (table_str == strings[i])
        ++lookup_count;
    }
    row[(int)STColumns::Lookups] = lookup_count;
    int duplicate_count = 0;
    for (int i = 0; i < strings.size(); ++i) {
      char *table_str = nullptr;
      std::string str = "Function" + std::to_string(i);
      xpti::string_id_t id = xptiRegisterString(str.c_str(), &table_str);
      if (str == table_str && id == ids[i] && table_str == strings[i])
        ++duplicate_count;
    }
    row[(int)STColumns::DuplicateInserts] = duplicate_count;
    row[(int)STColumns::PassRate] =
        (double)(strings.size() + lookup_count + duplicate_count) /
        (num_strings * 3) * 100;
  } else {
    tbb::task_arena a(num_threads);

    a.execute([&]() {
      std::vector<char *> strings;
      std::vector<xpti::string_id_t> ids;
      strings.resize(num_strings);
      ids.resize(num_strings);
      tbb::parallel_for(tbb::blocked_range<int>(0, num_strings),
                        [&](tbb::blocked_range<int> &r) {
                          for (int i = r.begin(); i != r.end(); ++i) {
                            char *table_str = nullptr;
                            std::string str = "Function" + std::to_string(i);
                            ids[i] =
                                xptiRegisterString(str.c_str(), &table_str);
                            strings[i] = table_str;
                          }
                        });

      std::string row_title = "Threads " + std::to_string(num_threads);
      auto &row = t.add_row(run_no, row_title);
      row[(int)STColumns::Threads] = num_threads;
      row[(int)STColumns::Insertions] = (long double)strings.size();
      std::atomic<int> lookup_count = {0}, duplicate_count = {0};
      tbb::parallel_for(tbb::blocked_range<int>(0, num_strings),
                        [&](tbb::blocked_range<int> &r) {
                          for (int i = r.begin(); i != r.end(); ++i) {
                            const char *table_str = xptiLookupString(ids[i]);
                            if (table_str == strings[i])
                              ++lookup_count;
                          }
                        });
      row[(int)STColumns::Lookups] = lookup_count;
      tbb::parallel_for(tbb::blocked_range<int>(0, num_strings),
                        [&](tbb::blocked_range<int> &r) {
                          for (int i = r.begin(); i != r.end(); ++i) {
                            char *table_str = nullptr;
                            std::string str = "Function" + std::to_string(i);
                            xpti::string_id_t id =
                                xptiRegisterString(str.c_str(), &table_str);
                            if (str == table_str && id == ids[i] &&
                                table_str == strings[i])
                              ++duplicate_count;
                          }
                        });
      row[(int)STColumns::DuplicateInserts] = duplicate_count;

      row[(int)STColumns::PassRate] =
          (double)(strings.size() + lookup_count + duplicate_count) /
          (num_strings * 3) * 100;
    });
  }
}

void test_correctness::run_string_table_tests() {
  test::utils::table_model table;

  test::utils::titles_t columns{"Threads", "Insert", "Lookup", "Duplicate",
                                "Pass rate"};
  std::cout << std::setw(25) << "String Table Tests\n";
  table.set_headers(columns);

  if (m_threads.size()) {
    int run_no = 0;
    for (auto thread : m_threads) {
      run_string_table_test_threads(run_no++, thread, table);
    }
  }

  table.print();
}

void test_correctness::run_tracepoint_test_threads(
    int run_no, int num_threads, test::utils::table_model &t) {
  xptiReset();
  constexpr long tracepoint_count = 1000;

  if (!num_threads) {
    std::vector<xpti::payload_t *> payloads;
    std::vector<int64_t> uids;
    std::vector<xpti::trace_event_data_t *> events;
    payloads.resize(tracepoint_count);
    uids.resize(tracepoint_count);
    events.resize(tracepoint_count);

    for (uint64_t i = 0; i < tracepoint_count; ++i) {
      std::string fn = "Function" + std::to_string(i);
      xpti::payload_t p = xpti::payload_t(fn.c_str(), m_source, (int)i,
                                          (int)(i % 80), (void *)i);
      xpti::trace_event_data_t *e = xptiMakeEvent(
          fn.c_str(), &p, (uint16_t)xpti::trace_event_type_t::algorithm,
          xpti::trace_activity_type_t::active, &m_instance_id);
      if (e) {
        uids[i] = e->unique_id;
        payloads[i] = e->reserved.payload;
        events[i] = e;
      }
    }
    auto &row = t.add_row(run_no, "Serial");
    row[(int)TPColumns::Threads] = num_threads;
    row[(int)TPColumns::Insertions] = (long double)events.size();

    std::atomic<int> lookup_count = {0};
    for (int i = 0; i < events.size(); ++i) {
      const xpti::trace_event_data_t *e = xptiFindEvent(uids[i]);
      if (e && e->unique_id == uids[i])
        ++lookup_count;
    }
    row[(int)TPColumns::Lookups] = lookup_count;
    std::atomic<int> duplicate_count = {0};
    std::atomic<int> payload_count = {0};
    for (uint64_t i = 0; i < events.size(); ++i) {
      std::string fn = "Function" + std::to_string(i);
      xpti::payload_t p =
          xpti::payload_t(fn.c_str(), m_source, (int)i, (int)i % 80, (void *)i);
      xpti::trace_event_data_t *e = xptiMakeEvent(
          fn.c_str(), &p, (uint16_t)xpti::trace_event_type_t::algorithm,
          xpti::trace_activity_type_t::active, &m_instance_id);
      if (e) {
        if (e->unique_id == uids[i]) {
          ++duplicate_count;
        }
        xpti::payload_t *rp = e->reserved.payload;
        if (e->unique_id == uids[i] && rp &&
            std::string(rp->name) == std::string(p.name) &&
            std::string(rp->source_file) == std::string(p.source_file) &&
            rp->line_no == p.line_no && rp->column_no == p.column_no)
          ++payload_count;
      }
    }
    row[(int)TPColumns::DuplicateInserts] = duplicate_count;
    row[(int)TPColumns::PayloadLookup] = payload_count;
    row[(int)TPColumns::PassRate] = (double)(events.size() + lookup_count +
                                             duplicate_count + payload_count) /
                                    (tracepoint_count * 4) * 100;
  } else {
    tbb::task_arena a(num_threads);

    a.execute([&]() {
      std::vector<xpti::payload_t *> payloads;
      std::vector<int64_t> uids;
      std::vector<xpti::trace_event_data_t *> events;
      payloads.resize(tracepoint_count);
      uids.resize(tracepoint_count);
      events.resize(tracepoint_count);

      tbb::spin_mutex m_lock;
      tbb::parallel_for(
          tbb::blocked_range<int>(0, tracepoint_count),
          [&](tbb::blocked_range<int> &r) {
            for (uint64_t i = r.begin(); i != r.end(); ++i) {
              std::string fn = "Function" + std::to_string(i);
              xpti::payload_t p = xpti::payload_t(fn.c_str(), m_source, (int)i,
                                                  (int)i % 80, (void *)i);
              xpti::trace_event_data_t *e = xptiMakeEvent(
                  fn.c_str(), &p, (uint16_t)xpti::trace_event_type_t::algorithm,
                  xpti::trace_activity_type_t::active, &m_instance_id);
              if (e) {
                uids[i] = e->unique_id;
                payloads[i] = e->reserved.payload;
                events[i] = e;
              }
            }
          });

      std::string row_title = "Threads " + std::to_string(num_threads);
      auto &row = t.add_row(run_no, row_title);
      row[(int)TPColumns::Threads] = num_threads;
      row[(int)TPColumns::Insertions] = (long double)events.size();
      std::atomic<int> lookup_count = {0}, duplicate_count = {0},
                       payload_count = {0};
      tbb::parallel_for(tbb::blocked_range<int>(0, tracepoint_count),
                        [&](tbb::blocked_range<int> &r) {
                          for (int i = r.begin(); i != r.end(); ++i) {
                            const xpti::trace_event_data_t *e =
                                xptiFindEvent(uids[i]);
                            if (e && e->unique_id == uids[i])
                              lookup_count++;
                          }
                        });

      row[(int)TPColumns::Lookups] = lookup_count;
      tbb::parallel_for(
          tbb::blocked_range<int>(0, tracepoint_count),
          [&](tbb::blocked_range<int> &r) {
            for (uint64_t i = r.begin(); i != r.end(); ++i) {
              std::string fn = "Function" + std::to_string(i);
              xpti::payload_t p = xpti::payload_t(fn.c_str(), m_source, (int)i,
                                                  (int)i % 80, (void *)i);
              xpti::trace_event_data_t *e = xptiMakeEvent(
                  fn.c_str(), &p, (uint16_t)xpti::trace_event_type_t::algorithm,
                  xpti::trace_activity_type_t::active, &m_instance_id);
              if (e) {
                if (e->unique_id == uids[i]) {
                  ++duplicate_count;
                }
                xpti::payload_t *rp = e->reserved.payload;
                if (e->unique_id == uids[i] && rp &&
                    std::string(rp->name) == std::string(p.name) &&
                    std::string(rp->source_file) ==
                        std::string(p.source_file) &&
                    rp->line_no == p.line_no && rp->column_no == p.column_no)
                  ++payload_count;
              }
            }
          });
      row[(int)TPColumns::DuplicateInserts] = duplicate_count;
      row[(int)TPColumns::PayloadLookup] = payload_count;
      row[(int)TPColumns::PassRate] =
          (double)(events.size() + lookup_count + duplicate_count +
                   payload_count) /
          (tracepoint_count * 4) * 100;
    });
  }
}

void test_correctness::run_tracepoint_tests() {
  test::utils::table_model table;

  test::utils::titles_t columns{"Threads",   "Create",  "Lookup",
                                "Duplicate", "Payload", "Pass rate"};
  std::cout << std::setw(25) << "Tracepoint Tests\n";
  table.set_headers(columns);

  if (m_threads.size()) {
    int run_no = 0;
    for (auto thread : m_threads) {
      run_tracepoint_test_threads(run_no++, thread, table);
    }
  }

  table.print();
}

void test_correctness::run_notification_test_threads(
    int run_no, int num_threads, test::utils::table_model &t) {
  xptiReset();
  constexpr long tp_count = 30, callback_count = tp_count * 30;
  std::vector<xpti::payload_t *> payloads;
  std::vector<int64_t> uids;
  std::vector<xpti::trace_event_data_t *> events;
  payloads.resize(tp_count);
  uids.resize(tp_count);
  events.resize(tp_count);

  if (!num_threads) {

    // assumes tp creation is thread safe
    std::atomic<int> notify_count = {0};
    for (uint64_t i = 0; i < tp_count; ++i) {
      int index = (int)i;
      std::string fn = "Function" + std::to_string(i);
      xpti::payload_t p = xpti::payload_t(fn.c_str(), m_source, index,
                                          index % 80, (void *)(i % 10));
      xpti::trace_event_data_t *e = xptiMakeEvent(
          fn.c_str(), &p, (uint16_t)xpti::trace_event_type_t::algorithm,
          xpti::trace_activity_type_t::active, &m_instance_id);
      if (e) {
        uids[index] = e->unique_id;
        payloads[index] = e->reserved.payload;
        events[index] = e;
      }
      notify_count++;
    }

    auto &row = t.add_row(run_no, "Serial");
    row[(int)NColumns::Threads] = num_threads;

    for (int i = tp_count; i < callback_count; ++i) {
      int index = (int)i % tp_count;
      void *addr = (void *)(index % 10);
      std::string fn = "Function" + std::to_string(index);
      xpti::payload_t p = xpti::payload_t(fn.c_str(), m_source, (int)index,
                                          (int)index % 80, addr);
      xpti::trace_event_data_t *e = xptiMakeEvent(
          fn.c_str(), &p, (uint16_t)xpti::trace_event_type_t::algorithm,
          xpti::trace_activity_type_t::active, &m_instance_id);
      if (e && e->unique_id == uids[index]) {
        uint8_t tp = (index % 10) + 1;
        uint16_t tp_type = (uint16_t)(tp << 1);
        xpti::framework::scoped_notify ev("xpti", tp_type, nullptr, e,
                                          m_instance_id, nullptr);
        notify_count++;
      }
    }
    uint64_t acc = 0;
    for (int i = 0; i < tp_count; ++i) {
      acc += events[i]->instance_id;
    }

    // Accumulator contains 'callback_count' number of
    // instances that are invoked after creation, so
    // each event has 101 instances * tp_count = 1010
    //
    // total instances = callback_count + tp_count;

    row[(int)NColumns::Notifications] = (long double)acc;
    row[(int)NColumns::PassRate] = (long double)(acc) / (notify_count)*100;
  } else {
    tbb::task_arena a(num_threads);

    a.execute([&]() {
      std::atomic<int> notify_count = {0};
      tbb::spin_mutex m_lock;
      tbb::parallel_for(
          tbb::blocked_range<int>(0, tp_count),
          [&](tbb::blocked_range<int> &r) {
            for (uint64_t i = r.begin(); i != r.end(); ++i) {
              int index = (int)i;
              std::string fn = "Function" + std::to_string(i);
              xpti::payload_t p =
                  xpti::payload_t(fn.c_str(), m_source, (int)index,
                                  (int)index % 80, (void *)(i % 10));
              xpti::trace_event_data_t *e = xptiMakeEvent(
                  fn.c_str(), &p, (uint16_t)xpti::trace_event_type_t::algorithm,
                  xpti::trace_activity_type_t::active, &m_instance_id);
              if (e) {
                uids[index] = e->unique_id;
                payloads[index] = e->reserved.payload;
                events[index] = e;
              }
              ++notify_count;
            }
          });

      std::string row_title = "Threads " + std::to_string(num_threads);
      auto &row = t.add_row(run_no, row_title);
      row[(int)NColumns::Threads] = num_threads;

      tbb::parallel_for(
          tbb::blocked_range<int>(tp_count, callback_count),
          [&](tbb::blocked_range<int> &r) {
            for (int i = r.begin(); i != r.end(); ++i) {
              int index = (int)i % tp_count;
              void *addr = (void *)(index % 10);
              std::string fn = "Function" + std::to_string(index);
              xpti::payload_t p = xpti::payload_t(
                  fn.c_str(), m_source, (int)index, (int)index % 80, addr);
              xpti::trace_event_data_t *e = xptiMakeEvent(
                  fn.c_str(), &p, (uint16_t)xpti::trace_event_type_t::algorithm,
                  xpti::trace_activity_type_t::active, &m_instance_id);
              if (e && e->unique_id == uids[index]) {
                uint8_t tp = (index % 10) + 1;
                uint16_t tp_type = (uint16_t)(tp << 1);
                xpti::framework::scoped_notify ev("xpti", tp_type, nullptr, e,
                                                  m_instance_id, nullptr);
                notify_count++;
              }
            }
          });

      uint64_t acc = 0;
      for (int i = 0; i < tp_count; ++i) {
        acc += events[i]->instance_id;
      }

      row[(int)NColumns::Notifications] = (long double)acc;
      row[(int)NColumns::PassRate] = (double)(acc) / (notify_count)*100;
    });
  }
}

void test_correctness::run_notification_tests() {
  test::utils::table_model table;

  test::utils::titles_t columns{"Threads", "Notify", "Pass rate"};
  std::cout << std::setw(25) << "Notification Tests\n";
  table.set_headers(columns);

  uint8_t sid = xptiRegisterStream("xpti");
  // We do not need to register callback for correctness tests

  if (m_threads.size()) {
    int run_no = 0;
    for (auto thread : m_threads) {
      run_notification_test_threads(run_no++, thread, table);
    }
  }

  table.print();
}

} // namespace semantic
} // namespace test
