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
#include "tbb/concurrent_vector.h"
#include "tbb/parallel_for.h"
#include "tbb/spin_mutex.h"
#include "tbb/task_arena.h"
#include "tbb/task_group.h"

#include "cl_processor.hpp"
#include "xpti_trace_framework.h"

#include <atomic>
#include <chrono>
#include <random>

static void tpCallback(uint16_t trace_type, xpti::trace_event_data_t *parent,
                       xpti::trace_event_data_t *event, uint64_t instance,
                       const void *ud) {}

namespace test {
void registerCallbacks(uint8_t sid) {
  xptiRegisterCallback(sid, (uint16_t)xpti::trace_point_type_t::graph_create,
                       tpCallback);
  xptiRegisterCallback(sid, (uint16_t)xpti::trace_point_type_t::node_create,
                       tpCallback);
  xptiRegisterCallback(sid, (uint16_t)xpti::trace_point_type_t::edge_create,
                       tpCallback);
  xptiRegisterCallback(sid, (uint16_t)xpti::trace_point_type_t::region_begin,
                       tpCallback);
  xptiRegisterCallback(sid, (uint16_t)xpti::trace_point_type_t::region_end,
                       tpCallback);
  xptiRegisterCallback(sid, (uint16_t)xpti::trace_point_type_t::task_begin,
                       tpCallback);
  xptiRegisterCallback(sid, (uint16_t)xpti::trace_point_type_t::task_end,
                       tpCallback);
  xptiRegisterCallback(sid, (uint16_t)xpti::trace_point_type_t::barrier_begin,
                       tpCallback);
  xptiRegisterCallback(sid, (uint16_t)xpti::trace_point_type_t::barrier_end,
                       tpCallback);
  xptiRegisterCallback(sid, (uint16_t)xpti::trace_point_type_t::lock_begin,
                       tpCallback);
  xptiRegisterCallback(sid, (uint16_t)xpti::trace_point_type_t::lock_end,
                       tpCallback);
  xptiRegisterCallback(sid, (uint16_t)xpti::trace_point_type_t::transfer_begin,
                       tpCallback);
  xptiRegisterCallback(sid, (uint16_t)xpti::trace_point_type_t::transfer_end,
                       tpCallback);
  xptiRegisterCallback(sid, (uint16_t)xpti::trace_point_type_t::thread_begin,
                       tpCallback);
  xptiRegisterCallback(sid, (uint16_t)xpti::trace_point_type_t::thread_end,
                       tpCallback);
  xptiRegisterCallback(sid, (uint16_t)xpti::trace_point_type_t::wait_begin,
                       tpCallback);
  xptiRegisterCallback(sid, (uint16_t)xpti::trace_point_type_t::wait_end,
                       tpCallback);
  xptiRegisterCallback(sid, (uint16_t)xpti::trace_point_type_t::signal,
                       tpCallback);
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

void TestCorrectness::runStringTableTestThreads(
    int RunNo, int NumThreads, test::utils::TableModel &Model) {
  xptiReset();
  constexpr int NumStrings = 1000;

  if (!NumThreads) {
    std::vector<char *> Strings;
    std::vector<xpti::string_id_t> IDs;
    IDs.resize(NumStrings);
    Strings.resize(NumStrings);
    for (int i = 0; i < NumStrings; ++i) {
      char *TableStrRef = nullptr;
      std::string StrName = "Function" + std::to_string(i);
      IDs[i] = xptiRegisterString(StrName.c_str(), &TableStrRef);
      Strings[i] = TableStrRef;
    }
    auto &ModelRow = Model.addRow(RunNo, "Serial");
    ModelRow[(int)STColumns::Threads] = NumThreads;
    ModelRow[(int)STColumns::Insertions] = (long double)Strings.size();
    int LookupCount = 0;
    for (int i = 0; i < Strings.size(); ++i) {
      const char *TableStrRef = xptiLookupString(IDs[i]);
      if (TableStrRef == Strings[i])
        ++LookupCount;
    }
    ModelRow[(int)STColumns::Lookups] = LookupCount;
    int DuplicateCount = 0;
    for (int i = 0; i < Strings.size(); ++i) {
      char *TableStrRef = nullptr;
      std::string StrName = "Function" + std::to_string(i);
      xpti::string_id_t id = xptiRegisterString(StrName.c_str(), &TableStrRef);
      if (StrName == TableStrRef && id == IDs[i] && TableStrRef == Strings[i])
        ++DuplicateCount;
    }
    ModelRow[(int)STColumns::DuplicateInserts] = DuplicateCount;
    ModelRow[(int)STColumns::PassRate] =
        (double)(Strings.size() + LookupCount + DuplicateCount) /
        (NumStrings * 3) * 100;
  } else {
    tbb::task_arena a(NumThreads);

    a.execute([&]() {
      std::vector<char *> Strings;
      std::vector<xpti::string_id_t> IDs;
      Strings.resize(NumStrings);
      IDs.resize(NumStrings);
      tbb::parallel_for(
          tbb::blocked_range<int>(0, NumStrings),
          [&](tbb::blocked_range<int> &r) {
            for (int i = r.begin(); i != r.end(); ++i) {
              char *TableStrRef = nullptr;
              std::string StrName = "Function" + std::to_string(i);
              IDs[i] = xptiRegisterString(StrName.c_str(), &TableStrRef);
              Strings[i] = TableStrRef;
            }
          });

      std::string RowTitle = "Threads " + std::to_string(NumThreads);
      auto &ModelRow = Model.addRow(RunNo, RowTitle);
      ModelRow[(int)STColumns::Threads] = NumThreads;
      ModelRow[(int)STColumns::Insertions] = (long double)Strings.size();
      std::atomic<int> LookupCount = {0}, DuplicateCount = {0};
      tbb::parallel_for(tbb::blocked_range<int>(0, NumStrings),
                        [&](tbb::blocked_range<int> &r) {
                          for (int i = r.begin(); i != r.end(); ++i) {
                            const char *TableStrRef = xptiLookupString(IDs[i]);
                            if (TableStrRef == Strings[i])
                              ++LookupCount;
                          }
                        });
      ModelRow[(int)STColumns::Lookups] = LookupCount;
      tbb::parallel_for(tbb::blocked_range<int>(0, NumStrings),
                        [&](tbb::blocked_range<int> &r) {
                          for (int i = r.begin(); i != r.end(); ++i) {
                            char *TableStrRef = nullptr;
                            std::string StrName =
                                "Function" + std::to_string(i);
                            xpti::string_id_t id = xptiRegisterString(
                                StrName.c_str(), &TableStrRef);
                            if (StrName == TableStrRef && id == IDs[i] &&
                                TableStrRef == Strings[i])
                              ++DuplicateCount;
                          }
                        });
      ModelRow[(int)STColumns::DuplicateInserts] = DuplicateCount;

      ModelRow[(int)STColumns::PassRate] =
          (double)(Strings.size() + LookupCount + DuplicateCount) /
          (NumStrings * 3) * 100;
    });
  }
}

void TestCorrectness::runStringTableTests() {
  test::utils::TableModel Model;

  test::utils::titles_t Columns{"Threads", "Insert", "Lookup", "Duplicate",
                                "Pass rate"};
  std::cout << std::setw(25) << "String Table Tests\n";
  Model.setHeaders(Columns);

  if (MThreads.size()) {
    int RunNo = 0;
    for (auto Thread : MThreads) {
      runStringTableTestThreads(RunNo++, Thread, Model);
    }
  }

  Model.print();
}

void TestCorrectness::runTracepointTestThreads(int RunNo, int NumThreads,
                                               test::utils::TableModel &Model) {
  xptiReset();
  constexpr int TracepointCount = 1000;

  if (!NumThreads) {
    std::vector<xpti::payload_t *> Payloads;
    std::vector<uint64_t> UIds;
    std::vector<xpti::trace_event_data_t *> Events;
    Payloads.resize(TracepointCount);
    UIds.resize(TracepointCount);
    Events.resize(TracepointCount);

    for (uint64_t i = 0; i < TracepointCount; ++i) {
      std::string fn = "Function" + std::to_string(i);
      xpti::payload_t P = xpti::payload_t(fn.c_str(), MSource, (int)i,
                                          (int)(i % 80), (void *)i);
      xpti::trace_event_data_t *Ev = xptiMakeEvent(
          fn.c_str(), &P, (uint16_t)xpti::trace_event_type_t::algorithm,
          xpti::trace_activity_type_t::active, &MInstanceID);
      if (Ev) {
        UIds[i] = Ev->unique_id;
        Payloads[i] = Ev->reserved.payload;
        Events[i] = Ev;
      }
    }
    auto &ModelRow = Model.addRow(RunNo, "Serial");
    ModelRow[(int)TPColumns::Threads] = NumThreads;
    ModelRow[(int)TPColumns::Insertions] = (long double)Events.size();

    std::atomic<int> LookupCount = {0};
    for (int i = 0; i < Events.size(); ++i) {
      const xpti::trace_event_data_t *Ev = xptiFindEvent(UIds[i]);
      if (Ev && Ev->unique_id == UIds[i])
        ++LookupCount;
    }
    ModelRow[(int)TPColumns::Lookups] = LookupCount;
    std::atomic<int> DuplicateCount = {0};
    std::atomic<int> PayloadCount = {0};
    for (uint64_t i = 0; i < Events.size(); ++i) {
      std::string fn = "Function" + std::to_string(i);
      xpti::payload_t P =
          xpti::payload_t(fn.c_str(), MSource, (int)i, (int)i % 80, (void *)i);
      xpti::trace_event_data_t *Ev = xptiMakeEvent(
          fn.c_str(), &P, (uint16_t)xpti::trace_event_type_t::algorithm,
          xpti::trace_activity_type_t::active, &MInstanceID);
      if (Ev) {
        if (Ev->unique_id == UIds[i]) {
          ++DuplicateCount;
        }
        xpti::payload_t *RP = Ev->reserved.payload;
        if (Ev->unique_id == UIds[i] && RP &&
            std::string(RP->name) == std::string(P.name) &&
            std::string(RP->source_file) == std::string(P.source_file) &&
            RP->line_no == P.line_no && RP->column_no == P.column_no)
          ++PayloadCount;
      }
    }
    ModelRow[(int)TPColumns::DuplicateInserts] = DuplicateCount;
    ModelRow[(int)TPColumns::PayloadLookup] = PayloadCount;
    ModelRow[(int)TPColumns::PassRate] =
        (double)(Events.size() + LookupCount + DuplicateCount + PayloadCount) /
        (TracepointCount * 4) * 100;
  } else {
    tbb::task_arena a(NumThreads);

    a.execute([&]() {
      std::vector<xpti::payload_t *> Payloads;
      std::vector<int64_t> UIds;
      std::vector<xpti::trace_event_data_t *> Events;
      Payloads.resize(TracepointCount);
      UIds.resize(TracepointCount);
      Events.resize(TracepointCount);

      tbb::spin_mutex MLock;
      tbb::parallel_for(
          tbb::blocked_range<int>(0, TracepointCount),
          [&](tbb::blocked_range<int> &r) {
            for (uint64_t i = r.begin(); i != r.end(); ++i) {
              std::string fn = "Function" + std::to_string(i);
              xpti::payload_t P = xpti::payload_t(fn.c_str(), MSource, (int)i,
                                                  (int)i % 80, (void *)i);
              xpti::trace_event_data_t *Ev = xptiMakeEvent(
                  fn.c_str(), &P, (uint16_t)xpti::trace_event_type_t::algorithm,
                  xpti::trace_activity_type_t::active, &MInstanceID);
              if (Ev) {
                UIds[i] = Ev->unique_id;
                Payloads[i] = Ev->reserved.payload;
                Events[i] = Ev;
              }
            }
          });

      std::string RowTitle = "Threads " + std::to_string(NumThreads);
      auto &ModelRow = Model.addRow(RunNo, RowTitle);
      ModelRow[(int)TPColumns::Threads] = NumThreads;
      ModelRow[(int)TPColumns::Insertions] = (long double)Events.size();
      std::atomic<int> LookupCount = {0}, DuplicateCount = {0},
                       PayloadCount = {0};
      tbb::parallel_for(tbb::blocked_range<int>(0, TracepointCount),
                        [&](tbb::blocked_range<int> &r) {
                          for (int i = r.begin(); i != r.end(); ++i) {
                            const xpti::trace_event_data_t *Ev =
                                xptiFindEvent(UIds[i]);
                            if (Ev && Ev->unique_id == UIds[i])
                              LookupCount++;
                          }
                        });

      ModelRow[(int)TPColumns::Lookups] = LookupCount;
      tbb::parallel_for(
          tbb::blocked_range<int>(0, TracepointCount),
          [&](tbb::blocked_range<int> &r) {
            for (uint64_t i = r.begin(); i != r.end(); ++i) {
              std::string fn = "Function" + std::to_string(i);
              xpti::payload_t P = xpti::payload_t(fn.c_str(), MSource, (int)i,
                                                  (int)i % 80, (void *)i);
              xpti::trace_event_data_t *Ev = xptiMakeEvent(
                  fn.c_str(), &P, (uint16_t)xpti::trace_event_type_t::algorithm,
                  xpti::trace_activity_type_t::active, &MInstanceID);
              if (Ev) {
                if (Ev->unique_id == UIds[i]) {
                  ++DuplicateCount;
                }
                xpti::payload_t *RP = Ev->reserved.payload;
                if (Ev->unique_id == UIds[i] && RP &&
                    std::string(RP->name) == std::string(P.name) &&
                    std::string(RP->source_file) ==
                        std::string(P.source_file) &&
                    RP->line_no == P.line_no && RP->column_no == P.column_no)
                  ++PayloadCount;
              }
            }
          });
      ModelRow[(int)TPColumns::DuplicateInserts] = DuplicateCount;
      ModelRow[(int)TPColumns::PayloadLookup] = PayloadCount;
      ModelRow[(int)TPColumns::PassRate] =
          (double)(Events.size() + LookupCount + DuplicateCount +
                   PayloadCount) /
          (TracepointCount * 4) * 100;
    });
  }
}

void TestCorrectness::runTracepointTests() {
  test::utils::TableModel Model;

  test::utils::titles_t Columns{"Threads",   "Create",  "Lookup",
                                "Duplicate", "Payload", "Pass rate"};
  std::cout << std::setw(25) << "Tracepoint Tests\n";
  Model.setHeaders(Columns);

  if (MThreads.size()) {
    int RunNo = 0;
    for (auto Thread : MThreads) {
      runTracepointTestThreads(RunNo++, Thread, Model);
    }
  }

  Model.print();
}

void TestCorrectness::runNotificationTestThreads(
    int RunNo, int NumThreads, test::utils::TableModel &Model) {
  xptiReset();
  int TPCount = 30, CallbackCount = TPCount * 30;
  std::vector<xpti::payload_t *> Payloads;
  std::vector<int64_t> UIds;
  std::vector<xpti::trace_event_data_t *> Events;
  Payloads.resize(TPCount);
  UIds.resize(TPCount);
  Events.resize(TPCount);

  if (!NumThreads) {

    // assumes tp creation is thread safe
    std::atomic<int> NotifyCount = {0};
    for (uint64_t i = 0; i < TPCount; ++i) {
      int Index = (int)i;
      std::string fn = "Function" + std::to_string(i);
      xpti::payload_t P = xpti::payload_t(fn.c_str(), MSource, Index,
                                          Index % 80, (void *)(i % 10));
      xpti::trace_event_data_t *Ev = xptiMakeEvent(
          fn.c_str(), &P, (uint16_t)xpti::trace_event_type_t::algorithm,
          xpti::trace_activity_type_t::active, &MInstanceID);
      if (Ev) {
        UIds[Index] = Ev->unique_id;
        Payloads[Index] = Ev->reserved.payload;
        Events[Index] = Ev;
      }
      NotifyCount++;
    }

    auto &ModelRow = Model.addRow(RunNo, "Serial");
    ModelRow[(int)NColumns::Threads] = NumThreads;

    for (int i = TPCount; i < CallbackCount; ++i) {
      int Index = (int)i % TPCount;
      void *Address = (void *)(Index % 10);
      std::string fn = "Function" + std::to_string(Index);
      xpti::payload_t P = xpti::payload_t(fn.c_str(), MSource, (int)Index,
                                          (int)Index % 80, Address);
      xpti::trace_event_data_t *Ev = xptiMakeEvent(
          fn.c_str(), &P, (uint16_t)xpti::trace_event_type_t::algorithm,
          xpti::trace_activity_type_t::active, &MInstanceID);
      if (Ev && Ev->unique_id == UIds[Index]) {
        uint8_t TP = (Index % 10) + 1;
        uint16_t TPType = (uint16_t)(TP << 1);
        xpti::framework::scoped_notify ev("xpti", TPType, nullptr, Ev,
                                          MInstanceID, nullptr);
        NotifyCount++;
      }
    }
    uint64_t Acc = 0;
    for (int i = 0; i < TPCount; ++i) {
      Acc += Events[i]->instance_id;
    }

    // Accumulator contains 'CallbackCount' number of
    // instances that are invoked after creation, so
    // each event has 101 instances * TPCount = 1010
    //
    // total instances = CallbackCount + TPCount;

    ModelRow[(int)NColumns::Notifications] = (long double)Acc;
    ModelRow[(int)NColumns::PassRate] = (long double)(Acc) / (NotifyCount)*100;
  } else {
    tbb::task_arena a(NumThreads);

    a.execute([&]() {
      std::atomic<int> NotifyCount = {0};
      tbb::spin_mutex MLock;
      tbb::parallel_for(
          tbb::blocked_range<int>(0, TPCount), [&](tbb::blocked_range<int> &r) {
            for (uint64_t i = r.begin(); i != r.end(); ++i) {
              int Index = (int)i;
              std::string fn = "Function" + std::to_string(i);
              xpti::payload_t P =
                  xpti::payload_t(fn.c_str(), MSource, (int)Index,
                                  (int)Index % 80, (void *)(i % 10));
              xpti::trace_event_data_t *Ev = xptiMakeEvent(
                  fn.c_str(), &P, (uint16_t)xpti::trace_event_type_t::algorithm,
                  xpti::trace_activity_type_t::active, &MInstanceID);
              if (Ev) {
                UIds[Index] = Ev->unique_id;
                Payloads[Index] = Ev->reserved.payload;
                Events[Index] = Ev;
              }
              ++NotifyCount;
            }
          });

      std::string RowTitle = "Threads " + std::to_string(NumThreads);
      auto &ModelRow = Model.addRow(RunNo, RowTitle);
      ModelRow[(int)NColumns::Threads] = NumThreads;

      tbb::parallel_for(
          tbb::blocked_range<int>(TPCount, CallbackCount),
          [&](tbb::blocked_range<int> &r) {
            for (int i = r.begin(); i != r.end(); ++i) {
              int Index = (int)i % TPCount;
              void *Address = (void *)(Index % 10);
              std::string fn = "Function" + std::to_string(Index);
              xpti::payload_t P = xpti::payload_t(
                  fn.c_str(), MSource, (int)Index, (int)Index % 80, Address);
              xpti::trace_event_data_t *Ev = xptiMakeEvent(
                  fn.c_str(), &P, (uint16_t)xpti::trace_event_type_t::algorithm,
                  xpti::trace_activity_type_t::active, &MInstanceID);
              if (Ev && Ev->unique_id == UIds[Index]) {
                uint8_t TP = (Index % 10) + 1;
                uint16_t TPType = (uint16_t)(TP << 1);
                xpti::framework::scoped_notify ev("xpti", TPType, nullptr, Ev,
                                                  MInstanceID, nullptr);
                NotifyCount++;
              }
            }
          });

      uint64_t Acc = 0;
      for (int i = 0; i < TPCount; ++i) {
        Acc += Events[i]->instance_id;
      }

      ModelRow[(int)NColumns::Notifications] = (long double)Acc;
      ModelRow[(int)NColumns::PassRate] = (double)(Acc) / (NotifyCount)*100;
    });
  }
}

void TestCorrectness::runNotificationTests() {
  test::utils::TableModel Model;

  test::utils::titles_t Columns{"Threads", "Notify", "Pass rate"};
  std::cout << std::setw(25) << "Notification Tests\n";
  Model.setHeaders(Columns);

  uint8_t SID = xptiRegisterStream("xpti");
  // We do not need to register callback for correctness tests

  if (MThreads.size()) {
    int RunNo = 0;
    for (auto Thread : MThreads) {
      runNotificationTestThreads(RunNo++, Thread, Model);
    }
  }

  Model.print();
}

} // namespace semantic
} // namespace test
