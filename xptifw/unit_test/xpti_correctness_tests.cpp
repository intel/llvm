//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
#include "xpti/xpti_trace_framework.h"
#include "xpti/xpti_trace_framework.hpp"

#include <gtest/gtest.h>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <unordered_map>

static bool TPCBCalled = false;

class xptiCorrectnessTest : public ::testing::Test {
protected:
  void SetUp() override { TPCBCalled = false; }

  void TearDown() override { xptiReset(); }
};

void tpCallback(uint16_t trace_type, xpti::trace_event_data_t *parent,
                xpti::trace_event_data_t *event, uint64_t instance,
                const void *user_data) {
  TPCBCalled = true;
}

#define NOTIFY(stream, tt, event)                                              \
  {                                                                            \
    int data;                                                                  \
    xpti::result_t Result =                                                    \
        xptiNotifySubscribers(stream, tt, nullptr, event, 0, (void *)(&data)); \
    EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);                    \
    EXPECT_TRUE(TPCBCalled);                                                   \
    TPCBCalled = false;                                                        \
  }

TEST_F(xptiCorrectnessTest, xptiMakeEvent) {
  uint64_t Instance = 0;
  xpti::payload_t p("foo", "foo.cpp", 1, 0, (void *)13);
  auto Result =
      xptiMakeEvent("foo", &p, 0, (xpti::trace_activity_type_t)1, &Instance);
  ASSERT_NE(Result, nullptr);
  p = xpti::payload_t("foo", "foo.cpp", 1, 0, (void *)13);
  auto NewResult =
      xptiMakeEvent("foo", &p, 0, (xpti::trace_activity_type_t)1, &Instance);
  ASSERT_NE(NewResult, nullptr);
  // Since we create a new trace event for each instance, the instance IDs will
  // be different for each event
  EXPECT_NE(Result, NewResult);
  EXPECT_NE(Result->instance_id, NewResult->instance_id);
  EXPECT_GT(NewResult->instance_id, Result->instance_id);
  EXPECT_NE(Result->unique_id, NewResult->unique_id);
  // We create a new payload instance for each tracepoint instance
  EXPECT_NE(Result->reserved.payload, NewResult->reserved.payload);
  EXPECT_STREQ(Result->reserved.payload->name, "foo");
  EXPECT_STREQ(Result->reserved.payload->source_file, "foo.cpp");
  EXPECT_EQ(Result->reserved.payload->line_no, 1u);
}

TEST_F(xptiCorrectnessTest, xptiUnregisterStream) {
  xptiForceSetTraceEnabled(true);
  auto ID = xptiRegisterStream("foo1");
  EXPECT_NE(ID, xpti::invalid_id);
  auto Result = xptiUnregisterStream("foo2");
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_NOTFOUND);
  // Event though stream exists, no callbacks registered
  auto NewResult = xptiUnregisterStream("foo1");
  EXPECT_EQ(NewResult, xpti::result_t::XPTI_RESULT_NOTFOUND);
  // Register again
  ID = xptiRegisterStream("foo1");
  EXPECT_NE(ID, xpti::invalid_id);
  Result = xptiRegisterCallback(
      ID, (uint16_t)xpti::trace_point_type_t::graph_create, tpCallback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      ID, (uint16_t)xpti::trace_point_type_t::node_create, tpCallback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      ID, (uint16_t)xpti::trace_point_type_t::edge_create, tpCallback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      ID, (uint16_t)xpti::trace_point_type_t::region_begin, tpCallback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      ID, (uint16_t)xpti::trace_point_type_t::region_end, tpCallback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      ID, (uint16_t)xpti::trace_point_type_t::task_begin, tpCallback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      ID, (uint16_t)xpti::trace_point_type_t::task_end, tpCallback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);

  bool Check = xptiCheckTraceEnabled(
      ID, (uint16_t)xpti::trace_point_type_t::edge_create);
  EXPECT_EQ(Check, true);
  Check =
      xptiCheckTraceEnabled(ID, (uint16_t)xpti::trace_point_type_t::task_begin);
  EXPECT_EQ(Check, true);
  auto Result2 = xptiUnregisterStream("foo1");
  EXPECT_EQ(Result2, xpti::result_t::XPTI_RESULT_SUCCESS);
  Check =
      xptiCheckTraceEnabled(ID, (uint16_t)xpti::trace_point_type_t::task_begin);
  EXPECT_EQ(Check, false);
  Check =
      xptiCheckTraceEnabled(ID, (uint16_t)xpti::trace_point_type_t::task_end);
  EXPECT_EQ(Check, false);
  Check = xptiCheckTraceEnabled(
      ID, (uint16_t)xpti::trace_point_type_t::edge_create);
  EXPECT_EQ(Check, false);
  Check = xptiCheckTraceEnabled(
      ID, (uint16_t)xpti::trace_point_type_t::graph_create);
  EXPECT_EQ(Check, false);
  Check = xptiCheckTraceEnabled(
      ID, (uint16_t)xpti::trace_point_type_t::node_create);
  EXPECT_EQ(Check, false);
  Check = xptiCheckTraceEnabled(
      ID, (uint16_t)xpti::trace_point_type_t::region_begin);
  EXPECT_EQ(Check, false);
  Check =
      xptiCheckTraceEnabled(ID, (uint16_t)xpti::trace_point_type_t::region_end);
  EXPECT_EQ(Check, false);
}

TEST_F(xptiCorrectnessTest, xptiCreateTracepoint) {
  uint64_t Instance = 0, Instance2 = 0;
  xpti::payload_t p("foo", "foo.cpp", 1, 0, (void *)13);
  auto TP = xptiCreateTracepoint(p.name, p.source_file, p.line_no, p.column_no);
  xpti::trace_event_data_t *Ev = TP->event_ref();
  ASSERT_NE(Ev, nullptr);
  p = xpti::payload_t("foo", "foo.cpp", 1, 0, (void *)13);
  auto TP1 =
      xptiCreateTracepoint(p.name, p.source_file, p.line_no, p.column_no);
  xpti::trace_event_data_t *Ev1 = TP1->event_ref();
  ASSERT_NE(Ev1, nullptr);
  EXPECT_NE(Ev->instance_id, Ev1->instance_id);
  EXPECT_GT(Ev1->instance_id, Ev->instance_id);
  // Since we create a new trace event for each instance, the instance IDs will
  // be different for each event
  EXPECT_NE(TP, TP1);
  EXPECT_EQ(TP->instance(), Ev->instance_id);
  EXPECT_NE(Ev->unique_id, xpti::invalid_uid);
  EXPECT_NE(Ev1->unique_id, xpti::invalid_uid);
  EXPECT_NE(Ev1->unique_id, Ev->unique_id);
  // We have a new instance of Payload associated with each Event
  EXPECT_NE(Ev->reserved.payload, Ev1->reserved.payload);
  EXPECT_STREQ(Ev->reserved.payload->name, "foo");
  EXPECT_STREQ(Ev->reserved.payload->source_file, "foo.cpp");
  EXPECT_EQ(Ev->reserved.payload->line_no, 1u);
  bool test =
      Ev->flags & static_cast<uint64_t>(xpti::trace_event_flag_t::UIDAvailable);
  EXPECT_EQ(test, true);
  test = Ev->flags &
         static_cast<uint64_t>(xpti::trace_event_flag_t::PayloadAvailable);
  EXPECT_EQ(test, true);
  test = Ev->flags &
         static_cast<uint64_t>(xpti::trace_event_flag_t::ActivityTypeAvailable);
  EXPECT_EQ(test, true);
  test = Ev->flags &
         static_cast<uint64_t>(xpti::trace_event_flag_t::EventTypeAvailable);
  EXPECT_EQ(test, true);
  test = Ev1->flags &
         static_cast<uint64_t>(xpti::trace_event_flag_t::UIDAvailable);
  EXPECT_EQ(test, true);
  test = Ev1->flags &
         static_cast<uint64_t>(xpti::trace_event_flag_t::PayloadAvailable);
  EXPECT_EQ(test, true);
  test = Ev1->flags &
         static_cast<uint64_t>(xpti::trace_event_flag_t::ActivityTypeAvailable);
  EXPECT_EQ(test, true);
  test = Ev1->flags &
         static_cast<uint64_t>(xpti::trace_event_flag_t::EventTypeAvailable);
  EXPECT_EQ(test, true);
  EXPECT_EQ(Ev->source_id, xpti::invalid_uid);
  EXPECT_EQ(Ev->target_id, xpti::invalid_uid);
  EXPECT_EQ(Ev1->source_id, xpti::invalid_uid);
  EXPECT_EQ(Ev1->target_id, xpti::invalid_uid);
}

TEST_F(xptiCorrectnessTest, xptiRegisterString) {
  char *TStr = nullptr;
  auto ID = xptiRegisterString("foo", &TStr);
  EXPECT_NE(ID, xpti::invalid_id);
  EXPECT_NE(TStr, nullptr);
  EXPECT_STREQ("foo", TStr);

  const char *LUTStr = xptiLookupString(ID);
  EXPECT_EQ(TStr, LUTStr);
  EXPECT_STREQ(LUTStr, TStr);
}

TEST_F(xptiCorrectnessTest, xptiTracePointScopeDataTest) {
  xpti::payload_t p("foo", "foo.cpp", 1, 4, nullptr);
  auto ScopeData = xptiGetTracepointScopeData();
  EXPECT_EQ(ScopeData, nullptr);
  auto TP = xptiCreateTracepoint(p.name, p.source_file, p.line_no, p.column_no);

  auto Result = xptiSetTracepointScopeData(TP);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);

  ScopeData = xptiGetTracepointScopeData();
  EXPECT_EQ(const_cast<xpti_tracepoint_t *>(ScopeData), TP);

  xptiUnsetTracepointScopeData();
}

void nestedTest(xpti::payload_t *p, std::vector<uint64_t> &uids) {
  xpti::framework::tracepoint_t t(p);
  uint64_t hash = t.universal_id();
  uids.push_back(hash);

  if (uids.size() < 5) {
    xpti::payload_t pp;
    nestedTest(&pp, uids);
  }
}

TEST_F(xptiCorrectnessTest, xptiTracePointTest) {
  std::vector<uint64_t> uids;
  xpti::payload_t p("foo", "foo.cpp", 10, 0, (void *)(0xdeadbeefull));

  (void)xptiRegisterPayload(&p);
  EXPECT_NE(p.internal, 0);
  EXPECT_NE(p.flags, 0);

  uint64_t id = xpti::invalid_uid;
  nestedTest(&p, uids);
  for (auto &e : uids) {
    EXPECT_NE(e, xpti::invalid_uid);
    if (id != xpti::invalid_uid) {
      EXPECT_EQ(e, id);
      id = e;
    }
  }

  uids.clear();
  xpti::payload_t p1("bar", "foo.cpp", 15, 0, (void *)(0xdeaddeadull));

  (void)xptiRegisterPayload(&p1);

  id = xpti::invalid_uid;
  nestedTest(&p1, uids);
  for (auto &e : uids) {
    EXPECT_NE(e, xpti::invalid_uid);
    if (id != xpti::invalid_uid) {
      EXPECT_EQ(e, id);
      id = e;
    }
  }
}

void nestedScopeTest(xpti::payload_t *p, std::vector<uint64_t> &uids) {
  xpti::framework::tracepoint_scope_t t(p->name, p->source_file, p->line_no,
                                        p->column_no, false);
  xpti::hash_t Hash;

  auto Tuid = t.uid64();
  uids.push_back(Tuid);

  if (uids.size() < 5) {
    xpti::payload_t pp;
    nestedScopeTest(&pp, uids);
  }
}

TEST_F(xptiCorrectnessTest, xptiTracePointScopeTest) {
  std::vector<uint64_t> uids;
  xpti::payload_t p("mickey", "mouse.cpp", 10, 0, nullptr);
  auto TP = xptiCreateTracepoint(p.name, p.source_file, p.line_no, p.column_no);
  xpti::trace_event_data_t *Ev1 = TP->event_ref();
  auto UID = Ev1->unique_id;

  uint64_t id = xpti::invalid_uid;
  nestedScopeTest(&p, uids);
  for (auto &e : uids) {
    EXPECT_NE(e, xpti::invalid_uid);
    if (id != xpti::invalid_uid) {
      EXPECT_EQ(e, id);
    }
    id = e;
  }
  // UID should be able top query an event as it was created for the first
  // instance of seeing the payload; nestedScopeTest() creates its own event
  // that is released when tracepoint_scope_t goes out of scope.
  auto Event = xptiLookupEvent(UID);
  EXPECT_NE(Event, nullptr);
  // UID should be able to query the payload data here as it is still valid
  auto Payload = xptiLookupPayload(UID);
  EXPECT_NE(Payload, nullptr);

  uids.clear();
  xpti::payload_t p1("bar", "foo.cpp", 15, 0, nullptr);
  {
    uint64_t uid;
    {
      xpti::framework::tracepoint_scope_t t(p1.name, p1.source_file, p1.line_no,
                                            p1.column_no, false);
      uid = t.uid64();
      EXPECT_NE(t.traceEvent(), nullptr);
      auto ScopeData = xptiGetTracepointScopeData();
      // it has beeen set by tracepoint_scope_t
      EXPECT_NE(ScopeData, nullptr);
      id = xpti::invalid_uid;
      nestedScopeTest(&p1, uids);
      for (auto &e : uids) {
        EXPECT_NE(e, xpti::invalid_uid);
        if (id != xpti::invalid_uid) {
          EXPECT_EQ(e, id);
        }
        id = e;
      }
      EXPECT_NE(t.traceEvent(), nullptr);

      // UID should be able to query both payload and event in this case
      auto UID = t.uid64();
      auto Event = xptiLookupEvent(UID);
      EXPECT_NE(Event, nullptr);
      auto Payload = xptiLookupPayload(UID);
      EXPECT_NE(Payload, nullptr);
    }
    // The Event for the uid would have goine out of scope and deleted
    auto Event = xptiLookupEvent(uid);
    EXPECT_EQ(Event, nullptr);
  }
}

TEST_F(xptiCorrectnessTest, xptiQueryMetadata) {
  xpti::uid128_t Id0;
  uint64_t instance;
  /// Simulates the specialization of a Kernel as used by MKL where
  /// the same kernel may be compiled multiple times
  xpti::payload_t Payload("foo", "foo.cpp", 1, 0, (void *)&Id0);
  auto Result = xptiMakeEvent("foo", &Payload, 0,
                              (xpti::trace_activity_type_t)1, &instance);
  auto Result1 = xptiMakeEvent("foo", &Payload, 0,
                               (xpti::trace_activity_type_t)1, &instance);
  auto Metadata = xptiQueryMetadata(Result);
  auto Metadata1 = xptiQueryMetadata(Result1);
  EXPECT_NE(Metadata, Metadata1);
  EXPECT_EQ(Metadata->size(), 0);
  EXPECT_EQ(Metadata1->size(), 0);
}

template <typename T> inline T queryMetadata(const xpti::object_id_t &ID) {
  xpti::object_data_t RawData = xptiLookupObject(ID);
  assert(RawData.size == sizeof(T));
  T Value = *reinterpret_cast<const T *>(RawData.data);
  return Value;
}

template <typename T>
T getMetadataByKey(xpti::metadata_t *Metadata, const char *key) {
  char *RetString;
  auto StringId = xptiRegisterString(key, &RetString);
  auto Item = Metadata->find(StringId);
  return queryMetadata<T>(Item->second);
}

TEST_F(xptiCorrectnessTest, xptiAddMetadata) {
  xpti::uid128_t Id0;
  uint64_t instance;
  /// Simulates the specialization of a Kernel as used by MKL where
  /// the same kernel may be compiled multiple times
  xpti::payload_t Payload("foo", "foo.cpp", 1, 0, (void *)&Id0);
  auto Result = xptiMakeEvent("foo", &Payload, 0,
                              (xpti::trace_activity_type_t)1, &instance);
  auto Result1 = xptiMakeEvent("foo", &Payload, 0,
                               (xpti::trace_activity_type_t)1, &instance);
  auto Metadata = xptiQueryMetadata(Result);
  auto Metadata1 = xptiQueryMetadata(Result1);
  EXPECT_NE(Metadata, Metadata1);
  EXPECT_EQ(Metadata->size(), 0);
  EXPECT_EQ(Metadata1->size(), 0);
  xpti::addMetadata(Result, "int_value", 1);
  xpti::addMetadata(Result1, "int_value", 2);

  auto Val = getMetadataByKey<int>(Metadata, "int_value");
  auto Val1 = getMetadataByKey<int>(Metadata1, "int_value");
  EXPECT_EQ(Val, 1);
  EXPECT_EQ(Val1, 2);
  EXPECT_NE(Val, Val1);
}

TEST_F(xptiCorrectnessTest, xptiInitializeForDefaultTracePointTypes) {
  // We will test functionality of a subscriber
  // without actually creating a plugin
  uint8_t StreamID = xptiRegisterStream("test_foo");
  auto Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::graph_create, tpCallback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::node_create, tpCallback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::edge_create, tpCallback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::region_begin, tpCallback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::region_end, tpCallback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::task_begin, tpCallback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::task_end, tpCallback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::barrier_begin, tpCallback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::barrier_end, tpCallback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::lock_begin, tpCallback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::lock_end, tpCallback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::transfer_begin, tpCallback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::transfer_end, tpCallback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::thread_begin, tpCallback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::thread_end, tpCallback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::wait_begin, tpCallback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::wait_end, tpCallback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::signal, tpCallback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
}

TEST_F(xptiCorrectnessTest, xptiNotifySubscribersForDefaultTracePointTypes) {
  uint64_t Instance;
  xpti::payload_t p("foo", "foo.cpp", 1, 0, (void *)13);
  xptiForceSetTraceEnabled(true);

  uint8_t StreamID = xptiRegisterStream("test_foo");
  auto Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::graph_create, tpCallback);
  ASSERT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::node_create, tpCallback);
  ASSERT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::edge_create, tpCallback);
  ASSERT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::region_begin, tpCallback);
  ASSERT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::region_end, tpCallback);
  ASSERT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::task_begin, tpCallback);
  ASSERT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::task_end, tpCallback);
  ASSERT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::barrier_begin, tpCallback);
  ASSERT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::barrier_end, tpCallback);
  ASSERT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::lock_begin, tpCallback);
  ASSERT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::lock_end, tpCallback);
  ASSERT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::transfer_begin, tpCallback);
  ASSERT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::transfer_end, tpCallback);
  ASSERT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::thread_begin, tpCallback);
  ASSERT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::thread_end, tpCallback);
  ASSERT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::wait_begin, tpCallback);
  ASSERT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::wait_end, tpCallback);
  ASSERT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::signal, tpCallback);
  ASSERT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);

  auto GE =
      xptiMakeEvent("foo", &p, 0, (xpti::trace_activity_type_t)1, &Instance);
  EXPECT_NE(GE, nullptr);

  NOTIFY(StreamID, (uint16_t)xpti::trace_point_type_t::graph_create, GE);
  NOTIFY(StreamID, (uint16_t)xpti::trace_point_type_t::node_create, GE);
  NOTIFY(StreamID, (uint16_t)xpti::trace_point_type_t::edge_create, GE);
  NOTIFY(StreamID, (uint16_t)xpti::trace_point_type_t::region_begin, GE);
  NOTIFY(StreamID, (uint16_t)xpti::trace_point_type_t::region_end, GE);
  NOTIFY(StreamID, (uint16_t)xpti::trace_point_type_t::task_begin, GE);
  NOTIFY(StreamID, (uint16_t)xpti::trace_point_type_t::task_end, GE);
  NOTIFY(StreamID, (uint16_t)xpti::trace_point_type_t::barrier_begin, GE);
  NOTIFY(StreamID, (uint16_t)xpti::trace_point_type_t::barrier_end, GE);
  NOTIFY(StreamID, (uint16_t)xpti::trace_point_type_t::lock_begin, GE);
  NOTIFY(StreamID, (uint16_t)xpti::trace_point_type_t::lock_end, GE);
  NOTIFY(StreamID, (uint16_t)xpti::trace_point_type_t::transfer_begin, GE);
  NOTIFY(StreamID, (uint16_t)xpti::trace_point_type_t::transfer_end, GE);
  NOTIFY(StreamID, (uint16_t)xpti::trace_point_type_t::thread_begin, GE);
  NOTIFY(StreamID, (uint16_t)xpti::trace_point_type_t::thread_end, GE);
  NOTIFY(StreamID, (uint16_t)xpti::trace_point_type_t::wait_begin, GE);
  NOTIFY(StreamID, (uint16_t)xpti::trace_point_type_t::wait_end, GE);
  NOTIFY(StreamID, (uint16_t)xpti::trace_point_type_t::signal, GE);
}

TEST_F(xptiCorrectnessTest, xptiCheckTraceEnabledForDefaultTracePointTypes) {
  uint64_t Instance;
  xpti::payload_t p("foo", "foo.cpp", 1, 0, (void *)13);
  xptiForceSetTraceEnabled(true);

  // We are testing clean streams by creating new ones as previous tests may
  // have polluted the streams
  uint8_t StreamID1 = xptiRegisterStream("CheckDefault");
  uint8_t StreamID2 = xptiRegisterStream("Check");
  auto Result = xptiRegisterCallback(
      StreamID1, (uint16_t)xpti::trace_point_type_t::graph_create, tpCallback);
  ASSERT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      StreamID1, (uint16_t)xpti::trace_point_type_t::node_create, tpCallback);
  ASSERT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      StreamID1, (uint16_t)xpti::trace_point_type_t::edge_create, tpCallback);
  ASSERT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      StreamID1, (uint16_t)xpti::trace_point_type_t::region_begin, tpCallback);
  ASSERT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      StreamID1, (uint16_t)xpti::trace_point_type_t::region_end, tpCallback);
  ASSERT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      StreamID1, (uint16_t)xpti::trace_point_type_t::task_begin, tpCallback);
  ASSERT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      StreamID1, (uint16_t)xpti::trace_point_type_t::task_end, tpCallback);
  ASSERT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      StreamID1, (uint16_t)xpti::trace_point_type_t::barrier_begin, tpCallback);
  ASSERT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      StreamID1, (uint16_t)xpti::trace_point_type_t::barrier_end, tpCallback);
  ASSERT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);

  Result = xptiRegisterCallback(
      StreamID2, (uint16_t)xpti::trace_point_type_t::lock_begin, tpCallback);
  ASSERT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      StreamID2, (uint16_t)xpti::trace_point_type_t::lock_end, tpCallback);
  ASSERT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      StreamID2, (uint16_t)xpti::trace_point_type_t::transfer_begin,
      tpCallback);
  ASSERT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      StreamID2, (uint16_t)xpti::trace_point_type_t::transfer_end, tpCallback);
  ASSERT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      StreamID2, (uint16_t)xpti::trace_point_type_t::thread_begin, tpCallback);
  ASSERT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      StreamID2, (uint16_t)xpti::trace_point_type_t::thread_end, tpCallback);
  ASSERT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      StreamID2, (uint16_t)xpti::trace_point_type_t::wait_begin, tpCallback);
  ASSERT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      StreamID2, (uint16_t)xpti::trace_point_type_t::wait_end, tpCallback);
  ASSERT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      StreamID2, (uint16_t)xpti::trace_point_type_t::signal, tpCallback);
  ASSERT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);

  auto GE =
      xptiMakeEvent("foo", &p, 0, (xpti::trace_activity_type_t)1, &Instance);
  EXPECT_NE(GE, nullptr);

  if (xptiCheckTraceEnabled(StreamID1)) {
    NOTIFY(StreamID1, (uint16_t)xpti::trace_point_type_t::graph_create, GE);
    NOTIFY(StreamID1, (uint16_t)xpti::trace_point_type_t::node_create, GE);
    NOTIFY(StreamID1, (uint16_t)xpti::trace_point_type_t::edge_create, GE);
    NOTIFY(StreamID1, (uint16_t)xpti::trace_point_type_t::region_begin, GE);
    NOTIFY(StreamID1, (uint16_t)xpti::trace_point_type_t::region_end, GE);
    NOTIFY(StreamID1, (uint16_t)xpti::trace_point_type_t::task_begin, GE);
    NOTIFY(StreamID1, (uint16_t)xpti::trace_point_type_t::task_end, GE);
    NOTIFY(StreamID1, (uint16_t)xpti::trace_point_type_t::barrier_begin, GE);
    NOTIFY(StreamID1, (uint16_t)xpti::trace_point_type_t::barrier_end, GE);
  }

  if (xptiCheckTraceEnabled(StreamID2)) {
    NOTIFY(StreamID2, (uint16_t)xpti::trace_point_type_t::lock_begin, GE);
    NOTIFY(StreamID2, (uint16_t)xpti::trace_point_type_t::lock_end, GE);
    NOTIFY(StreamID2, (uint16_t)xpti::trace_point_type_t::transfer_begin, GE);
    NOTIFY(StreamID2, (uint16_t)xpti::trace_point_type_t::transfer_end, GE);
    NOTIFY(StreamID2, (uint16_t)xpti::trace_point_type_t::thread_begin, GE);
    NOTIFY(StreamID2, (uint16_t)xpti::trace_point_type_t::thread_end, GE);
    NOTIFY(StreamID2, (uint16_t)xpti::trace_point_type_t::wait_begin, GE);
    NOTIFY(StreamID2, (uint16_t)xpti::trace_point_type_t::wait_end, GE);
    NOTIFY(StreamID2, (uint16_t)xpti::trace_point_type_t::signal, GE);
  }

  if (xptiCheckTraceEnabled(StreamID1,
                            (uint16_t)xpti::trace_point_type_t::graph_create)) {
    NOTIFY(StreamID1, (uint16_t)xpti::trace_point_type_t::graph_create, GE);
    auto Check = xptiCheckTraceEnabled(
        StreamID2, (uint16_t)xpti::trace_point_type_t::graph_create);
    EXPECT_NE(Check, true);
  }
  if (xptiCheckTraceEnabled(StreamID1,
                            (uint16_t)xpti::trace_point_type_t::node_create)) {
    NOTIFY(StreamID1, (uint16_t)xpti::trace_point_type_t::node_create, GE);
    auto Check = xptiCheckTraceEnabled(
        StreamID2, (uint16_t)xpti::trace_point_type_t::node_create);
    EXPECT_NE(Check, true);
  }
  if (xptiCheckTraceEnabled(StreamID1,
                            (uint16_t)xpti::trace_point_type_t::edge_create)) {
    NOTIFY(StreamID1, (uint16_t)xpti::trace_point_type_t::edge_create, GE);
    auto Check = xptiCheckTraceEnabled(
        StreamID2, (uint16_t)xpti::trace_point_type_t::edge_create);
    EXPECT_NE(Check, true);
  }
  if (xptiCheckTraceEnabled(StreamID1,
                            (uint16_t)xpti::trace_point_type_t::region_begin)) {
    NOTIFY(StreamID1, (uint16_t)xpti::trace_point_type_t::region_begin, GE);
    auto Check = xptiCheckTraceEnabled(
        StreamID2, (uint16_t)xpti::trace_point_type_t::region_begin);
    EXPECT_NE(Check, true);
  }
  if (xptiCheckTraceEnabled(StreamID1,
                            (uint16_t)xpti::trace_point_type_t::region_end)) {
    NOTIFY(StreamID1, (uint16_t)xpti::trace_point_type_t::region_end, GE);
    auto Check = xptiCheckTraceEnabled(
        StreamID2, (uint16_t)xpti::trace_point_type_t::region_end);
    EXPECT_NE(Check, true);
  }
  if (xptiCheckTraceEnabled(StreamID1,
                            (uint16_t)xpti::trace_point_type_t::task_begin)) {
    NOTIFY(StreamID1, (uint16_t)xpti::trace_point_type_t::task_begin, GE);
    auto Check = xptiCheckTraceEnabled(
        StreamID2, (uint16_t)xpti::trace_point_type_t::task_begin);
    EXPECT_NE(Check, true);
  }
  if (xptiCheckTraceEnabled(StreamID1,
                            (uint16_t)xpti::trace_point_type_t::task_end)) {
    NOTIFY(StreamID1, (uint16_t)xpti::trace_point_type_t::task_end, GE);
    auto Check = xptiCheckTraceEnabled(
        StreamID2, (uint16_t)xpti::trace_point_type_t::task_end);
    EXPECT_NE(Check, true);
  }
  if (xptiCheckTraceEnabled(
          StreamID1, (uint16_t)xpti::trace_point_type_t::barrier_begin)) {
    NOTIFY(StreamID1, (uint16_t)xpti::trace_point_type_t::barrier_begin, GE);
    auto Check = xptiCheckTraceEnabled(
        StreamID2, (uint16_t)xpti::trace_point_type_t::barrier_begin);
    EXPECT_NE(Check, true);
  }
  if (xptiCheckTraceEnabled(StreamID1,
                            (uint16_t)xpti::trace_point_type_t::barrier_end)) {
    NOTIFY(StreamID1, (uint16_t)xpti::trace_point_type_t::barrier_end, GE);
    auto Check = xptiCheckTraceEnabled(
        StreamID2, (uint16_t)xpti::trace_point_type_t::barrier_end);
    EXPECT_NE(Check, true);
  }

  if (xptiCheckTraceEnabled(StreamID2,
                            (uint16_t)xpti::trace_point_type_t::lock_begin)) {
    NOTIFY(StreamID2, (uint16_t)xpti::trace_point_type_t::lock_begin, GE);
    auto Check = xptiCheckTraceEnabled(
        StreamID1, (uint16_t)xpti::trace_point_type_t::lock_begin);
    EXPECT_NE(Check, true);
  }
  if (xptiCheckTraceEnabled(StreamID2,
                            (uint16_t)xpti::trace_point_type_t::lock_end)) {
    NOTIFY(StreamID2, (uint16_t)xpti::trace_point_type_t::lock_end, GE);
    auto Check = xptiCheckTraceEnabled(
        StreamID1, (uint16_t)xpti::trace_point_type_t::lock_end);
    EXPECT_NE(Check, true);
  }
  if (xptiCheckTraceEnabled(
          StreamID2, (uint16_t)xpti::trace_point_type_t::transfer_begin)) {
    NOTIFY(StreamID2, (uint16_t)xpti::trace_point_type_t::transfer_begin, GE);
    auto Check = xptiCheckTraceEnabled(
        StreamID1, (uint16_t)xpti::trace_point_type_t::transfer_begin);
    EXPECT_NE(Check, true);
  }
  if (xptiCheckTraceEnabled(StreamID2,
                            (uint16_t)xpti::trace_point_type_t::transfer_end)) {
    NOTIFY(StreamID2, (uint16_t)xpti::trace_point_type_t::transfer_end, GE);
    auto Check = xptiCheckTraceEnabled(
        StreamID1, (uint16_t)xpti::trace_point_type_t::transfer_end);
    EXPECT_NE(Check, true);
  }
  if (xptiCheckTraceEnabled(StreamID2,
                            (uint16_t)xpti::trace_point_type_t::thread_begin)) {
    NOTIFY(StreamID2, (uint16_t)xpti::trace_point_type_t::thread_begin, GE);
    auto Check = xptiCheckTraceEnabled(
        StreamID1, (uint16_t)xpti::trace_point_type_t::thread_begin);
    EXPECT_NE(Check, true);
  }
  if (xptiCheckTraceEnabled(StreamID2,
                            (uint16_t)xpti::trace_point_type_t::thread_end)) {
    NOTIFY(StreamID2, (uint16_t)xpti::trace_point_type_t::thread_end, GE);
    auto Check = xptiCheckTraceEnabled(
        StreamID1, (uint16_t)xpti::trace_point_type_t::thread_end);
    EXPECT_NE(Check, true);
  }
  if (xptiCheckTraceEnabled(StreamID2,
                            (uint16_t)xpti::trace_point_type_t::wait_begin)) {
    NOTIFY(StreamID2, (uint16_t)xpti::trace_point_type_t::wait_begin, GE);
    auto Check = xptiCheckTraceEnabled(
        StreamID1, (uint16_t)xpti::trace_point_type_t::wait_begin);
    EXPECT_NE(Check, true);
  }
  if (xptiCheckTraceEnabled(StreamID2,
                            (uint16_t)xpti::trace_point_type_t::wait_end)) {
    NOTIFY(StreamID2, (uint16_t)xpti::trace_point_type_t::wait_end, GE);
    auto Check = xptiCheckTraceEnabled(
        StreamID1, (uint16_t)xpti::trace_point_type_t::wait_end);
    EXPECT_NE(Check, true);
  }
  if (xptiCheckTraceEnabled(StreamID2,
                            (uint16_t)xpti::trace_point_type_t::signal)) {
    NOTIFY(StreamID2, (uint16_t)xpti::trace_point_type_t::signal, GE);
    auto Check = xptiCheckTraceEnabled(
        StreamID1, (uint16_t)xpti::trace_point_type_t::signal);
    EXPECT_NE(Check, true);
  }
}

TEST_F(xptiCorrectnessTest, xptiInitializeForUserDefinedTracePointTypes) {
  // We will test functionality of a subscriber
  // without actually creating a plugin
  uint8_t StreamID = xptiRegisterStream("test_foo");
  enum {
    extn1_begin = XPTI_TRACE_POINT_BEGIN(0),
    extn1_end = XPTI_TRACE_POINT_END(0),
    extn2_begin = XPTI_TRACE_POINT_BEGIN(1),
    extn2_end = XPTI_TRACE_POINT_END(1)
  };

  auto TTType = xptiRegisterUserDefinedTracePoint("test_foo_tool", extn1_begin);
  auto Result = xptiRegisterCallback(StreamID, TTType, tpCallback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  TTType = xptiRegisterUserDefinedTracePoint("test_foo_tool", extn1_end);
  Result = xptiRegisterCallback(StreamID, TTType, tpCallback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  TTType = xptiRegisterUserDefinedTracePoint("test_foo_tool", extn2_begin);
  Result = xptiRegisterCallback(StreamID, TTType, tpCallback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  TTType = xptiRegisterUserDefinedTracePoint("test_foo_tool", extn2_end);
  Result = xptiRegisterCallback(StreamID, TTType, tpCallback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
}

TEST_F(xptiCorrectnessTest,
       xptiNotifySubscribersForUserDefinedTracePointTypes) {
  uint64_t Instance;
  xpti::payload_t p("foo", "foo.cpp", 1, 0, (void *)13);
  xptiForceSetTraceEnabled(true);

  uint8_t StreamID = xptiRegisterStream("test_foo");
  enum {
    extn1_begin = XPTI_TRACE_POINT_BEGIN(0),
    extn1_end = XPTI_TRACE_POINT_END(0),
    extn2_begin = XPTI_TRACE_POINT_BEGIN(1),
    extn2_end = XPTI_TRACE_POINT_END(1)
  };

  auto TTType1 =
      xptiRegisterUserDefinedTracePoint("test_foo_tool", extn1_begin);
  auto Result = xptiRegisterCallback(StreamID, TTType1, tpCallback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  auto TTType2 = xptiRegisterUserDefinedTracePoint("test_foo_tool", extn1_end);
  Result = xptiRegisterCallback(StreamID, TTType2, tpCallback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  auto TTType3 =
      xptiRegisterUserDefinedTracePoint("test_foo_tool", extn2_begin);
  Result = xptiRegisterCallback(StreamID, TTType3, tpCallback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  auto TTType4 = xptiRegisterUserDefinedTracePoint("test_foo_tool", extn2_end);
  Result = xptiRegisterCallback(StreamID, TTType4, tpCallback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);

  auto GE =
      xptiMakeEvent("foo", &p, 0, (xpti::trace_activity_type_t)1, &Instance);
  EXPECT_NE(GE, nullptr);

  NOTIFY(StreamID, TTType1, GE);
  NOTIFY(StreamID, TTType2, GE);
  NOTIFY(StreamID, TTType3, GE);
  NOTIFY(StreamID, TTType4, GE);

  auto ToolID1 = XPTI_TOOL_ID(TTType1);
  auto ToolID2 = XPTI_TOOL_ID(TTType2);
  auto ToolID3 = XPTI_TOOL_ID(TTType3);
  auto ToolID4 = XPTI_TOOL_ID(TTType4);
  EXPECT_EQ(ToolID1, ToolID2);
  EXPECT_EQ(ToolID2, ToolID3);
  EXPECT_EQ(ToolID3, ToolID4);
  EXPECT_EQ(ToolID4, ToolID1);

  auto TpID1 = XPTI_EXTRACT_USER_DEFINED_ID(TTType1);
  auto TpID2 = XPTI_EXTRACT_USER_DEFINED_ID(TTType2);
  auto TpID3 = XPTI_EXTRACT_USER_DEFINED_ID(TTType3);
  auto TpID4 = XPTI_EXTRACT_USER_DEFINED_ID(TTType4);
  EXPECT_NE(TpID1, TpID2);
  EXPECT_NE(TpID2, TpID3);
  EXPECT_NE(TpID3, TpID4);
  EXPECT_NE(TpID4, TpID1);
}

TEST_F(xptiCorrectnessTest, xptiGetUniqueId) {
  auto Result = xptiGetUniqueId();
  EXPECT_NE(Result, 0u);
  auto Result1 = xptiGetUniqueId();
  EXPECT_NE(Result, Result1);
}

TEST_F(xptiCorrectnessTest, xptiUniversalIDTest) {
  xpti::uid_t Id0, Id1;
  uint64_t instance;
  /// Simulates the specialization of a Kernel as used by MKL where
  /// the same kernel may be compiled multiple times
  xpti::payload_t Payload("foo", "foo.cpp", 1, 0, (void *)&Id0);
  auto Result = xptiMakeEvent("foo", &Payload, 0,
                              (xpti::trace_activity_type_t)1, &instance);

  Id0.p1 = XPTI_PACK32_RET64(Payload.source_file_sid(), Payload.name_sid());
  Id0.p2 = XPTI_PACK32_RET64(Payload.column_no, Payload.line_no);
  Id0.p3 = (uint64_t)Payload.code_ptr_va;
  xpti::payload_t P("foo", "foo.cpp", 1, 0, (void *)&Id1);
  auto Result1 =
      xptiMakeEvent("foo", &P, 0, (xpti::trace_activity_type_t)1, &instance);
  Id1.p1 = XPTI_PACK32_RET64(P.source_file_sid(), P.name_sid());
  Id1.p2 = XPTI_PACK32_RET64(P.column_no, P.line_no);
  Id1.p3 = (uint64_t)P.code_ptr_va;
  EXPECT_NE(Result, Result1);
  EXPECT_NE(Id0.hash(), Id1.hash());
}

TEST_F(xptiCorrectnessTest, xptiUniversalIDRandomTest) {
  using namespace std;
  set<uint64_t> HashSet;
  random_device QRd;
  mt19937_64 Gen(QRd());
  uniform_int_distribution<uint32_t> MStringID, MLineNo, MColNo, MAddr;

  MStringID = uniform_int_distribution<uint32_t>(1, 1000000);
  MLineNo = uniform_int_distribution<uint32_t>(1, 200000);
  MColNo = uniform_int_distribution<uint32_t>(1, 2000);
  MAddr = uniform_int_distribution<uint32_t>(0x10000000, 0xffffffff);

  for (int i = 0; i < 1000000; ++i) {
    xpti::uid_t id;
    id.p1 = XPTI_PACK32_RET64(MStringID(Gen), MStringID(Gen));
    id.p2 = XPTI_PACK32_RET64(MColNo(Gen), MLineNo(Gen));
    id.p3 = MAddr(Gen);
    uint64_t hash = id.hash();
    EXPECT_EQ(HashSet.count(hash), 0u);
    HashSet.insert(hash);
  }

  xpti::uid_t id1, id2;
  uint32_t sid = MStringID(Gen), ln = MLineNo(Gen), kid = MStringID(Gen),
           col = MColNo(Gen);
  uint64_t code_ptr_va = MAddr(Gen);
  id1.p1 = XPTI_PACK32_RET64(sid, kid);
  id1.p2 = XPTI_PACK32_RET64(ln, col);
  id1.p3 = code_ptr_va;

  id2.p1 = XPTI_PACK32_RET64(sid, kid);
  id2.p2 = XPTI_PACK32_RET64(ln, col);
  id2.p3 = code_ptr_va;

  EXPECT_EQ(id1.hash(), id2.hash());
}

TEST_F(xptiCorrectnessTest, xptiUniversalIDMapTest) {
  using namespace std;
  map<xpti::uid_t, uint64_t> MapTest;
  random_device QRd;
  mt19937_64 Gen(QRd());
  uniform_int_distribution<uint32_t> MStringID, MLineNo, MAddr, MColNo;

  MStringID = uniform_int_distribution<uint32_t>(1, 1000000);
  MLineNo = uniform_int_distribution<uint32_t>(1, 200000);
  MColNo = uniform_int_distribution<uint32_t>(1, 2000);
  MAddr = uniform_int_distribution<uint32_t>(0x10000000, 0xffffffff);

  constexpr unsigned int Count = 100000;
  for (unsigned int i = 0; i < Count; ++i) {
    xpti::uid_t id;
    id.p1 = XPTI_PACK32_RET64(MStringID(Gen), MStringID(Gen));
    id.p2 = XPTI_PACK32_RET64(MColNo(Gen), MLineNo(Gen));
    id.p3 = MAddr(Gen);

    uint64_t hash = id.hash();
    EXPECT_EQ(MapTest.count(id), 0u);
    MapTest[id] = hash;
  }

  EXPECT_EQ(Count, MapTest.size());
  for (auto &e : MapTest) {
    EXPECT_EQ(e.first.hash(), e.second);
  }
  // Check if the IDs are in sorted order
  xpti::uid_t prev;
  for (auto &e : MapTest) {
    bool test = prev < e.first;
    EXPECT_EQ(test, true);
  }
}

TEST_F(xptiCorrectnessTest, xptiUniversalIDUnorderedMapTest) {
  using namespace std;
  unordered_map<xpti::uid_t, uint64_t> MapTest;
  random_device QRd;
  mt19937_64 Gen(QRd());
  uniform_int_distribution<uint32_t> MStringID, MLineNo, MAddr, MColNo;

  MStringID = uniform_int_distribution<uint32_t>(1, 1000000);
  MLineNo = uniform_int_distribution<uint32_t>(1, 200000);
  MColNo = uniform_int_distribution<uint32_t>(1, 2000);
  MAddr = uniform_int_distribution<uint32_t>(0x10000000, 0xffffffff);

  constexpr unsigned int Count = 100000;
  for (unsigned int i = 0; i < Count; ++i) {
    xpti::uid_t id;
    id.p1 = XPTI_PACK32_RET64(MStringID(Gen), MStringID(Gen));
    id.p2 = XPTI_PACK32_RET64(MColNo(Gen), MLineNo(Gen));
    id.p3 = (uint64_t)MAddr(Gen);

    uint64_t hash = id.hash();
    EXPECT_EQ(MapTest.count(id), 0u);
    MapTest[id] = hash;
  }

  EXPECT_EQ(Count, MapTest.size());
  for (auto &e : MapTest) {
    EXPECT_EQ(e.first.hash(), e.second);
  }
}

TEST_F(xptiCorrectnessTest, xptiUserDefinedEventTypes) {
  xpti::payload_t p("foo", "foo.cpp", 1, 0, (void *)13);
  xptiForceSetTraceEnabled(true);

  (void)xptiRegisterStream("test_foo");
  enum {
    extn_ev1 = XPTI_EVENT(0),
    extn_ev2 = XPTI_EVENT(1),
    extn_ev3 = XPTI_EVENT(2),
    extn_ev4 = XPTI_EVENT(3)
  };

  auto EventType1 = xptiRegisterUserDefinedEventType("test_foo_tool", extn_ev1);
  auto EventType2 = xptiRegisterUserDefinedEventType("test_foo_tool", extn_ev2);
  auto EventType3 = xptiRegisterUserDefinedEventType("test_foo_tool", extn_ev3);
  auto EventType4 = xptiRegisterUserDefinedEventType("test_foo_tool", extn_ev4);
  EXPECT_NE(EventType1, EventType2);
  EXPECT_NE(EventType2, EventType3);
  EXPECT_NE(EventType3, EventType4);
  EXPECT_NE(EventType4, EventType1);

  auto ToolID1 = XPTI_TOOL_ID(EventType1);
  auto ToolID2 = XPTI_TOOL_ID(EventType2);
  auto ToolID3 = XPTI_TOOL_ID(EventType3);
  auto ToolID4 = XPTI_TOOL_ID(EventType4);
  EXPECT_EQ(ToolID1, ToolID2);
  EXPECT_EQ(ToolID2, ToolID3);
  EXPECT_EQ(ToolID3, ToolID4);
  EXPECT_EQ(ToolID4, ToolID1);

  auto TpID1 = XPTI_EXTRACT_USER_DEFINED_ID(EventType1);
  auto TpID2 = XPTI_EXTRACT_USER_DEFINED_ID(EventType2);
  auto TpID3 = XPTI_EXTRACT_USER_DEFINED_ID(EventType3);
  auto TpID4 = XPTI_EXTRACT_USER_DEFINED_ID(EventType4);
  EXPECT_NE(TpID1, TpID2);
  EXPECT_NE(TpID2, TpID3);
  EXPECT_NE(TpID3, TpID4);
  EXPECT_NE(TpID4, TpID1);
}
