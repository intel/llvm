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
  xpti::hash_t Hash;

  ASSERT_NE(NewResult, nullptr);
  EXPECT_EQ(Hash.combine_short(NewResult->uid),
            Hash.combine_short(Result->uid));
  EXPECT_EQ(Result->uid, NewResult->uid);
  EXPECT_EQ(Result->reserved.payload, NewResult->reserved.payload);
  EXPECT_STREQ(Result->reserved.payload->name, "foo");
  EXPECT_STREQ(Result->reserved.payload->source_file, "foo.cpp");
  EXPECT_EQ(Result->reserved.payload->line_no, 1u);
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

TEST_F(xptiCorrectnessTest, xptiTPayloadTest) {
  xpti::payload_t p("foo", "foo.cpp", 10, 0, (void *)(0xdeadbeefull));
  xpti::hash_t Hash;

  auto UID = xptiRegisterPayload(&p);
  EXPECT_EQ(UID, p.uid);
  EXPECT_EQ(UID.isValid(), true);
  EXPECT_EQ(p.isValid(), true);

  auto Payload = const_cast<xpti::payload_t *>(xptiQueryPayloadByUID(UID));
  EXPECT_NE(Payload, nullptr);
  EXPECT_EQ(Payload->uid, UID);
  EXPECT_EQ(Payload->uid.isValid(), true);
  EXPECT_GT(Payload->uid.instance, 0);
  // Registering it again should change the instance number
  auto NewUID = xptiRegisterPayload(&p);
  EXPECT_GT(NewUID.instance, UID.instance);
  EXPECT_EQ(Hash.combine_short(NewUID), Hash.combine_short(UID));
}

TEST_F(xptiCorrectnessTest, xptiTracePointScopeDataTest) {
  xpti::payload_t p("foo", "foo.cpp", 10, 0, (void *)(0xdeadbeefull));
  xpti::framework::tracepoint_scope_t t(p, false);
  auto ScopeData = xptiGetTracepointScopeData();
  EXPECT_EQ(ScopeData.isValid(), true);
  EXPECT_EQ(ScopeData.uid.isValid(), true);
  EXPECT_EQ(ScopeData.uid, t.uid());
  EXPECT_EQ(ScopeData.payload->isValid(), true);
  EXPECT_NE(ScopeData.payload, nullptr);
  EXPECT_EQ(ScopeData.payload, t.payload());
  EXPECT_NE(ScopeData.event, nullptr);
  EXPECT_EQ(ScopeData.event, t.traceEvent());

  EXPECT_EQ(ScopeData.event->uid, t.uid());
  EXPECT_EQ(ScopeData.event->uid, ScopeData.uid);
  EXPECT_EQ(ScopeData.payload->uid, ScopeData.uid);
  EXPECT_EQ(ScopeData.payload->uid, ScopeData.event->uid);
  EXPECT_EQ(ScopeData.payload->uid, t.uid());

  EXPECT_EQ(ScopeData.payload, ScopeData.event->reserved.payload);
  EXPECT_EQ(ScopeData.payload->uid, ScopeData.event->reserved.payload->uid);
  EXPECT_EQ(ScopeData.event->uid, ScopeData.uid);
}

void nestedScopeTest(xpti::payload_t *p, std::vector<uint64_t> &uids) {
  xpti::framework::tracepoint_scope_t t(*p, false);
  xpti::hash_t Hash;

  uint64_t hash = Hash.combine_short(t.uid());
  uids.push_back(hash);

  if (uids.size() < 5) {
    xpti::payload_t pp;
    nestedScopeTest(&pp, uids);
  }
}

TEST_F(xptiCorrectnessTest, xptiTracePointScopeTest) {
  std::vector<uint64_t> uids;
  xpti::payload_t p("foo", "foo.cpp", 10, 0);

  auto uid = xptiRegisterPayload(&p);

  uint64_t id = xpti::invalid_uid;
  nestedScopeTest(&p, uids);
  for (auto &e : uids) {
    EXPECT_NE(e, xpti::invalid_uid);
    if (id != xpti::invalid_uid) {
      EXPECT_EQ(e, id);
      std::cerr << "E: " << e << " ID: " << id << std::endl;
      id = e;
    }
  }
  // UID should not be able to query the trace point data here as it would have
  // been released when it went of out scope
  auto Event = xptiFindEvent(uid);
  EXPECT_EQ(Event, nullptr);
  // UID should be able to query the payload data here as it is still valid
  auto Payload = xptiQueryPayloadByUID(uid);
  EXPECT_NE(Payload, nullptr);

  uids.clear();
  xpti::payload_t p1("bar", "foo.cpp", 15, 0);
  {
    xpti::uid_t uid;
    {
      xpti::framework::tracepoint_scope_t t(p1, false);
      uid = t.uid();
      EXPECT_NE(t.traceEvent(), nullptr);
      auto ScopeData = xptiGetTracepointScopeData();
      EXPECT_EQ(ScopeData.isValid(), true);
      id = xpti::invalid_uid;
      nestedScopeTest(&p1, uids);
      for (auto &e : uids) {
        EXPECT_NE(e, xpti::invalid_uid);
        if (id != xpti::invalid_uid) {
          EXPECT_EQ(e, id);
          id = e;
        }
      }
      EXPECT_NE(t.traceEvent(), nullptr);

      // UID should be able to query both payload and event in this case
      auto Event = xptiFindEvent(t.uid());
      EXPECT_NE(Event, nullptr);
      auto Payload = xptiQueryPayloadByUID(t.uid());
      EXPECT_NE(Payload, nullptr);
    }
    // The Event for the uid would have goine out of scope and deleted
    auto Event = xptiFindEvent(uid);
    EXPECT_EQ(Event, nullptr);
  }
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

  uint8_t StreamID1 = xptiRegisterStream("bar");
  uint8_t StreamID2 = xptiRegisterStream("foo_bar");
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

TEST_F(xptiCorrectnessTest, xptiQueryMetadata) {
  xpti::uid_t Id0;
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
  xpti::uid_t Id0;
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

TEST_F(xptiCorrectnessTest, xptiUniversalIDTest) {
  xpti::uid_t Id0;
  uint64_t instance;
  /// Simulates the specialization of a Kernel as used by MKL where
  /// the same kernel may be compiled multiple times
  xpti::payload_t Payload("foo", "foo.cpp", 1, 0, (void *)&Id0);
  auto Result = xptiMakeEvent("foo", &Payload, 0,
                              (xpti::trace_activity_type_t)1, &instance);
  Id0.p1 = XPTI_PACK32_RET64(Result->uid.fileId(), Result->uid.functionId());
  Id0.p2 = XPTI_PACK32_RET64(0, Result->uid.lineNo());
  auto Result1 = xptiMakeEvent("foo", &Payload, 0,
                               (xpti::trace_activity_type_t)1, &instance);
  xpti::hash_t Hash;
  EXPECT_EQ(Hash.combine_short(Result->uid), Hash.combine_short(Id0));
  EXPECT_NE(Result, Result1);
  EXPECT_EQ(Hash.combine_short(Result->uid), Hash.combine_short(Result1->uid));
}

TEST_F(xptiCorrectnessTest, xptiUniversalIDRandomTest) {
  using namespace std;
  set<uint64_t> HashSet;
  random_device QRd;
  mt19937_64 Gen(QRd());
  uniform_int_distribution<uint32_t> MStringID, MLineNo, MAddr;

  MStringID = uniform_int_distribution<uint32_t>(1, 1000000);
  MLineNo = uniform_int_distribution<uint32_t>(1, 200000);
  MAddr = uniform_int_distribution<uint32_t>(0x10000000, 0xffffffff);

  for (int i = 0; i < 1000000; ++i) {
    xpti::hash_t Hash;
    xpti::uid_t id;
    id.p1 = XPTI_PACK32_RET64(MStringID(Gen), MStringID(Gen));
    id.p2 = XPTI_PACK32_RET64(0, MLineNo(Gen));

    uint64_t hash = Hash.combine_short(id);
    EXPECT_EQ(HashSet.count(hash), 0u);
    HashSet.insert(hash);
  }

  xpti::uid_t id1, id2;
  uint32_t sid = MStringID(Gen), ln = MLineNo(Gen), kid = MStringID(Gen),
           addr = MAddr(Gen);
  id1.p1 = XPTI_PACK32_RET64(sid, kid);
  id1.p2 = XPTI_PACK32_RET64(0, ln);

  id2.p1 = XPTI_PACK32_RET64(sid, kid);
  id2.p2 = XPTI_PACK32_RET64(0, ln);

  xpti::hash_t Hash;
  EXPECT_EQ(Hash.combine_short(id1), Hash.combine_short(id2));
}

TEST_F(xptiCorrectnessTest, xptiUniversalIDMapTest) {
  using namespace std;
  map<xpti::uid_t, uint64_t> MapTest;
  random_device QRd;
  mt19937_64 Gen(QRd());
  uniform_int_distribution<uint32_t> MStringID, MLineNo, MAddr;

  MStringID = uniform_int_distribution<uint32_t>(1, 1000000);
  MLineNo = uniform_int_distribution<uint32_t>(1, 200000);
  xpti::hash_t Hash;

  constexpr unsigned int Count = 100000;
  for (unsigned int i = 0; i < Count; ++i) {
    xpti::uid_t id;
    id.p1 = XPTI_PACK32_RET64(MStringID(Gen), MStringID(Gen));
    id.p2 = XPTI_PACK32_RET64(0, MLineNo(Gen));

    uint64_t hash = Hash.combine_short(id);
    EXPECT_EQ(MapTest.count(id), 0u);
    MapTest[id] = hash;
  }

  EXPECT_EQ(Count, MapTest.size());
  for (auto &e : MapTest) {
    EXPECT_EQ(Hash.combine_short(e.first), e.second);
  }
  xpti::uid_t prev;
  for (auto &e : MapTest) {
    bool test = prev < e.first;
    EXPECT_EQ(test, true);
    prev = e.first;
  }
}

TEST_F(xptiCorrectnessTest, xptiUniversalIDUnorderedMapTest) {
  using namespace std;
  unordered_map<xpti::uid_t, uint64_t> MapTest;
  random_device QRd;
  mt19937_64 Gen(QRd());
  uniform_int_distribution<uint32_t> MStringID, MLineNo, MAddr;

  MStringID = uniform_int_distribution<uint32_t>(1, 1000000);
  MLineNo = uniform_int_distribution<uint32_t>(1, 200000);
  MAddr = uniform_int_distribution<uint32_t>(0x10000000, 0xffffffff);
  xpti::hash_t Hash;

  constexpr unsigned int Count = 100000;
  for (unsigned int i = 0; i < Count; ++i) {
    xpti::uid_t id;
    id.p1 = XPTI_PACK32_RET64(MStringID(Gen), MLineNo(Gen));
    id.p2 = XPTI_PACK32_RET64(0, MStringID(Gen));

    uint64_t hash = Hash.combine_short(id);
    EXPECT_EQ(MapTest.count(id), 0u);
    MapTest[id] = hash;
  }

  EXPECT_EQ(Count, MapTest.size());
  for (auto &e : MapTest) {
    EXPECT_EQ(Hash.combine_short(e.first), e.second);
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
