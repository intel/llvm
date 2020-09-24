//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
#include "xpti_trace_framework.h"
#include "xpti_trace_framework.hpp"

#include <gtest/gtest.h>
#include <iostream>
#include <set>

XPTI_CALLBACK_API void tpCallback(uint16_t trace_type,
                                  xpti::trace_event_data_t *parent,
                                  xpti::trace_event_data_t *event,
                                  uint64_t instance, const void *user_data) {

  if (user_data)
    (*(int *)user_data) = trace_type;
}

#define NOTIFY(stream, tt, event, retval)                                      \
  {                                                                            \
    xpti::result_t Result = xptiNotifySubscribers(stream, tt, nullptr, event,  \
                                                  0, (void *)(&retval));       \
    EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);                    \
    EXPECT_EQ(retval, tt);                                                     \
  }

TEST(xptiCorrectnessTest, xptiMakeEvent) {
  uint64_t Instance = 0;
  xpti::payload_t p("foo", "foo.cpp", 1, 0, (void *)13);
  auto Result =
      xptiMakeEvent("foo", &p, 0, (xpti::trace_activity_type_t)1, &Instance);
  EXPECT_NE(Result, nullptr);
  p = xpti::payload_t("foo", "foo.cpp", 1, 0, (void *)13);
  auto NewResult =
      xptiMakeEvent("foo", &p, 0, (xpti::trace_activity_type_t)1, &Instance);
  EXPECT_EQ(Result, NewResult);
  EXPECT_EQ(Result->unique_id, NewResult->unique_id);
  EXPECT_EQ(Result->reserved.payload, NewResult->reserved.payload);
  EXPECT_STREQ(Result->reserved.payload->name, "foo");
  EXPECT_STREQ(Result->reserved.payload->source_file, "foo.cpp");
  EXPECT_EQ(Result->reserved.payload->line_no, 1);
}

TEST(xptiCorrectnessTest, xptiRegisterString) {
  char *TStr = nullptr;
  auto ID = xptiRegisterString("foo", &TStr);
  EXPECT_NE(ID, xpti::invalid_id);
  EXPECT_NE(TStr, nullptr);
  EXPECT_STREQ("foo", TStr);

  const char *LUTStr = xptiLookupString(ID);
  EXPECT_EQ(TStr, LUTStr);
  EXPECT_STREQ(LUTStr, TStr);
}

TEST(xptiCorrectnessTest, xptiInitializeForDefaultTracePointTypes) {
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

TEST(xptiCorrectnessTest, xptiNotifySubscribersForDefaultTracePointTypes) {
  uint64_t Instance;
  xpti::payload_t p("foo", "foo.cpp", 1, 0, (void *)13);
  xptiForceSetTraceEnabled(true);

  uint8_t StreamID = xptiRegisterStream("test_foo");
  auto Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::graph_create, tpCallback);
  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::node_create, tpCallback);
  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::edge_create, tpCallback);
  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::region_begin, tpCallback);
  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::region_end, tpCallback);
  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::task_begin, tpCallback);
  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::task_end, tpCallback);
  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::barrier_begin, tpCallback);
  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::barrier_end, tpCallback);
  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::lock_begin, tpCallback);
  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::lock_end, tpCallback);
  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::transfer_begin, tpCallback);
  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::transfer_end, tpCallback);
  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::thread_begin, tpCallback);
  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::thread_end, tpCallback);
  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::wait_begin, tpCallback);
  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::wait_end, tpCallback);
  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::signal, tpCallback);

  auto GE =
      xptiMakeEvent("foo", &p, 0, (xpti::trace_activity_type_t)1, &Instance);
  EXPECT_NE(GE, nullptr);

  int FooReturn = 0;
  NOTIFY(StreamID, (uint16_t)xpti::trace_point_type_t::graph_create, GE,
         FooReturn);
  NOTIFY(StreamID, (uint16_t)xpti::trace_point_type_t::node_create, GE,
         FooReturn);
  NOTIFY(StreamID, (uint16_t)xpti::trace_point_type_t::edge_create, GE,
         FooReturn);
  NOTIFY(StreamID, (uint16_t)xpti::trace_point_type_t::region_begin, GE,
         FooReturn);
  NOTIFY(StreamID, (uint16_t)xpti::trace_point_type_t::region_end, GE,
         FooReturn);
  NOTIFY(StreamID, (uint16_t)xpti::trace_point_type_t::task_begin, GE,
         FooReturn);
  NOTIFY(StreamID, (uint16_t)xpti::trace_point_type_t::task_end, GE, FooReturn);
  NOTIFY(StreamID, (uint16_t)xpti::trace_point_type_t::barrier_begin, GE,
         FooReturn);
  NOTIFY(StreamID, (uint16_t)xpti::trace_point_type_t::barrier_end, GE,
         FooReturn);
  NOTIFY(StreamID, (uint16_t)xpti::trace_point_type_t::lock_begin, GE,
         FooReturn);
  NOTIFY(StreamID, (uint16_t)xpti::trace_point_type_t::lock_end, GE, FooReturn);
  NOTIFY(StreamID, (uint16_t)xpti::trace_point_type_t::transfer_begin, GE,
         FooReturn);
  NOTIFY(StreamID, (uint16_t)xpti::trace_point_type_t::transfer_end, GE,
         FooReturn);
  NOTIFY(StreamID, (uint16_t)xpti::trace_point_type_t::thread_begin, GE,
         FooReturn);
  NOTIFY(StreamID, (uint16_t)xpti::trace_point_type_t::thread_end, GE,
         FooReturn);
  NOTIFY(StreamID, (uint16_t)xpti::trace_point_type_t::wait_begin, GE,
         FooReturn);
  NOTIFY(StreamID, (uint16_t)xpti::trace_point_type_t::wait_end, GE, FooReturn);
  NOTIFY(StreamID, (uint16_t)xpti::trace_point_type_t::signal, GE, FooReturn);
}

TEST(xptiCorrectnessTest, xptiInitializeForUserDefinedTracePointTypes) {
  // We will test functionality of a subscriber
  // without actually creating a plugin
  uint8_t StreamID = xptiRegisterStream("test_foo");
  typedef enum {
    extn1_begin = XPTI_TRACE_POINT_BEGIN(0),
    extn1_end = XPTI_TRACE_POINT_END(0),
    extn2_begin = XPTI_TRACE_POINT_BEGIN(1),
    extn2_end = XPTI_TRACE_POINT_END(1)
  } tp_extension_t;

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

TEST(xptiCorrectnessTest, xptiNotifySubscribersForUserDefinedTracePointTypes) {
  uint64_t Instance;
  xpti::payload_t p("foo", "foo.cpp", 1, 0, (void *)13);
  xptiForceSetTraceEnabled(true);

  uint8_t StreamID = xptiRegisterStream("test_foo");
  typedef enum {
    extn1_begin = XPTI_TRACE_POINT_BEGIN(0),
    extn1_end = XPTI_TRACE_POINT_END(0),
    extn2_begin = XPTI_TRACE_POINT_BEGIN(1),
    extn2_end = XPTI_TRACE_POINT_END(1)
  } tp_extension_t;

  auto TTType1 =
      xptiRegisterUserDefinedTracePoint("test_foo_tool", extn1_begin);
  auto Result = xptiRegisterCallback(StreamID, TTType1, tpCallback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_DUPLICATE);
  auto TTType2 = xptiRegisterUserDefinedTracePoint("test_foo_tool", extn1_end);
  Result = xptiRegisterCallback(StreamID, TTType2, tpCallback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_DUPLICATE);
  auto TTType3 =
      xptiRegisterUserDefinedTracePoint("test_foo_tool", extn2_begin);
  Result = xptiRegisterCallback(StreamID, TTType3, tpCallback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_DUPLICATE);
  auto TTType4 = xptiRegisterUserDefinedTracePoint("test_foo_tool", extn2_end);
  Result = xptiRegisterCallback(StreamID, TTType4, tpCallback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_DUPLICATE);

  auto GE =
      xptiMakeEvent("foo", &p, 0, (xpti::trace_activity_type_t)1, &Instance);
  EXPECT_NE(GE, nullptr);

  int FooReturn = 0;
  NOTIFY(StreamID, TTType1, GE, FooReturn);
  NOTIFY(StreamID, TTType2, GE, FooReturn);
  NOTIFY(StreamID, TTType3, GE, FooReturn);
  NOTIFY(StreamID, TTType4, GE, FooReturn);

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

TEST(xptiCorrectnessTest, xptiGetUniqueId) {
  auto Result = xptiGetUniqueId();
  EXPECT_NE(Result, 0);
  auto Result1 = xptiGetUniqueId();
  EXPECT_NE(Result, Result1);
}

TEST(xptiCorrectnessTest, xptiUserDefinedEventTypes) {
  uint64_t Instance;
  xpti::payload_t p("foo", "foo.cpp", 1, 0, (void *)13);
  xptiForceSetTraceEnabled(true);

  uint8_t StreamID = xptiRegisterStream("test_foo");
  typedef enum {
    extn_ev1 = XPTI_EVENT(0),
    extn_ev2 = XPTI_EVENT(1),
    extn_ev3 = XPTI_EVENT(2),
    extn_ev4 = XPTI_EVENT(3)
  } event_extension_t;

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
