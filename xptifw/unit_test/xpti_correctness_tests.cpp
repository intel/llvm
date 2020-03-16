//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
#include "xpti_trace_framework.h"
#include "xpti_trace_framework.hpp"

#include <iostream>
#include <set>

#include <gtest/gtest.h>

XPTI_CALLBACK_API void tp_callback(uint16_t trace_type,
                                   xpti::trace_event_data_t *parent,
                                   xpti::trace_event_data_t *event,
                                   uint64_t instance, const void *user_data) {

  if (user_data)
    (*(int *)user_data) = trace_type;
}

#define NOTIFY(stream, tt, event, retval)                                      \
  {                                                                            \
    xpti::result_t result = xptiNotifySubscribers(stream, tt, nullptr, event,  \
                                                  0, (void *)(&retval));       \
    EXPECT_EQ(result, xpti::result_t::XPTI_RESULT_SUCCESS);                    \
    EXPECT_EQ(retval, tt);                                                     \
  }

TEST(xptiCorrectnessTest, xptiMakeEvent) {
  uint64_t instance = 0;
  xpti::payload_t p("foo", "foo.cpp", 1, 0, (void *)13);
  auto result =
      xptiMakeEvent("foo", &p, 0, (xpti::trace_activity_type_t)1, &instance);
  EXPECT_NE(result, nullptr);
  p = xpti::payload_t("foo", "foo.cpp", 1, 0, (void *)13);
  auto new_result =
      xptiMakeEvent("foo", &p, 0, (xpti::trace_activity_type_t)1, &instance);
  EXPECT_EQ(result, new_result);
  EXPECT_EQ(result->unique_id, new_result->unique_id);
  EXPECT_EQ(result->reserved.payload, new_result->reserved.payload);
  EXPECT_STREQ(result->reserved.payload->name, "foo");
  EXPECT_STREQ(result->reserved.payload->source_file, "foo.cpp");
  EXPECT_EQ(result->reserved.payload->line_no, 1);
}

TEST(xptiCorrectnessTest, xptiRegisterString) {
  char *tstr = nullptr;
  auto id = xptiRegisterString("foo", &tstr);
  EXPECT_NE(id, xpti::invalid_id);
  EXPECT_NE(tstr, nullptr);
  EXPECT_STREQ("foo", tstr);

  const char *lutstr = xptiLookupString(id);
  EXPECT_EQ(tstr, lutstr);
  EXPECT_STREQ(lutstr, tstr);
}

TEST(xptiCorrectnessTest, xptiInitializeForDefaultTracePointTypes) {
  // We will test functionality of a subscriber
  // without actually creating a plugin
  uint8_t stream_id = xptiRegisterStream("test_foo");
  auto result = xptiRegisterCallback(
      stream_id, (uint16_t)xpti::trace_point_type_t::graph_create, tp_callback);
  EXPECT_EQ(result, xpti::result_t::XPTI_RESULT_SUCCESS);
  result = xptiRegisterCallback(
      stream_id, (uint16_t)xpti::trace_point_type_t::node_create, tp_callback);
  EXPECT_EQ(result, xpti::result_t::XPTI_RESULT_SUCCESS);
  result = xptiRegisterCallback(
      stream_id, (uint16_t)xpti::trace_point_type_t::edge_create, tp_callback);
  EXPECT_EQ(result, xpti::result_t::XPTI_RESULT_SUCCESS);
  result = xptiRegisterCallback(
      stream_id, (uint16_t)xpti::trace_point_type_t::region_begin, tp_callback);
  EXPECT_EQ(result, xpti::result_t::XPTI_RESULT_SUCCESS);
  result = xptiRegisterCallback(
      stream_id, (uint16_t)xpti::trace_point_type_t::region_end, tp_callback);
  EXPECT_EQ(result, xpti::result_t::XPTI_RESULT_SUCCESS);
  result = xptiRegisterCallback(
      stream_id, (uint16_t)xpti::trace_point_type_t::task_begin, tp_callback);
  EXPECT_EQ(result, xpti::result_t::XPTI_RESULT_SUCCESS);
  result = xptiRegisterCallback(
      stream_id, (uint16_t)xpti::trace_point_type_t::task_end, tp_callback);
  EXPECT_EQ(result, xpti::result_t::XPTI_RESULT_SUCCESS);
  result = xptiRegisterCallback(
      stream_id, (uint16_t)xpti::trace_point_type_t::barrier_begin,
      tp_callback);
  EXPECT_EQ(result, xpti::result_t::XPTI_RESULT_SUCCESS);
  result = xptiRegisterCallback(
      stream_id, (uint16_t)xpti::trace_point_type_t::barrier_end, tp_callback);
  EXPECT_EQ(result, xpti::result_t::XPTI_RESULT_SUCCESS);
  result = xptiRegisterCallback(
      stream_id, (uint16_t)xpti::trace_point_type_t::lock_begin, tp_callback);
  EXPECT_EQ(result, xpti::result_t::XPTI_RESULT_SUCCESS);
  result = xptiRegisterCallback(
      stream_id, (uint16_t)xpti::trace_point_type_t::lock_end, tp_callback);
  EXPECT_EQ(result, xpti::result_t::XPTI_RESULT_SUCCESS);
  result = xptiRegisterCallback(
      stream_id, (uint16_t)xpti::trace_point_type_t::transfer_begin,
      tp_callback);
  EXPECT_EQ(result, xpti::result_t::XPTI_RESULT_SUCCESS);
  result = xptiRegisterCallback(
      stream_id, (uint16_t)xpti::trace_point_type_t::transfer_end, tp_callback);
  EXPECT_EQ(result, xpti::result_t::XPTI_RESULT_SUCCESS);
  result = xptiRegisterCallback(
      stream_id, (uint16_t)xpti::trace_point_type_t::thread_begin, tp_callback);
  EXPECT_EQ(result, xpti::result_t::XPTI_RESULT_SUCCESS);
  result = xptiRegisterCallback(
      stream_id, (uint16_t)xpti::trace_point_type_t::thread_end, tp_callback);
  EXPECT_EQ(result, xpti::result_t::XPTI_RESULT_SUCCESS);
  result = xptiRegisterCallback(
      stream_id, (uint16_t)xpti::trace_point_type_t::wait_begin, tp_callback);
  EXPECT_EQ(result, xpti::result_t::XPTI_RESULT_SUCCESS);
  result = xptiRegisterCallback(
      stream_id, (uint16_t)xpti::trace_point_type_t::wait_end, tp_callback);
  EXPECT_EQ(result, xpti::result_t::XPTI_RESULT_SUCCESS);
  result = xptiRegisterCallback(
      stream_id, (uint16_t)xpti::trace_point_type_t::signal, tp_callback);
  EXPECT_EQ(result, xpti::result_t::XPTI_RESULT_SUCCESS);
}

TEST(xptiCorrectnessTest, xptiNotifySubscribersForDefaultTracePointTypes) {
  uint64_t instance;
  xpti::payload_t p("foo", "foo.cpp", 1, 0, (void *)13);
  xptiForceSetTraceEnabled(true);

  uint8_t stream_id = xptiRegisterStream("test_foo");
  auto result = xptiRegisterCallback(
      stream_id, (uint16_t)xpti::trace_point_type_t::graph_create, tp_callback);
  result = xptiRegisterCallback(
      stream_id, (uint16_t)xpti::trace_point_type_t::node_create, tp_callback);
  result = xptiRegisterCallback(
      stream_id, (uint16_t)xpti::trace_point_type_t::edge_create, tp_callback);
  result = xptiRegisterCallback(
      stream_id, (uint16_t)xpti::trace_point_type_t::region_begin, tp_callback);
  result = xptiRegisterCallback(
      stream_id, (uint16_t)xpti::trace_point_type_t::region_end, tp_callback);
  result = xptiRegisterCallback(
      stream_id, (uint16_t)xpti::trace_point_type_t::task_begin, tp_callback);
  result = xptiRegisterCallback(
      stream_id, (uint16_t)xpti::trace_point_type_t::task_end, tp_callback);
  result = xptiRegisterCallback(
      stream_id, (uint16_t)xpti::trace_point_type_t::barrier_begin,
      tp_callback);
  result = xptiRegisterCallback(
      stream_id, (uint16_t)xpti::trace_point_type_t::barrier_end, tp_callback);
  result = xptiRegisterCallback(
      stream_id, (uint16_t)xpti::trace_point_type_t::lock_begin, tp_callback);
  result = xptiRegisterCallback(
      stream_id, (uint16_t)xpti::trace_point_type_t::lock_end, tp_callback);
  result = xptiRegisterCallback(
      stream_id, (uint16_t)xpti::trace_point_type_t::transfer_begin,
      tp_callback);
  result = xptiRegisterCallback(
      stream_id, (uint16_t)xpti::trace_point_type_t::transfer_end, tp_callback);
  result = xptiRegisterCallback(
      stream_id, (uint16_t)xpti::trace_point_type_t::thread_begin, tp_callback);
  result = xptiRegisterCallback(
      stream_id, (uint16_t)xpti::trace_point_type_t::thread_end, tp_callback);
  result = xptiRegisterCallback(
      stream_id, (uint16_t)xpti::trace_point_type_t::wait_begin, tp_callback);
  result = xptiRegisterCallback(
      stream_id, (uint16_t)xpti::trace_point_type_t::wait_end, tp_callback);
  result = xptiRegisterCallback(
      stream_id, (uint16_t)xpti::trace_point_type_t::signal, tp_callback);

  auto ge =
      xptiMakeEvent("foo", &p, 0, (xpti::trace_activity_type_t)1, &instance);
  EXPECT_NE(ge, nullptr);

  int foo_return = 0;
  NOTIFY(stream_id, (uint16_t)xpti::trace_point_type_t::graph_create, ge,
         foo_return);
  NOTIFY(stream_id, (uint16_t)xpti::trace_point_type_t::node_create, ge,
         foo_return);
  NOTIFY(stream_id, (uint16_t)xpti::trace_point_type_t::edge_create, ge,
         foo_return);
  NOTIFY(stream_id, (uint16_t)xpti::trace_point_type_t::region_begin, ge,
         foo_return);
  NOTIFY(stream_id, (uint16_t)xpti::trace_point_type_t::region_end, ge,
         foo_return);
  NOTIFY(stream_id, (uint16_t)xpti::trace_point_type_t::task_begin, ge,
         foo_return);
  NOTIFY(stream_id, (uint16_t)xpti::trace_point_type_t::task_end, ge,
         foo_return);
  NOTIFY(stream_id, (uint16_t)xpti::trace_point_type_t::barrier_begin, ge,
         foo_return);
  NOTIFY(stream_id, (uint16_t)xpti::trace_point_type_t::barrier_end, ge,
         foo_return);
  NOTIFY(stream_id, (uint16_t)xpti::trace_point_type_t::lock_begin, ge,
         foo_return);
  NOTIFY(stream_id, (uint16_t)xpti::trace_point_type_t::lock_end, ge,
         foo_return);
  NOTIFY(stream_id, (uint16_t)xpti::trace_point_type_t::transfer_begin, ge,
         foo_return);
  NOTIFY(stream_id, (uint16_t)xpti::trace_point_type_t::transfer_end, ge,
         foo_return);
  NOTIFY(stream_id, (uint16_t)xpti::trace_point_type_t::thread_begin, ge,
         foo_return);
  NOTIFY(stream_id, (uint16_t)xpti::trace_point_type_t::thread_end, ge,
         foo_return);
  NOTIFY(stream_id, (uint16_t)xpti::trace_point_type_t::wait_begin, ge,
         foo_return);
  NOTIFY(stream_id, (uint16_t)xpti::trace_point_type_t::wait_end, ge,
         foo_return);
  NOTIFY(stream_id, (uint16_t)xpti::trace_point_type_t::signal, ge, foo_return);
}

TEST(xptiCorrectnessTest, xptiInitializeForUserDefinedTracePointTypes) {
  // We will test functionality of a subscriber
  // without actually creating a plugin
  uint8_t stream_id = xptiRegisterStream("test_foo");
  typedef enum {
    extn1_begin = XPTI_TRACE_POINT_BEGIN(0),
    extn1_end = XPTI_TRACE_POINT_END(0),
    extn2_begin = XPTI_TRACE_POINT_BEGIN(1),
    extn2_end = XPTI_TRACE_POINT_END(1)
  } tp_extension_t;

  auto tt_type =
      xptiRegisterUserDefinedTracePoint("test_foo_tool", extn1_begin);
  auto result = xptiRegisterCallback(stream_id, tt_type, tp_callback);
  EXPECT_EQ(result, xpti::result_t::XPTI_RESULT_SUCCESS);
  tt_type = xptiRegisterUserDefinedTracePoint("test_foo_tool", extn1_end);
  result = xptiRegisterCallback(stream_id, tt_type, tp_callback);
  EXPECT_EQ(result, xpti::result_t::XPTI_RESULT_SUCCESS);
  tt_type = xptiRegisterUserDefinedTracePoint("test_foo_tool", extn2_begin);
  result = xptiRegisterCallback(stream_id, tt_type, tp_callback);
  EXPECT_EQ(result, xpti::result_t::XPTI_RESULT_SUCCESS);
  tt_type = xptiRegisterUserDefinedTracePoint("test_foo_tool", extn2_end);
  result = xptiRegisterCallback(stream_id, tt_type, tp_callback);
  EXPECT_EQ(result, xpti::result_t::XPTI_RESULT_SUCCESS);
}

TEST(xptiCorrectnessTest, xptiNotifySubscribersForUserDefinedTracePointTypes) {
  uint64_t instance;
  xpti::payload_t p("foo", "foo.cpp", 1, 0, (void *)13);
  xptiForceSetTraceEnabled(true);

  uint8_t stream_id = xptiRegisterStream("test_foo");
  typedef enum {
    extn1_begin = XPTI_TRACE_POINT_BEGIN(0),
    extn1_end = XPTI_TRACE_POINT_END(0),
    extn2_begin = XPTI_TRACE_POINT_BEGIN(1),
    extn2_end = XPTI_TRACE_POINT_END(1)
  } tp_extension_t;

  auto tt_type1 =
      xptiRegisterUserDefinedTracePoint("test_foo_tool", extn1_begin);
  auto result = xptiRegisterCallback(stream_id, tt_type1, tp_callback);
  EXPECT_EQ(result, xpti::result_t::XPTI_RESULT_DUPLICATE);
  auto tt_type2 = xptiRegisterUserDefinedTracePoint("test_foo_tool", extn1_end);
  result = xptiRegisterCallback(stream_id, tt_type2, tp_callback);
  EXPECT_EQ(result, xpti::result_t::XPTI_RESULT_DUPLICATE);
  auto tt_type3 =
      xptiRegisterUserDefinedTracePoint("test_foo_tool", extn2_begin);
  result = xptiRegisterCallback(stream_id, tt_type3, tp_callback);
  EXPECT_EQ(result, xpti::result_t::XPTI_RESULT_DUPLICATE);
  auto tt_type4 = xptiRegisterUserDefinedTracePoint("test_foo_tool", extn2_end);
  result = xptiRegisterCallback(stream_id, tt_type4, tp_callback);
  EXPECT_EQ(result, xpti::result_t::XPTI_RESULT_DUPLICATE);

  auto ge =
      xptiMakeEvent("foo", &p, 0, (xpti::trace_activity_type_t)1, &instance);
  EXPECT_NE(ge, nullptr);

  int foo_return = 0;
  NOTIFY(stream_id, tt_type1, ge, foo_return);
  NOTIFY(stream_id, tt_type2, ge, foo_return);
  NOTIFY(stream_id, tt_type3, ge, foo_return);
  NOTIFY(stream_id, tt_type4, ge, foo_return);

  auto tool_id1 = XPTI_TOOL_ID(tt_type1);
  auto tool_id2 = XPTI_TOOL_ID(tt_type2);
  auto tool_id3 = XPTI_TOOL_ID(tt_type3);
  auto tool_id4 = XPTI_TOOL_ID(tt_type4);
  EXPECT_EQ(tool_id1, tool_id2);
  EXPECT_EQ(tool_id2, tool_id3);
  EXPECT_EQ(tool_id3, tool_id4);
  EXPECT_EQ(tool_id4, tool_id1);

  auto tp_id1 = XPTI_EXTRACT_USER_DEFINED_ID(tt_type1);
  auto tp_id2 = XPTI_EXTRACT_USER_DEFINED_ID(tt_type2);
  auto tp_id3 = XPTI_EXTRACT_USER_DEFINED_ID(tt_type3);
  auto tp_id4 = XPTI_EXTRACT_USER_DEFINED_ID(tt_type4);
  EXPECT_NE(tp_id1, tp_id2);
  EXPECT_NE(tp_id2, tp_id3);
  EXPECT_NE(tp_id3, tp_id4);
  EXPECT_NE(tp_id4, tp_id1);
}

TEST(xptiCorrectnessTest, xptiUserDefinedEventTypes) {
  uint64_t instance;
  xpti::payload_t p("foo", "foo.cpp", 1, 0, (void *)13);
  xptiForceSetTraceEnabled(true);

  uint8_t stream_id = xptiRegisterStream("test_foo");
  typedef enum {
    extn_ev1 = XPTI_EVENT(0),
    extn_ev2 = XPTI_EVENT(1),
    extn_ev3 = XPTI_EVENT(2),
    extn_ev4 = XPTI_EVENT(3)
  } event_extension_t;

  auto ev_type1 = xptiRegisterUserDefinedEventType("test_foo_tool", extn_ev1);
  auto ev_type2 = xptiRegisterUserDefinedEventType("test_foo_tool", extn_ev2);
  auto ev_type3 = xptiRegisterUserDefinedEventType("test_foo_tool", extn_ev3);
  auto ev_type4 = xptiRegisterUserDefinedEventType("test_foo_tool", extn_ev4);
  EXPECT_NE(ev_type1, ev_type2);
  EXPECT_NE(ev_type2, ev_type3);
  EXPECT_NE(ev_type3, ev_type4);
  EXPECT_NE(ev_type4, ev_type1);

  auto tool_id1 = XPTI_TOOL_ID(ev_type1);
  auto tool_id2 = XPTI_TOOL_ID(ev_type2);
  auto tool_id3 = XPTI_TOOL_ID(ev_type3);
  auto tool_id4 = XPTI_TOOL_ID(ev_type4);
  EXPECT_EQ(tool_id1, tool_id2);
  EXPECT_EQ(tool_id2, tool_id3);
  EXPECT_EQ(tool_id3, tool_id4);
  EXPECT_EQ(tool_id4, tool_id1);

  auto tp_id1 = XPTI_EXTRACT_USER_DEFINED_ID(ev_type1);
  auto tp_id2 = XPTI_EXTRACT_USER_DEFINED_ID(ev_type2);
  auto tp_id3 = XPTI_EXTRACT_USER_DEFINED_ID(ev_type3);
  auto tp_id4 = XPTI_EXTRACT_USER_DEFINED_ID(ev_type4);
  EXPECT_NE(tp_id1, tp_id2);
  EXPECT_NE(tp_id2, tp_id3);
  EXPECT_NE(tp_id3, tp_id4);
  EXPECT_NE(tp_id4, tp_id1);
}
