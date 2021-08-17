//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
#include "xpti_trace_framework.hpp"

#include <gtest/gtest.h>
#include <iostream>
#include <set>

static int func_callback_update = 0;

TEST(xptiApiTest, xptiInitializeBadInput) {
  auto Result = xptiInitialize(nullptr, 0, 0, nullptr);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_INVALIDARG);
}

TEST(xptiApiTest, xptiRegisterStringBadInput) {
  char *TStr;

  auto ID = xptiRegisterString(nullptr, nullptr);
  EXPECT_EQ(ID, xpti::invalid_id);
  ID = xptiRegisterString(nullptr, &TStr);
  EXPECT_EQ(ID, xpti::invalid_id);
  ID = xptiRegisterString("foo", nullptr);
  EXPECT_EQ(ID, xpti::invalid_id);
}

TEST(xptiApiTest, xptiRegisterStringGoodInput) {
  char *TStr = nullptr;

  auto ID = xptiRegisterString("foo", &TStr);
  EXPECT_NE(ID, xpti::invalid_id);
  EXPECT_NE(TStr, nullptr);
  EXPECT_STREQ("foo", TStr);
}

TEST(xptiApiTest, xptiLookupStringBadInput) {
  const char *TStr;
  xptiReset();
  TStr = xptiLookupString(-1);
  EXPECT_EQ(TStr, nullptr);
}

TEST(xptiApiTest, xptiLookupStringGoodInput) {
  char *TStr = nullptr;

  auto ID = xptiRegisterString("foo", &TStr);
  EXPECT_NE(ID, xpti::invalid_id);
  EXPECT_NE(TStr, nullptr);
  EXPECT_STREQ("foo", TStr);

  const char *LookUpString = xptiLookupString(ID);
  EXPECT_EQ(LookUpString, TStr);
  EXPECT_STREQ(LookUpString, TStr);
  EXPECT_STREQ("foo", LookUpString);
}

TEST(xptiApiTest, xptiGetUniqueId) {
  std::set<uint64_t> IDs;
  for (int i = 0; i < 10; ++i) {
    auto ID = xptiGetUniqueId();
    auto Loc = IDs.find(ID);
    EXPECT_EQ(Loc, IDs.end());
    IDs.insert(ID);
  }
}

TEST(xptiApiTest, xptiRegisterStreamBadInput) {
  auto ID = xptiRegisterStream(nullptr);
  EXPECT_EQ(ID, (uint8_t)xpti::invalid_id);
}

TEST(xptiApiTest, xptiRegisterStreamGoodInput) {
  auto ID = xptiRegisterStream("foo");
  EXPECT_NE(ID, xpti::invalid_id);
  auto NewID = xptiRegisterStream("foo");
  EXPECT_EQ(ID, NewID);
}

TEST(xptiApiTest, xptiUnregisterStreamBadInput) {
  auto Result = xptiUnregisterStream(nullptr);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_INVALIDARG);
}

TEST(xptiApiTest, xptiUnregisterStreamGoodInput) {
  auto ID = xptiRegisterStream("foo");
  EXPECT_NE(ID, xpti::invalid_id);
  auto Result = xptiUnregisterStream("NoSuchStream");
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_NOTFOUND);
  // Event though stream exists, no callbacks registered
  auto NewResult = xptiUnregisterStream("foo");
  EXPECT_EQ(NewResult, xpti::result_t::XPTI_RESULT_NOTFOUND);
}

TEST(xptiApiTest, xptiMakeEventBadInput) {
  xpti::payload_t P;
  auto Result =
      xptiMakeEvent(nullptr, &P, 0, (xpti::trace_activity_type_t)1, nullptr);
  EXPECT_EQ(Result, nullptr);
  P = xpti::payload_t("foo", "foo.cpp", 1, 0, (void *)13);
  EXPECT_NE(P.flags, 0);
  Result =
      xptiMakeEvent(nullptr, &P, 0, (xpti::trace_activity_type_t)1, nullptr);
  EXPECT_EQ(Result, nullptr);
  Result = xptiMakeEvent("foo", &P, 0, (xpti::trace_activity_type_t)1, nullptr);
  EXPECT_EQ(Result, nullptr);
}

TEST(xptiApiTest, xptiMakeEventGoodInput) {
  uint64_t instance;
  xpti::payload_t Payload("foo", "foo.cpp", 1, 0, (void *)13);
  auto Result = xptiMakeEvent("foo", &Payload, 0,
                              (xpti::trace_activity_type_t)1, &instance);
  EXPECT_NE(Result, nullptr);
  EXPECT_EQ(instance, 1);
  Payload = xpti::payload_t("foo", "foo.cpp", 1, 0, (void *)13);
  auto NewResult = xptiMakeEvent("foo", &Payload, 0,
                                 (xpti::trace_activity_type_t)1, &instance);
  EXPECT_EQ(Result, NewResult);
  EXPECT_EQ(instance, 2);
}

TEST(xptiApiTest, xptiFindEventBadInput) {
  auto Result = xptiFindEvent(0);
  EXPECT_EQ(Result, nullptr);
  Result = xptiFindEvent(1000000);
  EXPECT_EQ(Result, nullptr);
}

TEST(xptiApiTest, xptiFindEventGoodInput) {
  uint64_t Instance;
  xpti::payload_t Payload("foo", "foo.cpp", 1, 0, (void *)13);

  auto Result = xptiMakeEvent("foo", &Payload, 0,
                              (xpti::trace_activity_type_t)1, &Instance);
  EXPECT_NE(Result, nullptr);
  EXPECT_GT(Instance, 1);
  auto NewResult = xptiFindEvent(Result->unique_id);
  EXPECT_EQ(Result, NewResult);
}

TEST(xptiApiTest, xptiQueryPayloadBadInput) {
  auto Result = xptiQueryPayload(nullptr);
  EXPECT_EQ(Result, nullptr);
}

TEST(xptiApiTest, xptiQueryPayloadGoodInput) {
  uint64_t instance;
  xpti::payload_t Payload("foo", "foo.cpp", 1, 0, (void *)13);
  auto Result = xptiMakeEvent("foo", &Payload, 0,
                              (xpti::trace_activity_type_t)1, &instance);
  EXPECT_NE(Result, nullptr);
  EXPECT_GT(instance, 1);
  auto NewResult = xptiQueryPayload(Result);
  EXPECT_STREQ(Payload.name, NewResult->name);
  EXPECT_STREQ(Payload.source_file, NewResult->source_file);
  // NewResult->name_sid will have a string ID whereas 'Payload' will not
  EXPECT_NE(Payload.name_sid(), NewResult->name_sid());
  EXPECT_NE(Payload.source_file_sid(), NewResult->source_file_sid());
  EXPECT_EQ(Payload.line_no, NewResult->line_no);
}

TEST(xptiApiTest, xptiTraceEnabled) {
  // If no env is set, this should be false
  // The state is determined at app startup
  // XPTI_TRACE_ENABLE=1 or 0 and XPTI_FRAMEWORK_DISPATCHER=
  //   Result false
  auto Result = xptiTraceEnabled();
  EXPECT_EQ(Result, false);
}

XPTI_CALLBACK_API void trace_point_callback(uint16_t trace_type,
                                            xpti::trace_event_data_t *parent,
                                            xpti::trace_event_data_t *event,
                                            uint64_t instance,
                                            const void *user_data) {

  if (user_data)
    (*(int *)user_data) = 1;
}

XPTI_CALLBACK_API void trace_point_callback2(uint16_t trace_type,
                                             xpti::trace_event_data_t *parent,
                                             xpti::trace_event_data_t *event,
                                             uint64_t instance,
                                             const void *user_data) {
  if (user_data)
    (*(int *)user_data) = 1;
}

XPTI_CALLBACK_API void fn_callback(uint16_t trace_type,
                                   xpti::trace_event_data_t *parent,
                                   xpti::trace_event_data_t *event,
                                   uint64_t instance, const void *user_data) {
  func_callback_update++;
}

TEST(xptiApiTest, xptiRegisterCallbackBadInput) {
  uint8_t StreamID = xptiRegisterStream("foo");
  auto Result = xptiRegisterCallback(StreamID, 1, nullptr);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_INVALIDARG);
}

TEST(xptiApiTest, xptiRegisterCallbackGoodInput) {
  uint64_t instance;
  xpti::payload_t Payload("foo", "foo.cpp", 1, 0, (void *)13);

  auto Event = xptiMakeEvent("foo", &Payload, 0, (xpti::trace_activity_type_t)1,
                             &instance);
  EXPECT_NE(Event, nullptr);

  uint8_t StreamID = xptiRegisterStream("foo");
  auto Result = xptiRegisterCallback(StreamID, 1, trace_point_callback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(StreamID, 1, trace_point_callback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_DUPLICATE);

  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::function_begin,
      fn_callback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
}

TEST(xptiApiTest, xptiUnregisterCallbackBadInput) {
  uint8_t StreamID = xptiRegisterStream("foo");
  auto Result = xptiUnregisterCallback(StreamID, 1, nullptr);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_INVALIDARG);
}

TEST(xptiApiTest, xptiUnregisterCallbackGoodInput) {
  uint64_t instance;
  xpti::payload_t Payload("foo", "foo.cpp", 1, 0, (void *)13);

  auto Event = xptiMakeEvent("foo", &Payload, 0, (xpti::trace_activity_type_t)1,
                             &instance);
  EXPECT_NE(Event, nullptr);

  uint8_t StreamID = xptiRegisterStream("foo");
  auto Result = xptiUnregisterCallback(StreamID, 1, trace_point_callback2);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_NOTFOUND);
  Result = xptiRegisterCallback(StreamID, 1, trace_point_callback2);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiUnregisterCallback(StreamID, 1, trace_point_callback2);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiUnregisterCallback(StreamID, 1, trace_point_callback2);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_DUPLICATE);
  Result = xptiRegisterCallback(StreamID, 1, trace_point_callback2);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_UNDELETE);
}

TEST(xptiApiTest, xptiNotifySubscribersBadInput) {
  uint8_t StreamID = xptiRegisterStream("foo");
  auto Result =
      xptiNotifySubscribers(StreamID, 1, nullptr, nullptr, 0, nullptr);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_FALSE);
  xptiForceSetTraceEnabled(true);
  Result = xptiNotifySubscribers(StreamID, 1, nullptr, nullptr, 0, nullptr);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_INVALIDARG);
  Result = xptiNotifySubscribers(
      StreamID, (uint16_t)xpti::trace_point_type_t::function_begin, nullptr,
      nullptr, 0, nullptr);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_INVALIDARG);
}

TEST(xptiApiTest, xptiNotifySubscribersGoodInput) {
  uint64_t instance;
  xpti::payload_t Payload("foo", "foo.cpp", 1, 0, (void *)13);

  auto Event = xptiMakeEvent("foo", &Payload, 0, (xpti::trace_activity_type_t)1,
                             &instance);
  EXPECT_NE(Event, nullptr);

  uint8_t StreamID = xptiRegisterStream("foo");
  xptiForceSetTraceEnabled(true);
  int foo_return = 0;
  auto Result = xptiRegisterCallback(StreamID, 1, trace_point_callback2);
  Result = xptiNotifySubscribers(StreamID, 1, nullptr, Event, 0,
                                 (void *)(&foo_return));
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  EXPECT_EQ(foo_return, 1);
  int tmp = func_callback_update;
  // We allow notification with parent and event being null, only for trace
  // point type of function_begin/end; This test update checks for that
  Result = xptiNotifySubscribers(
      StreamID, (uint16_t)xpti::trace_point_type_t::function_begin, nullptr,
      nullptr, 0, "foo");
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  EXPECT_NE(tmp, func_callback_update);
  tmp = func_callback_update;
  Result = xptiNotifySubscribers(
      StreamID, (uint16_t)xpti::trace_point_type_t::function_begin, nullptr,
      (xpti::trace_event_data_t *)1, 0, "foo");
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  EXPECT_NE(tmp, func_callback_update);
}

TEST(xptiApiTest, xptiAddMetadataBadInput) {
  uint64_t instance;
  xpti::payload_t Payload("foo", "foo.cpp", 1, 0, (void *)13);

  auto Event = xptiMakeEvent("foo", &Payload, 0, (xpti::trace_activity_type_t)1,
                             &instance);
  EXPECT_NE(Event, nullptr);

  auto Result = xptiAddMetadata(nullptr, nullptr, nullptr);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_INVALIDARG);
  Result = xptiAddMetadata(Event, nullptr, nullptr);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_INVALIDARG);
  Result = xptiAddMetadata(Event, "foo", nullptr);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_INVALIDARG);
  Result = xptiAddMetadata(Event, nullptr, "bar");
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_INVALIDARG);
}

TEST(xptiApiTest, xptiAddMetadataGoodInput) {
  uint64_t instance;
  xpti::payload_t Payload("foo", "foo.cpp", 1, 0, (void *)13);

  auto Event = xptiMakeEvent("foo", &Payload, 0, (xpti::trace_activity_type_t)1,
                             &instance);
  EXPECT_NE(Event, nullptr);

  auto Result = xptiAddMetadata(Event, "foo", "bar");
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiAddMetadata(Event, "foo", "bar");
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_DUPLICATE);
}

TEST(xptiApiTest, xptiQueryMetadata) {
  uint64_t instance;
  xpti::payload_t Payload("foo", "foo.cpp", 1, 0, (void *)13);

  auto Event = xptiMakeEvent("foo", &Payload, 0, (xpti::trace_activity_type_t)1,
                             &instance);
  EXPECT_NE(Event, nullptr);

  auto md = xptiQueryMetadata(Event);
  EXPECT_NE(md, nullptr);

  auto Result = xptiAddMetadata(Event, "foo1", "bar1");
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);

  char *ts;
  EXPECT_TRUE(md->size() > 1);
  auto ID = (*md)[xptiRegisterString("foo1", &ts)];
  auto str = xptiLookupString(ID);
  EXPECT_STREQ(str, "bar1");
}
