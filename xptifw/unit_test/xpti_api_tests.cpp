//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
#include "xpti_trace_framework.hpp"

#include <iostream>
#include <set>

#include <gtest/gtest.h>

TEST(xptiApiTest, xptiInitializeBadInput) {
  auto result = xptiInitialize(nullptr, 0, 0, nullptr);
  EXPECT_EQ(result, xpti::result_t::XPTI_RESULT_INVALIDARG);
}

TEST(xptiApiTest, xptiRegisterStringBadInput) {
  char *tstr;

  auto id = xptiRegisterString(nullptr, nullptr);
  EXPECT_EQ(id, xpti::invalid_id);
  id = xptiRegisterString(nullptr, &tstr);
  EXPECT_EQ(id, xpti::invalid_id);
  id = xptiRegisterString("foo", nullptr);
  EXPECT_EQ(id, xpti::invalid_id);
}

TEST(xptiApiTest, xptiRegisterStringGoodInput) {
  char *tstr = nullptr;

  auto id = xptiRegisterString("foo", &tstr);
  EXPECT_NE(id, xpti::invalid_id);
  EXPECT_NE(tstr, nullptr);
  EXPECT_STREQ("foo", tstr);
}

TEST(xptiApiTest, xptiLookupStringBadInput) {
  const char *tstr;
  xptiReset();
  tstr = xptiLookupString(-1);
  EXPECT_EQ(tstr, nullptr);
}

TEST(xptiApiTest, xptiLookupStringGoodInput) {
  char *tstr = nullptr;

  auto id = xptiRegisterString("foo", &tstr);
  EXPECT_NE(id, xpti::invalid_id);
  EXPECT_NE(tstr, nullptr);
  EXPECT_STREQ("foo", tstr);

  const char *lstr = xptiLookupString(id);
  EXPECT_EQ(lstr, tstr);
  EXPECT_STREQ(lstr, tstr);
  EXPECT_STREQ("foo", lstr);
}

TEST(xptiApiTest, xptiGetUniqueId) {
  std::set<uint64_t> ids;
  for (int i = 0; i < 10; ++i) {
    auto id = xptiGetUniqueId();
    auto loc = ids.find(id);
    EXPECT_EQ(loc, ids.end());
    ids.insert(id);
  }
}

TEST(xptiApiTest, xptiRegisterStreamBadInput) {
  auto id = xptiRegisterStream(nullptr);
  EXPECT_EQ(id, (uint8_t)xpti::invalid_id);
}

TEST(xptiApiTest, xptiRegisterStreamGoodInput) {
  auto id = xptiRegisterStream("foo");
  EXPECT_NE(id, xpti::invalid_id);
  auto new_id = xptiRegisterStream("foo");
  EXPECT_EQ(id, new_id);
}

TEST(xptiApiTest, xptiUnregisterStreamBadInput) {
  auto result = xptiUnregisterStream(nullptr);
  EXPECT_EQ(result, xpti::result_t::XPTI_RESULT_INVALIDARG);
}

TEST(xptiApiTest, xptiUnregisterStreamGoodInput) {
  auto id = xptiRegisterStream("foo");
  EXPECT_NE(id, xpti::invalid_id);
  auto res = xptiUnregisterStream("NoSuchStream");
  EXPECT_EQ(res, xpti::result_t::XPTI_RESULT_NOTFOUND);
  // Event though stream exists, no callbacks registered
  auto new_res = xptiUnregisterStream("foo");
  EXPECT_EQ(new_res, xpti::result_t::XPTI_RESULT_NOTFOUND);
}

TEST(xptiApiTest, xptiMakeEventBadInput) {
  xpti::payload_t p;
  auto result =
      xptiMakeEvent(nullptr, &p, 0, (xpti::trace_activity_type_t)1, nullptr);
  EXPECT_EQ(result, nullptr);
  p = xpti::payload_t("foo", "foo.cpp", 1, 0, (void *)13);
  EXPECT_NE(p.flags, 0);
  result =
      xptiMakeEvent(nullptr, &p, 0, (xpti::trace_activity_type_t)1, nullptr);
  EXPECT_EQ(result, nullptr);
  result = xptiMakeEvent("foo", &p, 0, (xpti::trace_activity_type_t)1, nullptr);
  EXPECT_EQ(result, nullptr);
}

TEST(xptiApiTest, xptiMakeEventGoodInput) {
  uint64_t instance;
  xpti::payload_t p("foo", "foo.cpp", 1, 0, (void *)13);
  auto result =
      xptiMakeEvent("foo", &p, 0, (xpti::trace_activity_type_t)1, &instance);
  EXPECT_NE(result, nullptr);
  EXPECT_EQ(instance, 1);
  p = xpti::payload_t("foo", "foo.cpp", 1, 0, (void *)13);
  auto new_result =
      xptiMakeEvent("foo", &p, 0, (xpti::trace_activity_type_t)1, &instance);
  EXPECT_EQ(result, new_result);
  EXPECT_EQ(instance, 2);
}

TEST(xptiApiTest, xptiFindEventBadInput) {
  auto result = xptiFindEvent(0);
  EXPECT_EQ(result, nullptr);
  result = xptiFindEvent(1000000);
  EXPECT_EQ(result, nullptr);
}

TEST(xptiApiTest, xptiFindEventGoodInput) {
  uint64_t instance;
  xpti::payload_t p("foo", "foo.cpp", 1, 0, (void *)13);

  auto result =
      xptiMakeEvent("foo", &p, 0, (xpti::trace_activity_type_t)1, &instance);
  EXPECT_NE(result, nullptr);
  EXPECT_GT(instance, 1);
  auto new_result = xptiFindEvent(result->unique_id);
  EXPECT_EQ(result, new_result);
}

TEST(xptiApiTest, xptiQueryPayloadBadInput) {
  auto result = xptiQueryPayload(nullptr);
  EXPECT_EQ(result, nullptr);
}

TEST(xptiApiTest, xptiQueryPayloadGoodInput) {
  uint64_t instance;
  xpti::payload_t p("foo", "foo.cpp", 1, 0, (void *)13);
  ;
  auto result =
      xptiMakeEvent("foo", &p, 0, (xpti::trace_activity_type_t)1, &instance);
  EXPECT_NE(result, nullptr);
  EXPECT_GT(instance, 1);
  auto new_result = xptiQueryPayload(result);
  EXPECT_STREQ(p.name, new_result->name);
  EXPECT_STREQ(p.source_file, new_result->source_file);
  // new_result->name_sid will have a string id whereas 'p' will not
  EXPECT_NE(p.name_sid, new_result->name_sid);
  EXPECT_NE(p.source_file_sid, new_result->source_file_sid);
  EXPECT_EQ(p.line_no, new_result->line_no);
}

TEST(xptiApiTest, xptiTraceEnabled) {
  // If no env is set, this should be false
  // The state is determined at app startup
  // XPTI_TRACE_ENABLE=1 or 0 and XPTI_FRAMEWORK_DISPATCHER=
  //   result false
  auto result = xptiTraceEnabled();
  EXPECT_EQ(result, false);
}

XPTI_CALLBACK_API void trace_point_callback(
    uint16_t                  trace_type,
    xpti::trace_event_data_t *parent,
    xpti::trace_event_data_t *event,
    uint64_t                  instance,
    const void               *user_data) {

    if(user_data)
      (*(int *)user_data) = 1;
}

XPTI_CALLBACK_API void trace_point_callback2(
    uint16_t                  trace_type,
    xpti::trace_event_data_t *parent,
    xpti::trace_event_data_t *event,
    uint64_t                  instance,
    const void               *user_data) {
    if(user_data)
      (*(int *)user_data) = 1;
}

TEST(xptiApiTest, xptiRegisterCallbackBadInput) {
  uint8_t stream_id = xptiRegisterStream("foo");
  auto result = xptiRegisterCallback(stream_id, 1, nullptr);
  EXPECT_EQ(result, xpti::result_t::XPTI_RESULT_INVALIDARG);
}

TEST(xptiApiTest, xptiRegisterCallbackGoodInput) {
  uint64_t instance;
  xpti::payload_t p("foo", "foo.cpp", 1, 0, (void *)13);

  auto event =
      xptiMakeEvent("foo", &p, 0, (xpti::trace_activity_type_t)1, &instance);
  EXPECT_NE(event, nullptr);

  uint8_t stream_id = xptiRegisterStream("foo");
  auto result = xptiRegisterCallback(stream_id, 1, trace_point_callback);
  EXPECT_EQ(result, xpti::result_t::XPTI_RESULT_SUCCESS);
  result = xptiRegisterCallback(stream_id, 1, trace_point_callback);
  EXPECT_EQ(result, xpti::result_t::XPTI_RESULT_DUPLICATE);
}

TEST(xptiApiTest, xptiUnregisterCallbackBadInput) {
  uint8_t stream_id = xptiRegisterStream("foo");
  auto result = xptiUnregisterCallback(stream_id, 1, nullptr);
  EXPECT_EQ(result, xpti::result_t::XPTI_RESULT_INVALIDARG);
}

TEST(xptiApiTest, xptiUnregisterCallbackGoodInput) {
  uint64_t instance;
  xpti::payload_t p("foo", "foo.cpp", 1, 0, (void *)13);

  auto event =
      xptiMakeEvent("foo", &p, 0, (xpti::trace_activity_type_t)1, &instance);
  EXPECT_NE(event, nullptr);

  uint8_t stream_id = xptiRegisterStream("foo");
  auto result = xptiUnregisterCallback(stream_id, 1, trace_point_callback2);
  EXPECT_EQ(result, xpti::result_t::XPTI_RESULT_NOTFOUND);
  result = xptiRegisterCallback(stream_id, 1, trace_point_callback2);
  EXPECT_EQ(result, xpti::result_t::XPTI_RESULT_SUCCESS);
  result = xptiUnregisterCallback(stream_id, 1, trace_point_callback2);
  EXPECT_EQ(result, xpti::result_t::XPTI_RESULT_SUCCESS);
  result = xptiUnregisterCallback(stream_id, 1, trace_point_callback2);
  EXPECT_EQ(result, xpti::result_t::XPTI_RESULT_DUPLICATE);
  result = xptiRegisterCallback(stream_id, 1, trace_point_callback2);
  EXPECT_EQ(result, xpti::result_t::XPTI_RESULT_UNDELETE);
}

TEST(xptiApiTest, xptiNotifySubscribersBadInput) {
  uint8_t stream_id = xptiRegisterStream("foo");
  auto result = xptiNotifySubscribers(stream_id, 1, nullptr, nullptr, 0, nullptr);
  EXPECT_EQ(result, xpti::result_t::XPTI_RESULT_FALSE);
  xptiForceSetTraceEnabled(true);
  result = xptiNotifySubscribers(stream_id, 1, nullptr, nullptr, 0, nullptr);
  EXPECT_EQ(result, xpti::result_t::XPTI_RESULT_INVALIDARG);
}

TEST(xptiApiTest, xptiNotifySubscribersGoodInput) {
  uint64_t instance;
  xpti::payload_t p("foo", "foo.cpp", 1, 0, (void *)13);

  auto event =
      xptiMakeEvent("foo", &p, 0, (xpti::trace_activity_type_t)1, &instance);
  EXPECT_NE(event, nullptr);

  uint8_t stream_id = xptiRegisterStream("foo");
  xptiForceSetTraceEnabled(true);
  int foo_return = 0;
  auto result = xptiRegisterCallback(stream_id, 1, trace_point_callback2);
  result = xptiNotifySubscribers(stream_id, 1, nullptr, event, 0, (void *)(&foo_return));
  EXPECT_EQ(result, xpti::result_t::XPTI_RESULT_SUCCESS);
  EXPECT_EQ(foo_return, 1);
}

TEST(xptiApiTest, xptiAddMetadataBadInput) {
  uint64_t instance;
  xpti::payload_t p("foo", "foo.cpp", 1, 0, (void *)13);

  auto event =
      xptiMakeEvent("foo", &p, 0, (xpti::trace_activity_type_t)1, &instance);
  EXPECT_NE(event, nullptr);

  auto result = xptiAddMetadata(nullptr, nullptr, nullptr);
  EXPECT_EQ(result, xpti::result_t::XPTI_RESULT_INVALIDARG);
  result = xptiAddMetadata(event, nullptr, nullptr);
  EXPECT_EQ(result, xpti::result_t::XPTI_RESULT_INVALIDARG);
  result = xptiAddMetadata(event, "foo", nullptr);
  EXPECT_EQ(result, xpti::result_t::XPTI_RESULT_INVALIDARG);
  result = xptiAddMetadata(event, nullptr, "bar");
  EXPECT_EQ(result, xpti::result_t::XPTI_RESULT_INVALIDARG);
}

TEST(xptiApiTest, xptiAddMetadataGoodInput) {
  uint64_t instance;
  xpti::payload_t p("foo", "foo.cpp", 1, 0, (void *)13);

  auto event =
      xptiMakeEvent("foo", &p, 0, (xpti::trace_activity_type_t)1, &instance);
  EXPECT_NE(event, nullptr);

  auto result = xptiAddMetadata(event, "foo", "bar");
  EXPECT_EQ(result, xpti::result_t::XPTI_RESULT_SUCCESS);
  result = xptiAddMetadata(event, "foo", "bar");
  EXPECT_EQ(result, xpti::result_t::XPTI_RESULT_DUPLICATE);
}

TEST(xptiApiTest, xptiQueryMetadata) {
  uint64_t instance;
  xpti::payload_t p("foo", "foo.cpp", 1, 0, (void *)13);

  auto event =
      xptiMakeEvent("foo", &p, 0, (xpti::trace_activity_type_t)1, &instance);
  EXPECT_NE(event, nullptr);

  auto md = xptiQueryMetadata(event);
  EXPECT_NE(md, nullptr);

  auto result = xptiAddMetadata(event, "foo1", "bar1");
  EXPECT_EQ(result, xpti::result_t::XPTI_RESULT_SUCCESS);

  char *ts;
  EXPECT_TRUE(md->size() > 1);
  auto id = (*md)[xptiRegisterString("foo1", &ts)];
  auto str = xptiLookupString(id);
  EXPECT_STREQ(str, "bar1");
}
