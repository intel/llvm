//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
#include "xpti/xpti_trace_framework.h"
#include "xpti/xpti_trace_framework.hpp"

#include <gtest/gtest.h>
#include <iostream>
#include <set>
#include <vector>

static int func_callback_update = 0;
static int TPCB2Called = 0;
class xptiApiTest : public ::testing::Test {
protected:
  void SetUp() override {
    TPCB2Called = 0;
    func_callback_update = 0;
  }

  void TearDown() override { xptiReset(); }
};

TEST_F(xptiApiTest, xptiInitializeBadInput) {
  auto Result = xptiInitialize(nullptr, 0, 0, nullptr);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_INVALIDARG);
}

TEST_F(xptiApiTest, xptiRegisterStringBadInput) {
  char *TStr;

  auto ID = xptiRegisterString(nullptr, nullptr);
  EXPECT_EQ(ID, xpti::invalid_id);
  ID = xptiRegisterString(nullptr, &TStr);
  EXPECT_EQ(ID, xpti::invalid_id);
  ID = xptiRegisterString("foo", nullptr);
  EXPECT_EQ(ID, xpti::invalid_id);
}

TEST_F(xptiApiTest, xptiRegisterStringGoodInput) {
  char *TStr = nullptr;

  auto ID = xptiRegisterString("foo", &TStr);
  EXPECT_NE(ID, xpti::invalid_id);
  EXPECT_NE(TStr, nullptr);
  EXPECT_STREQ("foo", TStr);
}

TEST_F(xptiApiTest, xptiLookupStringBadInput) {
  const char *TStr;
  xptiReset();
  TStr = xptiLookupString(-1);
  EXPECT_EQ(TStr, nullptr);
}

TEST_F(xptiApiTest, xptiLookupStringGoodInput) {
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

TEST_F(xptiApiTest, xptiRegisterPayloadGoodInput) {
  xpti::payload_t p("foo", "foo.cpp", 10, 0, (void *)(0xdeadbeefull));

  auto ID = xptiRegisterPayload(&p);
  EXPECT_NE(ID, xpti::invalid_id);
  EXPECT_EQ(p.internal, ID);
  EXPECT_EQ(p.uid.hash(), ID);
}

TEST_F(xptiApiTest, xptiRegisterPayloadBadInput) {
  xpti::payload_t p;

  auto ID = xptiRegisterPayload(&p);
  EXPECT_EQ(ID, xpti::invalid_uid);
}

TEST_F(xptiApiTest, xptiGetUniqueId) {
  std::set<uint64_t> IDs;
  for (int i = 0; i < 10; ++i) {
    auto ID = xptiGetUniqueId();
    auto Loc = IDs.find(ID);
    EXPECT_EQ(Loc, IDs.end());
    IDs.insert(ID);
  }
}

TEST_F(xptiApiTest, xptiRegisterStreamBadInput) {
  auto ID = xptiRegisterStream(nullptr);
  EXPECT_EQ(ID, (uint8_t)xpti::invalid_id);
}

TEST_F(xptiApiTest, xptiRegisterStreamGoodInput) {
  auto ID = xptiRegisterStream("foo");
  EXPECT_NE(ID, xpti::invalid_id);
  auto NewID = xptiRegisterStream("foo");
  EXPECT_EQ(ID, NewID);
}

TEST_F(xptiApiTest, xptiUnregisterStreamBadInput) {
  auto Result = xptiUnregisterStream(nullptr);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_INVALIDARG);
}

TEST_F(xptiApiTest, xptiUnregisterStreamGoodInput) {
  auto ID = xptiRegisterStream("foo");
  EXPECT_NE(ID, xpti::invalid_id);
  auto Result = xptiUnregisterStream("NoSuchStream");
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_NOTFOUND);
  // Event though stream exists, no callbacks registered
  auto NewResult = xptiUnregisterStream("foo");
  EXPECT_EQ(NewResult, xpti::result_t::XPTI_RESULT_NOTFOUND);
}

TEST_F(xptiApiTest, xptiMakeEventBadInput) {
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

TEST_F(xptiApiTest, xptiMakeEventGoodInput) {
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

TEST_F(xptiApiTest, xptiFindEventBadInput) {
  auto Result = xptiFindEvent(0);
  EXPECT_EQ(Result, nullptr);
  Result = xptiFindEvent(1000000);
  EXPECT_EQ(Result, nullptr);
}

TEST_F(xptiApiTest, xptiFindEventGoodInput) {
  uint64_t Instance = 0;
  xpti::payload_t Payload("foo", "foo.cpp", 1, 0, (void *)13);

  auto Result = xptiMakeEvent("foo", &Payload, 0,
                              (xpti::trace_activity_type_t)1, &Instance);
  ASSERT_NE(Result, nullptr);
  EXPECT_EQ(Instance, 1);
  auto NewResult = xptiFindEvent(Result->unique_id);
  EXPECT_EQ(Result, NewResult);
}

TEST_F(xptiApiTest, xptiQueryPayloadBadInput) {
  auto Result = xptiQueryPayload(nullptr);
  EXPECT_EQ(Result, nullptr);
}

TEST_F(xptiApiTest, xptiQueryPayloadGoodInput) {
  uint64_t instance;
  xpti::payload_t Payload("foo", "foo.cpp", 1, 0, (void *)13);
  auto Result = xptiMakeEvent("foo", &Payload, 0,
                              (xpti::trace_activity_type_t)1, &instance);
  EXPECT_NE(Result, nullptr);
  EXPECT_EQ(instance, 1);
  auto NewResult = xptiQueryPayload(Result);
  ASSERT_NE(NewResult, nullptr);
  EXPECT_STREQ(Payload.name, NewResult->name);
  EXPECT_STREQ(Payload.source_file, NewResult->source_file);
  // NewResult->name_sid will have a string ID whereas 'Payload' will not
  EXPECT_NE(Payload.name_sid(), NewResult->name_sid());
  EXPECT_NE(Payload.source_file_sid(), NewResult->source_file_sid());
  EXPECT_EQ(Payload.line_no, NewResult->line_no);
}

TEST_F(xptiApiTest, xptiQueryPayloadByUIDGoodInput) {
  xpti::payload_t p("foo", "foo.cpp", 10, 0, (void *)(0xdeadbeefull));

  auto ID = xptiRegisterPayload(&p);
  EXPECT_NE(ID, xpti::invalid_id);
  EXPECT_EQ(p.internal, ID);
  EXPECT_EQ(p.uid.hash(), ID);

  auto pp = xptiQueryPayloadByUID(ID);
  EXPECT_EQ(p.internal, pp->internal);
  EXPECT_EQ(p.uid.hash(), pp->uid.hash());
}

TEST_F(xptiApiTest, xptiTraceEnabled) {
  // If no env is set, this should be false
  // The state is determined at app startup
  // XPTI_TRACE_ENABLE=1 or 0 and XPTI_FRAMEWORK_DISPATCHER=
  //   Result false
  auto Result = xptiTraceEnabled();
  EXPECT_EQ(Result, false);
}

void trace_point_callback(uint16_t /*trace_type*/,
                          xpti::trace_event_data_t * /*parent*/,
                          xpti::trace_event_data_t * /*event*/,
                          uint64_t /*instance*/, const void * /*user_data*/) {}

void trace_point_callback2(uint16_t /*trace_type*/,
                           xpti::trace_event_data_t * /*parent*/,
                           xpti::trace_event_data_t * /*event*/,
                           uint64_t /*instance*/, const void *user_data) {
  TPCB2Called = *static_cast<const int *>(user_data);
}

void fn_callback(uint16_t /*trace_type*/, xpti::trace_event_data_t * /*parent*/,
                 xpti::trace_event_data_t * /*event*/, uint64_t /*instance*/,
                 const void * /*user_data*/) {
  func_callback_update++;
}

TEST_F(xptiApiTest, xptiRegisterCallbackBadInput) {
  uint8_t StreamID = xptiRegisterStream("foo");
  auto Result = xptiRegisterCallback(StreamID, 1, nullptr);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_INVALIDARG);
}

TEST_F(xptiApiTest, xptiRegisterCallbackGoodInput) {
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

  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::offload_alloc_construct,
      fn_callback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);

  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::offload_alloc_associate,
      fn_callback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);

  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::offload_alloc_destruct,
      fn_callback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);

  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::offload_alloc_release,
      fn_callback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);

  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::offload_alloc_accessor,
      fn_callback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);

  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::mem_alloc_begin,
      fn_callback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);

  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::mem_alloc_end, fn_callback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);

  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::mem_release_begin,
      fn_callback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);

  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::mem_release_end,
      fn_callback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
}

TEST_F(xptiApiTest, xptiUnregisterCallbackBadInput) {
  uint8_t StreamID = xptiRegisterStream("foo");
  auto Result = xptiUnregisterCallback(StreamID, 1, nullptr);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_INVALIDARG);
}

TEST_F(xptiApiTest, xptiUnregisterCallbackGoodInput) {
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

TEST_F(xptiApiTest, xptiNotifySubscribersBadInput) {
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

  Result = xptiNotifySubscribers(
      StreamID, (uint16_t)xpti::trace_point_type_t::offload_alloc_construct,
      nullptr, nullptr, 0, nullptr);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_INVALIDARG);

  Result = xptiNotifySubscribers(
      StreamID, (uint16_t)xpti::trace_point_type_t::offload_alloc_associate,
      nullptr, nullptr, 0, nullptr);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_INVALIDARG);

  Result = xptiNotifySubscribers(
      StreamID, (uint16_t)xpti::trace_point_type_t::offload_alloc_destruct,
      nullptr, nullptr, 0, nullptr);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_INVALIDARG);

  Result = xptiNotifySubscribers(
      StreamID, (uint16_t)xpti::trace_point_type_t::offload_alloc_release,
      nullptr, nullptr, 0, nullptr);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_INVALIDARG);

  Result = xptiNotifySubscribers(
      StreamID, (uint16_t)xpti::trace_point_type_t::offload_alloc_accessor,
      nullptr, nullptr, 0, nullptr);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_INVALIDARG);

  Result = xptiNotifySubscribers(
      StreamID, (uint16_t)xpti::trace_point_type_t::mem_alloc_begin, nullptr,
      nullptr, 0, nullptr);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_INVALIDARG);

  Result = xptiNotifySubscribers(
      StreamID, (uint16_t)xpti::trace_point_type_t::mem_alloc_end, nullptr,
      nullptr, 0, nullptr);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_INVALIDARG);

  Result = xptiNotifySubscribers(
      StreamID, (uint16_t)xpti::trace_point_type_t::mem_release_begin, nullptr,
      nullptr, 0, nullptr);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_INVALIDARG);

  Result = xptiNotifySubscribers(
      StreamID, (uint16_t)xpti::trace_point_type_t::mem_release_end, nullptr,
      nullptr, 0, nullptr);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_INVALIDARG);
}

TEST_F(xptiApiTest, xptiNotifySubscribersGoodInput) {
  uint64_t instance;
  xpti::payload_t Payload("foo", "foo.cpp", 1, 0, (void *)13);

  auto Event = xptiMakeEvent("foo", &Payload, 0, (xpti::trace_activity_type_t)1,
                             &instance);
  EXPECT_NE(Event, nullptr);

  uint8_t StreamID = xptiRegisterStream("foo");
  xptiForceSetTraceEnabled(true);
  int FooUserData = 42;
  auto Result = xptiRegisterCallback(StreamID, 1, trace_point_callback2);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiNotifySubscribers(StreamID, 1, nullptr, Event, 0, &FooUserData);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  EXPECT_EQ(TPCB2Called, FooUserData);
  int tmp = func_callback_update;
  Result = xptiRegisterCallback(
      StreamID, static_cast<uint16_t>(xpti::trace_point_type_t::function_begin),
      fn_callback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
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

  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::offload_alloc_construct,
      fn_callback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::offload_alloc_associate,
      fn_callback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::offload_alloc_destruct,
      fn_callback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::offload_alloc_release,
      fn_callback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::offload_alloc_accessor,
      fn_callback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);

  xpti::offload_buffer_data_t UserBufferData{1, 5, "int", 4, 2, {3, 2, 0}};
  xpti::offload_buffer_association_data_t AssociationData{0x01020304,
                                                          0x05060708};
  xpti::offload_accessor_data_t UserAccessorData{0x01020304, 0x09000102, 1, 2};

  tmp = func_callback_update;
  Result = xptiNotifySubscribers(
      StreamID, (uint16_t)xpti::trace_point_type_t::offload_alloc_construct,
      nullptr, (xpti::trace_event_data_t *)1, 0, &UserBufferData);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiNotifySubscribers(
      StreamID, (uint16_t)xpti::trace_point_type_t::offload_alloc_associate,
      nullptr, (xpti::trace_event_data_t *)1, 0, &AssociationData);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiNotifySubscribers(
      StreamID, (uint16_t)xpti::trace_point_type_t::offload_alloc_release,
      nullptr, (xpti::trace_event_data_t *)1, 0, &AssociationData);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiNotifySubscribers(
      StreamID, (uint16_t)xpti::trace_point_type_t::offload_alloc_destruct,
      nullptr, (xpti::trace_event_data_t *)1, 0, &UserBufferData);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiNotifySubscribers(
      StreamID, (uint16_t)xpti::trace_point_type_t::offload_alloc_accessor,
      nullptr, (xpti::trace_event_data_t *)1, 0, &UserAccessorData);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);

  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::mem_alloc_begin,
      fn_callback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::mem_alloc_end, fn_callback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::mem_release_begin,
      fn_callback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::mem_release_end,
      fn_callback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);

  xpti::mem_alloc_data_t AllocData{10, 100, 1, 0};

  tmp = func_callback_update;
  Result = xptiNotifySubscribers(
      StreamID, (uint16_t)xpti::trace_point_type_t::mem_alloc_begin, nullptr,
      nullptr, 0, &AllocData);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  EXPECT_NE(tmp, func_callback_update);
  tmp = func_callback_update;
  Result = xptiNotifySubscribers(
      StreamID, (uint16_t)xpti::trace_point_type_t::mem_alloc_end, nullptr,
      (xpti::trace_event_data_t *)1, 0, &AllocData);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  EXPECT_NE(tmp, func_callback_update);

  Result = xptiNotifySubscribers(
      StreamID, (uint16_t)xpti::trace_point_type_t::mem_release_begin, nullptr,
      nullptr, 0, &AllocData);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  EXPECT_NE(tmp, func_callback_update);
  tmp = func_callback_update;
  Result = xptiNotifySubscribers(
      StreamID, (uint16_t)xpti::trace_point_type_t::mem_release_end, nullptr,
      (xpti::trace_event_data_t *)1, 0, &AllocData);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  EXPECT_NE(tmp, func_callback_update);
}

TEST_F(xptiApiTest, xptiAddMetadataBadInput) {
  uint64_t instance;
  xpti::payload_t Payload("foo", "foo.cpp", 1, 0, (void *)13);

  auto Event = xptiMakeEvent("foo", &Payload, 0, (xpti::trace_activity_type_t)1,
                             &instance);
  EXPECT_NE(Event, nullptr);

  auto Result = xptiAddMetadata(nullptr, nullptr, 0);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_INVALIDARG);
  Result = xptiAddMetadata(Event, nullptr, 0);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_INVALIDARG);
}

TEST_F(xptiApiTest, xptiAddMetadataGoodInput) {
  uint64_t instance;
  xpti::payload_t Payload("foo", "foo.cpp", 1, 0, (void *)13);

  auto Event = xptiMakeEvent("foo", &Payload, 0, (xpti::trace_activity_type_t)1,
                             &instance);
  EXPECT_NE(Event, nullptr);

  xpti::object_id_t ID = xptiRegisterObject("bar", 3, 0);
  auto Result = xptiAddMetadata(Event, "foo", ID);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiAddMetadata(Event, "foo", ID);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_DUPLICATE);
}

TEST_F(xptiApiTest, xptiQueryMetadata) {
  uint64_t instance;
  xpti::payload_t Payload("foo", "foo.cpp", 1, 0, (void *)13);

  auto Event = xptiMakeEvent("foo", &Payload, 0, (xpti::trace_activity_type_t)1,
                             &instance);
  EXPECT_NE(Event, nullptr);

  auto md = xptiQueryMetadata(Event);
  EXPECT_NE(md, nullptr);

  xpti::object_id_t ID = xptiRegisterObject("bar1", 4, 0);
  auto Result = xptiAddMetadata(Event, "foo1", ID);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);

  char *ts;
  EXPECT_EQ(md->size(), 1);
  auto MDID = (*md)[xptiRegisterString("foo1", &ts)];
  auto obj = xptiLookupObject(MDID);
  std::string str{obj.data, obj.size};
  EXPECT_EQ(str, "bar1");
}
