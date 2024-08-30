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
  EXPECT_NE(ID, xpti::invalid_uid);
  EXPECT_EQ(p.internal, ID);
  // EXPECT_EQ(p.uid.hash(), ID);

  auto TP = xptiCreateTracepoint(p.name, p.source_file, p.line_no, p.column_no);
  xpti::trace_event_data_t *Ev = TP->event_ref();

  EXPECT_NE(Ev, nullptr);
  EXPECT_EQ(std::string(Ev->reserved.payload->name), std::string(p.name));
}

TEST_F(xptiApiTest, xptiRegisterPayloadBadInput) {
  xpti::payload_t p;

  auto ID = xptiRegisterPayload(nullptr);
  EXPECT_EQ(ID, xpti::invalid_uid);
  ID = xptiRegisterPayload(&p);
  EXPECT_EQ(ID, xpti::invalid_uid);
}

TEST_F(xptiApiTest, xptiPayloadBadInput) {
  xpti::payload_t p("foo", "foo.cpp", 10, 0, (void *)(0xdeadbeefull));

  auto ID = xptiCreateTracepoint(p.name, p.source_file, p.line_no, p.column_no);
  EXPECT_NE(ID, nullptr);

  auto UID = xptiRegisterPayload(&p);
  EXPECT_NE(UID, xpti::invalid_uid);
  EXPECT_EQ(p.internal, UID);

  auto TP = xptiCreateTracepoint(p.name, p.source_file, p.line_no, p.column_no);
  EXPECT_GT(TP->instance(), ID->instance());
  EXPECT_NE(TP->uid64(), ID->uid64());
  xpti::trace_event_data_t *Ev = TP->event_ref();

  EXPECT_NE(Ev, nullptr);
  EXPECT_EQ(std::string(Ev->reserved.payload->name), std::string(p.name));

  xpti::payload_t pp;
  auto NewTP =
      xptiCreateTracepoint(pp.name, pp.source_file, pp.line_no, pp.column_no);
  // Earlier, this should have been a nullptr
  EXPECT_NE(NewTP, nullptr);
  xpti::trace_event_data_t *NewEv = NewTP->event_ref();

  EXPECT_NE(NewEv, nullptr);
  EXPECT_NE(NewEv->reserved.payload->name, nullptr);
  EXPECT_EQ(std::string(NewEv->reserved.payload->name), std::string("unknown"));
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

TEST_F(xptiApiTest, xptiCreateTracepoint) {
  xpti::payload_t p("foo", "foo.cpp", 10, 0, (void *)(0xdeadbeefull));
  xpti::uid128_t UID;

  EXPECT_EQ(xpti::is_valid_uid(UID), false);
  EXPECT_EQ(xpti::is_valid_payload(&p), true);
  EXPECT_EQ(UID.p1, 0u);
  EXPECT_EQ(UID.p2, 0u);
  EXPECT_EQ(UID.instance, 0u);
  EXPECT_EQ(UID.uid64, 0u);
  bool test =
      p.flags & static_cast<uint16_t>(xpti::payload_flag_t::PayloadRegistered);
  EXPECT_EQ(test, false);
  test = p.flags & static_cast<uint16_t>(xpti::payload_flag_t::NameAvailable);
  EXPECT_EQ(test, true);
  test = p.flags &
         static_cast<uint64_t>(xpti::payload_flag_t::SourceFileAvailable);
  EXPECT_EQ(test, true);
  test =
      p.flags & static_cast<uint64_t>(xpti::payload_flag_t::LineInfoAvailable);
  EXPECT_EQ(test, true);
  test = p.flags &
         static_cast<uint64_t>(xpti::payload_flag_t::ColumnInfoAvailable);

  auto TP = xptiCreateTracepoint(p.name, p.source_file, p.line_no, p.column_no);
  xpti::trace_event_data_t *Ev = TP->event_ref();
  EXPECT_NE(TP, nullptr);
  auto payload = TP->payload_ref();
  test = payload->flags &
         static_cast<uint64_t>(xpti::payload_flag_t::PayloadRegistered);
  EXPECT_EQ(test, true);
  auto event = TP->event_ref();
  test = event->flags &
         static_cast<uint64_t>(xpti::trace_event_flag_t::PayloadAvailable);
  EXPECT_EQ(test, true);
  test = event->flags &
         static_cast<uint64_t>(xpti::trace_event_flag_t::UIDAvailable);
  EXPECT_EQ(test, true);
  test = event->flags &
         static_cast<uint64_t>(xpti::trace_event_flag_t::ActivityTypeAvailable);
  EXPECT_EQ(test, true);
  test = event->flags &
         static_cast<uint64_t>(xpti::trace_event_flag_t::EventTypeAvailable);
  EXPECT_EQ(test, true);

  auto uid64 = TP->uid64();
  EXPECT_NE(uid64, xpti::invalid_uid);

  auto P1 = xptiLookupPayload(uid64);
  EXPECT_EQ(const_cast<xpti_payload_t *>(P1)->payload_ref(), payload);

  EXPECT_NE(payload, nullptr);
  EXPECT_NE(payload->internal, xpti::invalid_uid);
  // Since 'p' is not sent in for it to be updated
  EXPECT_NE(payload->internal, p.internal);
  // p.source_file_sid is not set as the payload has not been used to create an
  // event
  EXPECT_NE(payload->source_file_sid(), p.source_file_sid());
  EXPECT_EQ(payload->line_no, p.line_no);
  EXPECT_EQ(payload->column_no, p.column_no);
}

TEST_F(xptiApiTest, xptiQueryLookupPayloadGoodInput) {
  xpti::payload_t p("foo", "foo.cpp", 10, 0, (void *)(0xdeadbeefull));

  EXPECT_EQ(xpti::is_valid_payload(&p), true);
  bool test =
      p.flags & static_cast<uint16_t>(xpti::payload_flag_t::PayloadRegistered);
  EXPECT_EQ(test, false);
  test = p.flags & static_cast<uint16_t>(xpti::payload_flag_t::NameAvailable);
  EXPECT_EQ(test, true);
  test = p.flags &
         static_cast<uint64_t>(xpti::payload_flag_t::SourceFileAvailable);
  EXPECT_EQ(test, true);
  test =
      p.flags & static_cast<uint64_t>(xpti::payload_flag_t::LineInfoAvailable);
  EXPECT_EQ(test, true);
  test = p.flags &
         static_cast<uint64_t>(xpti::payload_flag_t::ColumnInfoAvailable);

  auto TP = xptiCreateTracepoint(p.name, p.source_file, p.line_no, p.column_no);
  xpti::trace_event_data_t *Ev = TP->event_ref();
  EXPECT_NE(TP, nullptr);
  auto payload = TP->payload_ref();

  auto tp1 = xptiLookupPayload(0);
  EXPECT_NE(tp1, TP->payload());
  EXPECT_EQ(tp1, nullptr);

  tp1 = xptiLookupPayload(10000);
  EXPECT_NE(tp1, TP->payload());
  EXPECT_EQ(tp1, nullptr);

  test = payload->flags &
         static_cast<uint64_t>(xpti::payload_flag_t::PayloadRegistered);
  EXPECT_EQ(test, true);
  auto event = TP->event_ref();
  test = event->flags &
         static_cast<uint64_t>(xpti::trace_event_flag_t::PayloadAvailable);
  EXPECT_EQ(test, true);
  test = event->flags &
         static_cast<uint64_t>(xpti::trace_event_flag_t::UIDAvailable);
  EXPECT_EQ(test, true);
  test = event->flags &
         static_cast<uint64_t>(xpti::trace_event_flag_t::ActivityTypeAvailable);
  EXPECT_EQ(test, true);
  test = event->flags &
         static_cast<uint64_t>(xpti::trace_event_flag_t::EventTypeAvailable);
  EXPECT_EQ(test, true);

  auto uid64 = TP->uid64();
  EXPECT_NE(uid64, xpti::invalid_uid);

  auto P1 = xptiLookupPayload(uid64);
  EXPECT_EQ(const_cast<xpti_payload_t *>(P1)->payload_ref(), payload);
  auto pp = xptiQueryPayloadByUID(uid64);
  EXPECT_EQ(pp, payload);
  EXPECT_EQ(p.uid.p1, 0);
  EXPECT_EQ(p.uid.p2, 0);
  // Also, with the current 128-bit version, we are not using the hash value,
  // hence the payload->internal field will always be set to xpti::invalid_uid
  // for lookup by UID
  // EXPECT_EQ(p.internal, pp->internal);
}

TEST_F(xptiApiTest, xptiGetAndSetDefaultEventType) {
  auto ID = xptiGetDefaultEventType();
  EXPECT_EQ((int)ID, (int)xpti::trace_event_type_t::algorithm);
  auto Result =
      xptiSetDefaultEventType(xpti::trace_event_type_t::unknown_event);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_INVALIDARG);
  Result = xptiSetDefaultEventType(xpti::trace_event_type_t::graph);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  auto ID2 = xptiGetDefaultEventType();
  EXPECT_EQ((int)ID2, (int)xpti::trace_event_type_t::graph);
}

TEST_F(xptiApiTest, xptiGetAndSetDefaultStreamID) {
  auto ID = xptiGetDefaultStreamID();
  EXPECT_NE(ID, 0); // As we have a new default stream "xpti.framework" as the
                    // deault is nothing is set
  auto Result = xptiSetDefaultStreamID(-1);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_INVALIDARG);
  Result = xptiSetDefaultStreamID(42);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  auto ID2 = xptiGetDefaultStreamID();
  EXPECT_EQ(ID2, 42);
}

TEST_F(xptiApiTest, xptiGetAndSetDefaultTraceType) {
  auto ID = xptiGetDefaultTraceType();
  EXPECT_EQ((int)ID, (int)xpti::trace_point_type_t::function_begin);
  auto Result = xptiSetDefaultTraceType(xpti::trace_point_type_t::unknown_type);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_INVALIDARG);
  Result = xptiSetDefaultTraceType(xpti::trace_point_type_t::task_begin);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  auto ID2 = xptiGetDefaultTraceType();
  EXPECT_EQ((int)ID2, (int)xpti::trace_point_type_t::task_begin);
}

TEST_F(xptiApiTest, xptiGetTracePointScopeData) {
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

TEST_F(xptiApiTest, xptiQueryLookupPayloadBadInput) {
  auto UID = xpti::invalid_uid;
  auto Payload = xptiLookupPayload(UID);
  EXPECT_EQ(Payload, nullptr);
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
  EXPECT_NE(P.flags, 0u);
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
  EXPECT_EQ(instance, 1u);
  Payload = xpti::payload_t("foo", "foo.cpp", 1, 0, (void *)13);
  auto NewResult = xptiMakeEvent("foo", &Payload, 0,
                                 (xpti::trace_activity_type_t)1, &instance);
  // New implementation with 128-bit keys will return a new trace event for each
  // instance
  EXPECT_NE(Result, NewResult);
  EXPECT_EQ(instance, 2u);
}

TEST_F(xptiApiTest, xptiCreateTracepointBadInput) {
  auto Result = xptiCreateTracepoint(nullptr, nullptr, 0, 0);
  EXPECT_NE(Result, nullptr);
  auto Payload = Result->payload_ref();
  EXPECT_NE(Payload->name, nullptr);
  EXPECT_EQ(std::string(Payload->name), std::string("unknown"));
  EXPECT_EQ(std::string(Payload->source_file), std::string("unknown-file"));
  EXPECT_EQ(Payload->line_no, 0);
  EXPECT_EQ(Payload->column_no, 0);
}

TEST_F(xptiApiTest, xptiCreateTracepointGoodInput) {
  xpti::payload_t p("foo", "foo.cpp", 10, 0, (void *)(0xdeadbeefull));

  auto TP1 =
      xptiCreateTracepoint(p.name, p.source_file, p.line_no, p.column_no);
  xpti::trace_event_data_t *Ev1 = TP1->event_ref();
  auto TP2 =
      xptiCreateTracepoint(p.name, p.source_file, p.line_no, p.column_no);
  xpti::trace_event_data_t *Ev2 = TP2->event_ref();
  EXPECT_NE(TP1, nullptr);
  EXPECT_NE(TP2, nullptr);
  EXPECT_NE(TP1, TP2);
  auto Instance1 = TP1->instance();
  auto Instance2 = TP2->instance();
  EXPECT_NE(Instance1, Instance2);
  EXPECT_GT(Instance2, Instance1);
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
  EXPECT_EQ(Instance, 1u);
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
  EXPECT_EQ(instance, 1u);
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
  EXPECT_NE(ID, xpti::invalid_uid);
  EXPECT_EQ(p.internal, ID);
  // p.uid.hash() is legacy way of generating 64-bit hash values; current
  // implementation is using 128-bit keys, so the uid.hash() is never used
  // EXPECT_EQ(p.uid.hash(), ID);

  auto pp = xptiQueryPayloadByUID(ID);
  EXPECT_EQ(p.uid.p1, pp->uid.p1);
  EXPECT_EQ(p.uid.p2, pp->uid.p2);
  // Also, with the current 128-bit version, we are not using the hash value,
  // hence the payload->internal field will always be set to xpti::invalid_uid
  // for lookup by UID
  // EXPECT_EQ(p.internal, pp->internal);
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
      StreamID,
      (uint16_t)xpti::trace_point_type_t::offload_alloc_memory_object_construct,
      fn_callback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);

  Result = xptiRegisterCallback(
      StreamID,
      (uint16_t)xpti::trace_point_type_t::offload_alloc_memory_object_associate,
      fn_callback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);

  Result = xptiRegisterCallback(
      StreamID,
      (uint16_t)xpti::trace_point_type_t::offload_alloc_memory_object_destruct,
      fn_callback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);

  Result = xptiRegisterCallback(
      StreamID,
      (uint16_t)xpti::trace_point_type_t::offload_alloc_memory_object_release,
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

TEST_F(xptiApiTest, xptiCheckTraceEnabledGoodInput) {
  uint64_t instance;
  xptiForceSetTraceEnabled(true);
  xpti::payload_t Payload("foo", "foo.cpp", 1, 0, (void *)13);

  auto Event = xptiMakeEvent("foo", &Payload, 0, (xpti::trace_activity_type_t)1,
                             &instance);
  EXPECT_NE(Event, nullptr);

  auto ID = xptiRegisterStream("CheckTest");

  auto Result = xptiRegisterCallback(
      ID, (uint16_t)xpti::trace_point_type_t::function_begin, fn_callback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);

  auto Check = xptiCheckTraceEnabled(ID, 0);
  EXPECT_EQ(Check, true);

  Check = xptiCheckTraceEnabled(
      ID, (uint16_t)xpti::trace_point_type_t::function_begin);
  EXPECT_EQ(Check, true);
  Check =
      xptiCheckTraceEnabled(ID, (uint16_t)xpti::trace_point_type_t::task_begin);
  EXPECT_NE(Check, true);

  Result = xptiRegisterCallback(
      ID, (uint16_t)xpti::trace_point_type_t::task_begin, fn_callback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Check =
      xptiCheckTraceEnabled(ID, (uint16_t)xpti::trace_point_type_t::task_begin);
  EXPECT_EQ(Check, true);

  Check =
      xptiCheckTraceEnabled(ID, (uint16_t)xpti::trace_point_type_t::task_end);
  EXPECT_NE(Check, true);
  Check = xptiCheckTraceEnabled(
      ID, (uint16_t)xpti::trace_point_type_t::mem_alloc_begin);
  EXPECT_NE(Check, true);
  Check = xptiCheckTraceEnabled(
      ID, (uint16_t)
              xpti::trace_point_type_t::offload_alloc_memory_object_construct);
  EXPECT_NE(Check, true);
  Check = xptiCheckTraceEnabled(
      ID, (uint16_t)
              xpti::trace_point_type_t::offload_alloc_memory_object_associate);
  EXPECT_NE(Check, true);
  Check = xptiCheckTraceEnabled(
      ID,
      (uint16_t)xpti::trace_point_type_t::offload_alloc_memory_object_destruct);
  EXPECT_NE(Check, true);
  Check = xptiCheckTraceEnabled(
      ID,
      (uint16_t)xpti::trace_point_type_t::offload_alloc_memory_object_release);
  EXPECT_NE(Check, true);
  Check = xptiCheckTraceEnabled(
      ID, (uint16_t)xpti::trace_point_type_t::offload_alloc_accessor);
  EXPECT_NE(Check, true);
  Check = xptiCheckTraceEnabled(
      ID, (uint16_t)xpti::trace_point_type_t::mem_release_begin);
  EXPECT_NE(Check, true);
  Check = xptiCheckTraceEnabled(
      ID, (uint16_t)xpti::trace_point_type_t::mem_release_end);
  EXPECT_NE(Check, true);
  // We expect to reset TraceEnabled() == false
  xptiForceSetTraceEnabled(false);
}

TEST_F(xptiApiTest, xptiCheckTraceEnabledBadInput) {
  uint64_t instance;
  xptiForceSetTraceEnabled(true);
  xpti::payload_t Payload("foo", "foo.cpp", 1, 0, (void *)13);

  auto Event = xptiMakeEvent("foo", &Payload, 0, (xpti::trace_activity_type_t)1,
                             &instance);
  EXPECT_NE(Event, nullptr);

  auto ID = xptiRegisterStream("foo");
  auto Result = xptiRegisterCallback(
      ID, (uint16_t)xpti::trace_point_type_t::function_begin, fn_callback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);

  auto Check = xptiCheckTraceEnabled(0, 0);
  EXPECT_EQ(Check, false);

  Check =
      xptiCheckTraceEnabled(35, (uint16_t)xpti::trace_point_type_t::task_begin);
  EXPECT_EQ(Check, false);
  // We expect to reset TraceEnabled() == false
  xptiForceSetTraceEnabled(false);
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
      StreamID,
      (uint16_t)xpti::trace_point_type_t::offload_alloc_memory_object_construct,
      nullptr, nullptr, 0, nullptr);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_INVALIDARG);

  Result = xptiNotifySubscribers(
      StreamID,
      (uint16_t)xpti::trace_point_type_t::offload_alloc_memory_object_associate,
      nullptr, nullptr, 0, nullptr);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_INVALIDARG);

  Result = xptiNotifySubscribers(
      StreamID,
      (uint16_t)xpti::trace_point_type_t::offload_alloc_memory_object_destruct,
      nullptr, nullptr, 0, nullptr);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_INVALIDARG);

  Result = xptiNotifySubscribers(
      StreamID,
      (uint16_t)xpti::trace_point_type_t::offload_alloc_memory_object_release,
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
  xptiForceSetTraceEnabled(false);
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
      StreamID,
      (uint16_t)xpti::trace_point_type_t::offload_alloc_memory_object_construct,
      fn_callback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      StreamID,
      (uint16_t)xpti::trace_point_type_t::offload_alloc_memory_object_associate,
      fn_callback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      StreamID,
      (uint16_t)xpti::trace_point_type_t::offload_alloc_memory_object_destruct,
      fn_callback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      StreamID,
      (uint16_t)xpti::trace_point_type_t::offload_alloc_memory_object_release,
      fn_callback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiRegisterCallback(
      StreamID, (uint16_t)xpti::trace_point_type_t::offload_alloc_accessor,
      fn_callback);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);

  xpti::offload_buffer_data_t UserBufferData{1, 5, "int", 4, 2, {3, 2, 0}};
  xpti::offload_image_data_t UserSampledImageData{1, 5, 4, {3, 2, 0},
                                                  4, 3, 2, 1};
  xpti::offload_image_data_t UserUnsampledImageData{
      1, 5, 4, {3, 2, 0}, 4, std::nullopt, std::nullopt, std::nullopt};
  xpti::offload_association_data_t BufferAssociationData{0x01020304,
                                                         0x05060708};
  xpti::offload_association_data_t SampledImageAssociationData{0x01020404,
                                                               0x05060808};
  xpti::offload_association_data_t UnsampledImageAssociationData{0x01020504,
                                                                 0x05060908};
  xpti::offload_association_data_t SampledImageHostAssociationData{0x01020604,
                                                                   0x05060818};
  xpti::offload_association_data_t UnsampledImageHostAssociationData{
      0x01020704, 0x05060918};
  xpti::offload_accessor_data_t UserAccessorData{0x01020304, 0x09000102, 1, 2};
  xpti::offload_image_accessor_data_t UserSampledImageAccessorData{
      0x01020404, 0x09000103, 1, std::nullopt, "uint4", 16};
  xpti::offload_image_accessor_data_t UserUnsampledImageAccessorData{
      0x01020504, 0x09000104, 1, 2, "uint4", 16};
  xpti::offload_image_accessor_data_t UserSampledImageHostAccessorData{
      0x01020604, 0x09000105, std::nullopt, std::nullopt, "uint4", 16};
  xpti::offload_image_accessor_data_t UserUnsampledImageHostAccessorData{
      0x01020704, 0x09000106, std::nullopt, 2, "uint4", 16};

  tmp = func_callback_update;
  Result = xptiNotifySubscribers(
      StreamID,
      (uint16_t)xpti::trace_point_type_t::offload_alloc_memory_object_construct,
      nullptr, (xpti::trace_event_data_t *)1, 0, &UserBufferData);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiNotifySubscribers(
      StreamID,
      (uint16_t)xpti::trace_point_type_t::offload_alloc_memory_object_associate,
      nullptr, (xpti::trace_event_data_t *)1, 0, &BufferAssociationData);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiNotifySubscribers(
      StreamID,
      (uint16_t)xpti::trace_point_type_t::offload_alloc_memory_object_release,
      nullptr, (xpti::trace_event_data_t *)1, 0, &BufferAssociationData);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiNotifySubscribers(
      StreamID,
      (uint16_t)xpti::trace_point_type_t::offload_alloc_memory_object_destruct,
      nullptr, (xpti::trace_event_data_t *)1, 0, &UserBufferData);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiNotifySubscribers(
      StreamID,
      (uint16_t)xpti::trace_point_type_t::offload_alloc_memory_object_construct,
      nullptr, (xpti::trace_event_data_t *)1, 0, &UserSampledImageData);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiNotifySubscribers(
      StreamID,
      (uint16_t)xpti::trace_point_type_t::offload_alloc_memory_object_associate,
      nullptr, (xpti::trace_event_data_t *)1, 0, &SampledImageAssociationData);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiNotifySubscribers(
      StreamID,
      (uint16_t)xpti::trace_point_type_t::offload_alloc_memory_object_release,
      nullptr, (xpti::trace_event_data_t *)1, 0, &SampledImageAssociationData);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiNotifySubscribers(
      StreamID,
      (uint16_t)xpti::trace_point_type_t::offload_alloc_memory_object_associate,
      nullptr, (xpti::trace_event_data_t *)1, 0,
      &SampledImageHostAssociationData);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiNotifySubscribers(
      StreamID,
      (uint16_t)xpti::trace_point_type_t::offload_alloc_memory_object_release,
      nullptr, (xpti::trace_event_data_t *)1, 0,
      &SampledImageHostAssociationData);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiNotifySubscribers(
      StreamID,
      (uint16_t)xpti::trace_point_type_t::offload_alloc_memory_object_destruct,
      nullptr, (xpti::trace_event_data_t *)1, 0, &UserSampledImageData);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiNotifySubscribers(
      StreamID,
      (uint16_t)xpti::trace_point_type_t::offload_alloc_memory_object_construct,
      nullptr, (xpti::trace_event_data_t *)1, 0, &UserUnsampledImageData);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiNotifySubscribers(
      StreamID,
      (uint16_t)xpti::trace_point_type_t::offload_alloc_memory_object_associate,
      nullptr, (xpti::trace_event_data_t *)1, 0,
      &UnsampledImageAssociationData);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiNotifySubscribers(
      StreamID,
      (uint16_t)xpti::trace_point_type_t::offload_alloc_memory_object_release,
      nullptr, (xpti::trace_event_data_t *)1, 0,
      &UnsampledImageAssociationData);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiNotifySubscribers(
      StreamID,
      (uint16_t)xpti::trace_point_type_t::offload_alloc_memory_object_associate,
      nullptr, (xpti::trace_event_data_t *)1, 0,
      &UnsampledImageHostAssociationData);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiNotifySubscribers(
      StreamID,
      (uint16_t)xpti::trace_point_type_t::offload_alloc_memory_object_release,
      nullptr, (xpti::trace_event_data_t *)1, 0,
      &UnsampledImageHostAssociationData);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiNotifySubscribers(
      StreamID,
      (uint16_t)xpti::trace_point_type_t::offload_alloc_memory_object_destruct,
      nullptr, (xpti::trace_event_data_t *)1, 0, &UserUnsampledImageData);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiNotifySubscribers(
      StreamID, (uint16_t)xpti::trace_point_type_t::offload_alloc_accessor,
      nullptr, (xpti::trace_event_data_t *)1, 0, &UserAccessorData);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiNotifySubscribers(
      StreamID, (uint16_t)xpti::trace_point_type_t::offload_alloc_accessor,
      nullptr, (xpti::trace_event_data_t *)1, 0, &UserSampledImageAccessorData);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiNotifySubscribers(
      StreamID, (uint16_t)xpti::trace_point_type_t::offload_alloc_accessor,
      nullptr, (xpti::trace_event_data_t *)1, 0,
      &UserUnsampledImageAccessorData);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiNotifySubscribers(
      StreamID, (uint16_t)xpti::trace_point_type_t::offload_alloc_accessor,
      nullptr, (xpti::trace_event_data_t *)1, 0,
      &UserSampledImageHostAccessorData);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiNotifySubscribers(
      StreamID, (uint16_t)xpti::trace_point_type_t::offload_alloc_accessor,
      nullptr, (xpti::trace_event_data_t *)1, 0,
      &UserUnsampledImageHostAccessorData);
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
  xptiForceSetTraceEnabled(false);
}

TEST_F(xptiApiTest, xptiAddMetadataBadInput) {
  uint64_t instance;
  xptiForceSetTraceEnabled(true);
  xpti::payload_t Payload("foo", "foo.cpp", 1, 0, (void *)13);

  auto Event = xptiMakeEvent("foo", &Payload, 0, (xpti::trace_activity_type_t)1,
                             &instance);
  EXPECT_NE(Event, nullptr);

  auto Result = xptiAddMetadata(nullptr, nullptr, 0);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_INVALIDARG);
  Result = xptiAddMetadata(Event, nullptr, 0);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_INVALIDARG);
  xptiReleaseEvent(Event);
  xptiForceSetTraceEnabled(false);
}

TEST_F(xptiApiTest, xptiAddMetadataGoodInput) {
  uint64_t instance;
  xptiForceSetTraceEnabled(true);
  xpti::payload_t Payload("foo", "foo.cpp", 1, 0, (void *)13);

  auto Event = xptiMakeEvent("foo", &Payload, 0, (xpti::trace_activity_type_t)1,
                             &instance);
  EXPECT_NE(Event, nullptr);

  xpti::object_id_t ID = xptiRegisterObject("bar", 3, 0);
  auto Result = xptiAddMetadata(Event, "foo", ID);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);
  Result = xptiAddMetadata(Event, "foo", ID);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_DUPLICATE);
  xptiReleaseEvent(Event);
  xptiForceSetTraceEnabled(false);
}

TEST_F(xptiApiTest, xptiQueryMetadata) {
  uint64_t instance;
  xptiForceSetTraceEnabled(true);
  xpti::payload_t Payload("fubar", "foobar.cpp", 100, 0, (void *)13);

  auto Event = xptiMakeEvent("foo", &Payload, 0, (xpti::trace_activity_type_t)1,
                             &instance);
  EXPECT_NE(Event, nullptr);

  auto md = xptiQueryMetadata(Event);
  EXPECT_NE(md, nullptr);

  xpti::object_id_t ID = xptiRegisterObject("bar1", 4, 0);
  auto Result = xptiAddMetadata(Event, "foo1", ID);
  EXPECT_EQ(Result, xpti::result_t::XPTI_RESULT_SUCCESS);

  char *ts;
  EXPECT_EQ(md->size(), 1u);
  auto MDID = (*md)[xptiRegisterString("foo1", &ts)];
  auto obj = xptiLookupObject(MDID);
  std::string str{obj.data, obj.size};
  EXPECT_EQ(str, "bar1");
  xptiForceSetTraceEnabled(false);
}
