//==------------------- ExtOneapiBarrierOpt.cpp ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>
#include <helpers/ScopedEnvVar.hpp>
#include <helpers/UrMock.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;

inline thread_local uint32_t NumEventsInWaitList;

static ur_result_t redefinedEnqueueEventsWaitWithBarrierExt(void *pParams) {
  auto params =
      *static_cast<ur_enqueue_events_wait_with_barrier_ext_params_t *>(pParams);
  NumEventsInWaitList = *(params.pnumEventsInWaitList);
  return UR_RESULT_SUCCESS;
}

class ExtOneapiBarrierOptTest : public ::testing::Test {
public:
  ExtOneapiBarrierOptTest() : Mock{} {}

protected:
  void SetUp() override { NumEventsInWaitList = 0; }

protected:
  sycl::unittest::UrMock<> Mock;
};

// Check that ext_oneapi_submit_barrier works fine in the scenarios
// when provided waitlist consists of only empty events.
// Tets for https://github.com/intel/llvm/pull/12951
TEST_F(ExtOneapiBarrierOptTest, EmptyEventTest) {
  sycl::queue q1{{sycl::property::queue::in_order()}};

  mock::getCallbacks().set_after_callback(
      "urEnqueueEventsWaitWithBarrierExt",
      &redefinedEnqueueEventsWaitWithBarrierExt);

  NumEventsInWaitList = 100;
  q1.ext_oneapi_submit_barrier();
  ASSERT_EQ(0u, NumEventsInWaitList);

  // ext_oneapi_submit_barrier should ignore empty, default constructed events.
  sycl::event E1{};
  NumEventsInWaitList = 100;
  q1.ext_oneapi_submit_barrier({E1});
  ASSERT_EQ(0u, NumEventsInWaitList);

  sycl::event E2{};
  NumEventsInWaitList = 100;
  q1.ext_oneapi_submit_barrier({E1, E2});
  ASSERT_EQ(0u, NumEventsInWaitList);
}
