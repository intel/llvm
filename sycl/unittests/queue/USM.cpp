//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/event_impl.hpp>
#include <sycl/usm.hpp>

#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

namespace {
using namespace sycl;

struct {
  std::vector<event> Deps;
} TestContext;

// Dummy event values for bookkeeping
ur_event_handle_t WAIT = nullptr;
ur_event_handle_t MEMCPY = nullptr;
ur_event_handle_t MEMSET = nullptr;

template <typename T> auto getVal(T obj) {
  return detail::getSyclObjImpl(obj)->getHandle();
}

ur_result_t redefinedEnqueueEventsWaitAfter(void *pParams) {
  auto params = *static_cast<ur_enqueue_events_wait_params_t *>(pParams);
  EXPECT_EQ(*params.pnumEventsInWaitList, TestContext.Deps.size());
  for (size_t i = 0; i < *params.pnumEventsInWaitList; ++i) {
    EXPECT_EQ((*params.pphEventWaitList)[i], getVal(TestContext.Deps[i]));
  }
  WAIT = **params.pphEvent;
  return UR_RESULT_SUCCESS;
}

ur_result_t redefinedUSMEnqueueMemcpyAfter(void *pParams) {
  auto params = *static_cast<ur_enqueue_usm_memcpy_params_t *>(pParams);
  // Set MEMCPY to the event produced by the original USMEnqueueMemcpy
  MEMCPY = **params.pphEvent;
  return UR_RESULT_SUCCESS;
}

ur_result_t redefinedUSMEnqueueMemFillAfter(void *pParams) {
  auto params = *static_cast<ur_enqueue_usm_fill_params_t *>(pParams);
  // Set MEMSET to the event produced by the original USMEnqueueMemcpy
  MEMSET = **params.pphEvent;
  return UR_RESULT_SUCCESS;
}

// Check that zero-length USM memset/memcpy use urEnqueueEventsWait.
TEST(USM, NoOpPreservesDependencyChain) {
  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();
  mock::getCallbacks().set_after_callback("urEnqueueEventsWait",
                                          &redefinedEnqueueEventsWaitAfter);
  mock::getCallbacks().set_after_callback("urEnqueueUSMMemcpy",
                                          &redefinedUSMEnqueueMemcpyAfter);
  mock::getCallbacks().set_after_callback("urEnqueueUSMFill",
                                          &redefinedUSMEnqueueMemFillAfter);

  context Ctx{Plt.get_devices()[0]};
  queue Q{Ctx, default_selector()};

  uint8_t *Src = malloc_host<uint8_t>(1, Q);
  uint8_t *Dst = malloc_host<uint8_t>(1, Q);

  event E1 = Q.memset(Src, 1, 1);
  ASSERT_EQ(getVal(E1), MEMSET);

  TestContext.Deps = {E1};
  event E2 = Q.memset(Dst, 2, 0, TestContext.Deps);
  ASSERT_EQ(getVal(E2), WAIT);

  TestContext.Deps = {E1, E2};
  event E3 = Q.memcpy(Dst, Src, 1, TestContext.Deps);
  ASSERT_EQ(getVal(E3), MEMCPY);

  TestContext.Deps = {E1, E2, E3};
  event E4 = Q.memcpy(Dst, Src, 0, TestContext.Deps);
  ASSERT_EQ(getVal(E4), WAIT);

  free(Src, Q);
  free(Dst, Q);
  TestContext.Deps.clear();
}
} // namespace
