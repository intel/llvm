//==--------------- USM.cpp --- dependency chain unit tests ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/event_impl.hpp>
#include <sycl/usm.hpp>

#include <helpers/PiMock.hpp>

#include <gtest/gtest.h>

namespace {
using namespace sycl;

struct {
  std::vector<event> Deps;
} TestContext;

// Dummy event values for bookkeeping
pi_event WAIT = nullptr;
pi_event MEMCPY = nullptr;
pi_event MEMSET = nullptr;

template <typename T> auto getVal(T obj) {
  return detail::getSyclObjImpl(obj)->getHandleRef();
}

pi_result redefinedEnqueueEventsWaitAfter(pi_queue, pi_uint32 NumDeps,
                                          const pi_event *Deps,
                                          pi_event *Event) {
  EXPECT_EQ(NumDeps, TestContext.Deps.size());
  for (size_t i = 0; i < NumDeps; ++i) {
    EXPECT_EQ(Deps[i], getVal(TestContext.Deps[i]));
  }
  WAIT = *Event;
  return PI_SUCCESS;
}

pi_result redefinedUSMEnqueueMemcpyAfter(pi_queue, pi_bool, void *,
                                         const void *, size_t, pi_uint32,
                                         const pi_event *, pi_event *Event) {
  // Set MEMCPY to the event produced by the original USMEnqueueMemcpy
  MEMCPY = *Event;
  return PI_SUCCESS;
}

pi_result redefinedUSMEnqueueMemsetAfter(pi_queue, void *, const void *, size_t,
                                         size_t, pi_uint32, const pi_event *,
                                         pi_event *Event) {
  // Set MEMSET to the event produced by the original USMEnqueueMemcpy
  MEMSET = *Event;
  return PI_SUCCESS;
}

// Check that zero-length USM memset/memcpy use piEnqueueEventsWait.
TEST(USM, NoOpPreservesDependencyChain) {
  sycl::unittest::PiMock Mock;
  sycl::platform Plt = Mock.getPlatform();
  Mock.redefineAfter<detail::PiApiKind::piEnqueueEventsWait>(
      redefinedEnqueueEventsWaitAfter);
  Mock.redefineAfter<detail::PiApiKind::piextUSMEnqueueMemcpy>(
      redefinedUSMEnqueueMemcpyAfter);
  Mock.redefineAfter<detail::PiApiKind::piextUSMEnqueueFill>(
      redefinedUSMEnqueueMemsetAfter);

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
