//==--------------- USM.cpp --- dependency chain unit tests ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/usm.hpp>
#include <detail/event_impl.hpp>

#include <helpers/PiMock.hpp>

#include <gtest/gtest.h>

namespace {
using namespace sycl;

struct {
  std::vector<event> Deps;
} TestContext;

// Dummy event values for bookkeeping
pi_event WAIT = reinterpret_cast<pi_event>(1);
pi_event MEMCPY = reinterpret_cast<pi_event>(2);
pi_event MEMSET = reinterpret_cast<pi_event>(3);

template <typename T> auto getVal(T obj) {
  return detail::getSyclObjImpl(obj)->getHandleRef();
}

pi_result redefinedEnqueueEventsWait(pi_queue, pi_uint32 NumDeps,
                                     const pi_event *Deps, pi_event *Event) {
  EXPECT_EQ(NumDeps, TestContext.Deps.size());
  for (size_t i = 0; i < NumDeps; ++i) {
    EXPECT_EQ(Deps[i], getVal(TestContext.Deps[i]));
  }
  *Event = WAIT;
  return PI_SUCCESS;
}

pi_result redefinedUSMEnqueueMemcpy(pi_queue, pi_bool, void *, const void *,
                                    size_t, pi_uint32, const pi_event *,
                                    pi_event *Event) {
  *Event = MEMCPY;
  return PI_SUCCESS;
}

pi_result redefinedUSMEnqueueMemset(pi_queue, void *, pi_int32, size_t,
                                    pi_uint32, const pi_event *,
                                    pi_event *Event) {
  *Event = MEMSET;
  return PI_SUCCESS;
}

pi_result redefinedEventRelease(pi_event) { return PI_SUCCESS; }

bool preparePiMock(platform &Plt) {
  if (Plt.is_host()) {
    std::cout << "Not run on host - no PI events created in that case"
              << std::endl;
    return false;
  }

  unittest::PiMock Mock{Plt};
  Mock.redefine<detail::PiApiKind::piEnqueueEventsWait>(
      redefinedEnqueueEventsWait);
  Mock.redefine<detail::PiApiKind::piextUSMEnqueueMemcpy>(
      redefinedUSMEnqueueMemcpy);
  Mock.redefine<detail::PiApiKind::piextUSMEnqueueMemset>(
      redefinedUSMEnqueueMemset);
  Mock.redefine<detail::PiApiKind::piEventRelease>(redefinedEventRelease);
  return true;
}

// Check that zero-length USM memset/memcpy use piEnqueueEventsWait.
TEST(USM, NoOpPreservesDependencyChain) {
  platform Plt{default_selector()};
  if (!preparePiMock(Plt))
    return;

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
}

} // namespace
