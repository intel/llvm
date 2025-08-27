//==------------------------- USMPrefetch.cpp ------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Test SYCL prefetch calls UR prefetch functions with the correct arguments.

#include <gtest/gtest.h>

#include <helpers/UrMock.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/properties/all_properties.hpp>
#include <sycl/queue.hpp>
#include <sycl/usm.hpp>

using namespace sycl;

namespace {

static ur_usm_migration_flags_t SubmittedPrefetchType =
    UR_USM_MIGRATION_FLAG_FORCE_UINT32;

inline ur_result_t replace_urUSMEnqueuePrefetch(void *pParams) {
  auto params = *static_cast<ur_enqueue_usm_prefetch_params_t *>(pParams);
  SubmittedPrefetchType = *params.pflags;
  return UR_RESULT_SUCCESS;
}

static constexpr size_t N = 1024;
class USMPrefetchTests : public ::testing::Test {
public:
  USMPrefetchTests()
      : Mock{}, Q{context(sycl::platform()), default_selector_v,
                  property::queue::in_order{}} {}

protected:
  void SetUp() override {
    SubmittedPrefetchType = UR_USM_MIGRATION_FLAG_FORCE_UINT32;
    Dst = malloc_shared<int>(N, Q);
  }

  unittest::UrMock<> Mock;
  queue Q;
  int *Dst;
};

TEST_F(USMPrefetchTests, QueuePrefetch) {
  mock::getCallbacks().set_replace_callback("urEnqueueUSMPrefetch",
                                            replace_urUSMEnqueuePrefetch);

  Q.prefetch(Dst, sizeof(int) * N);
  ASSERT_EQ(SubmittedPrefetchType, UR_USM_MIGRATION_FLAG_HOST_TO_DEVICE);

  free(Dst, Q);
}

TEST_F(USMPrefetchTests, HandlerPrefetch) {
  mock::getCallbacks().set_replace_callback("urEnqueueUSMPrefetch",
                                            replace_urUSMEnqueuePrefetch);

  Q.submit([&](handler &CGH) { CGH.prefetch(Dst, sizeof(int) * N); });
  ASSERT_EQ(SubmittedPrefetchType, UR_USM_MIGRATION_FLAG_HOST_TO_DEVICE);

  free(Dst, Q);
}

} // namespace
