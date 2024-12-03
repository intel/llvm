//==--- OutOfResources.cpp --- OutOfResources unit test --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define SYCL2020_DISABLE_DEPRECATION_WARNINGS

#include "detail/context_impl.hpp"
#include "detail/kernel_bundle_impl.hpp"
#include "detail/kernel_program_cache.hpp"
#include <helpers/MockDeviceImage.hpp>
#include <helpers/MockKernelInfo.hpp>
#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

#include <iostream>

using namespace sycl;

class OutOfResourcesKernel1;
class OutOfResourcesKernel2;

MOCK_INTEGRATION_HEADER(OutOfResourcesKernel1)
MOCK_INTEGRATION_HEADER(OutOfResourcesKernel2)

static sycl::unittest::MockDeviceImage Img[2] = {
    sycl::unittest::generateDefaultImage({"OutOfResourcesKernel1"}),
    sycl::unittest::generateDefaultImage({"OutOfResourcesKernel2"})};

static sycl::unittest::MockDeviceImageArray<2> ImgArray{Img};

static int nProgramCreate = 0;
static volatile bool outOfResourcesToggle = false;
static volatile ur_result_t ErrorCode = UR_RESULT_SUCCESS;

static ur_result_t redefinedProgramCreateWithIL(void *) {
  ++nProgramCreate;
  if (outOfResourcesToggle) {
    outOfResourcesToggle = false;
    return ErrorCode;
  }
  return UR_RESULT_SUCCESS;
}

// Parameterized test fixture
class OutOfResourcesTestSuite : public ::testing::TestWithParam<ur_result_t> {};

TEST_P(OutOfResourcesTestSuite, urProgramCreate) {
  nProgramCreate = 0;
  sycl::unittest::UrMock<> Mock;
  ErrorCode = GetParam();
  mock::getCallbacks().set_before_callback("urProgramCreateWithIL",
                                           &redefinedProgramCreateWithIL);

  sycl::platform Plt{sycl::platform()};
  sycl::context Ctx{Plt};
  auto CtxImpl = detail::getSyclObjImpl(Ctx);
  queue q(Ctx, default_selector_v);

  int runningTotal = 0;
  // Cache is empty, so one urProgramCreateWithIL call.
  q.single_task<class OutOfResourcesKernel1>([] {});
  EXPECT_EQ(nProgramCreate, runningTotal += 1);

  // Now, we make the next urProgramCreateWithIL call fail with
  // UR_RESULT_ERROR_OUT_OF_RESOURCES. The caching mechanism should catch this,
  // clear the cache, and retry the urProgramCreateWithIL.
  outOfResourcesToggle = true;
  q.single_task<class OutOfResourcesKernel2>([] {});
  EXPECT_FALSE(outOfResourcesToggle);
  EXPECT_EQ(nProgramCreate, runningTotal += 2);
  {
    detail::KernelProgramCache::ProgramCache &Cache =
        CtxImpl->getKernelProgramCache().acquireCachedPrograms().get();
    EXPECT_EQ(Cache.size(), 1U) << "Expected 1 program in the cache";
  }

  // The next urProgramCreateWithIL call will fail with
  // UR_RESULT_ERROR_OUT_OF_RESOURCES. But OutOfResourcesKernel2 is in
  // the cache, so we expect no new urProgramCreateWithIL calls.
  outOfResourcesToggle = true;
  q.single_task<class OutOfResourcesKernel2>([] {});
  EXPECT_TRUE(outOfResourcesToggle);
  EXPECT_EQ(nProgramCreate, runningTotal);

  // OutOfResourcesKernel1 is not in the cache, so we have to
  // build it. From what we set before, this call will fail,
  // the cache will clear out, and will try again.
  q.single_task<class OutOfResourcesKernel1>([] {});
  EXPECT_FALSE(outOfResourcesToggle);
  EXPECT_EQ(nProgramCreate, runningTotal += 2);
  {
    detail::KernelProgramCache::ProgramCache &Cache =
        CtxImpl->getKernelProgramCache().acquireCachedPrograms().get();
    EXPECT_EQ(Cache.size(), 1U) << "Expected 1 program in the cache";
  }

  // Finally, OutOfResourcesKernel1 will be in the cache, but
  // OutOfResourceKenel2 will not, so one more urProgramCreateWithIL.
  // Toggle is not set, so this should succeed.
  q.single_task<class OutOfResourcesKernel1>([] {});
  q.single_task<class OutOfResourcesKernel2>([] {});
  EXPECT_EQ(nProgramCreate, runningTotal += 1);
  {
    detail::KernelProgramCache::ProgramCache &Cache =
        CtxImpl->getKernelProgramCache().acquireCachedPrograms().get();
    EXPECT_EQ(Cache.size(), 2U) << "Expected 2 program in the cache";
  }
}

static int nProgramLink = 0;

static ur_result_t redefinedProgramLink(void *) {
  ++nProgramLink;
  if (outOfResourcesToggle) {
    outOfResourcesToggle = false;
    return ErrorCode;
  }
  return UR_RESULT_SUCCESS;
}

TEST_P(OutOfResourcesTestSuite, urProgramLink) {
  nProgramLink = 0;
  sycl::unittest::UrMock<> Mock;
  ErrorCode = GetParam();
  mock::getCallbacks().set_before_callback("urProgramLinkExp",
                                           &redefinedProgramLink);

  sycl::platform Plt{sycl::platform()};
  sycl::context Ctx{Plt};
  auto CtxImpl = detail::getSyclObjImpl(Ctx);
  queue q(Ctx, default_selector_v);
  // Put some programs in the cache
  q.single_task<class OutOfResourcesKernel1>([] {});
  q.single_task<class OutOfResourcesKernel2>([] {});
  {
    detail::KernelProgramCache::ProgramCache &Cache =
        CtxImpl->getKernelProgramCache().acquireCachedPrograms().get();
    EXPECT_EQ(Cache.size(), 2U) << "Expect 2 programs in the cache";
  }

  auto b1 = sycl::get_kernel_bundle<OutOfResourcesKernel1,
                                    sycl::bundle_state::object>(Ctx);
  auto b2 = sycl::get_kernel_bundle<OutOfResourcesKernel2,
                                    sycl::bundle_state::object>(Ctx);
  outOfResourcesToggle = true;
  EXPECT_EQ(nProgramLink, 0);
  auto b3 = sycl::link({b1, b2});
  EXPECT_FALSE(outOfResourcesToggle);
  // one restart due to out of resources, one link per each of b1 and b2.
  EXPECT_EQ(nProgramLink, 3);
  // no programs should be in the cache due to out of resources.
  {
    detail::KernelProgramCache::ProgramCache &Cache =
        CtxImpl->getKernelProgramCache().acquireCachedPrograms().get();
    EXPECT_EQ(Cache.size(), 0u) << "Expect no programs in the cache";
  }
}

INSTANTIATE_TEST_SUITE_P(
    OutOfResourcesParameterizedRun, OutOfResourcesTestSuite,
    ::testing::Values(UR_RESULT_ERROR_OUT_OF_RESOURCES,
                      UR_RESULT_ERROR_OUT_OF_HOST_MEMORY,
                      UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY));
