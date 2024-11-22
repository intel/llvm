//==----- InMemCacheEviction.cpp --- In-memory cache eviction tests -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file contains tests covering eviction in in-memory program cache.

#define SYCL2020_DISABLE_DEPRECATION_WARNINGS

#include "../thread_safety/ThreadUtils.h"
#include "detail/context_impl.hpp"
#include "detail/kernel_program_cache.hpp"
#include <detail/config.hpp>
#include <helpers/MockDeviceImage.hpp>
#include <helpers/MockKernelInfo.hpp>
#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

#include <iostream>

using namespace sycl;

class Kernel1;
class Kernel2;
class Kernel3;

MOCK_INTEGRATION_HEADER(Kernel1)
MOCK_INTEGRATION_HEADER(Kernel2)
MOCK_INTEGRATION_HEADER(Kernel3)

static sycl::unittest::MockDeviceImage Img[] = {
    sycl::unittest::generateDefaultImage({"Kernel1"}),
    sycl::unittest::generateDefaultImage({"Kernel2"}),
    sycl::unittest::generateDefaultImage({"Kernel3"})};

static sycl::unittest::MockDeviceImageArray<3> ImgArray{Img};

// Number of times urProgramCreateWithIL is called. This is used to check
// if the program is created or fetched from the cache.
static int NumProgramBuild = 0;

constexpr int ProgramSize = 10000;

static ur_result_t redefinedProgramCreateWithIL(void *) {
  ++NumProgramBuild;
  return UR_RESULT_SUCCESS;
}

static ur_result_t redefinedProgramGetInfoAfter(void *pParams) {
  auto params = *static_cast<ur_program_get_info_params_t *>(pParams);
  if (*params.ppropName == UR_PROGRAM_INFO_NUM_DEVICES) {
    auto value = reinterpret_cast<unsigned int *>(*params.ppPropValue);
    *value = 1;
  }

  if (*params.ppropName == UR_PROGRAM_INFO_BINARY_SIZES) {
    auto value = reinterpret_cast<size_t *>(*params.ppPropValue);
    value[0] = ProgramSize;
  }

  if (*params.ppropName == UR_PROGRAM_INFO_BINARIES) {
    auto value = reinterpret_cast<unsigned char **>(*params.ppPropValue);
    value[0] = 0;
  }

  return UR_RESULT_SUCCESS;
}

// Function to set SYCL_IN_MEM_CACHE_EVICTION_THRESHOLD.
static void setCacheEvictionEnv(const char *value) {
#ifdef _WIN32
  _putenv_s("SYCL_IN_MEM_CACHE_EVICTION_THRESHOLD", value);
#else
  if (value)
    setenv("SYCL_IN_MEM_CACHE_EVICTION_THRESHOLD", value, 1);
  else
    (void)unsetenv("SYCL_IN_MEM_CACHE_EVICTION_THRESHOLD");
#endif

  sycl::detail::readConfig(true);
  sycl::detail::SYCLConfig<
      sycl::detail::SYCL_IN_MEM_CACHE_EVICTION_THRESHOLD>::reset();
}

// Function to check number of entries in the cache and eviction list.
static inline void
CheckNumberOfEntriesInCacheAndEvictionList(detail::context_impl &CtxImpl,
                                           size_t ExpectedNumEntries) {
  auto &KPCache = CtxImpl.getKernelProgramCache();
  EXPECT_EQ(KPCache.acquireCachedPrograms().get().size(), ExpectedNumEntries)
      << "Unexpected number of entries in the cache";
  auto EvcList = KPCache.acquireEvictionList();
  EXPECT_EQ(EvcList.get().size(), ExpectedNumEntries)
      << "Unexpected number of entries in the eviction list";
}

class InMemCacheEvictionTests : public ::testing::Test {
protected:
  void TearDown() override { setCacheEvictionEnv(""); }
};

TEST(InMemCacheEvictionTests, TestBasicEvictionAndLRU) {
  NumProgramBuild = 0;
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_before_callback("urProgramCreateWithIL",
                                           &redefinedProgramCreateWithIL);
  mock::getCallbacks().set_after_callback("urProgramGetInfo",
                                          &redefinedProgramGetInfoAfter);

  sycl::platform Plt{sycl::platform()};
  sycl::context Ctx{Plt};
  auto CtxImpl = detail::getSyclObjImpl(Ctx);
  queue q(Ctx, default_selector_v);

  // One program is of 10000 bytes, so 20005 eviction threshold can
  // accommodate two programs.
  setCacheEvictionEnv("20005");

  // Cache is empty, so one urProgramCreateWithIL call.
  q.single_task<class Kernel1>([] {});
  EXPECT_EQ(NumProgramBuild, 1);
  CheckNumberOfEntriesInCacheAndEvictionList(*CtxImpl, 1);

  q.single_task<class Kernel2>([] {});
  EXPECT_EQ(NumProgramBuild, 2);
  CheckNumberOfEntriesInCacheAndEvictionList(*CtxImpl, 2);

  // Move first program to end of eviction list.
  q.single_task<class Kernel1>([] {});
  EXPECT_EQ(NumProgramBuild, 2);

  // Calling Kernel3, Kernel2, and Kernel1 in a cyclic manner to
  // verify LRU's working.

  // Kernel2's program should have been evicted.
  q.single_task<class Kernel3>([] {});
  EXPECT_EQ(NumProgramBuild, 3);
  CheckNumberOfEntriesInCacheAndEvictionList(*CtxImpl, 2);

  // Calling Kernel2 again should trigger urProgramCreateWithIL and
  // should evict Kernel1's program.
  q.single_task<class Kernel2>([] {});
  EXPECT_EQ(NumProgramBuild, 3);
  CheckNumberOfEntriesInCacheAndEvictionList(*CtxImpl, 2);

  // Calling Kernel1 again should trigger urProgramCreateWithIL and
  // should evict Kernel3's program.
  q.single_task<class Kernel1>([] {});
  EXPECT_EQ(NumProgramBuild, 4);
  CheckNumberOfEntriesInCacheAndEvictionList(*CtxImpl, 2);
}

// Test to verify eviction using concurrent kernel invocation.
TEST(InMemCacheEvictionTests, TestConcurrentEvictionSameQueue) {
  NumProgramBuild = 0;
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_before_callback("urProgramCreateWithIL",
                                           &redefinedProgramCreateWithIL);
  mock::getCallbacks().set_after_callback("urProgramGetInfo",
                                          &redefinedProgramGetInfoAfter);

  sycl::platform Plt{sycl::platform()};
  context Ctx{Plt};
  auto CtxImpl = detail::getSyclObjImpl(Ctx);
  queue q(Ctx, default_selector_v);

  // One program is of 10000 bytes, so 20005 eviction threshold can
  // accommodate two programs.
  setCacheEvictionEnv("20005");

  constexpr size_t ThreadCount = 200;
  Barrier barrier(ThreadCount);
  {
    auto ConcurrentInvokeKernels = [&](std::size_t threadId) {
      barrier.wait();
      q.single_task<class Kernel1>([] {});
      q.single_task<class Kernel2>([] {});
      q.single_task<class Kernel3>([] {});
    };

    ThreadPool MPool(ThreadCount, ConcurrentInvokeKernels);
  }
  q.wait_and_throw();

  CheckNumberOfEntriesInCacheAndEvictionList(*CtxImpl, 2);
}

// Test to verify eviction using concurrent kernel invocation when
// cache size is very less so as to trigger immediate eviction.
TEST(InMemCacheEvictionTests, TestConcurrentEvictionSmallCache) {
  NumProgramBuild = 0;
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_before_callback("urProgramCreateWithIL",
                                           &redefinedProgramCreateWithIL);
  mock::getCallbacks().set_after_callback("urProgramGetInfo",
                                          &redefinedProgramGetInfoAfter);

  context Ctx{platform()};
  auto CtxImpl = detail::getSyclObjImpl(Ctx);
  queue q(Ctx, default_selector_v);

  // One program is of 10000 bytes, so 100 eviction threshold will
  // trigger immediate eviction.
  setCacheEvictionEnv("100");

  // Fetch the same kernel concurrently from multiple threads.
  // This should cause some threads to insert a program and other
  // threads to evict the same program.
  constexpr size_t ThreadCount = 300;
  Barrier barrier(ThreadCount);
  {
    auto ConcurrentInvokeKernels = [&](std::size_t threadId) {
      barrier.wait();
      q.single_task<class Kernel1>([] {});
    };

    ThreadPool MPool(ThreadCount, ConcurrentInvokeKernels);
  }
  q.wait_and_throw();

  CheckNumberOfEntriesInCacheAndEvictionList(*CtxImpl, 0);
}
