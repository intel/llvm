//==--- KPCache.cpp --- KernelProgramCache for multiple devices unit test --==//
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
#include <helpers/MockKernelInfo.hpp>
#include <helpers/UrImage.hpp>
#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

#include <iostream>

using namespace sycl;

class MultipleDevsCacheTestKernel;

MOCK_INTEGRATION_HEADER(MultipleDevsCacheTestKernel)

static sycl::unittest::UrImage Img =
    sycl::unittest::generateDefaultImage({"MultipleDevsCacheTestKernel"});
static sycl::unittest::UrImageArray<1> ImgArray{&Img};

static ur_result_t redefinedDeviceGetAfter(void *pParams) {
  auto params = *static_cast<ur_device_get_params_t *>(pParams);
  if (*params.ppNumDevices) {
    **params.ppNumDevices = static_cast<uint32_t>(2);
    return UR_RESULT_SUCCESS;
  }

  if (*params.pNumEntries == 2 && *params.pphDevices) {
    (*params.pphDevices)[0] = reinterpret_cast<ur_device_handle_t>(1111);
    (*params.pphDevices)[1] = reinterpret_cast<ur_device_handle_t>(2222);
  }
  return UR_RESULT_SUCCESS;
}

static ur_result_t redefinedDeviceGetInfo(void *pParams) {
  auto params = *static_cast<ur_device_get_info_params_t *>(pParams);
  if (*params.ppropName == UR_DEVICE_INFO_TYPE) {
    auto *Result = reinterpret_cast<ur_device_type_t *>(*params.ppPropValue);
    *Result = UR_DEVICE_TYPE_GPU;
  }
  if (*params.ppropName == UR_DEVICE_INFO_COMPILER_AVAILABLE) {
    auto *Result = reinterpret_cast<ur_bool_t *>(*params.ppPropValue);
    *Result = true;
  }

  // This mock device has no sub-devices
  if (*params.ppropName == UR_DEVICE_INFO_SUPPORTED_PARTITIONS) {
    if (*params.ppPropSizeRet) {
      **params.ppPropSizeRet = 0;
    }
  }
  if (*params.ppropName == UR_DEVICE_INFO_PARTITION_AFFINITY_DOMAIN) {
    assert(*params.ppropSize == sizeof(ur_device_affinity_domain_flags_t));
    if (*params.ppPropValue) {
      *static_cast<ur_device_affinity_domain_flags_t *>(*params.ppPropValue) =
          0;
    }
  }
  return UR_RESULT_SUCCESS;
}

static int RetainCounter = 0;
static ur_result_t redefinedProgramRetain(void *) {
  ++RetainCounter;
  return UR_RESULT_SUCCESS;
}

static int KernelReleaseCounter = 0;
static ur_result_t redefinedKernelRelease(void *) {
  ++KernelReleaseCounter;
  return UR_RESULT_SUCCESS;
}

class MultipleDeviceCacheTest : public ::testing::Test {
public:
  MultipleDeviceCacheTest() : Mock{}, Plt{sycl::platform()} {}

protected:
  void SetUp() override {
    mock::getCallbacks().set_after_callback("urDeviceGet",
                                            &redefinedDeviceGetAfter);
    mock::getCallbacks().set_before_callback("urDeviceGetInfo",
                                             &redefinedDeviceGetInfo);
    mock::getCallbacks().set_before_callback("urProgramRetain",
                                             &redefinedProgramRetain);
    mock::getCallbacks().set_before_callback("urKernelRelease",
                                             &redefinedKernelRelease);
  }

protected:
  unittest::UrMock<> Mock;
  platform Plt;
};

// Test that program is retained for each device and each kernel is released
// once
TEST_F(MultipleDeviceCacheTest, ProgramRetain) {
  {
    std::vector<sycl::device> Devices = Plt.get_devices(info::device_type::gpu);
    sycl::context Context(Devices);
    sycl::queue Queue(Context, Devices[0]);
    assert(Devices.size() == 2 && Context.get_devices().size() == 2);

    auto KernelID = sycl::get_kernel_id<MultipleDevsCacheTestKernel>();
    auto Bundle = sycl::get_kernel_bundle<sycl::bundle_state::input>(
        Queue.get_context(), {KernelID});
    assert(Bundle.get_devices().size() == 2);

    Queue.submit([&](sycl::handler &cgh) {
      cgh.single_task<MultipleDevsCacheTestKernel>([]() {});
    });

    auto BundleObject = sycl::build(Bundle, Bundle.get_devices());
    auto Kernel = BundleObject.get_kernel(KernelID);

    // Because of emulating 2 devices program is retained for each one in
    // build(). It is also depends on number of device images. This test has one
    // image, but other tests can create other images. Additional variable is
    // added to control count of urProgramRetain calls
    auto BundleImpl = getSyclObjImpl(Bundle);

    // Bundle should only contain a single image, specifically the one with
    // MultipleDevsCacheTestKernel.
    EXPECT_EQ(BundleImpl->size(), size_t{1});

    int NumRetains = 1 + BundleImpl->size() * 2;
    EXPECT_EQ(RetainCounter, NumRetains)
        << "Expect " << NumRetains << " piProgramRetain calls";

    auto CtxImpl = detail::getSyclObjImpl(Context);
    detail::KernelProgramCache::KernelCacheT &KernelCache =
        CtxImpl->getKernelProgramCache().acquireKernelsPerProgramCache().get();

    EXPECT_EQ(KernelCache.size(), (size_t)1)
        << "Expect 1 program in kernel cache";
    for (auto &KernelProgIt : KernelCache)
      EXPECT_EQ(KernelProgIt.second.size(), (size_t)1)
          << "Expect 1 kernel cache";
  }
  // The kernel creating is called in handler::single_task().
  // kernel_bundle::get_kernel() creates a kernel and shares it with created
  // programs. Also the kernel is retained in kernel_bundle::get_kernel(). A
  // kernel is removed from cache if urKernelRelease was called for it, so it
  // will not be removed twice for the other programs. As a result we must
  // expect 3 urKernelRelease calls.
  EXPECT_EQ(KernelReleaseCounter, 3) << "Expect 3 piKernelRelease calls";
}
