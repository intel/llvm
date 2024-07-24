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
#include <helpers/PiImage.hpp>
#include <helpers/PiMock.hpp>

#include <gtest/gtest.h>

#include <iostream>

using namespace sycl;

class MultipleDevsCacheTestKernel;

MOCK_INTEGRATION_HEADER(MultipleDevsCacheTestKernel)

static sycl::unittest::PiImage Img =
    sycl::unittest::generateDefaultImage({"MultipleDevsCacheTestKernel"});
static sycl::unittest::PiImageArray<1> ImgArray{&Img};

static pi_result redefinedDevicesGetAfter(pi_platform platform,
                                          pi_device_type device_type,
                                          pi_uint32 num_entries,
                                          pi_device *devices,
                                          pi_uint32 *num_devices) {
  if (num_devices) {
    *num_devices = static_cast<pi_uint32>(2);
    return PI_SUCCESS;
  }

  if (num_entries == 2 && devices) {
    devices[0] = reinterpret_cast<pi_device>(1111);
    devices[1] = reinterpret_cast<pi_device>(2222);
  }
  return PI_SUCCESS;
}

static pi_result redefinedDeviceGetInfo(pi_device device,
                                        pi_device_info param_name,
                                        size_t param_value_size,
                                        void *param_value,
                                        size_t *param_value_size_ret) {
  if (param_name == PI_DEVICE_INFO_TYPE) {
    auto *Result = reinterpret_cast<_pi_device_type *>(param_value);
    *Result = PI_DEVICE_TYPE_GPU;
  }
  if (param_name == PI_DEVICE_INFO_COMPILER_AVAILABLE) {
    auto *Result = reinterpret_cast<pi_bool *>(param_value);
    *Result = true;
  }

  // This mock device has no sub-devices
  if (param_name == PI_DEVICE_INFO_PARTITION_PROPERTIES) {
    if (param_value_size_ret) {
      *param_value_size_ret = 0;
    }
  }
  if (param_name == PI_DEVICE_INFO_PARTITION_AFFINITY_DOMAIN) {
    assert(param_value_size == sizeof(pi_device_affinity_domain));
    if (param_value) {
      *static_cast<pi_device_affinity_domain *>(param_value) = 0;
    }
  }
  return PI_SUCCESS;
}

static int RetainCounter = 0;
static pi_result redefinedProgramRetain(pi_program program) {
  ++RetainCounter;
  return PI_SUCCESS;
}

static int KernelReleaseCounter = 0;
static pi_result redefinedKernelRelease(pi_kernel kernel) {
  ++KernelReleaseCounter;
  return PI_SUCCESS;
}

class MultipleDeviceCacheTest : public ::testing::Test {
public:
  MultipleDeviceCacheTest() : Mock{}, Plt{Mock.getPlatform()} {}

protected:
  void SetUp() override {
    Mock.redefineAfter<detail::PiApiKind::piDevicesGet>(
        redefinedDevicesGetAfter);
    Mock.redefineBefore<detail::PiApiKind::piDeviceGetInfo>(
        redefinedDeviceGetInfo);
    Mock.redefineBefore<detail::PiApiKind::piProgramRetain>(
        redefinedProgramRetain);
    Mock.redefineBefore<detail::PiApiKind::piKernelRelease>(
        redefinedKernelRelease);
  }

protected:
  unittest::PiMock Mock;
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
    // added to control count of piProgramRetain calls
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
  // kernel is removed from cache if piKernelRelease was called for it, so it
  // will not be removed twice for the other programs. As a result we must
  // expect 3 piKernelRelease calls.
  EXPECT_EQ(KernelReleaseCounter, 3) << "Expect 3 piKernelRelease calls";
}
