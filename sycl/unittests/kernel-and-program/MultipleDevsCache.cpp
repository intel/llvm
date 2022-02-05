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
#include <helpers/PiImage.hpp>
#include <helpers/TestKernel.hpp>
#include <helpers/sycl_test.hpp>

#include <gtest/gtest.h>

#include <iostream>

using namespace sycl;

static pi_result redefinedContextCreate(
    const pi_context_properties *properties, pi_uint32 num_devices,
    const pi_device *devices,
    void (*pfn_notify)(const char *errinfo, const void *private_info, size_t cb,
                       void *user_data),
    void *user_data, pi_context *ret_context) {
  *ret_context = reinterpret_cast<pi_context>(123);
  return PI_SUCCESS;
}

static pi_result redefinedContextRelease(pi_context context) {
  return PI_SUCCESS;
}

static pi_result redefinedDevicesGet(pi_platform platform,
                                     pi_device_type device_type,
                                     pi_uint32 num_entries, pi_device *devices,
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
  return PI_SUCCESS;
}

static pi_result redefinedDeviceRetain(pi_device device) { return PI_SUCCESS; }

static pi_result redefinedDeviceRelease(pi_device device) { return PI_SUCCESS; }

static pi_result redefinedQueueCreate(pi_context context, pi_device device,
                                      pi_queue_properties properties,
                                      pi_queue *queue) {
  *queue = reinterpret_cast<pi_queue>(1234);
  return PI_SUCCESS;
}

static pi_result redefinedQueueRelease(pi_queue command_queue) {
  return PI_SUCCESS;
}

static size_t ProgramNum = 12345;
static pi_result redefinedProgramCreate(pi_context context, const void *il,
                                        size_t length,
                                        pi_program *res_program) {
  size_t CurrentProgram = ProgramNum;
  *res_program = reinterpret_cast<pi_program>(CurrentProgram);
  ++ProgramNum;
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

class MultipleDeviceCacheTest
    : public unittest::SYCLUnitTest<MultipleDeviceCacheTest> {
protected:
  void SetUp() override {
    unittest::SYCLUnitTest<MultipleDeviceCacheTest>::SetUp();
    Plt = sycl::platform(default_selector());

    if (Plt.is_host() || Plt.get_backend() != backend::opencl) {
      return;
    }

    using namespace sycl::unittest;

    redefine<detail::PiApiKind::piDevicesGet>(redefinedDevicesGet);
    redefine<detail::PiApiKind::piDeviceGetInfo>(redefinedDeviceGetInfo);
    redefine<detail::PiApiKind::piDeviceRetain>(redefinedDeviceRetain);
    redefine<detail::PiApiKind::piDeviceRelease>(redefinedDeviceRelease);
    redefine<detail::PiApiKind::piContextCreate>(redefinedContextCreate);
    redefine<detail::PiApiKind::piContextRelease>(redefinedContextRelease);
    redefine<detail::PiApiKind::piQueueCreate>(redefinedQueueCreate);
    redefine<detail::PiApiKind::piQueueRelease>(redefinedQueueRelease);
    redefine<detail::PiApiKind::piProgramRetain>(redefinedProgramRetain);
    redefine<detail::PiApiKind::piProgramCreate>(redefinedProgramCreate);
    redefine<detail::PiApiKind::piKernelRelease>(redefinedKernelRelease);
  }

protected:
  platform Plt;
};

// Test that program is retained for each device and each kernel is released
// once
TEST_F(MultipleDeviceCacheTest, ProgramRetain) {
  if (Plt.is_host() || Plt.get_backend() != backend::opencl) {
    return;
  }
  {
    std::vector<sycl::device> Devices = Plt.get_devices(info::device_type::gpu);
    sycl::context Context(Devices);
    sycl::queue Queue(Context, Devices[0]);
    assert(Devices.size() == 2);

    auto Bundle = cl::sycl::get_kernel_bundle<sycl::bundle_state::input>(
        Queue.get_context());
    Queue.submit(
        [&](cl::sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });

    auto BundleObject = cl::sycl::build(Bundle, Bundle.get_devices());
    auto KernelID = cl::sycl::get_kernel_id<TestKernel>();
    auto Kernel = BundleObject.get_kernel(KernelID);

    // Because of emulating 2 devices program is retained for each one in
    // build(). It is also depends on number of device images. This test has one
    // image, but other tests can create other images. Additional variable is
    // added to control count of piProgramRetain calls
    auto BundleImpl = getSyclObjImpl(Bundle);
    int NumRetains = BundleImpl->size() * 2;

    EXPECT_EQ(RetainCounter, NumRetains)
        << "Expect " << NumRetains << " piProgramRetain calls";

    auto CtxImpl = detail::getSyclObjImpl(Context);
    detail::KernelProgramCache::KernelCacheT &KernelCache =
        CtxImpl->getKernelProgramCache().acquireKernelsPerProgramCache().get();

    EXPECT_EQ(KernelCache.size(), (size_t)2) << "Expect 2 kernels in cache";
  }
  // First kernel creating is called in handler::single_task().
  // kernel_bundle::get_kernel() creates a kernel and shares it with created
  // programs. Also the kernel is retained in kernel_bundle::get_kernel(). A
  // kernel is removed from cache if piKernelRelease was called for it, so it
  // will not be removed twice for the other programs. As a result we must
  // expect 3 piKernelRelease calls.
  EXPECT_EQ(KernelReleaseCounter, 3) << "Expect 3 piKernelRelease calls";
}
