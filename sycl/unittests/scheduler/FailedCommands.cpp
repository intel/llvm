//==----------- FailedCommands.cpp ---- Scheduler unit tests ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SchedulerTest.hpp"
#include "SchedulerTestUtils.hpp"

#include <helpers/CommonRedefinitions.hpp>
#include <helpers/PiImage.hpp>
#include <helpers/PiMock.hpp>

using namespace cl::sycl;

class TestKernel;

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
template <> struct KernelInfo<TestKernel> {
  static constexpr unsigned getNumParams() { return 0; }
  static const kernel_param_desc_t &getParamDesc(int) {
    static kernel_param_desc_t Dummy;
    return Dummy;
  }
  static constexpr const char *getName() { return "TestKernel"; }
  static constexpr bool isESIMD() { return false; }
  static constexpr bool callsThisItem() { return false; }
  static constexpr bool callsAnyThisFreeFunction() { return false; }
};

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

static sycl::unittest::PiImage generateDefaultImage() {
  using namespace sycl::unittest;

  PiPropertySet PropSet;

  std::vector<unsigned char> Bin{0, 1, 2, 3, 4, 5}; // Random data

  PiArray<PiOffloadEntry> Entries = makeEmptyKernels({"TestKernel"});

  PiImage Img{PI_DEVICE_BINARY_TYPE_SPIRV,            // Format
              __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64, // DeviceTargetSpec
              "",                                     // Compile options
              "",                                     // Link options
              std::move(Bin),
              std::move(Entries),
              std::move(PropSet)};

  return Img;
}

static sycl::unittest::PiImage Img = generateDefaultImage();
static sycl::unittest::PiImageArray<1> ImgArray{&Img};

using namespace sycl;

TEST_F(SchedulerTest, FailedDependency) {
  detail::Requirement MockReq = getMockRequirement();
  MockCommand MDep(detail::getSyclObjImpl(MQueue));
  MockCommand MUser(detail::getSyclObjImpl(MQueue));
  MDep.addUser(&MUser);
  std::vector<detail::Command *> ToCleanUp;
  (void)MUser.addDep(detail::DepDesc{&MDep, &MockReq, nullptr}, ToCleanUp);
  MUser.MEnqueueStatus = detail::EnqueueResultT::SyclEnqueueReady;
  MDep.MEnqueueStatus = detail::EnqueueResultT::SyclEnqueueFailed;

  MockScheduler MS;
  auto Lock = MS.acquireGraphReadLock();
  detail::EnqueueResultT Res;
  bool Enqueued =
      MockScheduler::enqueueCommand(&MUser, Res, detail::NON_BLOCKING);

  ASSERT_FALSE(Enqueued) << "Enqueue process must fail\n";
  ASSERT_EQ(Res.MCmd, &MDep) << "Wrong failed command\n";
  ASSERT_EQ(Res.MResult, detail::EnqueueResultT::SyclEnqueueFailed)
      << "Enqueue process must fail\n";
  ASSERT_EQ(MUser.MEnqueueStatus, detail::EnqueueResultT::SyclEnqueueReady)
      << "MUser shouldn't be marked as failed\n";
  ASSERT_EQ(MDep.MEnqueueStatus, detail::EnqueueResultT::SyclEnqueueFailed)
      << "MDep should be marked as failed\n";
}

pi_result redefinedFailingEnqueueKernelLaunch(pi_queue, pi_kernel, pi_uint32,
                                              const size_t *, const size_t *,
                                              const size_t *, pi_uint32,
                                              const pi_event *, pi_event *) {
  throw sycl::runtime_error(
      "Exception from redefinedFailingEnqueueKernelLaunch.",
      PI_INVALID_OPERATION);
}

size_t MemBufRefCount = 0u;

pi_result redefinedMemBufferCreate(pi_context, pi_mem_flags, size_t, void *,
                                   pi_mem *ret_mem, const pi_mem_properties *) {
  *ret_mem = reinterpret_cast<pi_mem>(0x1);
  ++MemBufRefCount;
  return PI_SUCCESS;
}

pi_result redefinedMemBufferPartition(pi_mem, pi_mem_flags,
                                      pi_buffer_create_type, void *,
                                      pi_mem *ret_mem) {
  *ret_mem = reinterpret_cast<pi_mem>(0x1);
  ++MemBufRefCount;
  return PI_SUCCESS;
}

pi_result redefinedMemRetain(pi_mem) {
  ++MemBufRefCount;
  return PI_SUCCESS;
}

pi_result redefinedMemRelease(pi_mem) {
  --MemBufRefCount;
  return PI_SUCCESS;
}

TEST_F(SchedulerTest, FailedCommandAccessorCleanup) {
  default_selector Selector;
  platform Plt{default_selector()};
  if (Plt.is_host()) {
    std::cout << "Not run due to host-only environment\n";
    return;
  }
  if (Plt.get_backend() == sycl::backend::ext_oneapi_cuda ||
      Plt.get_backend() == sycl::backend::ext_oneapi_hip) {
    std::cout << "CUDA and HIP backends do not currently support this test\n";
    return;
  }

  unittest::PiMock Mock{Plt};
  setupDefaultMockAPIs(Mock);
  MemBufRefCount = 0u;
  Mock.redefine<detail::PiApiKind::piEnqueueKernelLaunch>(
      redefinedFailingEnqueueKernelLaunch);
  Mock.redefine<detail::PiApiKind::piMemBufferCreate>(redefinedMemBufferCreate);
  Mock.redefine<detail::PiApiKind::piMemRetain>(redefinedMemRetain);
  Mock.redefine<detail::PiApiKind::piMemRelease>(redefinedMemRelease);

  {
    context Ctx{Plt};
    queue Q{Ctx, Selector};

    kernel_bundle KernelBundle =
        sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx);
    auto ExecBundle = sycl::build(KernelBundle);

    buffer<int, 1> Buff{cl::sycl::range<1>(1)};

    try {
      Q.submit([&](sycl::handler &CGH) {
        auto Acc = Buff.get_access<cl::sycl::access::mode::read_write>(CGH);
        CGH.use_kernel_bundle(ExecBundle);
        CGH.single_task<TestKernel>([=] {});
      });
      FAIL() << "No exception was thrown.";
    } catch (...) {
    }
  }

  ASSERT_EQ(MemBufRefCount, 0u) << "Memory leak detected.";
}

TEST_F(SchedulerTest, FailedCommandStreamCleanup) {
  default_selector Selector;
  platform Plt{default_selector()};
  if (Plt.is_host()) {
    std::cout << "Not run due to host-only environment\n";
    return;
  }
  if (Plt.get_backend() == sycl::backend::ext_oneapi_cuda ||
      Plt.get_backend() == sycl::backend::ext_oneapi_hip) {
    std::cout << "CUDA and HIP backends do not currently support this test\n";
    return;
  }

  unittest::PiMock Mock{Plt};
  setupDefaultMockAPIs(Mock);
  MemBufRefCount = 0u;
  Mock.redefine<detail::PiApiKind::piEnqueueKernelLaunch>(
      redefinedFailingEnqueueKernelLaunch);
  Mock.redefine<detail::PiApiKind::piMemBufferCreate>(redefinedMemBufferCreate);
  Mock.redefine<detail::PiApiKind::piMemBufferPartition>(
      redefinedMemBufferPartition);
  Mock.redefine<detail::PiApiKind::piMemRetain>(redefinedMemRetain);
  Mock.redefine<detail::PiApiKind::piMemRelease>(redefinedMemRelease);

  {
    context Ctx{Plt};
    queue Q{Ctx, Selector};

    kernel_bundle KernelBundle =
        sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx);
    auto ExecBundle = sycl::build(KernelBundle);

    try {
      Q.submit([&](sycl::handler &CGH) {
        sycl::stream KernelStream(108 * 64 + 128, 64, CGH);
        CGH.use_kernel_bundle(ExecBundle);
        CGH.single_task<TestKernel>([=] {});
      });
      FAIL() << "No exception was thrown.";
    } catch (...) {
    }
    Q.wait();
  }

  ASSERT_EQ(MemBufRefCount, 0u) << "Memory leak detected.";
}
