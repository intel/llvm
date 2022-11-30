//==----------- FailedCommands.cpp ---- Scheduler unit tests ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SchedulerTest.hpp"
#include "SchedulerTestUtils.hpp"

#include <helpers/PiImage.hpp>
#include <helpers/PiMock.hpp>
#include <helpers/TestKernel.hpp>

using namespace sycl;

TEST_F(SchedulerTest, FailedDependency) {
  unittest::PiMock Mock;
  platform Plt = Mock.getPlatform();
  queue Queue(context(Plt), default_selector_v);

  detail::Requirement MockReq = getMockRequirement();
  MockCommand MDep(detail::getSyclObjImpl(Queue));
  MockCommand MUser(detail::getSyclObjImpl(Queue));
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
      PI_ERROR_INVALID_OPERATION);
  return PI_SUCCESS;
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
  unittest::PiMock Mock;
  platform Plt = Mock.getPlatform();
  MemBufRefCount = 0u;
  Mock.redefine<detail::PiApiKind::piEnqueueKernelLaunch>(
      redefinedFailingEnqueueKernelLaunch);
  Mock.redefine<detail::PiApiKind::piMemBufferCreate>(redefinedMemBufferCreate);
  Mock.redefine<detail::PiApiKind::piMemRetain>(redefinedMemRetain);
  Mock.redefine<detail::PiApiKind::piMemRelease>(redefinedMemRelease);

  {
    context Ctx{Plt};
    queue Q{Ctx, default_selector_v};

    kernel_bundle KernelBundle =
        sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx);
    auto ExecBundle = sycl::build(KernelBundle);

    buffer<int, 1> Buff{sycl::range<1>(1)};

    try {
      Q.submit([&](sycl::handler &CGH) {
        auto Acc = Buff.get_access<sycl::access::mode::read_write>(CGH);
        CGH.use_kernel_bundle(ExecBundle);
        CGH.single_task<TestKernel<>>([=] {});
      });
      FAIL() << "No exception was thrown.";
    } catch (...) {
    }
  }

  ASSERT_EQ(MemBufRefCount, 0u) << "Memory leak detected.";
}

TEST_F(SchedulerTest, FailedCommandStreamCleanup) {
  unittest::PiMock Mock;
  platform Plt = Mock.getPlatform();
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
        CGH.single_task<TestKernel<>>([=] {});
      });
      FAIL() << "No exception was thrown.";
    } catch (...) {
    }
    Q.wait();
  }

  ASSERT_EQ(MemBufRefCount, 0u) << "Memory leak detected.";
}
