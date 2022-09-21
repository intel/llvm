//==----------- RunOnHostIntelCG.cpp --- Scheduler unit tests --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//

#include "SchedulerTest.hpp"
#include "SchedulerTestUtils.hpp"

#include <helpers/PiMock.hpp>

#include <detail/event_impl.hpp>

using namespace sycl;

bool CGDeleted = false;
class MockCGExecKernel : public detail::CGExecKernel {
public:
  MockCGExecKernel(detail::NDRDescT NDRDesc,
                   std::unique_ptr<detail::HostKernelBase> HostKernel)
      : CGExecKernel(NDRDesc, std::move(HostKernel), /*SyclKernel*/ nullptr,
                     /*Kernelbundle*/ nullptr,
                     /*ArgsStorage*/ {}, /*AccStorage*/ {},
                     /*SharedPtrStorage*/ {}, /*Requirements*/ {},
                     /*Events*/ {}, /*Args*/ {}, /*KernelName*/ "",
                     detail::OSUtil::ExeModuleHandle, /*Streams*/ {},
                     /*AuxilaryResources*/ {}, detail::CG::RunOnHostIntel) {}
  ~MockCGExecKernel() override { CGDeleted = true; }
};

// Check that the command group associated with run_on_host_intel is properly
// released on command destruction.
TEST_F(SchedulerTest, RunOnHostIntelCG) {
  sycl::unittest::PiMock Mock;
  sycl::queue Q{Mock.getPlatform().get_devices()[0], MAsyncHandler};

  MockScheduler MS;
  detail::QueueImplPtr QueueImpl = detail::getSyclObjImpl(Q);

  detail::NDRDescT NDRDesc;
  NDRDesc.set(range<1>{1}, id<1>{0});
  std::unique_ptr<detail::HostKernelBase> HostKernel{
      new detail::HostKernel<std::function<void()>, void, 1>([]() {})};
  std::unique_ptr<detail::CG> CommandGroup{
      new MockCGExecKernel(std::move(NDRDesc), std::move(HostKernel))};
  MS.addCG(std::move(CommandGroup), QueueImpl);
  EXPECT_TRUE(CGDeleted);
}
