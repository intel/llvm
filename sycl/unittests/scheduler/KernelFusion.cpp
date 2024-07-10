//==----------- KernelFusion.cpp - Kernel Fusion scheduler unit tests ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SchedulerTest.hpp"
#include "SchedulerTestUtils.hpp"

#include <helpers/PiMock.hpp>
#include <helpers/ScopedEnvVar.hpp>
#include <helpers/TestKernel.hpp>

#include <vector>

using namespace sycl;
using EventImplPtr = std::shared_ptr<detail::event_impl>;

template <typename T, int Dim>
detail::Command *CreateTaskCommand(MockScheduler &MS,
                                   detail::QueueImplPtr DevQueue,
                                   buffer<T, Dim> &buf) {
  MockHandlerCustomFinalize MockCGH(DevQueue,
                                    /*CallerNeedsEvent=*/true);

  auto acc = buf.get_access(static_cast<sycl::handler &>(MockCGH));

  kernel_bundle KernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(
          DevQueue->get_context());
  auto ExecBundle = sycl::build(KernelBundle);
  MockCGH.use_kernel_bundle(ExecBundle);
  MockCGH.single_task<TestKernel<>>([] {});

  auto CmdGrp = MockCGH.finalize();

  std::vector<detail::Command *> ToEnqueue;
  detail::Command *NewCmd =
      MS.addCG(std::move(CmdGrp), DevQueue, ToEnqueue, /*EventNeeded=*/true);
  EXPECT_EQ(ToEnqueue.size(), 0u);
  return NewCmd;
}

bool CheckTestExecRequirements(const platform &plt) {
  // This test only contains device image for SPIR-V capable devices.
  if (plt.get_backend() != sycl::backend::opencl &&
      plt.get_backend() != sycl::backend::ext_oneapi_level_zero) {
    std::cout << "Only OpenCL and Level Zero are supported for this test\n";
    return false;
  }
  return true;
}

bool containsCommand(detail::Command *Cmd,
                     std::vector<detail::Command *> &List) {
  return std::find(List.begin(), List.end(), Cmd) != List.end();
}

bool dependsOnViaDep(detail::Command *Dependent, detail::Command *Dependee) {
  return std::find_if(Dependent->MDeps.begin(), Dependent->MDeps.end(),
                      [=](detail::DepDesc &Desc) {
                        return Desc.MDepCommand == Dependee;
                      }) != Dependent->MDeps.end();
}

bool dependsOnViaEvent(detail::Command *Dependent, detail::Command *Dependee) {
  auto &DepEvents = Dependent->getPreparedDepsEvents();
  return std::find_if(DepEvents.begin(), DepEvents.end(),
                      [=](const EventImplPtr &Ev) {
                        return Ev->getCommand() && Ev->getCommand() == Dependee;
                      }) != DepEvents.end();
}

TEST_F(SchedulerTest, CancelKernelFusion) {
  unittest::PiMock Mock;
  platform Plt = Mock.getPlatform();
  if (!CheckTestExecRequirements(Plt))
    return;

  queue QueueDev(context(Plt), default_selector_v);
  MockScheduler MS;

  detail::QueueImplPtr QueueDevImpl = detail::getSyclObjImpl(QueueDev);

  // Test scenario: Create four memory objects (buffers) and one command for
  // each memory object before starting fusion. Then start fusion, again adding
  // one command with a requirement for each of the memory objects. Then cancel
  // fusion and check for correct dependencies.

  buffer<int, 1> b1{range<1>{4}};
  buffer<int, 1> b2{range<1>{4}};
  buffer<int, 1> b3{range<1>{4}};
  buffer<int, 1> b4{range<1>{4}};

  auto *nonFusionCmd1 = CreateTaskCommand(MS, QueueDevImpl, b1);
  auto *nonFusionCmd2 = CreateTaskCommand(MS, QueueDevImpl, b2);
  auto *nonFusionCmd3 = CreateTaskCommand(MS, QueueDevImpl, b3);
  auto *nonFusionCmd4 = CreateTaskCommand(MS, QueueDevImpl, b4);

  MS.startFusion(QueueDevImpl);

  auto *fusionCmd1 = CreateTaskCommand(MS, QueueDevImpl, b1);
  auto *fusionCmd2 = CreateTaskCommand(MS, QueueDevImpl, b2);
  auto *fusionCmd3 = CreateTaskCommand(MS, QueueDevImpl, b3);
  auto *fusionCmd4 = CreateTaskCommand(MS, QueueDevImpl, b4);

  std::vector<detail::Command *> ToEnqueue;
  MS.cancelFusion(QueueDevImpl, ToEnqueue);

  // The list of commands filled by cancelFusion should contain the four
  // commands submitted while in fusion mode, plus the placeholder command.
  EXPECT_EQ(ToEnqueue.size(), 5u);
  EXPECT_TRUE(containsCommand(fusionCmd1, ToEnqueue));
  EXPECT_TRUE(containsCommand(fusionCmd2, ToEnqueue));
  EXPECT_TRUE(containsCommand(fusionCmd3, ToEnqueue));
  EXPECT_TRUE(containsCommand(fusionCmd4, ToEnqueue));

  // Each of the commands submitted while in fusion mode should have exactly one
  // dependency on the command not participating in fusion, but accessing the
  // same memory object.
  EXPECT_TRUE(dependsOnViaDep(fusionCmd1, nonFusionCmd1));
  EXPECT_EQ(fusionCmd1->MDeps.size(), 1u);
  EXPECT_TRUE(dependsOnViaDep(fusionCmd2, nonFusionCmd2));
  EXPECT_EQ(fusionCmd2->MDeps.size(), 1u);
  EXPECT_TRUE(dependsOnViaDep(fusionCmd3, nonFusionCmd3));
  EXPECT_EQ(fusionCmd3->MDeps.size(), 1u);
  EXPECT_TRUE(dependsOnViaDep(fusionCmd4, nonFusionCmd4));
  EXPECT_EQ(fusionCmd4->MDeps.size(), 1u);

  // There should be one placeholder command in the command list.
  auto FusionCmdIt = std::find_if(
      ToEnqueue.begin(), ToEnqueue.end(), [](detail::Command *Cmd) {
        return Cmd->getType() == sycl::_V1::detail::Command::FUSION;
      });
  EXPECT_NE(FusionCmdIt, ToEnqueue.end());

  // Check that the placeholder command has an event dependency on each of the
  // commands submitted while in fusion mode.
  auto *placeHolderCmd =
      static_cast<detail::KernelFusionCommand *>(*FusionCmdIt);
  EXPECT_EQ(placeHolderCmd->getPreparedDepsEvents().size(), 4u);
  EXPECT_TRUE(dependsOnViaEvent(placeHolderCmd, fusionCmd2));
  EXPECT_TRUE(dependsOnViaEvent(placeHolderCmd, fusionCmd3));
  EXPECT_TRUE(dependsOnViaEvent(placeHolderCmd, fusionCmd4));
  EXPECT_TRUE(dependsOnViaEvent(placeHolderCmd, fusionCmd1));
}
