//==------------ Regression.cpp --- Scheduler unit tests -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SchedulerTest.hpp"
#include "SchedulerTestUtils.hpp"

#include <helpers/PiMock.hpp>

using namespace sycl;

static std::tuple<detail::NDRDescT, detail::Requirement> initializeRefValues() {
  detail::NDRDescT NDRDesc;
  NDRDesc.GlobalSize = range<3>(1, 2, 3);
  NDRDesc.LocalSize = range<3>(1, 1, 1);

  int val;
  buffer<int, 1> Buf(&val, range<1>(1));
  detail::Requirement MockReq = getMockRequirement(Buf);
  return {NDRDesc, MockReq};
}

static pi_result redefinedEnqueueNativeKernel(
    pi_queue queue, void (*user_func)(void *), void *args, size_t cb_args,
    pi_uint32 num_mem_objects, const pi_mem *mem_list,
    const void **args_mem_loc, pi_uint32 num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event) {
  void **CastedBlob = (void **)args;
  auto [NDRDesc, MockReq] = initializeRefValues();

  std::vector<detail::Requirement *> Reqs =
      *(static_cast<std::vector<detail::Requirement *> *>(CastedBlob[0]));
  EXPECT_EQ(Reqs[0]->MAccessRange[0], MockReq.MAccessRange[0]);
  EXPECT_EQ(Reqs[0]->MAccessRange[1], MockReq.MAccessRange[1]);
  EXPECT_EQ(Reqs[0]->MAccessRange[2], MockReq.MAccessRange[2]);

  detail::HostKernelBase *HostKernel =
      static_cast<detail::HostKernelBase *>(CastedBlob[1]);
  testing::internal::CaptureStdout();
  HostKernel->call(NDRDesc, nullptr);
  std::string Output = testing::internal::GetCapturedStdout();
  EXPECT_EQ(Output, "Blablabla");

  detail::NDRDescT *NDRDescActual =
      static_cast<detail::NDRDescT *>(CastedBlob[2]);
  EXPECT_EQ(NDRDescActual->GlobalSize[0], NDRDesc.GlobalSize[0]);
  EXPECT_EQ(NDRDescActual->GlobalSize[1], NDRDesc.GlobalSize[1]);
  EXPECT_EQ(NDRDescActual->GlobalSize[2], NDRDesc.GlobalSize[2]);

  return PI_SUCCESS;
}

TEST_F(SchedulerTest, CheckArgsBlobInPiEnqueueNativeKernelIsValid) {
  sycl::unittest::PiMock Mock;
  sycl::platform Plt = Mock.getPlatform();
  Mock.redefineBefore<detail::PiApiKind::piEnqueueNativeKernel>(
      redefinedEnqueueNativeKernel);

  auto Kernel = []() { std::cout << "Blablabla"; };
  detail::HostKernel<decltype(Kernel), void, 1> HKernel(Kernel);
  auto [NDRDesc, MockReq] = initializeRefValues();

  detail::CG::StorageInitHelper CGData(
      /*ArgsStorage*/ {},
      /*AccStorage*/ {},
      /*SharedPtrStorage*/ {},
      /*Requirements*/ {&MockReq},
      /*Events*/ {});
  std::unique_ptr<detail::CG> CG{new detail::CGExecKernel(
      /*NDRDesc*/ NDRDesc,
      /*HKernel*/
      std::make_shared<detail::HostKernel<decltype(Kernel), void, 1>>(HKernel),
      /*SyclKernel*/ nullptr,
      /*KernelBundle*/ nullptr, std::move(CGData),
      /*Args*/ {},
      /*KernelName*/ "",
      /*Streams*/ {},
      /*AuxiliaryResources*/ {},
      /*Type*/ detail::CG::RunOnHostIntel,
      /*KernelCacheConfig*/ {})};

  context Ctx{Plt};
  queue Queue{Ctx, default_selector_v};
  detail::QueueImplPtr QueueImpl = detail::getSyclObjImpl(Queue);

  detail::ExecCGCommand ExecCGCmd{std::move(CG), QueueImpl};
  detail::EnqueueResultT EnqueueResult = detail::EnqueueResultT(
      detail::EnqueueResultT::SyclEnqueueReady, &ExecCGCmd);
  std::vector<sycl::detail::Command *> ToCleanUp;
  ExecCGCmd.enqueue(EnqueueResult, detail::BlockingT::NON_BLOCKING, ToCleanUp);
}
