//==------- ReadOnlyLeafEviction.cpp --- Scheduler unit tests -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Kernels which only read shared buffers are independent. Once the read leaves
// of a buffer are full, the oldest one becomes a dependency of the new kernel,
// so read leaves are allowed to grow beyond the generic leaf limit first.

#include "SchedulerTest.hpp"
#include "SchedulerTestUtils.hpp"

#include <detail/config.hpp>
#include <helpers/ScopedEnvVar.hpp>
#include <helpers/TestKernel.hpp>
#include <helpers/UrMock.hpp>

#include <map>
#include <set>
#include <vector>

namespace {
using namespace sycl;

constexpr size_t LeafLimit = 8;
constexpr size_t ReadLeafLimit = 16;

// Kernels launched so far, by output event, and the kernels each one waits on.
std::map<ur_event_handle_t, size_t> EventToKernel;
std::vector<std::set<size_t>> KernelDeps;

ur_result_t afterKernelLaunch(void *pParams) {
  auto &Params =
      *static_cast<ur_enqueue_kernel_launch_with_args_exp_params_t *>(pParams);
  std::set<size_t> Deps;
  for (uint32_t I = 0; I < *Params.pnumEventsInWaitList; ++I)
    if (auto It = EventToKernel.find((*Params.pphEventWaitList)[I]);
        It != EventToKernel.end())
      Deps.insert(It->second);
  EventToKernel[**Params.pphEvent] = KernelDeps.size();
  KernelDeps.push_back(std::move(Deps));
  return UR_RESULT_SUCCESS;
}

class ReadOnlyLeafEviction : public testing::TestWithParam<size_t> {
protected:
  // Keep leaves from being removed by cleanup in a timing dependent way.
  unittest::ScopedEnvVar DisableGraphCleanup{
      "SYCL_DISABLE_EXECUTION_GRAPH_CLEANUP", "1",
      detail::SYCLConfig<detail::SYCL_DISABLE_EXECUTION_GRAPH_CLEANUP>::reset};
  unittest::UrMock<> Mock;
  context Ctx{platform().get_devices()[0]};
  std::vector<queue> Queues;
  int A[1] = {0}, B[1] = {0};
  buffer<int, 1> ABuf{A, range<1>{1}};
  buffer<int, 1> BBuf{B, range<1>{1}};

  void SetUp() override {
    EventToKernel.clear();
    KernelDeps.clear();
    mock::getCallbacks().set_after_callback("urEnqueueKernelLaunchWithArgsExp",
                                            &afterKernelLaunch);
    for (size_t I = 0; I < GetParam(); ++I)
      Queues.emplace_back(Ctx, default_selector_v);
  }

  // Submits NumKernels round-robin to the queues, each reading both buffers.
  void submitReaders(size_t NumKernels) {
    for (size_t I = 0; I < NumKernels; ++I)
      Queues[I % Queues.size()].submit([&](handler &CGH) {
        accessor AAcc{ABuf, CGH, read_only};
        accessor BAcc{BBuf, CGH, read_only};
        CGH.single_task<TestKernel>([] {});
      });
  }
};

TEST_P(ReadOnlyLeafEviction, NoDependenciesBetweenReaders) {
  submitReaders(ReadLeafLimit);

  detail::Requirement Req = getMockRequirement(ABuf);
  detail::MemObjRecord *Record = detail::Scheduler::getMemObjRecord(&Req);
  EXPECT_EQ(Record->MReadLeaves.genericCommandsCapacity(), ReadLeafLimit);
  EXPECT_EQ(Record->MWriteLeaves.genericCommandsCapacity(), LeafLimit);

  ASSERT_EQ(KernelDeps.size(), ReadLeafLimit);
  for (size_t I = 0; I < KernelDeps.size(); ++I)
    EXPECT_TRUE(KernelDeps[I].empty()) << "kernel " << I;
}

TEST_P(ReadOnlyLeafEviction, EvictedReaderBecomesDependency) {
  constexpr size_t NumReaders = 2 * ReadLeafLimit + 1;
  submitReaders(NumReaders);
  // A writer depends on the read leaves, and on the evicted readers through
  // them.
  Queues[0].submit([&](handler &CGH) {
    accessor AAcc{ABuf, CGH, read_write};
    CGH.single_task<TestKernel>([] {});
  });

  ASSERT_EQ(KernelDeps.size(), NumReaders + 1);
  for (size_t I = 0; I < NumReaders; ++I) {
    std::set<size_t> Expected;
    if (I >= ReadLeafLimit)
      Expected.insert(I - ReadLeafLimit);
    EXPECT_EQ(KernelDeps[I], Expected) << "kernel " << I;
  }
  std::set<size_t> LastReaders;
  for (size_t I = NumReaders - ReadLeafLimit; I < NumReaders; ++I)
    LastReaders.insert(I);
  EXPECT_EQ(KernelDeps[NumReaders], LastReaders);
}

INSTANTIATE_TEST_SUITE_P(SchedulerTest, ReadOnlyLeafEviction,
                         testing::Values(1, 2, 3, 4));

} // namespace
