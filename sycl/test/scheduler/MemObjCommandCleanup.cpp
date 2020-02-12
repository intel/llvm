// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out
#include <CL/sycl.hpp>

#include <memory>
#include <utility>

#include "SchedulerTestUtils.hpp"

// This test checks that the execution graph cleanup on memory object
// destruction traverses the entire graph, rather than only the immediate users
// of deleted commands.

using namespace cl::sycl;

int main() {
  TestScheduler TS;
  queue Queue;
  buffer<int, 1> BufA(range<1>(1));
  buffer<int, 1> BufB(range<1>(1));
  detail::Requirement FakeReqA = getFakeRequirement(BufA);
  detail::Requirement FakeReqB = getFakeRequirement(BufB);
  detail::MemObjRecord *RecA =
      TS.getOrInsertMemObjRecord(detail::getSyclObjImpl(Queue), &FakeReqA);

  // Create 2 fake allocas, one of which will be cleaned up
  detail::AllocaCommand *FakeAllocaA =
      new detail::AllocaCommand(detail::getSyclObjImpl(Queue), FakeReqA);
  std::unique_ptr<detail::AllocaCommand> FakeAllocaB{
      new detail::AllocaCommand(detail::getSyclObjImpl(Queue), FakeReqB)};
  RecA->MAllocaCommands.push_back(FakeAllocaA);

  // Create a direct user of both allocas
  std::unique_ptr<FakeCommand> FakeDirectUser{
      new FakeCommand(detail::getSyclObjImpl(Queue), FakeReqA)};
  addEdge(FakeDirectUser.get(), FakeAllocaA, FakeAllocaA);
  addEdge(FakeDirectUser.get(), FakeAllocaB.get(), FakeAllocaB.get());

  // Create an indirect user of the soon-to-be deleted alloca
  bool IndirectUserDeleted = false;
  std::function<void()> Callback = [&]() { IndirectUserDeleted = true; };
  FakeCommand *FakeIndirectUser = new FakeCommandWithCallback(
      detail::getSyclObjImpl(Queue), FakeReqA, Callback);
  addEdge(FakeIndirectUser, FakeDirectUser.get(), FakeAllocaA);

  TS.cleanupCommandsForRecord(RecA);
  TS.removeRecordForMemObj(detail::getSyclObjImpl(BufA).get());

  // Check that the direct user has been left with the second alloca
  // as the only dependency, while the indirect user has been cleaned up.
  assert(FakeDirectUser->MUsers.size() == 0);
  assert(FakeDirectUser->MDeps.size() == 1);
  assert(FakeDirectUser->MDeps[0].MDepCommand == FakeAllocaB.get());
  assert(IndirectUserDeleted);
}
