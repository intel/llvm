// RUN: %clangxx -fsycl -I %sycl_source_dir %s -o %t.out
// RUN: %t.out
#include <CL/sycl.hpp>
#include <detail/scheduler/scheduler.hpp>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <vector>

#include "SchedulerTestUtils.hpp"

// This test checks the leaf limit imposed on the execution graph

using namespace cl::sycl;

int main() {
  TestScheduler TS;
  queue Queue;
  buffer<int, 1> Buf(range<1>(1));
  detail::Requirement FakeReq{{0, 0, 0},
                              {0, 0, 0},
                              {0, 0, 0},
                              access::mode::read_write,
                              detail::getSyclObjImpl(Buf).get(),
                              0,
                              0,
                              0};
  FakeCommand *FakeDepCmd =
      new FakeCommand(detail::getSyclObjImpl(Queue), FakeReq);
  detail::MemObjRecord *Rec =
      TS.getOrInsertMemObjRecord(detail::getSyclObjImpl(Queue), &FakeReq);

  // Create commands that will be added as leaves exceeding the limit by 1
  std::vector<FakeCommand *> LeavesToAdd;
  for (std::size_t i = 0; i < Rec->MWriteLeaves.capacity() + 1; ++i) {
    LeavesToAdd.push_back(
        new FakeCommand(detail::getSyclObjImpl(Queue), FakeReq));
  }
  // Create edges: all soon-to-be leaves are direct users of FakeDep
  for (auto Leaf : LeavesToAdd) {
    FakeDepCmd->addUser(Leaf);
    Leaf->addDep(detail::DepDesc{FakeDepCmd, Leaf->getRequirement(), nullptr});
  }
  // Add edges as leaves and exceed the leaf limit
  for (auto LeafPtr : LeavesToAdd) {
    TS.addNodeToLeaves(Rec, LeafPtr);
  }
  // Check that the oldest leaf has been removed from the leaf list
  // and added as a dependency of the newest one instead
  const detail::CircularBuffer<detail::Command *> &Leaves = Rec->MWriteLeaves;
  assert(std::find(Leaves.begin(), Leaves.end(), LeavesToAdd.front()) ==
         Leaves.end());
  for (std::size_t i = 1; i < LeavesToAdd.size(); ++i) {
    assert(std::find(Leaves.begin(), Leaves.end(), LeavesToAdd[i]) !=
           Leaves.end());
  }
  FakeCommand *OldestLeaf = LeavesToAdd.front();
  FakeCommand *NewestLeaf = LeavesToAdd.back();
  assert(OldestLeaf->MUsers.size() == 1);
  assert(OldestLeaf->MUsers.count(NewestLeaf));
  assert(NewestLeaf->MDeps.size() == 2);
  assert(std::any_of(
      NewestLeaf->MDeps.begin(), NewestLeaf->MDeps.end(),
      [&](const detail::DepDesc &DD) { return DD.MDepCommand == OldestLeaf; }));
}
