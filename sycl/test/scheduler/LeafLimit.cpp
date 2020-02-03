// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out
#include <CL/sycl.hpp>

#include <memory>
#include <vector>

// This test checks the leaf limit imposed on the execution graph

using namespace cl::sycl;

class FakeCommand : public detail::Command {
public:
  FakeCommand(detail::QueueImplPtr Queue, detail::Requirement Req)
      : Command{detail::Command::ALLOCA, Queue}, MRequirement{std::move(Req)} {}

  void printDot(std::ostream &Stream) const override {}

  const detail::Requirement *getRequirement() const final {
    return &MRequirement;
  };

  cl_int enqueueImp() override { return MRetVal; }

  cl_int MRetVal = CL_SUCCESS;

protected:
  detail::Requirement MRequirement;
};

class TestScheduler : public detail::Scheduler {
public:
  void AddNodeToLeaves(detail::MemObjRecord *Rec, detail::Command *Cmd,
                       access::mode Mode) {
    return MGraphBuilder.AddNodeToLeaves(Rec, Cmd, Mode);
  }

  detail::MemObjRecord *
  getOrInsertMemObjRecord(const detail::QueueImplPtr &Queue,
                          detail::Requirement *Req) {
    return MGraphBuilder.getOrInsertMemObjRecord(Queue, Req);
  }
};

int main() {
  TestScheduler TS;
  queue Queue;
  buffer<int, 1> Buf(range<1>(1));
  detail::Requirement FakeReq{{0, 0, 0},
                              {0, 0, 0},
                              {0, 0, 0},
                              access::mode::read_write,
                              detail::getSyclObjImpl(Buf),
                              0,
                              0,
                              0};
  FakeCommand *FakeDepCmd =
      new FakeCommand(detail::getSyclObjImpl(Queue), FakeReq);
  detail::MemObjRecord *Rec =
      TS.getOrInsertMemObjRecord(detail::getSyclObjImpl(Queue), &FakeReq);

  // Create commands that will be added as leaves exceeding the limit by 1
  std::vector<FakeCommand *> LeavesToAdd;
  for (size_t i = 0; i < Rec->MWriteLeaves.capacity() + 1; ++i) {
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
    TS.AddNodeToLeaves(Rec, LeafPtr, access::mode::read_write);
  }
  // Check that the oldest leaf has been removed from the leaf list
  // and added as a dependency of the newest one instead
  const detail::CircularBuffer<detail::Command *> &Leaves = Rec->MWriteLeaves;
  assert(std::find(Leaves.begin(), Leaves.end(), LeavesToAdd.front()) ==
         Leaves.end());
  for (size_t i = 1; i < LeavesToAdd.size(); ++i) {
    assert(std::find(Leaves.begin(), Leaves.end(), LeavesToAdd[i]) !=
           Leaves.end());
  }
  FakeCommand *OldestLeaf = LeavesToAdd.front();
  FakeCommand *NewestLeaf = LeavesToAdd.back();
  assert(OldestLeaf->MUsers.size() == 1);
  assert(OldestLeaf->MUsers[0] == NewestLeaf);
  assert(NewestLeaf->MDeps.size() == 2);
  assert(NewestLeaf->MDeps[1].MDepCommand == OldestLeaf);
}
