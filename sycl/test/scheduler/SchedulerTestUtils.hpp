#pragma once
#include <CL/sycl.hpp>

#include <functional>
// This header contains a few common classes/methods used in
// execution graph testing.

class FakeCommand : public cl::sycl::detail::Command {
public:
  FakeCommand(cl::sycl::detail::QueueImplPtr Queue,
              cl::sycl::detail::Requirement Req)
      : Command{cl::sycl::detail::Command::EMPTY_TASK, Queue},
        MRequirement{std::move(Req)} {}

  void printDot(std::ostream &Stream) const override {}

  const cl::sycl::detail::Requirement *getRequirement() const final {
    return &MRequirement;
  };

  cl_int enqueueImp() override { return MRetVal; }

  cl_int MRetVal = CL_SUCCESS;

protected:
  cl::sycl::detail::Requirement MRequirement;
};

class FakeCommandWithCallback : public FakeCommand {
public:
  FakeCommandWithCallback(cl::sycl::detail::QueueImplPtr Queue,
                          cl::sycl::detail::Requirement Req,
                          std::function<void()> Callback)
      : FakeCommand(Queue, Req), MCallback(std::move(Callback)) {}

  ~FakeCommandWithCallback() override { MCallback(); }

protected:
  std::function<void()> MCallback;
};

class TestScheduler : public cl::sycl::detail::Scheduler {
public:
  cl::sycl::detail::MemObjRecord *
  getOrInsertMemObjRecord(const cl::sycl::detail::QueueImplPtr &Queue,
                          cl::sycl::detail::Requirement *Req) {
    return MGraphBuilder.getOrInsertMemObjRecord(Queue, Req);
  }

  void removeRecordForMemObj(cl::sycl::detail::SYCLMemObjI *MemObj) {
    MGraphBuilder.removeRecordForMemObj(MemObj);
  }

  void cleanupCommandsForRecord(cl::sycl::detail::MemObjRecord *Rec) {
    MGraphBuilder.cleanupCommandsForRecord(Rec);
  }

  void addNodeToLeaves(
      cl::sycl::detail::MemObjRecord *Rec, cl::sycl::detail::Command *Cmd,
      cl::sycl::access::mode Mode = cl::sycl::access::mode::read_write) {
    return MGraphBuilder.addNodeToLeaves(Rec, Cmd, Mode);
  }
};

void addEdge(cl::sycl::detail::Command *User, cl::sycl::detail::Command *Dep,
             cl::sycl::detail::AllocaCommandBase *Alloca) {
  User->addDep(cl::sycl::detail::DepDesc{Dep, User->getRequirement(), Alloca});
  Dep->addUser(User);
}

template <typename MemObjT>
cl::sycl::detail::Requirement getFakeRequirement(const MemObjT &MemObj) {
  return {{0, 0, 0},
          {0, 0, 0},
          {0, 0, 0},
          cl::sycl::access::mode::read_write,
          cl::sycl::detail::getSyclObjImpl(MemObj).get(),
          0,
          0,
          0};
}
