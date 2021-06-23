//==---------- SchedulerTestUtils.hpp --- Scheduler unit tests -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <sycl/sycl.hpp>
#include <sycl/__impl/detail/cl.h>
#include <detail/queue_impl.hpp>
#include <detail/scheduler/scheduler.hpp>
#include <detail/stream_impl.hpp>

#include <functional>
#include <gmock/gmock.h>
#include <vector>

// This header contains a few common classes/methods used in
// execution graph testing.

__sycl_internal::__v1::detail::Requirement getMockRequirement();

#ifdef __SYCL_ENABLE_SYCL121_NAMESPACE
__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
#else
namespace __sycl_internal {
inline namespace __v1 {
#endif
namespace detail {
class Command;
} // namespace detail
} // namespace __v1
} // namespace __sycl_internal

class MockCommand : public __sycl_internal::__v1::detail::Command {
public:
  MockCommand(__sycl_internal::__v1::detail::QueueImplPtr Queue,
              __sycl_internal::__v1::detail::Requirement Req,
              __sycl_internal::__v1::detail::Command::CommandType Type =
                  __sycl_internal::__v1::detail::Command::RUN_CG)
      : Command{Type, Queue}, MRequirement{std::move(Req)} {
    using namespace testing;
    ON_CALL(*this, enqueue(_, _))
        .WillByDefault(Invoke(this, &MockCommand::enqueueOrigin));
    EXPECT_CALL(*this, enqueue(_, _)).Times(AnyNumber());
  }

  MockCommand(__sycl_internal::__v1::detail::QueueImplPtr Queue,
              __sycl_internal::__v1::detail::Command::CommandType Type =
                  __sycl_internal::__v1::detail::Command::RUN_CG)
      : Command{Type, Queue}, MRequirement{std::move(getMockRequirement())} {
    using namespace testing;
    ON_CALL(*this, enqueue(_, _))
        .WillByDefault(Invoke(this, &MockCommand::enqueueOrigin));
    EXPECT_CALL(*this, enqueue(_, _)).Times(AnyNumber());
  }

  void printDot(std::ostream &) const override {}
  void emitInstrumentationData() override {}

  const __sycl_internal::__v1::detail::Requirement *getRequirement() const final {
    return &MRequirement;
  };

  cl_int enqueueImp() override { return MRetVal; }

  MOCK_METHOD2(enqueue, bool(__sycl_internal::__v1::detail::EnqueueResultT &,
                             __sycl_internal::__v1::detail::BlockingT));
  bool enqueueOrigin(__sycl_internal::__v1::detail::EnqueueResultT &EnqueueResult,
                     __sycl_internal::__v1::detail::BlockingT Blocking) {
    return __sycl_internal::__v1::detail::Command::enqueue(EnqueueResult, Blocking);
  }

  cl_int MRetVal = CL_SUCCESS;

  void waitForEventsCall(
      std::shared_ptr<__sycl_internal::__v1::detail::queue_impl> Queue,
      std::vector<std::shared_ptr<__sycl_internal::__v1::detail::event_impl>> &RawEvents,
      pi_event &Event) {
    Command::waitForEvents(Queue, RawEvents, Event);
  }

protected:
  __sycl_internal::__v1::detail::Requirement MRequirement;
};

class MockCommandWithCallback : public MockCommand {
public:
  MockCommandWithCallback(__sycl_internal::__v1::detail::QueueImplPtr Queue,
                          __sycl_internal::__v1::detail::Requirement Req,
                          std::function<void()> Callback)
      : MockCommand(Queue, Req), MCallback(std::move(Callback)) {}

  ~MockCommandWithCallback() override { MCallback(); }

protected:
  std::function<void()> MCallback;
};

class MockScheduler : public __sycl_internal::__v1::detail::Scheduler {
public:
  __sycl_internal::__v1::detail::MemObjRecord *
  getOrInsertMemObjRecord(const __sycl_internal::__v1::detail::QueueImplPtr &Queue,
                          __sycl_internal::__v1::detail::Requirement *Req,
                          std::vector<__sycl_internal::__v1::detail::Command *> &ToEnqueue) {
    return MGraphBuilder.getOrInsertMemObjRecord(Queue, Req, ToEnqueue);
  }

  void removeRecordForMemObj(__sycl_internal::__v1::detail::SYCLMemObjI *MemObj) {
    MGraphBuilder.removeRecordForMemObj(MemObj);
  }

  void cleanupCommandsForRecord(__sycl_internal::__v1::detail::MemObjRecord *Rec) {
    std::vector<std::shared_ptr<__sycl_internal::__v1::detail::stream_impl>>
        StreamsToDeallocate;
    MGraphBuilder.cleanupCommandsForRecord(Rec, StreamsToDeallocate);
  }

  void addNodeToLeaves(__sycl_internal::__v1::detail::MemObjRecord *Rec,
                       __sycl_internal::__v1::detail::Command *Cmd,
                       __sycl_internal::__v1::access::mode Mode,
                       std::vector<__sycl_internal::__v1::detail::Command *> &ToEnqueue) {
    return MGraphBuilder.addNodeToLeaves(Rec, Cmd, Mode, ToEnqueue);
  }

  static bool enqueueCommand(__sycl_internal::__v1::detail::Command *Cmd,
                             __sycl_internal::__v1::detail::EnqueueResultT &EnqueueResult,
                             __sycl_internal::__v1::detail::BlockingT Blocking) {
    return GraphProcessor::enqueueCommand(Cmd, EnqueueResult, Blocking);
  }

  __sycl_internal::__v1::detail::AllocaCommandBase *
  getOrCreateAllocaForReq(__sycl_internal::__v1::detail::MemObjRecord *Record,
                          const __sycl_internal::__v1::detail::Requirement *Req,
                          __sycl_internal::__v1::detail::QueueImplPtr Queue,
                          std::vector<__sycl_internal::__v1::detail::Command *> &ToEnqueue) {
    return MGraphBuilder.getOrCreateAllocaForReq(Record, Req, Queue, ToEnqueue);
  }

  ReadLockT acquireGraphReadLock() { return ReadLockT{MGraphLock}; }

  __sycl_internal::__v1::detail::Command *
  insertMemoryMove(__sycl_internal::__v1::detail::MemObjRecord *Record,
                   __sycl_internal::__v1::detail::Requirement *Req,
                   const __sycl_internal::__v1::detail::QueueImplPtr &Queue,
                   std::vector<__sycl_internal::__v1::detail::Command *> &ToEnqueue) {
    return MGraphBuilder.insertMemoryMove(Record, Req, Queue, ToEnqueue);
  }

  __sycl_internal::__v1::detail::Command *
  addCG(std::unique_ptr<__sycl_internal::__v1::detail::CG> CommandGroup,
        __sycl_internal::__v1::detail::QueueImplPtr Queue,
        std::vector<__sycl_internal::__v1::detail::Command *> &ToEnqueue) {
    return MGraphBuilder.addCG(std::move(CommandGroup), Queue, ToEnqueue);
  }
};

void addEdge(__sycl_internal::__v1::detail::Command *User, __sycl_internal::__v1::detail::Command *Dep,
             __sycl_internal::__v1::detail::AllocaCommandBase *Alloca);

template <typename MemObjT>
__sycl_internal::__v1::detail::Requirement getMockRequirement(const MemObjT &MemObj) {
  return {/*Offset*/ {0, 0, 0},
          /*AccessRange*/ {0, 0, 0},
          /*MemoryRange*/ {0, 0, 0},
          /*AccessMode*/ sycl::access::mode::read_write,
          /*SYCLMemObj*/ __sycl_internal::__v1::detail::getSyclObjImpl(MemObj).get(),
          /*Dims*/ 0,
          /*ElementSize*/ 0};
}
