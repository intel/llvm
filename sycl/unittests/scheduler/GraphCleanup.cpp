//==--------- GraphCleanup.cpp --- Scheduler unit tests --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SchedulerTest.hpp"
#include "SchedulerTestUtils.hpp"

#include <helpers/ScopedEnvVar.hpp>
#include <helpers/TestKernel.hpp>
#include <helpers/UrMock.hpp>

#include <detail/buffer_impl.hpp>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <mutex>
#include <vector>

namespace {

using namespace sycl;

inline constexpr auto HostUnifiedMemoryName = "SYCL_HOST_UNIFIED_MEMORY";

int val;
static ur_result_t redefinedEnqueueMemBufferMap(void *pParams) {
  auto params = *static_cast<ur_enqueue_mem_buffer_map_params_t *>(pParams);
  **params.pphEvent = reinterpret_cast<ur_event_handle_t>(new int{});
  **params.pppRetMap = &val;
  return UR_RESULT_SUCCESS;
}

static ur_result_t redefinedEnqueueMemUnmap(void *pParams) {
  auto params = *static_cast<ur_enqueue_mem_unmap_params_t *>(pParams);
  **params.pphEvent = reinterpret_cast<ur_event_handle_t>(new int{});
  return UR_RESULT_SUCCESS;
}

static ur_result_t redefinedEnqueueMemBufferFill(void *pParams) {
  auto params = *static_cast<ur_enqueue_mem_buffer_fill_params_t *>(pParams);
  **params.pphEvent = reinterpret_cast<ur_event_handle_t>(new int{});
  return UR_RESULT_SUCCESS;
}

static void verifyCleanup(detail::MemObjRecord *Record,
                          detail::AllocaCommandBase *AllocaCmd,
                          detail::Command *DeletedCmd, bool &CmdDeletedFlag) {
  EXPECT_TRUE(CmdDeletedFlag);
  CmdDeletedFlag = false;
  EXPECT_EQ(
      std::find(AllocaCmd->MUsers.begin(), AllocaCmd->MUsers.end(), DeletedCmd),
      AllocaCmd->MUsers.end());
  detail::Command *Leaf = *Record->MWriteLeaves.begin();
  EXPECT_FALSE(std::any_of(Leaf->MDeps.begin(), Leaf->MDeps.end(),
                           [&](const detail::DepDesc &Dep) {
                             return Dep.MDepCommand == DeletedCmd;
                           }));
}

// Check that any non-leaf commands enqueued as part of high level scheduler
// calls are cleaned up.
static void checkCleanupOnEnqueue(MockScheduler &MS,
                                  detail::QueueImplPtr &QueueImpl,
                                  buffer<int, 1> &Buf,
                                  detail::Requirement &MockReq) {
  bool CommandDeleted = false;
  std::vector<detail::Command *> ToCleanUp;
  std::vector<detail::Command *> ToEnqueue;
  detail::MemObjRecord *Record =
      MS.getOrInsertMemObjRecord(QueueImpl, &MockReq);
  detail::AllocaCommandBase *AllocaCmd =
      MS.getOrCreateAllocaForReq(Record, &MockReq, QueueImpl, ToEnqueue);
  std::function<void()> Callback = [&CommandDeleted]() {
    CommandDeleted = true;
  };

  // Check addCG.
  MockCommand *MockCmd =
      new MockCommandWithCallback(QueueImpl, MockReq, Callback);
  (void)MockCmd->addDep(detail::DepDesc(AllocaCmd, &MockReq, nullptr),
                        ToCleanUp);
  EXPECT_TRUE(ToCleanUp.empty());
  MS.addNodeToLeaves(Record, MockCmd, access::mode::read_write, ToEnqueue);
  detail::MapOfDependentCmds DependentCmdsOfNewCmd(MockCmd->MDeps);
  MS.updateLeaves(MockCmd, {AllocaCmd}, Record, access::mode::read_write,
                  DependentCmdsOfNewCmd, QueueImpl, ToCleanUp, ToEnqueue);

  EXPECT_TRUE(ToCleanUp.empty());
  std::unique_ptr<detail::CG> CG{
      new detail::CGFill(/*Pattern*/ {}, &MockReq,
                         detail::CG::StorageInitHelper(
                             /*ArgsStorage*/ {},
                             /*AccStorage*/ {},
                             /*SharedPtrStorage*/ {},
                             /*Requirements*/ {&MockReq},
                             /*Events*/ {}))};
  detail::EventImplPtr Event =
      MS.addCG(std::move(CG), QueueImpl, /*EventNeeded=*/true);
  auto *Cmd = static_cast<detail::Command *>(Event->getCommand());
  verifyCleanup(Record, AllocaCmd, MockCmd, CommandDeleted);

  // Check add/releaseHostAccessor.
  CommandDeleted = false;
  MockCmd = new MockCommandWithCallback(QueueImpl, MockReq, Callback);
  addEdge(MockCmd, Cmd, AllocaCmd);
  MS.addNodeToLeaves(Record, MockCmd, access::mode::read_write, ToEnqueue);
  DependentCmdsOfNewCmd.addDeps(MockCmd->MDeps);
  MS.updateLeaves(MockCmd, {Cmd}, Record, access::mode::read_write,
                  DependentCmdsOfNewCmd, QueueImpl, ToCleanUp, ToEnqueue);
  MS.addHostAccessor(&MockReq);
  verifyCleanup(Record, AllocaCmd, MockCmd, CommandDeleted);

  CommandDeleted = false;
  MockCmd = new MockCommandWithCallback(QueueImpl, MockReq, Callback);
  addEdge(MockCmd, AllocaCmd, AllocaCmd);
  MockCommand *LeafMockCmd =
      new MockCommandWithCallback(QueueImpl, MockReq, Callback);
  addEdge(LeafMockCmd, MockCmd, AllocaCmd);
  MS.addNodeToLeaves(Record, LeafMockCmd, access::mode::read_write, ToEnqueue);
  MS.releaseHostAccessor(&MockReq);
  MockReq.MBlockedCmd = nullptr;
  verifyCleanup(Record, AllocaCmd, MockCmd, CommandDeleted);

  auto addNewMockCmds = [&]() -> MockCommand * {
    CommandDeleted = false;
    MockCmd = LeafMockCmd;
    LeafMockCmd = new MockCommandWithCallback(QueueImpl, MockReq, Callback);
    addEdge(LeafMockCmd, MockCmd, AllocaCmd);
    MS.addNodeToLeaves(Record, LeafMockCmd, access::mode::read_write,
                       ToEnqueue);
    // Since this mock command has already been enqueued, it's expected to be
    // cleaned up during removal from leaves.
    ToCleanUp.clear();
    detail::MapOfDependentCmds DependentCmdsOfNewCmd(LeafMockCmd->MDeps);
    MS.updateLeaves(LeafMockCmd, {MockCmd}, Record, access::mode::read_write,
                    DependentCmdsOfNewCmd, QueueImpl, ToCleanUp, ToEnqueue);
    EXPECT_EQ(ToCleanUp.size(), 1U);
    EXPECT_EQ(ToCleanUp[0], MockCmd);
    MS.cleanupCommands({MockCmd});
    verifyCleanup(Record, AllocaCmd, MockCmd, CommandDeleted);
    CommandDeleted = false;
    MockCmd = LeafMockCmd;
    LeafMockCmd = new MockCommandWithCallback(QueueImpl, MockReq, Callback);
    addEdge(LeafMockCmd, MockCmd, AllocaCmd);
    MS.addNodeToLeaves(Record, LeafMockCmd, access::mode::read_write,
                       ToEnqueue);
    DependentCmdsOfNewCmd.addDeps(LeafMockCmd->MDeps);
    MS.updateLeaves(LeafMockCmd, {MockCmd}, Record, access::mode::read_write,
                    DependentCmdsOfNewCmd, QueueImpl, ToCleanUp, ToEnqueue);
    return MockCmd;
  };

  // Check waitForEvent
  MockCmd = addNewMockCmds();
  MS.waitForEvent(LeafMockCmd->getEvent());
  verifyCleanup(Record, AllocaCmd, MockCmd, CommandDeleted);

  // Check addCopyBack
  MockCmd = addNewMockCmds();
  LeafMockCmd->getEvent()->setHandle(
      reinterpret_cast<ur_event_handle_t>(new int{}));
  MS.addCopyBack(&MockReq);
  verifyCleanup(Record, AllocaCmd, MockCmd, CommandDeleted);

  MS.removeRecordForMemObj(detail::getSyclObjImpl(Buf).get());
}

static void checkCleanupOnLeafUpdate(
    MockScheduler &MS, detail::QueueImplPtr QueueImpl, buffer<int, 1> &Buf,
    detail::Requirement &MockReq,
    std::function<void(detail::MemObjRecord *)> SchedulerCall) {
  bool CommandDeleted = false;
  std::vector<detail::Command *> ToCleanUp;
  std::vector<detail::Command *> ToEnqueue;
  detail::MemObjRecord *Record =
      MS.getOrInsertMemObjRecord(QueueImpl, &MockReq);
  detail::AllocaCommandBase *AllocaCmd =
      MS.getOrCreateAllocaForReq(Record, &MockReq, QueueImpl, ToEnqueue);
  std::function<void()> Callback = [&CommandDeleted]() {
    CommandDeleted = true;
  };

  // Add a mock command as a leaf and enqueue it.
  MockCommand *MockCmd =
      new MockCommandWithCallback(QueueImpl, MockReq, Callback);
  (void)MockCmd->addDep(detail::DepDesc(AllocaCmd, &MockReq, nullptr),
                        ToCleanUp);
  EXPECT_TRUE(ToCleanUp.empty());
  MS.addNodeToLeaves(Record, MockCmd, access::mode::read_write, ToEnqueue);
  const detail::MapOfDependentCmds DependentCmdsOfNewCmd(MockCmd->MDeps);
  MS.updateLeaves(MockCmd, {AllocaCmd}, Record, access::mode::read_write,
                  DependentCmdsOfNewCmd, QueueImpl, ToCleanUp, ToEnqueue);
  detail::EnqueueResultT Res;
  MockScheduler::enqueueCommand(MockCmd, Res, detail::BLOCKING);

  EXPECT_FALSE(CommandDeleted);
  SchedulerCall(Record);
  EXPECT_TRUE(CommandDeleted);
  MS.removeRecordForMemObj(detail::getSyclObjImpl(Buf).get());
}

TEST_F(SchedulerTest, PostEnqueueCleanup) {
  // Enforce creation of linked commands to test all sites of calling cleanup.
  unittest::ScopedEnvVar HostUnifiedMemoryVar{
      HostUnifiedMemoryName, "1",
      detail::SYCLConfig<detail::SYCL_HOST_UNIFIED_MEMORY>::reset};
  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();
  mock::getCallbacks().set_before_callback("urEnqueueMemBufferMap",
                                           &redefinedEnqueueMemBufferMap);
  mock::getCallbacks().set_before_callback("urEnqueueMemUnmap",
                                           &redefinedEnqueueMemUnmap);
  mock::getCallbacks().set_before_callback("urEnqueueMemBufferFill",
                                           &redefinedEnqueueMemBufferFill);

  context Ctx{Plt};
  queue Queue{Ctx, default_selector_v};
  detail::QueueImplPtr QueueImpl = detail::getSyclObjImpl(Queue);
  MockScheduler MS;

  buffer<int, 1> Buf{range<1>(1)};
  std::shared_ptr<detail::buffer_impl> BufImpl = detail::getSyclObjImpl(Buf);
  detail::Requirement MockReq = getMockRequirement(Buf);
  MockReq.MDims = 1;
  MockReq.MSYCLMemObj = BufImpl.get();

  checkCleanupOnEnqueue(MS, QueueImpl, Buf, MockReq);
  std::vector<detail::Command *> ToEnqueue;
  checkCleanupOnLeafUpdate(MS, QueueImpl, Buf, MockReq,
                           [&](detail::MemObjRecord *Record) {
                             MS.decrementLeafCountersForRecord(Record);
                           });
  checkCleanupOnLeafUpdate(
      MS, QueueImpl, Buf, MockReq, [&](detail::MemObjRecord *Record) {
        MS.insertMemoryMove(Record, &MockReq, QueueImpl, ToEnqueue);
      });
  checkCleanupOnLeafUpdate(MS, QueueImpl, Buf, MockReq,
                           [&](detail::MemObjRecord *Record) {
                             Record->MMemModified = true;
                             MS.addCopyBack(&MockReq, ToEnqueue);
                           });
  checkCleanupOnLeafUpdate(
      MS, QueueImpl, Buf, MockReq, [&](detail::MemObjRecord *Record) {
        detail::Command *Leaf = *Record->MWriteLeaves.begin();
        MS.addEmptyCmd(Leaf, {&MockReq}, detail::Command::BlockReason::HostTask,
                       ToEnqueue);
      });
  checkCleanupOnLeafUpdate(
      MS, nullptr, Buf, MockReq, [&](detail::MemObjRecord *Record) {
        MS.getOrCreateAllocaForReq(Record, &MockReq, QueueImpl, ToEnqueue);
      });
  // Check cleanup on exceeding leaf limit.
  checkCleanupOnLeafUpdate(
      MS, QueueImpl, Buf, MockReq, [&](detail::MemObjRecord *Record) {
        std::vector<std::unique_ptr<MockCommand>> Leaves;
        for (std::size_t I = 0;
             I < Record->MWriteLeaves.genericCommandsCapacity(); ++I)
          Leaves.push_back(std::make_unique<MockCommand>(QueueImpl, MockReq));

        detail::AllocaCommandBase *AllocaCmd = Record->MAllocaCommands[0];
        std::vector<detail::Command *> ToCleanUp;
        for (std::unique_ptr<MockCommand> &MockCmd : Leaves) {
          (void)MockCmd->addDep(detail::DepDesc(AllocaCmd, &MockReq, AllocaCmd),
                                ToCleanUp);
          MS.addNodeToLeaves(Record, MockCmd.get(), access::mode::read_write,
                             ToEnqueue);
        }
        const detail::MapOfDependentCmds DependentCmdsOfNewCmd(
            AllocaCmd->MDeps);
        for (std::unique_ptr<MockCommand> &MockCmd : Leaves)
          MS.updateLeaves(AllocaCmd, {MockCmd.get()}, Record,
                          access::mode::read_write, DependentCmdsOfNewCmd,
                          QueueImpl, ToCleanUp, ToEnqueue);
        EXPECT_TRUE(ToCleanUp.empty());
      });
}

// Check that host tasks are cleaned up after completion.
TEST_F(SchedulerTest, HostTaskCleanup) {
  unittest::UrMock<> Mock;
  platform Plt = sycl::platform();
  context Ctx{Plt};
  queue Queue{Ctx, default_selector_v};

  std::mutex Mutex;
  std::unique_lock<std::mutex> Lock{Mutex};
  event Event = Queue.submit([&](sycl::handler &cgh) {
    cgh.host_task([&]() { std::unique_lock<std::mutex> Lock{Mutex}; });
  });
  detail::EventImplPtr EventImpl = detail::getSyclObjImpl(Event);

  // Unlike other commands, host task should be kept alive until its
  // completion.
  auto *Cmd = static_cast<detail::Command *>(EventImpl->getCommand());
  ASSERT_NE(Cmd, nullptr);
  EXPECT_TRUE(Cmd->isSuccessfullyEnqueued());

  Lock.unlock();
  Event.wait();
  // The command should be sent to graph cleanup as part of the task
  // submitted to the thread pool, shortly after the event is marked
  // as complete.
  detail::GlobalHandler::instance().drainThreadPool();
  ASSERT_EQ(EventImpl->getCommand(), nullptr);
}

struct AttachSchedulerWrapper {
  AttachSchedulerWrapper(MockScheduler *MSPtr) {
    sycl::detail::GlobalHandler::instance().attachScheduler(
        static_cast<sycl::detail::Scheduler *>(MSPtr));
  }
  ~AttachSchedulerWrapper() {
    sycl::detail::GlobalHandler::instance().attachScheduler(nullptr);
  }
};

// Check that stream buffers are released alongside graph cleanup.
TEST_F(SchedulerTest, StreamBufferDeallocation) {
  unittest::UrMock<> Mock;
  platform Plt = sycl::platform();
  context Ctx{Plt};
  queue Queue{Ctx, default_selector_v};
  detail::QueueImplPtr QueueImplPtr = detail::getSyclObjImpl(Queue);

  MockScheduler *MSPtr = new MockScheduler();
  AttachSchedulerWrapper AttachScheduler{MSPtr};
  detail::EventImplPtr EventImplPtr;
  {
    MockHandlerCustomFinalize MockCGH(QueueImplPtr,
                                      /*CallerNeedsEvent=*/true);
    kernel_bundle KernelBundle =
        sycl::get_kernel_bundle<sycl::bundle_state::input>(
            QueueImplPtr->get_context());
    auto ExecBundle = sycl::build(KernelBundle);
    MockCGH.use_kernel_bundle(ExecBundle);
    stream Stream{1, 1, MockCGH};
    MockCGH.addStream(detail::getSyclObjImpl(Stream));
    MockCGH.single_task<TestKernel<>>([] {});
    std::unique_ptr<detail::CG> CG = MockCGH.finalize();

    EventImplPtr =
        MSPtr->addCG(std::move(CG), QueueImplPtr, /*EventNeeded=*/true);
  }

  // The buffers should have been released with graph cleanup once the work is
  // finished.
  EventImplPtr->wait(EventImplPtr);
  // Drain the thread pool to ensure that the cleanup is able to acquire
  // the graph lock.
  detail::GlobalHandler::instance().drainThreadPool();
  MSPtr->cleanupCommands({});
  ASSERT_EQ(MSPtr->MDeferredMemObjRelease.size(), 0u);
}

class MockAuxResource {
public:
  MockAuxResource(bool &Flag) : MFlag{Flag} {}
  ~MockAuxResource() { MFlag = true; }
  char *getDataPtr() { return &Data; };

private:
  bool &MFlag;
  char Data;
};

bool EventCompleted = false;

ur_result_t redefinedEventGetInfo(void *pParams) {
  auto params = *static_cast<ur_event_get_info_params_t *>(pParams);
  EXPECT_EQ(*params.ppropName, UR_EVENT_INFO_COMMAND_EXECUTION_STATUS)
      << "Unknown param name";
  EXPECT_EQ(*params.ppropSize, 4u);
  *(static_cast<int32_t *>(*params.ppPropValue)) =
      EventCompleted ? UR_EVENT_STATUS_COMPLETE : UR_EVENT_STATUS_SUBMITTED;
  return UR_RESULT_SUCCESS;
}

// Check that auxiliary resources are released alongside graph cleanup.
TEST_F(SchedulerTest, AuxiliaryResourcesDeallocation) {
  unittest::UrMock<> Mock;
  mock::getCallbacks().set_replace_callback("urEventGetInfo",
                                            &redefinedEventGetInfo);
  platform Plt = sycl::platform();
  context Ctx{Plt};
  queue Queue{Ctx, default_selector_v};
  detail::QueueImplPtr QueueImplPtr = detail::getSyclObjImpl(Queue);

  MockScheduler *MSPtr = new MockScheduler();
  AttachSchedulerWrapper AttachScheduler{MSPtr};
  detail::EventImplPtr EventImplPtr;
  bool MockAuxResourceDeleted = false;
  {
    MockHandlerCustomFinalize MockCGH(QueueImplPtr,
                                      /*CallerNeedsEvent=*/true);
    kernel_bundle KernelBundle =
        sycl::get_kernel_bundle<sycl::bundle_state::input>(
            QueueImplPtr->get_context());
    auto ExecBundle = sycl::build(KernelBundle);
    auto MockAuxResourcePtr =
        std::make_shared<MockAuxResource>(MockAuxResourceDeleted);
    // Create a buffer that would normally be released in a blocking manner to
    // check that we perform a deferred release instead.
    auto BufPtr = std::make_shared<buffer<char, 1>>(
        MockAuxResourcePtr->getDataPtr(), range<1>{1});
    detail::Requirement MockReq = getMockRequirement(*BufPtr);
    MSPtr->getOrInsertMemObjRecord(QueueImplPtr, &MockReq);
    MockCGH.use_kernel_bundle(ExecBundle);
    MockCGH.addReduction(std::move(MockAuxResourcePtr));
    MockCGH.addReduction(std::move(BufPtr));
    MockCGH.single_task<TestKernel<>>([] {});
    std::unique_ptr<detail::CG> CG = MockCGH.finalize();

    EventImplPtr =
        MSPtr->addCG(std::move(CG), QueueImplPtr, /*EventNeeded=*/true);
  }

  EventCompleted = false;
  MSPtr->cleanupCommands({});
  ASSERT_FALSE(MockAuxResourceDeleted);

  EventCompleted = true;
  // Acquire lock to keep deferred mem obj from releasing so that they can be
  // checked.
  {
    auto Lock = MSPtr->acquireGraphReadLock();
    MSPtr->cleanupCommands({});
    ASSERT_TRUE(MockAuxResourceDeleted);
    ASSERT_EQ(MSPtr->MDeferredMemObjRelease.size(), 1u);
  }

  MSPtr->cleanupCommands({});
  ASSERT_EQ(MSPtr->MDeferredMemObjRelease.size(), 0u);
}
} // anonymous namespace
