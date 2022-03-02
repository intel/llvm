//==-------- LeavesCollection.cpp --- Scheduler unit tests -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SchedulerTestUtils.hpp"

#include <CL/sycl.hpp>
#include <detail/scheduler/leaves_collection.hpp>
#include <gtest/gtest.h>
#include <memory>

using namespace cl::sycl::detail;

class LeavesCollectionTest : public ::testing::Test {
protected:
  cl::sycl::async_handler MAsyncHandler =
      [](cl::sycl::exception_list ExceptionList) {
        for (std::exception_ptr ExceptionPtr : ExceptionList) {
          try {
            std::rethrow_exception(ExceptionPtr);
          } catch (cl::sycl::exception &E) {
            std::cerr << E.what();
          } catch (...) {
            std::cerr << "Unknown async exception was caught." << std::endl;
          }
        }
      };
  cl::sycl::queue MQueue = cl::sycl::queue(cl::sycl::device(), MAsyncHandler);
};

std::shared_ptr<Command>
createGenericCommand(const std::shared_ptr<queue_impl> &Q) {
  return std::shared_ptr<Command>{new MockCommand(Q, Command::RUN_CG)};
}

std::shared_ptr<Command>
createEmptyCommand(const std::shared_ptr<queue_impl> &Q,
                   const Requirement &Req) {
  EmptyCommand *Cmd = new EmptyCommand(Q);
  Cmd->addRequirement(/* DepCmd = */ nullptr, /* AllocaCmd = */ nullptr, &Req);
  Cmd->MBlockReason = Command::BlockReason::HostAccessor;
  Cmd->MEnqueueStatus = EnqueueResultT::SyclEnqueueBlocked;
  return std::shared_ptr<Command>{Cmd};
}

TEST_F(LeavesCollectionTest, PushBack) {
  static constexpr size_t GenericCmdsCapacity = 8;

  size_t TimesGenericWasFull;

  std::vector<cl::sycl::detail::Command *> ToEnqueue;

  LeavesCollection::AllocateDependencyF AllocateDependency =
      [&](Command *, Command *, MemObjRecord *,
          std::vector<cl::sycl::detail::Command *> &) {
        ++TimesGenericWasFull;
      };

  // add only generic commands
  {
    LeavesCollection LE =
        LeavesCollection(nullptr, GenericCmdsCapacity, AllocateDependency);
    std::vector<std::shared_ptr<Command>> Cmds;

    TimesGenericWasFull = 0;

    for (size_t Idx = 0; Idx < GenericCmdsCapacity * 2; ++Idx) {
      Cmds.push_back(createGenericCommand(getSyclObjImpl(MQueue)));

      LE.push_back(Cmds.back().get(), ToEnqueue);
    }

    ASSERT_EQ(TimesGenericWasFull, GenericCmdsCapacity)
        << "IfGenericIsFull call count mismatch.";

    ASSERT_EQ(LE.getGenericCommands().size(), GenericCmdsCapacity)
        << "Generic commands container size overflow";

    ASSERT_EQ(LE.getHostAccessorCommands().size(), 0ul)
        << "Host accessor commands container isn't empty, but it should be.";
  }

  // add mix of generic and empty commands
  {
    cl::sycl::buffer<int, 1> Buf(cl::sycl::range<1>(1));

    Requirement MockReq = getMockRequirement(Buf);

    LeavesCollection LE =
        LeavesCollection(nullptr, GenericCmdsCapacity, AllocateDependency);
    std::vector<std::shared_ptr<Command>> Cmds;

    TimesGenericWasFull = 0;

    for (size_t Idx = 0; Idx < GenericCmdsCapacity * 4; ++Idx) {
      auto Cmd = Idx % 2 ? createGenericCommand(getSyclObjImpl(MQueue))
                         : createEmptyCommand(getSyclObjImpl(MQueue), MockReq);
      Cmds.push_back(Cmd);

      LE.push_back(Cmds.back().get(), ToEnqueue);
    }

    ASSERT_EQ(TimesGenericWasFull, GenericCmdsCapacity)
        << "IfGenericIsFull call count mismatch.";

    ASSERT_EQ(LE.getGenericCommands().size(), GenericCmdsCapacity)
        << "Generic commands container size overflow";

    ASSERT_EQ(LE.getHostAccessorCommands().size(), 2 * GenericCmdsCapacity)
        << "Host accessor commands container isn't empty, but it should be.";
  }
}

TEST_F(LeavesCollectionTest, Remove) {
  static constexpr size_t GenericCmdsCapacity = 8;

  std::vector<cl::sycl::detail::Command *> ToEnqueue;

  LeavesCollection::AllocateDependencyF AllocateDependency =
      [](Command *, Command *Old, MemObjRecord *,
         std::vector<cl::sycl::detail::Command *> &) { --Old->MLeafCounter; };

  {
    cl::sycl::buffer<int, 1> Buf(cl::sycl::range<1>(1));

    Requirement MockReq = getMockRequirement(Buf);

    LeavesCollection LE =
        LeavesCollection(nullptr, GenericCmdsCapacity, AllocateDependency);
    std::vector<std::shared_ptr<Command>> Cmds;

    for (size_t Idx = 0; Idx < GenericCmdsCapacity * 4; ++Idx) {
      auto Cmd = Idx % 2 ? createGenericCommand(getSyclObjImpl(MQueue))
                         : createEmptyCommand(getSyclObjImpl(MQueue), MockReq);
      Cmds.push_back(Cmd);

      if (LE.push_back(Cmds.back().get(), ToEnqueue))
        ++Cmd->MLeafCounter;
    }

    for (const auto &Cmd : Cmds) {
      size_t Count = LE.remove(Cmd.get());

      ASSERT_EQ(Count, Cmd->MLeafCounter) << "Command not removed";

      Count = LE.remove(Cmd.get());

      ASSERT_EQ(Count, 0ul) << "Command removed for the second time";
    }
  }
}
