//==-------- LeavesCollection.cpp --- Scheduler unit tests -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SchedulerTestUtils.hpp"

#include <detail/scheduler/leaves_collection.hpp>
#include <gtest/gtest.h>
#include <helpers/PiMock.hpp>
#include <memory>
#include <sycl/sycl.hpp>

using namespace sycl::detail;

class LeavesCollectionTest : public ::testing::Test {
protected:
  sycl::async_handler MAsyncHandler = [](sycl::exception_list ExceptionList) {
    for (std::exception_ptr ExceptionPtr : ExceptionList) {
      try {
        std::rethrow_exception(ExceptionPtr);
      } catch (sycl::exception &E) {
        std::cerr << E.what();
      } catch (...) {
        std::cerr << "Unknown async exception was caught." << std::endl;
      }
    }
  };
};

std::shared_ptr<Command>
createGenericCommand(const std::shared_ptr<queue_impl> &Q) {
  return std::shared_ptr<Command>{new MockCommand(Q, Command::RUN_CG)};
}

std::shared_ptr<Command> createEmptyCommand(const Requirement &Req) {
  EmptyCommand *Cmd = new EmptyCommand();
  Cmd->addRequirement(/* DepCmd = */ nullptr, /* AllocaCmd = */ nullptr, &Req);
  Cmd->MBlockReason = Command::BlockReason::HostAccessor;
  return std::shared_ptr<Command>{Cmd};
}

TEST_F(LeavesCollectionTest, PushBack) {
  sycl::unittest::PiMock Mock;
  sycl::queue Q{Mock.getPlatform().get_devices()[0], MAsyncHandler};

  static constexpr size_t GenericCmdsCapacity = 8;

  size_t TimesGenericWasFull;

  std::vector<sycl::detail::Command *> ToEnqueue;

  LeavesCollection::AllocateDependencyF AllocateDependency =
      [&](Command *, Command *, MemObjRecord *,
          std::vector<sycl::detail::Command *> &) { ++TimesGenericWasFull; };

  // add only generic commands
  {
    LeavesCollection LE =
        LeavesCollection(nullptr, GenericCmdsCapacity, AllocateDependency);
    std::vector<std::shared_ptr<Command>> Cmds;

    TimesGenericWasFull = 0;

    for (size_t Idx = 0; Idx < GenericCmdsCapacity * 2; ++Idx) {
      Cmds.push_back(createGenericCommand(getSyclObjImpl(Q)));

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
    sycl::buffer<int, 1> Buf(sycl::range<1>(1));

    Requirement MockReq = getMockRequirement(Buf);

    LeavesCollection LE =
        LeavesCollection(nullptr, GenericCmdsCapacity, AllocateDependency);
    std::vector<std::shared_ptr<Command>> Cmds;

    TimesGenericWasFull = 0;

    for (size_t Idx = 0; Idx < GenericCmdsCapacity * 4; ++Idx) {
      auto Cmd = Idx % 2 ? createGenericCommand(getSyclObjImpl(Q))
                         : createEmptyCommand(MockReq);
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
  sycl::unittest::PiMock Mock;
  sycl::queue Q{Mock.getPlatform().get_devices()[0], MAsyncHandler};

  static constexpr size_t GenericCmdsCapacity = 8;

  std::vector<sycl::detail::Command *> ToEnqueue;

  LeavesCollection::AllocateDependencyF AllocateDependency =
      [](Command *, Command *Old, MemObjRecord *,
         std::vector<sycl::detail::Command *> &) { --Old->MLeafCounter; };

  {
    sycl::buffer<int, 1> Buf(sycl::range<1>(1));

    Requirement MockReq = getMockRequirement(Buf);

    LeavesCollection LE =
        LeavesCollection(nullptr, GenericCmdsCapacity, AllocateDependency);
    std::vector<std::shared_ptr<Command>> Cmds;

    for (size_t Idx = 0; Idx < GenericCmdsCapacity * 4; ++Idx) {
      auto Cmd = Idx % 2 ? createGenericCommand(getSyclObjImpl(Q))
                         : createEmptyCommand(MockReq);
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
