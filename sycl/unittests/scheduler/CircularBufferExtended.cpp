//==-------- CircularBufferExtended.cpp --- Scheduler unit tests -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SchedulerTestUtils.hpp"

#include <CL/sycl.hpp>
#include <detail/scheduler/circular_buffer_extended.hpp>
#include <gtest/gtest.h>
#include <memory>

using namespace cl::sycl::detail;

class CircularBufferExtendedTest : public ::testing::Test {
protected:
  cl::sycl::async_handler MAsyncHandler =
      [](cl::sycl::exception_list ExceptionList) {
        for (cl::sycl::exception_ptr_class ExceptionPtr : ExceptionList) {
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
  return std::shared_ptr<Command>{Cmd};
}

TEST_F(CircularBufferExtendedTest, PushBack) {
  static constexpr size_t GenericCmdsCapacity = 8;

  size_t TimesGenericWasFull;

  CircularBufferExtended::AllocateDependencyF AllocateDependency =
      [&](Command *, Command *, MemObjRecord *) {
        ++TimesGenericWasFull;
      };

  // add only generic commands
  {
    CircularBufferExtended CBE = CircularBufferExtended(
        GenericCmdsCapacity, AllocateDependency);
    std::vector<std::shared_ptr<Command>> Cmds;

    TimesGenericWasFull = 0;

    for (size_t Idx = 0; Idx < GenericCmdsCapacity * 2; ++Idx) {
      Cmds.push_back(createGenericCommand(getSyclObjImpl(MQueue)));

      CBE.push_back(Cmds.back().get(), nullptr);
    }

    ASSERT_EQ(TimesGenericWasFull, GenericCmdsCapacity)
        << "IfGenericIsFull call count mismatch.";

    ASSERT_EQ(CBE.getGenericCommands().size(), GenericCmdsCapacity)
        << "Generic commands container size overflow";

    ASSERT_EQ(CBE.getHostAccessorCommands().size(), 0ul)
        << "Host accessor commands container isn't empty, but it should be.";
  }

  // add mix of generic and empty commands
  {
    cl::sycl::buffer<int, 1> Buf(cl::sycl::range<1>(1));

    Requirement MockReq = getMockRequirement(Buf);

    CircularBufferExtended CBE = CircularBufferExtended(
        GenericCmdsCapacity, AllocateDependency);
    std::vector<std::shared_ptr<Command>> Cmds;

    TimesGenericWasFull = 0;

    for (size_t Idx = 0; Idx < GenericCmdsCapacity * 4; ++Idx) {
      auto Cmd = Idx % 2 ? createGenericCommand(getSyclObjImpl(MQueue))
                         : createEmptyCommand(getSyclObjImpl(MQueue), MockReq);
      Cmds.push_back(Cmd);

      CBE.push_back(Cmds.back().get(), nullptr);
    }

    ASSERT_EQ(TimesGenericWasFull, GenericCmdsCapacity)
        << "IfGenericIsFull call count mismatch.";

    ASSERT_EQ(CBE.getGenericCommands().size(), GenericCmdsCapacity)
        << "Generic commands container size overflow";

    ASSERT_EQ(CBE.getHostAccessorCommands().size(), 2 * GenericCmdsCapacity)
        << "Host accessor commands container isn't empty, but it should be.";
  }
}

TEST_F(CircularBufferExtendedTest, Remove) {
  static constexpr size_t GenericCmdsCapacity = 8;

  CircularBufferExtended::AllocateDependencyF AllocateDependency =
      [](Command *, Command *Old, MemObjRecord *) {
        --Old->MLeafCounter;
      };

  {
    cl::sycl::buffer<int, 1> Buf(cl::sycl::range<1>(1));

    Requirement MockReq = getMockRequirement(Buf);

    CircularBufferExtended CBE = CircularBufferExtended(
        GenericCmdsCapacity, AllocateDependency);
    std::vector<std::shared_ptr<Command>> Cmds;

    for (size_t Idx = 0; Idx < GenericCmdsCapacity * 4; ++Idx) {
      auto Cmd = Idx % 2 ? createGenericCommand(getSyclObjImpl(MQueue))
                         : createEmptyCommand(getSyclObjImpl(MQueue), MockReq);
      Cmds.push_back(Cmd);
      ++Cmd->MLeafCounter;

      CBE.push_back(Cmds.back().get(), nullptr);
    }

    for (const auto &Cmd : Cmds) {
      size_t Count = CBE.remove(Cmd.get());

      ASSERT_EQ(Count, Cmd->MLeafCounter) << "Command not removed";

      Count = CBE.remove(Cmd.get());

      ASSERT_EQ(Count, 0ul) << "Command removed for the second time";
    }
  }
}
