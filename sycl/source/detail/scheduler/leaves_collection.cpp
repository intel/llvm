//==---- leaves_collection.hpp - Container for leaves of execution graph ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/scheduler/leaves_collection.hpp>
#include <detail/scheduler/scheduler.hpp>

#include <algorithm>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

// TODO merge with GraphBuilder's version of doOverlap (see graph_builder.cpp).
static inline bool doOverlap(const Requirement *LHS, const Requirement *RHS) {
  size_t LHSStart = LHS->MOffsetInBytes;
  size_t LHSEnd = LHSStart + LHS->MAccessRange.size() * LHS->MElemSize;

  size_t RHSStart = RHS->MOffsetInBytes;
  size_t RHSEnd = RHSStart + RHS->MAccessRange.size() * RHS->MElemSize;

  if (LHSStart < RHSStart) {
    return (RHSStart < LHSEnd) && (LHSEnd <= RHSEnd);
  } else {
    return (LHSStart < RHSEnd) && (RHSEnd <= LHSEnd);
  }
}

static inline bool isHostAccessorCmd(Command *Cmd) {
  return Cmd->getType() == Command::EMPTY_TASK &&
         Cmd->MEnqueueStatus == EnqueueResultT::SyclEnqueueBlocked &&
         Cmd->MBlockReason == Command::BlockReason::HostAccessor;
}

size_t LeavesCollection::remove(value_type Cmd) {
  if (!isHostAccessorCmd(Cmd)) {
    auto NewEnd =
        std::remove(MGenericCommands.begin(), MGenericCommands.end(), Cmd);
    size_t RemovedCount = std::distance(NewEnd, MGenericCommands.end());
    MGenericCommands.erase(NewEnd, MGenericCommands.end());

    return RemovedCount;
  }

  // host accessor commands part
  return eraseHostAccessorCommand(static_cast<EmptyCommand *>(Cmd));
}

bool LeavesCollection::push_back(value_type Cmd, EnqueueListT &ToEnqueue) {
  bool Result = false;

  if (isHostAccessorCmd(Cmd))
    Result =
        addHostAccessorCommand(static_cast<EmptyCommand *>(Cmd), ToEnqueue);
  else
    Result = addGenericCommand(Cmd, ToEnqueue);

  return Result;
}

std::vector<LeavesCollection::value_type> LeavesCollection::toVector() const {
  std::vector<value_type> Result;
  Result.reserve(MGenericCommands.size() + MHostAccessorCommands.size());

  Result.insert(Result.end(), MGenericCommands.begin(), MGenericCommands.end());

  for (EmptyCommand *Cmd : MHostAccessorCommands)
    Result.push_back(Cmd);

  return Result;
}

bool LeavesCollection::addHostAccessorCommand(EmptyCommand *Cmd,
                                              EnqueueListT &ToEnqueue) {
  // 1. find the oldest command with doOverlap() = true amongst the List
  //      => OldCmd
  HostAccessorCommandSingleXRefT OldCmdIt;

  // HACK we believe here that read accessors never overlap as it doesn't add
  // any real dependency (e.g. data copy to device) except for blocking.
  if (Cmd->getRequirement()->MAccessMode == cl::sycl::access::mode::read)
    OldCmdIt = MHostAccessorCommands.end();
  else
    OldCmdIt = std::find_if(
        MHostAccessorCommands.begin(), MHostAccessorCommands.end(),
        [&](const EmptyCommand *Test) -> bool {
          return doOverlap(Test->getRequirement(), Cmd->getRequirement());
        });

  // FIXME this 'if' is a workaround for duplicate leaves, remove once fixed
  if (OldCmdIt != MHostAccessorCommands.end() && *OldCmdIt == Cmd)
    return false;

  // 2.1  If OldCmd != null:
  //          Put a dependency in the same way as we would for generic commands
  //          when circular buffer is full.
  if (OldCmdIt != MHostAccessorCommands.end()) {
    // allocate dependency
    MAllocateDependency(Cmd, *OldCmdIt, MRecord, ToEnqueue);

    // erase the old cmd as it's tracked via dependency now
    eraseHostAccessorCommand(static_cast<EmptyCommand *>(*OldCmdIt));
  }

  // 2.2  If OldCmd == null:
  //          Put cmd to the List
  insertHostAccessorCommand(Cmd);
  return true;
}

bool LeavesCollection::addGenericCommand(Command *Cmd,
                                         EnqueueListT &ToEnqueue) {
  if (MGenericCommands.full()) {
    Command *OldLeaf = MGenericCommands.front();

    // FIXME this 'if' is a workaround for duplicate leaves, remove once fixed
    if (OldLeaf == Cmd)
      return false;

    MAllocateDependency(Cmd, OldLeaf, MRecord, ToEnqueue);
  }

  MGenericCommands.push_back(Cmd);

  return true;
}

void LeavesCollection::insertHostAccessorCommand(EmptyCommand *Cmd) {
  MHostAccessorCommandsXRef[Cmd] =
      MHostAccessorCommands.insert(MHostAccessorCommands.end(), Cmd);
}

size_t LeavesCollection::eraseHostAccessorCommand(EmptyCommand *Cmd) {
  auto XRefIt = MHostAccessorCommandsXRef.find(Cmd);

  if (XRefIt == MHostAccessorCommandsXRef.end())
    return 0;

  MHostAccessorCommands.erase(XRefIt->second);
  MHostAccessorCommandsXRef.erase(XRefIt);
  return 1;
}

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
