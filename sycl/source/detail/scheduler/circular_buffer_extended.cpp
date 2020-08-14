//==-- circular_buffer_extended.hpp - Circular buffer with host accessor ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/scheduler/circular_buffer_extended.hpp>
#include <detail/scheduler/scheduler.hpp>

#include <algorithm>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

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
         Cmd->MBlockReason == Command::BlockReason::HostAccessor;
}

size_t CircularBufferExtended::remove(value_type Cmd) {
  if (!isHostAccessorCmd(Cmd))
  {
    auto NewEnd = std::remove(MGenericCommands.begin(),
                              MGenericCommands.end(), Cmd);
    size_t RemovedCount = std::distance(NewEnd, MGenericCommands.end());
    MGenericCommands.erase(NewEnd, MGenericCommands.end());

    return RemovedCount;
  }

  // host accessor commands part
  size_t RemovedCount = 0;

  HostAccessorIt<false> It{
      &MHostAccessorCommands, MHostAccessorCommands.begin()};
  HostAccessorIt<false> End{
      &MHostAccessorCommands, MHostAccessorCommands.end()};

  while (It != End) {
    auto Next = It;
    ++It;

    if (*It == Cmd) {
      eraseHostAccessorCommand(static_cast<EmptyCommand *>(Cmd));
      ++RemovedCount;
    }

    It = Next;
  }

  return RemovedCount;
}

void CircularBufferExtended::push_back(value_type Cmd, MemObjRecord *Record) {
  // if EmptyCommand add to HA
  if (isHostAccessorCmd(Cmd)) {
    addHostAccessorCommand(static_cast<EmptyCommand *>(Cmd), Record);
  } else
    addGenericCommand(Cmd, Record);
}

std::vector<CircularBufferExtended::value_type>
CircularBufferExtended::toVector() const {
  std::vector<value_type> Result;
  Result.reserve(MGenericCommands.size() + MHostAccessorCommandsXRef.size());

  Result.insert(Result.end(), MGenericCommands.begin(), MGenericCommands.end());

  for (const auto &It : MHostAccessorCommandsXRef)
    Result.push_back(It.first);

  return Result;
}

void CircularBufferExtended::addHostAccessorCommand(
    EmptyCommand *Cmd, MemObjRecord *Record) {
//  MemObjRecord *Record = Scheduler::getMemObjRecord(Cmd->getRequirement());

  // 1. find list of commands for the same MemObj. => List
  HostAccessorCommandListT &List = MHostAccessorCommands[Record];

  // 2. find the oldest command with doOverlap() = true amongst the List
  //      => OldCmd
  HostAccessorCommandSingleXRefT OldCmdIt;

  if (Cmd->getRequirement()->MAccessMode == cl::sycl::access::mode::read)
    OldCmdIt = List.end();
  else
    OldCmdIt = std::find_if(List.begin(), List.end(),
        [&] (const EmptyCommand * Test) -> bool {
          return doOverlap(Test->getRequirement(), Cmd->getRequirement());
        }
    );

  // 3.1  If OldCmd != null:
  //          Put a dependency in the same way as we would for generic commands
  //          when circular buffer is full.
  if (OldCmdIt != List.end()) {
    // allocate dependency
    MAllocateDependency(Cmd, *OldCmdIt, Record);

    // erase the old cmd as it's tracked via dependency now
    eraseHostAccessorCommand(static_cast<EmptyCommand *>(*OldCmdIt));
  }

  // 3.2  If OldCmd == null:
  //          Put cmd to the List
  insertHostAccessorCommand(Cmd);
}

void CircularBufferExtended::addGenericCommand(
    Command *Cmd, MemObjRecord *Record) {
  if (MGenericCommands.full())
    MIfGenericIsFull(Cmd, Record, MGenericCommands);

  MGenericCommands.push_back(Cmd);
}

void CircularBufferExtended::insertHostAccessorCommand(EmptyCommand *Cmd) {
  MemObjRecord *Record = Scheduler::getMemObjRecord(Cmd->getRequirement());

  HostAccessorCommandListT &List = MHostAccessorCommands[Record];
  MHostAccessorCommandsXRef[Cmd] = List.insert(List.end(), Cmd);
}

void CircularBufferExtended::eraseHostAccessorCommand(EmptyCommand *Cmd) {
  MemObjRecord *Record = Scheduler::getMemObjRecord(Cmd->getRequirement());

  HostAccessorCommandsT::iterator It = MHostAccessorCommands.find(Record);
  assert(It != MHostAccessorCommands.end());

  HostAccessorCommandListT &List = It->second;

  auto XRefIt = MHostAccessorCommandsXRef.find(Cmd);
  assert(XRefIt != MHostAccessorCommandsXRef.end());

  List.erase(XRefIt->second);
  MHostAccessorCommandsXRef.erase(XRefIt);

  if (List.empty()) {
    MHostAccessorCommands.erase(It);
  }
}

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
