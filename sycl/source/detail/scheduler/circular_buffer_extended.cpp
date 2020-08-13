//==-- circular_buffer_extended.hpp - Circular buffer with host accessor ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/scheduler/circular_buffer_extended.hpp>
#include <detail/scheduler/scheduler.hpp>

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

void CircularBufferExtended::push_back(value_type Cmd) {
  // if EmptyCommand add to HA
  if (Cmd->getType() == Command::EMPTY_TASK &&
      Cmd->MBlockReason == Command::BlockReason::HostAccessor) {
    addHostAccessorCommand(static_cast<EmptyCommand *>(Cmd));
  } else
    addGenericCommand(Cmd);
}

void CircularBufferExtended::addHostAccessorCommand(EmptyCommand *Cmd) {
  // TODO
}

void CircularBufferExtended::addGenericCommand(Command *Cmd) {
  if (MGenericCommands.full())
    MIfGenericIsFull(Cmd, MGenericCommands);

  MGenericCommands.push_back(Cmd);
}

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
