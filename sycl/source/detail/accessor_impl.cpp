//==---------------- accessor_impl.cpp - SYCL standard source file ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/accessor_impl.hpp>
#include <detail/event_impl.hpp>
#include <detail/scheduler/scheduler.hpp>

#include <algorithm>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

AccessorImplHost::~AccessorImplHost() {
  try {
    std::set<const Command *> BlockedCmds;
    size_t Count =
        countBlockedCommand([&BlockedCmds](const Command *const Cmd) {
          if (Cmd->MBlockReason == Command::BlockReason::HostAccessor) {
            BlockedCmds.insert(Cmd);
            return true;
          }

          return false;
        });

    for (const Command *Cmd : BlockedCmds)
      if (EventImplPtr Event = Cmd->getEvent())
        if (Event->is_host())
          Event->setComplete();

    if (Count)
      detail::Scheduler::getInstance().releaseHostAccessor(this);
  } catch (...) {
  }
}

void AccessorImplHost::addBlockedCommand(Command *BlockedCmd) {
  std::lock_guard<std::mutex> Lock(MBlockedCmdsMutex);

  MBlockedCmds.insert(BlockedCmd);
}

size_t AccessorImplHost::countBlockedCommand(const CheckCmdFn &Check) {
  std::lock_guard<std::mutex> Lock(MBlockedCmdsMutex);

  return std::count_if(MBlockedCmds.begin(), MBlockedCmds.end(), Check);
}

Command *AccessorImplHost::findBlockedCommand(const CheckCmdFn &Check) {
  std::lock_guard<std::mutex> Lock(MBlockedCmdsMutex);

  auto FoundIt = std::find_if(MBlockedCmds.begin(), MBlockedCmds.end(), Check);

  return FoundIt == MBlockedCmds.end() ? nullptr : *FoundIt;
}

bool AccessorImplHost::removeBlockedCommand(Command *BlockedCmd) {
  std::lock_guard<std::mutex> Lock(MBlockedCmdsMutex);

  MBlockedCmds.erase(BlockedCmd);

  return MBlockedCmds.empty();
}

void addHostAccessorAndWait(Requirement *Req) {
  detail::EventImplPtr Event =
      detail::Scheduler::getInstance().addHostAccessor(Req);
  Event->wait(Event);
}
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
