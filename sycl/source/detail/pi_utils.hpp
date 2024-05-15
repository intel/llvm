//==------------- pi_utils.hpp - Common PI utilities -----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <detail/plugin.hpp>
#include <sycl/detail/defines_elementary.hpp>
#include <sycl/detail/pi.hpp>

#include <optional>

namespace sycl {
inline namespace _V1 {
namespace detail {

// RAII object for keeping ownership of a PI event.
struct OwnedUrEvent {
  OwnedUrEvent(const PluginPtr &Plugin)
      : MEvent{std::nullopt}, MPlugin{Plugin} {}
  OwnedUrEvent(ur_event_handle_t Event, const PluginPtr &Plugin,
               bool TakeOwnership = false)
      : MEvent(Event), MPlugin(Plugin) {
    // If it is not instructed to take ownership, retain the event to share
    // ownership of it.
    if (!TakeOwnership)
      MPlugin->call(urEventRetain, *MEvent);
  }
  ~OwnedUrEvent() {
    // Release the event if the ownership was not transferred.
    if (MEvent.has_value())
      MPlugin->call(urEventRelease, *MEvent);
  }

  OwnedUrEvent(OwnedUrEvent &&Other)
      : MEvent(Other.MEvent), MPlugin(Other.MPlugin) {
    Other.MEvent = std::nullopt;
  }

  // Copy constructor explicitly deleted for simplicity as it is not currently
  // used. Implement if needed.
  OwnedUrEvent(const OwnedUrEvent &Other) = delete;

  operator bool() { return MEvent.has_value(); }

  ur_event_handle_t GetEvent() { return *MEvent; }

  // Transfers the ownership of the event to the caller. The destructor will
  // no longer release the event.
  ur_event_handle_t TransferOwnership() {
    ur_event_handle_t Event = *MEvent;
    MEvent = std::nullopt;
    return Event;
  }

private:
  std::optional<ur_event_handle_t> MEvent;
  const PluginPtr &MPlugin;
};

} // namespace detail
} // namespace _V1
} // namespace sycl
