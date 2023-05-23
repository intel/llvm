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
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {

// RAII object for keeping ownership of a PI event.
struct OwnedPiEvent {
  OwnedPiEvent(const PluginPtr &Plugin)
      : MEvent{std::nullopt}, MPlugin{Plugin} {}
  OwnedPiEvent(RT::PiEvent Event, const PluginPtr &Plugin,
               bool TakeOwnership = false)
      : MEvent(Event), MPlugin(Plugin) {
    // If it is not instructed to take ownership, retain the event to share
    // ownership of it.
    if (!TakeOwnership)
      MPlugin->call<PiApiKind::piEventRetain>(*MEvent);
  }
  ~OwnedPiEvent() {
    // Release the event if the ownership was not transferred.
    if (MEvent.has_value())
      MPlugin->call<PiApiKind::piEventRelease>(*MEvent);
  }

  OwnedPiEvent(OwnedPiEvent &&Other)
      : MEvent(Other.MEvent), MPlugin(Other.MPlugin) {
    Other.MEvent = std::nullopt;
  }

  // Copy constructor explicitly deleted for simplicity as it is not currently
  // used. Implement if needed.
  OwnedPiEvent(const OwnedPiEvent &Other) = delete;

  operator bool() { return MEvent.has_value(); }

  RT::PiEvent GetEvent() { return *MEvent; }

  // Transfers the ownership of the event to the caller. The destructor will
  // no longer release the event.
  RT::PiEvent TransferOwnership() {
    RT::PiEvent Event = *MEvent;
    MEvent = std::nullopt;
    return Event;
  }

private:
  std::optional<RT::PiEvent> MEvent;
  const PluginPtr &MPlugin;
};

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
