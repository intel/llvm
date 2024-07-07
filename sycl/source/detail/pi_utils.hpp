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
struct OwnedPiEvent {
  OwnedPiEvent(const PluginPtr &Plugin)
      : MEvent{std::nullopt}, MPlugin{Plugin} {}
  OwnedPiEvent(sycl::detail::pi::PiEvent Event, const PluginPtr &Plugin,
               bool TakeOwnership = false)
      : MEvent(Event), MPlugin(Plugin) {
    // If it is not instructed to take ownership, retain the event to share
    // ownership of it.
    if (!TakeOwnership)
      MPlugin->call<PiApiKind::piEventRetain>(*MEvent);
  }
  ~OwnedPiEvent() {
    try {
      // Release the event if the ownership was not transferred.
      if (MEvent.has_value())
        MPlugin->call<PiApiKind::piEventRelease>(*MEvent);

    } catch (std::exception &e) {
      __SYCL_REPORT_EXCEPTION_TO_STREAM("exception in ~OwnedPiEvent", e);
    }
  }

  OwnedPiEvent(OwnedPiEvent &&Other)
      : MEvent(Other.MEvent), MPlugin(Other.MPlugin) {
    Other.MEvent = std::nullopt;
  }

  // Copy constructor explicitly deleted for simplicity as it is not currently
  // used. Implement if needed.
  OwnedPiEvent(const OwnedPiEvent &Other) = delete;

  operator bool() { return MEvent.has_value(); }

  sycl::detail::pi::PiEvent GetEvent() { return *MEvent; }

  // Transfers the ownership of the event to the caller. The destructor will
  // no longer release the event.
  sycl::detail::pi::PiEvent TransferOwnership() {
    sycl::detail::pi::PiEvent Event = *MEvent;
    MEvent = std::nullopt;
    return Event;
  }

private:
  std::optional<sycl::detail::pi::PiEvent> MEvent;
  const PluginPtr &MPlugin;
};

} // namespace detail
} // namespace _V1
} // namespace sycl
