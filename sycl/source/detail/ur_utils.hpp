//==------------- ur_utils.hpp - Common UR utilities -----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <detail/adapter.hpp>
#include <detail/compiler.hpp>
#include <sycl/detail/defines_elementary.hpp>
#include <sycl/detail/ur.hpp>

#include <optional>

namespace sycl {
inline namespace _V1 {
namespace detail {

// RAII object for keeping ownership of a UR event.
struct OwnedUrEvent {
  OwnedUrEvent(const AdapterPtr &Adapter)
      : MEvent{std::nullopt}, MAdapter{Adapter} {}
  OwnedUrEvent(ur_event_handle_t Event, const AdapterPtr &Adapter,
               bool TakeOwnership = false)
      : MEvent(Event), MAdapter(Adapter) {
    // If it is not instructed to take ownership, retain the event to share
    // ownership of it.
    if (!TakeOwnership)
      MAdapter->call<UrApiKind::urEventRetain>(*MEvent);
  }
  ~OwnedUrEvent() {
    try {
      // Release the event if the ownership was not transferred.
      if (MEvent.has_value())
        MAdapter->call<UrApiKind::urEventRelease>(*MEvent);

    } catch (std::exception &e) {
      __SYCL_REPORT_EXCEPTION_TO_STREAM("exception in ~OwnedUrEvent", e);
    }
  }

  OwnedUrEvent(OwnedUrEvent &&Other)
      : MEvent(Other.MEvent), MAdapter(Other.MAdapter) {
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
  const AdapterPtr &MAdapter;
};

namespace ur {
using DeviceBinaryType = ::sycl_device_binary_type;

/// Tries to determine the device binary image foramat. Returns
/// SYCL_DEVICE_BINARY_TYPE_NONE if unsuccessful.
DeviceBinaryType getBinaryImageFormat(const unsigned char *ImgData,
                                      size_t ImgSize);
} // namespace ur

} // namespace detail
} // namespace _V1
} // namespace sycl
