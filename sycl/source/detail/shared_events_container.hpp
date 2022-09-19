//==--------- shared_events_container.hpp --- shared_events_container ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/event.hpp>

#include <vector>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {

/// Helper class to manage events tracked by queue.
///
/// This class is a wrapper around vector of events. When threshold of 128 is
/// reached we start to work with underlying vector as with ring buffer. We
/// track number of available slots in the ring buffer and if there is no any
/// free slot then we start cleanup process. We cleanup in a circular way, i.e.
/// we check the status of older events first. During cleanup we don't erase
/// elements but only release ownership to avoid overhead from destroy and
/// push_back when adding new event. If there is an available slot then we just
/// put event there. If ring is full then we push_back. As soon as whole ring is
/// cleaned up we extend the ring size to the overflow area.
class shared_events_container {
public:
  void push(const event &Event) {
    // Grow till threshold.
    if (MEventsShared.size() < EventThreshold) {
      MEventsShared.push_back(Event);
      return;
    }

    // We reached threshold.
    // If there are no slots, start cleanup process where we release ownership
    // of events but don't remove elements of the vector.
    if (NumAvailableSlots == 0) {
      // Remember the first index
      AvailableIndex = CleanupStartIndex;
      for (size_t I = 0; I < RingSize; I++) {
        // We start cleanup process from the index we stopped last time.
        size_t Index = (I + CleanupStartIndex) % RingSize;
        if (MEventsShared[Index]
                .get_info<info::event::command_execution_status>() ==
            info::event_command_status::complete) {
          // Release ownership of the event, so that it can get released.
          getSyclObjImpl(MEventsShared[Index]).reset();
          NumAvailableSlots++;
        } else {
          // During next cleanup we need to start from this index because it is
          // the first uncompleted event.
          CleanupStartIndex = Index;
          break;
        }
      }
      // If we consumed everything then increase ring size to full capacity.
      if (NumAvailableSlots == RingSize) {
        AvailableIndex = 0;
        if (RingSize < MEventsShared.size()) {
          CleanupStartIndex = RingSize;
          RingSize = MEventsShared.size();
        }
      }
      if (NumAvailableSlots > 0) {
        MEventsShared[AvailableIndex % RingSize] = Event;
        AvailableIndex++;
        NumAvailableSlots--;
      } else {
        // If no available slots after cleanup, then push beyond the ring.
        MEventsShared.push_back(Event);
      }
    } else {
      MEventsShared[AvailableIndex % RingSize] = Event;
      AvailableIndex++;
      NumAvailableSlots--;
    }
  }

  // Get list of uncompleted events.
  std::vector<event> get_event_list() const {
    if (MEventsShared.size() < EventThreshold) {
      return MEventsShared;
    }

    size_t Size =
        (RingSize - NumAvailableSlots) + (MEventsShared.size() - RingSize);
    if (Size > 0) {
      std::vector<event> Events(Size);

      size_t I = 0;
      for (; I < (RingSize - NumAvailableSlots); I++) {
        size_t Index = (I + CleanupStartIndex) % RingSize;
        Events[I] = MEventsShared[Index];
      }

      for (size_t Index = RingSize; Index < (MEventsShared.size() - RingSize);
           Index++, I++) {
        Events[I] = MEventsShared[Index];
      }
      return Events;
    } else {
      return {};
    }
  }

private:
  std::vector<event> MEventsShared;

  const size_t EventThreshold = 128;
  size_t NumAvailableSlots = 0;     // Number of available slots
  size_t AvailableIndex = 0;        // Index of the available slot
  size_t CleanupStartIndex = 0;     // Index to start cleanup process
  size_t RingSize = EventThreshold; // Size of the ring
};

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
