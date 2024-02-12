//===--------- event.hpp - Level Zero Adapter -----------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cassert>
#include <list>
#include <map>
#include <mutex>
#include <optional>
#include <stdarg.h>
#include <string>
#include <unordered_map>
#include <vector>

#include <ur/ur.hpp>
#include <ur_api.h>
#include <ze_api.h>
#include <zes_api.h>

#include "common.hpp"
#include "queue.hpp"

extern "C" {
ur_result_t urEventReleaseInternal(ur_event_handle_t Event);
ur_result_t EventCreate(ur_context_handle_t Context, ur_queue_handle_t Queue,
                        bool IsMultiDevice, bool HostVisible,
                        ur_event_handle_t *RetEvent);
} // extern "C"

// This is an experimental option that allows to disable caching of events in
// the context.
const bool DisableEventsCaching = [] {
  const char *UrRet = std::getenv("UR_L0_DISABLE_EVENTS_CACHING");
  const char *PiRet = std::getenv("SYCL_PI_LEVEL_ZERO_DISABLE_EVENTS_CACHING");
  const char *DisableEventsCachingFlag =
      UrRet ? UrRet : (PiRet ? PiRet : nullptr);
  if (!DisableEventsCachingFlag)
    return false;
  return std::atoi(DisableEventsCachingFlag) != 0;
}();

// This is an experimental option that allows reset and reuse of uncompleted
// events in the in-order queue with discard_events property.
const bool ReuseDiscardedEvents = [] {
  const char *UrRet = std::getenv("UR_L0_REUSE_DISCARDED_EVENTS");
  const char *PiRet = std::getenv("SYCL_PI_LEVEL_ZERO_REUSE_DISCARDED_EVENTS");
  const char *ReuseDiscardedEventsFlag =
      UrRet ? UrRet : (PiRet ? PiRet : nullptr);
  if (!ReuseDiscardedEventsFlag)
    return true;
  return std::atoi(ReuseDiscardedEventsFlag) > 0;
}();

const bool FilterEventWaitList = [] {
  const char *Ret = std::getenv("SYCL_PI_LEVEL_ZERO_FILTER_EVENT_WAIT_LIST");
  const bool RetVal = Ret ? std::stoi(Ret) : 0;
  return RetVal;
}();

struct _ur_ze_event_list_t {
  // List of level zero events for this event list.
  ze_event_handle_t *ZeEventList = {nullptr};

  // List of ur_events for this event list.
  ur_event_handle_t *UrEventList = {nullptr};

  // length of both the lists.  The actual allocation of these lists
  // may be longer than this length.  This length is the actual number
  // of elements in the above arrays that are valid.
  uint32_t Length = {0};

  // Initialize this using the array of events in EventList, and retain
  // all the ur_event_handle_t in the created data structure.
  // CurQueue is the ur_queue_handle_t that the command with this event wait
  // list is going to be added to.  That is needed to flush command
  // batches for wait events that are in other queues.
  // UseCopyEngine indicates if the next command (the one that this
  // event wait-list is for) is going to go to copy or compute
  // queue. This is used to properly submit the dependent open
  // command-lists.
  ur_result_t createAndRetainUrZeEventList(uint32_t EventListLength,
                                           const ur_event_handle_t *EventList,
                                           ur_queue_handle_t CurQueue,
                                           bool UseCopyEngine);

  // Add all the events in this object's UrEventList to the end
  // of the list EventsToBeReleased. Destroy ur_ze_event_list_t data
  // structure fields making it look empty.
  ur_result_t collectEventsForReleaseAndDestroyUrZeEventList(
      std::list<ur_event_handle_t> &EventsToBeReleased);

  // Had to create custom assignment operator because the mutex is
  // not assignment copyable. Just field by field copy of the other
  // fields.
  _ur_ze_event_list_t &operator=(const _ur_ze_event_list_t &other) {
    if (this != &other) {
      this->ZeEventList = other.ZeEventList;
      this->UrEventList = other.UrEventList;
      this->Length = other.Length;
    }
    return *this;
  }

  // This function allows to merge two _ur_ze_event_lists
  // The ur_ze_event_list "other" is added to the caller list.
  // Note that new containers are allocated to contains the additional elements.
  // Elements are moved to the new containers.
  // other list can not be used after the call to this function.
  ur_result_t insert(_ur_ze_event_list_t &Other);

  bool isEmpty() const { return (this->ZeEventList == nullptr); }
};

void printZeEventList(const _ur_ze_event_list_t &PiZeEventList);

struct ur_event_handle_t_ : _ur_object {
  ur_event_handle_t_(ze_event_handle_t ZeEvent,
                     ze_event_pool_handle_t ZeEventPool,
                     ur_context_handle_t Context, ur_command_t CommandType,
                     bool OwnZeEvent)
      : ZeEvent{ZeEvent}, ZeEventPool{ZeEventPool}, Context{Context},
        CommandType{CommandType}, CommandData{nullptr} {
    OwnNativeHandle = OwnZeEvent;
  }

  // Level Zero event handle.
  ze_event_handle_t ZeEvent;

  // Level Zero event pool handle.
  ze_event_pool_handle_t ZeEventPool;

  // In case we use device-only events this holds their host-visible
  // counterpart. If this event is itself host-visble then HostVisibleEvent
  // points to this event. If this event is not host-visible then this field can
  // be: 1) null, meaning that a host-visible event wasn't yet created 2) a PI
  // event created internally that host will actually be redirected
  //    to wait/query instead of this PI event.
  //
  // The HostVisibleEvent is a reference counted PI event and can be used more
  // than by just this one event, depending on the mode (see EventsScope).
  //
  ur_event_handle_t HostVisibleEvent = {nullptr};
  bool isHostVisible() const {
    return this ==
           const_cast<const ur_event_handle_t_ *>(
               reinterpret_cast<ur_event_handle_t_ *>(HostVisibleEvent));
  }

  // Provide direct access to Context, instead of going via queue.
  // Not every PI event has a queue, and we need a handle to Context
  // to get to event pool related information.
  ur_context_handle_t Context;

  // Keeps the command-queue and command associated with the event.
  // These are NULL for the user events.
  ur_queue_handle_t UrQueue = {nullptr};
  ur_command_t CommandType;

  // Opaque data to hold any data needed for CommandType.
  void *CommandData;

  // Command list associated with the ur_event_handle_t
  std::optional<ur_command_list_ptr_t> CommandList;

  // List of events that were in the wait list of the command that will
  // signal this event.  These events must be retained when the command is
  // enqueued, and must then be released when this event has signalled.
  // This list must be destroyed once the event has signalled.
  _ur_ze_event_list_t WaitList;

  // Tracks if the needed cleanup was already performed for
  // a completed event. This allows to control that some cleanup
  // actions are performed only once.
  //
  bool CleanedUp = {false};

  // Indicates that this PI event had already completed in the sense
  // that no other synchromization is needed. Note that the underlying
  // L0 event (if any) is not guranteed to have been signalled, or
  // being visible to the host at all.
  bool Completed = {false};

  // Indicates that this event is discarded, i.e. it is not visible outside of
  // plugin.
  bool IsDiscarded = {false};

  // Indicates that this event is needed to be visible by multiple devices.
  // When possible, allocate Event from single device pool for optimal
  // performance
  bool IsMultiDevice = {false};

  // Besides each PI object keeping a total reference count in
  // _ur_object::RefCount we keep special track of the event *external*
  // references. This way we are able to tell when the event is not referenced
  // externally anymore, i.e. it can't be passed as a dependency event to
  // piEnqueue* functions and explicitly waited meaning that we can do some
  // optimizations:
  // 1. For in-order queues we can reset and reuse event even if it was not yet
  // completed by submitting a reset command to the queue (since there are no
  // external references, we know that nobody can wait this event somewhere in
  // parallel thread or pass it as a dependency which may lead to hang)
  // 2. We can avoid creating host proxy event.
  // This counter doesn't track the lifetime of an event object. Even if it
  // reaches zero an event object may not be destroyed and can be used
  // internally in the plugin.
  std::atomic<uint32_t> RefCountExternal{0};

  bool hasExternalRefs() { return RefCountExternal != 0; }

  // Reset ur_event_handle_t object.
  ur_result_t reset();

  // Tells if this event is with profiling capabilities.
  bool isProfilingEnabled() const;

  // Get the host-visible event or create one and enqueue its signal.
  ur_result_t getOrCreateHostVisibleEvent(ze_event_handle_t &HostVisibleEvent);
};

// Helper function to implement zeHostSynchronize.
// The behavior is to avoid infinite wait during host sync under ZE_DEBUG.
// This allows for a much more responsive debugging of hangs.
//
template <typename T, typename Func>
ze_result_t zeHostSynchronizeImpl(Func Api, T Handle);

// Template function to do various types of host synchronizations.
// This is intended to be used instead of direct calls to specific
// Level-Zero synchronization APIs.
//
template <typename T> ze_result_t zeHostSynchronize(T Handle);
template <> ze_result_t zeHostSynchronize(ze_event_handle_t Handle);
template <> ze_result_t zeHostSynchronize(ze_command_queue_handle_t Handle);

// Perform any necessary cleanup after an event has been signalled.
// This currently makes sure to release any kernel that may have been used by
// the event, updates the last command event in the queue and cleans up all dep
// events of the event.
// If the caller locks queue mutex then it must pass 'true' to QueueLocked.
ur_result_t CleanupCompletedEvent(ur_event_handle_t Event,
                                  bool QueueLocked = false,
                                  bool SetEventCompleted = false);

// Get value of device scope events env var setting or default setting
static const EventsScope DeviceEventsSetting = [] {
  char *UrRet = std::getenv("UR_L0_DEVICE_SCOPE_EVENTS");
  char *PiRet = std::getenv("SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS");
  const char *DeviceEventsSettingStr =
      UrRet ? UrRet : (PiRet ? PiRet : nullptr);
  if (DeviceEventsSettingStr) {
    // Override the default if user has explicitly chosen the events scope.
    switch (std::stoi(DeviceEventsSettingStr)) {
    case 0:
      return AllHostVisible;
    case 1:
      return OnDemandHostVisibleProxy;
    case 2:
      return LastCommandInBatchHostVisible;
    default:
      // fallthrough to default setting
      break;
    }
  }
  // This is our default setting, which is expected to be the fastest
  // with the modern GPU drivers.
  return AllHostVisible;
}();
