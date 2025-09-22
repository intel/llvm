//==---------- xpti_registry.hpp ----- XPTI Stream Registry ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mutex>
#include <string>
#include <unordered_set>

#include <sycl/detail/common.hpp>
#include <sycl/version.hpp>

#ifdef XPTI_ENABLE_INSTRUMENTATION
// Include the headers necessary for emitting
// traces using the trace framework
#include "xpti/xpti_trace_framework.hpp"
#endif

namespace sycl {
inline namespace _V1 {
namespace detail {

#ifdef XPTI_ENABLE_INSTRUMENTATION
#define STR(x) #x
#define TO_STRING(x) STR(x)
#define SYCL_VERSION_STR                                                       \
  "sycl " TO_STRING(__LIBSYCL_MAJOR_VERSION) "." TO_STRING(                    \
      __LIBSYCL_MINOR_VERSION)

/// Constants being used as placeholder until one is able to reliably get the
/// version of the SYCL runtime
constexpr uint32_t GMajVer = __LIBSYCL_MAJOR_VERSION;
constexpr uint32_t GMinVer = __LIBSYCL_MINOR_VERSION;
constexpr const char *GVerStr = SYCL_VERSION_STR;

/// We define all the streams used the instrumentation framework here
inline constexpr const char *SYCL_STREAM_NAME = "sycl";
// We will use "sycl.debug" stream name as an indicator of needing debugging
// information; in this case, the tool will have to subscribe to the sycl.debug
// stream to get additional debug metadata, but the metadata will still be sent
// through the regular streams.
inline constexpr const char *SYCL_DEBUG_STREAM_NAME = "sycl.debug";
inline constexpr auto SYCL_MEM_ALLOC_STREAM_NAME =
    "sycl.experimental.mem_alloc";
// Stream name being used to notify about buffer objects.
inline constexpr const char *SYCL_BUFFER_STREAM_NAME =
    "sycl.experimental.buffer";
// Stream name being used to notify about image objects.
inline constexpr const char *SYCL_IMAGE_STREAM_NAME = "sycl.experimental.image";
inline constexpr const char *UR_API_STREAM_NAME = "ur.api";

extern uint8_t GBufferStreamID;
extern uint8_t GImageStreamID;
extern uint8_t GMemAllocStreamID;
extern uint8_t GSYCLStreamID;
extern uint8_t GSYCLDebugStreamID;
extern uint8_t GUrApiStreamID;

extern xpti::trace_event_data_t *GMemAllocEvent;
extern xpti::trace_event_data_t *GSYCLGraphEvent;
extern xpti::trace_event_data_t *GSYCLCallEvent;
extern xpti::trace_event_data_t *GApiEvent;

// We will pick a global constant so that the pointer in TLS never goes stale
inline constexpr auto XPTI_QUEUE_INSTANCE_ID_KEY = "queue_id";

// Helper to check if xpti stream is debug.
inline bool isDebugStream(xpti::stream_id_t StreamID) {
  return StreamID == detail::GSYCLDebugStreamID;
}

inline uint8_t getActiveXPTIStreamID() {
  return xptiCheckTraceEnabled(detail::GSYCLDebugStreamID)
             ? detail::GSYCLDebugStreamID
             : detail::GSYCLStreamID;
}

inline bool anyTraceEnabled(uint16_t TraceType) {
  return xptiCheckTraceEnabled(detail::GSYCLDebugStreamID, TraceType) ||
         xptiCheckTraceEnabled(detail::GSYCLStreamID, TraceType);
}
#endif

class XPTIRegistry {
public:
  void initializeFrameworkOnce() {
#ifdef XPTI_ENABLE_INSTRUMENTATION
    std::call_once(MInitialized, [this] {
      if (!xptiTraceEnabled())
        // If tracing is not enabled, do not initialize the framework
        return;

      // Initialize the XPTI framework
      xptiFrameworkInitialize();
      // Register the streams that we will be using
      // SYCL events
      detail::GSYCLStreamID =
          this->initializeStream(SYCL_STREAM_NAME, GMajVer, GMinVer, GVerStr);
      // Register the SYCL Debug event stream; tools subscribing to this stream
      // will receive additional metadata in the regular "sycl" stream.
      detail::GSYCLDebugStreamID = this->initializeStream(
          SYCL_DEBUG_STREAM_NAME, GMajVer, GMinVer, GVerStr);
      // SYCL buffer events
      detail::GBufferStreamID = this->initializeStream(
          SYCL_BUFFER_STREAM_NAME, GMajVer, GMinVer, GVerStr);
      // SYCL image events
      detail::GImageStreamID = this->initializeStream(
          SYCL_IMAGE_STREAM_NAME, GMajVer, GMinVer, GVerStr);
      // Memory allocation events
      detail::GMemAllocStreamID = this->initializeStream(
          SYCL_MEM_ALLOC_STREAM_NAME, GMajVer, GMinVer, GVerStr);
      // UR API events
      detail::GUrApiStreamID =
          this->initializeStream(UR_API_STREAM_NAME, GMajVer, GMinVer, GVerStr);

      auto SYCLEventTP = xptiCreateTracepoint("sycl.application.graph", nullptr,
                                              0, 0, nullptr);
      detail::GSYCLGraphEvent = SYCLEventTP->event_ref();
      if (detail::GSYCLGraphEvent) {
        // The graph event is a global event and will be used as the parent for
        // all nodes (command groups, memory allocations, etc)
        xptiNotifySubscribers(detail::getActiveXPTIStreamID(),
                              xpti::trace_graph_create, nullptr,
                              detail::GSYCLGraphEvent,
                              detail::GSYCLGraphEvent->instance_id, nullptr);
      }
      auto MemAllocEventTP =
          xptiCreateTracepoint("sycl.memory.alloc", nullptr, 0, 0, nullptr);
      detail::GMemAllocEvent = MemAllocEventTP->event_ref();

      // We capture all API calls in a single event, so that we can minimize
      // XPTI infra calls
      auto APIEventTP =
          xptiCreateTracepoint("api.function", nullptr, 0, 0, nullptr);
      detail::GApiEvent = APIEventTP->event_ref();

      auto SYCLExceptionsTP =
          xptiCreateTracepoint("sycl.exceptions", nullptr, 0, 0, nullptr);
      detail::GSYCLCallEvent = SYCLExceptionsTP->event_ref();
    });
#endif
  }

  /// Registers and notifies XPTI subscribers about new stream.
  ///
  /// \param StreamName is a name of newly initialized stream.
  /// \param MajVer is a stream major version.
  /// \param MinVer is a stream minor version.
  /// \param VerStr is a string of "MajVer.MinVer" format.
  uint8_t initializeStream(const std::string &StreamName, uint32_t MajVer,
                           uint32_t MinVer, const std::string &VerStr) {
    // We need to return an invalid ID if XPTI is not enabled
    uint8_t StreamID = std::numeric_limits<uint8_t>::max();
#ifdef XPTI_ENABLE_INSTRUMENTATION
    StreamID = xptiRegisterStream(StreamName.c_str());
    MActiveStreams.insert(StreamName);
    xptiInitialize(StreamName.c_str(), MajVer, MinVer, VerStr.c_str());
#endif // XPTI_ENABLE_INSTRUMENTATION
    return StreamID;
  }

  ~XPTIRegistry() {
#ifdef XPTI_ENABLE_INSTRUMENTATION
    for (const auto &StreamName : MActiveStreams) {
      xptiFinalize(StreamName.c_str());
    }
    xptiFrameworkFinalize();
#endif // XPTI_ENABLE_INSTRUMENTATION
  }

  static void bufferConstructorNotification(const void *,
                                            const detail::code_location &,
                                            const void *, const void *,
                                            uint32_t, uint32_t, size_t[3]);
  static void bufferAssociateNotification(const void *, const void *);
  static void bufferReleaseNotification(const void *, const void *);
  static void bufferDestructorNotification(const void *);
  static void bufferAccessorNotification(const void *, const void *, uint32_t,
                                         uint32_t,
                                         const detail::code_location &);

  static void sampledImageConstructorNotification(const void *,
                                                  const detail::code_location &,
                                                  const void *, uint32_t,
                                                  size_t[3], uint32_t, uint32_t,
                                                  uint32_t, uint32_t);
  static void sampledImageDestructorNotification(const void *);

  static void unsampledImageConstructorNotification(
      const void *, const detail::code_location &, const void *, uint32_t,
      size_t[3], uint32_t);
  static void unsampledImageDestructorNotification(const void *);

  static void unsampledImageAccessorNotification(const void *, const void *,
                                                 uint32_t, uint32_t,
                                                 const void *, uint32_t,
                                                 const detail::code_location &);
  static void
  unsampledImageHostAccessorNotification(const void *, const void *, uint32_t,
                                         const void *, uint32_t,
                                         const detail::code_location &);
  static void sampledImageAccessorNotification(const void *, const void *,
                                               uint32_t, const void *, uint32_t,
                                               const detail::code_location &);
  static void
  sampledImageHostAccessorNotification(const void *, const void *, const void *,
                                       uint32_t, const detail::code_location &);

private:
  std::unordered_set<std::string> MActiveStreams;
  std::once_flag MInitialized;

#ifdef XPTI_ENABLE_INSTRUMENTATION
  static xpti::trace_event_data_t *
  createTraceEvent(const void *Obj, const void *ObjName, uint64_t &IId,
                   const detail::code_location &CodeLoc,
                   uint16_t TraceEventType);
#endif // XPTI_ENABLE_INSTRUMENTATION
};

/// @brief Helper class to enable XPTI implementation
/// @details This class simplifies the instrumentation and encapsulates the
/// verbose call sequences
#if XPTI_ENABLE_INSTRUMENTATION
class XPTIScope {
public:
  using TracePoint = xpti::framework::tracepoint_t;
  /// @brief Scoped class for XPTI instrumentation using TLS data
  /// @param CodePtr  The address of the class/function to help differentiate
  /// actions in case the code location information is not available
  /// @param TraceType The type of trace event being created
  /// @param StreamName  The stream which will emit these notifications
  /// @param InstanceID The instance ID associated with an object, otherwise 0
  /// will auto-generate
  /// @param UserData String value that provides metadata about the
  /// instrumentation
  XPTIScope(void *CodePtr, uint16_t TraceType, const char *StreamName,
            uint64_t InstanceID, const char *UserData)
      : MUserData(UserData), MStreamID(0), MInstanceID(InstanceID),
        MScopedNotify(false), MTraceType(0) {
    detail::tls_code_loc_t Tls;
    auto TData = Tls.query();
    // If TLS is not set, we can still genertate universal IDs with user data
    // and CodePtr information
    const char *FuncName = TData.functionName();
    if (!TData.functionName() && !TData.fileName())
      FuncName = UserData;
    // Create a tracepoint object that has a lifetime of this class
    MTP = new TracePoint(TData.fileName(), FuncName, TData.lineNumber(),
                         TData.columnNumber(), CodePtr);
    if (TraceType == (uint16_t)xpti::trace_point_type_t::graph_create ||
        TraceType == (uint16_t)xpti::trace_point_type_t::node_create ||
        TraceType == (uint16_t)xpti::trace_point_type_t::edge_create ||
        TraceType == (uint16_t)xpti::trace_point_type_t::queue_create)
      MTP->parent_event(GSYCLGraphEvent);
    // Now if tracing is enabled, create trace events and notify
    if (xptiTraceEnabled() && MTP) {
      MTP->stream(StreamName).trace_type((xpti::trace_point_type_t)TraceType);
      MTraceEvent = const_cast<xpti::trace_event_data_t *>(MTP->trace_event());
      MStreamID = MTP->stream_id();
      // This constructor uses a manual override for the instance ID as some
      // objects such as queues keep track of instance IDs
      MTP->override_instance_id(MInstanceID);
    }
  }

  /// @brief Scoped class for XPTI instrumentation using TLS data
  /// @param CodePtr  The address of the class/function to help differentiate
  /// actions in case the code location information is not available
  /// @param TraceType The type of trace event being created
  /// @param StreamName  The stream which will emit these notifications
  /// @param UserData String value that provides metadata about the
  /// instrumentation
  XPTIScope(void *CodePtr, uint16_t TraceType, const char *StreamName,
            const char *UserData)
      : MUserData(UserData), MStreamID(0), MInstanceID(0), MScopedNotify(false),
        MTraceType(0) {
    detail::tls_code_loc_t Tls;
    auto TData = Tls.query();
    // If TLS is not set, we can still genertate universal IDs with user data
    // and CodePtr information
    const char *FuncName = TData.functionName();
    if (!TData.functionName() && !TData.fileName())
      FuncName = UserData;
    // Create a tracepoint object that has a lifetime of this class
    MTP = new TracePoint(TData.fileName(), FuncName, TData.lineNumber(),
                         TData.columnNumber(), CodePtr);
    if (TraceType == (uint16_t)xpti::trace_point_type_t::graph_create ||
        TraceType == (uint16_t)xpti::trace_point_type_t::node_create ||
        TraceType == (uint16_t)xpti::trace_point_type_t::edge_create ||
        TraceType == (uint16_t)xpti::trace_point_type_t::queue_create)
      MTP->parent_event(GSYCLGraphEvent);
    // Now if tracing is enabled, create trace events and notify
    if (xptiTraceEnabled() && MTP) {
      MTP->stream(StreamName).trace_type((xpti::trace_point_type_t)TraceType);
      MTraceEvent = const_cast<xpti::trace_event_data_t *>(MTP->trace_event());
      MStreamID = MTP->stream_id();
      MInstanceID = MTP->instance_id();
    }
  }

  XPTIScope(const XPTIScope &rhs) = delete;

  XPTIScope &operator=(const XPTIScope &rhs) = delete;

  xpti::trace_event_data_t *traceEvent() { return MTraceEvent; }

  uint8_t streamID() { return MStreamID; }

  uint64_t instanceID() { return MTP ? MTP->instance_id() : 0; }

  XPTIScope &
  addMetadata(const std::function<void(xpti::trace_event_data_t *)> &Callback) {
    if (xptiTraceEnabled() && MTP) {
      auto TEvent = const_cast<xpti::trace_event_data_t *>(MTP->trace_event());
      Callback(TEvent);
    }
    return *this;
  }

  XPTIScope &notify() {
    MTP->notify(static_cast<const void *>(MUserData));
    return *this;
  }

  /// @brief Method that emits begin/end trace notifications
  /// @return Current class
  XPTIScope &scopedNotify(uint16_t TraceType) {
    // Keep this data even if no subscribers are for this TraceType (begin).
    // Someone could still use (end) emitted from destructor.
    MTraceType = TraceType & 0xfffe;
    MScopedNotify = true;
    if (xptiCheckTraceEnabled(MStreamID, TraceType) && MTP) {
      xptiNotifySubscribers(MStreamID, MTraceType, nullptr, MTraceEvent,
                            MInstanceID, static_cast<const void *>(MUserData));
    }
    return *this;
  }
  ~XPTIScope() {
    MTraceType = MTraceType | 1;
    if (xptiCheckTraceEnabled(MStreamID, MTraceType) && MTP && MScopedNotify) {
      if (MTraceType == (uint16_t)xpti::trace_point_type_t::signal ||
          MTraceType == (uint16_t)xpti::trace_point_type_t::graph_create ||
          MTraceType == (uint16_t)xpti::trace_point_type_t::node_create ||
          MTraceType == (uint16_t)xpti::trace_point_type_t::edge_create ||
          MTraceType == (uint16_t)xpti::trace_point_type_t::queue_create ||
          MTraceType == (uint16_t)xpti::trace_point_type_t::queue_destroy ||
          MTraceType == (uint16_t)xpti::trace_point_type_t::diagnostics)
        return;

      // Only notify for a trace type that has a begin/end
      xptiNotifySubscribers(MStreamID, MTraceType, nullptr, MTraceEvent,
                            MInstanceID, static_cast<const void *>(MUserData));
    }
    // Delete the tracepoint object which will clear TLS if it is the top of
    // the scope
    delete MTP;
  }

private:
  // Tracepoint_t object who's lifetime is that of the class
  TracePoint *MTP = nullptr;
  // Trace event created from the TLS data, if it exists
  xpti::trace_event_data_t *MTraceEvent = nullptr;
  // The const string that indicates the operation
  const char *MUserData = nullptr;
  // The stream on which the notifications occur
  uint8_t MStreamID;
  // The instance ID for the trace event; if it is called in a loop, then the
  // trace event ID will remain the same, but the instance ID will increment
  uint64_t MInstanceID;
  // If scoped notifcation is requested, this tracks the request
  bool MScopedNotify;
  // The trace type information for scoped notifications
  uint16_t MTraceType;
}; // class XPTIScope
#endif

class queue_impl;
std::string queueDeviceToString(const detail::queue_impl *const &Queue);

} // namespace detail
} // namespace _V1
} // namespace sycl
