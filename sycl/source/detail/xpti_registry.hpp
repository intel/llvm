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

#include <CL/sycl/detail/common.hpp>

#ifdef XPTI_ENABLE_INSTRUMENTATION
// Include the headers necessary for emitting
// traces using the trace framework
#include "xpti/xpti_trace_framework.h"
#endif

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
// We define a sycl stream name and this will be used by the instrumentation
// framework
inline constexpr const char *SYCL_STREAM_NAME = "sycl";
// Stream name being used for traces generated from the SYCL plugin layer
inline constexpr const char *SYCL_PICALL_STREAM_NAME = "sycl.pi";
// Stream name being used for traces generated from PI calls. This stream
// contains information about function arguments.
inline constexpr const char *SYCL_PIDEBUGCALL_STREAM_NAME = "sycl.pi.debug";
inline constexpr auto SYCL_MEM_ALLOC_STREAM_NAME =
    "sycl.experimental.mem_alloc";

#ifdef XPTI_ENABLE_INSTRUMENTATION
extern uint8_t GBufferStreamID;
extern uint8_t GMemAllocStreamID;
extern xpti::trace_event_data_t *GMemAllocEvent;
#endif

// Stream name being used to notify about buffer objects.
inline constexpr const char *SYCL_BUFFER_STREAM_NAME =
    "sycl.experimental.buffer";

class XPTIRegistry {
public:
  void initializeFrameworkOnce() {
#ifdef XPTI_ENABLE_INSTRUMENTATION
    std::call_once(MInitialized, [this] {
      xptiFrameworkInitialize();
      // SYCL buffer events
      GBufferStreamID = xptiRegisterStream(SYCL_BUFFER_STREAM_NAME);
      this->initializeStream(SYCL_BUFFER_STREAM_NAME, 0, 1, "0.1");

      // Memory allocation events
      GMemAllocStreamID = xptiRegisterStream(SYCL_MEM_ALLOC_STREAM_NAME);
      this->initializeStream(SYCL_MEM_ALLOC_STREAM_NAME, 0, 1, "0.1");
      xpti::payload_t MAPayload("SYCL Memory Allocations Layer");
      uint64_t MAInstanceNo = 0;
      GMemAllocEvent = xptiMakeEvent("SYCL Memory Allocations", &MAPayload,
                                     xpti::trace_algorithm_event,
                                     xpti_at::active, &MAInstanceNo);
    });
#endif
  }

  /// Notifies XPTI subscribers about new stream.
  ///
  /// \param StreamName is a name of newly initialized stream.
  /// \param MajVer is a stream major version.
  /// \param MinVer is a stream minor version.
  /// \param VerStr is a string of "MajVer.MinVer" format.
  void initializeStream(const std::string &StreamName, uint32_t MajVer,
                        uint32_t MinVer, const std::string &VerStr) {
#ifdef XPTI_ENABLE_INSTRUMENTATION
    MActiveStreams.insert(StreamName);
    xptiInitialize(StreamName.c_str(), MajVer, MinVer, VerStr.c_str());
#endif // XPTI_ENABLE_INSTRUMENTATION
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
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
