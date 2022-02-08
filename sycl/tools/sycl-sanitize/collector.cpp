//==-------------- collector.cpp -------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// \file collector.cpp
/// The SYCL sanitizer collector intercepts PI calls to find memory leaks in
/// usages of USM pointers.

#include "xpti/xpti_trace_framework.h"

#include "pi_arguments_handler.hpp"

#include <detail/plugin_printers.hpp>

#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>

struct TracepointInfo {
  std::string Source;
  std::string Function;
  uint32_t Line;
};

enum class AllocKind { host, device, shared };

struct AllocationInfo {
  size_t Length;
  AllocKind Kind;
  TracepointInfo Location;
};

struct GlobalState {
  std::mutex IOMutex;
  std::map<void *, AllocationInfo> ActivePointers;
  TracepointInfo LastTracepoint;
  sycl::xpti_helpers::PiArgumentsHandler ArgHandlerPostCall;
  sycl::xpti_helpers::PiArgumentsHandler ArgHandlerPreCall;
};

GlobalState *GS = nullptr;

static void handleUSMHostAlloc(const pi_plugin &, std::optional<pi_result>,
                               void **ResultPtr, pi_context,
                               pi_usm_mem_properties *, size_t Size,
                               pi_uint32) {
  AllocationInfo Info;
  Info.Location = GS->LastTracepoint;
  Info.Length = Size;
  Info.Kind = AllocKind::host;
  GS->ActivePointers[*ResultPtr] = Info;
}

static void handleUSMDeviceAlloc(const pi_plugin &, std::optional<pi_result>,
                                 void **ResultPtr, pi_context, pi_device,
                                 pi_usm_mem_properties *, size_t Size,
                                 pi_uint32) {
  AllocationInfo Info;
  Info.Location = GS->LastTracepoint;
  Info.Length = Size;
  Info.Kind = AllocKind::device;
  GS->ActivePointers[*ResultPtr] = Info;
}

static void handleUSMSharedAlloc(const pi_plugin &, std::optional<pi_result>,
                                 void **ResultPtr, pi_context, pi_device,
                                 pi_usm_mem_properties *, size_t Size,
                                 pi_uint32) {
  AllocationInfo Info;
  Info.Location = GS->LastTracepoint;
  Info.Length = Size;
  Info.Kind = AllocKind::shared;
  GS->ActivePointers[*ResultPtr] = Info;
}

static void handleUSMFree(const pi_plugin &, std::optional<pi_result>,
                          pi_context, void *Ptr) {
  if (GS->ActivePointers.count(Ptr) == 0) {
    std::cerr << "Attempt to free pointer " << std::hex << Ptr;
    std::cerr << " that was not allocated with SYCL USM APIs.\n";
    std::cerr << "  Location: function " << GS->LastTracepoint.Function;
    std::cerr << " at " << GS->LastTracepoint.Source << ":";
    std::cerr << std::dec << GS->LastTracepoint.Line << "\n";
    std::terminate();
  }
  GS->ActivePointers.erase(Ptr);
}

static void handleMemBufferCreate(const pi_plugin &, std::optional<pi_result>,
                                  pi_context, pi_mem_flags, size_t Size,
                                  void *HostPtr, pi_mem *,
                                  const pi_mem_properties *) {
  for (const auto &Alloc : GS->ActivePointers) {
    const void *Begin = Alloc.first;
    const void *End =
        static_cast<const char *>(Alloc.first) + Alloc.second.Length;
    // Host pointer was allocated with USM APIs
    if (HostPtr >= Begin && HostPtr <= End) {
      bool NeedsTerminate = false;
      if (Alloc.second.Kind != AllocKind::host) {
        std::cerr << "Attempt to construct a buffer with non-host pointer.\n";
        NeedsTerminate = true;
      }

      const void *HostEnd = static_cast<char *>(HostPtr) + Size;
      if (HostEnd > End) {
        std::cerr << "Buffer size exceeds allocated host memory size.\n";
        NeedsTerminate = true;
      }

      if (NeedsTerminate) {
        std::cerr << "  Allocation location: ";
        std::cerr << " function " << Alloc.second.Location.Function << " at ";
        std::cerr << Alloc.second.Location.Source << ":"
                  << Alloc.second.Location.Line << "\n";
        std::cerr << "  Buffer location: ";
        std::cerr << " function " << GS->LastTracepoint.Function << " at ";
        std::cerr << GS->LastTracepoint.Source << ":" << GS->LastTracepoint.Line
                  << "\n";
        std::terminate();
      }
      break;
    }
  }
}

XPTI_CALLBACK_API void tpCallback(uint16_t trace_type,
                                  xpti::trace_event_data_t *parent,
                                  xpti::trace_event_data_t *event,
                                  uint64_t instance, const void *user_data);

XPTI_CALLBACK_API void xptiTraceInit(unsigned int /*major_version*/,
                                     unsigned int /*minor_version*/,
                                     const char * /*version_str*/,
                                     const char *StreamName) {
  if (std::string_view(StreamName) == "sycl.pi.debug") {
    GS = new GlobalState;
    uint8_t StreamID = xptiRegisterStream(StreamName);
    xptiRegisterCallback(
        StreamID, (uint16_t)xpti::trace_point_type_t::function_with_args_begin,
        tpCallback);
    xptiRegisterCallback(
        StreamID, (uint16_t)xpti::trace_point_type_t::function_with_args_end,
        tpCallback);

    GS->ArgHandlerPostCall.set_piextUSMHostAlloc(handleUSMHostAlloc);
    GS->ArgHandlerPostCall.set_piextUSMDeviceAlloc(handleUSMDeviceAlloc);
    GS->ArgHandlerPostCall.set_piextUSMSharedAlloc(handleUSMSharedAlloc);
    GS->ArgHandlerPreCall.set_piextUSMFree(handleUSMFree);
    GS->ArgHandlerPreCall.set_piMemBufferCreate(handleMemBufferCreate);
  }
}

XPTI_CALLBACK_API void xptiTraceFinish(const char *StreamName) {
  if (std::string_view(StreamName) == "sycl.pi.debug") {
    bool hadLeak = false;
    if (GS->ActivePointers.size() > 0) {
      hadLeak = true;
      std::cerr << "Found " << GS->ActivePointers.size()
                << " leaked memory allocations\n";
      for (const auto &Ptr : GS->ActivePointers) {
        std::cerr << "Leaked pointer: " << std::hex << Ptr.first << "\n";
        std::cerr << "  Location: "
                  << "function " << Ptr.second.Location.Function << " at "
                  << Ptr.second.Location.Source << ":" << std::dec
                  << Ptr.second.Location.Line << "\n";
      }
    }

    delete GS;
    if (hadLeak)
      exit(-1);
  }
}

XPTI_CALLBACK_API void tpCallback(uint16_t TraceType,
                                  xpti::trace_event_data_t *,
                                  xpti::trace_event_data_t *,
                                  uint64_t /*Instance*/, const void *UserData) {
  auto *Payload = xptiQueryPayloadByUID(xptiGetUniversalId());

  if (Payload) {
    if (Payload->source_file)
      GS->LastTracepoint.Source = Payload->source_file;
    else
      GS->LastTracepoint.Source = "<unknown>";
    GS->LastTracepoint.Function = Payload->name;
    GS->LastTracepoint.Line = Payload->line_no;
  } else {
    GS->LastTracepoint.Function = "<unknown>";
    GS->LastTracepoint.Source = "<unknown>";
    GS->LastTracepoint.Line = 0;
  }

  auto Type = static_cast<xpti::trace_point_type_t>(TraceType);
  // Lock while we capture information
  std::lock_guard<std::mutex> Lock(GS->IOMutex);

  const auto *Data = static_cast<const xpti::function_with_args_t *>(UserData);
  const auto *Plugin = static_cast<pi_plugin *>(Data->user_data);
  if (Type == xpti::trace_point_type_t::function_with_args_begin) {
    GS->ArgHandlerPreCall.handle(Data->function_id, *Plugin, std::nullopt,
                                 Data->args_data);
  } else if (Type == xpti::trace_point_type_t::function_with_args_end) {
    const pi_result Result = *static_cast<pi_result *>(Data->ret_data);
    GS->ArgHandlerPostCall.handle(Data->function_id, *Plugin, Result,
                                  Data->args_data);
  }
}
