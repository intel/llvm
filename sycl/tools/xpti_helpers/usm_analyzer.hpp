//==----------------- usm_analyzer.hpp -------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

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

class USMAnalyzer {
private:
  USMAnalyzer(){};
  // TO DO: mem allocations could be effectively validated with
  // piextUSMGetMemAllocInfo - could be more robust
  static void CheckPointerValidness(std::string ParameterDesc, const void *Ptr,
                                    size_t size, std::string FunctionName) {
    void *PtrToValidate = (const_cast<void *>(Ptr));

    bool PointerFound = false;
    auto &GS = USMAnalyzer::getInstance();
    auto &OutStream = GS.getOutStream();
    if (PtrToValidate == nullptr) {
      OutStream << std::endl;
      OutStream << PrintPrefix << "Function uses nullptr as " << ParameterDesc
                << ".\n";
      OutStream << PrintIndentation << FunctionName << " location: ";
      OutStream << " function " << GS.LastTracepoint.Function << " at ";
      OutStream << GS.LastTracepoint.Source << ":" << GS.LastTracepoint.Line
                << std::endl;
      if (GS.TerminateOnError)
        std::terminate();
      return;
    }

    for (const auto &Alloc : GS.ActivePointers) {
      const void *Begin = Alloc.first;
      const void *End =
          static_cast<const char *>(Alloc.first) + Alloc.second.Length;

      if (PtrToValidate >= Begin && PtrToValidate <= End) {
        PointerFound = true;
        const void *CopyRegionEnd =
            static_cast<const char *>(PtrToValidate) + size;
        if (CopyRegionEnd > End) {
          OutStream << std::endl;
          OutStream << PrintPrefix << "Requested " << FunctionName
                    << " range exceeds allocated USM memory size for "
                    << ParameterDesc << ".\n";
          OutStream << PrintIndentation << "Allocation location: ";
          OutStream << " function " << Alloc.second.Location.Function << " at ";
          OutStream << Alloc.second.Location.Source << ":"
                    << Alloc.second.Location.Line << "\n";
          OutStream << PrintIndentation << FunctionName << " location: ";
          OutStream << " function " << GS.LastTracepoint.Function << " at ";
          OutStream << GS.LastTracepoint.Source << ":" << GS.LastTracepoint.Line
                    << std::endl;
          if (GS.TerminateOnError)
            std::terminate();
        }
        break;
      }
    }

    if (!PointerFound) {
      OutStream << std::endl;
      OutStream << PrintPrefix
                << "Function uses unknown USM pointer (could be already "
                   "released or not allocated as USM) as "
                << ParameterDesc << ".\n";
      OutStream << PrintIndentation << FunctionName << " location: ";
      OutStream << " function " << GS.LastTracepoint.Function << " at ";
      OutStream << GS.LastTracepoint.Source << ":" << GS.LastTracepoint.Line
                << std::endl;
      if (GS.TerminateOnError)
        std::terminate();
    }
  }

  static void CheckPointerValidness(std::string ParameterDesc, const void *Ptr,
                                    size_t pitch, size_t width, size_t length,
                                    std::string FunctionName) {
    void *PtrToValidate = *(void **)(const_cast<void *>(Ptr));
    bool PointerFound = false;
    auto &GS = USMAnalyzer::getInstance();
    auto &OutStream = GS.getOutStream();
    if (width > length) {
      OutStream << std::endl;
      OutStream << PrintPrefix << "Requested " << FunctionName
                << " width is greater than pitch for  " << ParameterDesc
                << ".\n";
      OutStream << PrintIndentation << FunctionName << " location: ";
      OutStream << " function " << GS.LastTracepoint.Function << " at ";
      OutStream << GS.LastTracepoint.Source << ":" << GS.LastTracepoint.Line
                << std::endl;
      if (GS.TerminateOnError)
        std::terminate();
      return;
    }

    if (PtrToValidate == nullptr) {
      OutStream << std::endl;
      OutStream << PrintPrefix << "Function uses nullptr as " << ParameterDesc
                << ".\n";
      OutStream << PrintIndentation << FunctionName << " location: ";
      OutStream << " function " << GS.LastTracepoint.Function << " at ";
      OutStream << GS.LastTracepoint.Source << ":" << GS.LastTracepoint.Line
                << std::endl;
      if (GS.TerminateOnError)
        std::terminate();
      return;
    }

    for (const auto &Alloc : GS.ActivePointers) {
      const void *Begin = Alloc.first;
      const void *End =
          static_cast<const char *>(Alloc.first) + Alloc.second.Length;
      if (PtrToValidate >= Begin && PtrToValidate <= End) {
        PointerFound = true;
        const void *CopyRegionEnd =
            static_cast<const char *>(PtrToValidate) + pitch * length;
        if (CopyRegionEnd > End) {
          OutStream << std::endl;
          OutStream << PrintPrefix << "Requested " << FunctionName
                    << " range exceeds allocated USM memory size for "
                    << ParameterDesc << ".\n";
          OutStream << PrintIndentation << "Allocation location: ";
          OutStream << " function " << Alloc.second.Location.Function << " at ";
          OutStream << Alloc.second.Location.Source << ":"
                    << Alloc.second.Location.Line << "\n";
          OutStream << PrintIndentation << FunctionName << " location: ";
          OutStream << " function " << GS.LastTracepoint.Function << " at ";
          OutStream << GS.LastTracepoint.Source << ":" << GS.LastTracepoint.Line
                    << std::endl;
          if (GS.TerminateOnError)
            std::terminate();
        }

        break;
      }
    }
    if (!PointerFound) {
      OutStream << std::endl;
      OutStream << PrintPrefix
                << "Function uses unknown USM pointer (could be already "
                   "released or not allocated as USM).\n";
      OutStream << PrintIndentation << FunctionName << " location: ";
      OutStream << " function " << GS.LastTracepoint.Function << " at ";
      OutStream << GS.LastTracepoint.Source << ":" << GS.LastTracepoint.Line
                << std::endl;
      if (GS.TerminateOnError)
        std::terminate();
    }
  }

  static constexpr char PrintPrefix[] = "[USM] ";
  static constexpr char PrintIndentation[] = "      | ";
  bool PrintToError = false;

  std::ostream &getOutStream() { return PrintToError ? std::cerr : std::cout; }

public:
  // TO DO: allocations must be tracked with device
  std::map<void *, AllocationInfo> ActivePointers;
  TracepointInfo LastTracepoint;
  sycl::xpti_helpers::PiArgumentsHandler ArgHandlerPostCall;
  sycl::xpti_helpers::PiArgumentsHandler ArgHandlerPreCall;
  bool TerminateOnError = false;

  USMAnalyzer(const USMAnalyzer &obj) = delete;
  USMAnalyzer &operator=(const USMAnalyzer &rhs) = delete;

  static USMAnalyzer &getInstance() {
    static USMAnalyzer s;
    return s;
  }

  void changeTerminationOnErrorState(bool EnableTermination) {
    TerminateOnError = EnableTermination;
  }

  void printToErrorStream() { PrintToError = true; }

  void setupUSMHandlers() {
    ArgHandlerPostCall.set_piextUSMHostAlloc(USMAnalyzer::handleUSMHostAlloc);
    ArgHandlerPostCall.set_piextUSMDeviceAlloc(
        USMAnalyzer::handleUSMDeviceAlloc);
    ArgHandlerPostCall.set_piextUSMSharedAlloc(
        USMAnalyzer::handleUSMSharedAlloc);
    ArgHandlerPreCall.set_piextUSMFree(USMAnalyzer::handleUSMFree);
    ArgHandlerPreCall.set_piMemBufferCreate(USMAnalyzer::handleMemBufferCreate);
    ArgHandlerPreCall.set_piextUSMEnqueueFill(
        USMAnalyzer::handleUSMEnqueueFill);
    ArgHandlerPreCall.set_piextUSMEnqueueMemcpy(
        USMAnalyzer::handleUSMEnqueueMemcpy);
    ArgHandlerPreCall.set_piextUSMEnqueuePrefetch(
        USMAnalyzer::handleUSMEnqueuePrefetch);
    ArgHandlerPreCall.set_piextUSMEnqueueMemAdvise(
        USMAnalyzer::handleUSMEnqueueMemAdvise);
    ArgHandlerPreCall.set_piextUSMEnqueueFill2D(
        USMAnalyzer::handleUSMEnqueueFill2D);
    ArgHandlerPreCall.set_piextUSMEnqueueMemset2D(
        USMAnalyzer::handleUSMEnqueueMemset2D);
    ArgHandlerPreCall.set_piextUSMEnqueueMemcpy2D(
        USMAnalyzer::handleUSMEnqueueMemcpy2D);
    ArgHandlerPreCall.set_piextKernelSetArgPointer(
        USMAnalyzer::handleKernelSetArgPointer);
  }

  void fillLastTracepointData(const xpti::trace_event_data_t *ObjectEvent) {
    const xpti::payload_t *Payload =
        ObjectEvent && ObjectEvent->reserved.payload
            ? ObjectEvent->reserved.payload
            : xptiQueryPayloadByUID(xptiGetUniversalId());

    if (Payload) {
      if (Payload->source_file)
        LastTracepoint.Source = Payload->source_file;
      else
        LastTracepoint.Source = "<unknown>";
      LastTracepoint.Function = Payload->name;
      LastTracepoint.Line = Payload->line_no;
    } else {
      LastTracepoint.Function = "<unknown>";
      LastTracepoint.Source = "<unknown>";
      LastTracepoint.Line = 0;
    }
  }

  static void handleUSMHostAlloc(const pi_plugin &, std::optional<pi_result>,
                                 void **ResultPtr, pi_context,
                                 pi_usm_mem_properties *, size_t Size,
                                 pi_uint32) {
    auto &GS = USMAnalyzer::getInstance();
    AllocationInfo Info;
    Info.Location = GS.LastTracepoint;
    Info.Length = Size;
    Info.Kind = AllocKind::host;
    GS.ActivePointers[*ResultPtr] = Info;
  }

  static void handleUSMDeviceAlloc(const pi_plugin &, std::optional<pi_result>,
                                   void **ResultPtr, pi_context, pi_device,
                                   pi_usm_mem_properties *, size_t Size,
                                   pi_uint32) {
    auto &GS = USMAnalyzer::getInstance();

    AllocationInfo Info;
    Info.Location = GS.LastTracepoint;
    Info.Length = Size;
    Info.Kind = AllocKind::device;
    GS.ActivePointers[*ResultPtr] = Info;
  }

  static void handleUSMSharedAlloc(const pi_plugin &, std::optional<pi_result>,
                                   void **ResultPtr, pi_context, pi_device,
                                   pi_usm_mem_properties *, size_t Size,
                                   pi_uint32) {
    auto &GS = USMAnalyzer::getInstance();
    AllocationInfo Info;
    Info.Location = GS.LastTracepoint;
    Info.Length = Size;
    Info.Kind = AllocKind::shared;
    GS.ActivePointers[*ResultPtr] = Info;
  }

  static void handleUSMFree(const pi_plugin &, std::optional<pi_result>,
                            pi_context, void *Ptr) {
    auto &GS = USMAnalyzer::getInstance();
    auto &OutStream = GS.getOutStream();
    if (GS.ActivePointers.count(Ptr) == 0) {
      OutStream << std::endl;
      OutStream << PrintPrefix << "Attempt to free pointer " << std::hex << Ptr;
      OutStream << " that was not allocated with SYCL USM APIs.\n";
      OutStream << PrintIndentation << "Location: function "
                << GS.LastTracepoint.Function;
      OutStream << " at " << GS.LastTracepoint.Source << ":";
      OutStream << std::dec << GS.LastTracepoint.Line << "\n";
      if (GS.TerminateOnError)
        std::terminate();
    }
    GS.ActivePointers.erase(Ptr);
  }

  static void handleMemBufferCreate(const pi_plugin &, std::optional<pi_result>,
                                    pi_context, pi_mem_flags, size_t Size,
                                    void *HostPtr, pi_mem *,
                                    const pi_mem_properties *) {
    auto &GS = USMAnalyzer::getInstance();
    auto &OutStream = GS.getOutStream();
    for (const auto &Alloc : GS.ActivePointers) {
      const void *Begin = Alloc.first;
      const void *End =
          static_cast<const char *>(Alloc.first) + Alloc.second.Length;
      // Host pointer was allocated with USM APIs
      if (HostPtr >= Begin && HostPtr <= End) {
        bool NeedsTerminate = false;
        if (Alloc.second.Kind != AllocKind::host) {
          OutStream << PrintPrefix
                    << "Attempt to construct a buffer with non-host pointer.\n";
          NeedsTerminate = true;
        }

        const void *HostEnd = static_cast<char *>(HostPtr) + Size;
        if (HostEnd > End) {
          OutStream << PrintPrefix
                    << "Buffer size exceeds allocated host memory size.\n";
          NeedsTerminate = true;
        }

        if (NeedsTerminate) {
          OutStream << PrintIndentation << "Allocation location: ";
          OutStream << " function " << Alloc.second.Location.Function << " at ";
          OutStream << Alloc.second.Location.Source << ":"
                    << Alloc.second.Location.Line << "\n";
          OutStream << PrintIndentation << "Buffer location: ";
          OutStream << " function " << GS.LastTracepoint.Function << " at ";
          OutStream << GS.LastTracepoint.Source << ":" << GS.LastTracepoint.Line
                    << "\n";
          if (GS.TerminateOnError)
            std::terminate();
        }
        break;
      }
    }
  }

  static void handleUSMEnqueueFill(const pi_plugin &, std::optional<pi_result>,
                                   pi_queue, void *ptr, const void *, size_t,
                                   size_t numBytes, pi_uint32, const pi_event *,
                                   pi_event *) {
    CheckPointerValidness("input parameter", ptr, numBytes, "fill");
  }

  static void handleUSMEnqueueMemcpy(const pi_plugin &,
                                     std::optional<pi_result>, pi_queue,
                                     pi_bool, void *dst_ptr,
                                     const void *src_ptr, size_t size,
                                     pi_uint32, const pi_event *, pi_event *) {
    CheckPointerValidness("source memory block", src_ptr, size, "memcpy");
    CheckPointerValidness("destination memory block", dst_ptr, size, "memcpy");
  }

  static void handleUSMEnqueuePrefetch(const pi_plugin &,
                                       std::optional<pi_result>, pi_queue,
                                       const void *ptr, size_t size,
                                       pi_usm_migration_flags, pi_uint32,
                                       const pi_event *, pi_event *) {
    CheckPointerValidness("input parameter", ptr, size, "prefetch");
  }

  static void handleUSMEnqueueMemAdvise(const pi_plugin &,
                                        std::optional<pi_result>, pi_queue,
                                        const void *ptr, size_t length,
                                        pi_mem_advice, pi_event *) {
    CheckPointerValidness("input parameter", ptr, length, "mem_advise");
  }

  static void handleUSMEnqueueFill2D(const pi_plugin &,
                                     std::optional<pi_result>, pi_queue,
                                     void *ptr, size_t pitch, size_t,
                                     const void *, size_t width, size_t height,
                                     pi_uint32, const pi_event *, pi_event *) {
    // TO DO: add checks for pattern validity
    CheckPointerValidness("input parameter", ptr, pitch, width, height,
                          "ext_oneapi_fill2d");
  }

  static void handleUSMEnqueueMemset2D(const pi_plugin &,
                                       std::optional<pi_result>, pi_queue,
                                       void *ptr, size_t pitch, int,
                                       size_t width, size_t height, pi_uint32,
                                       const pi_event *, pi_event *) {
    CheckPointerValidness("input parameter", ptr, pitch, width, height,
                          "ext_oneapi_memset2d");
  }

  static void handleUSMEnqueueMemcpy2D(const pi_plugin &,
                                       std::optional<pi_result>, pi_queue,
                                       pi_bool, void *dst_ptr, size_t dst_pitch,
                                       const void *src_ptr, size_t src_pitch,
                                       size_t width, size_t height, pi_uint32,
                                       const pi_event *, pi_event *) {
    CheckPointerValidness("source parameter", src_ptr, src_pitch, width, height,
                          "ext_oneapi_copy2d/ext_oneapi_memcpy2d");
    CheckPointerValidness("destination parameter", dst_ptr, dst_pitch, width,
                          height, "ext_oneapi_copy2d/ext_oneapi_memcpy2d");
  }

  static void handleKernelSetArgPointer(const pi_plugin &,
                                        std::optional<pi_result>, pi_kernel,
                                        pi_uint32 arg_index, size_t arg_size,
                                        const void *arg_value) {
    // no clarity how to handle complex types so check only simple pointers here
    if (arg_size == sizeof(arg_value)) {
      void *Ptr = *(void **)(const_cast<void *>(arg_value));
      CheckPointerValidness(
          "kernel parameter with index = " + std::to_string(arg_index), Ptr,
          0 /*no data how it will be used in kernel*/, "kernel");
    }
  }
};
