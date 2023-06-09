//==----------------- usm_analyzer.hpp -------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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

public:
  std::mutex IOMutex;
  std::map<void *, AllocationInfo> ActivePointers;
  TracepointInfo LastTracepoint;
  sycl::xpti_helpers::PiArgumentsHandler ArgHandlerPostCall;
  sycl::xpti_helpers::PiArgumentsHandler ArgHandlerPreCall;
  bool TerminateOnError = false;

  USMAnalyzer(const USMAnalyzer &obj) = delete;

  static USMAnalyzer &getInstance() {
    static USMAnalyzer s;
    return s;
  }

  void changeTerminationOnErrorState(bool EnableTermination) {
    TerminateOnError = EnableTermination;
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
    if (GS.ActivePointers.count(Ptr) == 0) {
      std::cerr << "Attempt to free pointer " << std::hex << Ptr;
      std::cerr << " that was not allocated with SYCL USM APIs.\n";
      std::cerr << "  Location: function " << GS.LastTracepoint.Function;
      std::cerr << " at " << GS.LastTracepoint.Source << ":";
      std::cerr << std::dec << GS.LastTracepoint.Line << "\n";
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
    for (const auto &Alloc : GS.ActivePointers) {
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
          std::cerr << " function " << GS.LastTracepoint.Function << " at ";
          std::cerr << GS.LastTracepoint.Source << ":" << GS.LastTracepoint.Line
                    << "\n";
          if (GS.TerminateOnError)
            std::terminate();
        }
        break;
      }
    }
  }

  static void handleUSMEnqueueMemset(pi_queue, void *ptr, pi_int32,
                                     size_t numBytes, pi_uint32,
                                     const pi_event *, pi_event *) {
    auto &GS = USMAnalyzer::getInstance();
    bool PointerFound = false;
    bool NeedsTerminate = false;
    for (const auto &Alloc : GS.ActivePointers) {
      const void *Begin = Alloc.first;
      const void *End =
          static_cast<const char *>(Alloc.first) + Alloc.second.Length;
      if (ptr >= Begin && ptr <= End) {
        PointerFound = true;
        const void *MemsetRegionEnd = static_cast<char *>(ptr) + numBytes;
        if (MemsetRegionEnd > End) {
          std::cerr
              << "Requested memset range exceeds allocated USM memory size.\n";
          NeedsTerminate = true;
        }

        if (NeedsTerminate) {
          std::cerr << "  Allocation location: ";
          std::cerr << " function " << Alloc.second.Location.Function << " at ";
          std::cerr << Alloc.second.Location.Source << ":"
                    << Alloc.second.Location.Line << "\n";
          std::cerr << "  Memset location: ";
          std::cerr << " function " << GS.LastTracepoint.Function << " at ";
          std::cerr << GS.LastTracepoint.Source << ":" << GS.LastTracepoint.Line
                    << "\n";
          if (GS.TerminateOnError)
            std::terminate();
        }
        break;
      }
    }
    if (!PointerFound) {
      std::cerr << "Function uses unknown USM pointer (could be already "
                   "released or not allocated as USM).\n";
      std::cerr << "  Memset location: ";
      std::cerr << " function " << GS.LastTracepoint.Function << " at ";
      std::cerr << GS.LastTracepoint.Source << ":" << GS.LastTracepoint.Line
                << "\n";
      if (GS.TerminateOnError)
        std::terminate();
    }
  }

  static void handleUSMEnqueueMemcpy(pi_queue, pi_bool, void *dst_ptr,
                                     const void *src_ptr, size_t size,
                                     pi_uint32, const pi_event *, pi_event *) {
    auto &GS = USMAnalyzer::getInstance();
    bool SrcPointerFound = false;
    bool DstPointerFound = false;
    auto CheckPointerValidness = [&](std::string ParameterDesc,
                                     const void *PtrToValidate) {
      bool NeedsTerminate = false;
      bool PointerFound = false;
      for (const auto &Alloc : GS.ActivePointers) {
        const void *Begin = Alloc.first;
        const void *End =
            static_cast<const char *>(Alloc.first) + Alloc.second.Length;
        if (PtrToValidate >= Begin && PtrToValidate <= End) {
          PointerFound = true;
          const void *CopyRegionEnd =
              static_cast<const char *>(PtrToValidate) + size;
          if (CopyRegionEnd > End) {
            std::cerr
                << "Requested copy range exceeds allocated USM memory size for "
                << ParameterDesc << ".\n";
            NeedsTerminate = true;
          }

          if (NeedsTerminate) {
            std::cerr << "  Allocation location: ";
            std::cerr << " function " << Alloc.second.Location.Function
                      << " at ";
            std::cerr << Alloc.second.Location.Source << ":"
                      << Alloc.second.Location.Line << "\n";
            std::cerr << "  Memcpy location: ";
            std::cerr << " function " << GS.LastTracepoint.Function << " at ";
            std::cerr << GS.LastTracepoint.Source << ":"
                      << GS.LastTracepoint.Line << "\n";
            if (GS.TerminateOnError)
              std::terminate();
          }
          break;
        }
      }
      if (!PointerFound) {
        std::cerr << "Function uses unknown USM pointer (could be already "
                     "released or not allocated as USM).\n";
        std::cerr << "  Memcpy location: ";
        std::cerr << " function " << GS.LastTracepoint.Function << " at ";
        std::cerr << GS.LastTracepoint.Source << ":" << GS.LastTracepoint.Line
                  << "\n";
        if (GS.TerminateOnError)
          std::terminate();
      }
    };
    CheckPointerValidness("source memory block", src_ptr);
    CheckPointerValidness("destination memory block", dst_ptr);
  }
};
