//==----------------- usm_analyzer.hpp -------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "xpti/xpti_trace_framework.h"

#include <ur_api.h>

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

      if (PtrToValidate >= Begin && PtrToValidate < End) {
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
      if (PtrToValidate >= Begin && PtrToValidate < End) {
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

  void handlePostCall(const xpti::function_with_args_t *Data) {
    switch (static_cast<ur_function_t>(Data->function_id)) {
    case UR_FUNCTION_USM_HOST_ALLOC:
      handleUSMHostAlloc(
          static_cast<ur_usm_host_alloc_params_t *>(Data->args_data));
      return;
    case UR_FUNCTION_USM_DEVICE_ALLOC:
      handleUSMDeviceAlloc(
          static_cast<ur_usm_device_alloc_params_t *>(Data->args_data));
      return;
    case UR_FUNCTION_USM_SHARED_ALLOC:
      handleUSMSharedAlloc(
          static_cast<ur_usm_shared_alloc_params_t *>(Data->args_data));
      return;
    default:
      return;
    }
  }

  void handlePreCall(const xpti::function_with_args_t *Data) {
    switch (static_cast<ur_function_t>(Data->function_id)) {
    case UR_FUNCTION_USM_FREE:
      handleUSMFree(static_cast<ur_usm_free_params_t *>(Data->args_data));
      return;
    case UR_FUNCTION_MEM_BUFFER_CREATE:
      handleMemBufferCreate(
          static_cast<ur_mem_buffer_create_params_t *>(Data->args_data));
      return;
    case UR_FUNCTION_ENQUEUE_USM_MEMCPY:
      handleUSMEnqueueMemcpy(
          static_cast<ur_enqueue_usm_memcpy_params_t *>(Data->args_data));
      return;
    case UR_FUNCTION_ENQUEUE_USM_PREFETCH:
      handleUSMEnqueuePrefetch(
          static_cast<ur_enqueue_usm_prefetch_params_t *>(Data->args_data));
      return;
    case UR_FUNCTION_ENQUEUE_USM_ADVISE:
      handleUSMEnqueueMemAdvise(
          static_cast<ur_enqueue_usm_advise_params_t *>(Data->args_data));
      return;
    case UR_FUNCTION_ENQUEUE_USM_FILL_2D:
      handleUSMEnqueueFill2D(
          static_cast<ur_enqueue_usm_fill_2d_params_t *>(Data->args_data));
      return;
    case UR_FUNCTION_ENQUEUE_USM_MEMCPY_2D:
      handleUSMEnqueueMemcpy2D(
          static_cast<ur_enqueue_usm_memcpy_2d_params_t *>(Data->args_data));
      return;
    case UR_FUNCTION_KERNEL_SET_ARG_POINTER:
      handleKernelSetArgPointer(
          static_cast<ur_kernel_set_arg_pointer_params_t *>(Data->args_data));
      return;
    default:
      return;
    }
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

  static void handleUSMHostAlloc(const ur_usm_host_alloc_params_t *Params) {
    auto &GS = USMAnalyzer::getInstance();
    AllocationInfo Info;
    Info.Location = GS.LastTracepoint;
    Info.Length = *Params->psize;
    Info.Kind = AllocKind::host;
    GS.ActivePointers[**Params->pppMem] = Info;
  }

  static void handleUSMDeviceAlloc(const ur_usm_device_alloc_params_t *Params) {
    auto &GS = USMAnalyzer::getInstance();

    AllocationInfo Info;
    Info.Location = GS.LastTracepoint;
    Info.Length = *Params->psize;
    Info.Kind = AllocKind::device;
    GS.ActivePointers[**Params->pppMem] = Info;
  }

  static void handleUSMSharedAlloc(const ur_usm_shared_alloc_params_t *Params) {
    auto &GS = USMAnalyzer::getInstance();
    AllocationInfo Info;
    Info.Location = GS.LastTracepoint;
    Info.Length = *Params->psize;
    Info.Kind = AllocKind::shared;
    GS.ActivePointers[**Params->pppMem] = Info;
  }

  static void handleUSMFree(const ur_usm_free_params_t *Params) {
    auto &GS = USMAnalyzer::getInstance();
    auto &OutStream = GS.getOutStream();
    if (GS.ActivePointers.count(*Params->ppMem) == 0) {
      OutStream << std::endl;
      OutStream << PrintPrefix << "Attempt to free pointer " << std::hex
                << *Params->ppMem;
      OutStream << " that was not allocated with SYCL USM APIs.\n";
      OutStream << PrintIndentation << "Location: function "
                << GS.LastTracepoint.Function;
      OutStream << " at " << GS.LastTracepoint.Source << ":";
      OutStream << std::dec << GS.LastTracepoint.Line << "\n";
      if (GS.TerminateOnError)
        std::terminate();
    }
    GS.ActivePointers.erase(*Params->ppMem);
  }

  static void
  handleMemBufferCreate(const ur_mem_buffer_create_params_t *Params) {
    auto &GS = USMAnalyzer::getInstance();
    auto &OutStream = GS.getOutStream();
    void *HostPtr = nullptr;
    if (*Params->ppProperties) {
      HostPtr = (*Params->ppProperties)->pHost;
    }
    for (const auto &Alloc : GS.ActivePointers) {
      const void *Begin = Alloc.first;
      const void *End =
          static_cast<const char *>(Alloc.first) + Alloc.second.Length;
      // Host pointer was allocated with USM APIs
      if (HostPtr >= Begin && HostPtr < End) {
        bool NeedsTerminate = false;
        if (Alloc.second.Kind != AllocKind::host) {
          OutStream << PrintPrefix
                    << "Attempt to construct a buffer with non-host pointer.\n";
          NeedsTerminate = true;
        }

        const void *HostEnd = static_cast<char *>(HostPtr) + *Params->psize;
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

  static void
  handleUSMEnqueueMemcpy(const ur_enqueue_usm_memcpy_params_t *Params) {
    CheckPointerValidness("source memory block", *Params->ppSrc, *Params->psize,
                          "memcpy");
    CheckPointerValidness("destination memory block", *Params->ppDst,
                          *Params->psize, "memcpy");
  }

  static void
  handleUSMEnqueueFill(const ur_enqueue_usm_memcpy_params_t *Params) {
    CheckPointerValidness("input parameter", *Params->ppDst, *Params->psize,
                          "fill");
  }

  static void
  handleUSMEnqueuePrefetch(const ur_enqueue_usm_prefetch_params_t *Params) {
    CheckPointerValidness("input parameter", *Params->ppMem, *Params->psize,
                          "prefetch");
  }

  static void
  handleUSMEnqueueMemAdvise(const ur_enqueue_usm_advise_params_t *Params) {
    CheckPointerValidness("input parameter", *Params->ppMem, *Params->psize,
                          "mem_advise");
  }

  static void
  handleUSMEnqueueFill2D(const ur_enqueue_usm_fill_2d_params_t *Params) {
    // TO DO: add checks for pattern validity
    CheckPointerValidness("input parameter", *Params->ppMem, *Params->ppitch,
                          *Params->pwidth, *Params->pheight,
                          "ext_oneapi_fill2d");
  }

  static void
  handleUSMEnqueueMemcpy2D(const ur_enqueue_usm_memcpy_2d_params_t *Params) {
    CheckPointerValidness("source parameter", *Params->ppSrc,
                          *Params->psrcPitch, *Params->pwidth, *Params->pheight,
                          "ext_oneapi_copy2d/ext_oneapi_memcpy2d");
    CheckPointerValidness("destination parameter", *Params->ppDst,
                          *Params->pdstPitch, *Params->pwidth, *Params->pheight,
                          "ext_oneapi_copy2d/ext_oneapi_memcpy2d");
  }

  static void
  handleKernelSetArgPointer(const ur_kernel_set_arg_pointer_params_t *Params) {
    void *Ptr = (const_cast<void *>(*Params->ppArgValue));
    CheckPointerValidness(
        "kernel parameter with index = " + std::to_string(*Params->pargIndex),
        Ptr, 0 /*no data how it will be used in kernel*/, "kernel");
  }
};
