//==----------- ze_trace_collector.cpp -------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// \file ze_trace_collector.cpp
/// Routines to collect and print Level Zero API calls.

#include "xpti/xpti_trace_framework.h"

#include <sycl/detail/spinlock.hpp>

#include <zet_api.h>

#include <iostream>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>

#define COLLECTOR_EXPORT_API __attribute__((__visibility__("default")))

int IndentationLevel = 0;

enum class ZEApiKind {
#define _ZE_API(call, domain, cb, params_type) call,
#include "ze_api.def"
#undef _ZE_API
};

bool PrintVerbose = false;

static std::string getResult(ze_result_t Res) {
  std::string ResultStr;
  switch (Res) {
  case ZE_RESULT_SUCCESS:
    ResultStr = "ZE_RESULT_SUCCESS";
    if (PrintVerbose)
      ResultStr += " (success)";
    break;
  case ZE_RESULT_NOT_READY:
    ResultStr = "ZE_RESULT_NOT_READY";
    if (PrintVerbose)
      ResultStr += " (synchronization primitive not signaled)";
    break;
  case ZE_RESULT_ERROR_DEVICE_LOST:
    ResultStr = "ZE_RESULT_ERROR_DEVICE_LOST";
    if (PrintVerbose)
      ResultStr +=
          " (device hung, reset, was removed, or driver update occurred)";
    break;
  case ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY:
    ResultStr = "ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY";
    if (PrintVerbose)
      ResultStr += " (insufficient host memory to satisfy call)";
    break;
  case ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY:
    ResultStr = "ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY";
    if (PrintVerbose)
      ResultStr += " (insufficient device memory to satisfy call)";
    break;
  case ZE_RESULT_ERROR_MODULE_BUILD_FAILURE:
    ResultStr = "ZE_RESULT_ERROR_MODULE_BUILD_FAILURE";
    if (PrintVerbose)
      ResultStr +=
          " (error occurred when building module, see build log for details)";
    break;
  case ZE_RESULT_ERROR_MODULE_LINK_FAILURE:
    ResultStr = "ZE_RESULT_ERROR_MODULE_LINK_FAILURE";
    if (PrintVerbose)
      ResultStr +=
          " (error occurred when linking modules, see build log for details)";
    break;
  case ZE_RESULT_ERROR_DEVICE_REQUIRES_RESET:
    ResultStr = "ZE_RESULT_ERROR_DEVICE_REQUIRES_RESET";
    if (PrintVerbose)
      ResultStr += " (device requires a reset)";
    break;
  case ZE_RESULT_ERROR_DEVICE_IN_LOW_POWER_STATE:
    ResultStr = "ZE_RESULT_ERROR_DEVICE_IN_LOW_POWER_STATE";
    if (PrintVerbose)
      ResultStr += " (device currently in low power state)";
    break;
  case ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS:
    ResultStr = "ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS";
    if (PrintVerbose)
      ResultStr += " (access denied due to permission level)";
    break;
  case ZE_RESULT_ERROR_NOT_AVAILABLE:
    ResultStr = "ZE_RESULT_ERROR_NOT_AVAILABLE";
    if (PrintVerbose)
      ResultStr += " (resource already in use and simultaneous access not "
                   "allowed or resource was removed)";
    break;
  case ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE:
    ResultStr = "ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE";
    if (PrintVerbose)
      ResultStr += " (external required dependency is unavailable or missing)";
    break;
  case ZE_RESULT_ERROR_UNINITIALIZED:
    ResultStr = "ZE_RESULT_ERROR_UNINITIALIZED";
    if (PrintVerbose)
      ResultStr += " (driver is not initialized)";
    break;
  case ZE_RESULT_ERROR_UNSUPPORTED_VERSION:
    ResultStr = "ZE_RESULT_ERROR_UNSUPPORTED_VERSION";
    if (PrintVerbose)
      ResultStr += " (generic error code for unsupported versions)";
    break;
  case ZE_RESULT_ERROR_UNSUPPORTED_FEATURE:
    ResultStr = "ZE_RESULT_ERROR_UNSUPPORTED_FEATURE";
    if (PrintVerbose)
      ResultStr += " (generic error code for unsupported features)";
    break;
  case ZE_RESULT_ERROR_INVALID_ARGUMENT:
    ResultStr = "ZE_RESULT_ERROR_INVALID_ARGUMENT";
    if (PrintVerbose)
      ResultStr += " (generic error code for invalid arguments)";
    break;
  case ZE_RESULT_ERROR_INVALID_NULL_HANDLE:
    ResultStr = "ZE_RESULT_ERROR_INVALID_NULL_HANDLE";
    if (PrintVerbose)
      ResultStr += " (handle argument is not valid)";
    break;
  case ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE:
    ResultStr = "ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE";
    if (PrintVerbose)
      ResultStr += " (object pointed to by handle still in-use by device)";
    break;
  case ZE_RESULT_ERROR_INVALID_NULL_POINTER:
    ResultStr = "ZE_RESULT_ERROR_INVALID_NULL_POINTER";
    if (PrintVerbose)
      ResultStr += " (pointer argument may not be nullptr)";
    break;
  case ZE_RESULT_ERROR_INVALID_SIZE:
    ResultStr = "ZE_RESULT_ERROR_INVALID_SIZE";
    if (PrintVerbose)
      ResultStr += " (size argument is invalid (e.g., must not be zero))";
    break;
  case ZE_RESULT_ERROR_UNSUPPORTED_SIZE:
    ResultStr = "ZE_RESULT_ERROR_UNSUPPORTED_SIZE";
    if (PrintVerbose)
      ResultStr +=
          " (size argument is not supported by the device (e.g., too large))";
    break;
  case ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT:
    ResultStr = "ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT";
    if (PrintVerbose)
      ResultStr += " (alignment argument is not supported by the device (e.g., "
                   "too small))";
    break;
  case ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT:
    ResultStr = "ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT";
    if (PrintVerbose)
      ResultStr += " (synchronization object in invalid state)";
    break;
  case ZE_RESULT_ERROR_INVALID_ENUMERATION:
    ResultStr = "ZE_RESULT_ERROR_INVALID_ENUMERATION";
    if (PrintVerbose)
      ResultStr += " (enumerator argument is not valid)";
    break;
  case ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION:
    ResultStr = "ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION";
    if (PrintVerbose)
      ResultStr += " (enumerator argument is not supported by the device)";
    break;
  case ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT:
    ResultStr = "ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT";
    if (PrintVerbose)
      ResultStr += " (image format is not supported by the device)";
    break;
  case ZE_RESULT_ERROR_INVALID_NATIVE_BINARY:
    ResultStr += "ZE_RESULT_ERROR_INVALID_NATIVE_BINARY";
    if (PrintVerbose)
      ResultStr += " (native binary is not supported by the device)";
    break;
  case ZE_RESULT_ERROR_INVALID_GLOBAL_NAME:
    ResultStr = "ZE_RESULT_ERROR_INVALID_GLOBAL_NAME";
    if (PrintVerbose)
      ResultStr += " (global variable is not found in the module)";
    break;
  case ZE_RESULT_ERROR_INVALID_KERNEL_NAME:
    ResultStr = "ZE_RESULT_ERROR_INVALID_KERNEL_NAME";
    if (PrintVerbose)
      ResultStr += " (kernel name is not found in the module)";
    break;
  case ZE_RESULT_ERROR_INVALID_FUNCTION_NAME:
    ResultStr = "ZE_RESULT_ERROR_INVALID_FUNCTION_NAME";
    if (PrintVerbose)
      ResultStr += " (function name is not found in the module)";
    break;
  case ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION:
    ResultStr = "ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION";
    if (PrintVerbose)
      ResultStr +=
          " (group size dimension is not valid for the kernel or device)";
    break;
  case ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION:
    ResultStr = "ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION";
    if (PrintVerbose)
      ResultStr +=
          " (global width dimension is not valid for the kernel or device)";
    break;
  case ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX:
    ResultStr = "ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX";
    if (PrintVerbose)
      ResultStr += " (kernel argument index is not valid for kernel)";
    break;
  case ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE:
    ResultStr = "ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE";
    if (PrintVerbose)
      ResultStr += " (kernel argument size does not match kernel)";
    break;
  case ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE:
    ResultStr = "ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE";
    if (PrintVerbose)
      ResultStr +=
          " (value of kernel attribute is not valid for the kernel or device)";
    break;
  case ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED:
    ResultStr = "ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED";
    if (PrintVerbose)
      ResultStr += " (module with imports needs to be linked before kernels "
                   "can be created from it)";
    break;
  case ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE:
    ResultStr = "ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE";
    if (PrintVerbose)
      ResultStr += " (command list type does not match command queue type)";
    break;
  case ZE_RESULT_ERROR_OVERLAPPING_REGIONS:
    ResultStr = "ZE_RESULT_ERROR_OVERLAPPING_REGIONS";
    if (PrintVerbose)
      ResultStr +=
          " (copy operations do not support overlapping regions of memory)";
    break;
  default:
    ResultStr = "UNKNOWN ERROR";
    break;
  }

  return ResultStr;
}

extern "C" {

COLLECTOR_EXPORT_API void callback(uint16_t TraceType,
                                   xpti::trace_event_data_t * /*Parent*/,
                                   xpti::trace_event_data_t * /*Event*/,
                                   uint64_t /*Instance*/,
                                   const void *UserData) {
  const auto *Data = static_cast<const xpti::function_with_args_t *>(UserData);
  const auto PrintPrefix = [] {
    if (IndentationLevel)
      std::cout << "*  ";
  };
  if (TraceType == xpti::trace_function_with_args_begin) {

    const auto PrintOffset = [PrintPrefix]() {
      PrintPrefix();
      std::cout << "   ";
    };

    PrintPrefix();
    if (PrintVerbose) {
      std::string Source = "<unknown>";
      size_t Line = 0;

      auto *Payload = xptiQueryPayloadByUID(xptiGetUniversalId());

      if (Payload) {
        if (Payload->source_file != nullptr) {
          Source = Payload->source_file;
          Line = Payload->line_no;
        }
      }

      auto TID = std::this_thread::get_id();
      std::cout << "[L0:TID " << TID << ":";
      std::cout << Source << ":" << Line << "]\n";
      PrintPrefix();
    } else {
      std::cout << "[L0] ";
    }

    std::cout << Data->function_name << "(\n";

    switch (Data->function_id) {
#include "ze_printers.def"
    default:
      break; // unknown API
    }

    if (IndentationLevel) {
      std::cout << "*  ";
    }
    std::cout << std::flush;
  } else if (TraceType == xpti::trace_function_with_args_end) {
    std::cout << ") ---> "
              << getResult(*static_cast<ze_result_t *>(Data->ret_data))
              << std::endl;
    PrintPrefix();
    std::cout << std::endl;
  }
}

COLLECTOR_EXPORT_API void init() {
  std::string_view PrinterType(std::getenv("SYCL_TRACE_PRINT_FORMAT"));
  if (PrinterType == "classic") {
    std::cerr << "Classic output is unsupported for Level Zero\n";
  } else if (PrinterType == "verbose") {
    PrintVerbose = true;
  } else if (PrinterType == "compact") {
    PrintVerbose = false;
  }
}

// For unification purpose
COLLECTOR_EXPORT_API void finish() {}

COLLECTOR_EXPORT_API void setIndentationLevel(int NewLevel) {
  IndentationLevel = NewLevel;
}
}
