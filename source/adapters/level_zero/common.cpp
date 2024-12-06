//===--------- common.cpp - Level Zero Adapter ----------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "common.hpp"
#include "logger/ur_logger.hpp"
#include "usm.hpp"
#include <level_zero/include/ze_intel_gpu.h>

ur_result_t ze2urResult(ze_result_t ZeResult) {
  if (ZeResult == ZE_RESULT_SUCCESS)
    return UR_RESULT_SUCCESS;

  switch (ZeResult) {
  case ZE_RESULT_ERROR_DEVICE_LOST:
    return UR_RESULT_ERROR_DEVICE_LOST;
  case ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS:
    return UR_RESULT_ERROR_INVALID_OPERATION;
  case ZE_RESULT_ERROR_NOT_AVAILABLE:
    return UR_RESULT_ERROR_INVALID_OPERATION;
  case ZE_RESULT_ERROR_UNINITIALIZED:
    return UR_RESULT_ERROR_UNINITIALIZED;
  case ZE_RESULT_ERROR_INVALID_ARGUMENT:
    return UR_RESULT_ERROR_INVALID_ARGUMENT;
  case ZE_RESULT_ERROR_INVALID_NULL_POINTER:
    return UR_RESULT_ERROR_INVALID_VALUE;
  case ZE_RESULT_ERROR_INVALID_SIZE:
    return UR_RESULT_ERROR_INVALID_VALUE;
  case ZE_RESULT_ERROR_UNSUPPORTED_SIZE:
    return UR_RESULT_ERROR_INVALID_VALUE;
  case ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT:
    return UR_RESULT_ERROR_INVALID_VALUE;
  case ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT:
    return UR_RESULT_ERROR_INVALID_EVENT;
  case ZE_RESULT_ERROR_INVALID_ENUMERATION:
    return UR_RESULT_ERROR_INVALID_VALUE;
  case ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION:
    return UR_RESULT_ERROR_INVALID_VALUE;
  case ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT:
    return UR_RESULT_ERROR_INVALID_VALUE;
  case ZE_RESULT_ERROR_INVALID_NATIVE_BINARY:
    return UR_RESULT_ERROR_INVALID_BINARY;
  case ZE_RESULT_ERROR_INVALID_KERNEL_NAME:
  case ZE_RESULT_ERROR_INVALID_FUNCTION_NAME:
    return UR_RESULT_ERROR_INVALID_KERNEL_NAME;
  case ZE_RESULT_ERROR_OVERLAPPING_REGIONS:
    return UR_RESULT_ERROR_INVALID_OPERATION;
  case ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION:
    return UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE;
  case ZE_RESULT_ERROR_MODULE_BUILD_FAILURE:
    return UR_RESULT_ERROR_PROGRAM_BUILD_FAILURE;
  case ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY:
    return UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY;
  case ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY:
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  case ZE_RESULT_ERROR_UNSUPPORTED_FEATURE:
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  case ZE_RESULT_ERROR_MODULE_LINK_FAILURE:
    return UR_RESULT_ERROR_PROGRAM_LINK_FAILURE;
  default:
    return UR_RESULT_ERROR_UNKNOWN;
  }
}

// This function will ensure compatibility with both Linux and Windows for
// setting environment variables.
bool setEnvVar(const char *name, const char *value) {
#ifdef _WIN32
  int Res = _putenv_s(name, value);
#else
  int Res = setenv(name, value, 1);
#endif
  if (Res != 0) {
    logger::debug(
        "UR L0 Adapter was unable to set the environment variable: {}", name);
    return false;
  }
  return true;
}

ZeUSMImportExtension ZeUSMImport;

std::map<std::string, int> *ZeCallCount = nullptr;

inline void zeParseError(ze_result_t ZeError, const char *&ErrorString) {
  switch (ZeError) {
#define ZE_ERRCASE(ERR)                                                        \
  case ERR:                                                                    \
    ErrorString = "" #ERR;                                                     \
    break;

    ZE_ERRCASE(ZE_RESULT_SUCCESS)
    ZE_ERRCASE(ZE_RESULT_NOT_READY)
    ZE_ERRCASE(ZE_RESULT_ERROR_DEVICE_LOST)
    ZE_ERRCASE(ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY)
    ZE_ERRCASE(ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY)
    ZE_ERRCASE(ZE_RESULT_ERROR_MODULE_BUILD_FAILURE)
    ZE_ERRCASE(ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS)
    ZE_ERRCASE(ZE_RESULT_ERROR_NOT_AVAILABLE)
    ZE_ERRCASE(ZE_RESULT_ERROR_UNINITIALIZED)
    ZE_ERRCASE(ZE_RESULT_ERROR_UNSUPPORTED_VERSION)
    ZE_ERRCASE(ZE_RESULT_ERROR_UNSUPPORTED_FEATURE)
    ZE_ERRCASE(ZE_RESULT_ERROR_INVALID_ARGUMENT)
    ZE_ERRCASE(ZE_RESULT_ERROR_INVALID_NULL_HANDLE)
    ZE_ERRCASE(ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE)
    ZE_ERRCASE(ZE_RESULT_ERROR_INVALID_NULL_POINTER)
    ZE_ERRCASE(ZE_RESULT_ERROR_INVALID_SIZE)
    ZE_ERRCASE(ZE_RESULT_ERROR_UNSUPPORTED_SIZE)
    ZE_ERRCASE(ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT)
    ZE_ERRCASE(ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT)
    ZE_ERRCASE(ZE_RESULT_ERROR_INVALID_ENUMERATION)
    ZE_ERRCASE(ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION)
    ZE_ERRCASE(ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT)
    ZE_ERRCASE(ZE_RESULT_ERROR_INVALID_NATIVE_BINARY)
    ZE_ERRCASE(ZE_RESULT_ERROR_INVALID_GLOBAL_NAME)
    ZE_ERRCASE(ZE_RESULT_ERROR_INVALID_KERNEL_NAME)
    ZE_ERRCASE(ZE_RESULT_ERROR_INVALID_FUNCTION_NAME)
    ZE_ERRCASE(ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION)
    ZE_ERRCASE(ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION)
    ZE_ERRCASE(ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX)
    ZE_ERRCASE(ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE)
    ZE_ERRCASE(ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE)
    ZE_ERRCASE(ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE)
    ZE_ERRCASE(ZE_RESULT_ERROR_OVERLAPPING_REGIONS)
    ZE_ERRCASE(ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED)
    ZE_ERRCASE(ZE_RESULT_ERROR_UNKNOWN)

#undef ZE_ERRCASE
  default:
    assert(false && "Unexpected Error code");
  } // switch
}

ze_result_t ZeCall::doCall(ze_result_t ZeResult, const char *ZeName,
                           const char *ZeArgs, bool TraceError) {
  logger::debug("ZE ---> {}{}", ZeName, ZeArgs);

  if (ZeResult == ZE_RESULT_SUCCESS) {
    if (UrL0LeaksDebug) {
      ++(*ZeCallCount)[ZeName];
    }
    return ZE_RESULT_SUCCESS;
  }

  if (TraceError) {
    const char *ErrorString = "Unknown";
    zeParseError(ZeResult, ErrorString);
    logger::error("Error ({}) in {}", ErrorString, ZeName);
  }
  return ZeResult;
}

// Specializations for various L0 structures
template <> ze_structure_type_t getZeStructureType<ze_event_pool_desc_t>() {
  return ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
}
template <> ze_structure_type_t getZeStructureType<ze_fence_desc_t>() {
  return ZE_STRUCTURE_TYPE_FENCE_DESC;
}
template <> ze_structure_type_t getZeStructureType<ze_command_list_desc_t>() {
  return ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
}
template <>
ze_structure_type_t
getZeStructureType<ze_mutable_command_list_exp_properties_t>() {
  return ZE_STRUCTURE_TYPE_MUTABLE_COMMAND_LIST_EXP_PROPERTIES;
}
template <>
ze_structure_type_t getZeStructureType<ze_mutable_command_list_exp_desc_t>() {
  return ZE_STRUCTURE_TYPE_MUTABLE_COMMAND_LIST_EXP_DESC;
}
template <>
ze_structure_type_t getZeStructureType<ze_mutable_command_id_exp_desc_t>() {
  return ZE_STRUCTURE_TYPE_MUTABLE_COMMAND_ID_EXP_DESC;
}
template <>
ze_structure_type_t getZeStructureType<ze_mutable_group_count_exp_desc_t>() {
  return ZE_STRUCTURE_TYPE_MUTABLE_GROUP_COUNT_EXP_DESC;
}
template <>
ze_structure_type_t getZeStructureType<ze_mutable_group_size_exp_desc_t>() {
  return ZE_STRUCTURE_TYPE_MUTABLE_GROUP_SIZE_EXP_DESC;
}
template <>
ze_structure_type_t getZeStructureType<ze_mutable_global_offset_exp_desc_t>() {
  return ZE_STRUCTURE_TYPE_MUTABLE_GLOBAL_OFFSET_EXP_DESC;
}
template <>
ze_structure_type_t
getZeStructureType<ze_mutable_kernel_argument_exp_desc_t>() {
  return ZE_STRUCTURE_TYPE_MUTABLE_KERNEL_ARGUMENT_EXP_DESC;
}
template <>
ze_structure_type_t getZeStructureType<ze_mutable_commands_exp_desc_t>() {
  return ZE_STRUCTURE_TYPE_MUTABLE_COMMANDS_EXP_DESC;
}
template <> ze_structure_type_t getZeStructureType<ze_context_desc_t>() {
  return ZE_STRUCTURE_TYPE_CONTEXT_DESC;
}
template <>
ze_structure_type_t
getZeStructureType<ze_relaxed_allocation_limits_exp_desc_t>() {
  return ZE_STRUCTURE_TYPE_RELAXED_ALLOCATION_LIMITS_EXP_DESC;
}
template <>
ze_structure_type_t
getZeStructureType<ze_kernel_max_group_size_properties_ext_t>() {
  return ZE_STRUCTURE_TYPE_KERNEL_MAX_GROUP_SIZE_EXT_PROPERTIES;
}
template <> ze_structure_type_t getZeStructureType<ze_host_mem_alloc_desc_t>() {
  return ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
}
template <>
ze_structure_type_t getZeStructureType<ze_device_mem_alloc_desc_t>() {
  return ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
}
template <> ze_structure_type_t getZeStructureType<ze_command_queue_desc_t>() {
  return ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
}
template <> ze_structure_type_t getZeStructureType<ze_image_desc_t>() {
  return ZE_STRUCTURE_TYPE_IMAGE_DESC;
}
template <>
ze_structure_type_t getZeStructureType<ze_image_bindless_exp_desc_t>() {
  return ZE_STRUCTURE_TYPE_BINDLESS_IMAGE_EXP_DESC;
}
template <>
ze_structure_type_t getZeStructureType<ze_image_pitched_exp_desc_t>() {
  return ZE_STRUCTURE_TYPE_PITCHED_IMAGE_EXP_DESC;
}
template <> ze_structure_type_t getZeStructureType<ze_module_desc_t>() {
  return ZE_STRUCTURE_TYPE_MODULE_DESC;
}
template <>
ze_structure_type_t getZeStructureType<ze_module_program_exp_desc_t>() {
  return ZE_STRUCTURE_TYPE_MODULE_PROGRAM_EXP_DESC;
}
template <> ze_structure_type_t getZeStructureType<ze_kernel_desc_t>() {
  return ZE_STRUCTURE_TYPE_KERNEL_DESC;
}
template <> ze_structure_type_t getZeStructureType<ze_event_desc_t>() {
  return ZE_STRUCTURE_TYPE_EVENT_DESC;
}
template <> ze_structure_type_t getZeStructureType<ze_sampler_desc_t>() {
  return ZE_STRUCTURE_TYPE_SAMPLER_DESC;
}
template <> ze_structure_type_t getZeStructureType<ze_physical_mem_desc_t>() {
  return ZE_STRUCTURE_TYPE_PHYSICAL_MEM_DESC;
}
template <> ze_structure_type_t getZeStructureType<ze_driver_properties_t>() {
  return ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES;
}
template <> ze_structure_type_t getZeStructureType<ze_device_properties_t>() {
  return ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
}
template <>
ze_structure_type_t getZeStructureType<ze_device_p2p_properties_t>() {
  return ZE_STRUCTURE_TYPE_DEVICE_P2P_PROPERTIES;
}
template <>
ze_structure_type_t getZeStructureType<ze_device_compute_properties_t>() {
  return ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES;
}
template <>
ze_structure_type_t getZeStructureType<ze_command_queue_group_properties_t>() {
  return ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
}
template <>
ze_structure_type_t getZeStructureType<ze_device_image_properties_t>() {
  return ZE_STRUCTURE_TYPE_DEVICE_IMAGE_PROPERTIES;
}
template <>
ze_structure_type_t getZeStructureType<ze_device_module_properties_t>() {
  return ZE_STRUCTURE_TYPE_DEVICE_MODULE_PROPERTIES;
}
template <>
ze_structure_type_t getZeStructureType<ze_device_cache_properties_t>() {
  return ZE_STRUCTURE_TYPE_DEVICE_CACHE_PROPERTIES;
}
template <>
ze_structure_type_t getZeStructureType<ze_device_memory_properties_t>() {
  return ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES;
}
template <>
ze_structure_type_t getZeStructureType<ze_device_memory_ext_properties_t>() {
  return ZE_STRUCTURE_TYPE_DEVICE_MEMORY_EXT_PROPERTIES;
}
template <>
ze_structure_type_t getZeStructureType<ze_device_ip_version_ext_t>() {
  return ZE_STRUCTURE_TYPE_DEVICE_IP_VERSION_EXT;
}
template <>
ze_structure_type_t getZeStructureType<ze_device_memory_access_properties_t>() {
  return ZE_STRUCTURE_TYPE_DEVICE_MEMORY_ACCESS_PROPERTIES;
}
template <> ze_structure_type_t getZeStructureType<ze_module_properties_t>() {
  return ZE_STRUCTURE_TYPE_MODULE_PROPERTIES;
}
template <> ze_structure_type_t getZeStructureType<ze_kernel_properties_t>() {
  return ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES;
}
template <>
ze_structure_type_t getZeStructureType<ze_memory_allocation_properties_t>() {
  return ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES;
}

template <> ze_structure_type_t getZeStructureType<ze_pci_ext_properties_t>() {
  return ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES;
}

template <> zes_structure_type_t getZesStructureType<ze_pci_address_ext_t>() {
  return ZES_STRUCTURE_TYPE_PCI_PROPERTIES;
}

template <> zes_structure_type_t getZesStructureType<zes_mem_state_t>() {
  return ZES_STRUCTURE_TYPE_MEM_STATE;
}

template <> zes_structure_type_t getZesStructureType<zes_mem_properties_t>() {
  return ZES_STRUCTURE_TYPE_MEM_PROPERTIES;
}

#ifdef ZE_INTEL_DEVICE_BLOCK_ARRAY_EXP_NAME
template <>
ze_structure_type_t
getZeStructureType<ze_intel_device_block_array_exp_properties_t>() {
  return ZE_INTEL_DEVICE_BLOCK_ARRAY_EXP_PROPERTIES;
}
#endif // ZE_INTEL_DEVICE_BLOCK_ARRAY_EXP_NAME

// Global variables for ZER_EXT_RESULT_ADAPTER_SPECIFIC_ERROR
thread_local ur_result_t ErrorMessageCode = UR_RESULT_SUCCESS;
thread_local char ErrorMessage[MaxMessageSize];
thread_local int32_t ErrorAdapterNativeCode;

// Utility function for setting a message and warning
[[maybe_unused]] void setErrorMessage(const char *pMessage,
                                      ur_result_t ErrorCode,
                                      int32_t AdapterErrorCode) {
  assert(strlen(pMessage) <= MaxMessageSize);
  strcpy(ErrorMessage, pMessage);
  ErrorMessageCode = ErrorCode;
  ErrorAdapterNativeCode = AdapterErrorCode;
}

ur_result_t zerPluginGetLastError(char **message) {
  *message = &ErrorMessage[0];
  return ErrorMessageCode;
}
