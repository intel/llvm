/*
 * Copyright (C) 2020-2025 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */

#ifndef _ZEX_MODULE_H
#define _ZEX_MODULE_H
#if defined(__cplusplus)
#pragma once
#endif

#include <level_zero/ze_api.h>

#if defined(__cplusplus)
extern "C" {
#endif

ZE_APIEXPORT ze_result_t ZE_APICALL
zexKernelGetBaseAddress(ze_kernel_handle_t hKernel, uint64_t *baseAddress);

ZE_APIEXPORT ze_result_t ZE_APICALL zexKernelGetArgumentSize(
    ze_kernel_handle_t hKernel, uint32_t argIndex, uint32_t *pArgSize);

ZE_APIEXPORT ze_result_t ZE_APICALL
zexKernelGetArgumentType(ze_kernel_handle_t hKernel, uint32_t argIndex,
                         uint32_t *pSize, char *pString);

#if defined(__cplusplus)
} // extern "C"
#endif

///////////////////////////////////////////////////////////////////////////////
/// @brief General Register File descriptor.
/// Must be passed to zeDeviceGetModuleProperties via pNext member of
/// ze_device_module_properties_t.
typedef struct _zex_device_module_register_file_exp_t {
  ze_structure_type_ext_t stype; ///< [in] type of this structure
  const void
      *pNext; ///< [in, out][optional] pointer to extension-specific structure
  uint32_t
      registerFileSizesCount; ///< [out] Size of array of supported GRF sizes
  uint32_t
      *registerFileSizes; ///< [in, out][optional] Array of supported GRF sizes
} zex_device_module_register_file_exp_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Kernel register file size information
/// Must be passed to zeKernelGetProperties via pNext member of
/// ze_kernel_properties_t
typedef struct _zex_kernel_register_file_size_exp_t {
  ze_structure_type_ext_t stype; ///< [in] type of this structure
  const void
      *pNext; ///< [in, out][optional] pointer to extension-specific structure
  uint32_t registerFileSize; ///< [out] Register file size used in kernel
} zex_kernel_register_file_size_exp_t;

#endif // _ZEX_MODULE_H
