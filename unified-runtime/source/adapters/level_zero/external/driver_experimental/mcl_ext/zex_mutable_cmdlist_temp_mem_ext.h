/*
 * Copyright (C) 2025 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */

#pragma once

#include "level_zero/driver_experimental/mcl_ext/zex_mutable_cmdlist_variable_ext.h"
#include "level_zero/ze_stypes.h"

///////////////////////////////////////////////////////////////////////////////
/// @brief Temporary variable flags
typedef uint32_t zex_temp_variable_flags_t;
typedef enum _zex_temp_variable_flag_t {
  ZEX_TEMP_VARIABLE_FLAGS_IS_CONST_SIZE =
      ZE_BIT(0), ///< temp variable has constant size
  ZEX_TEMP_VARIABLE_FLAGS_IS_SCALABLE =
      ZE_BIT(1) ///< temp variable scales with num elements
} zex_temp_variable_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Temporary variable type
typedef enum _zex_temp_variable_type_t {
  ZEX_TEMP_VARIABLE_TYPE_CONST_SIZE = 0, ///< temp variable has constant size
  ZEX_TEMP_VARIABLE_TYPE_SCALABLE =
      1 ///< temp variable scales with num elements
} zex_temp_variable_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Temporary variable descriptor
typedef struct _zex_temp_variable_desc_t {
  ze_structure_type_ext_t stype = ZEX_STRUCTURE_TYPE_TEMP_VARIABLE_DESCRIPTOR;
  const void *pNext = nullptr;

  zex_temp_variable_flags_t flags;
  size_t size;
} zex_temp_variable_desc_t;

#if defined(__cplusplus)
extern "C" {
#endif

///////////////////////////////////////////////////////////////////////////////
/// @brief Sets number of elements used by scalable temporary variables.
ze_result_t ZE_APICALL zexCommandListTempMemSetEleCount(
    ze_command_list_handle_t
        hCommandList, ///< [in] handle of mutable command list
    size_t eleCount); ///< [in] number of elements

///////////////////////////////////////////////////////////////////////////////
/// @brief Returns size of the buffer required for all temporary variables.
ze_result_t ZE_APICALL zexCommandListTempMemGetSize(
    ze_command_list_handle_t
        hCommandList,      ///< [in] handle of mutable command list
    size_t *pTempMemSize); ///< [in,out] size of temporary memory buffer

///////////////////////////////////////////////////////////////////////////////
/// @brief Sets buffer used by temporary variables.
ze_result_t ZE_APICALL zexCommandListTempMemSet(
    ze_command_list_handle_t
        hCommandList,      ///< [in] handle of mutable command list
    const void *pTempMem); ///< [in] ptr to temporary memory buffer

typedef ze_result_t(ZE_APICALL *zex_pfnCommandListTempMemSetEleCountCb_t)(
    ze_command_list_handle_t hCommandList, size_t eleCount);

typedef ze_result_t(ZE_APICALL *zex_pfnCommandListTempMemGetSizeCb_t)(
    ze_command_list_handle_t hCommandList, size_t *pTempMemSize);

typedef ze_result_t(ZE_APICALL *zex_pfnCommandListTempMemSetCb_t)(
    ze_command_list_handle_t hCommandList, const void *pTempMem);

#if defined(__cplusplus)
} // extern "C"
#endif
