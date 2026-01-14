/*
 * Copyright (C) 2025 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */

#pragma once

#include "level_zero/ze_stypes.h"
#include <level_zero/ze_api.h>

///////////////////////////////////////////////////////////////////////////////
/// @brief Label handle.
typedef struct _zex_label_handle_t *zex_label_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Label descriptor.
typedef struct _zex_label_desc_t {
  ze_structure_type_ext_t stype =
      ZEX_STRUCTURE_TYPE_LABEL_DESCRIPTOR; ///< [in] type of this structure
  const void *pNext =
      nullptr; ///< [in][optional] pointer to extension-specific structure

  const char *name;   ///< [in][optional] null-terminated name of the label
  uint32_t alignment; ///< [in][optional] minimum alignment of the label
} zex_label_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Operand flags
typedef uint32_t zex_operand_flags_t;
typedef enum _zex_operand_flag_t {
  ZEX_OPERAND_FLAG_USES_VARIABLE =
      ZE_BIT(0), // variable is being used - passed via memory
  ZEX_OPERAND_FLAG_JUMP_ON_CLEAR =
      ZE_BIT(1) // jump on '0', instead of default '1'
} zex_operand_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Operand descriptor.
typedef struct _zex_operand_desc_t {
  ze_structure_type_ext_t stype =
      ZEX_STRUCTURE_TYPE_OPERAND_DESCRIPTOR; ///< [in] type of this structure
  const void *pNext =
      nullptr; ///< [in][optional] pointer to extension-specific structure

  void *memory;  // if memory is NULL then offset is interpreted as MMIO
                 // if flag ZEX_OPERAND_FLAG_USES_VARIABLE is set memory is
                 // interpreted as variable
  size_t offset; // offset within memory or register MMIO
  uint32_t size; // operand size - dword
  zex_operand_flags_t flags; // flags
} zex_operand_desc_t;

#if defined(__cplusplus)
extern "C" {
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Creates/returns label based on name provided in
///        label descriptor.
///
/// @details
///     - When label with the name provided in label descriptor does not
///       exist new label is created.
///     - If label with provided name exists it's returned.
///     - Label at creation is undefined (points to nothing).
///
/// @returns
///     - ZE_RESULT_SUCCESS
///     - ZE_RESULT_ERROR_INVALID_ARGUMENT
///         + nullptr == hCommandList
///         + nullptr == pLabelDesc
///         + nullptr == phLabel
///         + pLabelDesc->alignment & (pLabelDesc->alignment - 1) != 0
///     - ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///         + bad_alloc
ze_result_t ZE_APICALL zexCommandListGetLabel(
    ze_command_list_handle_t hCommandList, ///< [in] handle of command list
    const zex_label_desc_t *pLabelDesc,    ///< [in] pointer to label descriptor
    zex_label_handle_t *phLabel            ///< [out] pointer to handle of label
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Sets label to point to current location in command list.
///
/// @details
///     - Sets label address to current location.
///     - All previous jumps to this label are patched with label's address.
///     - Future jumps to this label will be patched with label's address.
///     - Label can be set only once.
///     - Label must be located in the same command list, as it was created
///
/// @returns
///     - ZE_RESULT_SUCCESS
///     - ZE_RESULT_ERROR_INVALID_ARGUMENT
///         + nullptr == hCommandList
///         + nullptr == hLabel
///     - ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + variable already set
///     - ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT
///         + label's address is not aligned to it's alignment
ze_result_t ZE_APICALL zexCommandListSetLabel(
    ze_command_list_handle_t hCommandList, ///< [in] handle of command list
    zex_label_handle_t hLabel              ///< [in] handle of label
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Appends jump to label predicated by condition.
///
/// @details
///     If condition is present:
///        If condition->memory is present:
///            If ZEX_OPERAND_FLAG_USES_VARIABLE flag is set:
///              - Append MI_LOAD_REGISTER_MEM
///                 reg = PREDICATE_RESULT_2
///                 memAddr = gpu address of buffer variable
///            Else:
///              - Append MI_LOAD_REGISTER_MEM
///                reg = PREDICATE_RESULT_2
///                memAddr = pCondition->memory + pCondition->offset
///        Else:
///          - Append MI_LOAD_REGISTER_REG
///            regDst = PREDICATE_RESULT_2
///            regSrc = pCondition->offset (need to pass MMIO)
///
///        If ZEX_OPERAND_FLAG_JUMP_ON_CLEAR flag is set: //  jumps when '0'
///          - Append MI_SET_PREDICATE(ENABLE_ON_SET)
///        Else:                                          //  jumps when '1'
///          - Append MI_SET_PREDICATE(ENABLE_ON_CLEAR)
///
///     - Append MI_BATCH_BUFFER_START
///       predicationEnabled = condition is present
///       jumpAddress = label's address
///     - Append MI_SET_PREDICATE(DISABLE) - after BB_START
///     - Append MI_SET_PREDICATE(DISABLE) - at label's address
///
/// @returns
///     - ZE_RESULT_SUCCESS
///     - ZE_RESULT_ERROR_INVALID_ARGUMENT
///         + nullptr == hCommandList
///         + nullptr == hLabel
ze_result_t ZE_APICALL zexCommandListAppendJump(
    ze_command_list_handle_t hCommandList, ///< [in] handle of command list
    zex_label_handle_t hLabel,             ///< [in] handle of label
    zex_operand_desc_t *pCondition         ///< [in][opt] pointer to operand
);

typedef ze_result_t(ZE_APICALL *zex_pfnCommandListGetLabelCb_t)(
    ze_command_list_handle_t hCommandList, const zex_label_desc_t *pLabelDesc,
    zex_label_handle_t *phLabel);

typedef ze_result_t(ZE_APICALL *zex_pfnCommandListSetLabelCb_t)(
    ze_command_list_handle_t hCommandList, zex_label_handle_t hLabel);

typedef ze_result_t(ZE_APICALL *zex_pfnCommandListAppendJumpCb_t)(
    ze_command_list_handle_t hCommandList, zex_label_handle_t hLabel,
    zex_operand_desc_t *pCondition);

#if defined(__cplusplus)
} // extern "C"
#endif
