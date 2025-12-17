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
/// @brief State of the variable object
typedef enum _zex_variable_state_t {
  ZEX_VARIABLE_STATE_DECLARED, ///< not associated to any operation
  ZEX_VARIABLE_STATE_DEFINED, ///< associated to an operation, but value not set
  ZEX_VARIABLE_STATE_INITIALIZED ///< value is set
} zex_variable_state_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Variable object flags
typedef uint32_t zex_variable_flags_t;
typedef enum _zex_variable_flag_t {
  ZEX_VARIABLE_FLAGS_NONE = 0,
  ZEX_VARIABLE_FLAGS_INPUT = ZE_BIT(0),  ///< used as input
  ZEX_VARIABLE_FLAGS_OUTPUT = ZE_BIT(1), ///< used as output
  ZEX_VARIABLE_FLAGS_TEMPORARY =
      ZE_BIT(2), ///< used as temporary (not visible to host) object
  ZEX_VARIABLE_FLAGS_SCALABLE = ZE_BIT(3), ///< variable is scalable
} zex_variable_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Variable object value type
typedef enum _zex_variable_type_t {
  ZEX_VARIABLE_TYPE_NONE,          ///< ZEX_VARIABLE_STATE_DECLARED
  ZEX_VARIABLE_TYPE_IMAGE,         ///< variable represents an image
  ZEX_VARIABLE_TYPE_VALUE,         ///< variable represents a value
  ZEX_VARIABLE_TYPE_BUFFER,        ///< variable represents a buffer
  ZEX_VARIABLE_TYPE_SAMPLER,       ///< variable represents a sampler
  ZEX_VARIABLE_TYPE_EVENT,         ///< variable represents an event
  ZEX_VARIABLE_TYPE_PARAM,         ///< variable represents host param
  ZEX_VARIABLE_TYPE_GROUP_SIZE,    ///< variable represents group size
  ZEX_VARIABLE_TYPE_GROUP_COUNT,   ///< variable represents group count
  ZEX_VARIABLE_TYPE_GLOBAL_OFFSET, ///< variable represents global offset
  ZEX_VARIABLE_TYPE_SIGNAL_EVENT,  ///< variable represents signal event
  ZEX_VARIABLE_TYPE_WAIT_EVENT,    ///< variable represents wait events
  ZEX_VARIABLE_TYPE_SLM_BUFFER,    ///< variable represents slm buffer
} zex_variable_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Variable information
typedef struct _zex_variable_info_t {
  ze_structure_type_ext_t stype =
      ZEX_STRUCTURE_TYPE_VARIABLE_INFO; ///< type of this structure
  const void *pNext =
      nullptr; ///< [optional] pointer to extension-specific structure

  zex_variable_handle_t handle = nullptr; ///< handle to the variable
  const char *name = nullptr;             ///< name of the variable
  size_t size = 0U;                       ///< size of the variable
  zex_variable_state_t state =
      ZEX_VARIABLE_STATE_DECLARED; ///< current state of the variable
  zex_variable_flags_t flags =
      ZEX_VARIABLE_FLAGS_NONE; ///< flags providing additional metadata about
                               ///< the variable
  zex_variable_type_t type =
      ZEX_VARIABLE_TYPE_NONE; ///< type of value represented by this variable
} zex_variable_info_t;

#if defined(__cplusplus)
extern "C" {
#endif

///////////////////////////////////////////////////////////////////////////////
/// @brief Variable information
///
/// @details
///     - Variable information represents set of metadata describing a variable
///     - Retrieved zex_variable_info_t lifetime is bound to lifetime of the
///     commandlist (zex_variable_info_t is not being copied into caller memory)
ze_result_t ZE_APICALL zexVariableGetInfo(
    zex_variable_handle_t hVariable, // [in] handle to the variable
    const zex_variable_info_t *
        *pTypeInfo // [in] pointer where pointer to requested variable info
                   // should be copied to
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieve list of all variables tied to a command list
///
/// @details
///     - pVariables should be big enough to hold pointers to variables
///     information representing all variables in the command list
///     - it's expected that user first calls zexCommandListGetVariablesList
///     with pVariables==nullptr to query total number of variables in the
///     command list.
///       Next, user should allocate big enough array and use it with second
///       call to zexCommandListGetVariablesList
///     - lifetime of zex_variable_info_t pointed to by pointers in pVariables
///     is tied to lifetime of the command list (similarly to
///     zexVariableGetInfo)
ze_result_t ZE_APICALL zexCommandListGetVariablesList(
    ze_command_list_handle_t
        hCommandList,          // [in] handle to the command list object
    uint32_t *pVariablesCount, // [in,out] number of variables defined in the
                               // command list
    const zex_variable_info_t *
        *pVariablesInfos // [in,out][optional] array to which pointers to
                         // variables information should be copied to
);

typedef ze_result_t(ZE_APICALL *zex_pfnCommandListGetVariableCb_t)(
    ze_command_list_handle_t hCmdList,
    const zex_variable_desc_t *pVariableDescriptor,
    zex_variable_handle_t *phVariable);

typedef ze_result_t(ZE_APICALL *zex_pfnCommandListGetVariablesListCb_t)(
    ze_command_list_handle_t hCommandList, uint32_t *pVariablesCount,
    const zex_variable_info_t **pVariablesInfos);

#if defined(__cplusplus)
} // extern "C"
#endif
