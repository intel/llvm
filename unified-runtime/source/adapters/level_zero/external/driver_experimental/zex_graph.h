// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef _ZEX_GRAPH_H
#define _ZEX_GRAPH_H
#if defined(__cplusplus)
#pragma once
#endif

#include <level_zero/ze_api.h>

#include "zex_common.h"

#ifndef ZE_RECORD_REPLAY_GRAPH_EXP_NAME
/// @brief Record and Replay Graph Extension Name
#define ZE_RECORD_REPLAY_GRAPH_EXP_NAME "ZE_experimental_record_replay_graph"

typedef enum _ze_record_replay_graph_exp_version_t {
  ZE_RECORD_REPLAY_GRAPH_EXP_VERSION_1_0 =
      ZE_MAKE_VERSION(1, 0), ///< version 1.0
  ZE_RECORD_REPLAY_GRAPH_EXP_VERSION_CURRENT =
      ZE_MAKE_VERSION(1, 0), ///< latest known version
  ZE_RECORD_REPLAY_GRAPH_EXP_VERSION_FORCE_UINT32 =
      0x7fffffff, ///< Value marking end of ZE_RECORD_REPLAY_GRAPH_EXP_VERSION_*
                  ///< ENUMs
} ze_record_replay_graph_exp_version_t;

typedef uint32_t ze_record_replay_graph_exp_flags_t;
typedef enum _ze_record_replay_graph_exp_flag_t {
  ZE_RECORD_REPLAY_GRAPH_EXP_FLAG_IMMUTABLE_GRAPH = ZE_BIT(0), ///< immutable
  ZE_RECORD_REPLAY_GRAPH_EXP_FLAG_MUTABLE_GRAPH = ZE_BIT(1),   ///< mutable
  ZE_RECORD_REPLAY_GRAPH_EXP_FLAG_FORCE_UINT32 =
      0x7fffffff, ///< Value marking end of ZE_RECORD_REPLAY_EXP_FLAG_* ENUMs
} ze_record_replay_graph_exp_flag_t;

typedef struct _ze_record_replay_graph_exp_properties_t {
  ze_structure_type_ext_t stype; ///< [in] type of this structure
  void *pNext; ///< [in,out][optional] must be null or a pointer to an
               ///< extension-specific
  ///< structure (i.e. contains stype and pNext).
  ze_record_replay_graph_exp_flags_t graphFlags; ///< [out] record replay flags
} ze_record_replay_graph_exp_properties_t;

typedef struct _ze_graph_handle_t *ze_graph_handle_t;
typedef struct _ze_executable_graph_handle_t *ze_executable_graph_handle_t;

typedef enum _ze_record_replay_graph_exp_dump_mode_t {
  ZE_RECORD_REPLAY_GRAPH_EXP_DUMP_MODE_DETAILED =
      0x0, ///< detailed mode (default)
  ZE_RECORD_REPLAY_GRAPH_EXP_DUMP_MODE_SIMPLE = 0x1, ///< simple mode
  ZE_RECORD_REPLAY_GRAPH_EXP_DUMP_MODE_FORCE_UINT32 =
      0x7fffffff, ///< Value marking end of
                  ///< ZE_RECORD_REPLAY_GRAPH_EXP_DUMP_MODE_* ENUMs
} ze_record_replay_graph_exp_dump_mode_t;

typedef struct _ze_record_replay_graph_exp_dump_desc_t {
  ze_structure_type_ext_t stype; ///< [in] type of this structure
  const void
      *pNext; ///< [in][optional] must be null or a pointer to an
              ///< extension-specific structure (i.e. contains stype and pNext).
  ze_record_replay_graph_exp_dump_mode_t mode; ///< [in] graph dump mode
} ze_record_replay_graph_exp_dump_desc_t;

#define ZE_RESULT_QUERY_TRUE EXTENDED_ENUM(ze_result_t, 0x7fff0000)
#define ZE_RESULT_QUERY_FALSE EXTENDED_ENUM(ze_result_t, 0x7fff0001)
#define ZE_RESULT_ERROR_INVALID_GRAPH EXTENDED_ENUM(ze_result_t, 0x7fff0002)

#if defined(__cplusplus)
extern "C" {
#endif

typedef void(ZE_CALLBACK *zex_mem_graph_free_callback_fn_t)(void *pUserData);

ze_result_t ZE_APICALL zeGraphCreateExp(ze_context_handle_t hContext,
                                        ze_graph_handle_t *phGraph,
                                        void *pNext);
ze_result_t ZE_APICALL zeCommandListBeginGraphCaptureExp(
    ze_command_list_handle_t hCommandList, void *pNext);
ze_result_t ZE_APICALL
zeCommandListBeginCaptureIntoGraphExp(ze_command_list_handle_t hCommandList,
                                      ze_graph_handle_t hGraph, void *pNext);
ze_result_t ZE_APICALL
zeCommandListEndGraphCaptureExp(ze_command_list_handle_t hCommandList,
                                ze_graph_handle_t *phGraph, void *pNext);
ze_result_t ZE_APICALL zeCommandListInstantiateGraphExp(
    ze_graph_handle_t hGraph, ze_executable_graph_handle_t *phExecutableGraph,
    void *pNext);
ze_result_t ZE_APICALL zeCommandListAppendGraphExp(
    ze_command_list_handle_t hCommandList, ze_executable_graph_handle_t hGraph,
    void *pNext, ze_event_handle_t hSignalEvent, uint32_t numWaitEvents,
    ze_event_handle_t *phWaitEvents);
ze_result_t ZE_APICALL zeGraphDestroyExp(ze_graph_handle_t hGraph);
ze_result_t ZE_APICALL
zeExecutableGraphDestroyExp(ze_executable_graph_handle_t hGraph);
ze_result_t ZE_APICALL
zeCommandListIsGraphCaptureEnabledExp(ze_command_list_handle_t hCommandList);
ze_result_t ZE_APICALL zeGraphIsEmptyExp(ze_graph_handle_t hGraph);
ze_result_t ZE_APICALL zeGraphDumpContentsExp(ze_graph_handle_t hGraph,
                                              const char *filePath,
                                              void *pNext);
ze_result_t ZE_APICALL zeCommandListGetGraphExp(
    ze_command_list_handle_t hCommandList, ze_graph_handle_t *phGraph);
ze_result_t ZE_APICALL zeGraphSetDestructionCallbackExp(
    ze_graph_handle_t hGraph, zex_mem_graph_free_callback_fn_t pfnCallback,
    void *pUserData, void *pNext);

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // ZE_RECORD_REPLAY_GRAPH_EXP_NAME
#endif // _ZEX_GRAPH_H
