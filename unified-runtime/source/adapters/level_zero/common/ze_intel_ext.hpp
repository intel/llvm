//===--------- ze_intel_ext.hpp - Level Zero Adapter ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cstdint>

#include <ze_api.h>

// UR copies of the Intel Level Zero driver experimental definitions used by
// the adapters, taken from the compute-runtime level_zero/include headers
// (ze_stypes.h, ze_intel_gpu.h and driver_experimental/zex_common.h). They are
// prefixed with ur_/UR_ so they don't clash with the original names once those
// get upstreamed to the Level Zero loader headers.

namespace ur::level_zero {

// Structure type of the Intel experimental extension structures.
using ur_ze_structure_type_ext_t = uint32_t;

///////////////////////////////////////////////////////////////////////////////
// ZE_intel_experimental_device_block_array_properties

constexpr ur_ze_structure_type_ext_t
    UR_ZE_INTEL_DEVICE_BLOCK_ARRAY_EXP_PROPERTIES = 0x00030007;

// Supported 2D block array flags
typedef uint32_t ur_ze_intel_device_block_array_exp_flags_t;
typedef enum _ur_ze_intel_device_block_array_exp_flag_t {
  UR_ZE_INTEL_DEVICE_EXP_FLAG_2D_BLOCK_STORE =
      ZE_BIT(0), ///< Supports store operation
  UR_ZE_INTEL_DEVICE_EXP_FLAG_2D_BLOCK_LOAD =
      ZE_BIT(1), ///< Supports load operation
  UR_ZE_INTEL_DEVICE_EXP_FLAG_2D_BLOCK_FORCE_UINT32 = 0x7fffffff
} ur_ze_intel_device_block_array_exp_flag_t;

// Device 2D block array properties, passed to zeDeviceGetProperties via the
// pNext member of ze_device_properties_t.
typedef struct _ur_ze_intel_device_block_array_exp_properties_t {
  ur_ze_structure_type_ext_t stype; ///< [in] type of this structure
  void *pNext; ///< [in,out][optional] must be null or a pointer to an
               ///< extension-specific structure
  ur_ze_intel_device_block_array_exp_flags_t
      flags; ///< [out] 0 (none) or a valid combination of
             ///< ur_ze_intel_device_block_array_exp_flag_t
} ur_ze_intel_device_block_array_exp_properties_t;

///////////////////////////////////////////////////////////////////////////////
// ZE_intel_experimental_xe_device_properties

constexpr ur_ze_structure_type_ext_t
    UR_ZE_STRUCTURE_TYPE_INTEL_XE_DEVICE_EXP_PROPERTIES = 0x00030021;

// Intel Xe device compute unit properties, passed to zeDeviceGetProperties via
// the pNext member of ze_device_properties_t.
typedef struct _ur_ze_intel_xe_device_exp_properties_t {
  ur_ze_structure_type_ext_t stype; ///< [in] type of this structure
  void *pNext;          ///< [in][optional] must be null or a pointer to an
                        ///< extension-specific structure
  uint32_t numXeStacks; ///< [out] number of Stacks (Tiles)
  uint32_t numXeRegionsPerStack;   ///< [out] number of Regions per stack
  uint32_t numXeClustersPerRegion; ///< [out] number of Clusters (Slices) per
                                   ///< Region
  uint32_t numXeCorePerCluster;    ///< [out] number of XE Cores per Cluster
  uint32_t numExecutionEnginesPerXeCore; ///< [out] number of Execution Engines
                                         ///< (EUs) per XE Core
  uint32_t maxNumHwThreadsPerExecutionEngine; ///< [out] maximal number of HW
                                              ///< threads per Execution Engine
  uint32_t maxNumLanesPerHwThread; ///< [out] maximal number of lanes (virtual
                                   ///< SIMD size) per HW thread
} ur_ze_intel_xe_device_exp_properties_t;

///////////////////////////////////////////////////////////////////////////////
// Queue priority levels (zeDeviceGetPriorityLevels)

constexpr ur_ze_structure_type_ext_t UR_ZE_STRUCTURE_TYPE_QUEUE_PRIORITY_DESC =
    0x00030028;

// Descriptor used for setting priority on command queues and immediate command
// lists, passed via the pNext member of ze_command_queue_desc_t.
typedef struct _ur_ze_queue_priority_desc_t {
  ur_ze_structure_type_ext_t stype; ///< [in] type of this structure
  const void *pNext; ///< [in][optional] must be null or a pointer to an
                     ///< extension-specific structure
  int priority;      ///< [in] priority of the queue
} ur_ze_queue_priority_desc_t;

///////////////////////////////////////////////////////////////////////////////
// ZEX_intel_experimental_queue_copy_operations_offload_hint (deprecated)

constexpr ur_ze_structure_type_ext_t
    UR_ZEX_INTEL_STRUCTURE_TYPE_QUEUE_COPY_OPERATIONS_OFFLOAD_HINT_EXP_PROPERTIES =
        0x0003001B;

typedef struct _ur_zex_intel_queue_copy_operations_offload_hint_exp_desc_t {
  ur_ze_structure_type_ext_t stype; ///< [in] type of this structure
  const void *pNext; ///< [in][optional] must be null or a pointer to an
                     ///< extension-specific structure
  ze_bool_t
      copyOffloadEnabled; ///< [in] If set, try to offload copy operations to
                          ///< different engines. Applicable only for compute
                          ///< queues. This is only a hint. Driver may ignore it
                          ///< per append call, based on platform capabilities
                          ///< or internal heuristics. If not set, driver will
                          ///< follow default behaviour. Copy operations will be
                          ///< submitted to same engine as compute operations.
} ur_zex_intel_queue_copy_operations_offload_hint_exp_desc_t;

///////////////////////////////////////////////////////////////////////////////
// ZEX_counter_based_event (deprecated, superseded by the core
// zeEventCounterBasedCreate API in Level Zero spec version 1.15)

constexpr ur_ze_structure_type_ext_t UR_ZEX_STRUCTURE_COUNTER_BASED_EVENT_DESC =
    0x0003001C;

// Supported event flags for defining counter-based event
typedef uint32_t ur_zex_counter_based_event_exp_flags_t;
typedef enum _ur_zex_counter_based_event_exp_flag_t {
  UR_ZEX_COUNTER_BASED_EVENT_FLAG_IMMEDIATE =
      ZE_BIT(0), ///< Counter-based event is used for immediate command lists
  UR_ZEX_COUNTER_BASED_EVENT_FLAG_NON_IMMEDIATE =
      ZE_BIT(1), ///< Counter-based event is used for non-immediate command
                 ///< lists
  UR_ZEX_COUNTER_BASED_EVENT_FLAG_HOST_VISIBLE =
      ZE_BIT(2), ///< Signals and waits are also visible to host
  UR_ZEX_COUNTER_BASED_EVENT_FLAG_IPC =
      ZE_BIT(3), ///< Event can be shared across processes for waiting
  UR_ZEX_COUNTER_BASED_EVENT_FLAG_KERNEL_TIMESTAMP =
      ZE_BIT(4), ///< Event contains kernel timestamps
  UR_ZEX_COUNTER_BASED_EVENT_FLAG_FORCE_UINT32 = 0x7fffffff
} ur_zex_counter_based_event_exp_flag_t;

typedef struct _ur_zex_counter_based_event_desc_t {
  ur_ze_structure_type_ext_t stype; ///< [in] type of this structure
  const void *pNext; ///< [in][optional] must be null or a pointer to an
                     ///< extension-specific structure
  ur_zex_counter_based_event_exp_flags_t
      flags; ///< [in] counter based event flags
  ze_event_scope_flags_t
      signalScope; ///< [in] scope of relevant cache hierarchies to flush on a
                   ///< signal action before the event is triggered
  ze_event_scope_flags_t
      waitScope; ///< [in] scope of relevant cache hierarchies to invalidate on
                 ///< a wait action after the event is complete
} ur_zex_counter_based_event_desc_t;

} // namespace ur::level_zero
