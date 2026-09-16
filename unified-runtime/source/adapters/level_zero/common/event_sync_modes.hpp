//===--------- event_sync_modes.hpp - Level Zero Adapter ---------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <ze_api.h>

typedef uint32_t ze_intel_event_sync_mode_exp_flags_t;
typedef enum _ze_intel_event_sync_mode_exp_flag_t {
  ZE_INTEL_EVENT_SYNC_MODE_EXP_FLAG_LOW_POWER_WAIT = ZE_BIT(0),
  ZE_INTEL_EVENT_SYNC_MODE_EXP_FLAG_SIGNAL_INTERRUPT = ZE_BIT(1),
  ZE_INTEL_EVENT_SYNC_MODE_EXP_EXP_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_intel_event_sync_mode_exp_flag_t;

#define ZE_INTEL_STRUCTURE_TYPE_EVENT_SYNC_MODE_EXP_DESC                       \
  (ze_structure_type_t)0x00030016

typedef struct _ze_intel_event_sync_mode_exp_desc_t {
  ze_structure_type_t stype;
  const void *pNext;

  ze_intel_event_sync_mode_exp_flags_t syncModeFlags;
} ze_intel_event_sync_mode_exp_desc_t;
