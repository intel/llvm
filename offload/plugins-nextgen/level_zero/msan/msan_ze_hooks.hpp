//===--- msan_ze_hooks.hpp - Level Zero calls hooked by the MSan shim -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Level Zero calls intercepted by liboffload_msan_ze.so (the shim the
// level_zero plugin binds to as "libze_loader.so") and forwarded to MSan.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <level_zero/ze_api.h>

// X(Name, (Params...), (Args...))
#define MSAN_ZE_HOOKS(X)                                                       \
  X(zeMemAllocDevice,                                                          \
    (ze_context_handle_t hContext, const ze_device_mem_alloc_desc_t *DDesc,    \
     size_t Size, size_t Align, ze_device_handle_t hDevice, void **Ptr),       \
    (hContext, DDesc, Size, Align, hDevice, Ptr))                              \
  X(zeMemAllocHost,                                                            \
    (ze_context_handle_t hContext, const ze_host_mem_alloc_desc_t *HDesc,      \
     size_t Size, size_t Align, void **Ptr),                                   \
    (hContext, HDesc, Size, Align, Ptr))                                       \
  X(zeMemAllocShared,                                                          \
    (ze_context_handle_t hContext, const ze_device_mem_alloc_desc_t *DDesc,    \
     const ze_host_mem_alloc_desc_t *HDesc, size_t Size, size_t Align,         \
     ze_device_handle_t hDevice, void **Ptr),                                  \
    (hContext, DDesc, HDesc, Size, Align, hDevice, Ptr))                       \
  X(zeCommandListAppendMemoryCopy,                                             \
    (ze_command_list_handle_t hCmdList, void *Dst, const void *Src,            \
     size_t Size, ze_event_handle_t hSignal, uint32_t NumWait,                 \
     ze_event_handle_t *phWait),                                               \
    (hCmdList, Dst, Src, Size, hSignal, NumWait, phWait))                      \
  X(zeCommandListAppendMemoryFill,                                             \
    (ze_command_list_handle_t hCmdList, void *Ptr, const void *Pattern,        \
     size_t PatternSize, size_t Size, ze_event_handle_t hSignal,               \
     uint32_t NumWait, ze_event_handle_t *phWait),                             \
    (hCmdList, Ptr, Pattern, PatternSize, Size, hSignal, NumWait, phWait))     \
  X(zeModuleCreate,                                                            \
    (ze_context_handle_t hContext, ze_device_handle_t hDevice,                 \
     const ze_module_desc_t *Desc, ze_module_handle_t *phModule,               \
     ze_module_build_log_handle_t *phLog),                                     \
    (hContext, hDevice, Desc, phModule, phLog))                                \
  X(zeKernelCreate,                                                            \
    (ze_module_handle_t hModule, const ze_kernel_desc_t *Desc,                 \
     ze_kernel_handle_t *phKernel),                                            \
    (hModule, Desc, phKernel))                                                 \
  X(zeKernelSetGroupSize,                                                      \
    (ze_kernel_handle_t hKernel, uint32_t X, uint32_t Y, uint32_t Z),          \
    (hKernel, X, Y, Z))                                                        \
  X(zeCommandListAppendLaunchKernel,                                           \
    (ze_command_list_handle_t hCmdList, ze_kernel_handle_t hKernel,            \
     const ze_group_count_t *GroupCount, ze_event_handle_t hSignal,            \
     uint32_t NumWait, ze_event_handle_t *phWait),                             \
    (hCmdList, hKernel, GroupCount, hSignal, NumWait, phWait))

struct MsanZeTable {
#define MSAN_ZE_FIELD(N, P, A) decltype(&::N) N;
  MSAN_ZE_HOOKS(MSAN_ZE_FIELD)
#undef MSAN_ZE_FIELD
};

// Exported by the shim. Installs Hooks (nullptr = forward everything to the
// real loader) and returns the real loader's entry points.
extern "C" const MsanZeTable *offloadMsanZeSetHooks(const MsanZeTable *Hooks);
using offloadMsanZeSetHooks_t = decltype(&offloadMsanZeSetHooks);
