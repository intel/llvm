//==--- itt_compiler_wrappers.cpp - compiler wrappers for ITT --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "device_itt.h"

#ifdef __SPIR__

DEVICE_EXTERN_C
void __itt_offload_wi_start_wrapper() {
  if (!isITTEnabled())
    return;

  size_t GroupID[3] = {__spirv_BuiltInWorkgroupId.x,
                       __spirv_BuiltInWorkgroupId.y,
                       __spirv_BuiltInWorkgroupId.z};
  size_t WIID = __spirv_BuiltInGlobalLinearId;
  uint32_t WGSize = static_cast<uint32_t>(__spirv_BuiltInWorkgroupSize.x *
                                          __spirv_BuiltInWorkgroupSize.y *
                                          __spirv_BuiltInWorkgroupSize.z);
  __itt_offload_wi_start_stub(GroupID, WIID, WGSize);
}

DEVICE_EXTERN_C
void __itt_offload_wi_finish_wrapper() {
  if (!isITTEnabled())
    return;

  size_t GroupID[3] = {__spirv_BuiltInWorkgroupId.x,
                       __spirv_BuiltInWorkgroupId.y,
                       __spirv_BuiltInWorkgroupId.z};
  size_t WIID = __spirv_BuiltInGlobalLinearId;
  __itt_offload_wi_finish_stub(GroupID, WIID);
}

DEVICE_EXTERN_C
void __itt_offload_wg_barrier_wrapper() {
  if (!isITTEnabled())
    return;

  __itt_offload_wg_barrier_stub(0);
}

DEVICE_EXTERN_C
void __itt_offload_wi_resume_wrapper() {
  if (!isITTEnabled())
    return;

  size_t GroupID[3] = {__spirv_BuiltInWorkgroupId.x,
                       __spirv_BuiltInWorkgroupId.y,
                       __spirv_BuiltInWorkgroupId.z};
  size_t WIID = __spirv_BuiltInGlobalLinearId;
  __itt_offload_wi_resume_stub(GroupID, WIID);
}

#endif // __SPIR__
