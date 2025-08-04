//==--- itt_compiler_wrappers.cpp - compiler wrappers for ITT --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "device_itt.h"

#if defined(__SPIR__) || defined(__SPIRV__)

SYCL_EXTERNAL EXTERN_C void __itt_offload_wi_start_wrapper() {
  if (!isITTEnabled())
    return;

  size_t GroupID[3] = {__spirv_BuiltInWorkgroupId(0),
                       __spirv_BuiltInWorkgroupId(1),
                       __spirv_BuiltInWorkgroupId(2)};
  size_t WIID = __spirv_BuiltInGlobalLinearId();
  uint32_t WGSize = static_cast<uint32_t>(__spirv_BuiltInWorkgroupSize(0) *
                                          __spirv_BuiltInWorkgroupSize(1) *
                                          __spirv_BuiltInWorkgroupSize(2));
  __itt_offload_wi_start_stub(GroupID, WIID, WGSize);
}

SYCL_EXTERNAL EXTERN_C void __itt_offload_wi_finish_wrapper() {
  if (!isITTEnabled())
    return;

  size_t GroupID[3] = {__spirv_BuiltInWorkgroupId(0),
                       __spirv_BuiltInWorkgroupId(1),
                       __spirv_BuiltInWorkgroupId(2)};
  size_t WIID = __spirv_BuiltInGlobalLinearId();
  __itt_offload_wi_finish_stub(GroupID, WIID);
}

SYCL_EXTERNAL EXTERN_C void __itt_offload_wg_barrier_wrapper() {
  if (!isITTEnabled())
    return;

  __itt_offload_wg_barrier_stub(0);
}

SYCL_EXTERNAL EXTERN_C void __itt_offload_wi_resume_wrapper() {
  if (!isITTEnabled())
    return;

  size_t GroupID[3] = {__spirv_BuiltInWorkgroupId(0),
                       __spirv_BuiltInWorkgroupId(1),
                       __spirv_BuiltInWorkgroupId(2)};
  size_t WIID = __spirv_BuiltInGlobalLinearId();
  __itt_offload_wi_resume_stub(GroupID, WIID);
}

#endif // __SPIR__ || __SPIRV__
