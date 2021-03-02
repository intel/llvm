//==------- device_itt.h - ITT devicelib functions declarations ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//==------------------------------------------------------------------------==//

#ifndef __LIBDEVICE_DEVICE_ITT_H__
#define __LIBDEVICE_DEVICE_ITT_H__

#include "device.h"

#ifdef __SPIR__
#include <cstddef>
#include <cstdint>

// Use SPIRV constants directly in place of OCL intrinsic functions.
#define __SPIRV_VAR_QUALIFIERS EXTERN_C const
typedef size_t size_t_vec __attribute__((ext_vector_type(3)));
__SPIRV_VAR_QUALIFIERS size_t __spirv_BuiltInGlobalLinearId;
__SPIRV_VAR_QUALIFIERS size_t_vec __spirv_BuiltInWorkgroupId;
__SPIRV_VAR_QUALIFIERS size_t_vec __spirv_BuiltInWorkgroupSize;

#define ITT_STUB_ATTRIBUTES __attribute__((noinline,optnone))

// FIXME: must be enabled via -fdeclare-spirv-builtins
DEVICE_EXTERN_C char __spirv_SpecConstant(int, char);

#define ITT_SPEC_CONSTANT 0xFF747469

static inline bool isITTEnabled() {
  return __spirv_SpecConstant(ITT_SPEC_CONSTANT, 0) != 0;
}

// Wrapper APIs that may be called by compiler-generated code.
DEVICE_EXTERN_C
void __itt_spirv_wi_start_wrapper();
DEVICE_EXTERN_C
void __itt_spirv_wi_finish_wrapper();
DEVICE_EXTERN_C
void __itt_spirv_wg_barrier_wrapper();
DEVICE_EXTERN_C
void __itt_spirv_wi_resume_wrapper();

// Non-inlinable and non-optimizable APIs that are recognized
// by profiling tools.
DEVICE_EXTERN_C ITT_STUB_ATTRIBUTES
void __itt_spirv_wi_start_stub(
    size_t *group_id, size_t wi_id, uint32_t wg_size);
DEVICE_EXTERN_C ITT_STUB_ATTRIBUTES
void __itt_spirv_wi_finish_stub(
    size_t *group_id, size_t wi_id);
DEVICE_EXTERN_C ITT_STUB_ATTRIBUTES
void __itt_spirv_wg_barrier_stub(uintptr_t barrier_id);
DEVICE_EXTERN_C ITT_STUB_ATTRIBUTES
void __itt_spirv_wi_resume_stub(size_t* group_id, size_t wi_id);

#endif // __SPIR__
#endif // __LIBDEVICE_DEVICE_ITT_H__
