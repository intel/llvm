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
#include "spirv_vars.h"

#define ITT_STUB_ATTRIBUTES __attribute__((noinline, optnone))
#define ITT_WRAPPER_ATTRIBUTES __attribute__((always_inline))

/// Atomic operation type
enum __itt_atomic_mem_op_t {
  __itt_mem_load = 0,
  __itt_mem_store = 1,
  __itt_mem_update = 2
};

/// Memory operation ordering semantic type
enum __itt_atomic_mem_order_t {
  __itt_mem_order_relaxed = 0,
  __itt_mem_order_acquire = 1,
  __itt_mem_order_release = 2,
  __itt_mem_order_acquire_release = 3
};

// FIXME: must be enabled via -fdeclare-spirv-builtins
DEVICE_EXTERN_C char __spirv_SpecConstant(int, char);

#define ITT_SPEC_CONSTANT 0xFF747469

static ITT_WRAPPER_ATTRIBUTES bool isITTEnabled() {
  return __spirv_SpecConstant(ITT_SPEC_CONSTANT, 0) != 0;
}

// Wrapper APIs that may be called by compiler-generated code.
// These are just parameterless helper APIs that call the corresponding
// stub APIs after preparing the arguments for them.
//
// Note that we do not provide compiler wrappers for all stub APIs.
// For example, there is no compiler wrapper for
// __itt_offload_sync_acquired_stub, since the API's parameter cannot
// be computed in the wrapper itself and has to be passed from outside.
// If a compiler needs to invoke such an API, it has to use the user
// visible API directly (i.e. __itt_offload_sync_acquired).
//
// FIXME: we need to add always_inline compiler wrappers
//        for atomic_op_start/finish. Compiler calls user
//        wrappers right now, and they may interfere with
//        debugging user code in non-ITT mode.
DEVICE_EXTERN_C ITT_WRAPPER_ATTRIBUTES void __itt_offload_wi_start_wrapper();
DEVICE_EXTERN_C ITT_WRAPPER_ATTRIBUTES void __itt_offload_wi_finish_wrapper();
DEVICE_EXTERN_C ITT_WRAPPER_ATTRIBUTES void __itt_offload_wg_barrier_wrapper();
DEVICE_EXTERN_C ITT_WRAPPER_ATTRIBUTES void __itt_offload_wi_resume_wrapper();

// Non-inlinable and non-optimizable APIs that are recognized
// by profiling tools.
DEVICE_EXTERN_C ITT_STUB_ATTRIBUTES void
__itt_offload_wi_start_stub(size_t *group_id, size_t wi_id, uint32_t wg_size);
DEVICE_EXTERN_C ITT_STUB_ATTRIBUTES void
__itt_offload_wi_finish_stub(size_t *group_id, size_t wi_id);
DEVICE_EXTERN_C ITT_STUB_ATTRIBUTES void
__itt_offload_wg_barrier_stub(uintptr_t barrier_id);
DEVICE_EXTERN_C ITT_STUB_ATTRIBUTES void
__itt_offload_wi_resume_stub(size_t *group_id, size_t wi_id);
DEVICE_EXTERN_C ITT_STUB_ATTRIBUTES void
__itt_offload_sync_acquired_stub(uintptr_t sync_id);
DEVICE_EXTERN_C ITT_STUB_ATTRIBUTES void
__itt_offload_sync_releasing_stub(uintptr_t sync_id);
DEVICE_EXTERN_C ITT_STUB_ATTRIBUTES void
__itt_offload_wg_local_range_stub(void *ptr, size_t size);
DEVICE_EXTERN_C ITT_STUB_ATTRIBUTES void
__itt_offload_atomic_op_start_stub(void *object, __itt_atomic_mem_op_t op_type,
                                   __itt_atomic_mem_order_t mem_order);
DEVICE_EXTERN_C ITT_STUB_ATTRIBUTES void
__itt_offload_atomic_op_finish_stub(void *object, __itt_atomic_mem_op_t op_type,
                                    __itt_atomic_mem_order_t mem_order);

// User visible APIs. These may called both from user code and from
// compiler generated code.
DEVICE_EXTERN_C void __itt_offload_wi_start(size_t *group_id, size_t wi_id,
                                            uint32_t wg_size);
DEVICE_EXTERN_C void __itt_offload_wi_finish(size_t *group_id, size_t wi_id);
DEVICE_EXTERN_C void __itt_offload_wg_barrier(uintptr_t barrier_id);
DEVICE_EXTERN_C void __itt_offload_wi_resume(size_t *group_id, size_t wi_id);
DEVICE_EXTERN_C void __itt_offload_sync_acquired(uintptr_t sync_id);
DEVICE_EXTERN_C void __itt_offload_sync_releasing(uintptr_t sync_id);
DEVICE_EXTERN_C void __itt_offload_wg_local_range(void *ptr, size_t size);
DEVICE_EXTERN_C void
__itt_offload_atomic_op_start(void *object, __itt_atomic_mem_op_t op_type,
                              __itt_atomic_mem_order_t mem_order);
DEVICE_EXTERN_C void
__itt_offload_atomic_op_finish(void *object, __itt_atomic_mem_op_t op_type,
                               __itt_atomic_mem_order_t mem_order);

#endif // __SPIR__
#endif // __LIBDEVICE_DEVICE_ITT_H__
