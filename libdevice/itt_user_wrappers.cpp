//==--- itt_user_wrappers.cpp - user visible functions for ITT  ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "device_itt.h"

#ifdef __SPIR__

DEVICE_EXTERN_C void __itt_offload_wi_start(size_t *group_id, size_t wi_id,
                                            uint32_t wg_size) {
  if (isITTEnabled())
    __itt_offload_wi_start_stub(group_id, wi_id, wg_size);
}

DEVICE_EXTERN_C void __itt_offload_wi_finish(size_t *group_id, size_t wi_id) {
  if (isITTEnabled())
    __itt_offload_wi_finish_stub(group_id, wi_id);
}

DEVICE_EXTERN_C void __itt_offload_wg_barrier(uintptr_t barrier_id) {
  if (isITTEnabled())
    __itt_offload_wg_barrier_stub(barrier_id);
}

DEVICE_EXTERN_C void __itt_offload_wi_resume(size_t *group_id, size_t wi_id) {
  if (isITTEnabled())
    __itt_offload_wi_resume_stub(group_id, wi_id);
}

DEVICE_EXTERN_C void __itt_offload_sync_acquired(uintptr_t sync_id) {
  if (isITTEnabled())
    __itt_offload_sync_acquired_stub(sync_id);
}

DEVICE_EXTERN_C void __itt_offload_sync_releasing(uintptr_t sync_id) {
  if (isITTEnabled())
    __itt_offload_sync_releasing_stub(sync_id);
}

DEVICE_EXTERN_C void __itt_offload_wg_local_range(void *ptr, size_t size) {
  if (isITTEnabled())
    __itt_offload_wg_local_range_stub(ptr, size);
}

DEVICE_EXTERN_C void
__itt_offload_atomic_op_start(void *object, __itt_atomic_mem_op_t op_type,
                              __itt_atomic_mem_order_t mem_order) {
  if (isITTEnabled())
    __itt_offload_atomic_op_start_stub(object, op_type, mem_order);
}

DEVICE_EXTERN_C void
__itt_offload_atomic_op_finish(void *object, __itt_atomic_mem_op_t op_type,
                               __itt_atomic_mem_order_t mem_order) {
  if (isITTEnabled())
    __itt_offload_atomic_op_finish_stub(object, op_type, mem_order);
}

#endif // __SPIR__
