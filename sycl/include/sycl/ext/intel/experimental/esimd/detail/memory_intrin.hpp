//==------------ memory_intrin.hpp - DPC++ Explicit SIMD API ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Declares experimental memory Explicit SIMD intrinsics.
//===----------------------------------------------------------------------===//

/// @cond ESIMD_DETAIL

#pragma once

#include <sycl/ext/intel/esimd/detail/defines_elementary.hpp>
#include <sycl/ext/intel/esimd/detail/memory_intrin.hpp>

// generic work-group split barrier
__ESIMD_INTRIN void
__esimd_sbarrier(__ESIMD_ENS::split_barrier_action flag) __ESIMD_INTRIN_END;

#ifdef __SYCL_DEVICE_ONLY__
// Create an explicit data and GPU scoreboard dependency.
__ESIMD_INTRIN void __esimd_wait(uint16_t value);
#endif // __SYCL_DEVICE_ONLY__

/// Represents named barrier synchronization for a subgroup of threads.
/// Available only on PVC
///
/// @param mode  - is wait(0) or signal(1)
///
/// @param id  - barrier id
///
/// @param thread_count  - number of threads, ignored in 'wait' mode
__ESIMD_INTRIN void __esimd_nbarrier(uint8_t mode, uint8_t id,
                                     uint8_t thread_count) __ESIMD_INTRIN_END;

/// Initialize number of named barriers for a kernel
/// Available only on PVC
///
/// @param count  - number of named barriers
__ESIMD_INTRIN void __esimd_nbarrier_init(uint8_t count) __ESIMD_INTRIN_END;

/// Perform signal operation on named barriers
/// Available only on PVC
/// @param id - barrier id
///
/// @param thread_role - thread role
///
/// @param num_producers - number of producers
///
/// @param num_consumers - number of consumers
__ESIMD_INTRIN void
__esimd_nbarrier_arrive(uint8_t id, uint8_t thread_role, uint8_t num_producers,
                        uint8_t num_consumers) __ESIMD_INTRIN_END;

/// Memory fence.
/// Supported platforms: DG2, PVC
///
/// @tparam Kind is the Sfid shaded function.
/// @tparam FenceOp is the fence operation.
/// @tparam Scope is the operation scope.
/// @tparam N is the SIMD size of operation (the number of addresses to access)
/// @param pred is predicates.
template <__ESIMD_ENS::lsc_memory_kind Kind, __ESIMD_ENS::lsc_fence_op FenceOp,
          __ESIMD_ENS::lsc_scope Scope, int N>
__ESIMD_INTRIN void
__esimd_lsc_fence(__ESIMD_DNS::simd_mask_storage_t<N> pred) __ESIMD_INTRIN_END;

__ESIMD_INTRIN uint32_t __esimd_slm_alloc(uint32_t size) __ESIMD_INTRIN_END;

__ESIMD_INTRIN void __esimd_slm_free(uint32_t id) __ESIMD_INTRIN_END;

/// @endcond ESIMD_DETAIL
