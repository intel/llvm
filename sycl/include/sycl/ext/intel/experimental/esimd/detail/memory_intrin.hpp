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

/// **************************** WARNING ************************************
/// When declaring new SPIR-V intrinsics (functions starting with __spirv),
/// it is imperitive to exactly follow the pattern of the existing SPIR-V
/// intrinsics. If not followed, the declaration may conflict with
/// the Clang-generated functions and cause compilation errors.
/// **************************** WARNING ************************************

#pragma once

#include <sycl/ext/intel/esimd/detail/defines_elementary.hpp>
#include <sycl/ext/intel/esimd/detail/memory_intrin.hpp>
#include <sycl/ext/intel/experimental/esimd/common.hpp>

// generic work-group split barrier
__ESIMD_INTRIN void
__esimd_sbarrier(__ESIMD_ENS::split_barrier_action flag) __ESIMD_INTRIN_END;

#ifdef __SYCL_DEVICE_ONLY__
// Create an explicit data and GPU scoreboard dependency.
__ESIMD_INTRIN void __esimd_wait(uint16_t value);
#endif // __SYCL_DEVICE_ONLY__

/// Memory fence.
/// Supported platforms: DG2, PVC
///
/// @tparam Kind is the Sfid shaded function.
/// @tparam FenceOp is the fence operation.
/// @tparam Scope is the operation scope.
/// @tparam N is the SIMD size of operation (the number of addresses to access)
/// @param pred is the predicate.
template <__ESIMD_NS::memory_kind Kind, __ESIMD_NS::fence_flush_op FenceOp,
          __ESIMD_NS::fence_scope Scope, int N>
__ESIMD_INTRIN void
__esimd_lsc_fence(__ESIMD_DNS::simd_mask_storage_t<N> pred) __ESIMD_INTRIN_END;

__ESIMD_INTRIN uint8_t __esimd_named_barrier_allocate(uint8_t NbarCount)
    __ESIMD_INTRIN_END;

/// 2D USM pointer block load.
/// Supported platforms: PVC
///
/// Collects elements located as described in the descriptor and returns them
/// as a single \ref simd object.
///
/// @tparam Ty is element type.
/// @tparam NBlocks is the number of blocks.
/// @tparam BlockWidth is the block width in number of elements.
/// @tparam BlockHeight is the block height in number of elements.
/// @tparam BlockXOffset is Memory block X immediate offset (in elements).
/// @tparam BlockYOffset is Memory block Y immediate offset (in elements).
/// @param Pred is the predicate.
/// @param Desc is the descriptor containing parameters for the operation.
/// @param PassThru is value to passthru when predicate is false on load.
/// @param Cache is vector containing cache hint information.
/// @return is a vector of type Ty
template <typename Ty, uint8_t NBlocks, uint8_t BlockWidth, uint8_t BlockHeight,
          uint32_t BlockXOffset, uint32_t BlockYOffset, int N>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<Ty, N> __esimd_lsc_load2d_descriptor(
    uint16_t Pred, __ESIMD_DNS::vector_type_t<uint32_t, 16> Desc,
    __ESIMD_DNS::vector_type_t<Ty, N> PassThru,
    __ESIMD_DNS::vector_type_t<uint8_t, 2> Cache) __ESIMD_INTRIN_END;

/// Collects elements located as described in the descriptor, performs
/// transposition and returns them as a single \ref simd object.
///
/// @tparam Ty is element type.
/// @tparam NBlocks is the number of blocks.
/// @tparam BlockWidth is the block width in number of elements.
/// @tparam BlockHeight is the block height in number of elements.
/// @tparam BlockXOffset is Memory block X immediate offset (in elements).
/// @tparam BlockYOffset is Memory block Y immediate offset (in elements).
/// @param Pred is the predicate.
/// @param Desc is the descriptor containing parameters for the operation.
/// @param PassThru is value to passthru when predicate is false on load.
/// @param Cache is vector containing cache hint information.
/// @return is a vector of type Ty
template <typename Ty, uint8_t NBlocks, uint8_t BlockWidth, uint8_t BlockHeight,
          uint32_t BlockXOffset, uint32_t BlockYOffset, int N>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<Ty, N>
__esimd_lsc_load2d_descriptor_transpose(
    uint16_t Pred, __ESIMD_DNS::vector_type_t<uint32_t, 16> Desc,
    __ESIMD_DNS::vector_type_t<Ty, N> PassThru,
    __ESIMD_DNS::vector_type_t<uint8_t, 2> Cache) __ESIMD_INTRIN_END;

/// Collects elements located as described in the descriptor, performs vnni
/// transform and returns them as a single \ref simd object.
///
/// @tparam Ty is element type.
/// @tparam NBlocks is the number of blocks.
/// @tparam BlockWidth is the block width in number of elements.
/// @tparam BlockHeight is the block height in number of elements.
/// @tparam BlockXOffset is Memory block X immediate offset (in elements).
/// @tparam BlockYOffset is Memory block Y immediate offset (in elements).
/// @param Pred is the predicate.
/// @param Desc is the descriptor containing parameters for the operation.
/// @param PassThru is value to passthru when predicate is false on load.
/// @param Cache is vector containing cache hint information.
/// @return is a vector of type Ty
template <typename Ty, uint8_t NBlocks, uint8_t BlockWidth, uint8_t BlockHeight,
          uint32_t BlockXOffset, uint32_t BlockYOffset, int N>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<Ty, N>
__esimd_lsc_load2d_descriptor_transform(
    uint16_t Pred, __ESIMD_DNS::vector_type_t<uint32_t, 16> Desc,
    __ESIMD_DNS::vector_type_t<Ty, N> PassThru,
    __ESIMD_DNS::vector_type_t<uint8_t, 2> Cache) __ESIMD_INTRIN_END;

/// 2D USM pointer block prefetch.
/// Supported platforms: PVC
///
/// Prefetches elements located as described in the descriptor.
///
/// @tparam Ty is element type.
/// @tparam NBlocks is the number of blocks.
/// @tparam BlockWidth is the block width in number of elements.
/// @tparam BlockHeight is the block height in number of elements.
/// @tparam BlockXOffset is Memory block X immediate offset (in elements).
/// @tparam BlockYOffset is Memory block Y immediate offset (in elements).
/// @param Pred is the predicate.
/// @param Desc is the descriptor containing parameters for the operation.
/// @param PassThru is dummy value to obtain type of the elements.
/// @param Cache is vector containing cache hint information.
template <typename Ty, uint8_t NBlocks, uint8_t BlockWidth, uint8_t BlockHeight,
          uint32_t BlockXOffset, uint32_t BlockYOffset, int N>
__ESIMD_INTRIN void __esimd_lsc_prefetch_descriptor(
    uint16_t Pred, __ESIMD_DNS::vector_type_t<uint32_t, 16> Desc,
    __ESIMD_DNS::vector_type_t<Ty, N> PassThru,
    __ESIMD_DNS::vector_type_t<uint8_t, 2> Cache) __ESIMD_INTRIN_END;

/// 2D USM pointer block store.
/// Supported platforms: PVC
///
/// Stores elements as described in the descriptor.
///
/// @tparam Ty is element type.
/// @tparam NBlocks is the number of blocks.
/// @tparam BlockWidth is the block width in number of elements.
/// @tparam BlockHeight is the block height in number of elements.
/// @tparam BlockXOffset is Memory block X immediate offset (in elements).
/// @tparam BlockYOffset is Memory block Y immediate offset (in elements).
/// @param Pred is the predicate.
/// @param Desc is the descriptor containing parameters for the operation.
/// @param Values is value to to store.
/// @param Cache is vector containing cache hint information.
template <typename Ty, uint8_t NBlocks, uint8_t BlockWidth, uint8_t BlockHeight,
          uint32_t BlockXOffset, uint32_t BlockYOffset, int N>
__ESIMD_INTRIN void __esimd_lsc_store_descriptor(
    uint16_t Pred, __ESIMD_DNS::vector_type_t<uint32_t, 16> Desc,
    __ESIMD_DNS::vector_type_t<Ty, N> Values,
    __ESIMD_DNS::vector_type_t<uint8_t, 2> Cache) __ESIMD_INTRIN_END;

/// @endcond ESIMD_DETAIL
