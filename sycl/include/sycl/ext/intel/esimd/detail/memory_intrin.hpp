//==------------ memory_intrin.hpp - DPC++ Explicit SIMD API ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Declares Explicit SIMD intrinsics used to implement working with
// the SIMD classes objects.
//===----------------------------------------------------------------------===//

/// @cond ESIMD_DETAIL

#pragma once

#include <sycl/accessor.hpp>
#include <sycl/ext/intel/esimd/common.hpp>
#include <sycl/ext/intel/esimd/detail/types.hpp>
#include <sycl/ext/intel/esimd/detail/util.hpp>
#include <sycl/types.hpp>

#include <cstdint>

#ifndef __SYCL_DEVICE_ONLY__
// ESIMD_CPU Emulation support using esimd_cpu plugin

#include <sycl/backend_types.hpp>
#include <sycl/detail/pi.hpp>
#include <sycl/ext/intel/esimd/detail/atomic_intrin.hpp>
#include <sycl/ext/intel/esimd/emu/detail/esimd_emulator_device_interface.hpp>

// Channel Mask Array for scaled-gather/scatter
const std::array<__ESIMD_NS::rgba_channel, 4> ChannelMaskArray{
    __ESIMD_NS::rgba_channel::R, __ESIMD_NS::rgba_channel::G,
    __ESIMD_NS::rgba_channel::B, __ESIMD_NS::rgba_channel::A};

#endif // ifndef __SYCL_DEVICE_ONLY__

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext::intel::esimd::detail {

// Provides access to sycl accessor class' private members.
class AccessorPrivateProxy {
public:
  template <typename AccessorTy>
  static auto getNativeImageObj(const AccessorTy &Acc) {
#ifdef __SYCL_DEVICE_ONLY__
    return Acc.getNativeImageObj();
#else  // __SYCL_DEVICE_ONLY__
    return Acc;
#endif // __SYCL_DEVICE_ONLY__
  }
#ifndef __SYCL_DEVICE_ONLY__
  static void *getPtr(const sycl::detail::AccessorBaseHost &Acc) {
    return Acc.getPtr();
  }
#endif // __SYCL_DEVICE_ONLY__
};

template <int ElemsPerAddr,
          typename = std::enable_if_t<(ElemsPerAddr == 1 || ElemsPerAddr == 2 ||
                                       ElemsPerAddr == 4)>>
constexpr unsigned int ElemsPerAddrEncoding() {
  // encoding requires log2 of ElemsPerAddr
  if constexpr (ElemsPerAddr == 1)
    return 0;
  else if constexpr (ElemsPerAddr == 2)
    return 1;
  else if constexpr (ElemsPerAddr == 4)
    return 2;

  // other cases not needed since enable_if disallows other values
}

constexpr unsigned int ElemsPerAddrDecoding(unsigned int ElemsPerAddrEncoded) {
  // encoding requires 2^ElemsPerAddrEncoded
  return (1 << ElemsPerAddrEncoded);
}

} // namespace ext::intel::esimd::detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl

// flat_read does flat-address gather
template <typename Ty, int N, int NumBlk = 0, int ElemsPerAddr = 0>
__ESIMD_INTRIN
    __ESIMD_DNS::vector_type_t<Ty,
                               N * __ESIMD_DNS::ElemsPerAddrDecoding(NumBlk)>
    __esimd_svm_gather(__ESIMD_DNS::vector_type_t<uint64_t, N> addrs,
                       __ESIMD_DNS::simd_mask_storage_t<N> pred = 1)
#ifdef __SYCL_DEVICE_ONLY__
        ;
#else
{
  auto NumBlkDecoded = __ESIMD_DNS::ElemsPerAddrDecoding(NumBlk);
  __ESIMD_DNS::vector_type_t<Ty, N * __ESIMD_DNS::ElemsPerAddrDecoding(NumBlk)>
      V = 0;
  auto ElemsPerAddrDecoded = __ESIMD_DNS::ElemsPerAddrDecoding(ElemsPerAddr);
  if (sizeof(Ty) == 2)
    ElemsPerAddrDecoded = ElemsPerAddrDecoded / 2;

  for (int I = 0; I < N; I++) {
    if (pred[I]) {
      Ty *Addr = reinterpret_cast<Ty *>(addrs[I]);
      if (sizeof(Ty) <= 2) {
        for (int J = 0; J < NumBlkDecoded && J < ElemsPerAddrDecoded; J++)
          V[I * NumBlkDecoded + J] = *(Addr + J);
      } else {
        for (int J = 0; J < NumBlkDecoded && J < ElemsPerAddrDecoded; J++)
          V[J * N + I] = *(Addr + J);
      }
    }
  }
  return V;
}
#endif // __SYCL_DEVICE_ONLY__

// flat_write does flat-address scatter
template <typename Ty, int N, int NumBlk = 0, int ElemsPerAddr = 0>
__ESIMD_INTRIN void __esimd_svm_scatter(
    __ESIMD_DNS::vector_type_t<uint64_t, N> addrs,
    __ESIMD_DNS::vector_type_t<Ty,
                               N * __ESIMD_DNS::ElemsPerAddrDecoding(NumBlk)>
        vals,
    __ESIMD_DNS::simd_mask_storage_t<N> pred = 1)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  auto NumBlkDecoded = __ESIMD_DNS::ElemsPerAddrDecoding(NumBlk);
  auto ElemsPerAddrDecoded = __ESIMD_DNS::ElemsPerAddrDecoding(ElemsPerAddr);
  if (sizeof(Ty) == 2)
    ElemsPerAddrDecoded = ElemsPerAddrDecoded / 2;

  for (int I = 0; I < N; I++) {
    if (pred[I]) {
      Ty *Addr = reinterpret_cast<Ty *>(addrs[I]);
      if (sizeof(Ty) <= 2) {
        for (int J = 0; J < NumBlkDecoded && J < ElemsPerAddrDecoded; J++)
          *(Addr + J) = vals[I * NumBlkDecoded + J];
      } else {
        for (int J = 0; J < NumBlkDecoded && J < ElemsPerAddrDecoded; J++)
          *(Addr + J) = vals[J * N + I];
      }
    }
  }
}
#endif // __SYCL_DEVICE_ONLY__

// flat_block_read reads a block of data from one flat address
template <typename Ty, int N>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<Ty, N>
__esimd_svm_block_ld_unaligned(uint64_t addr)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  __ESIMD_DNS::vector_type_t<Ty, N> V;

  for (int I = 0; I < N; I++) {
    Ty *Addr = reinterpret_cast<Ty *>(addr + I * sizeof(Ty));
    V[I] = *Addr;
  }
  return V;
}
#endif // __SYCL_DEVICE_ONLY__

// Read a block of data from the given address. Address must be 16-byte aligned.
template <typename Ty, int N>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<Ty, N>
__esimd_svm_block_ld(uint64_t addr)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  __ESIMD_DNS::vector_type_t<Ty, N> V;

  for (int I = 0; I < N; I++) {
    Ty *Addr = reinterpret_cast<Ty *>(addr + I * sizeof(Ty));
    V[I] = *Addr;
  }
  return V;
}
#endif // __SYCL_DEVICE_ONLY__

// flat_block_write writes a block of data using one flat address
template <typename Ty, int N>
__ESIMD_INTRIN void __esimd_svm_block_st(uint64_t addr,
                                         __ESIMD_DNS::vector_type_t<Ty, N> vals)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  for (int I = 0; I < N; I++) {
    Ty *Addr = reinterpret_cast<Ty *>(addr + I * sizeof(Ty));
    *Addr = vals[I];
  }
}
#endif // __SYCL_DEVICE_ONLY__

// Reads a block of data from given surface at given offset.
template <typename Ty, int N, typename SurfIndAliasTy, int32_t IsModified = 0>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<Ty, N>
__esimd_oword_ld_unaligned(SurfIndAliasTy surf_ind, uint32_t offset)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  __ESIMD_DNS::vector_type_t<Ty, N> retv;
  sycl::detail::ESIMDDeviceInterface *I =
      sycl::detail::getESIMDDeviceInterface();

  if (surf_ind == __ESIMD_NS::detail::SLM_BTI) {
    // O-word/Block load for Shared Local Memory
    // __ESIMD_NS::detail::SLM_BTI is special binding table index for SLM
    char *SlmBase = I->__cm_emu_get_slm_ptr();
    for (int i = 0; i < N; ++i) {
      Ty *SlmAddr = reinterpret_cast<Ty *>(offset + SlmBase);
      retv[i] = *SlmAddr;
      offset += sizeof(Ty);
    }
  } else {
    // O-word/Block load for regular surface indexed by surf_ind
    char *readBase;
    uint32_t width;
    std::mutex *mutexLock;

    I->sycl_get_cm_buffer_params_ptr(surf_ind, &readBase, &width, &mutexLock);

    std::lock_guard<std::mutex> lock(*mutexLock);

    for (int idx = 0; idx < N; idx++) {
      if (offset >= width) {
        retv[idx] = 0;
      } else {
        retv[idx] = *((Ty *)(readBase + offset));
      }
      offset += (uint32_t)sizeof(Ty);
    }
  }
  return retv;
}
#endif // __SYCL_DEVICE_ONLY__

// Writes given block of data to a surface with given index at given offset.
template <typename Ty, int N, typename SurfIndAliasTy>
__ESIMD_INTRIN void __esimd_oword_st(SurfIndAliasTy surf_ind, uint32_t offset,
                                     __ESIMD_DNS::vector_type_t<Ty, N> vals)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  offset <<= 4;

  sycl::detail::ESIMDDeviceInterface *I =
      sycl::detail::getESIMDDeviceInterface();
  if (surf_ind == __ESIMD_NS::detail::SLM_BTI) {
    // O-word/Block store for Shared Local Memory
    // __ESIMD_NS::detail::SLM_BTI is special binding table index for SLM
    char *SlmBase = I->__cm_emu_get_slm_ptr();
    for (int i = 0; i < N; ++i) {
      Ty *SlmAddr = reinterpret_cast<Ty *>(offset + SlmBase);
      *SlmAddr = vals[i];
      offset += sizeof(Ty);
    }
  } else {
    // O-word/Block store for regular surface indexed by surf_ind
    char *writeBase;
    uint32_t width;
    std::mutex *mutexLock;

    I->sycl_get_cm_buffer_params_ptr(surf_ind, &writeBase, &width, &mutexLock);

    std::lock_guard<std::mutex> lock(*mutexLock);

    for (int idx = 0; idx < N; idx++) {
      if (offset < width) {
        *((Ty *)(writeBase + offset)) = vals[idx];
      } else {
        break;
      }
      offset += (uint32_t)sizeof(Ty);
    }

    // TODO : Optimize
    I->cm_fence_ptr();
  }
}
#endif // __SYCL_DEVICE_ONLY__

// flat_read4 does flat-address gather4
template <typename Ty, int N, __ESIMD_NS::rgba_channel_mask Mask>
__ESIMD_DNS::vector_type_t<Ty, N * get_num_channels_enabled(Mask)>
    __ESIMD_INTRIN
    __esimd_svm_gather4_scaled(__ESIMD_DNS::vector_type_t<uint64_t, N> addrs,
                               __ESIMD_DNS::simd_mask_storage_t<N> pred = 1)
#ifdef __SYCL_DEVICE_ONLY__
        ;
#else
{
  __ESIMD_DNS::vector_type_t<Ty, N * get_num_channels_enabled(Mask)> V = 0;
  unsigned int Next = 0;
  uint64_t Offset = 0;

  for (const auto &channel : ChannelMaskArray) {
    if (__ESIMD_NS::is_channel_enabled(Mask, channel)) {
      for (int I = 0; I < N; I++, Next++) {
        if (pred[I]) {
          Ty *Addr = reinterpret_cast<Ty *>(addrs[I] + Offset);
          V[Next] = *Addr;
        }
      }
    }
    Offset += (uint64_t)sizeof(Ty);
  }

  return V;
}
#endif // __SYCL_DEVICE_ONLY__

// flat_write does flat-address scatter
template <typename Ty, int N, __ESIMD_NS::rgba_channel_mask Mask>
__ESIMD_INTRIN void __esimd_svm_scatter4_scaled(
    __ESIMD_DNS::vector_type_t<uint64_t, N> addrs,
    __ESIMD_DNS::vector_type_t<Ty, N * get_num_channels_enabled(Mask)> vals,
    __ESIMD_DNS::simd_mask_storage_t<N> pred = 1)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  __ESIMD_DNS::vector_type_t<Ty, N * get_num_channels_enabled(Mask)> V;
  unsigned int Next = 0;
  uint64_t Offset = 0;

  for (const auto &channel : ChannelMaskArray) {
    if (__ESIMD_NS::is_channel_enabled(Mask, channel)) {
      for (int I = 0; I < N; I++, Next++) {
        if (pred[I]) {
          Ty *Addr = reinterpret_cast<Ty *>(addrs[I] + Offset);
          *Addr = vals[Next];
        }
      }
    }
    Offset += (uint64_t)sizeof(Ty);
  }
}
#endif // __SYCL_DEVICE_ONLY__

// Low-level surface-based gather. Collects elements located at given offsets in
// a surface and returns them as a single \ref simd object. Element can be
// 1, 2 or 4-byte value, but is always returned as a 4-byte value within the
// resulting simd object, with upper 2 or 3 bytes undefined.
// Template (compile-time constant) parameters:
// @tparam Ty - element type; can only be a 4-byte integer or \c float,
// @tparam N  - the number of elements
// @tparam SurfIndAliasTy - "surface index alias" type - internal type in the
//   accessor used to denote the surface
// @tparam TySizeLog2 - Log2 of the number of bytes read per element:
//   0 - 1 byte, 1 - 2 bytes, 2 - 4 bytes
// @tparam Scale - offset scaling factor; must be zero currently
// @tparam L1H - L1 cache hint
// @tparam L3H - L3 cache hint
//
// Formal parameters:
// @param surf_ind - the surface index, taken from the SYCL memory object
// @param global_offset - offset added to each individual element's offset to
//   compute actual memory access offset for that element
// @param elem_offsets - per-element offsets
//
template <typename Ty, int N, typename SurfIndAliasTy, int TySizeLog2,
          int16_t Scale = 0>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<Ty, N>
__esimd_gather_scaled2(SurfIndAliasTy surf_ind, uint32_t global_offset,
                       __ESIMD_DNS::vector_type_t<uint32_t, N> elem_offsets)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  static_assert(N == 1 || N == 8 || N == 16 || N == 32);
  static_assert(TySizeLog2 <= 2 && Scale == 0);
  static_assert(std::is_integral<Ty>::value || TySizeLog2 == 2);
  __ESIMD_UNSUPPORTED_ON_HOST;
}
#endif // __SYCL_DEVICE_ONLY__

// Low-level surface-based scatter. Writes elements of a \ref simd object into a
// surface at given offsets. Element can be a 1, 2 or 4-byte value, but it is
// always represented as a 4-byte value within the input simd object,
// unused (not written) upper bytes are ignored.
// Template (compile-time constant) parameters:
// @tparam Ty - element type; can only be a 4-byte integer or \c float,
// @tparam N  - the number of elements to write
// @tparam SurfIndAliasTy - "surface index alias" type - internal type in the
//   accessor used to denote the surface
// @tparam TySizeLog2 - Log2 of the number of bytes written per element:
//   0 - 1 byte, 1 - 2 bytes, 2 - 4 bytes
// @tparam Scale - offset scale; only 0 is supported for now
// @tparam L1H - L1 cache hint
// @tparam L3H - L3 cache hint
//
// Formal parameters:
// @param pred - per-element predicates; elements with zero corresponding
//   predicates are not written
// @param surf_ind - the surface index, taken from the SYCL memory object
// @param global_offset - offset added to each individual element's offset to
//   compute actual memory access offset for that element
// @param elem_offsets - per-element offsets
// @param vals - values to write
//
template <typename Ty, int N, typename SurfIndAliasTy, int TySizeLog2,
          int16_t Scale = 0>
__ESIMD_INTRIN void
__esimd_scatter_scaled(__ESIMD_DNS::simd_mask_storage_t<N> pred,
                       SurfIndAliasTy surf_ind, uint32_t global_offset,
                       __ESIMD_DNS::vector_type_t<uint32_t, N> elem_offsets,
                       __ESIMD_DNS::vector_type_t<Ty, N> vals)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  static_assert(N == 1 || N == 8 || N == 16 || N == 32);
  static_assert(TySizeLog2 <= 2);
  static_assert(std::is_integral<Ty>::value || TySizeLog2 == 2);

  // determine the original element's type size (as __esimd_scatter_scaled
  // requires vals to be a vector of 4-byte integers)
  constexpr size_t OrigSize = __ESIMD_DNS::ElemsPerAddrDecoding(TySizeLog2);
  using RestoredTy = __ESIMD_DNS::uint_type_t<OrigSize>;

  sycl::detail::ESIMDDeviceInterface *I =
      sycl::detail::getESIMDDeviceInterface();

  __ESIMD_DNS::vector_type_t<RestoredTy, N> TypeAdjustedVals;
  if constexpr (OrigSize == 4) {
    TypeAdjustedVals = __ESIMD_DNS::bitcast<RestoredTy, Ty, N>(vals);
  } else {
    static_assert(OrigSize == 1 || OrigSize == 2);
    TypeAdjustedVals = __ESIMD_DNS::convert_vector<RestoredTy, Ty, N>(vals);
  }

  if (surf_ind == __ESIMD_NS::detail::SLM_BTI) {
    // Scattered-store for Shared Local Memory
    // __ESIMD_NS::detail::SLM_BTI is special binding table index for SLM
    assert(global_offset == 0);
    char *SlmBase = I->__cm_emu_get_slm_ptr();
    for (int i = 0; i < N; ++i) {
      if (pred[i]) {
        RestoredTy *addr =
            reinterpret_cast<RestoredTy *>(elem_offsets[i] + SlmBase);
        *addr = TypeAdjustedVals[i];
      }
    }
  } else {
    // Scattered-store for regular surface indexed by surf_ind
    char *writeBase;
    uint32_t width;
    std::mutex *mutexLock;

    I->sycl_get_cm_buffer_params_ptr(surf_ind, &writeBase, &width, &mutexLock);
    writeBase += global_offset;

    std::lock_guard<std::mutex> lock(*mutexLock);

    for (int idx = 0; idx < N; idx++) {
      if (pred[idx]) {
        RestoredTy *addr =
            reinterpret_cast<RestoredTy *>(elem_offsets[idx] + writeBase);
        *addr = TypeAdjustedVals[idx];
      }
    }

    // TODO : Optimize
    I->cm_fence_ptr();
  }
}
#endif // __SYCL_DEVICE_ONLY__

// flat_atomic: flat-address atomic
template <__ESIMD_NS::atomic_op Op, typename Ty, int N>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<Ty, N>
__esimd_svm_atomic0(__ESIMD_DNS::vector_type_t<uint64_t, N> addrs,
                    __ESIMD_DNS::simd_mask_storage_t<N> pred)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  __ESIMD_DNS::vector_type_t<Ty, N> Oldval = 0;

  for (int AddrIdx = 0; AddrIdx < N; AddrIdx += 1) {
    if (pred[AddrIdx] == 0) {
      // Skip Oldval vector elements correpsonding to
      // predicates whose value is zero
      continue;
    }
    if constexpr (Op == __ESIMD_NS::atomic_op::load) {
      Oldval[AddrIdx] = __ESIMD_DNS::atomic_load<Ty>((Ty *)addrs[AddrIdx]);
    } else if constexpr (Op == __ESIMD_NS::atomic_op::inc) {
      Oldval[AddrIdx] =
          __ESIMD_DNS::atomic_add<Ty>((Ty *)addrs[AddrIdx], static_cast<Ty>(1));
    } else if constexpr (Op == __ESIMD_NS::atomic_op::dec) {
      Oldval[AddrIdx] =
          __ESIMD_DNS::atomic_sub<Ty>((Ty *)addrs[AddrIdx], static_cast<Ty>(1));
    }
  }
  return Oldval;
}
#endif // __SYCL_DEVICE_ONLY__

template <__ESIMD_NS::atomic_op Op, typename Ty, int N>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<Ty, N>
__esimd_svm_atomic1(__ESIMD_DNS::vector_type_t<uint64_t, N> addrs,
                    __ESIMD_DNS::vector_type_t<Ty, N> src0,
                    __ESIMD_DNS::simd_mask_storage_t<N> pred)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  __ESIMD_DNS::vector_type_t<Ty, N> Oldval;

  for (int AddrIdx = 0; AddrIdx < N; AddrIdx++) {
    if (pred[AddrIdx] == 0) {
      // Skip Output vector elements correpsonding to
      // predicates whose value is zero
      continue;
    }

    if constexpr (Op == __ESIMD_NS::atomic_op::store) {
      Oldval[AddrIdx] =
          __ESIMD_DNS::atomic_store<Ty>((Ty *)addrs[AddrIdx], src0[AddrIdx]);
    } else if constexpr ((Op == __ESIMD_NS::atomic_op::add) ||
                         (Op == __ESIMD_NS::atomic_op::fadd)) {
      Oldval[AddrIdx] =
          __ESIMD_DNS::atomic_add<Ty>((Ty *)addrs[AddrIdx], src0[AddrIdx]);
    } else if constexpr ((Op == __ESIMD_NS::atomic_op::sub) ||
                         (Op == __ESIMD_NS::atomic_op::fsub)) {
      Oldval[AddrIdx] =
          __ESIMD_DNS::atomic_sub<Ty>((Ty *)addrs[AddrIdx], src0[AddrIdx]);
    } else if constexpr ((Op == __ESIMD_NS::atomic_op::minsint) ||
                         (Op == __ESIMD_NS::atomic_op::min) ||
                         (Op == __ESIMD_NS::atomic_op::fmin)) {
      Oldval[AddrIdx] =
          __ESIMD_DNS::atomic_min<Ty>((Ty *)addrs[AddrIdx], src0[AddrIdx]);
    } else if constexpr ((Op == __ESIMD_NS::atomic_op::maxsint) ||
                         (Op == __ESIMD_NS::atomic_op::max) ||
                         (Op == __ESIMD_NS::atomic_op::fmax)) {
      Oldval[AddrIdx] =
          __ESIMD_DNS::atomic_max<Ty>((Ty *)addrs[AddrIdx], src0[AddrIdx]);
    } else if constexpr (Op == __ESIMD_NS::atomic_op::bit_and) {
      Oldval[AddrIdx] =
          __ESIMD_DNS::atomic_and<Ty>((Ty *)addrs[AddrIdx], src0[AddrIdx]);
    } else if constexpr (Op == __ESIMD_NS::atomic_op::bit_or) {
      Oldval[AddrIdx] =
          __ESIMD_DNS::atomic_or<Ty>((Ty *)addrs[AddrIdx], src0[AddrIdx]);
    } else if constexpr (Op == __ESIMD_NS::atomic_op::bit_xor) {
      Oldval[AddrIdx] =
          __ESIMD_DNS::atomic_xor<Ty>((Ty *)addrs[AddrIdx], src0[AddrIdx]);
    }
  }

  return Oldval;
}
#endif // __SYCL_DEVICE_ONLY__

template <__ESIMD_NS::atomic_op Op, typename Ty, int N>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<Ty, N>
__esimd_svm_atomic2(__ESIMD_DNS::vector_type_t<uint64_t, N> addrs,
                    __ESIMD_DNS::vector_type_t<Ty, N> src0,
                    __ESIMD_DNS::vector_type_t<Ty, N> src1,
                    __ESIMD_DNS::simd_mask_storage_t<N> pred)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  __ESIMD_DNS::vector_type_t<Ty, N> Oldval;

  for (int AddrIdx = 0; AddrIdx < N; AddrIdx++) {
    if (pred[AddrIdx] == 0) {
      // Skip Output vector elements correpsonding to
      // predicates whose value is zero
      continue;
    }
    static_assert((Op == __ESIMD_NS::atomic_op::cmpxchg) ||
                  (Op == __ESIMD_NS::atomic_op::fcmpxchg));
    Oldval[AddrIdx] = __ESIMD_DNS::atomic_cmpxchg((Ty *)addrs[AddrIdx],
                                                  src0[AddrIdx], src1[AddrIdx]);
  }
  return Oldval;
}
#endif // __SYCL_DEVICE_ONLY__

__ESIMD_INTRIN void __esimd_slm_init(uint32_t size)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  sycl::detail::getESIMDDeviceInterface()->cm_slm_init_ptr(size);
}
#endif // ifndef __SYCL_DEVICE_ONLY__

// esimd_barrier, generic group barrier
__ESIMD_INTRIN void __esimd_barrier()
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  sycl::detail::getESIMDDeviceInterface()->cm_barrier_ptr();
}
#endif // __SYCL_DEVICE_ONLY__

// slm_fence sets the SLM read/write order
__ESIMD_INTRIN void __esimd_fence(uint8_t cntl)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  // CM_EMU's 'cm_fence' is NOP. Disabled.
  // sycl::detail::getESIMDDeviceInterface()->cm_fence_ptr();
  __ESIMD_DNS::atomic_fence();
}
#endif // __SYCL_DEVICE_ONLY__

// Scaled gather from a surface.
template <typename Ty, int N, typename SurfIndAliasTy, int TySizeLog2,
          int16_t Scale = 0>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<Ty, N>
__esimd_gather_scaled(__ESIMD_DNS::simd_mask_storage_t<N> pred,
                      SurfIndAliasTy surf_ind, uint32_t global_offset,
                      __ESIMD_DNS::vector_type_t<uint32_t, N> addrs)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  __ESIMD_DNS::vector_type_t<Ty, N> retv = 0;
  sycl::detail::ESIMDDeviceInterface *I =
      sycl::detail::getESIMDDeviceInterface();
  if (surf_ind == __ESIMD_NS::detail::SLM_BTI) {
    // Scattered-load for Shared Local Memory
    // __ESIMD_NS::detail::SLM_BTI is special binding table index for SLM
    assert(global_offset == 0);
    char *SlmBase = I->__cm_emu_get_slm_ptr();
    for (int i = 0; i < N; ++i) {
      if (pred[i]) {
        Ty *addr = reinterpret_cast<Ty *>(addrs[i] + SlmBase);
        retv[i] = *addr;
      }
    }
  } else {
    // Scattered-load for regular surface indexed by surf_ind
    char *readBase;
    uint32_t width;
    std::mutex *mutexLock;

    I->sycl_get_cm_buffer_params_ptr(surf_ind, &readBase, &width, &mutexLock);
    readBase += global_offset;

    std::lock_guard<std::mutex> lock(*mutexLock);

    for (int idx = 0; idx < N; idx++) {
      if (pred[idx]) {
        Ty *addr = reinterpret_cast<Ty *>(addrs[idx] + readBase);
        retv[idx] = *addr;
      }
    }

    // TODO : Optimize
    I->cm_fence_ptr();
  }

  return retv;
}
#endif // __SYCL_DEVICE_ONLY__

// Predicated (masked) scaled gather from a surface.
//
// Template (compile-time constant) parameters:
// @tparam Ty - element type
// @tparam N  - the number of elements to read
// @tparam SurfIndAliasTy - "surface index alias" type - internal type in the
//   accessor used to denote the surface
// @tparam TySizeLog2 - Log2 of the number of bytes written per element:
//   0 - 1 byte, 1 - 2 bytes, 2 - 4 bytes
// @tparam Scale - offset scale; only 0 is supported for now
//
// Formal parameters:
// @param surf_ind - the surface index, taken from the SYCL memory object
// @param global_offset - offset added to each individual element's offset to
//   compute actual memory access offset for that element
// @param offsets - per-element offsets
// @param pred - per-element predicates; elements with zero corresponding
//   predicates are not written
// @return - elements read ("gathered") from memory

template <typename Ty, int N, typename SurfIndAliasTy, int TySizeLog2,
          int16_t Scale = 0>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<Ty, N>
__esimd_gather_masked_scaled2(SurfIndAliasTy surf_ind, uint32_t global_offset,
                              __ESIMD_DNS::vector_type_t<uint32_t, N> offsets,
                              __ESIMD_DNS::simd_mask_storage_t<N> pred)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  static_assert(Scale == 0);

  // determine the original element's type size (as __esimd_scatter_scaled
  // requires vals to be a vector of 4-byte integers)
  constexpr size_t OrigSize = __ESIMD_DNS::ElemsPerAddrDecoding(TySizeLog2);
  using RestoredTy = __ESIMD_DNS::uint_type_t<OrigSize>;

  __ESIMD_DNS::vector_type_t<RestoredTy, N> retv = 0;
  sycl::detail::ESIMDDeviceInterface *I =
      sycl::detail::getESIMDDeviceInterface();

  if (surf_ind == __ESIMD_NS::detail::SLM_BTI) {
    // __ESIMD_NS::detail::SLM_BTI is special binding table index for SLM
    assert(global_offset == 0);
    char *SlmBase = I->__cm_emu_get_slm_ptr();
    for (int idx = 0; idx < N; ++idx) {
      if (pred[idx]) {
        RestoredTy *addr =
            reinterpret_cast<RestoredTy *>(offsets[idx] + SlmBase);
        retv[idx] = *addr;
      }
    }
  } else {
    char *readBase;
    uint32_t width;
    std::mutex *mutexLock;

    I->sycl_get_cm_buffer_params_ptr(surf_ind, &readBase, &width, &mutexLock);

    readBase += global_offset;
    std::lock_guard<std::mutex> lock(*mutexLock);
    for (int idx = 0; idx < N; idx++) {
      if (pred[idx]) {
        RestoredTy *addr =
            reinterpret_cast<RestoredTy *>(offsets[idx] + readBase);
        retv[idx] = *addr;
      }
    }

    // TODO : Optimize
    I->cm_fence_ptr();
  }

  if constexpr (OrigSize == 4) {
    return __ESIMD_DNS::bitcast<Ty, RestoredTy, N>(retv);
  } else {
    return __ESIMD_DNS::convert_vector<Ty, RestoredTy, N>(retv);
  }
}
#endif // __SYCL_DEVICE_ONLY__

// Reads a block of data from given surface at given offset, offset must be
// 16-byte-aligned.
template <typename Ty, int N, typename SurfIndAliasTy, int32_t IsModified = 0>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<Ty, N>
__esimd_oword_ld(SurfIndAliasTy surf_ind, uint32_t addr)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  addr <<= 4;

  __ESIMD_DNS::vector_type_t<Ty, N> retv;
  sycl::detail::ESIMDDeviceInterface *I =
      sycl::detail::getESIMDDeviceInterface();

  if (surf_ind == __ESIMD_NS::detail::SLM_BTI) {
    // O-word/Block load for Shared Local Memory
    // __ESIMD_NS::detail::SLM_BTI is special binding table index for SLM
    char *SlmBase = I->__cm_emu_get_slm_ptr();
    for (int i = 0; i < N; ++i) {
      Ty *SlmAddr = reinterpret_cast<Ty *>(addr + SlmBase);
      retv[i] = *SlmAddr;
      addr += sizeof(Ty);
    }
  } else {
    // O-word/Block load for regular surface indexed by surf_ind
    char *readBase;
    uint32_t width;
    std::mutex *mutexLock;

    I->sycl_get_cm_buffer_params_ptr(surf_ind, &readBase, &width, &mutexLock);

    std::lock_guard<std::mutex> lock(*mutexLock);

    for (int idx = 0; idx < N; idx++) {
      if (addr >= width) {
        retv[idx] = 0;
      } else {
        retv[idx] = *((Ty *)(readBase + addr));
      }
      addr += (uint32_t)sizeof(Ty);
    }
  }
  return retv;
}
#endif // __SYCL_DEVICE_ONLY__

// gather4 scaled masked from a surface/SLM
template <typename Ty, int N, __ESIMD_NS::rgba_channel_mask Mask,
          typename SurfIndAliasTy, int16_t Scale = 0>
__ESIMD_INTRIN
    __ESIMD_DNS::vector_type_t<Ty, N * get_num_channels_enabled(Mask)>
    __esimd_gather4_masked_scaled2(
        SurfIndAliasTy surf_ind, int global_offset,
        __ESIMD_DNS::vector_type_t<uint32_t, N> offsets,
        __ESIMD_DNS::simd_mask_storage_t<N> pred)
#ifdef __SYCL_DEVICE_ONLY__
        ;
#else
{
  __ESIMD_DNS::vector_type_t<Ty, N * get_num_channels_enabled(Mask)> retv = 0;
  sycl::detail::ESIMDDeviceInterface *I =
      sycl::detail::getESIMDDeviceInterface();
  char *ReadBase;
  unsigned int Next = 0;

  if (surf_ind == __ESIMD_NS::detail::SLM_BTI) {
    ReadBase = I->__cm_emu_get_slm_ptr();
  } else {
    uint32_t width;
    std::mutex *mutexLock;
    I->sycl_get_cm_buffer_params_ptr(surf_ind, &ReadBase, &width, &mutexLock);
    std::lock_guard<std::mutex> lock(*mutexLock);
  }

  ReadBase += global_offset;

  for (const auto &channel : ChannelMaskArray) {
    if (__ESIMD_NS::is_channel_enabled(Mask, channel)) {
      for (int I = 0; I < N; I++, Next++) {
        if (pred[I]) {
          Ty *Addr = reinterpret_cast<Ty *>(ReadBase + offsets[I]);
          retv[Next] = *Addr;
        }
      }
    }
    ReadBase += (uint64_t)sizeof(Ty);
  }

  return retv;
}
#endif // __SYCL_DEVICE_ONLY__

// scatter4 scaled to a surface/SLM
template <typename Ty, int N, typename SurfIndAliasTy,
          __ESIMD_NS::rgba_channel_mask Mask, int16_t Scale = 0>
__ESIMD_INTRIN void __esimd_scatter4_scaled(
    __ESIMD_DNS::simd_mask_storage_t<N> pred, SurfIndAliasTy surf_ind,
    int global_offset, __ESIMD_DNS::vector_type_t<uint32_t, N> offsets,
    __ESIMD_DNS::vector_type_t<Ty, N * get_num_channels_enabled(Mask)> vals)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  sycl::detail::ESIMDDeviceInterface *I =
      sycl::detail::getESIMDDeviceInterface();
  char *WriteBase;
  unsigned int Next = 0;

  if (surf_ind == __ESIMD_NS::detail::SLM_BTI) {
    WriteBase = I->__cm_emu_get_slm_ptr();
  } else {
    uint32_t width;
    std::mutex *mutexLock;
    I->sycl_get_cm_buffer_params_ptr(surf_ind, &WriteBase, &width, &mutexLock);
    std::lock_guard<std::mutex> lock(*mutexLock);
  }

  WriteBase += global_offset;

  for (const auto &channel : ChannelMaskArray) {
    if (__ESIMD_NS::is_channel_enabled(Mask, channel)) {
      for (int I = 0; I < N; I++, Next++) {
        if (pred[I]) {
          Ty *Addr = reinterpret_cast<Ty *>(WriteBase + offsets[I]);
          *Addr = vals[Next];
        }
      }
    }
    WriteBase += (uint64_t)sizeof(Ty);
  }
}
#endif // __SYCL_DEVICE_ONLY__

// Surface-based atomic operations
template <__ESIMD_NS::atomic_op Op, typename Ty, int N, typename SurfIndAliasTy>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<Ty, N>
__esimd_dword_atomic0(__ESIMD_DNS::simd_mask_storage_t<N> pred,
                      SurfIndAliasTy surf_ind,
                      __ESIMD_DNS::vector_type_t<uint32_t, N> addrs)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  __ESIMD_DNS::vector_type_t<Ty, N> retv;

  if (surf_ind == __ESIMD_NS::detail::SLM_BTI) {
    char *WriteBase =
        sycl::detail::getESIMDDeviceInterface()->__cm_emu_get_slm_ptr();

    for (int i = 0; i < N; i++) {
      if (pred[i]) {
        Ty *p = reinterpret_cast<Ty *>(addrs[i] + WriteBase);

        switch (Op) {
        case __ESIMD_NS::atomic_op::inc:
          retv[i] = __ESIMD_DNS::atomic_add<Ty>(p, 1);
          break;
        default:
          __ESIMD_UNSUPPORTED_ON_HOST;
        }
      }
    }
  } else {
    __ESIMD_UNSUPPORTED_ON_HOST;
  }
  return retv;
}
#endif // __SYCL_DEVICE_ONLY__

template <__ESIMD_NS::atomic_op Op, typename Ty, int N, typename SurfIndAliasTy>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<Ty, N>
__esimd_dword_atomic1(__ESIMD_DNS::simd_mask_storage_t<N> pred,
                      SurfIndAliasTy surf_ind,
                      __ESIMD_DNS::vector_type_t<uint32_t, N> addrs,
                      __ESIMD_DNS::vector_type_t<Ty, N> src0)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  __ESIMD_UNSUPPORTED_ON_HOST;
}
#endif // __SYCL_DEVICE_ONLY__

template <__ESIMD_NS::atomic_op Op, typename Ty, int N, typename SurfIndAliasTy>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<Ty, N>
__esimd_dword_atomic2(__ESIMD_DNS::simd_mask_storage_t<N> pred,
                      SurfIndAliasTy surf_ind,
                      __ESIMD_DNS::vector_type_t<uint32_t, N> addrs,
                      __ESIMD_DNS::vector_type_t<Ty, N> src0,
                      __ESIMD_DNS::vector_type_t<Ty, N> src1)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  __ESIMD_UNSUPPORTED_ON_HOST;
}
#endif // __SYCL_DEVICE_ONLY__

// Media block load.
//
// @tparam Ty the element data type.
// @tparam M the hight of the 2D block.
// @tparam N the width of the 2D block.
// @tparam Modifier top/bottom field surface access control.
// @tparam TACC type of the surface handle.
// @tparam Plane planar surface index.
// @tparam BlockWidth the width of the return block.
// @param handle the surface handle.
// @param x X-coordinate of the left upper rectangle corner in BYTES.
// @param y Y-coordinate of the left upper rectangle corner in ROWS.
//
// @return the linearized 2D block data read from surface.
//
template <typename Ty, int M, int N, int Modifier, typename TACC, int Plane,
          int BlockWidth>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<Ty, M * N>
__esimd_media_ld(TACC handle, unsigned x, unsigned y)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  __ESIMD_DNS::vector_type_t<Ty, M * N> vals;
  char *readBase;
  uint32_t bpp;
  uint32_t imgWidth;
  uint32_t imgHeight;
  std::mutex *mutexLock;

  assert((handle != __ESIMD_NS::detail::SLM_BTI) &&
         "__esimd_media_ld cannot access SLM");

  sycl::detail::getESIMDDeviceInterface()->sycl_get_cm_image_params_ptr(
      handle, &readBase, &imgWidth, &imgHeight, &bpp, &mutexLock);

  std::lock_guard<std::mutex> lock(*mutexLock);

  int x_pos_a, y_pos_a, offset, index;

  // TODO : Remove intermediate 'in' matrix
  std::vector<std::vector<Ty>> in(M, std::vector<Ty>(N));
  int R = M;
  int C = N;
  for (int i = 0; i < R; i++) {
    for (int j = 0; j < C; j++) {
      x_pos_a = x + j * sizeof(Ty);
      { y_pos_a = y + i; }
      // We should check the boundary condition based on sizeof(Ty), x_pos_a is
      // 0-based Note: Use a signed variable; otherwise sizeof(Ty) is unsigned
      if ((x_pos_a + sizeof(Ty)) > imgWidth) {
        // If we're trying to read outside the boundary, limit the value of
        // x_pos_a Assumption -- We don't this situation:
        //         x_pos_a  width's boundary
        //           |      |
        //           <---type(Ty)--->
        // At most x_pos_a+sizeof(Ty) is exactly at the boundary.
        x_pos_a = imgWidth;
      }
      if (y_pos_a > imgHeight - 1) {
        y_pos_a = imgHeight - 1;
      }
      if (y_pos_a < 0) {
        y_pos_a = 0;
      }
      {
        if (x_pos_a < 0) {
          // Need to align x position to bbp
          int offset = x % bpp;
          x_pos_a -= offset;
        }
        while (x_pos_a < 0) {
          // If we're trying to read outside the left boundary, increase x_pos_a
          x_pos_a += bpp;
        }
      }

      if (x_pos_a >= imgWidth) {
        {
          x_pos_a = x_pos_a - bpp;
          for (uint byte_count = 0; byte_count < sizeof(Ty); byte_count++) {
            if (x_pos_a >= imgWidth) {
              x_pos_a = x_pos_a - bpp;
            }
            offset = y_pos_a * imgWidth + x_pos_a;

            /*
              If destination size per element is less then or equal pixel size
              of the surface move the pixel value accross the destination
              elements. If destination size per element is greater then pixel
              size of the surface replicate pixel value in the destination
              element.
            */
            if (sizeof(Ty) <= bpp) {
              for (uint bpp_count = 0; j < C && bpp_count < bpp;
                   j++, bpp_count += sizeof(Ty)) {
                in[i][j] = *((Ty *)(readBase + offset + bpp_count));
              }
              j--;
              break;
            } else {
              // ((unsigned char*)in.get_addr(i*C+j))[byte_count] = *((unsigned
              // char*)((char*)buff_iter->p + offset));
              unsigned char *pTempBase =
                  ((unsigned char *)in[i].data()) + j * sizeof(Ty);
              pTempBase[byte_count] = *((unsigned char *)(readBase + offset));
            }

            x_pos_a = x_pos_a + 1;
          }
          x_pos_a = imgWidth;
        }
      } else {
        offset = y_pos_a * imgWidth + x_pos_a;
        { in[i][j] = *((Ty *)(readBase + offset)); }
      }
    }
  }

  for (auto i = 0, k = 0; i < M; i++) {
    for (auto j = 0; j < N; j++) {
      vals[k++] = in[i][j];
    }
  }

  return vals;
}
#endif // __SYCL_DEVICE_ONLY__

// Media block store
//
// @tparam Ty the element data type.
// @tparam M the hight of the 2D block.
// @tparam N the width of the 2D block.
// @tparam Modifier top/bottom field surface access control.
// @tparam TACC type of the surface handle.
// @tparam Plane planar surface index.
// @tparam BlockWidth the width of the return block.
// @param handle the surface handle.
// @param x X-coordinate of the left upper rectangle corner in BYTES.
// @param y Y-coordinate of the left upper rectangle corner in ROWS.
// @param vals the linearized 2D block data to be written to surface.
//
template <typename Ty, int M, int N, int Modifier, typename TACC, int Plane,
          int BlockWidth>
__ESIMD_INTRIN void __esimd_media_st(TACC handle, unsigned x, unsigned y,
                                     __ESIMD_DNS::vector_type_t<Ty, M * N> vals)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  sycl::detail::ESIMDDeviceInterface *I =
      sycl::detail::getESIMDDeviceInterface();

  char *writeBase;
  uint32_t bpp;
  uint32_t imgWidth;
  uint32_t imgHeight;
  std::mutex *mutexLock;

  assert((handle != __ESIMD_NS::detail::SLM_BTI) &&
         "__esimd_media_ld cannot access SLM");

  I->sycl_get_cm_image_params_ptr(handle, &writeBase, &imgWidth, &imgHeight,
                                  &bpp, &mutexLock);

  int x_pos_a, y_pos_a, offset;

  assert((x % 4) == 0);
  assert((N * sizeof(Ty)) % 4 == 0);

  // TODO : Remove intermediate 'out' matrix
  std::vector<std::vector<Ty>> out(M, std::vector<Ty>(N));

  std::lock_guard<std::mutex> lock(*mutexLock);

  for (int i = 0, k = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      out[i][j] = vals[k++];
    }
  }

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      x_pos_a = x + j * sizeof(Ty);
      { y_pos_a = y + i; }
      if ((int)x_pos_a < 0) {
        continue;
      }
      if ((int)y_pos_a < 0) {
        continue;
      }
      if ((int)(x_pos_a + sizeof(Ty)) > imgWidth) {
        continue;
      }

      if ((int)y_pos_a > imgHeight - 1) {
        continue;
      }
      offset = y_pos_a * imgWidth + x_pos_a;
      *((Ty *)(writeBase + offset)) = out[i][j];
    }
  }

  // TODO : Optimize
  I->cm_fence_ptr();
}
#endif // __SYCL_DEVICE_ONLY__

// getter methods returning surface index are not available when stateless
// memory accesses are enforced.
#ifndef __ESIMD_FORCE_STATELESS_MEM

// \brief Converts given value to a surface index.
// The input must always be a result of
//   detail::AccessorPrivateProxy::getNativeImageObj(acc)
// where acc is a buffer or image accessor. If the result is, say, 'obj', then
// 'obj' is really a value of the surface index kept in a differently typed
// accessor field. Front-end compilation time type of 'obj' is either
//   ConcreteASPtrType (detail::DecoratedType<DataT, AS>::type *), for a buffer
// or
//   image{1,2,3}d_t OpenCL type for an image
// But when doing code generation, FE replaces e.g. '__read_only image2d_t' FE
// type with '%opencl.image2d_ro_t addrspace(1) *' LLVM type.
// image2d_t can neither be reinterpret_cast'ed from pointer to intptr_t
// (because it is not a pointer at FE translation time), nor it can be
// bit_cast'ed to intptr_t (because it is not trivially copyable). This
// intrinsic takes advantage of the fact that in LLVM IR 'obj' is always a
// pointer, where we can do ptr to uint32_t conversion.
// This intrinsic can be called only from the device code, as
// accessor => memory handle translation for host is different.
// @param acc the SYCL accessor.
//   getNativeImageObj.
// Returns the binding table index value.
template <typename MemObjTy>
__ESIMD_INTRIN __ESIMD_NS::SurfaceIndex __esimd_get_surface_index(MemObjTy obj)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else  // __SYCL_DEVICE_ONLY__
{
  return sycl::detail::getESIMDDeviceInterface()->sycl_get_cm_surface_index_ptr(
      __ESIMD_DNS::AccessorPrivateProxy::getPtr(obj));
}
#endif // __SYCL_DEVICE_ONLY__

#endif // !__ESIMD_FORCE_STATELESS_MEM

/// @endcond ESIMD_DETAIL
