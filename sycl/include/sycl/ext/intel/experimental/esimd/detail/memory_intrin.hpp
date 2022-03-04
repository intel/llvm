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

#include <CL/sycl/detail/accessor_impl.hpp>
#include <CL/sycl/types.hpp>
#include <sycl/ext/intel/experimental/esimd/common.hpp>
#include <sycl/ext/intel/experimental/esimd/detail/types.hpp>
#include <sycl/ext/intel/experimental/esimd/detail/util.hpp>

#include <cstdint>

#ifndef __SYCL_DEVICE_ONLY__
// ESIMD_CPU Emulation support using esimd_cpu plugin

#include <CL/sycl/backend_types.hpp>
#include <CL/sycl/detail/pi.hpp>
#include <sycl/ext/intel/experimental/esimd/detail/atomic_intrin.hpp>
#include <sycl/ext/intel/experimental/esimd/emu/detail/esimd_emulator_device_interface.hpp>

// Channel Mask Array for scaled-gather/scatter
const std::array<__SEIEE::rgba_channel, 4> ChannelMaskArray{
    __SEIEE::rgba_channel::R, __SEIEE::rgba_channel::G,
    __SEIEE::rgba_channel::B, __SEIEE::rgba_channel::A};

#endif // ifndef __SYCL_DEVICE_ONLY__

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace intel {
namespace experimental {
namespace esimd {
namespace detail {

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

} // namespace detail

} // namespace esimd
} // namespace experimental
} // namespace intel
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

// flat_read does flat-address gather
template <typename Ty, int N, int NumBlk = 0>
__ESIMD_INTRIN
    __SEIEED::vector_type_t<Ty, N * __SEIEED::ElemsPerAddrDecoding(NumBlk)>
    __esimd_svm_gather(__SEIEED::vector_type_t<uint64_t, N> addrs,
                       int ElemsPerAddr = NumBlk,
                       __SEIEED::simd_mask_storage_t<N> pred = 1)
#ifdef __SYCL_DEVICE_ONLY__
        ;
#else
{
  auto NumBlkDecoded = __SEIEED::ElemsPerAddrDecoding(NumBlk);
  __SEIEED::vector_type_t<Ty, N * __SEIEED::ElemsPerAddrDecoding(NumBlk)> V = 0;
  ElemsPerAddr = __SEIEED::ElemsPerAddrDecoding(ElemsPerAddr);
  if (sizeof(Ty) == 2)
    ElemsPerAddr = ElemsPerAddr / 2;

  for (int I = 0; I < N; I++) {
    if (pred[I]) {
      Ty *Addr = reinterpret_cast<Ty *>(addrs[I]);
      if (sizeof(Ty) <= 2) {
        for (int J = 0; J < NumBlkDecoded && J < ElemsPerAddr; J++)
          V[I * NumBlkDecoded + J] = *(Addr + J);
      } else {
        for (int J = 0; J < NumBlkDecoded && J < ElemsPerAddr; J++)
          V[J * N + I] = *(Addr + J);
      }
    }
  }
  return V;
}
#endif // __SYCL_DEVICE_ONLY__

// flat_write does flat-address scatter
template <typename Ty, int N, int NumBlk = 0>
__ESIMD_INTRIN void __esimd_svm_scatter(
    __SEIEED::vector_type_t<uint64_t, N> addrs,
    __SEIEED::vector_type_t<Ty, N * __SEIEED::ElemsPerAddrDecoding(NumBlk)>
        vals,
    int ElemsPerAddr = NumBlk, __SEIEED::simd_mask_storage_t<N> pred = 1)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  auto NumBlkDecoded = __SEIEED::ElemsPerAddrDecoding(NumBlk);
  ElemsPerAddr = __SEIEED::ElemsPerAddrDecoding(ElemsPerAddr);
  if (sizeof(Ty) == 2)
    ElemsPerAddr = ElemsPerAddr / 2;

  for (int I = 0; I < N; I++) {
    if (pred[I]) {
      Ty *Addr = reinterpret_cast<Ty *>(addrs[I]);
      if (sizeof(Ty) <= 2) {
        for (int J = 0; J < NumBlkDecoded && J < ElemsPerAddr; J++)
          *(Addr + J) = vals[I * NumBlkDecoded + J];
      } else {
        for (int J = 0; J < NumBlkDecoded && J < ElemsPerAddr; J++)
          *(Addr + J) = vals[J * N + I];
      }
    }
  }
}
#endif // __SYCL_DEVICE_ONLY__

// flat_block_read reads a block of data from one flat address
template <typename Ty, int N>
__ESIMD_INTRIN __SEIEED::vector_type_t<Ty, N>
__esimd_svm_block_ld_unaligned(uint64_t addr)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  __SEIEED::vector_type_t<Ty, N> V;

  for (int I = 0; I < N; I++) {
    Ty *Addr = reinterpret_cast<Ty *>(addr + I * sizeof(Ty));
    V[I] = *Addr;
  }
  return V;
}
#endif // __SYCL_DEVICE_ONLY__

// Read a block of data from the given address. Address must be 16-byte aligned.
template <typename Ty, int N>
__ESIMD_INTRIN __SEIEED::vector_type_t<Ty, N>
__esimd_svm_block_ld(uint64_t addr)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  __SEIEED::vector_type_t<Ty, N> V;

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
                                         __SEIEED::vector_type_t<Ty, N> vals)
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
__ESIMD_INTRIN __SEIEED::vector_type_t<Ty, N>
__esimd_oword_ld_unaligned(SurfIndAliasTy surf_ind, uint32_t offset)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  __SEIEED::vector_type_t<Ty, N> retv;
  sycl::detail::ESIMDDeviceInterface *I =
      sycl::detail::getESIMDDeviceInterface();

  if (surf_ind == __SEIEE::detail::SLM_BTI) {
    // O-word/Block load for Shared Local Memory
    // __SEIEE::detail::SLM_BTI is special binding table index for SLM
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

    I->sycl_get_cm_buffer_params_index_ptr(surf_ind, &readBase, &width,
                                           &mutexLock);

    std::unique_lock<std::mutex> lock(*mutexLock);

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
                                     __SEIEED::vector_type_t<Ty, N> vals)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  offset <<= 4;

  sycl::detail::ESIMDDeviceInterface *I =
      sycl::detail::getESIMDDeviceInterface();
  if (surf_ind == __SEIEE::detail::SLM_BTI) {
    // O-word/Block store for Shared Local Memory
    // __SEIEE::detail::SLM_BTI is special binding table index for SLM
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

    I->sycl_get_cm_buffer_params_index_ptr(surf_ind, &writeBase, &width,
                                           &mutexLock);

    std::unique_lock<std::mutex> lock(*mutexLock);

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
template <typename Ty, int N, __SEIEE::rgba_channel_mask Mask>
__SEIEED::vector_type_t<Ty, N * get_num_channels_enabled(Mask)> __ESIMD_INTRIN
__esimd_svm_gather4_scaled(__SEIEED::vector_type_t<uint64_t, N> addrs,
                           __SEIEED::simd_mask_storage_t<N> pred = 1)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  __SEIEED::vector_type_t<Ty, N * get_num_channels_enabled(Mask)> V = 0;
  unsigned int Next = 0;
  uint64_t Offset = 0;

  for (const auto &channel : ChannelMaskArray) {
    if (__SEIEE::is_channel_enabled(Mask, channel)) {
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
template <typename Ty, int N, __SEIEE::rgba_channel_mask Mask>
__ESIMD_INTRIN void __esimd_svm_scatter4_scaled(
    __SEIEED::vector_type_t<uint64_t, N> addrs,
    __SEIEED::vector_type_t<Ty, N * get_num_channels_enabled(Mask)> vals,
    __SEIEED::simd_mask_storage_t<N> pred = 1)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  __SEIEED::vector_type_t<Ty, N * get_num_channels_enabled(Mask)> V;
  unsigned int Next = 0;
  uint64_t Offset = 0;

  for (const auto &channel : ChannelMaskArray) {
    if (__SEIEE::is_channel_enabled(Mask, channel)) {
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
__ESIMD_INTRIN __SEIEED::vector_type_t<Ty, N>
__esimd_gather_scaled2(SurfIndAliasTy surf_ind, uint32_t global_offset,
                       __SEIEED::vector_type_t<uint32_t, N> elem_offsets)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  static_assert(N == 1 || N == 8 || N == 16 || N == 32);
  static_assert(TySizeLog2 <= 2 && Scale == 0);
  static_assert(std::is_integral<Ty>::value || TySizeLog2 == 2);
  throw cl::sycl::feature_not_supported();
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
__esimd_scatter_scaled(__SEIEED::simd_mask_storage_t<N> pred,
                       SurfIndAliasTy surf_ind, uint32_t global_offset,
                       __SEIEED::vector_type_t<uint32_t, N> elem_offsets,
                       __SEIEED::vector_type_t<Ty, N> vals)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  static_assert(N == 1 || N == 8 || N == 16 || N == 32);
  static_assert(TySizeLog2 <= 2);
  static_assert(std::is_integral<Ty>::value || TySizeLog2 == 2);

  // determine the original element's type size (as __esimd_scatter_scaled
  // requires vals to be a vector of 4-byte integers)
  constexpr size_t OrigSize = __SEIEED::ElemsPerAddrDecoding(TySizeLog2);
  using OrigTyAsInt = __SEIEED::uint_type_t<OrigSize>;

  sycl::detail::ESIMDDeviceInterface *I =
      sycl::detail::getESIMDDeviceInterface();

  if (surf_ind == __SEIEE::detail::SLM_BTI) {
    // Scattered-store for Shared Local Memory
    // __SEIEE::detail::SLM_BTI is special binding table index for SLM
    assert(global_offset == 0);
    char *SlmBase = I->__cm_emu_get_slm_ptr();
    for (int i = 0; i < N; ++i) {
      if (pred[i]) {
        OrigTyAsInt *addr =
            reinterpret_cast<OrigTyAsInt *>(elem_offsets[i] + SlmBase);
        *addr = static_cast<OrigTyAsInt>(vals[i]);
      }
    }
  } else {
    // Scattered-store for regular surface indexed by surf_ind
    char *writeBase;
    uint32_t width;
    std::mutex *mutexLock;

    I->sycl_get_cm_buffer_params_index_ptr(surf_ind, &writeBase, &width,
                                           &mutexLock);
    writeBase += global_offset;

    std::unique_lock<std::mutex> lock(*mutexLock);

    for (int idx = 0; idx < N; idx++) {
      if (pred[idx]) {
        OrigTyAsInt *addr =
            reinterpret_cast<OrigTyAsInt *>(elem_offsets[idx] + writeBase);
        *addr = static_cast<OrigTyAsInt>(vals[idx]);
      }
    }

    // TODO : Optimize
    I->cm_fence_ptr();
  }
}
#endif // __SYCL_DEVICE_ONLY__

// flat_atomic: flat-address atomic
template <__SEIEE::atomic_op Op, typename Ty, int N>
__ESIMD_INTRIN __SEIEED::vector_type_t<Ty, N>
__esimd_svm_atomic0(__SEIEED::vector_type_t<uint64_t, N> addrs,
                    __SEIEED::simd_mask_storage_t<N> pred)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  throw cl::sycl::feature_not_supported();
}
#endif // __SYCL_DEVICE_ONLY__

template <__SEIEE::atomic_op Op, typename Ty, int N>
__ESIMD_INTRIN __SEIEED::vector_type_t<Ty, N>
__esimd_svm_atomic1(__SEIEED::vector_type_t<uint64_t, N> addrs,
                    __SEIEED::vector_type_t<Ty, N> src0,
                    __SEIEED::simd_mask_storage_t<N> pred)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  __SEIEED::vector_type_t<Ty, N> retv;

  for (int i = 0; i < N; i++) {
    if (pred[i]) {
      Ty *p = reinterpret_cast<Ty *>(addrs[i]);

      switch (Op) {
      case __SEIEE::atomic_op::add:
        retv[i] = atomic_add_fetch<Ty>(p, src0[i]);
        break;
      default:
        throw cl::sycl::feature_not_supported();
      }
    }
  }

  return retv;
}
#endif // __SYCL_DEVICE_ONLY__

template <__SEIEE::atomic_op Op, typename Ty, int N>
__ESIMD_INTRIN __SEIEED::vector_type_t<Ty, N>
__esimd_svm_atomic2(__SEIEED::vector_type_t<uint64_t, N> addrs,
                    __SEIEED::vector_type_t<Ty, N> src0,
                    __SEIEED::vector_type_t<Ty, N> src1,
                    __SEIEED::simd_mask_storage_t<N> pred)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  throw cl::sycl::feature_not_supported();
}
#endif // __SYCL_DEVICE_ONLY__

__ESIMD_INTRIN void __esimd_slm_init(size_t size)
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

// generic work-group split barrier
__ESIMD_INTRIN void __esimd_sbarrier(__SEIEE::split_barrier_action flag)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  sycl::detail::getESIMDDeviceInterface()->cm_sbarrier_ptr((uint32_t)flag);
}
#endif // __SYCL_DEVICE_ONLY__

// slm_fence sets the SLM read/write order
__ESIMD_INTRIN void __esimd_fence(uint8_t cntl)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  sycl::detail::getESIMDDeviceInterface()->cm_fence_ptr();
}
#endif // __SYCL_DEVICE_ONLY__

// Scaled gather from a surface.
template <typename Ty, int N, typename SurfIndAliasTy, int TySizeLog2,
          int16_t Scale = 0>
__ESIMD_INTRIN __SEIEED::vector_type_t<Ty, N>
__esimd_gather_scaled(__SEIEED::simd_mask_storage_t<N> pred,
                      SurfIndAliasTy surf_ind, uint32_t global_offset,
                      __SEIEED::vector_type_t<uint32_t, N> addrs)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  __SEIEED::vector_type_t<Ty, N> retv = 0;
  sycl::detail::ESIMDDeviceInterface *I =
      sycl::detail::getESIMDDeviceInterface();
  if (surf_ind == __SEIEE::detail::SLM_BTI) {
    // Scattered-load for Shared Local Memory
    // __SEIEE::detail::SLM_BTI is special binding table index for SLM
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

    I->sycl_get_cm_buffer_params_index_ptr(surf_ind, &readBase, &width,
                                           &mutexLock);
    readBase += global_offset;

    std::unique_lock<std::mutex> lock(*mutexLock);

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
__ESIMD_INTRIN __SEIEED::vector_type_t<Ty, N>
__esimd_gather_masked_scaled2(SurfIndAliasTy surf_ind, uint32_t global_offset,
                              __SEIEED::vector_type_t<uint32_t, N> offsets,
                              __SEIEED::simd_mask_storage_t<N> pred)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  static_assert(Scale == 0);

  // determine the original element's type size (as __esimd_scatter_scaled
  // requires vals to be a vector of 4-byte integers)
  constexpr size_t OrigSize = __SEIEED::ElemsPerAddrDecoding(TySizeLog2);
  using OrigTyAsInt = __SEIEED::uint_type_t<OrigSize>;

  __SEIEED::vector_type_t<Ty, N> retv = 0;
  sycl::detail::ESIMDDeviceInterface *I =
      sycl::detail::getESIMDDeviceInterface();

  if (surf_ind == __SEIEE::detail::SLM_BTI) {
    // __SEIEE::detail::SLM_BTI is special binding table index for SLM
    assert(global_offset == 0);
    char *SlmBase = I->__cm_emu_get_slm_ptr();
    for (int idx = 0; idx < N; ++idx) {
      if (pred[idx]) {
        OrigTyAsInt *addr =
            reinterpret_cast<OrigTyAsInt *>(offsets[idx] + SlmBase);
        retv[idx] = static_cast<Ty>(*addr);
      }
    }
  } else {
    char *readBase;
    uint32_t width;
    std::mutex *mutexLock;

    I->sycl_get_cm_buffer_params_index_ptr(surf_ind, &readBase, &width,
                                           &mutexLock);

    readBase += global_offset;
    std::unique_lock<std::mutex> lock(*mutexLock);
    for (int idx = 0; idx < N; idx++) {
      if (pred[idx]) {
        OrigTyAsInt *addr =
            reinterpret_cast<OrigTyAsInt *>(offsets[idx] + readBase);
        retv[idx] = static_cast<Ty>(*addr);
      }
    }

    // TODO : Optimize
    I->cm_fence_ptr();
  }
  return retv;
}
#endif // __SYCL_DEVICE_ONLY__

// Reads a block of data from given surface at given offset, offset must be
// 16-byte-aligned.
template <typename Ty, int N, typename SurfIndAliasTy, int32_t IsModified = 0>
__ESIMD_INTRIN __SEIEED::vector_type_t<Ty, N>
__esimd_oword_ld(SurfIndAliasTy surf_ind, uint32_t addr)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  addr <<= 4;

  __SEIEED::vector_type_t<Ty, N> retv;
  sycl::detail::ESIMDDeviceInterface *I =
      sycl::detail::getESIMDDeviceInterface();

  if (surf_ind == __SEIEE::detail::SLM_BTI) {
    // O-word/Block load for Shared Local Memory
    // __SEIEE::detail::SLM_BTI is special binding table index for SLM
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

    I->sycl_get_cm_buffer_params_index_ptr(surf_ind, &readBase, &width,
                                           &mutexLock);

    std::unique_lock<std::mutex> lock(*mutexLock);

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

// gather4 scaled from a surface/SLM
template <typename Ty, int N, typename SurfIndAliasTy,
          __SEIEE::rgba_channel_mask Mask, int16_t Scale = 0>
__ESIMD_INTRIN __SEIEED::vector_type_t<Ty, N * get_num_channels_enabled(Mask)>
__esimd_gather4_scaled(__SEIEED::simd_mask_storage_t<N> pred,
                       SurfIndAliasTy surf_ind, int global_offset,
                       __SEIEED::vector_type_t<uint32_t, N> offsets)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  __SEIEED::vector_type_t<Ty, N * get_num_channels_enabled(Mask)> retv = 0;
  sycl::detail::ESIMDDeviceInterface *I =
      sycl::detail::getESIMDDeviceInterface();
  char *ReadBase;
  unsigned int Next = 0;

  if (surf_ind == __SEIEE::detail::SLM_BTI) {
    ReadBase = I->__cm_emu_get_slm_ptr();
  } else {
    uint32_t width;
    std::mutex *mutexLock;
    I->sycl_get_cm_buffer_params_index_ptr(surf_ind, &ReadBase, &width,
                                           &mutexLock);
    std::unique_lock<std::mutex> lock(*mutexLock);
  }

  ReadBase += global_offset;

  for (const auto &channel : ChannelMaskArray) {
    if (__SEIEE::is_channel_enabled(Mask, channel)) {
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
          __SEIEE::rgba_channel_mask Mask, int16_t Scale = 0>
__ESIMD_INTRIN void __esimd_scatter4_scaled(
    __SEIEED::simd_mask_storage_t<N> pred, SurfIndAliasTy surf_ind,
    int global_offset, __SEIEED::vector_type_t<uint32_t, N> offsets,
    __SEIEED::vector_type_t<Ty, N * get_num_channels_enabled(Mask)> vals)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  sycl::detail::ESIMDDeviceInterface *I =
      sycl::detail::getESIMDDeviceInterface();
  char *WriteBase;
  unsigned int Next = 0;

  if (surf_ind == __SEIEE::detail::SLM_BTI) {
    WriteBase = I->__cm_emu_get_slm_ptr();
  } else {
    uint32_t width;
    std::mutex *mutexLock;
    I->sycl_get_cm_buffer_params_index_ptr(surf_ind, &WriteBase, &width,
                                           &mutexLock);
    std::unique_lock<std::mutex> lock(*mutexLock);
  }

  WriteBase += global_offset;

  for (const auto &channel : ChannelMaskArray) {
    if (__SEIEE::is_channel_enabled(Mask, channel)) {
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
template <__SEIEE::atomic_op Op, typename Ty, int N, typename SurfIndAliasTy>
__ESIMD_INTRIN __SEIEED::vector_type_t<Ty, N>
__esimd_dword_atomic0(__SEIEED::simd_mask_storage_t<N> pred,
                      SurfIndAliasTy surf_ind,
                      __SEIEED::vector_type_t<uint32_t, N> addrs)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  __SEIEED::vector_type_t<Ty, N> retv;

  if (surf_ind == __SEIEE::detail::SLM_BTI) {
    char *WriteBase =
        sycl::detail::getESIMDDeviceInterface()->__cm_emu_get_slm_ptr();

    for (int i = 0; i < N; i++) {
      if (pred[i]) {
        Ty *p = reinterpret_cast<Ty *>(addrs[i] + WriteBase);

        switch (Op) {
        case __SEIEE::atomic_op::inc:
          retv[i] = atomic_add_fetch<Ty>(p, 1);
          break;
        default:
          throw cl::sycl::feature_not_supported();
        }
      }
    }
  } else {
    throw cl::sycl::feature_not_supported();
  }
  return retv;
}
#endif // __SYCL_DEVICE_ONLY__

template <__SEIEE::atomic_op Op, typename Ty, int N, typename SurfIndAliasTy>
__ESIMD_INTRIN __SEIEED::vector_type_t<Ty, N>
__esimd_dword_atomic1(__SEIEED::simd_mask_storage_t<N> pred,
                      SurfIndAliasTy surf_ind,
                      __SEIEED::vector_type_t<uint32_t, N> addrs,
                      __SEIEED::vector_type_t<Ty, N> src0)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  throw cl::sycl::feature_not_supported();
}
#endif // __SYCL_DEVICE_ONLY__

template <__SEIEE::atomic_op Op, typename Ty, int N, typename SurfIndAliasTy>
__ESIMD_INTRIN __SEIEED::vector_type_t<Ty, N> __esimd_dword_atomic2(
    __SEIEED::simd_mask_storage_t<N> pred, SurfIndAliasTy surf_ind,
    __SEIEED::vector_type_t<uint32_t, N> addrs,
    __SEIEED::vector_type_t<Ty, N> src0, __SEIEED::vector_type_t<Ty, N> src1)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  throw cl::sycl::feature_not_supported();
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
__ESIMD_INTRIN __SEIEED::vector_type_t<Ty, M * N>
__esimd_media_ld(TACC handle, unsigned x, unsigned y)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  __SEIEED::vector_type_t<Ty, M * N> vals;
  char *readBase;
  uint32_t bpp;
  uint32_t imgWidth;
  uint32_t imgHeight;
  std::mutex *mutexLock;

  assert((handle != __SEIEE::detail::SLM_BTI) &&
         "__esimd_media_ld cannot access SLM");

  sycl::detail::getESIMDDeviceInterface()->sycl_get_cm_image_params_index_ptr(
      handle, &readBase, &imgWidth, &imgHeight, &bpp, &mutexLock);

  std::unique_lock<std::mutex> lock(*mutexLock);

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
                                     __SEIEED::vector_type_t<Ty, M * N> vals)
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

  assert((handle != __SEIEE::detail::SLM_BTI) &&
         "__esimd_media_ld cannot access SLM");

  I->sycl_get_cm_image_params_index_ptr(handle, &writeBase, &imgWidth,
                                        &imgHeight, &bpp, &mutexLock);

  int x_pos_a, y_pos_a, offset;

  assert((x % 4) == 0);
  assert((N * sizeof(Ty)) % 4 == 0);

  // TODO : Remove intermediate 'out' matrix
  std::vector<std::vector<Ty>> out(M, std::vector<Ty>(N));

  std::unique_lock<std::mutex> lock(*mutexLock);

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
__ESIMD_INTRIN __SEIEE::SurfaceIndex __esimd_get_surface_index(MemObjTy obj)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else  // __SYCL_DEVICE_ONLY__
{
  return sycl::detail::getESIMDDeviceInterface()->sycl_get_cm_surface_index_ptr(
      __SEIEED::AccessorPrivateProxy::getPtr(obj));
}
#endif // __SYCL_DEVICE_ONLY__

// \brief Raw sends load.
//
// @param modifier	the send message flags (Bit-0: isSendc, Bit-1: isEOT).
//
// @param execSize the execution size, which must be a compile time constant.
//
// @param pred the predicate to specify enabled channels.
//
// @param numSrc0 the number of GRFs for source-0, which must be a compile time
// constant.
//
// @param numSrc1 the number of GRFs for source-1, which must be a compile time
// constant.
//
// @param numDst the number of GRFs for destination, which must be a compile
// time constant.
//
// @param sfid the shared function ID, which must be a compile time constant.
//
// @param exDesc the extended message descriptor.
//
// @param msgDesc the message descriptor.
//
// @param msgSrc0 the first source operand of send message.
//
// @param msgSrc1 the second source operand of send message.
//
// @param msgDst the destination operand of send message.
//
// Returns a simd vector of type Ty1 and size N1.
//
template <typename Ty1, int N1, typename Ty2, int N2, typename Ty3, int N3,
          int N = 16>
__ESIMD_INTRIN __SEIEED::vector_type_t<Ty1, N1> __esimd_raw_sends2(
    uint8_t modifier, uint8_t execSize, __SEIEED::simd_mask_storage_t<N> pred,
    uint8_t numSrc0, uint8_t numSrc1, uint8_t numDst, uint8_t sfid,
    uint32_t exDesc, uint32_t msgDesc, __SEIEED::vector_type_t<Ty2, N2> msgSrc0,
    __SEIEED::vector_type_t<Ty3, N3> msgSrc1,
    __SEIEED::vector_type_t<Ty1, N1> msgDst)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  throw cl::sycl::feature_not_supported();
}
#endif // __SYCL_DEVICE_ONLY__

// \brief Raw send load.
//
// @param modifier	the send message flags (Bit-0: isSendc, Bit-1: isEOT).
//
// @param execSize the execution size, which must be a compile time constant.
//
// @param pred the predicate to specify enabled channels.
//
// @param numSrc0 the number of GRFs for source-0, which must be a compile time
// constant.
//
// @param numDst the number of GRFs for destination, which must be a compile
// time constant.
//
// @param sfid the shared function ID, which must be a compile time constant.
//
// @param exDesc the extended message descriptor.
//
// @param msgDesc the message descriptor.
//
// @param msgSrc0 the first source operand of send message.
//
// @param msgDst the destination operand of send message.
//
// Returns a simd vector of type Ty1 and size N1.
//
template <typename Ty1, int N1, typename Ty2, int N2, int N = 16>
__ESIMD_INTRIN __SEIEED::vector_type_t<Ty1, N1>
__esimd_raw_send2(uint8_t modifier, uint8_t execSize,
                  __SEIEED::simd_mask_storage_t<N> pred, uint8_t numSrc0,
                  uint8_t numDst, uint8_t sfid, uint32_t exDesc,
                  uint32_t msgDesc, __SEIEED::vector_type_t<Ty2, N2> msgSrc0,
                  __SEIEED::vector_type_t<Ty1, N1> msgDst)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  throw cl::sycl::feature_not_supported();
}
#endif // __SYCL_DEVICE_ONLY__

// \brief Raw sends store.
//
// @param modifier	the send message flags (Bit-0: isSendc, Bit-1: isEOT).
//
// @param execSize the execution size, which must be a compile time constant.
//
// @param pred the predicate to specify enabled channels.
//
// @param numSrc0 the number of GRFs for source-0, which must be a compile time
// constant.
//
// @param numSrc1 the number of GRFs for source-1, which must be a compile time
// constant.
//
// @param sfid the shared function ID, which must be a compile time constant.
//
// @param exDesc the extended message descriptor.
//
// @param msgDesc the message descriptor.
//
// @param msgSrc0 the first source operand of send message.
//
// @param msgSrc1 the second source operand of send message.
//
template <typename Ty1, int N1, typename Ty2, int N2, int N = 16>
__ESIMD_INTRIN void __esimd_raw_sends2_noresult(
    uint8_t modifier, uint8_t execSize, __SEIEED::simd_mask_storage_t<N> pred,
    uint8_t numSrc0, uint8_t numSrc1, uint8_t sfid, uint32_t exDesc,
    uint32_t msgDesc, __SEIEED::vector_type_t<Ty1, N1> msgSrc0,
    __SEIEED::vector_type_t<Ty2, N2> msgSrc1)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  throw cl::sycl::feature_not_supported();
}
#endif // __SYCL_DEVICE_ONLY__

// \brief Raw send store.
//
// @param modifier	the send message flags (Bit-0: isSendc, Bit-1: isEOT).
//
// @param execSize the execution size, which must be a compile time constant.
//
// @param pred the predicate to specify enabled channels.
//
// @param numSrc0 the number of GRFs for source-0, which must be a compile time
// constant.
//
// @param sfid the shared function ID, which must be a compile time constant.
//
// @param exDesc the extended message descriptor.
//
// @param msgDesc the message descriptor.
//
// @param msgSrc0 the first source operand of send message.
//
template <typename Ty1, int N1, int N = 16>
__ESIMD_INTRIN void __esimd_raw_send2_noresult(
    uint8_t modifier, uint8_t execSize, __SEIEED::simd_mask_storage_t<N> pred,
    uint8_t numSrc0, uint8_t sfid, uint32_t exDesc, uint32_t msgDesc,
    __SEIEED::vector_type_t<Ty1, N1> msgSrc0)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  throw cl::sycl::feature_not_supported();
}
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
                                     uint8_t thread_count)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else  // __SYCL_DEVICE_ONLY__
{
  throw cl::sycl::feature_not_supported();
}
#endif // __SYCL_DEVICE_ONLY__

/// Initialize number of named barriers for a kernel
/// Available only on PVC
///
/// @param count  - number of named barriers
__ESIMD_INTRIN void __esimd_nbarrier_init(uint8_t count)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else  // __SYCL_DEVICE_ONLY__
{
  throw cl::sycl::feature_not_supported();
}
#endif // __SYCL_DEVICE_ONLY__

/// Raw send signal to perform signal operation on named barriers
/// Available only on PVC
/// @tparam Ty  - message element type
///
/// @tparam N  - message length
///
/// @param is_sendc  - is sendc
///
/// @param extended_descriptor  - extended message descriptor
///
/// @param descriptor  - message descriptor
///
/// @param msg_var  - source operand of send message
///
/// @param pred  - predicate for enabled channels
template <typename Ty, int N>
__ESIMD_INTRIN void __esimd_raw_send_nbarrier_signal(
    uint32_t is_sendc, uint32_t extended_descriptor, uint32_t descriptor,
    __SEIEED::vector_type_t<Ty, N> msg_var, uint16_t pred = 1)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else  // __SYCL_DEVICE_ONLY__
{
  throw cl::sycl::feature_not_supported();
}
#endif // __SYCL_DEVICE_ONLY__

/// SLM gather.
/// Supported platforms: DG2, PVC
///
/// Collects elements located at slm and returns them
/// as a single \ref simd object.
///
/// @tparam Ty is element type.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint
/// @tparam AddressScale is the address scale.
/// @tparam ImmOffset is the immediate offset added to each address.
/// @tparam DS is the data size.
/// @tparam VS is the number of elements to load per address.
/// @tparam Transposed indicates if the data is transposed during the transfer.
/// @tparam N is the number of channels (platform dependent).
/// @param pred is predicates.
/// @param offsets is the zero-based offsets for SLM buffer in bytes.
/// @return is a vector of type T and size N * to_int<VS>()
template <typename Ty, __SEIEE::cache_hint L1H, __SEIEE::cache_hint L3H,
          uint16_t AddressScale, int ImmOffset, __SEIEE::lsc_data_size DS,
          __SEIEED::lsc_vector_size VS, __SEIEED::lsc_data_order _Transposed,
          int N>
__ESIMD_INTRIN __SEIEED::vector_type_t<Ty, N * __SEIEED::to_int<VS>()>
__esimd_lsc_load_slm(__SEIEED::simd_mask_storage_t<N> pred,
                     __SEIEED::vector_type_t<uint32_t, N> offsets)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else  // __SYCL_DEVICE_ONLY__
{
  throw cl::sycl::feature_not_supported();
  return 0;
}
#endif // __SYCL_DEVICE_ONLY__

/// Surface-based gather.
/// Supported platforms: DG2, PVC
///
/// Collects elements located at surface and returns them
/// as a single \ref simd object.
///
/// @tparam Ty is element type.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam AddressScale is the address scale.
/// @tparam ImmOffset is the immediate offset added to each address.
/// @tparam DS is the data size.
/// @tparam VS is the number of elements to load per address.
/// @tparam Transposed indicates if the data is transposed during the transfer.
/// @tparam N is the number of channels (platform dependent).
/// @tparam SurfIndAliasTy is the \ref sycl::accessor type.
/// @param pred is predicates.
/// @param offsets is the zero-based offsets in bytes.
/// @param surf_ind is the surface index.
/// @return is a vector of type T and N * to_int<VS>()
template <typename Ty, __SEIEE::cache_hint L1H, __SEIEE::cache_hint L3H,
          uint16_t AddressScale, int ImmOffset, __SEIEE::lsc_data_size DS,
          __SEIEED::lsc_vector_size VS, __SEIEED::lsc_data_order _Transposed,
          int N, typename SurfIndAliasTy>
__ESIMD_INTRIN __SEIEED::vector_type_t<Ty, N * __SEIEED::to_int<VS>()>
__esimd_lsc_load_bti(__SEIEED::simd_mask_storage_t<N> pred,
                     __SEIEED::vector_type_t<uint32_t, N> offsets,
                     SurfIndAliasTy surf_ind)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else  // __SYCL_DEVICE_ONLY__
{
  throw cl::sycl::feature_not_supported();
  return 0;
}
#endif // __SYCL_DEVICE_ONLY__

/// USM pointer gather.
/// Supported platforms: DG2, PVC
///
/// Collects elements located at specified address and returns them
/// as a single \ref simd object.
///
/// @tparam Ty is element type.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam AddressScale is the address scale.
/// @tparam ImmOffset is the immediate offset added to each address.
/// @tparam DS is the data size.
/// @tparam VS is the number of elements to load per address.
/// @tparam Transposed indicates if the data is transposed during the transfer.
/// @tparam N is the number of channels (platform dependent).
/// @param pred is predicates.
/// @param addrs is the load addresses.
/// @return is a vector of type T and N * to_int<VS>()
template <typename Ty, __SEIEE::cache_hint L1H, __SEIEE::cache_hint L3H,
          uint16_t AddressScale, int ImmOffset, __SEIEE::lsc_data_size DS,
          __SEIEED::lsc_vector_size VS, __SEIEED::lsc_data_order _Transposed,
          int N>
__ESIMD_INTRIN __SEIEED::vector_type_t<Ty, N * __SEIEED::to_int<VS>()>
__esimd_lsc_load_stateless(__SEIEED::simd_mask_storage_t<N> pred,
                           __SEIEED::vector_type_t<uintptr_t, N> addrs)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else  // __SYCL_DEVICE_ONLY__
{
  throw cl::sycl::feature_not_supported();
  return 0;
}
#endif // __SYCL_DEVICE_ONLY__

/// Surface-based prefetch gather.
/// Supported platforms: DG2, PVC
///
/// Prefetches elements located at surface.
///
/// @tparam Ty is element type.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam AddressScale is the address scale.
/// @tparam ImmOffset is the immediate offset added to each address.
/// @tparam DS is the data size.
/// @tparam VS is the number of elements to load per address.
/// @tparam Transposed indicates if the data is transposed during the transfer.
/// @tparam N is the number of channels (platform dependent).
/// @tparam SurfIndAliasTy is the \ref sycl::accessor type.
/// @param pred is predicates.
/// @param offsets is the zero-based offsets in bytes.
/// @param surf_ind is the surface index.
template <typename Ty, __SEIEE::cache_hint L1H, __SEIEE::cache_hint L3H,
          uint16_t AddressScale, int ImmOffset, __SEIEE::lsc_data_size DS,
          __SEIEED::lsc_vector_size VS, __SEIEED::lsc_data_order _Transposed,
          int N, typename SurfIndAliasTy>
__ESIMD_INTRIN void
__esimd_lsc_prefetch_bti(__SEIEED::simd_mask_storage_t<N> pred,
                         __SEIEED::vector_type_t<uint32_t, N> offsets,
                         SurfIndAliasTy surf_ind)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else  // __SYCL_DEVICE_ONLY__
{
  throw cl::sycl::feature_not_supported();
}
#endif // __SYCL_DEVICE_ONLY__

/// USM pointer prefetch gather.
/// Supported platforms: DG2, PVC
///
/// Prefetches elements located at specified address.
///
/// @tparam Ty is element type.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam AddressScale is the address scale.
/// @tparam ImmOffset is the immediate offset added to each address.
/// @tparam DS is the data size.
/// @tparam VS is the number of elements to load per address.
/// @tparam Transposed indicates if the data is transposed during the transfer.
/// @tparam N is the number of channels (platform dependent).
/// @param pred is predicates.
/// @param addrs is the prefetch addresses.
template <typename Ty, __SEIEE::cache_hint L1H, __SEIEE::cache_hint L3H,
          uint16_t AddressScale, int ImmOffset, __SEIEE::lsc_data_size DS,
          __SEIEED::lsc_vector_size VS, __SEIEED::lsc_data_order _Transposed,
          int N>
__ESIMD_INTRIN void
__esimd_lsc_prefetch_stateless(__SEIEED::simd_mask_storage_t<N> pred,
                               __SEIEED::vector_type_t<uintptr_t, N> addrs)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else  // __SYCL_DEVICE_ONLY__
{
  throw cl::sycl::feature_not_supported();
}
#endif // __SYCL_DEVICE_ONLY__

/// SLM scatter.
/// Supported platforms: DG2, PVC
///
/// Scatters elements located to slm.
///
/// @tparam Ty is element type.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam AddressScale is the address scale.
/// @tparam ImmOffset is the immediate offset added to each address.
/// @tparam DS is the data size.
/// @tparam VS is the number of elements to load per address.
/// @tparam Transposed indicates if the data is transposed during the transfer.
/// @tparam N is the number of channels (platform dependent).
/// @param pred is predicates.
/// @param offsets is the zero-based offsets for SLM buffer in bytes.
/// @param vals is values to store.
template <typename Ty, __SEIEE::cache_hint L1H, __SEIEE::cache_hint L3H,
          uint16_t AddressScale, int ImmOffset, __SEIEE::lsc_data_size DS,
          __SEIEED::lsc_vector_size VS, __SEIEED::lsc_data_order _Transposed,
          int N>
__ESIMD_INTRIN void __esimd_lsc_store_slm(
    __SEIEED::simd_mask_storage_t<N> pred,
    __SEIEED::vector_type_t<uint32_t, N> offsets,
    __SEIEED::vector_type_t<Ty, N * __SEIEED::to_int<VS>()> vals)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else  // __SYCL_DEVICE_ONLY__
{
  throw cl::sycl::feature_not_supported();
}
#endif // __SYCL_DEVICE_ONLY__

/// Surface-based scatter.
/// Supported platforms: DG2, PVC
///
/// Scatters elements to surface.
///
/// @tparam Ty is element type.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam AddressScale is the address scale.
/// @tparam ImmOffset is the immediate offset added to each address.
/// @tparam DS is the data size.
/// @tparam VS is the number of elements to load per address.
/// @tparam Transposed indicates if the data is transposed during the transfer.
/// @tparam N is the number of channels (platform dependent).
/// @tparam SurfIndAliasTy is the \ref sycl::accessor type.
/// @param pred is predicates.
/// @param offsets is the zero-based offsets in bytes.
/// @param vals is values to store.
/// @param surf_ind is the surface index.
template <typename Ty, __SEIEE::cache_hint L1H, __SEIEE::cache_hint L3H,
          uint16_t AddressScale, int ImmOffset, __SEIEE::lsc_data_size DS,
          __SEIEED::lsc_vector_size VS, __SEIEED::lsc_data_order _Transposed,
          int N, typename SurfIndAliasTy>
__ESIMD_INTRIN void __esimd_lsc_store_bti(
    __SEIEED::simd_mask_storage_t<N> pred,
    __SEIEED::vector_type_t<uint32_t, N> offsets,
    __SEIEED::vector_type_t<Ty, N * __SEIEED::to_int<VS>()> vals,
    SurfIndAliasTy surf_ind)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else  // __SYCL_DEVICE_ONLY__
{
  throw cl::sycl::feature_not_supported();
}
#endif // __SYCL_DEVICE_ONLY__

/// USM pointer scatter.
/// Supported platforms: DG2, PVC
///
/// Scatters elements to specific address.
///
/// @tparam Ty is element type.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam AddressScale is the address scale.
/// @tparam ImmOffset is the immediate offset added to each address.
/// @tparam DS is the data size.
/// @tparam VS is the number of elements to load per address.
/// @tparam Transposed indicates if the data is transposed during the transfer.
/// @tparam N is the number of channels (platform dependent).
/// @param pred is predicates.
/// @param addrs is the prefetch addresses.
/// @param vals is values to store.
template <typename Ty, __SEIEE::cache_hint L1H, __SEIEE::cache_hint L3H,
          uint16_t AddressScale, int ImmOffset, __SEIEE::lsc_data_size DS,
          __SEIEED::lsc_vector_size VS, __SEIEED::lsc_data_order _Transposed,
          int N>
__ESIMD_INTRIN void __esimd_lsc_store_stateless(
    __SEIEED::simd_mask_storage_t<N> pred,
    __SEIEED::vector_type_t<uintptr_t, N> addrs,
    __SEIEED::vector_type_t<Ty, N * __SEIEED::to_int<VS>()> vals)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else  // __SYCL_DEVICE_ONLY__
{
  throw cl::sycl::feature_not_supported();
}
#endif // __SYCL_DEVICE_ONLY__

/// 2D USM pointer block load.
/// Supported platforms: PVC
///
/// Collects elements located at specified address and returns them
/// as a single \ref simd object.
///
/// @tparam Ty is element type.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam DS is the data size.
/// @tparam Transposed is the transposed version or not.
/// @tparam NBlocks is the number of blocks.
/// @tparam BlockWidth is the block width in number of elements.
/// @tparam BlockHeight is the block height in number of elements.
/// @tparam Transformed is apply VNNI transform or not.
/// @tparam N is the data size
/// @param Pred is predicates.
/// @param Ptr is the surface base address for this operation.
/// @param SurfaceWidth is the surface width minus 1 in bytes
/// @param SurfaceHeight is the surface height minus 1 in rows
/// @param SurfacePitch is the surface pitch minus 1 in bytes
/// @param X is zero based X-coordinate of the left upper rectangle corner in
/// number of elements.
/// @param Y is zero based Y-coordinate of the left upper rectangle corner in
/// rows.
/// @return is a vector of type T and size N, where N is
///  BlockWidth * BlockHeight * NBlocks, if transformed;
///  otherwise,
///  N = roundUpNextMultiple(BlockHeight, 4 / sizeof(T)) *
///   getNextPowerOf2(BlockWidth) * NBlocks
template <typename Ty, __SEIEE::cache_hint L1H, __SEIEE::cache_hint L3H,
          __SEIEE::lsc_data_size DS, __SEIEED::lsc_data_order _Transposed,
          uint8_t NBlocks, int BlockWidth, int BlockHeight, bool Transformed,
          int N>
__ESIMD_INTRIN __SEIEED::vector_type_t<Ty, N>
__esimd_lsc_load2d_stateless(__SEIEED::simd_mask_storage_t<N> Pred,
                             uintptr_t Ptr, int SurfaceWidth, int SurfaceHeight,
                             int SurfacePitch, int X, int Y)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else  // __SYCL_DEVICE_ONLY__
{
  throw cl::sycl::feature_not_supported();
  return 0;
}
#endif // __SYCL_DEVICE_ONLY__

/// 2D USM pointer block prefetch.
/// Supported platforms: PVC
///
/// Prefetches elements located at specified address.
///
/// @tparam Ty is element type.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam DS is the data size.
/// @tparam NBlocks is the number of blocks.
/// @tparam Transposed is the transposed version or not.
/// @tparam BlockWidth is the block width in number of elements.
/// @tparam BlockHeight is the block height in number of elements.
/// @tparam Transformed is apply VNNI transform or not.
/// @tparam N is the data size
/// @param Pred is predicates.
/// @param Ptr is the surface base address for this operation.
/// @param SurfaceWidth is the surface width minus 1 in bytes
/// @param SurfaceHeight is the surface height minus 1 in rows
/// @param SurfacePitch is the surface pitch minus 1 in bytes
/// @param X is zero based X-coordinate of the left upper rectangle corner in
/// number of elements.
/// @param Y is zero based Y-coordinate of the left upper rectangle corner in
/// rows.
template <typename Ty, __SEIEE::cache_hint L1H, __SEIEE::cache_hint L3H,
          __SEIEE::lsc_data_size DS, __SEIEED::lsc_data_order _Transposed,
          uint8_t NBlocks, int BlockWidth, int BlockHeight, bool Transformed,
          int N>
__ESIMD_INTRIN void __esimd_lsc_prefetch2d_stateless(
    __SEIEED::simd_mask_storage_t<N> Pred, uintptr_t Ptr, int SurfaceWidth,
    int SurfaceHeight, int SurfacePitch, int X, int Y)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else  // __SYCL_DEVICE_ONLY__
{
  throw cl::sycl::feature_not_supported();
}
#endif // __SYCL_DEVICE_ONLY__

/// 2D USM pointer block store.
/// Supported platforms: PVC
///
/// Stores elements at specified address.
///
/// @tparam Ty is element type.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam DS is the data size.
/// @tparam Transposed is the transposed version or not.
/// @tparam NBlocks is the number of blocks.
/// @tparam BlockWidth is the block width in number of elements.
/// @tparam BlockHeight is the block height in number of elements.
/// @tparam Transformed is apply VNNI transform or not.
/// @tparam N is the data size
/// @param Pred is predicates.
/// @param Ptr is the surface base address for this operation.
/// @param SurfaceWidth is the surface width minus 1 in bytes
/// @param SurfaceHeight is the surface height minus 1 in rows
/// @param SurfacePitch is the surface pitch minus 1 in bytes
/// @param X is zero based X-coordinate of the left upper rectangle corner in
/// number of elements.
/// @param Y is zero based Y-coordinate of the left upper rectangle corner in
/// rows.
/// @param Vals is a vector to store of type T and size N, where N is
///  BlockWidth * BlockHeight * NBlocks, if transformed;
///  otherwise,
///  N = roundUpNextMultiple(BlockHeight, 4 / sizeof(T)) *
///   getNextPowerOf2(BlockWidth) * NBlocks
template <typename Ty, __SEIEE::cache_hint L1H, __SEIEE::cache_hint L3H,
          __SEIEE::lsc_data_size DS, __SEIEED::lsc_data_order _Transposed,
          uint8_t NBlocks, int BlockWidth, int BlockHeight, bool Transformed,
          int N>
__ESIMD_INTRIN void
__esimd_lsc_store2d_stateless(__SEIEED::simd_mask_storage_t<N> Pred,
                              uintptr_t Ptr, int SurfaceWidth,
                              int SurfaceHeight, int SurfacePitch, int X, int Y,
                              __SEIEED::vector_type_t<Ty, N> vals)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else  // __SYCL_DEVICE_ONLY__
{
  throw cl::sycl::feature_not_supported();
}
#endif // __SYCL_DEVICE_ONLY__

/// SLM atomic.
/// Supported platforms: DG2, PVC
///
/// @tparam Ty is element type.
/// @tparam Op is operation type.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam AddressScale is the address scale.
/// @tparam ImmOffset is the immediate offset added to each address.
/// @tparam DS is the data size.
/// @tparam VS is the number of elements per address.
/// @tparam Transposed indicates if the data is transposed during the transfer.
/// @tparam N is the number of channels (platform dependent).
/// @param pred is predicates.
/// @param offsets is the zero-based offsets.
template <typename Ty, __SEIEED::lsc_atomic_op Op, __SEIEE::cache_hint L1H,
          __SEIEE::cache_hint L3H, uint16_t AddressScale, int ImmOffset,
          __SEIEE::lsc_data_size DS, __SEIEED::lsc_vector_size VS,
          __SEIEED::lsc_data_order _Transposed, int N>
__ESIMD_INTRIN __SEIEED::vector_type_t<Ty, N * __SEIEED::to_int<VS>()>
__esimd_lsc_xatomic_slm_0(__SEIEED::simd_mask_storage_t<N> pred,
                          __SEIEED::vector_type_t<uint32_t, N> offsets)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else  // __SYCL_DEVICE_ONLY__
{
  throw cl::sycl::feature_not_supported();
  return 0;
}
#endif // __SYCL_DEVICE_ONLY__

/// SLM atomic.
/// Supported platforms: DG2, PVC
///
/// @tparam Ty is element type.
/// @tparam Op is operation type.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam AddressScale is the address scale.
/// @tparam ImmOffset is the immediate offset added to each address.
/// @tparam DS is the data size.
/// @tparam VS is the number of elements per address.
/// @tparam Transposed indicates if the data is transposed during the transfer.
/// @tparam N is the number of channels (platform dependent).
/// @param pred is predicates.
/// @param offsets is the zero-based offsets.
/// @param src0 is the first atomic operand.
template <typename Ty, __SEIEED::lsc_atomic_op Op, __SEIEE::cache_hint L1H,
          __SEIEE::cache_hint L3H, uint16_t AddressScale, int ImmOffset,
          __SEIEE::lsc_data_size DS, __SEIEED::lsc_vector_size VS,
          __SEIEED::lsc_data_order _Transposed, int N>
__ESIMD_INTRIN __SEIEED::vector_type_t<Ty, N * __SEIEED::to_int<VS>()>
__esimd_lsc_xatomic_slm_1(
    __SEIEED::simd_mask_storage_t<N> pred,
    __SEIEED::vector_type_t<uint32_t, N> offsets,
    __SEIEED::vector_type_t<Ty, N * __SEIEED::to_int<VS>()> src0)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else  // __SYCL_DEVICE_ONLY__
{
  throw cl::sycl::feature_not_supported();
  return 0;
}
#endif // __SYCL_DEVICE_ONLY__

/// SLM atomic.
/// Supported platforms: DG2, PVC
///
/// @tparam Ty is element type.
/// @tparam Op is operation type.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam AddressScale is the address scale.
/// @tparam ImmOffset is the immediate offset added to each address.
/// @tparam DS is the data size.
/// @tparam VS is the number of elements per address.
/// @tparam Transposed indicates if the data is transposed during the transfer.
/// @tparam N is the number of channels (platform dependent).
/// @param pred is predicates.
/// @param offsets is the zero-based offsets.
/// @param src0 is the first atomic operand.
/// @param src1 is the second atomic operand.
template <typename Ty, __SEIEED::lsc_atomic_op Op, __SEIEE::cache_hint L1H,
          __SEIEE::cache_hint L3H, uint16_t AddressScale, int ImmOffset,
          __SEIEE::lsc_data_size DS, __SEIEED::lsc_vector_size VS,
          __SEIEED::lsc_data_order _Transposed, int N>
__ESIMD_INTRIN __SEIEED::vector_type_t<Ty, N * __SEIEED::to_int<VS>()>
__esimd_lsc_xatomic_slm_2(
    __SEIEED::simd_mask_storage_t<N> pred,
    __SEIEED::vector_type_t<uint32_t, N> offsets,
    __SEIEED::vector_type_t<Ty, N * __SEIEED::to_int<VS>()> src0,
    __SEIEED::vector_type_t<Ty, N * __SEIEED::to_int<VS>()> src1)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else  // __SYCL_DEVICE_ONLY__
{
  throw cl::sycl::feature_not_supported();
  return 0;
}
#endif // __SYCL_DEVICE_ONLY__

/// Accessor-based atomic.
/// Supported platforms: DG2, PVC
///
/// @tparam Ty is element type.
/// @tparam Op is operation type.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam AddressScale is the address scale.
/// @tparam ImmOffset is the immediate offset added to each address.
/// @tparam DS is the data size.
/// @tparam VS is the number of elements per address.
/// @tparam Transposed indicates if the data is transposed during the transfer.
/// @tparam N is the number of channels (platform dependent).
/// @tparam SurfIndAliasTy is the \ref sycl::accessor type.
/// @param pred is predicates.
/// @param offsets is the zero-based offsets.
/// @param surf_ind is the surface index.
template <typename Ty, __SEIEED::lsc_atomic_op Op, __SEIEE::cache_hint L1H,
          __SEIEE::cache_hint L3H, uint16_t AddressScale, int ImmOffset,
          __SEIEE::lsc_data_size DS, __SEIEED::lsc_vector_size VS,
          __SEIEED::lsc_data_order _Transposed, int N, typename SurfIndAliasTy>
__ESIMD_INTRIN __SEIEED::vector_type_t<Ty, N * __SEIEED::to_int<VS>()>
__esimd_lsc_xatomic_bti_0(__SEIEED::simd_mask_storage_t<N> pred,
                          __SEIEED::vector_type_t<uint32_t, N> offsets,
                          SurfIndAliasTy surf_ind)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else  // __SYCL_DEVICE_ONLY__
{
  throw cl::sycl::feature_not_supported();
  return 0;
}
#endif // __SYCL_DEVICE_ONLY__

/// Accessor-based atomic.
/// Supported platforms: DG2, PVC
///
/// @tparam Ty is element type.
/// @tparam Op is operation type.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam AddressScale is the address scale.
/// @tparam ImmOffset is the immediate offset added to each address.
/// @tparam DS is the data size.
/// @tparam VS is the number of elements per address.
/// @tparam Transposed indicates if the data is transposed during the transfer.
/// @tparam N is the number of channels (platform dependent).
/// @tparam SurfIndAliasTy is the \ref sycl::accessor type.
/// @param pred is predicates.
/// @param offsets is the zero-based offsets.
/// @param src0 is the first atomic operand.
/// @param surf_ind is the surface index.
template <typename Ty, __SEIEED::lsc_atomic_op Op, __SEIEE::cache_hint L1H,
          __SEIEE::cache_hint L3H, uint16_t AddressScale, int ImmOffset,
          __SEIEE::lsc_data_size DS, __SEIEED::lsc_vector_size VS,
          __SEIEED::lsc_data_order _Transposed, int N, typename SurfIndAliasTy>
__ESIMD_INTRIN __SEIEED::vector_type_t<Ty, N * __SEIEED::to_int<VS>()>
__esimd_lsc_xatomic_bti_1(
    __SEIEED::simd_mask_storage_t<N> pred,
    __SEIEED::vector_type_t<uint32_t, N> offsets,
    __SEIEED::vector_type_t<Ty, N * __SEIEED::to_int<VS>()> src0,
    SurfIndAliasTy surf_ind)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else  // __SYCL_DEVICE_ONLY__
{
  throw cl::sycl::feature_not_supported();
  return 0;
}
#endif // __SYCL_DEVICE_ONLY__

/// Accessor-based atomic.
/// Supported platforms: DG2, PVC
///
/// @tparam Ty is element type.
/// @tparam Op is operation type.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam AddressScale is the address scale.
/// @tparam ImmOffset is the immediate offset added to each address.
/// @tparam DS is the data size.
/// @tparam VS is the number of elements per address.
/// @tparam Transposed indicates if the data is transposed during the transfer.
/// @tparam N is the number of channels (platform dependent).
/// @tparam SurfIndAliasTy is the \ref sycl::accessor type.
/// @param pred is predicates.
/// @param offsets is the zero-based offsets.
/// @param src0 is the first atomic operand.
/// @param src1 is the second atomic operand.
/// @param surf_ind is the surface index.
template <typename Ty, __SEIEED::lsc_atomic_op Op, __SEIEE::cache_hint L1H,
          __SEIEE::cache_hint L3H, uint16_t AddressScale, int ImmOffset,
          __SEIEE::lsc_data_size DS, __SEIEED::lsc_vector_size VS,
          __SEIEED::lsc_data_order _Transposed, int N, typename SurfIndAliasTy>
__ESIMD_INTRIN __SEIEED::vector_type_t<Ty, N * __SEIEED::to_int<VS>()>
__esimd_lsc_xatomic_bti_2(
    __SEIEED::simd_mask_storage_t<N> pred,
    __SEIEED::vector_type_t<uint32_t, N> offsets,
    __SEIEED::vector_type_t<Ty, N * __SEIEED::to_int<VS>()> src0,
    __SEIEED::vector_type_t<Ty, N * __SEIEED::to_int<VS>()> src1,
    SurfIndAliasTy surf_ind)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else  // __SYCL_DEVICE_ONLY__
{
  throw cl::sycl::feature_not_supported();
  return 0;
}
#endif // __SYCL_DEVICE_ONLY__

/// USM pointer atomic.
/// Supported platforms: DG2, PVC
///
/// @tparam Ty is element type.
/// @tparam Op is operation type.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam AddressScale is the address scale.
/// @tparam ImmOffset is the immediate offset added to each address.
/// @tparam DS is the data size.
/// @tparam VS is the number of elements per address.
/// @tparam Transposed indicates if the data is transposed during the transfer.
/// @tparam N is the number of channels (platform dependent).
/// @param pred is predicates.
/// @param addrs is the prefetch addresses.
template <typename Ty, __SEIEED::lsc_atomic_op Op, __SEIEE::cache_hint L1H,
          __SEIEE::cache_hint L3H, uint16_t AddressScale, int ImmOffset,
          __SEIEE::lsc_data_size DS, __SEIEED::lsc_vector_size VS,
          __SEIEED::lsc_data_order _Transposed, int N>
__ESIMD_INTRIN __SEIEED::vector_type_t<Ty, N * __SEIEED::to_int<VS>()>
__esimd_lsc_xatomic_stateless_0(__SEIEED::simd_mask_storage_t<N> pred,
                                __SEIEED::vector_type_t<uintptr_t, N> addrs)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else  // __SYCL_DEVICE_ONLY__
{
  throw cl::sycl::feature_not_supported();
  return 0;
}
#endif // __SYCL_DEVICE_ONLY__

/// USM pointer atomic.
/// Supported platforms: DG2, PVC
///
/// @tparam Ty is element type.
/// @tparam Op is operation type.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam AddressScale is the address scale.
/// @tparam ImmOffset is the immediate offset added to each address.
/// @tparam DS is the data size.
/// @tparam VS is the number of elements per address.
/// @tparam Transposed indicates if the data is transposed during the transfer.
/// @tparam N is the number of channels (platform dependent).
/// @param pred is predicates.
/// @param addrs is the prefetch addresses.
/// @param src0 is the first atomic operand.
template <typename Ty, __SEIEED::lsc_atomic_op Op, __SEIEE::cache_hint L1H,
          __SEIEE::cache_hint L3H, uint16_t AddressScale, int ImmOffset,
          __SEIEE::lsc_data_size DS, __SEIEED::lsc_vector_size VS,
          __SEIEED::lsc_data_order _Transposed, int N>
__ESIMD_INTRIN __SEIEED::vector_type_t<Ty, N * __SEIEED::to_int<VS>()>
__esimd_lsc_xatomic_stateless_1(
    __SEIEED::simd_mask_storage_t<N> pred,
    __SEIEED::vector_type_t<uintptr_t, N> addrs,
    __SEIEED::vector_type_t<Ty, N * __SEIEED::to_int<VS>()> src0)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else  // __SYCL_DEVICE_ONLY__
{
  throw cl::sycl::feature_not_supported();
  return 0;
}
#endif // __SYCL_DEVICE_ONLY__

/// USM pointer atomic.
/// Supported platforms: DG2, PVC
///
/// @tparam Ty is element type.
/// @tparam Op is operation type.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam AddressScale is the address scale.
/// @tparam ImmOffset is the immediate offset added to each address.
/// @tparam DS is the data size.
/// @tparam VS is the number of elements per address.
/// @tparam Transposed indicates if the data is transposed during the transfer.
/// @tparam N is the number of channels (platform dependent).
/// @param pred is predicates.
/// @param addrs is the prefetch addresses.
/// @param src0 is the first atomic operand.
/// @param src1 is the second atomic operand.
template <typename Ty, __SEIEED::lsc_atomic_op Op, __SEIEE::cache_hint L1H,
          __SEIEE::cache_hint L3H, uint16_t AddressScale, int ImmOffset,
          __SEIEE::lsc_data_size DS, __SEIEED::lsc_vector_size VS,
          __SEIEED::lsc_data_order _Transposed, int N>
__ESIMD_INTRIN __SEIEED::vector_type_t<Ty, N * __SEIEED::to_int<VS>()>
__esimd_lsc_xatomic_stateless_2(
    __SEIEED::simd_mask_storage_t<N> pred,
    __SEIEED::vector_type_t<uintptr_t, N> addrs,
    __SEIEED::vector_type_t<Ty, N * __SEIEED::to_int<VS>()> src0,
    __SEIEED::vector_type_t<Ty, N * __SEIEED::to_int<VS>()> src1)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else  // __SYCL_DEVICE_ONLY__
{
  throw cl::sycl::feature_not_supported();
  return 0;
}
#endif // __SYCL_DEVICE_ONLY__

/// Memory fence.
/// Supported platforms: DG2, PVC
///
/// @tparam Kind is the Sfid shaded function.
/// @tparam FenceOp is the fence operation.
/// @tparam Scope is the operation scope.
/// @tparam N is the number of channels (platform dependent).
/// @param pred is predicates.
template <__SEIEE::lsc_memory_kind Kind, __SEIEE::lsc_fence_op FenceOp,
          __SEIEE::lsc_scope Scope, int N>
__ESIMD_INTRIN void __esimd_lsc_fence(__SEIEED::simd_mask_storage_t<N> pred)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else  // __SYCL_DEVICE_ONLY__
{
  throw cl::sycl::feature_not_supported();
}
#endif // __SYCL_DEVICE_ONLY__

/// @endcond ESIMD_DETAIL
