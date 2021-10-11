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

#pragma once

#include <CL/sycl/detail/accessor_impl.hpp>
#include <CL/sycl/types.hpp>
#include <sycl/ext/intel/experimental/esimd/common.hpp>
#include <sycl/ext/intel/experimental/esimd/detail/types.hpp>
#include <sycl/ext/intel/experimental/esimd/detail/util.hpp>

#include <cstdint>

#ifndef __SYCL_DEVICE_ONLY__
/// ESIMD_CPU Emulation support using esimd_cpu plugin

/// Definition macro to be referenced in CM header files for
/// preventing build failure caused by symbol conflicts between llvm
/// and CM - e.g. vector.
#define __SYCL_EXPLICIT_SIMD_PLUGIN__

// Header files required for accessing CM-managed resources - image,
// buffer, runtime API etc.
namespace cm_support {
#include <CL/cm_rt.h>
} // namespace cm_support

#include <CL/sycl/backend_types.hpp>
#include <CL/sycl/detail/pi.hpp>
#include <sycl/ext/intel/experimental/esimd/detail/atomic_intrin.hpp>
#include <sycl/ext/intel/experimental/esimd/emu/detail/esimdcpu_device_interface.hpp>

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
#ifdef __SYCL_DEVICE_ONLY__
  template <typename AccessorTy>
  static auto getNativeImageObj(const AccessorTy &Acc) {
    return Acc.getNativeImageObj();
  }
#else  // __SYCL_DEVICE_ONLY__
  template <typename AccessorTy>
  static auto getImageRange(const AccessorTy &Acc) {
    return Acc.getAccessRange();
  }
  static auto getElemSize(const sycl::detail::AccessorBaseHost &Acc) {
    return Acc.getElemSize();
  }
  static void *getPtr(const sycl::detail::AccessorBaseHost &Acc) {
    return Acc.getPtr();
  }
#endif // __SYCL_DEVICE_ONLY__
};

template <int ElemsPerAddr,
          typename = sycl::detail::enable_if_t<
              (ElemsPerAddr == 1 || ElemsPerAddr == 2 || ElemsPerAddr == 4)>>
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
template <typename Ty, int N, int NumBlk = 0,
          __SEIEE::CacheHint L1H = __SEIEE::CacheHint::None,
          __SEIEE::CacheHint L3H = __SEIEE::CacheHint::None>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION
    __SEIEED::vector_type_t<Ty, N * __SEIEED::ElemsPerAddrDecoding(NumBlk)>
    __esimd_flat_read(__SEIEED::vector_type_t<uint64_t, N> addrs,
                      int ElemsPerAddr = NumBlk,
                      __SEIEED::simd_mask_storage_t<N> pred = 1);

// flat_write does flat-address scatter
template <typename Ty, int N, int NumBlk = 0,
          __SEIEE::CacheHint L1H = __SEIEE::CacheHint::None,
          __SEIEE::CacheHint L3H = __SEIEE::CacheHint::None>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION void __esimd_flat_write(
    __SEIEED::vector_type_t<uint64_t, N> addrs,
    __SEIEED::vector_type_t<Ty, N * __SEIEED::ElemsPerAddrDecoding(NumBlk)>
        vals,
    int ElemsPerAddr = NumBlk, __SEIEED::simd_mask_storage_t<N> pred = 1);

// flat_block_read reads a block of data from one flat address
template <typename Ty, int N, __SEIEE::CacheHint L1H = __SEIEE::CacheHint::None,
          __SEIEE::CacheHint L3H = __SEIEE::CacheHint::None>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<Ty, N>
__esimd_flat_block_read_unaligned(uint64_t addr);

// flat_block_write writes a block of data using one flat address
template <typename Ty, int N, __SEIEE::CacheHint L1H = __SEIEE::CacheHint::None,
          __SEIEE::CacheHint L3H = __SEIEE::CacheHint::None>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION void
__esimd_flat_block_write(uint64_t addr, __SEIEED::vector_type_t<Ty, N> vals);

// Reads a block of data from given surface at given offset.
template <typename Ty, int N, typename SurfIndAliasTy>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<Ty, N>
__esimd_block_read(SurfIndAliasTy surf_ind, uint32_t offset);

// Writes given block of data to a surface with given index at given offset.
template <typename Ty, int N, typename SurfIndAliasTy>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION void
__esimd_block_write(SurfIndAliasTy surf_ind, uint32_t offset,
                    __SEIEED::vector_type_t<Ty, N> vals);

// flat_read4 does flat-address gather4
template <typename Ty, int N, __SEIEE::rgba_channel_mask Mask,
          __SEIEE::CacheHint L1H = __SEIEE::CacheHint::None,
          __SEIEE::CacheHint L3H = __SEIEE::CacheHint::None>
__SEIEED::vector_type_t<Ty, N * get_num_channels_enabled(Mask)>
    SYCL_EXTERNAL SYCL_ESIMD_FUNCTION
    __esimd_flat_read4(__SEIEED::vector_type_t<uint64_t, N> addrs,
                       __SEIEED::simd_mask_storage_t<N> pred = 1);

// flat_write does flat-address scatter
template <typename Ty, int N, __SEIEE::rgba_channel_mask Mask,
          __SEIEE::CacheHint L1H = __SEIEE::CacheHint::None,
          __SEIEE::CacheHint L3H = __SEIEE::CacheHint::None>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION void __esimd_flat_write4(
    __SEIEED::vector_type_t<uint64_t, N> addrs,
    __SEIEED::vector_type_t<Ty, N * get_num_channels_enabled(Mask)> vals,
    __SEIEED::simd_mask_storage_t<N> pred = 1);

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
// @tparam L1H - L1 cache hint
// @tparam L3H - L3 cache hint
//
// Formal parameters:
// @param scale - the scale; must be 0
// @param surf_ind - the surface index, taken from the SYCL memory object
// @param global_offset - offset added to each individual element's offset to
//   compute actual memory access offset for that element
// @param elem_offsets - per-element offsets
//
template <typename Ty, int N, typename SurfIndAliasTy, int TySizeLog2,
          __SEIEE::CacheHint L1H = __SEIEE::CacheHint::None,
          __SEIEE::CacheHint L3H = __SEIEE::CacheHint::None>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<Ty, N>
__esimd_surf_read(int16_t scale, SurfIndAliasTy surf_ind,
                  uint32_t global_offset,
                  __SEIEED::vector_type_t<uint32_t, N> elem_offsets);

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
// @tparam L1H - L1 cache hint
// @tparam L3H - L3 cache hint
//
// Formal parameters:
// @param pred - per-element predicates; elements with zero corresponding
//   predicates are not written
// @param scale - the scale; must be 0
// @param surf_ind - the surface index, taken from the SYCL memory object
// @param global_offset - offset added to each individual element's offset to
//   compute actual memory access offset for that element
// @param elem_offsets - per-element offsets
// @param vals - values to write
//
template <typename Ty, int N, typename SurfIndAliasTy, int TySizeLog2,
          __SEIEE::CacheHint L1H = __SEIEE::CacheHint::None,
          __SEIEE::CacheHint L3H = __SEIEE::CacheHint::None>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION void
__esimd_surf_write(__SEIEED::simd_mask_storage_t<N> pred, int16_t scale,
                   SurfIndAliasTy surf_ind, uint32_t global_offset,
                   __SEIEED::vector_type_t<uint32_t, N> elem_offsets,
                   __SEIEED::vector_type_t<Ty, N> vals);

// TODO bring the parameter order of __esimd* intrinsics in accordance with the
// correponsing BE intrinsicics parameter order.

// flat_atomic: flat-address atomic
template <__SEIEE::atomic_op Op, typename Ty, int N,
          __SEIEE::CacheHint L1H = __SEIEE::CacheHint::None,
          __SEIEE::CacheHint L3H = __SEIEE::CacheHint::None>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<Ty, N>
__esimd_flat_atomic0(__SEIEED::vector_type_t<uint64_t, N> addrs,
                     __SEIEED::simd_mask_storage_t<N> pred);

template <__SEIEE::atomic_op Op, typename Ty, int N,
          __SEIEE::CacheHint L1H = __SEIEE::CacheHint::None,
          __SEIEE::CacheHint L3H = __SEIEE::CacheHint::None>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<Ty, N>
__esimd_flat_atomic1(__SEIEED::vector_type_t<uint64_t, N> addrs,
                     __SEIEED::vector_type_t<Ty, N> src0,
                     __SEIEED::simd_mask_storage_t<N> pred);

template <__SEIEE::atomic_op Op, typename Ty, int N,
          __SEIEE::CacheHint L1H = __SEIEE::CacheHint::None,
          __SEIEE::CacheHint L3H = __SEIEE::CacheHint::None>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<Ty, N>
__esimd_flat_atomic2(__SEIEED::vector_type_t<uint64_t, N> addrs,
                     __SEIEED::vector_type_t<Ty, N> src0,
                     __SEIEED::vector_type_t<Ty, N> src1,
                     __SEIEED::simd_mask_storage_t<N> pred);

// esimd_barrier, generic group barrier
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION void __esimd_barrier();

// generic work-group split barrier
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION void
__esimd_sbarrier(__SEIEE::split_barrier_action flag);

// slm_fence sets the SLM read/write order
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION void __esimd_slm_fence(uint8_t cntl);

// slm_read does SLM gather
template <typename Ty, int N>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<Ty, N>
__esimd_slm_read(__SEIEED::vector_type_t<uint32_t, N> addrs,
                 __SEIEED::simd_mask_storage_t<N> pred = 1);

// slm_write does SLM scatter
template <typename Ty, int N>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION void
__esimd_slm_write(__SEIEED::vector_type_t<uint32_t, N> addrs,
                  __SEIEED::vector_type_t<Ty, N> vals,
                  __SEIEED::simd_mask_storage_t<N> pred = 1);

// slm_block_read reads a block of data from SLM
template <typename Ty, int N>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<Ty, N>
__esimd_slm_block_read(uint32_t addr);

// slm_block_write writes a block of data to SLM
template <typename Ty, int N>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION void
__esimd_slm_block_write(uint32_t addr, __SEIEED::vector_type_t<Ty, N> vals);

// slm_read4 does SLM gather4
template <typename Ty, int N, __SEIEE::rgba_channel_mask Mask>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION
    __SEIEED::vector_type_t<Ty, N * get_num_channels_enabled(Mask)>
    __esimd_slm_read4(__SEIEED::vector_type_t<uint32_t, N> addrs,
                      __SEIEED::simd_mask_storage_t<N> pred = 1);

// slm_write4 does SLM scatter4
template <typename Ty, int N, __SEIEE::rgba_channel_mask Mask>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION void __esimd_slm_write4(
    __SEIEED::vector_type_t<uint32_t, N> addrs,
    __SEIEED::vector_type_t<Ty, N * get_num_channels_enabled(Mask)> vals,
    __SEIEED::simd_mask_storage_t<N> pred = 1);

// slm_atomic: SLM atomic
template <__SEIEE::atomic_op Op, typename Ty, int N>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<Ty, N>
__esimd_slm_atomic0(__SEIEED::vector_type_t<uint32_t, N> addrs,
                    __SEIEED::simd_mask_storage_t<N> pred);

template <__SEIEE::atomic_op Op, typename Ty, int N>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<Ty, N>
__esimd_slm_atomic1(__SEIEED::vector_type_t<uint32_t, N> addrs,
                    __SEIEED::vector_type_t<Ty, N> src0,
                    __SEIEED::simd_mask_storage_t<N> pred);

template <__SEIEE::atomic_op Op, typename Ty, int N>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<Ty, N>
__esimd_slm_atomic2(__SEIEED::vector_type_t<uint32_t, N> addrs,
                    __SEIEED::vector_type_t<Ty, N> src0,
                    __SEIEED::vector_type_t<Ty, N> src1,
                    __SEIEED::simd_mask_storage_t<N> pred);

// Media block load
//
// @param Ty the element data type.
//
// @param M the hight of the 2D block.
//
// @param N the width of the 2D block.
//
// @param TACC type of the surface handle.
//
// @param modifier top/bottom field surface access control.
//
// @param handle the surface handle.
//
// @param plane planar surface index.
//
// @param width the width of the return block.
//
// @param x X-coordinate of the left upper rectangle corner in BYTES.
//
// @param y Y-coordinate of the left upper rectangle corner in ROWS.
//
// @return the linearized 2D block data read from surface.
//
template <typename Ty, int M, int N, typename TACC>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<Ty, M * N>
__esimd_media_block_load(unsigned modififer, TACC handle, unsigned plane,
                         unsigned width, unsigned x, unsigned y);

// Media block store
//
// @param Ty the element data type.
//
// @param M the hight of the 2D block.
//
// @param N the width of the 2D block.
//
// @param TACC type of the surface handle.
//
// @param modifier top/bottom field surface access control.
//
// @param handle the surface handle.
//
// @param plane planar surface index.
//
// @param width the width of the return block.
//
// @param x X-coordinate of the left upper rectangle corner in BYTES.
//
// @param y Y-coordinate of the left upper rectangle corner in ROWS.
//
// @param vals the linearized 2D block data to be written to surface.
//
template <typename Ty, int M, int N, typename TACC>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION void
__esimd_media_block_store(unsigned modififer, TACC handle, unsigned plane,
                          unsigned width, unsigned x, unsigned y,
                          __SEIEED::vector_type_t<Ty, M * N> vals);

/// \brief esimd_get_value
///
/// @param sid the SYCL accessor.
///
/// Returns the binding table index value.
///
template <typename SurfIndAliasTy>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION uint32_t
__esimd_get_value(SurfIndAliasTy sid);

/// \brief Raw sends load.
///
/// @param modifier	the send message flags (Bit-0: isSendc, Bit-1: isEOT).
///
/// @param execSize the execution size, which must be a compile time constant.
///
/// @param pred the predicate to specify enabled channels.
///
/// @param numSrc0 the number of GRFs for source-0, which must be a compile time
/// constant.
///
/// @param numSrc1 the number of GRFs for source-1, which must be a compile time
/// constant.
///
/// @param numDst the number of GRFs for destination, which must be a compile
/// time constant.
///
/// @param sfid the shared function ID, which must be a compile time constant.
///
/// @param exDesc the extended message descriptor.
///
/// @param msgDesc the message descriptor.
///
/// @param msgSrc0 the first source operand of send message.
///
/// @param msgSrc1 the second source operand of send message.
///
/// @param msgDst the destination operand of send message.
///
/// Returns a simd vector of type Ty1 and size N1.
///
template <typename Ty1, int N1, typename Ty2, int N2, typename Ty3, int N3,
          int N = 16>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<Ty1, N1>
__esimd_raw_sends_load(uint8_t modifier, uint8_t execSize,
                       __SEIEED::simd_mask_storage_t<N> pred, uint8_t numSrc0,
                       uint8_t numSrc1, uint8_t numDst, uint8_t sfid,
                       uint32_t exDesc, uint32_t msgDesc,
                       __SEIEED::vector_type_t<Ty2, N2> msgSrc0,
                       __SEIEED::vector_type_t<Ty3, N3> msgSrc1,
                       __SEIEED::vector_type_t<Ty1, N1> msgDst);

/// \brief Raw send load.
///
/// @param modifier	the send message flags (Bit-0: isSendc, Bit-1: isEOT).
///
/// @param execSize the execution size, which must be a compile time constant.
///
/// @param pred the predicate to specify enabled channels.
///
/// @param numSrc0 the number of GRFs for source-0, which must be a compile time
/// constant.
///
/// @param numDst the number of GRFs for destination, which must be a compile
/// time constant.
///
/// @param sfid the shared function ID, which must be a compile time constant.
///
/// @param exDesc the extended message descriptor.
///
/// @param msgDesc the message descriptor.
///
/// @param msgSrc0 the first source operand of send message.
///
/// @param msgDst the destination operand of send message.
///
/// Returns a simd vector of type Ty1 and size N1.
///
template <typename Ty1, int N1, typename Ty2, int N2, int N = 16>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<Ty1, N1>
__esimd_raw_send_load(uint8_t modifier, uint8_t execSize,
                      __SEIEED::simd_mask_storage_t<N> pred, uint8_t numSrc0,
                      uint8_t numDst, uint8_t sfid, uint32_t exDesc,
                      uint32_t msgDesc,
                      __SEIEED::vector_type_t<Ty2, N2> msgSrc0,
                      __SEIEED::vector_type_t<Ty1, N1> msgDst);

/// \brief Raw sends store.
///
/// @param modifier	the send message flags (Bit-0: isSendc, Bit-1: isEOT).
///
/// @param execSize the execution size, which must be a compile time constant.
///
/// @param pred the predicate to specify enabled channels.
///
/// @param numSrc0 the number of GRFs for source-0, which must be a compile time
/// constant.
///
/// @param numSrc1 the number of GRFs for source-1, which must be a compile time
/// constant.
///
/// @param sfid the shared function ID, which must be a compile time constant.
///
/// @param exDesc the extended message descriptor.
///
/// @param msgDesc the message descriptor.
///
/// @param msgSrc0 the first source operand of send message.
///
/// @param msgSrc1 the second source operand of send message.
///
template <typename Ty1, int N1, typename Ty2, int N2, int N = 16>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION void __esimd_raw_sends_store(
    uint8_t modifier, uint8_t execSize, __SEIEED::simd_mask_storage_t<N> pred,
    uint8_t numSrc0, uint8_t numSrc1, uint8_t sfid, uint32_t exDesc,
    uint32_t msgDesc, __SEIEED::vector_type_t<Ty1, N1> msgSrc0,
    __SEIEED::vector_type_t<Ty2, N2> msgSrc1);

/// \brief Raw send store.
///
/// @param modifier	the send message flags (Bit-0: isSendc, Bit-1: isEOT).
///
/// @param execSize the execution size, which must be a compile time constant.
///
/// @param pred the predicate to specify enabled channels.
///
/// @param numSrc0 the number of GRFs for source-0, which must be a compile time
/// constant.
///
/// @param sfid the shared function ID, which must be a compile time constant.
///
/// @param exDesc the extended message descriptor.
///
/// @param msgDesc the message descriptor.
///
/// @param msgSrc0 the first source operand of send message.
///
template <typename Ty1, int N1, int N = 16>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION void
__esimd_raw_send_store(uint8_t modifier, uint8_t execSize,
                       __SEIEED::simd_mask_storage_t<N> pred, uint8_t numSrc0,
                       uint8_t sfid, uint32_t exDesc, uint32_t msgDesc,
                       __SEIEED::vector_type_t<Ty1, N1> msgSrc0);

#ifndef __SYCL_DEVICE_ONLY__

/// ESIMD_CPU Emulation support using esimd_cpu plugin

template <typename Ty, int N, int NumBlk, __SEIEE::CacheHint L1H,
          __SEIEE::CacheHint L3H>
inline __SEIEED::vector_type_t<Ty, N * __SEIEED::ElemsPerAddrDecoding(NumBlk)>
__esimd_flat_read(__SEIEED::vector_type_t<uint64_t, N> addrs, int ElemsPerAddr,
                  __SEIEED::simd_mask_storage_t<N> pred) {
  auto NumBlkDecoded = __SEIEED::ElemsPerAddrDecoding(NumBlk);
  __SEIEED::vector_type_t<Ty, N * __SEIEED::ElemsPerAddrDecoding(NumBlk)> V;
  ElemsPerAddr = __SEIEED::ElemsPerAddrDecoding(ElemsPerAddr);

  for (int I = 0; I < N; I++) {
    if (pred[I]) {
      Ty *Addr = reinterpret_cast<Ty *>(addrs[I]);
      if (sizeof(Ty) == 2)
        ElemsPerAddr = ElemsPerAddr / 2;
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

template <typename Ty, int N, __SEIEE::rgba_channel_mask Mask,
          __SEIEE::CacheHint L1H, __SEIEE::CacheHint L3H>
inline __SEIEED::vector_type_t<Ty, N * get_num_channels_enabled(Mask)>
__esimd_flat_read4(__SEIEED::vector_type_t<uint64_t, N> addrs,
                   __SEIEED::simd_mask_storage_t<N> pred) {
  __SEIEED::vector_type_t<Ty, N * get_num_channels_enabled(Mask)> V;
  unsigned int Next = 0;

  if constexpr (__SEIEE::is_channel_enabled(Mask, __SEIEE::rgba_channel::R)) {
    for (int I = 0; I < N; I++, Next++) {
      if (pred[I]) {
        Ty *Addr = reinterpret_cast<Ty *>(addrs[I]);
        V[Next] = *Addr;
      }
    }
  }

  if constexpr (__SEIEE::is_channel_enabled(Mask, __SEIEE::rgba_channel::G)) {
    for (int I = 0; I < N; I++, Next++) {
      if (pred[I]) {
        Ty *Addr = reinterpret_cast<Ty *>(addrs[I] + sizeof(Ty));
        V[Next] = *Addr;
      }
    }
  }

  if constexpr (__SEIEE::is_channel_enabled(Mask, __SEIEE::rgba_channel::B)) {
    for (int I = 0; I < N; I++, Next++) {
      if (pred[I]) {
        Ty *Addr = reinterpret_cast<Ty *>(addrs[I] + sizeof(Ty) + sizeof(Ty));
        V[Next] = *Addr;
      }
    }
  }

  if constexpr (__SEIEE::is_channel_enabled(Mask, __SEIEE::rgba_channel::A)) {
    for (int I = 0; I < N; I++, Next++) {
      if (pred[I]) {
        Ty *Addr = reinterpret_cast<Ty *>(addrs[I] + sizeof(Ty) + sizeof(Ty) +
                                          sizeof(Ty));
        V[Next] = *Addr;
      }
    }
  }

  return V;
}

template <typename Ty, int N, int NumBlk, __SEIEE::CacheHint L1H,
          __SEIEE::CacheHint L3H>
inline void __esimd_flat_write(
    __SEIEED::vector_type_t<uint64_t, N> addrs,
    __SEIEED::vector_type_t<Ty, N * __SEIEED::ElemsPerAddrDecoding(NumBlk)>
        vals,
    int ElemsPerAddr, __SEIEED::simd_mask_storage_t<N> pred) {
  auto NumBlkDecoded = __SEIEED::ElemsPerAddrDecoding(NumBlk);
  ElemsPerAddr = __SEIEED::ElemsPerAddrDecoding(ElemsPerAddr);

  for (int I = 0; I < N; I++) {
    if (pred[I]) {
      Ty *Addr = reinterpret_cast<Ty *>(addrs[I]);
      if (sizeof(Ty) == 2)
        ElemsPerAddr = ElemsPerAddr / 2;
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

template <typename Ty, int N, __SEIEE::rgba_channel_mask Mask,
          __SEIEE::CacheHint L1H, __SEIEE::CacheHint L3H>
inline void __esimd_flat_write4(
    __SEIEED::vector_type_t<uint64_t, N> addrs,
    __SEIEED::vector_type_t<Ty, N * get_num_channels_enabled(Mask)> vals,
    __SEIEED::simd_mask_storage_t<N> pred) {
  __SEIEED::vector_type_t<Ty, N * get_num_channels_enabled(Mask)> V;
  unsigned int Next = 0;

  if constexpr (__SEIEE::is_channel_enabled(Mask, __SEIEE::rgba_channel::R)) {
    for (int I = 0; I < N; I++, Next++) {
      if (pred[I]) {
        Ty *Addr = reinterpret_cast<Ty *>(addrs[I]);
        *Addr = vals[Next];
      }
    }
  }

  if constexpr (__SEIEE::is_channel_enabled(Mask, __SEIEE::rgba_channel::G)) {
    for (int I = 0; I < N; I++, Next++) {
      if (pred[I]) {
        Ty *Addr = reinterpret_cast<Ty *>(addrs[I] + sizeof(Ty));
        *Addr = vals[Next];
      }
    }
  }

  if constexpr (__SEIEE::is_channel_enabled(Mask, __SEIEE::rgba_channel::B)) {
    for (int I = 0; I < N; I++, Next++) {
      if (pred[I]) {
        Ty *Addr = reinterpret_cast<Ty *>(addrs[I] + sizeof(Ty) + sizeof(Ty));
        *Addr = vals[Next];
      }
    }
  }

  if constexpr (__SEIEE::is_channel_enabled(Mask, __SEIEE::rgba_channel::A)) {
    for (int I = 0; I < N; I++, Next++) {
      if (pred[I]) {
        Ty *Addr = reinterpret_cast<Ty *>(addrs[I] + sizeof(Ty) + sizeof(Ty) +
                                          sizeof(Ty));
        *Addr = vals[Next];
      }
    }
  }
}

template <typename Ty, int N, typename SurfIndAliasTy, int TySizeLog2,
          __SEIEE::CacheHint L1H, __SEIEE::CacheHint L3H>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<Ty, N>
__esimd_surf_read(int16_t scale, SurfIndAliasTy surf_ind,
                  uint32_t global_offset,
                  __SEIEED::vector_type_t<uint32_t, N> elem_offsets) {
  static_assert(N == 1 || N == 8 || N == 16);
  static_assert(TySizeLog2 <= 2);
  static_assert(std::is_integral<Ty>::value || TySizeLog2 == 2);
  throw cl::sycl::feature_not_supported();
}

template <typename Ty, int N, typename SurfIndAliasTy, int TySizeLog2,
          __SEIEE::CacheHint L1H, __SEIEE::CacheHint L3H>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION void
__esimd_surf_write(__SEIEED::vector_type_t<uint16_t, N> pred, int16_t scale,
                   SurfIndAliasTy surf_ind, uint32_t global_offset,
                   __SEIEED::vector_type_t<uint32_t, N> elem_offsets,
                   __SEIEED::vector_type_t<Ty, N> vals) {
  static_assert(N == 1 || N == 8 || N == 16);
  static_assert(TySizeLog2 <= 2);
  static_assert(std::is_integral<Ty>::value || TySizeLog2 == 2);
  throw cl::sycl::feature_not_supported();
}

template <typename Ty, int N, __SEIEE::CacheHint L1H, __SEIEE::CacheHint L3H>
inline __SEIEED::vector_type_t<Ty, N>
__esimd_flat_block_read_unaligned(uint64_t addr) {
  __SEIEED::vector_type_t<Ty, N> V;

  for (int I = 0; I < N; I++) {
    Ty *Addr = reinterpret_cast<Ty *>(addr + I * sizeof(Ty));
    V[I] = *Addr;
  }
  return V;
}

template <typename Ty, int N, __SEIEE::CacheHint L1H, __SEIEE::CacheHint L3H>
inline void __esimd_flat_block_write(uint64_t addr,
                                     __SEIEED::vector_type_t<Ty, N> vals) {
  for (int I = 0; I < N; I++) {
    Ty *Addr = reinterpret_cast<Ty *>(addr + I * sizeof(Ty));
    *Addr = vals[I];
  }
}

template <typename Ty, int M, int N, typename TACC>
inline __SEIEED::vector_type_t<Ty, M * N>
__esimd_media_block_load(unsigned modififer, TACC handle, unsigned plane,
                         unsigned width, unsigned x, unsigned y) {
  __SEIEED::vector_type_t<Ty, M * N> vals;

  sycl::detail::ESIMDDeviceInterface *I =
      sycl::detail::getESIMDDeviceInterface();

  char *readBase;
  uint32_t bpp;
  uint32_t imgWidth;
  uint32_t imgHeight;
  std::mutex *mutexLock;

  auto ImageHandle = __SEIEED::AccessorPrivateProxy::getPtr(handle);

  I->sycl_get_cm_image_params_ptr(ImageHandle, &readBase, &imgWidth, &imgHeight,
                                  &bpp, &mutexLock);

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

template <typename Ty, int M, int N, typename TACC>
inline void __esimd_media_block_store(unsigned modififer, TACC handle,
                                      unsigned plane, unsigned width,
                                      unsigned x, unsigned y,
                                      __SEIEED::vector_type_t<Ty, M * N> vals) {
  sycl::detail::ESIMDDeviceInterface *I =
      sycl::detail::getESIMDDeviceInterface();

  char *writeBase;
  uint32_t bpp;
  uint32_t imgWidth;
  uint32_t imgHeight;
  std::mutex *mutexLock;

  auto ImageHandle = __SEIEED::AccessorPrivateProxy::getPtr(handle);

  I->sycl_get_cm_image_params_ptr(ImageHandle, &writeBase, &imgWidth,
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

  /// TODO : Optimize
  I->cm_fence_ptr();
}

template <typename Ty, int N>
inline uint16_t __esimd_any(__SEIEED::vector_type_t<Ty, N> src) {
  for (unsigned int i = 0; i != N; i++) {
    if (src[i] != 0)
      return 1;
  }
  return 0;
}

template <typename Ty, int N>
inline uint16_t __esimd_all(__SEIEED::vector_type_t<Ty, N> src) {
  for (unsigned int i = 0; i != N; i++) {
    if (src[i] == 0)
      return 0;
  }
  return 1;
}

template <typename Ty, int N>
inline __SEIEED::vector_type_t<Ty, N>
__esimd_dp4(__SEIEED::vector_type_t<Ty, N> v1,
            __SEIEED::vector_type_t<Ty, N> v2) {
  __SEIEED::vector_type_t<Ty, N> retv;
  for (auto i = 0; i != N; i += 4) {
    Ty dp = (v1[i] * v2[i]) + (v1[i + 1] * v2[i + 1]) +
            (v1[i + 2] * v2[i + 2]) + (v1[i + 3] * v2[i + 3]);
    retv[i] = dp;
    retv[i + 1] = dp;
    retv[i + 2] = dp;
    retv[i + 3] = dp;
  }
  return retv;
}

inline void __esimd_slm_init(size_t size) {
  sycl::detail::ESIMDDeviceInterface *I =
      sycl::detail::getESIMDDeviceInterface();

  I->cm_slm_init_ptr(size);
}

inline void __esimd_barrier() {
  sycl::detail::ESIMDDeviceInterface *I =
      sycl::detail::getESIMDDeviceInterface();

  I->cm_barrier_ptr();
}

inline void __esimd_sbarrier(
    sycl::ext::intel::experimental::esimd::EsimdSbarrierType flag) {
  sycl::detail::ESIMDDeviceInterface *I =
      sycl::detail::getESIMDDeviceInterface();

  I->cm_sbarrier_ptr((uint32_t)flag);
}

inline void __esimd_slm_fence(uint8_t cntl) {}

template <typename Ty, int N>
inline __SEIEED::vector_type_t<Ty, N>
__esimd_slm_read(__SEIEED::vector_type_t<uint32_t, N> addrs,
                 __SEIEED::simd_mask_storage_t<N> pred) {
  __SEIEED::vector_type_t<Ty, N> retv;
  sycl::detail::ESIMDDeviceInterface *I =
      sycl::detail::getESIMDDeviceInterface();

  char *SlmBase = I->__cm_emu_get_slm_ptr();
  for (int i = 0; i < N; ++i) {
    if (pred[i]) {
      Ty *addr = reinterpret_cast<Ty *>(addrs[i] + SlmBase);
      retv[i] = *addr;
    }
  }

  return retv;
}

// slm_write does SLM scatter
template <typename Ty, int N>
inline void __esimd_slm_write(__SEIEED::vector_type_t<uint32_t, N> addrs,
                              __SEIEED::vector_type_t<Ty, N> vals,
                              __SEIEED::vector_type_t<uint16_t, N> pred) {
  sycl::detail::ESIMDDeviceInterface *I =
      sycl::detail::getESIMDDeviceInterface();

  char *SlmBase = I->__cm_emu_get_slm_ptr();
  for (int i = 0; i < N; ++i) {
    if (pred[i]) {
      Ty *addr = reinterpret_cast<Ty *>(addrs[i] + SlmBase);
      *addr = vals[i];
    }
  }
}

// slm_block_read reads a block of data from SLM
template <typename Ty, int N>
inline __SEIEED::vector_type_t<Ty, N> __esimd_slm_block_read(uint32_t addr) {
  __SEIEED::vector_type_t<Ty, N> retv;
  sycl::detail::ESIMDDeviceInterface *I =
      sycl::detail::getESIMDDeviceInterface();
  char *SlmBase = I->__cm_emu_get_slm_ptr();
  addr <<= 4;
  for (int i = 0; i < N; ++i) {
    Ty *SlmAddr = reinterpret_cast<Ty *>(addr + SlmBase);
    retv[i] = *SlmAddr;
    addr += sizeof(Ty);
  }
  return retv;
}

// slm_block_write writes a block of data to SLM
template <typename Ty, int N>
inline void __esimd_slm_block_write(uint32_t addr,
                                    __SEIEED::vector_type_t<Ty, N> vals) {
  sycl::detail::ESIMDDeviceInterface *I =
      sycl::detail::getESIMDDeviceInterface();
  char *SlmBase = I->__cm_emu_get_slm_ptr();
  addr <<= 4;
  for (int i = 0; i < N; ++i) {
    Ty *SlmAddr = reinterpret_cast<Ty *>(addr + SlmBase);
    *SlmAddr = vals[i];
    addr += sizeof(Ty);
  }
}

// slm_read4 does SLM gather4
template <typename Ty, int N, __SEIEE::rgba_channel_mask Mask>
inline __SEIEED::vector_type_t<Ty, N * get_num_channels_enabled(Mask)>
__esimd_slm_read4(__SEIEED::vector_type_t<uint32_t, N> addrs,
                  __SEIEED::simd_mask_storage_t<N> pred) {
  __SEIEED::vector_type_t<Ty, N * get_num_channels_enabled(Mask)> retv;
  sycl::detail::ESIMDDeviceInterface *I =
      sycl::detail::getESIMDDeviceInterface();
  char *ReadBase = I->__cm_emu_get_slm_ptr();

  unsigned int Next = 0;
  if (__SEIEE::is_channel_enabled(Mask, __SEIEE::rgba_channel::R)) {
    for (int I = 0; I < N; I++, Next++) {
      if (pred[I]) {
        Ty *addr = reinterpret_cast<Ty *>(addrs[I] + ReadBase);
        retv[Next] = *addr;
      }
    }
  }

  ReadBase += sizeof(Ty);

  if (__SEIEE::is_channel_enabled(Mask, __SEIEE::rgba_channel::G)) {
    for (int I = 0; I < N; I++, Next++) {
      if (pred[I]) {
        Ty *addr = reinterpret_cast<Ty *>(addrs[I] + ReadBase);
        retv[Next] = *addr;
      }
    }
  }

  ReadBase += sizeof(Ty);

  if (__SEIEE::is_channel_enabled(Mask, __SEIEE::rgba_channel::B)) {
    for (int I = 0; I < N; I++, Next++) {
      if (pred[I]) {
        Ty *addr = reinterpret_cast<Ty *>(addrs[I] + ReadBase);
        retv[Next] = *addr;
      }
    }
  }

  ReadBase += sizeof(Ty);

  if (__SEIEE::is_channel_enabled(Mask, __SEIEE::rgba_channel::A)) {
    for (int I = 0; I < N; I++, Next++) {
      if (pred[I]) {
        Ty *addr = reinterpret_cast<Ty *>(addrs[I] + ReadBase);
        retv[Next] = *addr;
      }
    }
  }
  return retv;
}

// slm_write4 does SLM scatter4
template <typename Ty, int N, __SEIEE::rgba_channel_mask Mask>
inline void __esimd_slm_write4(
    __SEIEED::vector_type_t<uint32_t, N> addrs,
    __SEIEED::vector_type_t<Ty, N * get_num_channels_enabled(Mask)> vals,
    __SEIEED::vector_type_t<uint16_t, N> pred) {

  sycl::detail::ESIMDDeviceInterface *I =
      sycl::detail::getESIMDDeviceInterface();
  char *WriteBase = I->__cm_emu_get_slm_ptr();

  unsigned int Next = 0;
  if (__SEIEE::is_channel_enabled(Mask, __SEIEE::rgba_channel::R)) {
    for (int I = 0; I < N; I++, Next++) {
      if (pred[I]) {
        Ty *addr = reinterpret_cast<Ty *>(addrs[I] + WriteBase);
        *addr = vals[Next];
      }
    }
  }

  WriteBase += sizeof(Ty);

  if (__SEIEE::is_channel_enabled(Mask, __SEIEE::rgba_channel::G)) {
    for (int I = 0; I < N; I++, Next++) {
      if (pred[I]) {
        Ty *addr = reinterpret_cast<Ty *>(addrs[I] + WriteBase);
        *addr = vals[Next];
      }
    }
  }

  WriteBase += sizeof(Ty);

  if (__SEIEE::is_channel_enabled(Mask, __SEIEE::rgba_channel::B)) {
    for (int I = 0; I < N; I++, Next++) {
      if (pred[I]) {
        Ty *addr = reinterpret_cast<Ty *>(addrs[I] + WriteBase);
        *addr = vals[Next];
      }
    }
  }

  WriteBase += sizeof(Ty);

  if (__SEIEE::is_channel_enabled(Mask, __SEIEE::rgba_channel::A)) {
    for (int I = 0; I < N; I++, Next++) {
      if (pred[I]) {
        Ty *addr = reinterpret_cast<Ty *>(addrs[I] + WriteBase);
        *addr = vals[Next];
      }
    }
  }
}

// slm_atomic: SLM atomic
template <__SEIEE::atomic_op Op, typename Ty, int N>
inline __SEIEED::vector_type_t<Ty, N>
__esimd_slm_atomic0(__SEIEED::vector_type_t<uint32_t, N> addrs,
                    __SEIEED::simd_mask_storage_t<N> pred) {
  __SEIEED::vector_type_t<Ty, N> retv;
  sycl::detail::ESIMDDeviceInterface *I =
      sycl::detail::getESIMDDeviceInterface();
  char *WriteBase = I->__cm_emu_get_slm_ptr();

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
  return retv;
}

template <__SEIEE::atomic_op Op, typename Ty, int N>
inline __SEIEED::vector_type_t<Ty, N>
__esimd_slm_atomic1(__SEIEED::vector_type_t<uint32_t, N> addrs,
                    __SEIEED::vector_type_t<Ty, N> src0,
                    __SEIEED::simd_mask_storage_t<N> pred) {
  __SEIEED::vector_type_t<Ty, N> retv;
  return retv;
}

template <__SEIEE::atomic_op Op, typename Ty, int N>
inline __SEIEED::vector_type_t<Ty, N>
__esimd_slm_atomic2(__SEIEED::vector_type_t<uint32_t, N> addrs,
                    __SEIEED::vector_type_t<Ty, N> src0,
                    __SEIEED::vector_type_t<Ty, N> src1,
                    __SEIEED::simd_mask_storage_t<N> pred) {
  __SEIEED::vector_type_t<Ty, N> retv;
  return retv;
}

template <__SEIEE::atomic_op Op, typename Ty, int N, __SEIEE::CacheHint L1H,
          __SEIEE::CacheHint L3H>
inline __SEIEED::vector_type_t<Ty, N>
__esimd_flat_atomic0(__SEIEED::vector_type_t<uint64_t, N> addrs,
                     __SEIEED::simd_mask_storage_t<N> pred) {
  __SEIEED::vector_type_t<Ty, N> retv;
  return retv;
}

template <__SEIEE::atomic_op Op, typename Ty, int N, __SEIEE::CacheHint L1H,
          __SEIEE::CacheHint L3H>
inline __SEIEED::vector_type_t<Ty, N>
__esimd_flat_atomic1(__SEIEED::vector_type_t<uint64_t, N> addrs,
                     __SEIEED::vector_type_t<Ty, N> src0,
                     __SEIEED::simd_mask_storage_t<N> pred) {
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

template <__SEIEE::atomic_op Op, typename Ty, int N, __SEIEE::CacheHint L1H,
          __SEIEE::CacheHint L3H>
inline __SEIEED::vector_type_t<Ty, N>
__esimd_flat_atomic2(__SEIEED::vector_type_t<uint64_t, N> addrs,
                     __SEIEED::vector_type_t<Ty, N> src0,
                     __SEIEED::vector_type_t<Ty, N> src1,
                     __SEIEED::simd_mask_storage_t<N> pred) {
  __SEIEED::vector_type_t<Ty, N> retv;
  return retv;
}

template <typename Ty, int N, typename SurfIndAliasTy>
inline __SEIEED::vector_type_t<Ty, N>
__esimd_block_read(SurfIndAliasTy surf_ind, uint32_t offset) {
  __SEIEED::vector_type_t<Ty, N> retv;
  sycl::detail::ESIMDDeviceInterface *I =
      sycl::detail::getESIMDDeviceInterface();

  char *readBase;
  uint32_t width;
  std::mutex *mutexLock;

  auto BufferHandle = __SEIEED::AccessorPrivateProxy::getPtr(surf_ind);

  I->sycl_get_cm_buffer_params_ptr(BufferHandle, &readBase, &width, &mutexLock);

  std::unique_lock<std::mutex> lock(*mutexLock);

  for (int idx = 0; idx < N; idx++) {
    if (offset >= width) {
      retv[idx] = 0;
    } else {
      retv[idx] = *((Ty *)(readBase + offset));
    }
    offset += (uint32_t)sizeof(Ty);
  }

  return retv;
}

template <typename Ty, int N, typename SurfIndAliasTy>
inline void __esimd_block_write(SurfIndAliasTy surf_ind, uint32_t offset,
                                __SEIEED::vector_type_t<Ty, N> vals) {
  sycl::detail::ESIMDDeviceInterface *I =
      sycl::detail::getESIMDDeviceInterface();

  char *writeBase;
  uint32_t width;
  std::mutex *mutexLock;

  auto BufferHandle = __SEIEED::AccessorPrivateProxy::getPtr(surf_ind);

  I->sycl_get_cm_buffer_params_ptr(BufferHandle, &writeBase, &width,
                                   &mutexLock);

  std::unique_lock<std::mutex> lock(*mutexLock);

  offset <<= 4;

  for (int idx = 0; idx < N; idx++) {
    if (offset < width) {
      *((Ty *)(writeBase + offset)) = vals[idx];
    } else {
      break;
    }
    offset += (uint32_t)sizeof(Ty);
  }

  /// TODO : Optimize
  I->cm_fence_ptr();
}

/// \brief esimd_get_value
///
/// @param acc the SYCL accessor.
///
/// Returns the binding table index value.
///
template <typename AccessorTy>
inline uint32_t __esimd_get_value(AccessorTy acc) {
  throw cl::sycl::feature_not_supported();
  return 0;
}

/// \brief Raw sends load.
///
/// @param modifier	the send message flags (Bit-0: isSendc, Bit-1: isEOT).
///
/// @param execSize the execution size, which must be a compile time constant.
///
/// @param pred the predicate to specify enabled channels.
///
/// @param numSrc0 the number of GRFs for source-0, which must be a compile time
/// constant.
///
/// @param numSrc1 the number of GRFs for source-1, which must be a compile time
/// constant.
///
/// @param numDst the number of GRFs for destination, which must be a compile
/// time constant.
///
/// @param sfid the shared function ID, which must be a compile time constant.
///
/// @param exDesc the extended message descriptor.
///
/// @param msgDesc the message descriptor.
///
/// @param msgSrc0 the first source operand of send message.
///
/// @param msgSrc1 the second source operand of send message.
///
/// @param msgDst the destination operand of send message.
///
/// Returns a simd vector of type Ty1 and size N1.
///
template <typename Ty1, int N1, typename Ty2, int N2, typename Ty3, int N3,
          int N>
inline __SEIEED::vector_type_t<Ty1, N1> __esimd_raw_sends_load(
    uint8_t modifier, uint8_t execSize, __SEIEED::simd_mask_storage_t<N> pred,
    uint8_t numSrc0, uint8_t numSrc1, uint8_t numDst, uint8_t sfid,
    uint32_t exDesc, uint32_t msgDesc, __SEIEED::vector_type_t<Ty2, N2> msgSrc0,
    __SEIEED::vector_type_t<Ty3, N3> msgSrc1,
    __SEIEED::vector_type_t<Ty1, N1> msgDst) {
  throw cl::sycl::feature_not_supported();
  return 0;
}

/// \brief Raw send load.
///
/// @param modifier	the send message flags (Bit-0: isSendc, Bit-1: isEOT).
///
/// @param execSize the execution size, which must be a compile time constant.
///
/// @param pred the predicate to specify enabled channels.
///
/// @param numSrc0 the number of GRFs for source-0, which must be a compile time
/// constant.
///
/// @param numDst the number of GRFs for destination, which must be a compile
/// time constant.
///
/// @param sfid the shared function ID, which must be a compile time constant.
///
/// @param exDesc the extended message descriptor.
///
/// @param msgDesc the message descriptor.
///
/// @param msgSrc0 the first source operand of send message.
///
/// @param msgDst the destination operand of send message.
///
/// Returns a simd vector of type Ty1 and size N1.
///
template <typename Ty1, int N1, typename Ty2, int N2, int N>
inline __SEIEED::vector_type_t<Ty1, N1> __esimd_raw_send_load(
    uint8_t modifier, uint8_t execSize, __SEIEED::simd_mask_storage_t<N> pred,
    uint8_t numSrc0, uint8_t numDst, uint8_t sfid, uint32_t exDesc,
    uint32_t msgDesc, __SEIEED::vector_type_t<Ty2, N2> msgSrc0,
    __SEIEED::vector_type_t<Ty1, N1> msgDst) {
  throw cl::sycl::feature_not_supported();
  return 0;
}

/// \brief Raw sends store.
///
/// @param modifier	the send message flags (Bit-0: isSendc, Bit-1: isEOT).
///
/// @param execSize the execution size, which must be a compile time constant.
///
/// @param pred the predicate to specify enabled channels.
///
/// @param numSrc0 the number of GRFs for source-0, which must be a compile time
/// constant.
///
/// @param numSrc1 the number of GRFs for source-1, which must be a compile time
/// constant.
///
/// @param sfid the shared function ID, which must be a compile time constant.
///
/// @param exDesc the extended message descriptor.
///
/// @param msgDesc the message descriptor.
///
/// @param msgSrc0 the first source operand of send message.
///
/// @param msgSrc1 the second source operand of send message.
///
template <typename Ty1, int N1, typename Ty2, int N2, int N>
inline void __esimd_raw_sends_store(uint8_t modifier, uint8_t execSize,
                                    __SEIEED::simd_mask_storage_t<N> pred,
                                    uint8_t numSrc0, uint8_t numSrc1,
                                    uint8_t sfid, uint32_t exDesc,
                                    uint32_t msgDesc,
                                    __SEIEED::vector_type_t<Ty1, N1> msgSrc0,
                                    __SEIEED::vector_type_t<Ty2, N2> msgSrc1) {
  throw cl::sycl::feature_not_supported();
}

/// \brief Raw send store.
///
/// @param modifier	the send message flags (Bit-0: isSendc, Bit-1: isEOT).
///
/// @param execSize the execution size, which must be a compile time constant.
///
/// @param pred the predicate to specify enabled channels.
///
/// @param numSrc0 the number of GRFs for source-0, which must be a compile time
/// constant.
///
/// @param sfid the shared function ID, which must be a compile time constant.
///
/// @param exDesc the extended message descriptor.
///
/// @param msgDesc the message descriptor.
///
/// @param msgSrc0 the first source operand of send message.
///
template <typename Ty1, int N1, int N>
inline void __esimd_raw_send_store(uint8_t modifier, uint8_t execSize,
                                   __SEIEED::simd_mask_storage_t<N> pred,
                                   uint8_t numSrc0, uint8_t sfid,
                                   uint32_t exDesc, uint32_t msgDesc,
                                   __SEIEED::vector_type_t<Ty1, N1> msgSrc0) {
  throw cl::sycl::feature_not_supported();
}

#endif // __SYCL_DEVICE_ONLY__
