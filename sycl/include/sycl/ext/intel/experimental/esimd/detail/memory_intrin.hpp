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
#else
  template <typename AccessorTy>
  static auto getImageRange(const AccessorTy &Acc) {
    return Acc.getAccessRange();
  }
  static auto getElemSize(const sycl::detail::AccessorBaseHost &Acc) {
    return Acc.getElemSize();
  }
#endif
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

#define __SEIEE sycl::ext::intel::experimental::esimd
#define __SEIEED sycl::ext::intel::experimental::esimd::detail

// flat_read does flat-address gather
template <typename Ty, int N, int NumBlk = 0,
          __SEIEE::CacheHint L1H = __SEIEE::CacheHint::None,
          __SEIEE::CacheHint L3H = __SEIEE::CacheHint::None>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION
    __SEIEED::vector_type_t<Ty, N * __SEIEED::ElemsPerAddrDecoding(NumBlk)>
    __esimd_flat_read(__SEIEED::vector_type_t<uint64_t, N> addrs,
                      int ElemsPerAddr = NumBlk,
                      __SEIEED::vector_type_t<uint16_t, N> pred = 1);

// flat_write does flat-address scatter
template <typename Ty, int N, int NumBlk = 0,
          __SEIEE::CacheHint L1H = __SEIEE::CacheHint::None,
          __SEIEE::CacheHint L3H = __SEIEE::CacheHint::None>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION void __esimd_flat_write(
    __SEIEED::vector_type_t<uint64_t, N> addrs,
    __SEIEED::vector_type_t<Ty, N * __SEIEED::ElemsPerAddrDecoding(NumBlk)>
        vals,
    int ElemsPerAddr = NumBlk, __SEIEED::vector_type_t<uint16_t, N> pred = 1);

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
                       __SEIEED::vector_type_t<uint16_t, N> pred = 1);

// flat_write does flat-address scatter
template <typename Ty, int N, __SEIEE::rgba_channel_mask Mask,
          __SEIEE::CacheHint L1H = __SEIEE::CacheHint::None,
          __SEIEE::CacheHint L3H = __SEIEE::CacheHint::None>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION void __esimd_flat_write4(
    __SEIEED::vector_type_t<uint64_t, N> addrs,
    __SEIEED::vector_type_t<Ty, N * get_num_channels_enabled(Mask)> vals,
    __SEIEED::vector_type_t<uint16_t, N> pred = 1);

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
                  __SEIEED::vector_type_t<uint32_t, N> elem_offsets)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  static_assert(N == 1 || N == 8 || N == 16);
  static_assert(TySizeLog2 <= 2);
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
__esimd_surf_write(__SEIEED::vector_type_t<uint16_t, N> pred, int16_t scale,
                   SurfIndAliasTy surf_ind, uint32_t global_offset,
                   __SEIEED::vector_type_t<uint32_t, N> elem_offsets,
                   __SEIEED::vector_type_t<Ty, N> vals)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  static_assert(N == 1 || N == 8 || N == 16);
  static_assert(TySizeLog2 <= 2);
  static_assert(std::is_integral<Ty>::value || TySizeLog2 == 2);
  throw cl::sycl::feature_not_supported();
}
#endif // __SYCL_DEVICE_ONLY__

// TODO bring the parameter order of __esimd* intrinsics in accordance with the
// correponsing BE intrinsicics parameter order.

// flat_atomic: flat-address atomic
template <__SEIEE::atomic_op Op, typename Ty, int N,
          __SEIEE::CacheHint L1H = __SEIEE::CacheHint::None,
          __SEIEE::CacheHint L3H = __SEIEE::CacheHint::None>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<Ty, N>
__esimd_flat_atomic0(__SEIEED::vector_type_t<uint64_t, N> addrs,
                     __SEIEED::vector_type_t<uint16_t, N> pred);

template <__SEIEE::atomic_op Op, typename Ty, int N,
          __SEIEE::CacheHint L1H = __SEIEE::CacheHint::None,
          __SEIEE::CacheHint L3H = __SEIEE::CacheHint::None>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<Ty, N>
__esimd_flat_atomic1(__SEIEED::vector_type_t<uint64_t, N> addrs,
                     __SEIEED::vector_type_t<Ty, N> src0,
                     __SEIEED::vector_type_t<uint16_t, N> pred);

template <__SEIEE::atomic_op Op, typename Ty, int N,
          __SEIEE::CacheHint L1H = __SEIEE::CacheHint::None,
          __SEIEE::CacheHint L3H = __SEIEE::CacheHint::None>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<Ty, N>
__esimd_flat_atomic2(__SEIEED::vector_type_t<uint64_t, N> addrs,
                     __SEIEED::vector_type_t<Ty, N> src0,
                     __SEIEED::vector_type_t<Ty, N> src1,
                     __SEIEED::vector_type_t<uint16_t, N> pred);

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
                 __SEIEED::vector_type_t<uint16_t, N> pred = 1);

// slm_write does SLM scatter
template <typename Ty, int N>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION void
__esimd_slm_write(__SEIEED::vector_type_t<uint32_t, N> addrs,
                  __SEIEED::vector_type_t<Ty, N> vals,
                  __SEIEED::vector_type_t<uint16_t, N> pred = 1);

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
                      __SEIEED::vector_type_t<uint16_t, N> pred = 1);

// slm_write4 does SLM scatter4
template <typename Ty, int N, __SEIEE::rgba_channel_mask Mask>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION void __esimd_slm_write4(
    __SEIEED::vector_type_t<uint32_t, N> addrs,
    __SEIEED::vector_type_t<Ty, N * get_num_channels_enabled(Mask)> vals,
    __SEIEED::vector_type_t<uint16_t, N> pred = 1);

// slm_atomic: SLM atomic
template <__SEIEE::atomic_op Op, typename Ty, int N>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<Ty, N>
__esimd_slm_atomic0(__SEIEED::vector_type_t<uint32_t, N> addrs,
                    __SEIEED::vector_type_t<uint16_t, N> pred);

template <__SEIEE::atomic_op Op, typename Ty, int N>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<Ty, N>
__esimd_slm_atomic1(__SEIEED::vector_type_t<uint32_t, N> addrs,
                    __SEIEED::vector_type_t<Ty, N> src0,
                    __SEIEED::vector_type_t<uint16_t, N> pred);

template <__SEIEE::atomic_op Op, typename Ty, int N>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<Ty, N>
__esimd_slm_atomic2(__SEIEED::vector_type_t<uint32_t, N> addrs,
                    __SEIEED::vector_type_t<Ty, N> src0,
                    __SEIEED::vector_type_t<Ty, N> src1,
                    __SEIEED::vector_type_t<uint16_t, N> pred);

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
                       __SEIEED::vector_type_t<uint16_t, N> pred,
                       uint8_t numSrc0, uint8_t numSrc1, uint8_t numDst,
                       uint8_t sfid, uint32_t exDesc, uint32_t msgDesc,
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
                      __SEIEED::vector_type_t<uint16_t, N> pred,
                      uint8_t numSrc0, uint8_t numDst, uint8_t sfid,
                      uint32_t exDesc, uint32_t msgDesc,
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
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION void
__esimd_raw_sends_store(uint8_t modifier, uint8_t execSize,
                        __SEIEED::vector_type_t<uint16_t, N> pred,
                        uint8_t numSrc0, uint8_t numSrc1, uint8_t sfid,
                        uint32_t exDesc, uint32_t msgDesc,
                        __SEIEED::vector_type_t<Ty1, N1> msgSrc0,
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
                       __SEIEED::vector_type_t<uint16_t, N> pred,
                       uint8_t numSrc0, uint8_t sfid, uint32_t exDesc,
                       uint32_t msgDesc,
                       __SEIEED::vector_type_t<Ty1, N1> msgSrc0);
#ifndef __SYCL_DEVICE_ONLY__

template <typename Ty, int N, int NumBlk, __SEIEE::CacheHint L1H,
          __SEIEE::CacheHint L3H>
inline __SEIEED::vector_type_t<Ty, N * __SEIEED::ElemsPerAddrDecoding(NumBlk)>
__esimd_flat_read(__SEIEED::vector_type_t<uint64_t, N> addrs, int ElemsPerAddr,
                  __SEIEED::vector_type_t<uint16_t, N> pred) {
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
                   __SEIEED::vector_type_t<uint16_t, N> pred) {
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
    int ElemsPerAddr, __SEIEED::vector_type_t<uint16_t, N> pred) {
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
    __SEIEED::vector_type_t<uint16_t, N> pred) {
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
  // On host the input surface is modeled as sycl image 2d object,
  // and the read/write access is done through accessor,
  // which is passed in as the handle argument.
  auto range = __SEIEED::AccessorPrivateProxy::getImageRange(handle);
  unsigned bpp = __SEIEED::AccessorPrivateProxy::getElemSize(handle);
  unsigned vpp = bpp / sizeof(Ty);
  unsigned int i = x / bpp;
  unsigned int j = y;

  assert(x % bpp == 0);
  unsigned int xbound = range[0] - 1;
  unsigned int ybound = range[1] - 1;

  __SEIEED::vector_type_t<Ty, M * N> vals;
  for (int row = 0; row < M; row++) {
    for (int col = 0; col < N; col += vpp) {
      unsigned int xoff = (i > xbound) ? xbound : i;
      unsigned int yoff = (j > ybound) ? ybound : j;
      auto coords = cl::sycl::cl_int2(xoff, yoff);
      cl::sycl::cl_uint4 data = handle.read(coords);

      __SEIEED::vector_type_t<unsigned int, 4> res;
      for (int idx = 0; idx < 4; idx++) {
        res[idx] = data[idx];
      }

      constexpr int refN = sizeof(cl::sycl::cl_uint4) / sizeof(Ty);
      unsigned int stride = sizeof(cl::sycl::cl_uint4) / bpp;
      using refTy = __SEIEED::vector_type_t<Ty, refN>;
      auto ref = reinterpret_cast<refTy>(res);

      unsigned int offset1 = col + row * N;
      unsigned int offset2 = 0;
      for (int idx = 0; idx < vpp; idx++) {
        vals[offset1] = ref[offset2];
        offset1++;
        offset2 += stride;
      }
      i++;
    }
    i = x / bpp;
    j++;
  }

  return vals;
}

template <typename Ty, int M, int N, typename TACC>
inline void __esimd_media_block_store(unsigned modififer, TACC handle,
                                      unsigned plane, unsigned width,
                                      unsigned x, unsigned y,
                                      __SEIEED::vector_type_t<Ty, M * N> vals) {
  unsigned bpp = __SEIEED::AccessorPrivateProxy::getElemSize(handle);
  unsigned vpp = bpp / sizeof(Ty);
  auto range = __SEIEED::AccessorPrivateProxy::getImageRange(handle);
  unsigned int i = x / bpp;
  unsigned int j = y;

  assert(x % bpp == 0);

  for (int row = 0; row < M; row++) {
    for (int col = 0; col < N; col += vpp) {
      constexpr int Sz = sizeof(cl::sycl::cl_uint4) / sizeof(Ty);
      __SEIEED::vector_type_t<Ty, Sz> res = 0;

      unsigned int offset1 = col + row * N;
      unsigned int offset2 = 0;
      unsigned int stride = sizeof(cl::sycl::cl_uint4) / bpp;
      for (int idx = 0; idx < vpp; idx++) {
        res[offset2] = vals[offset1];
        offset1++;
        offset2 += stride;
      }

      using refTy = __SEIEED::vector_type_t<unsigned int, 4>;
      auto ref = reinterpret_cast<refTy>(res);

      cl::sycl::cl_uint4 data;
      for (int idx = 0; idx < 4; idx++) {
        data[idx] = ref[idx];
      }

      if (i < range[0] && j < range[1]) {
        auto coords = cl::sycl::cl_int2(i, j);
        handle.write(coords, data);
      }
      i++;
    }
    i = x / bpp;
    j++;
  }
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

/// TODO
inline void __esimd_barrier() {}

inline void __esimd_sbarrier(__SEIEE::split_barrier_action flag) {}

inline void __esimd_slm_fence(uint8_t cntl) {}

template <typename Ty, int N>
inline __SEIEED::vector_type_t<Ty, N>
__esimd_slm_read(__SEIEED::vector_type_t<uint32_t, N> addrs,
                 __SEIEED::vector_type_t<uint16_t, N> pred) {
  __SEIEED::vector_type_t<Ty, N> retv;
  return retv;
}

// slm_write does SLM scatter
template <typename Ty, int N>
inline void __esimd_slm_write(__SEIEED::vector_type_t<uint32_t, N> addrs,
                              __SEIEED::vector_type_t<Ty, N> vals,
                              __SEIEED::vector_type_t<uint16_t, N> pred) {}

// slm_block_read reads a block of data from SLM
template <typename Ty, int N>
inline __SEIEED::vector_type_t<Ty, N> __esimd_slm_block_read(uint32_t addr) {
  __SEIEED::vector_type_t<Ty, N> retv;
  return retv;
}

// slm_block_write writes a block of data to SLM
template <typename Ty, int N>
inline void __esimd_slm_block_write(uint32_t addr,
                                    __SEIEED::vector_type_t<Ty, N> vals) {}

// slm_read4 does SLM gather4
template <typename Ty, int N, __SEIEE::rgba_channel_mask Mask>
inline __SEIEED::vector_type_t<Ty, N * get_num_channels_enabled(Mask)>
__esimd_slm_read4(__SEIEED::vector_type_t<uint32_t, N> addrs,
                  __SEIEED::vector_type_t<uint16_t, N> pred) {
  __SEIEED::vector_type_t<Ty, N * get_num_channels_enabled(Mask)> retv;
  return retv;
}

// slm_write4 does SLM scatter4
template <typename Ty, int N, __SEIEE::rgba_channel_mask Mask>
inline void __esimd_slm_write4(
    __SEIEED::vector_type_t<uint32_t, N> addrs,
    __SEIEED::vector_type_t<Ty, N * get_num_channels_enabled(Mask)> vals,
    __SEIEED::vector_type_t<uint16_t, N> pred) {}

// slm_atomic: SLM atomic
template <__SEIEE::atomic_op Op, typename Ty, int N>
inline __SEIEED::vector_type_t<Ty, N>
__esimd_slm_atomic0(__SEIEED::vector_type_t<uint32_t, N> addrs,
                    __SEIEED::vector_type_t<uint16_t, N> pred) {
  __SEIEED::vector_type_t<Ty, N> retv;
  return retv;
}

template <__SEIEE::atomic_op Op, typename Ty, int N>
inline __SEIEED::vector_type_t<Ty, N>
__esimd_slm_atomic1(__SEIEED::vector_type_t<uint32_t, N> addrs,
                    __SEIEED::vector_type_t<Ty, N> src0,
                    __SEIEED::vector_type_t<uint16_t, N> pred) {
  __SEIEED::vector_type_t<Ty, N> retv;
  return retv;
}

template <__SEIEE::atomic_op Op, typename Ty, int N>
inline __SEIEED::vector_type_t<Ty, N>
__esimd_slm_atomic2(__SEIEED::vector_type_t<uint32_t, N> addrs,
                    __SEIEED::vector_type_t<Ty, N> src0,
                    __SEIEED::vector_type_t<Ty, N> src1,
                    __SEIEED::vector_type_t<uint16_t, N> pred) {
  __SEIEED::vector_type_t<Ty, N> retv;
  return retv;
}

template <__SEIEE::atomic_op Op, typename Ty, int N, __SEIEE::CacheHint L1H,
          __SEIEE::CacheHint L3H>
inline __SEIEED::vector_type_t<Ty, N>
__esimd_flat_atomic0(__SEIEED::vector_type_t<uint64_t, N> addrs,
                     __SEIEED::vector_type_t<uint16_t, N> pred) {
  __SEIEED::vector_type_t<Ty, N> retv;
  return retv;
}

template <__SEIEE::atomic_op Op, typename Ty, int N, __SEIEE::CacheHint L1H,
          __SEIEE::CacheHint L3H>
inline __SEIEED::vector_type_t<Ty, N>
__esimd_flat_atomic1(__SEIEED::vector_type_t<uint64_t, N> addrs,
                     __SEIEED::vector_type_t<Ty, N> src0,
                     __SEIEED::vector_type_t<uint16_t, N> pred) {
  __SEIEED::vector_type_t<Ty, N> retv;
  return retv;
}

template <__SEIEE::atomic_op Op, typename Ty, int N, __SEIEE::CacheHint L1H,
          __SEIEE::CacheHint L3H>
inline __SEIEED::vector_type_t<Ty, N>
__esimd_flat_atomic2(__SEIEED::vector_type_t<uint64_t, N> addrs,
                     __SEIEED::vector_type_t<Ty, N> src0,
                     __SEIEED::vector_type_t<Ty, N> src1,
                     __SEIEED::vector_type_t<uint16_t, N> pred) {
  __SEIEED::vector_type_t<Ty, N> retv;
  return retv;
}

template <typename Ty, int N, typename SurfIndAliasTy>
inline __SEIEED::vector_type_t<Ty, N>
__esimd_block_read(SurfIndAliasTy surf_ind, uint32_t offset) {
  throw cl::sycl::feature_not_supported();
  return __SEIEED::vector_type_t<Ty, N>();
}

template <typename Ty, int N, typename SurfIndAliasTy>
inline void __esimd_block_write(SurfIndAliasTy surf_ind, uint32_t offset,
                                __SEIEED::vector_type_t<Ty, N> vals) {

  throw cl::sycl::feature_not_supported();
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
inline __SEIEED::vector_type_t<Ty1, N1>
__esimd_raw_sends_load(uint8_t modifier, uint8_t execSize,
                       __SEIEED::vector_type_t<uint16_t, N> pred,
                       uint8_t numSrc0, uint8_t numSrc1, uint8_t numDst,
                       uint8_t sfid, uint32_t exDesc, uint32_t msgDesc,
                       __SEIEED::vector_type_t<Ty2, N2> msgSrc0,
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
inline __SEIEED::vector_type_t<Ty1, N1>
__esimd_raw_send_load(uint8_t modifier, uint8_t execSize,
                      __SEIEED::vector_type_t<uint16_t, N> pred,
                      uint8_t numSrc0, uint8_t numDst, uint8_t sfid,
                      uint32_t exDesc, uint32_t msgDesc,
                      __SEIEED::vector_type_t<Ty2, N2> msgSrc0,
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
                                    __SEIEED::vector_type_t<uint16_t, N> pred,
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
                                   __SEIEED::vector_type_t<uint16_t, N> pred,
                                   uint8_t numSrc0, uint8_t sfid,
                                   uint32_t exDesc, uint32_t msgDesc,
                                   __SEIEED::vector_type_t<Ty1, N1> msgSrc0) {
  throw cl::sycl::feature_not_supported();
}

#endif // __SYCL_DEVICE_ONLY__

#undef __SEIEED
#undef __SEIEE
