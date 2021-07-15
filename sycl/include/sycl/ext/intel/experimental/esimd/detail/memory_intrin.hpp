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
#else  // __SYCL_DEVICE_ONLY__
  template <typename AccessorTy>
  static auto getImageRange(const AccessorTy &Acc) {
    return Acc.getAccessRange();
  }
  static auto getElemSize(const sycl::detail::AccessorBaseHost &Acc) {
    return Acc.getElemSize();
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
__esimd_surf_write(__SEIEED::vector_type_t<uint16_t, N> pred, int16_t scale,
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

/// ESIMD_CPU Emulation support using esimd_cpu plugin

#define __SYCL_EXPLICIT_SIMD_PLUGIN__

// Header files required for accessing CM-managed memory - Surface,
// buffer, etc
namespace cm_support {
#include <CL/cm_rt.h>
} // namespace cm_support

#include <CL/sycl/backend_types.hpp>
#include <CL/sycl/detail/pi.hpp>
#include <sycl/ext/intel/experimental/esimd/detail/atomic_intrin.hpp>
#include <sycl/include/CL/sycl/INTEL/esimd/detail/emu/esimdcpu_device_interface.hpp>

namespace raw_send {

enum class msgField : short {
  OP,
  VNNI,
  ADDRSIZE,
  DATASIZE,
  VECTSIZE,
  TRANSPOSE,
  CACHE,
  DSTLEN,
  SRC0LEN,
  ADDRTYPE
};

enum class msgOp : short {
  DP_LOAD = 0x0, // scatter/vector load
  LOAD_2D = 0x3,
  DP_STORE = 0x4, // scatter/vector store
  STORE_2D = 0x7,
  OP_MAX = 0x3F
};

typedef struct _bitfields_ {
  uint32_t offset;
  uint32_t mask;
} bitfields;

const bitfields BIT_FIELDS[10] = {
    {0, 0x3F},  // OP / 6 bits
    {7, 0x1},   // VNNI -> LOAD only
    {7, 0x3},   // Address size
    {9, 0x7},   // DATASIZE
    {12, 0x7},  // VECTSIZE
    {15, 0x1},  // TRANSPOSE -> LOAD only
    {17, 0x7},  // CACHE
    {20, 0x1F}, // DSTLEN
    {25, 0xF},  // SRC0LEN,
    {29, 0x3}   // ADDRTYPE
};
uint32_t inline getMsgField(uint32_t msg, msgField field) {
  uint32_t idx = static_cast<uint32_t>(field);
  return ((msg >> BIT_FIELDS[idx].offset) & BIT_FIELDS[idx].mask);
}

auto inline getMsgOp(uint32_t msg) {
  msgOp ret;
  ret = static_cast<msgOp>(getMsgField((uint32_t)msg, msgField::OP));
  return ret;
}

template <typename T, unsigned N>
uint64_t inline getSurfaceBaseAddr(__SEIEED::vector_type_t<T, N> addrMsg) {
  constexpr int sizeofT = sizeof(T);
  uint64_t Ret = 0;

  if constexpr (sizeofT == 4) {
    Ret = (uint64_t)addrMsg[1] << 32;
    Ret |= (uint64_t)addrMsg[0];
  } else if constexpr (sizeofT == 8) {
    Ret = addrMsg[0];
  }

  return Ret;
}

template <typename T, unsigned N>
uint64_t inline getLaneAddr(__SEIEED::vector_type_t<T, N> addrMsg,
                            unsigned lane_id) {
  // (matrix_ref<T, R, C> addrMsg)
  // vector_ref<uint64_t, 1> addr_ref = addrMsg.template select<1, 1, 2, 1>(0, 2
  // * lane_id).template format<uint64_t, 1, 1>(); return addr_ref(0);
  throw cl::sycl::feature_not_supported();
}

template <typename T, unsigned N>
auto inline getSurfaceDim(__SEIEED::vector_type_t<T, N> addrMsg) {
  __SEIEED::vector_type_t<uint32_t, 4> Ret;
  constexpr int sizeofT = sizeof(T);

  static_assert(sizeofT == 4, "Unsupported addrMsg format!!");

  if constexpr (sizeofT == 4) {
    for (int idx = 0; idx < 4; idx++) {
      Ret[idx] = addrMsg[idx + 2];
    }
  }

  return Ret;
}

template <typename T, unsigned N>
auto inline getBlockOffsets(__SEIEED::vector_type_t<T, N> addrMsg) {
  __SEIEED::vector_type_t<int32_t, 4> Ret;
  constexpr int sizeofT = sizeof(T);

  static_assert(sizeofT == 4, "Unsupported addrMsg format!!");

  if constexpr (sizeofT == 4) {
    for (int idx = 0; idx < 4; idx++) {
      Ret[idx] = static_cast<int32_t>(addrMsg[idx + 5]);
    }
  }

  return Ret;
}

template <typename T, unsigned N>
auto inline getBlockDim(__SEIEED::vector_type_t<T, N> addrMsg) {
  __SEIEED::vector_type_t<unsigned char, 4> Ret;
  constexpr int sizeofT = sizeof(T);
  T RawValue = 0;

  static_assert(sizeofT == 4, "Unsupported addrMsg format!!");

  if constexpr (sizeofT == 4) {
    RawValue = addrMsg[7];
    Ret[0] = (unsigned char)(RawValue & 0xFF);         // width
    Ret[1] = (unsigned char)((RawValue >> 8) & 0xFF);  // height
    Ret[2] = (unsigned char)((RawValue >> 24) & 0xFF); // For ArrayLen
  }

  assert(RawValue != 0);

  return Ret;
}

template <typename T, unsigned N>
auto inline getArrayLen(__SEIEED::vector_type_t<T, N> addrMsg) {
  auto blkDim = getBlockDim<T, N>(addrMsg);
  return (blkDim[2] >> 4);
}

} // namespace raw_send

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

template <typename Ty, int N, typename SurfIndAliasTy, int TySizeLog2,
          __SEIEE::CacheHint L1H = __SEIEE::CacheHint::None,
          __SEIEE::CacheHint L3H = __SEIEE::CacheHint::None>
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
          __SEIEE::CacheHint L1H = __SEIEE::CacheHint::None,
          __SEIEE::CacheHint L3H = __SEIEE::CacheHint::None>
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

  I->sycl_get_cm_image_params_ptr(static_cast<void *>(handle.get_pointer()),
                                  &readBase, &imgWidth, &imgHeight, &bpp,
                                  &mutexLock);

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

  I->sycl_get_cm_image_params_ptr(static_cast<void *>(handle.get_pointer()),
                                  &writeBase, &imgWidth, &imgHeight, &bpp,
                                  &mutexLock);

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
                 __SEIEED::vector_type_t<uint16_t, N> pred) {
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
                  __SEIEED::vector_type_t<uint16_t, N> pred) {
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
                    __SEIEED::vector_type_t<uint16_t, N> pred) {
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
                     __SEIEED::vector_type_t<uint16_t, N> pred) {
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

  I->sycl_get_cm_buffer_params_ptr(static_cast<void *>(surf_ind.get_pointer()),
                                   &readBase, &width, &mutexLock);

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

  I->sycl_get_cm_buffer_params_ptr(static_cast<void *>(surf_ind.get_pointer()),
                                   &writeBase, &width, &mutexLock);

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
  assert(sfid == 0xF); // UGM type only

  __SEIEED::vector_type_t<Ty1, N1> retv;

  auto op = raw_send::getMsgOp(msgDesc);
  assert(op == raw_send::msgOp::LOAD_2D);
  uint64_t surfaceBase = raw_send::getSurfaceBaseAddr<Ty2, N2>(msgSrc0);
  auto surfaceDim = raw_send::getSurfaceDim<Ty2, N2>(msgSrc0);
  auto blockOffset = raw_send::getBlockOffsets<Ty2, N2>(msgSrc0);
  auto blockDim = raw_send::getBlockDim<Ty2, N2>(msgSrc0);
  auto arrayLen = raw_send::getArrayLen<Ty2, N2>(msgSrc0);

  unsigned SurfaceWidth = surfaceDim[0] + 1;
  unsigned SurfaceHeight = surfaceDim[1] + 1;
  unsigned SurfacePitch = surfaceDim[2] + 1;

  int X = blockOffset[0];
  int Y = blockOffset[1];
  int Width = blockDim[0] + 1;
  int Height = blockDim[1] + 1;
  int NBlks = arrayLen + 1;

  bool Transposed =
      raw_send::getMsgField(msgDesc, raw_send::msgField::TRANSPOSE);
  bool Transformed = raw_send::getMsgField(msgDesc, raw_send::msgField::VNNI);

  constexpr unsigned sizeofT = sizeof(Ty1);

  char *buffBase = (char *)surfaceBase;

  // TODO : Acquire mutex for the surface pointed to by 'surfaceBase'
  int vecIdx = 0;
  int blkCount = 0;

  for (int xBase = X * sizeofT; blkCount < NBlks; xBase += sizeofT * Width) {
    if (Transformed == true) {
      constexpr int elems_per_DW = (sizeofT == 1) ? 4 : 2; /// VNNI_pack
      if (Transposed == false) { /// Transform only load
        int yRead = Y * SurfacePitch;
        for (int u = 0; u < Height;
             u += elems_per_DW, yRead += SurfacePitch * elems_per_DW) {
          if ((yRead < 0) || (yRead >= SurfacePitch * SurfaceHeight)) {
            /// Vertically out-of-bound, padding zero on out of boundary
            for (int v = 0; v < Width; v += 1) {
              for (int k = 0; k < elems_per_DW; k++, vecIdx += 1) {
                retv[vecIdx] = (Ty1)(0);
              } // k loop
            }
            // vecIdx += Width * elems_per_DW;;
            continue;
          }

          int xRead = xBase;
          for (int v = 0; v < Width; v += 1, xRead += sizeofT) {
            if ((xRead < 0) || (xRead >= SurfaceWidth)) {
              /// Horizontally out-of-bound
              for (int k = 0; k < elems_per_DW; k++, vecIdx += 1) {
                retv[vecIdx] = (Ty1)(0);
              } // k loop
              // vecIdx += elems_per_DW;
              continue;
            }

            char *base = buffBase + yRead + xRead;
            int offset = 0;
            for (int k = 0; k < elems_per_DW; k++, vecIdx += 1) {
              retv[vecIdx] = *((Ty1 *)(base + offset));
              // Increasing in Y-direction
              offset += SurfacePitch;
            } // k loop
          }   // v loop
        }     /// u loop
      }       // Transposed = false
      else    // Transposed == true
      {       /// Transform & Transpose load
        int xRead = xBase;
        for (int v = 0; v < Width;
             v += elems_per_DW, xRead += sizeofT * elems_per_DW) {
          if ((xRead < 0) || (xRead >= SurfaceWidth)) {
            // Horizontally out-of-bound
            for (int u = 0; u < Height; u += 1) {
              for (int k = 0; k < elems_per_DW; k++, vecIdx += 1) {
                retv[vecIdx] = (Ty1)(0);
              } // k loop
            }
            // vecIdx += Height * elems_per_DW;
            continue;
          }

          int yRead = Y * SurfacePitch;
          for (int u = 0; u < Height; u += 1, yRead += SurfacePitch) {
            if ((yRead < 0) || (yRead >= SurfacePitch * SurfaceHeight)) {
              /// Vertically out-of-bound
              for (int k = 0; k < elems_per_DW; k++, vecIdx += 1) {
                retv[vecIdx] = (Ty1)(0);
              } // k loop
              // vecIdx += elems_per_DW;
              continue;
            }

            char *base = buffBase + yRead + xRead;
            int offset = 0;
            for (int k = 0; k < elems_per_DW; k++, vecIdx += 1) {
              retv[vecIdx] = *((Ty1 *)(base + offset));
              // Increasing in X-direction
              offset += sizeofT;
            } // k loop
          }   // u loop
        }     // v loop
      }       // Transposed == true
    }         // Transformed == true
    else      // (Transformed == false)
    {
      if (Transposed == false) { /// Linear load - no transform, no transpose
        int yRead = Y * SurfacePitch;
        for (int u = 0; u < Height; u += 1, yRead += SurfacePitch) {
          if ((yRead < 0) || (yRead >= SurfacePitch * SurfaceHeight)) {
            // Vertically Out-of-bound
            for (int v = 0; v < Width; v += 1, vecIdx += 1) {
              retv[vecIdx] = (Ty1)(0);
            }
            // vecIdx += Width;
            continue;
          }

          int xRead = xBase;
          for (int v = 0; v < Width; v += 1, xRead += sizeofT, vecIdx += 1) {
            if ((xRead >= 0) && (xRead < SurfaceWidth)) {
              retv[vecIdx] = *((Ty1 *)(buffBase + yRead + xRead));
            } else {
              // Horizontally out of bound
              retv[vecIdx] = (Ty1)(0);
            }
          } // v loop
        }   // u loop
      }     /// Transposed == false
      else  // Transposed = true
      {     /// Transpose load - no transform
        int xRead = xBase;
        for (int v = 0; v < Width; v += 1, xRead += sizeofT) {
          if ((xRead < 0) || (xRead > SurfaceWidth)) {
            // Horizontally out-of-bound
            for (int u = 0; u < Height; u += 1, vecIdx += 1) {
              retv[vecIdx] = (Ty1)(0);
            }
            // vecIdx += Height;
            continue;
          }

          int yRead = Y * SurfacePitch;
          for (int u = 0; u < Height;
               u += 1, yRead += SurfacePitch, vecIdx += 1) {
            if ((yRead >= 0) && (yRead < SurfacePitch * SurfaceHeight)) {
              retv[vecIdx] = *((Ty1 *)(buffBase + yRead + xRead));
            } else {
              // Vertically out of bound
              retv[vecIdx] = (Ty1)(0);
            }
          } // u loop
        }   // v loop
      }     // Transposed == true
    }       // Transformed == false
    blkCount += 1;
    vecIdx = blkCount * Width * Height;
  } // xBase loop

  return retv;
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
  assert(sfid == 0xF); // UGM type only
  auto op = raw_send::getMsgOp(msgDesc);
  assert(op == raw_send::msgOp::STORE_2D);
  uint64_t surfaceBase = raw_send::getSurfaceBaseAddr<Ty1, N1>(msgSrc0);
  auto surfaceDim = raw_send::getSurfaceDim<Ty1, N1>(msgSrc0);
  auto blockOffset = raw_send::getBlockOffsets<Ty1, N1>(msgSrc0);
  auto blockDim = raw_send::getBlockDim<Ty1, N1>(msgSrc0);

  unsigned SurfaceWidth = surfaceDim[0] + 1;
  unsigned SurfaceHeight = surfaceDim[1] + 1;
  unsigned SurfacePitch = surfaceDim[2] + 1;

  int X = blockOffset[0];
  int Y = blockOffset[1];
  int Width = blockDim[0] + 1;
  int Height = blockDim[1] + 1;

  constexpr unsigned sizeofT = sizeof(Ty2);

  char *buffBase = (char *)surfaceBase;

  int vecIdx = 0;
  int rowCount = 0;
  for (int yWrite = Y * SurfacePitch; rowCount < Height;
       yWrite += SurfacePitch) {
    if (yWrite == SurfacePitch * SurfaceHeight) {
      // Vertically Out-of-bound
      break;
    }
    int writeCount = 0;
    for (int xWrite = X * sizeofT; writeCount < Width;
         xWrite += sizeofT, vecIdx += 1, writeCount += 1) {
      if (xWrite >= 0 && xWrite < SurfaceWidth) {
        *((Ty2 *)(buffBase + yWrite + xWrite)) = msgSrc1[vecIdx];
      }
    } // xWrite loop
    rowCount += 1;
  } // yWrite loop
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
  auto op = raw_send::getMsgOp(msgDesc);

  if (op == raw_send::msgOp::LOAD_2D) {
    // Prefetch?
    return;
  }

  throw cl::sycl::feature_not_supported();
}

#endif // __SYCL_DEVICE_ONLY__

#undef __SEIEED
#undef __SEIEE
