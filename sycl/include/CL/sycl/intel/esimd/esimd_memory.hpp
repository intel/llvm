//==-------------- esimd_memory.hpp - DPC++ Explicit SIMD API --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Implement Explicit SIMD memory-access APIs.
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/half_type.hpp>
#include <CL/sycl/intel/esimd/detail/esimd_memory_intrin.hpp>
#include <CL/sycl/intel/esimd/detail/esimd_types.hpp>
#include <CL/sycl/intel/esimd/detail/esimd_util.hpp>
#include <CL/sycl/intel/esimd/esimd.hpp>
#include <CL/sycl/intel/esimd/esimd_enum.hpp>
#include <cstdint>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace intel {
namespace gpu {

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

// TODO @Pennycook
// {quote}
// ...I'd like us to think more about what we can do to make these interfaces
// more user - friendly. A user providing cache hints has to provide a lot more
// template arguments than required.Could we make this nicer by providing the
// hints as tag - type arguments ?
// ...
//   // Without cache hints, type and length can be deduced from offsets
//   float* p;
//   simd<uint32_t, 16> offsets;
//   auto result = flat_load(p, offsets);
//
//   // With cache hints as templates, verbosity increases significantly:
//   // - Providing any cache hint forces the user to specify the type and
//   length float* p; simd<uint32_t, 16> offsets; auto result =
//   flat_load<uint32_t, 16, 1, CacheHint::Foo, CacheHint::Bar>(p, offsets);
//
//   // With cache hints as tag types, verbosity is reduced:
//   // - Providing a cache hint does not prevent deduction of type and length
//   float* p;
//   simd <uint32_t, 16> offsets;
//   auto result = flat_load(p, offsets, CacheHint::Foo{});
//
// Note also that the templated form prevents a developer from specifying an L3
// hint without also explicitly specifying an L1 hint. If flat_load accepted a
// list of hints, it might be possible to refactor the hints to specify them in
// any order, and it may be more extensible to future cache hints:
// {/quote}
//
// TODO @keryell
// {quote}
// An approach � la https ://github.com/chriskohlhoff/propria from
// @chriskohlhoff would be to add a property to the pointer, such as
//
//    auto result = flat_load(p, offsets);
//    auto result = flat_load(decorate<CacheHint::Foo, CacheHint::Bar>(p),
//    offsets);
// The advantage is that you do not have to change all tour API and all the uses
// of this decorated pointer will benefit from this. decorate is to be bikeshed
// accordingly.
// {/quote}
//
/// flat-address gather
template <typename T, int n, int ElemsPerAddr = 1,
          CacheHint L1H = CacheHint::None, CacheHint L3H = CacheHint::None>
ESIMD_INLINE ESIMD_NODEBUG
    typename std::enable_if<((n == 8 || n == 16 || n == 32) &&
                             (ElemsPerAddr == 1 || ElemsPerAddr == 2 ||
                              ElemsPerAddr == 4)),
                            simd<T, n * ElemsPerAddr>>::type
    gather(T *p, simd<uint32_t, n> offsets, simd<uint16_t, n> pred = 1) {

  simd<uint64_t, n> offsets_i = convert<uint64_t>(offsets);
  simd<uint64_t, n> addrs(reinterpret_cast<uint64_t>(p));
  addrs = addrs + offsets_i;

  if constexpr (sizeof(T) == 1 && ElemsPerAddr == 2) {
    auto Ret = __esimd_flat_read<T, n, ElemsPerAddrEncoding<4>(), L1H, L3H>(
        addrs.data(), ElemsPerAddrEncoding<ElemsPerAddr>(), pred.data());
    return __esimd_rdregion<T, n * 4, n * ElemsPerAddr, /*VS*/ 4, 2, 1>(Ret, 0);
  } else if constexpr (sizeof(T) == 1 && ElemsPerAddr == 1) {
    auto Ret = __esimd_flat_read<T, n, ElemsPerAddrEncoding<4>(), L1H, L3H>(
        addrs.data(), ElemsPerAddrEncoding<ElemsPerAddr>(), pred.data());
    return __esimd_rdregion<T, n * 4, n * ElemsPerAddr, /*VS*/ 0, n, 4>(Ret, 0);
  } else if constexpr (sizeof(T) == 2 && ElemsPerAddr == 1) {
    auto Ret = __esimd_flat_read<T, n, ElemsPerAddrEncoding<2>(), L1H, L3H>(
        addrs.data(), ElemsPerAddrEncoding<2>(), pred.data());
    return __esimd_rdregion<T, n * 2, n, /*VS*/ 0, n, 2>(Ret, 0);
  } else if constexpr (sizeof(T) == 2)
    return __esimd_flat_read<T, n, ElemsPerAddrEncoding<ElemsPerAddr>(), L1H,
                             L3H>(
        addrs.data(), ElemsPerAddrEncoding<2 * ElemsPerAddr>(), pred.data());
  else
    return __esimd_flat_read<T, n, ElemsPerAddrEncoding<ElemsPerAddr>(), L1H,
                             L3H>(
        addrs.data(), ElemsPerAddrEncoding<ElemsPerAddr>(), pred.data());
}

/// flat-address scatter
template <typename T, int n, int ElemsPerAddr = 1,
          CacheHint L1H = CacheHint::None, CacheHint L3H = CacheHint::None>
ESIMD_INLINE ESIMD_NODEBUG
    typename std::enable_if<((n == 8 || n == 16 || n == 32) &&
                             (ElemsPerAddr == 1 || ElemsPerAddr == 2 ||
                              ElemsPerAddr == 4)),
                            void>::type
    scatter(T *p, simd<T, n * ElemsPerAddr> vals, simd<uint32_t, n> offsets,
            simd<uint16_t, n> pred = 1) {
  simd<uint64_t, n> offsets_i = convert<uint64_t>(offsets);
  simd<uint64_t, n> addrs(reinterpret_cast<uint64_t>(p));
  addrs = addrs + offsets_i;
  if constexpr (sizeof(T) == 1 && ElemsPerAddr == 2) {
    simd<T, n * 4> D;
    D = __esimd_wrregion<T, n * 4, n * ElemsPerAddr, /*VS*/ 4, 2, 1>(
        D.data(), vals.data(), 0);
    __esimd_flat_write<T, n, ElemsPerAddrEncoding<4>(), L1H, L3H>(
        addrs.data(), D.data(), ElemsPerAddrEncoding<ElemsPerAddr>(),
        pred.data());
  } else if constexpr (sizeof(T) == 1 && ElemsPerAddr == 1) {
    simd<T, n * 4> D;
    D = __esimd_wrregion<T, n * 4, n * ElemsPerAddr, /*VS*/ 0, n, 4>(
        D.data(), vals.data(), 0);
    __esimd_flat_write<T, n, ElemsPerAddrEncoding<4>(), L1H, L3H>(
        addrs.data(), D.data(), ElemsPerAddrEncoding<ElemsPerAddr>(),
        pred.data());
  } else if constexpr (sizeof(T) == 2 && ElemsPerAddr == 1) {
    simd<T, n * 2> D;
    D = __esimd_wrregion<T, n * 2, n, /*VS*/ 0, n, 2>(D.data(), vals.data(), 0);
    __esimd_flat_write<T, n, ElemsPerAddrEncoding<2>(), L1H, L3H>(
        addrs.data(), D.data(), ElemsPerAddrEncoding<2>(), pred.data());
  } else if constexpr (sizeof(T) == 2)
    __esimd_flat_write<T, n, ElemsPerAddrEncoding<ElemsPerAddr>(), L1H, L3H>(
        addrs.data(), vals.data(), ElemsPerAddrEncoding<2 * ElemsPerAddr>(),
        pred.data());
  else
    __esimd_flat_write<T, n, ElemsPerAddrEncoding<ElemsPerAddr>(), L1H, L3H>(
        addrs.data(), vals.data(), ElemsPerAddrEncoding<ElemsPerAddr>(),
        pred.data());
}

// TODO @rolandschulz
// Should follow existing std::simd naming for similar APIs - "copy_from" and
// "copy_to" to avoid confusion.
//
/// flat-address block-load
template <typename T, int n, CacheHint L1H = CacheHint::None,
          CacheHint L3H = CacheHint::None>
ESIMD_INLINE ESIMD_NODEBUG simd<T, n> block_load(const T *const addr) {
  constexpr unsigned Sz = sizeof(T) * n;
  static_assert(Sz >= __esimd::OWORD, "block size must be at least 1 oword");
  static_assert(Sz % __esimd::OWORD == 0,
                "block size must be whole number of owords");
  static_assert(__esimd::isPowerOf2(Sz / __esimd::OWORD),
                "block must be 1, 2, 4 or 8 owords long");
  static_assert(Sz <= 8 * __esimd::OWORD,
                "block size must be at most 8 owords");

  uintptr_t Addr = reinterpret_cast<uintptr_t>(addr);
  return __esimd_flat_block_read_unaligned<T, n, L1H, L3H>(Addr);
}

/// accessor-based block-load
template <typename T, int n, typename AccessorTy>
ESIMD_INLINE ESIMD_NODEBUG simd<T, n> block_load(AccessorTy acc,
                                                 uint32_t offset) {
  constexpr unsigned Sz = sizeof(T) * n;
  static_assert(Sz >= __esimd::OWORD, "block size must be at least 1 oword");
  static_assert(Sz % __esimd::OWORD == 0,
                "block size must be whole number of owords");
  static_assert(__esimd::isPowerOf2(Sz / __esimd::OWORD),
                "block must be 1, 2, 4 or 8 owords long");
  static_assert(Sz <= 8 * __esimd::OWORD,
                "block size must be at most 8 owords");

#if defined(__SYCL_DEVICE_ONLY__) && defined(__SYCL_EXPLICIT_SIMD__)
  auto surf_ind = AccessorPrivateProxy::getNativeImageObj(acc);
  return __esimd_block_read<T, n>(surf_ind, offset);
#else
  return __esimd_block_read<T, n>(acc, offset);
#endif // __SYCL_DEVICE_ONLY__ && __SYCL_EXPLICIT_SIMD__
}

/// flat-address block-store
template <typename T, int n, CacheHint L1H = CacheHint::None,
          CacheHint L3H = CacheHint::None>
ESIMD_INLINE ESIMD_NODEBUG void block_store(T *p, simd<T, n> vals) {
  constexpr unsigned Sz = sizeof(T) * n;
  static_assert(Sz >= __esimd::OWORD, "block size must be at least 1 oword");
  static_assert(Sz % __esimd::OWORD == 0,
                "block size must be whole number of owords");
  static_assert(__esimd::isPowerOf2(Sz / __esimd::OWORD),
                "block must be 1, 2, 4 or 8 owords long");
  static_assert(Sz <= 8 * __esimd::OWORD,
                "block size must be at most 8 owords");

  uintptr_t Addr = reinterpret_cast<uintptr_t>(p);
  __esimd_flat_block_write<T, n, L1H, L3H>(Addr, vals.data());
}

/// accessor-based block-store
template <typename T, int n, typename AccessorTy>
ESIMD_INLINE ESIMD_NODEBUG void block_store(AccessorTy acc, uint32_t offset,
                                            simd<T, n> vals) {
  constexpr unsigned Sz = sizeof(T) * n;
  static_assert(Sz >= __esimd::OWORD, "block size must be at least 1 oword");
  static_assert(Sz % __esimd::OWORD == 0,
                "block size must be whole number of owords");
  static_assert(__esimd::isPowerOf2(Sz / __esimd::OWORD),
                "block must be 1, 2, 4 or 8 owords long");
  static_assert(Sz <= 8 * __esimd::OWORD,
                "block size must be at most 8 owords");

#if defined(__SYCL_DEVICE_ONLY__) && defined(__SYCL_EXPLICIT_SIMD__)
  auto surf_ind = AccessorPrivateProxy::getNativeImageObj(acc);
  __esimd_block_write<T, n>(surf_ind, offset >> 4, vals.data());
#else
  __esimd_block_write<T, n>(acc, offset >> 4, vals.data());
#endif // __SYCL_DEVICE_ONLY__ && __SYCL_EXPLICIT_SIMD__
}

// TODO @jasonsewall-intel
// Don't use '4' in the name - instead either make it a parameter or
// (if it must be constant) - try to deduce from other arguments.
//
/// flat-address gather4
/// only allow simd-16 and simd-32
template <typename T, int n, ChannelMaskType Mask,
          CacheHint L1H = CacheHint::None, CacheHint L3H = CacheHint::None>
ESIMD_INLINE ESIMD_NODEBUG
    typename std::enable_if<(n == 16 || n == 32) && (sizeof(T) == 4),
                            simd<T, n * NumChannels(Mask)>>::type
    gather4(T *p, simd<uint32_t, n> offsets, simd<uint16_t, n> pred = 1) {

  simd<uint64_t, n> offsets_i = convert<uint64_t>(offsets);
  simd<uint64_t, n> addrs(reinterpret_cast<uint64_t>(p));
  addrs = addrs + offsets_i;
  return __esimd_flat_read4<T, n, Mask, L1H, L3H>(addrs.data(), pred.data());
}

/// flat-address scatter4
template <typename T, int n, ChannelMaskType Mask,
          CacheHint L1H = CacheHint::None, CacheHint L3H = CacheHint::None>
ESIMD_INLINE ESIMD_NODEBUG
    typename std::enable_if<(n == 16 || n == 32) && (sizeof(T) == 4),
                            void>::type
    scatter4(T *p, simd<T, n * NumChannels(Mask)> vals,
             simd<uint32_t, n> offsets, simd<uint16_t, n> pred = 1) {
  simd<uint64_t, n> offsets_i = convert<uint64_t>(offsets);
  simd<uint64_t, n> addrs(reinterpret_cast<uint64_t>(p));
  addrs = addrs + offsets_i;
  __esimd_flat_write4<T, n, Mask, L1H, L3H>(addrs.data(), vals.data(),
                                            pred.data());
}

/// check the legality of an atomic call in terms of size and type
template <EsimdAtomicOpType Op, typename T, int N, unsigned NumSrc>
constexpr bool check_atomic() {
  if constexpr (!__esimd::isPowerOf2(N, 32)) {
    static_assert((__esimd::isPowerOf2(N, 32)),
                  "Execution size 1, 2, 4, 8, 16, 32 are supported");
    return false;
  }

  // No source operand.
  if constexpr (Op == EsimdAtomicOpType::ATOMIC_INC ||
                Op == EsimdAtomicOpType::ATOMIC_DEC) {
    if constexpr (NumSrc != 0) {
      static_assert(NumSrc == 0, "No source operands are expected");
      return false;
    }
    if constexpr (!is_type<T, uint16_t, uint32_t, uint64_t>()) {
      static_assert((is_type<T, uint16_t, uint32_t, uint64_t>()),
                    "Type UW, UD or UQ is expected");
      return false;
    }
    return true;
  }

  // One source integer operand.
  if constexpr (Op == EsimdAtomicOpType::ATOMIC_ADD ||
                Op == EsimdAtomicOpType::ATOMIC_SUB ||
                Op == EsimdAtomicOpType::ATOMIC_MIN ||
                Op == EsimdAtomicOpType::ATOMIC_MAX ||
                Op == EsimdAtomicOpType::ATOMIC_XCHG ||
                Op == EsimdAtomicOpType::ATOMIC_AND ||
                Op == EsimdAtomicOpType::ATOMIC_OR ||
                Op == EsimdAtomicOpType::ATOMIC_XOR ||
                Op == EsimdAtomicOpType::ATOMIC_MINSINT ||
                Op == EsimdAtomicOpType::ATOMIC_MAXSINT) {
    if constexpr (NumSrc != 1) {
      static_assert(NumSrc == 1, "One source operand is expected");
      return false;
    }
    if constexpr ((Op != EsimdAtomicOpType::ATOMIC_MINSINT &&
                   Op != EsimdAtomicOpType::ATOMIC_MAXSINT) &&
                  !is_type<T, uint16_t, uint32_t, uint64_t>()) {
      static_assert((is_type<T, uint16_t, uint32_t, uint64_t>()),
                    "Type UW, UD or UQ is expected");
      return false;
    }
    if constexpr ((Op == EsimdAtomicOpType::ATOMIC_MINSINT ||
                   Op == EsimdAtomicOpType::ATOMIC_MAXSINT) &&
                  !is_type<T, int16_t, int32_t, int64_t>()) {
      static_assert((is_type<T, int16_t, int32_t, int64_t>()),
                    "Type W, D or Q is expected");
      return false;
    }
    return true;
  }

  // One source float operand.
  if constexpr (Op == EsimdAtomicOpType::ATOMIC_FMAX ||
                Op == EsimdAtomicOpType::ATOMIC_FMIN) {
    if constexpr (NumSrc != 1) {
      static_assert(NumSrc == 1, "One source operand is expected");
      return false;
    }
    if constexpr (!is_type<T, float, cl::sycl::detail::half_impl::StorageT>()) {
      static_assert(
          (is_type<T, float, cl::sycl::detail::half_impl::StorageT>()),
          "Type F or HF is expected");
      return false;
    }
    return true;
  }

  // Two scouce operands.
  if constexpr (Op == EsimdAtomicOpType::ATOMIC_CMPXCHG ||
                Op == EsimdAtomicOpType::ATOMIC_FCMPWR) {
    if constexpr (NumSrc != 2) {
      static_assert(NumSrc == 2, "Two source operands are expected");
      return false;
    }
    if constexpr (Op == EsimdAtomicOpType::ATOMIC_CMPXCHG &&
                  !is_type<T, uint16_t, uint32_t, uint64_t>()) {
      static_assert((is_type<T, uint16_t, uint32_t, uint64_t>()),
                    "Type UW, UD or UQ is expected");
      return false;
    }
    if constexpr (Op == EsimdAtomicOpType::ATOMIC_FCMPWR &&
                  !is_type<T, float, cl::sycl::detail::half_impl::StorageT>()) {
      static_assert(
          (is_type<T, float, cl::sycl::detail::half_impl::StorageT>()),
          "Type F or HF is expected");
      return false;
    }
    return true;
  }
  // Unsupported svm atomic Op.
  return false;
}

// TODO @Pennycook
// {quote}
// We should look into what can be done to simplify these atomic functions and
// align their design with the other new atomic features.That is perhaps out of
// scope for this PR(the direction is less clear than for the reduce changes,
// for example) but we should open an issue to track it.
// {/quote}

/// flat-address atomic, zero source operand: inc and dec
template <EsimdAtomicOpType Op, typename T, int n,
          CacheHint L1H = CacheHint::None, CacheHint L3H = CacheHint::None>
ESIMD_NODEBUG ESIMD_INLINE
    typename std::enable_if<check_atomic<Op, T, n, 0>(), simd<T, n>>::type
    flat_atomic(T *p, simd<unsigned, n> offset, simd<ushort, n> pred) {
  simd<uintptr_t, n> vAddr(reinterpret_cast<uintptr_t>(p));
  simd<uintptr_t, n> offset_i1 = convert<uintptr_t>(offset);
  vAddr += offset_i1;
  return __esimd_flat_atomic0<Op, T, n, L1H, L3H>(vAddr.data(), pred.data());
}

/// flat-address atomic, one source operand, add/sub/min/max etc
template <EsimdAtomicOpType Op, typename T, int n,
          CacheHint L1H = CacheHint::None, CacheHint L3H = CacheHint::None>
ESIMD_NODEBUG ESIMD_INLINE
    typename std::enable_if<check_atomic<Op, T, n, 1>(), simd<T, n>>::type
    flat_atomic(T *p, simd<unsigned, n> offset, simd<T, n> src0,
                simd<ushort, n> pred) {
  simd<uintptr_t, n> vAddr(reinterpret_cast<uintptr_t>(p));
  simd<uintptr_t, n> offset_i1 = convert<uintptr_t>(offset);
  vAddr += offset_i1;
  return __esimd_flat_atomic1<Op, T, n, L1H, L3H>(vAddr.data(), src0.data(),
                                                  pred.data());
}

/// flat-address atomic, two source operands
template <EsimdAtomicOpType Op, typename T, int n,
          CacheHint L1H = CacheHint::None, CacheHint L3H = CacheHint::None>
ESIMD_NODEBUG ESIMD_INLINE
    typename std::enable_if<check_atomic<Op, T, n, 2>(), simd<T, n>>::type
    flat_atomic(T *p, simd<unsigned, n> offset, simd<T, n> src0,
                simd<T, n> src1, simd<ushort, n> pred) {
  simd<uintptr_t, n> vAddr(reinterpret_cast<uintptr_t>(p));
  simd<uintptr_t, n> offset_i1 = convert<uintptr_t>(offset);
  vAddr += offset_i1;
  return __esimd_flat_atomic2<Op, T, n, L1H, L3H>(vAddr.data(), src0.data(),
                                                  src1.data(), pred.data());
}

/// generic work-group barrier
inline ESIMD_NODEBUG void esimd_barrier() { __esimd_barrier(); }

/// SLM functions

/// declare per-work-group slm size
SYCL_EXTERNAL void slm_init(uint32_t size);

enum EsimdFenceMask {
  ESIMD_GLOBAL_COHERENT_FENCE = 0x1,
  ESIMD_L3_FLUSH_INSTRUCTIONS = 0x2,
  ESIMD_L3_FLUSH_TEXTURE_DATA = 0x4,
  ESIMD_L3_FLUSH_CONSTANT_DATA = 0x8,
  ESIMD_L3_FLUSH_RW_DATA = 0x10,
  ESIMD_LOCAL_BARRIER = 0x20,
  ESIMD_L1_FLUASH_RO_DATA = 0x40,
  ESIMD_SW_BARRIER = 0x80
};

/// slm_fence sets the SLM read/write order
inline ESIMD_NODEBUG void slm_fence(uint8_t cntl) { __esimd_slm_fence(cntl); }

/// SLM gather
/// only allow simd-16 and simd-32
template <typename T, int n>
ESIMD_INLINE ESIMD_NODEBUG
    typename std::enable_if<(n == 16 || n == 32), simd<T, n>>::type
    slm_load(simd<uint32_t, n> offsets, simd<uint16_t, n> pred = 1) {
  return __esimd_slm_read<T, n>(offsets.data(), pred.data());
}

/// SLM scatter
template <typename T, int n>
ESIMD_INLINE ESIMD_NODEBUG
    typename std::enable_if<(n == 16 || n == 32), void>::type
    slm_store(simd<T, n> vals, simd<uint32_t, n> offsets,
              simd<uint16_t, n> pred = 1) {
  __esimd_slm_write<T, n>(offsets.data(), vals.data(), pred.data());
}

/// SLM gather4
/// only allow simd-16 and simd-32
template <typename T, int n, ChannelMaskType Mask>
ESIMD_INLINE ESIMD_NODEBUG
    typename std::enable_if<(n == 16 || n == 32) && (sizeof(T) == 4),
                            simd<T, n * NumChannels(Mask)>>::type
    slm_load4(simd<uint32_t, n> offsets, simd<uint16_t, n> pred = 1) {
  return __esimd_slm_read4<T, n, Mask>(offsets.data(), pred.data());
}

/// SLM scatter4
template <typename T, int n, ChannelMaskType Mask>
typename std::enable_if<(n == 16 || n == 32) && (sizeof(T) == 4), void>::type
slm_store4(simd<T, n * NumChannels(Mask)> vals, simd<uint32_t, n> offsets,
           simd<uint16_t, n> pred = 1) {
  __esimd_slm_write4<T, n, Mask>(offsets.data(), vals.data(), pred.data());
}

/// SLM block-load
template <typename T, int n>
ESIMD_INLINE ESIMD_NODEBUG simd<T, n> slm_block_load(uint32_t offset) {
  constexpr unsigned Sz = sizeof(T) * n;
  static_assert(Sz >= __esimd::OWORD, "block size must be at least 1 oword");
  static_assert(Sz % __esimd::OWORD == 0,
                "block size must be whole number of owords");
  static_assert(__esimd::isPowerOf2(Sz / __esimd::OWORD),
                "block must be 1, 2, 4 or 8 owords long");
  static_assert(Sz <= 8 * __esimd::OWORD,
                "block size must be at most 8 owords");

  return __esimd_slm_block_read<T, n>(offset);
}

/// SLM block-store
template <typename T, int n>
ESIMD_INLINE ESIMD_NODEBUG void slm_block_store(uint32_t offset,
                                                simd<T, n> vals) {
  constexpr unsigned Sz = sizeof(T) * n;
  static_assert(Sz >= __esimd::OWORD, "block size must be at least 1 oword");
  static_assert(Sz % __esimd::OWORD == 0,
                "block size must be whole number of owords");
  static_assert(__esimd::isPowerOf2(Sz / __esimd::OWORD),
                "block must be 1, 2, 4 or 8 owords long");
  static_assert(Sz <= 8 * __esimd::OWORD,
                "block size must be at most 8 owords");

  // offset in genx.oword.st is in owords
  __esimd_slm_block_write<T, n>(offset >> 4, vals.data());
}

/// SLM atomic, zero source operand: inc and dec
template <EsimdAtomicOpType Op, typename T, int n>
ESIMD_NODEBUG ESIMD_INLINE
    typename std::enable_if<check_atomic<Op, T, n, 0>(), simd<T, n>>::type
    slm_atomic(simd<uint32_t, n> offsets, simd<ushort, n> pred) {
  return __esimd_slm_atomic0<Op, T, n>(offsets.data(), pred.data());
}

/// SLM atomic, one source operand, add/sub/min/max etc
template <EsimdAtomicOpType Op, typename T, int n>
ESIMD_NODEBUG ESIMD_INLINE
    typename std::enable_if<check_atomic<Op, T, n, 1>(), simd<T, n>>::type
    slm_atomic(simd<uint32_t, n> offsets, simd<T, n> src0,
               simd<ushort, n> pred) {
  return __esimd_slm_atomic1<Op, T, n>(offsets.data(), src0.data(),
                                       pred.data());
}

/// SLM atomic, two source operands
template <EsimdAtomicOpType Op, typename T, int n>
ESIMD_NODEBUG ESIMD_INLINE
    typename std::enable_if<check_atomic<Op, T, n, 2>(), simd<T, n>>::type
    slm_atomic(simd<uint32_t, n> offsets, simd<T, n> src0, simd<T, n> src1,
               simd<ushort, n> pred) {
  return __esimd_slm_atomic2<Op, T, n>(offsets.data(), src0.data(), src1.data(),
                                       pred.data());
}

// Media block load
//
// @param T the element data type.
//
// @param m the hight of the 2D block.
//
// @param n the width of the 2D block.
//
// @param AccessorTy type of the SYCL accessor.
//
// @param plane planar surface index.
//
// @param acc the SYCL accessor.
//
// @param x X-coordinate of the left upper rectangle corner in BYTES.
//
// @param y Y-coordinate of the left upper rectangle corner in ROWS.
//
// @return the linearized 2D block data read from surface.
//
template <typename T, int m, int n, typename AccessorTy, unsigned plane = 0>
ESIMD_INLINE ESIMD_NODEBUG simd<T, m * n>
media_block_load(AccessorTy acc, unsigned x, unsigned y) {
  constexpr unsigned Width = n * sizeof(T);
  static_assert(Width * m <= 256u,
                "data does not fit into a single dataport transaction");
  static_assert(Width <= 64u, "valid block width is in range [1, 64]");
  static_assert(m <= 64u, "valid block height is in range [1, 64]");
  static_assert(plane <= 3u, "valid plane index is in range [0, 3]");
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SYCL_EXPLICIT_SIMD__)
  constexpr unsigned int RoundedWidth =
      Width < 4 ? 4 : __esimd::getNextPowerOf2(Width);

  if constexpr (Width < RoundedWidth) {
    constexpr unsigned int n1 = RoundedWidth / sizeof(T);
    simd<T, m *n1> temp = __esimd_media_block_load<T, m, n1>(
        0, AccessorPrivateProxy::getNativeImageObj(acc), plane, sizeof(T) * n,
        x, y);
    return temp.template select<m, 1, n, 1>(0, 0);
  } else {
    return __esimd_media_block_load<T, m, n>(
        0, AccessorPrivateProxy::getNativeImageObj(acc), plane, sizeof(T) * n,
        x, y);
  }
#else
  return __esimd_media_block_load<T, m, n>(0, acc, plane, sizeof(T) * n, x, y);
#endif // __SYCL_DEVICE_ONLY__ && __SYCL_EXPLICIT_SIMD__
}

// Media block store
//
// @param T the element data type.
//
// @param m the hight of the 2D block.
//
// @param n the width of the 2D block.
//
// @param AccessorTy type of the SYCL accessor.
//
// @param plane planar surface index.
//
// @param acc the SYCL accessor.
//
// @param x X-coordinate of the left upper rectangle corner in BYTES.
//
// @param y Y-coordinate of the left upper rectangle corner in ROWS.
//
// @param vals the linearized 2D block data to be written to surface.
//
/// Media block store
template <typename T, int m, int n, typename AccessorTy, unsigned plane = 0>
ESIMD_INLINE ESIMD_NODEBUG void
media_block_store(AccessorTy acc, unsigned x, unsigned y, simd<T, m * n> vals) {
  constexpr unsigned Width = n * sizeof(T);
  static_assert(Width * m <= 256u,
                "data does not fit into a single dataport transaction");
  static_assert(Width <= 64u, "valid block width is in range [1, 64]");
  static_assert(m <= 64u, "valid block height is in range [1, 64]");
  static_assert(plane <= 3u, "valid plane index is in range [0, 3]");
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SYCL_EXPLICIT_SIMD__)
  constexpr unsigned int RoundedWidth =
      Width < 4 ? 4 : __esimd::getNextPowerOf2(Width);
  constexpr unsigned int n1 = RoundedWidth / sizeof(T);

  if constexpr (Width < RoundedWidth) {
    simd<T, m * n1> temp;
    auto temp_ref = temp.template format<T, m, n1>();
    auto vals_ref = vals.template format<T, m, n>();
    temp_ref.template select<m, 1, n, 1>() = vals_ref;
    __esimd_media_block_store<T, m, n1>(
        0, AccessorPrivateProxy::getNativeImageObj(acc), plane, sizeof(T) * n,
        x, y, temp);
  } else {
    __esimd_media_block_store<T, m, n>(
        0, AccessorPrivateProxy::getNativeImageObj(acc), plane, sizeof(T) * n,
        x, y, vals);
  }
#else
  __esimd_media_block_store<T, m, n>(0, acc, plane, sizeof(T) * n, x, y, vals);
#endif // __SYCL_DEVICE_ONLY__ && __SYCL_EXPLICIT_SIMD__
}

#ifndef __SYCL_DEVICE_ONLY__

SYCL_EXTERNAL void slm_init(uint32_t size) {}

#endif
} // namespace gpu
} // namespace intel
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
