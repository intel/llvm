//===-- spirv.hpp - Helpers to generate SPIR-V instructions ----*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <CL/__spirv/spirv_ops.hpp>
#include <CL/__spirv/spirv_types.hpp>
#include <CL/__spirv/spirv_vars.hpp>
#include <CL/sycl/ONEAPI/atomic_enums.hpp>
#include <CL/sycl/detail/generic_type_traits.hpp>
#include <CL/sycl/detail/helpers.hpp>
#include <CL/sycl/detail/type_traits.hpp>
#include <CL/sycl/id.hpp>

#ifdef __SYCL_DEVICE_ONLY__
__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ONEAPI {
struct sub_group;
} // namespace ONEAPI
namespace detail {
namespace spirv {

template <typename Group> struct group_scope {};

template <int Dimensions> struct group_scope<group<Dimensions>> {
  static constexpr __spv::Scope::Flag value = __spv::Scope::Flag::Workgroup;
};

template <> struct group_scope<::cl::sycl::ONEAPI::sub_group> {
  static constexpr __spv::Scope::Flag value = __spv::Scope::Flag::Subgroup;
};

// Generic shuffles and broadcasts may require multiple calls to
// intrinsics, and should use the fewest broadcasts possible
// - Loop over chunks until remaining bytes < chunk size
// - At most one 32-bit, 16-bit and 8-bit chunk left over
#ifndef __NVPTX__
using ShuffleChunkT = uint64_t;
#else
using ShuffleChunkT = uint32_t;
#endif
template <typename T, typename Functor>
void GenericCall(const Functor &ApplyToBytes) {
  if (sizeof(T) >= sizeof(ShuffleChunkT)) {
#pragma unroll
    for (size_t Offset = 0; Offset < sizeof(T);
         Offset += sizeof(ShuffleChunkT)) {
      ApplyToBytes(Offset, sizeof(ShuffleChunkT));
    }
  }
  if (sizeof(ShuffleChunkT) >= sizeof(uint64_t)) {
    if (sizeof(T) % sizeof(uint64_t) >= sizeof(uint32_t)) {
      size_t Offset = sizeof(T) / sizeof(uint64_t) * sizeof(uint64_t);
      ApplyToBytes(Offset, sizeof(uint32_t));
    }
  }
  if (sizeof(ShuffleChunkT) >= sizeof(uint32_t)) {
    if (sizeof(T) % sizeof(uint32_t) >= sizeof(uint16_t)) {
      size_t Offset = sizeof(T) / sizeof(uint32_t) * sizeof(uint32_t);
      ApplyToBytes(Offset, sizeof(uint16_t));
    }
  }
  if (sizeof(ShuffleChunkT) >= sizeof(uint16_t)) {
    if (sizeof(T) % sizeof(uint16_t) >= sizeof(uint8_t)) {
      size_t Offset = sizeof(T) / sizeof(uint16_t) * sizeof(uint16_t);
      ApplyToBytes(Offset, sizeof(uint8_t));
    }
  }
}

template <typename Group> bool GroupAll(bool pred) {
  return __spirv_GroupAll(group_scope<Group>::value, pred);
}

template <typename Group> bool GroupAny(bool pred) {
  return __spirv_GroupAny(group_scope<Group>::value, pred);
}

// Native broadcasts map directly to a SPIR-V GroupBroadcast intrinsic
template <typename T>
using is_native_broadcast = bool_constant<detail::is_arithmetic<T>::value>;

template <typename T, typename IdT = size_t>
using EnableIfNativeBroadcast = detail::enable_if_t<
    is_native_broadcast<T>::value && std::is_integral<IdT>::value, T>;

// Bitcast broadcasts can be implemented using a single SPIR-V GroupBroadcast
// intrinsic, but require type-punning via an appropriate integer type
template <typename T>
using is_bitcast_broadcast = bool_constant<
    !is_native_broadcast<T>::value && std::is_trivially_copyable<T>::value &&
    (sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8)>;

template <typename T, typename IdT = size_t>
using EnableIfBitcastBroadcast = detail::enable_if_t<
    is_bitcast_broadcast<T>::value && std::is_integral<IdT>::value, T>;

template <typename T>
using ConvertToNativeBroadcastType_t = select_cl_scalar_integral_unsigned_t<T>;

// Generic broadcasts may require multiple calls to SPIR-V GroupBroadcast
// intrinsics, and should use the fewest broadcasts possible
// - Loop over 64-bit chunks until remaining bytes < 64-bit
// - At most one 32-bit, 16-bit and 8-bit chunk left over
template <typename T>
using is_generic_broadcast =
    bool_constant<!is_native_broadcast<T>::value &&
                  !is_bitcast_broadcast<T>::value &&
                  std::is_trivially_copyable<T>::value>;

template <typename T, typename IdT = size_t>
using EnableIfGenericBroadcast = detail::enable_if_t<
    is_generic_broadcast<T>::value && std::is_integral<IdT>::value, T>;

// Broadcast with scalar local index
// Work-group supports any integral type
// Sub-group currently supports only uint32_t
template <typename Group> struct GroupId { using type = size_t; };
template <> struct GroupId<::cl::sycl::ONEAPI::sub_group> {
  using type = uint32_t;
};
template <typename Group, typename T, typename IdT>
EnableIfNativeBroadcast<T, IdT> GroupBroadcast(T x, IdT local_id) {
  using GroupIdT = typename GroupId<Group>::type;
  GroupIdT GroupLocalId = static_cast<GroupIdT>(local_id);
  using OCLT = detail::ConvertToOpenCLType_t<T>;
  using OCLIdT = detail::ConvertToOpenCLType_t<GroupIdT>;
  OCLT OCLX = detail::convertDataToType<T, OCLT>(x);
  OCLIdT OCLId = detail::convertDataToType<GroupIdT, OCLIdT>(GroupLocalId);
  return __spirv_GroupBroadcast(group_scope<Group>::value, OCLX, OCLId);
}
template <typename Group, typename T, typename IdT>
EnableIfBitcastBroadcast<T, IdT> GroupBroadcast(T x, IdT local_id) {
  using GroupIdT = typename GroupId<Group>::type;
  GroupIdT GroupLocalId = static_cast<GroupIdT>(local_id);
  using BroadcastT = ConvertToNativeBroadcastType_t<T>;
  using OCLIdT = detail::ConvertToOpenCLType_t<GroupIdT>;
  auto BroadcastX = bit_cast<BroadcastT>(x);
  OCLIdT OCLId = detail::convertDataToType<GroupIdT, OCLIdT>(GroupLocalId);
  BroadcastT Result =
      __spirv_GroupBroadcast(group_scope<Group>::value, BroadcastX, OCLId);
  return bit_cast<T>(Result);
}
template <typename Group, typename T, typename IdT>
EnableIfGenericBroadcast<T, IdT> GroupBroadcast(T x, IdT local_id) {
  T Result;
  char *XBytes = reinterpret_cast<char *>(&x);
  char *ResultBytes = reinterpret_cast<char *>(&Result);
  auto BroadcastBytes = [=](size_t Offset, size_t Size) {
    uint64_t BroadcastX, BroadcastResult;
    detail::memcpy(&BroadcastX, XBytes + Offset, Size);
    BroadcastResult = GroupBroadcast<Group>(BroadcastX, local_id);
    detail::memcpy(ResultBytes + Offset, &BroadcastResult, Size);
  };
  GenericCall<T>(BroadcastBytes);
  return Result;
}

// Broadcast with vector local index
template <typename Group, typename T, int Dimensions>
EnableIfNativeBroadcast<T> GroupBroadcast(T x, id<Dimensions> local_id) {
  if (Dimensions == 1) {
    return GroupBroadcast<Group>(x, local_id[0]);
  }
  using IdT = vec<size_t, Dimensions>;
  using OCLT = detail::ConvertToOpenCLType_t<T>;
  using OCLIdT = detail::ConvertToOpenCLType_t<IdT>;
  IdT VecId;
  for (int i = 0; i < Dimensions; ++i) {
    VecId[i] = local_id[Dimensions - i - 1];
  }
  OCLT OCLX = detail::convertDataToType<T, OCLT>(x);
  OCLIdT OCLId = detail::convertDataToType<IdT, OCLIdT>(VecId);
  return __spirv_GroupBroadcast(group_scope<Group>::value, OCLX, OCLId);
}
template <typename Group, typename T, int Dimensions>
EnableIfBitcastBroadcast<T> GroupBroadcast(T x, id<Dimensions> local_id) {
  if (Dimensions == 1) {
    return GroupBroadcast<Group>(x, local_id[0]);
  }
  using IdT = vec<size_t, Dimensions>;
  using BroadcastT = ConvertToNativeBroadcastType_t<T>;
  using OCLIdT = detail::ConvertToOpenCLType_t<IdT>;
  IdT VecId;
  for (int i = 0; i < Dimensions; ++i) {
    VecId[i] = local_id[Dimensions - i - 1];
  }
  auto BroadcastX = bit_cast<BroadcastT>(x);
  OCLIdT OCLId = detail::convertDataToType<IdT, OCLIdT>(VecId);
  BroadcastT Result =
      __spirv_GroupBroadcast(group_scope<Group>::value, BroadcastX, OCLId);
  return bit_cast<T>(Result);
}
template <typename Group, typename T, int Dimensions>
EnableIfGenericBroadcast<T> GroupBroadcast(T x, id<Dimensions> local_id) {
  if (Dimensions == 1) {
    return GroupBroadcast<Group>(x, local_id[0]);
  }
  T Result;
  char *XBytes = reinterpret_cast<char *>(&x);
  char *ResultBytes = reinterpret_cast<char *>(&Result);
  auto BroadcastBytes = [=](size_t Offset, size_t Size) {
    uint64_t BroadcastX, BroadcastResult;
    detail::memcpy(&BroadcastX, XBytes + Offset, Size);
    BroadcastResult = GroupBroadcast<Group>(BroadcastX, local_id);
    detail::memcpy(ResultBytes + Offset, &BroadcastResult, Size);
  };
  GenericCall<T>(BroadcastBytes);
  return Result;
}

// Single happens-before means semantics should always apply to all spaces
// Although consume is unsupported, forwarding to acquire is valid
static inline constexpr __spv::MemorySemanticsMask::Flag
getMemorySemanticsMask(ONEAPI::memory_order Order) {
  __spv::MemorySemanticsMask::Flag SpvOrder = __spv::MemorySemanticsMask::None;
  switch (Order) {
  case ONEAPI::memory_order::relaxed:
    SpvOrder = __spv::MemorySemanticsMask::None;
    break;
  case ONEAPI::memory_order::__consume_unsupported:
  case ONEAPI::memory_order::acquire:
    SpvOrder = __spv::MemorySemanticsMask::Acquire;
    break;
  case ONEAPI::memory_order::release:
    SpvOrder = __spv::MemorySemanticsMask::Release;
    break;
  case ONEAPI::memory_order::acq_rel:
    SpvOrder = __spv::MemorySemanticsMask::AcquireRelease;
    break;
  case ONEAPI::memory_order::seq_cst:
    SpvOrder = __spv::MemorySemanticsMask::SequentiallyConsistent;
    break;
  }
  return static_cast<__spv::MemorySemanticsMask::Flag>(
      SpvOrder | __spv::MemorySemanticsMask::SubgroupMemory |
      __spv::MemorySemanticsMask::WorkgroupMemory |
      __spv::MemorySemanticsMask::CrossWorkgroupMemory);
}

static inline constexpr __spv::Scope::Flag
getScope(ONEAPI::memory_scope Scope) {
  switch (Scope) {
  case ONEAPI::memory_scope::work_item:
    return __spv::Scope::Invocation;
  case ONEAPI::memory_scope::sub_group:
    return __spv::Scope::Subgroup;
  case ONEAPI::memory_scope::work_group:
    return __spv::Scope::Workgroup;
  case ONEAPI::memory_scope::device:
    return __spv::Scope::Device;
  case ONEAPI::memory_scope::system:
    return __spv::Scope::CrossDevice;
  }
}

template <typename T, access::address_space AddressSpace>
inline typename detail::enable_if_t<std::is_integral<T>::value, T>
AtomicCompareExchange(multi_ptr<T, AddressSpace> MPtr,
                      ONEAPI::memory_scope Scope, ONEAPI::memory_order Success,
                      ONEAPI::memory_order Failure, T Desired, T Expected) {
  auto SPIRVSuccess = getMemorySemanticsMask(Success);
  auto SPIRVFailure = getMemorySemanticsMask(Failure);
  auto SPIRVScope = getScope(Scope);
  auto *Ptr = MPtr.get();
  return __spirv_AtomicCompareExchange(Ptr, SPIRVScope, SPIRVSuccess,
                                       SPIRVFailure, Desired, Expected);
}

template <typename T, access::address_space AddressSpace>
inline typename detail::enable_if_t<std::is_floating_point<T>::value, T>
AtomicCompareExchange(multi_ptr<T, AddressSpace> MPtr,
                      ONEAPI::memory_scope Scope, ONEAPI::memory_order Success,
                      ONEAPI::memory_order Failure, T Desired, T Expected) {
  using I = detail::make_unsinged_integer_t<T>;
  auto SPIRVSuccess = getMemorySemanticsMask(Success);
  auto SPIRVFailure = getMemorySemanticsMask(Failure);
  auto SPIRVScope = getScope(Scope);
  auto *PtrInt =
      reinterpret_cast<typename multi_ptr<I, AddressSpace>::pointer_t>(
          MPtr.get());
  I DesiredInt = bit_cast<I>(Desired);
  I ExpectedInt = bit_cast<I>(Expected);
  I ResultInt = __spirv_AtomicCompareExchange(
      PtrInt, SPIRVScope, SPIRVSuccess, SPIRVFailure, DesiredInt, ExpectedInt);
  return bit_cast<T>(ResultInt);
}

template <typename T, access::address_space AddressSpace>
inline typename detail::enable_if_t<std::is_integral<T>::value, T>
AtomicLoad(multi_ptr<T, AddressSpace> MPtr, ONEAPI::memory_scope Scope,
           ONEAPI::memory_order Order) {
  auto *Ptr = MPtr.get();
  auto SPIRVOrder = getMemorySemanticsMask(Order);
  auto SPIRVScope = getScope(Scope);
  return __spirv_AtomicLoad(Ptr, SPIRVScope, SPIRVOrder);
}

template <typename T, access::address_space AddressSpace>
inline typename detail::enable_if_t<std::is_floating_point<T>::value, T>
AtomicLoad(multi_ptr<T, AddressSpace> MPtr, ONEAPI::memory_scope Scope,
           ONEAPI::memory_order Order) {
  using I = detail::make_unsinged_integer_t<T>;
  auto *PtrInt =
      reinterpret_cast<typename multi_ptr<I, AddressSpace>::pointer_t>(
          MPtr.get());
  auto SPIRVOrder = getMemorySemanticsMask(Order);
  auto SPIRVScope = getScope(Scope);
  I ResultInt = __spirv_AtomicLoad(PtrInt, SPIRVScope, SPIRVOrder);
  return bit_cast<T>(ResultInt);
}

template <typename T, access::address_space AddressSpace>
inline typename detail::enable_if_t<std::is_integral<T>::value>
AtomicStore(multi_ptr<T, AddressSpace> MPtr, ONEAPI::memory_scope Scope,
            ONEAPI::memory_order Order, T Value) {
  auto *Ptr = MPtr.get();
  auto SPIRVOrder = getMemorySemanticsMask(Order);
  auto SPIRVScope = getScope(Scope);
  __spirv_AtomicStore(Ptr, SPIRVScope, SPIRVOrder, Value);
}

template <typename T, access::address_space AddressSpace>
inline typename detail::enable_if_t<std::is_floating_point<T>::value>
AtomicStore(multi_ptr<T, AddressSpace> MPtr, ONEAPI::memory_scope Scope,
            ONEAPI::memory_order Order, T Value) {
  using I = detail::make_unsinged_integer_t<T>;
  auto *PtrInt =
      reinterpret_cast<typename multi_ptr<I, AddressSpace>::pointer_t>(
          MPtr.get());
  auto SPIRVOrder = getMemorySemanticsMask(Order);
  auto SPIRVScope = getScope(Scope);
  I ValueInt = bit_cast<I>(Value);
  __spirv_AtomicStore(PtrInt, SPIRVScope, SPIRVOrder, ValueInt);
}

template <typename T, access::address_space AddressSpace>
inline typename detail::enable_if_t<std::is_integral<T>::value, T>
AtomicExchange(multi_ptr<T, AddressSpace> MPtr, ONEAPI::memory_scope Scope,
               ONEAPI::memory_order Order, T Value) {
  auto *Ptr = MPtr.get();
  auto SPIRVOrder = getMemorySemanticsMask(Order);
  auto SPIRVScope = getScope(Scope);
  return __spirv_AtomicExchange(Ptr, SPIRVScope, SPIRVOrder, Value);
}

template <typename T, access::address_space AddressSpace>
inline typename detail::enable_if_t<std::is_floating_point<T>::value, T>
AtomicExchange(multi_ptr<T, AddressSpace> MPtr, ONEAPI::memory_scope Scope,
               ONEAPI::memory_order Order, T Value) {
  using I = detail::make_unsinged_integer_t<T>;
  auto *PtrInt =
      reinterpret_cast<typename multi_ptr<I, AddressSpace>::pointer_t>(
          MPtr.get());
  auto SPIRVOrder = getMemorySemanticsMask(Order);
  auto SPIRVScope = getScope(Scope);
  I ValueInt = bit_cast<I>(Value);
  I ResultInt =
      __spirv_AtomicExchange(PtrInt, SPIRVScope, SPIRVOrder, ValueInt);
  return bit_cast<T>(ResultInt);
}

template <typename T, access::address_space AddressSpace>
inline typename detail::enable_if_t<std::is_integral<T>::value, T>
AtomicIAdd(multi_ptr<T, AddressSpace> MPtr, ONEAPI::memory_scope Scope,
           ONEAPI::memory_order Order, T Value) {
  auto *Ptr = MPtr.get();
  auto SPIRVOrder = getMemorySemanticsMask(Order);
  auto SPIRVScope = getScope(Scope);
  return __spirv_AtomicIAdd(Ptr, SPIRVScope, SPIRVOrder, Value);
}

template <typename T, access::address_space AddressSpace>
inline typename detail::enable_if_t<std::is_integral<T>::value, T>
AtomicISub(multi_ptr<T, AddressSpace> MPtr, ONEAPI::memory_scope Scope,
           ONEAPI::memory_order Order, T Value) {
  auto *Ptr = MPtr.get();
  auto SPIRVOrder = getMemorySemanticsMask(Order);
  auto SPIRVScope = getScope(Scope);
  return __spirv_AtomicISub(Ptr, SPIRVScope, SPIRVOrder, Value);
}

template <typename T, access::address_space AddressSpace>
inline typename detail::enable_if_t<std::is_floating_point<T>::value, T>
AtomicFAdd(multi_ptr<T, AddressSpace> MPtr, ONEAPI::memory_scope Scope,
           ONEAPI::memory_order Order, T Value) {
  auto *Ptr = MPtr.get();
  auto SPIRVOrder = getMemorySemanticsMask(Order);
  auto SPIRVScope = getScope(Scope);
  return __spirv_AtomicFAddEXT(Ptr, SPIRVScope, SPIRVOrder, Value);
}

template <typename T, access::address_space AddressSpace>
inline typename detail::enable_if_t<std::is_integral<T>::value, T>
AtomicAnd(multi_ptr<T, AddressSpace> MPtr, ONEAPI::memory_scope Scope,
          ONEAPI::memory_order Order, T Value) {
  auto *Ptr = MPtr.get();
  auto SPIRVOrder = getMemorySemanticsMask(Order);
  auto SPIRVScope = getScope(Scope);
  return __spirv_AtomicAnd(Ptr, SPIRVScope, SPIRVOrder, Value);
}

template <typename T, access::address_space AddressSpace>
inline typename detail::enable_if_t<std::is_integral<T>::value, T>
AtomicOr(multi_ptr<T, AddressSpace> MPtr, ONEAPI::memory_scope Scope,
         ONEAPI::memory_order Order, T Value) {
  auto *Ptr = MPtr.get();
  auto SPIRVOrder = getMemorySemanticsMask(Order);
  auto SPIRVScope = getScope(Scope);
  return __spirv_AtomicOr(Ptr, SPIRVScope, SPIRVOrder, Value);
}

template <typename T, access::address_space AddressSpace>
inline typename detail::enable_if_t<std::is_integral<T>::value, T>
AtomicXor(multi_ptr<T, AddressSpace> MPtr, ONEAPI::memory_scope Scope,
          ONEAPI::memory_order Order, T Value) {
  auto *Ptr = MPtr.get();
  auto SPIRVOrder = getMemorySemanticsMask(Order);
  auto SPIRVScope = getScope(Scope);
  return __spirv_AtomicXor(Ptr, SPIRVScope, SPIRVOrder, Value);
}

template <typename T, access::address_space AddressSpace>
inline typename detail::enable_if_t<std::is_integral<T>::value, T>
AtomicMin(multi_ptr<T, AddressSpace> MPtr, ONEAPI::memory_scope Scope,
          ONEAPI::memory_order Order, T Value) {
  auto *Ptr = MPtr.get();
  auto SPIRVOrder = getMemorySemanticsMask(Order);
  auto SPIRVScope = getScope(Scope);
  return __spirv_AtomicMin(Ptr, SPIRVScope, SPIRVOrder, Value);
}

template <typename T, access::address_space AddressSpace>
inline typename detail::enable_if_t<std::is_floating_point<T>::value, T>
AtomicMin(multi_ptr<T, AddressSpace> MPtr, ONEAPI::memory_scope Scope,
          ONEAPI::memory_order Order, T Value) {
  auto *Ptr = MPtr.get();
  auto SPIRVOrder = getMemorySemanticsMask(Order);
  auto SPIRVScope = getScope(Scope);
  return __spirv_AtomicMin(Ptr, SPIRVScope, SPIRVOrder, Value);
}

template <typename T, access::address_space AddressSpace>
inline typename detail::enable_if_t<std::is_integral<T>::value, T>
AtomicMax(multi_ptr<T, AddressSpace> MPtr, ONEAPI::memory_scope Scope,
          ONEAPI::memory_order Order, T Value) {
  auto *Ptr = MPtr.get();
  auto SPIRVOrder = getMemorySemanticsMask(Order);
  auto SPIRVScope = getScope(Scope);
  return __spirv_AtomicMax(Ptr, SPIRVScope, SPIRVOrder, Value);
}

template <typename T, access::address_space AddressSpace>
inline typename detail::enable_if_t<std::is_floating_point<T>::value, T>
AtomicMax(multi_ptr<T, AddressSpace> MPtr, ONEAPI::memory_scope Scope,
          ONEAPI::memory_order Order, T Value) {
  auto *Ptr = MPtr.get();
  auto SPIRVOrder = getMemorySemanticsMask(Order);
  auto SPIRVScope = getScope(Scope);
  return __spirv_AtomicMax(Ptr, SPIRVScope, SPIRVOrder, Value);
}

// Native shuffles map directly to a shuffle intrinsic:
// - The Intel SPIR-V extension natively supports all arithmetic types
// - The CUDA shfl intrinsics do not support vectors, and we use the _i32
//   variants for all scalar types
#ifndef __NVPTX__
template <typename T>
using EnableIfNativeShuffle =
    detail::enable_if_t<detail::is_arithmetic<T>::value, T>;
#else
template <typename T>
using EnableIfNativeShuffle = detail::enable_if_t<
    std::is_integral<T>::value && (sizeof(T) <= sizeof(int32_t)), T>;

template <typename T>
using EnableIfVectorShuffle =
    detail::enable_if_t<detail::is_vector_arithmetic<T>::value, T>;
#endif

#ifdef __NVPTX__
inline uint32_t membermask() {
  uint32_t FULL_MASK = 0xFFFFFFFF;
  uint32_t max_size = __spirv_SubgroupMaxSize();
  uint32_t sg_size = __spirv_SubgroupSize();
  return FULL_MASK >> (max_size - sg_size);
}
#endif

template <typename T>
EnableIfNativeShuffle<T> SubgroupShuffle(T x, id<1> local_id) {
#ifndef __NVPTX__
  using OCLT = detail::ConvertToOpenCLType_t<T>;
  return __spirv_SubgroupShuffleINTEL(OCLT(x),
                                      static_cast<uint32_t>(local_id.get(0)));
#else
  return __nvvm_shfl_sync_idx_i32(membermask(), x, local_id.get(0), 0x1f);
#endif
}

template <typename T>
EnableIfNativeShuffle<T> SubgroupShuffleXor(T x, id<1> local_id) {
#ifndef __NVPTX__
  using OCLT = detail::ConvertToOpenCLType_t<T>;
  return __spirv_SubgroupShuffleXorINTEL(
      OCLT(x), static_cast<uint32_t>(local_id.get(0)));
#else
  return __nvvm_shfl_sync_bfly_i32(membermask(), x, local_id.get(0), 0x1f);
#endif
}

template <typename T>
EnableIfNativeShuffle<T> SubgroupShuffleDown(T x, id<1> local_id) {
#ifndef __NVPTX__
  using OCLT = detail::ConvertToOpenCLType_t<T>;
  return __spirv_SubgroupShuffleDownINTEL(
      OCLT(x), OCLT(x), static_cast<uint32_t>(local_id.get(0)));
#else
  return __nvvm_shfl_sync_down_i32(membermask(), x, local_id.get(0), 0x1f);
#endif
}

template <typename T>
EnableIfNativeShuffle<T> SubgroupShuffleUp(T x, id<1> local_id) {
#ifndef __NVPTX__
  using OCLT = detail::ConvertToOpenCLType_t<T>;
  return __spirv_SubgroupShuffleUpINTEL(OCLT(x), OCLT(x),
                                        static_cast<uint32_t>(local_id.get(0)));
#else
  return __nvvm_shfl_sync_up_i32(membermask(), x, local_id.get(0), 0);
#endif
}

#ifdef __NVPTX__
template <typename T>
EnableIfVectorShuffle<T> SubgroupShuffle(T x, id<1> local_id) {
  T result;
  for (int s = 0; s < x.get_size(); ++s) {
    result[s] = SubgroupShuffle(x[s], local_id);
  }
  return result;
}

template <typename T>
EnableIfVectorShuffle<T> SubgroupShuffleXor(T x, id<1> local_id) {
  T result;
  for (int s = 0; s < x.get_size(); ++s) {
    result[s] = SubgroupShuffleXor(x[s], local_id);
  }
  return result;
}

template <typename T>
EnableIfVectorShuffle<T> SubgroupShuffleDown(T x, id<1> local_id) {
  T result;
  for (int s = 0; s < x.get_size(); ++s) {
    result[s] = SubgroupShuffleDown(x[s], local_id);
  }
  return result;
}

template <typename T>
EnableIfVectorShuffle<T> SubgroupShuffleUp(T x, id<1> local_id) {
  T result;
  for (int s = 0; s < x.get_size(); ++s) {
    result[s] = SubgroupShuffleUp(x[s], local_id);
  }
  return result;
}
#endif

// Bitcast shuffles can be implemented using a single SubgroupShuffle
// intrinsic, but require type-punning via an appropriate integer type
#ifndef __NVPTX__
template <typename T>
using EnableIfBitcastShuffle =
    detail::enable_if_t<!detail::is_arithmetic<T>::value &&
                            (std::is_trivially_copyable<T>::value &&
                             (sizeof(T) == 1 || sizeof(T) == 2 ||
                              sizeof(T) == 4 || sizeof(T) == 8)),
                        T>;
#else
template <typename T>
using EnableIfBitcastShuffle = detail::enable_if_t<
    !(std::is_integral<T>::value && (sizeof(T) <= sizeof(int32_t))) &&
        !detail::is_vector_arithmetic<T>::value &&
        (std::is_trivially_copyable<T>::value &&
         (sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4)),
    T>;
#endif

template <typename T>
using ConvertToNativeShuffleType_t = select_cl_scalar_integral_unsigned_t<T>;

template <typename T>
EnableIfBitcastShuffle<T> SubgroupShuffle(T x, id<1> local_id) {
  using ShuffleT = ConvertToNativeShuffleType_t<T>;
  auto ShuffleX = bit_cast<ShuffleT>(x);
#ifndef __NVPTX__
  ShuffleT Result = __spirv_SubgroupShuffleINTEL(
      ShuffleX, static_cast<uint32_t>(local_id.get(0)));
#else
  ShuffleT Result =
      __nvvm_shfl_sync_idx_i32(membermask(), ShuffleX, local_id.get(0), 0x1f);
#endif
  return bit_cast<T>(Result);
}

template <typename T>
EnableIfBitcastShuffle<T> SubgroupShuffleXor(T x, id<1> local_id) {
  using ShuffleT = ConvertToNativeShuffleType_t<T>;
  auto ShuffleX = bit_cast<ShuffleT>(x);
#ifndef __NVPTX__
  ShuffleT Result = __spirv_SubgroupShuffleXorINTEL(
      ShuffleX, static_cast<uint32_t>(local_id.get(0)));
#else
  ShuffleT Result =
      __nvvm_shfl_sync_bfly_i32(membermask(), ShuffleX, local_id.get(0), 0x1f);
#endif
  return bit_cast<T>(Result);
}

template <typename T>
EnableIfBitcastShuffle<T> SubgroupShuffleDown(T x, id<1> local_id) {
  using ShuffleT = ConvertToNativeShuffleType_t<T>;
  auto ShuffleX = bit_cast<ShuffleT>(x);
#ifndef __NVPTX__
  ShuffleT Result = __spirv_SubgroupShuffleDownINTEL(
      ShuffleX, ShuffleX, static_cast<uint32_t>(local_id.get(0)));
#else
  ShuffleT Result =
      __nvvm_shfl_sync_down_i32(membermask(), ShuffleX, local_id.get(0), 0x1f);
#endif
  return bit_cast<T>(Result);
}

template <typename T>
EnableIfBitcastShuffle<T> SubgroupShuffleUp(T x, id<1> local_id) {
  using ShuffleT = ConvertToNativeShuffleType_t<T>;
  auto ShuffleX = bit_cast<ShuffleT>(x);
#ifndef __NVPTX__
  ShuffleT Result = __spirv_SubgroupShuffleUpINTEL(
      ShuffleX, ShuffleX, static_cast<uint32_t>(local_id.get(0)));
#else
  ShuffleT Result =
      __nvvm_shfl_sync_up_i32(membermask(), ShuffleX, local_id.get(0), 0);
#endif
  return bit_cast<T>(Result);
}

// Generic shuffles may require multiple calls to SubgroupShuffle
// intrinsics, and should use the fewest shuffles possible:
// - Loop over 64-bit chunks until remaining bytes < 64-bit
// - At most one 32-bit, 16-bit and 8-bit chunk left over
#ifndef __NVPTX__
template <typename T>
using EnableIfGenericShuffle =
    detail::enable_if_t<!detail::is_arithmetic<T>::value &&
                            !(std::is_trivially_copyable<T>::value &&
                              (sizeof(T) == 1 || sizeof(T) == 2 ||
                               sizeof(T) == 4 || sizeof(T) == 8)),
                        T>;
#else
template <typename T>
using EnableIfGenericShuffle = detail::enable_if_t<
    !(std::is_integral<T>::value && (sizeof(T) <= sizeof(int32_t))) &&
        !detail::is_vector_arithmetic<T>::value &&
        !(std::is_trivially_copyable<T>::value &&
          (sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4)),
    T>;
#endif

template <typename T>
EnableIfGenericShuffle<T> SubgroupShuffle(T x, id<1> local_id) {
  T Result;
  char *XBytes = reinterpret_cast<char *>(&x);
  char *ResultBytes = reinterpret_cast<char *>(&Result);
  auto ShuffleBytes = [=](size_t Offset, size_t Size) {
    ShuffleChunkT ShuffleX, ShuffleResult;
    detail::memcpy(&ShuffleX, XBytes + Offset, Size);
    ShuffleResult = SubgroupShuffle(ShuffleX, local_id);
    detail::memcpy(ResultBytes + Offset, &ShuffleResult, Size);
  };
  GenericCall<T>(ShuffleBytes);
  return Result;
}

template <typename T>
EnableIfGenericShuffle<T> SubgroupShuffleXor(T x, id<1> local_id) {
  T Result;
  char *XBytes = reinterpret_cast<char *>(&x);
  char *ResultBytes = reinterpret_cast<char *>(&Result);
  auto ShuffleBytes = [=](size_t Offset, size_t Size) {
    ShuffleChunkT ShuffleX, ShuffleResult;
    detail::memcpy(&ShuffleX, XBytes + Offset, Size);
    ShuffleResult = SubgroupShuffleXor(ShuffleX, local_id);
    detail::memcpy(ResultBytes + Offset, &ShuffleResult, Size);
  };
  GenericCall<T>(ShuffleBytes);
  return Result;
}

template <typename T>
EnableIfGenericShuffle<T> SubgroupShuffleDown(T x, id<1> local_id) {
  T Result;
  char *XBytes = reinterpret_cast<char *>(&x);
  char *ResultBytes = reinterpret_cast<char *>(&Result);
  auto ShuffleBytes = [=](size_t Offset, size_t Size) {
    ShuffleChunkT ShuffleX, ShuffleResult;
    detail::memcpy(&ShuffleX, XBytes + Offset, Size);
    ShuffleResult = SubgroupShuffleDown(ShuffleX, local_id);
    detail::memcpy(ResultBytes + Offset, &ShuffleResult, Size);
  };
  GenericCall<T>(ShuffleBytes);
  return Result;
}

template <typename T>
EnableIfGenericShuffle<T> SubgroupShuffleUp(T x, id<1> local_id) {
  T Result;
  char *XBytes = reinterpret_cast<char *>(&x);
  char *ResultBytes = reinterpret_cast<char *>(&Result);
  auto ShuffleBytes = [=](size_t Offset, size_t Size) {
    ShuffleChunkT ShuffleX, ShuffleResult;
    detail::memcpy(&ShuffleX, XBytes + Offset, Size);
    ShuffleResult = SubgroupShuffleUp(ShuffleX, local_id);
    detail::memcpy(ResultBytes + Offset, &ShuffleResult, Size);
  };
  GenericCall<T>(ShuffleBytes);
  return Result;
}

} // namespace spirv
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
#endif //  __SYCL_DEVICE_ONLY__
