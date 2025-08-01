//===-- spirv.hpp - Helpers to generate SPIR-V instructions ----*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#ifdef __SYCL_DEVICE_ONLY__

// Some __spirv_* inrinsics are automatically forward-declared by the compiler,
// but not all of them. For example:
//   __spirv_AtomicStore(unsigned long long*, ...)
// Therefore, we need the following include to get forward-declarations of those
// versions.
#include <sycl/__spirv/spirv_ops.hpp>
#include <sycl/__spirv/spirv_types.hpp>

#include <sycl/ext/oneapi/experimental/non_uniform_groups.hpp> // for IdToMaskPosition

#if defined(__NVPTX__)
#include <sycl/ext/oneapi/experimental/cuda/masked_shuffles.hpp>
#endif

#include <sycl/detail/memcpy.hpp> // sycl::detail::memcpy
#include <sycl/detail/type_traits.hpp>

namespace sycl {
inline namespace _V1 {
struct sub_group;
namespace ext {
namespace oneapi {
struct sub_group;
namespace experimental {
template <typename ParentGroup> class fragment;

template <int Dimensions> class root_group;
} // namespace experimental
} // namespace oneapi
} // namespace ext

namespace detail {

// Helper for reinterpret casting the decorated pointer inside a multi_ptr
// without losing the decorations.
template <typename ToT, typename FromT, access::address_space Space,
          access::decorated IsDecorated>
inline typename multi_ptr<ToT, Space, access::decorated::yes>::pointer
GetMultiPtrDecoratedAs(multi_ptr<FromT, Space, IsDecorated> MPtr) {
  if constexpr (IsDecorated == access::decorated::legacy)
    return reinterpret_cast<
        typename multi_ptr<ToT, Space, access::decorated::yes>::pointer>(
        MPtr.get());
  else
    return reinterpret_cast<
        typename multi_ptr<ToT, Space, access::decorated::yes>::pointer>(
        MPtr.get_decorated());
}

template <typename NonUniformGroup>
inline uint32_t IdToMaskPosition(NonUniformGroup Group, uint32_t Id);

namespace spirv {

template <typename Group> struct group_scope {};

template <int Dimensions>
struct group_scope<sycl::ext::oneapi::experimental::root_group<Dimensions>> {
  static constexpr __spv::Scope::Flag value = __spv::Scope::Flag::Device;
};

template <int Dimensions> struct group_scope<group<Dimensions>> {
  static constexpr __spv::Scope::Flag value = __spv::Scope::Flag::Workgroup;
};

template <> struct group_scope<::sycl::ext::oneapi::sub_group> {
  static constexpr __spv::Scope::Flag value = __spv::Scope::Flag::Subgroup;
};
template <> struct group_scope<::sycl::sub_group> {
  static constexpr __spv::Scope::Flag value = __spv::Scope::Flag::Subgroup;
};

template <typename ParentGroup>
struct group_scope<sycl::ext::oneapi::experimental::fragment<ParentGroup>> {
  static constexpr __spv::Scope::Flag value = group_scope<ParentGroup>::value;
};

template <typename ParentGroup>
struct group_scope<sycl::ext::oneapi::experimental::tangle<ParentGroup>> {
  static constexpr __spv::Scope::Flag value = group_scope<ParentGroup>::value;
};

template <size_t ChunkSize, typename ParentGroup>
struct group_scope<
    sycl::ext::oneapi::experimental::chunk<ChunkSize, ParentGroup>> {
  static constexpr __spv::Scope::Flag value = group_scope<ParentGroup>::value;
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
    for (size_t Offset = 0; Offset + sizeof(ShuffleChunkT) <= sizeof(T);
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

template <typename Group> bool GroupAll(Group, bool pred) {
  return __spirv_GroupAll(group_scope<Group>::value, pred);
}
template <typename ParentGroup>
bool GroupAll(ext::oneapi::experimental::fragment<ParentGroup> g, bool pred) {
  // fragment partitions its parent into two groups (0 and 1)
  // We have to force each group down different control flow
  // Work-items in the "false" group (0) may still be active
  if (g.get_group_id() == 1) {
    return __spirv_GroupNonUniformAll(group_scope<ParentGroup>::value, pred);
  } else {
    return __spirv_GroupNonUniformAll(group_scope<ParentGroup>::value, pred);
  }
  // TODO: adding support for fragments have partitioning into more than two
  // groups such as labeled_partition:
  //
  // 1. add size_t framgment::get_group_range() const definition
  // 2. const auto group_range{g.get_group_range()};
  // 3. for (size_t i{}; i != group_range; ++i)
  //      if (g.get_group_id() == i)
  //        return __spirv_GroupNonUniformAll(
  //          group_scope<ParentGroup>::value,
  //          pred);
}
template <typename ParentGroup>
bool GroupAll(ext::oneapi::experimental::tangle<ParentGroup>, bool pred) {
  return __spirv_GroupNonUniformAll(group_scope<ParentGroup>::value, pred);
}

template <size_t ChunkSize, typename ParentGroup>
bool GroupAll(ext::oneapi::experimental::chunk<ChunkSize, ParentGroup>,
              bool pred) {
  // Using reduction because the GroupNonUniformAll have no support of cluster
  // size
  return __spirv_GroupNonUniformBitwiseAnd(
      group_scope<ParentGroup>::value,
      static_cast<uint32_t>(__spv::GroupOperation::ClusteredReduce),
      static_cast<uint32_t>(pred), ChunkSize);
}

template <typename Group> bool GroupAny(Group, bool pred) {
  return __spirv_GroupAny(group_scope<Group>::value, pred);
}
template <typename ParentGroup>
bool GroupAny(ext::oneapi::experimental::fragment<ParentGroup> g, bool pred) {
  // fragment partitions its parent into two groups (0 and 1)
  // We have to force each group down different control flow
  // Work-items in the "false" group (0) may still be active
  if (g.get_group_id() == 1) {
    return __spirv_GroupNonUniformAny(group_scope<ParentGroup>::value, pred);
  } else {
    return __spirv_GroupNonUniformAny(group_scope<ParentGroup>::value, pred);
  }
  // TODO: adding support for fragments have partitioning into more than two
  // groups such as labeled_partition:
  //
  // 1. add size_t framgment::get_group_range() const definition
  // 2. const auto group_range{g.get_group_range()};
  // 3. for (size_t i{}; i != group_range; ++i)
  //      if (g.get_group_id() == i)
  //        return __spirv_GroupNonUniformAny(
  //          group_scope<ParentGroup>::value,
  //          pred);
}
template <typename ParentGroup>
bool GroupAny(ext::oneapi::experimental::tangle<ParentGroup>, bool pred) {
  return __spirv_GroupNonUniformAny(group_scope<ParentGroup>::value, pred);
}

template <size_t ChunkSize, typename ParentGroup>
bool GroupAny(ext::oneapi::experimental::chunk<ChunkSize, ParentGroup>,
              bool pred) {
  // Using reduction because the GroupNonUniformAll have no support of cluster
  // size
  return __spirv_GroupNonUniformBitwiseOr(
      group_scope<ParentGroup>::value,
      static_cast<uint32_t>(__spv::GroupOperation::ClusteredReduce),
      static_cast<uint32_t>(pred), ChunkSize);
}

// Native broadcasts map directly to a SPIR-V GroupBroadcast intrinsic
// FIXME: Do not special-case for half or vec once all backends support all data
// types.
template <typename T>
using is_native_broadcast =
    std::bool_constant<detail::is_arithmetic<T>::value &&
                       !std::is_same<T, half>::value && !detail::is_vec_v<T> &&
                       !detail::is_marray_v<T> && !std::is_pointer_v<T>>;

template <typename T, typename IdT = size_t>
using EnableIfNativeBroadcast = std::enable_if_t<
    is_native_broadcast<T>::value && std::is_integral<IdT>::value, T>;

// Bitcast broadcasts can be implemented using a single SPIR-V GroupBroadcast
// intrinsic, but require type-punning via an appropriate integer type
template <typename T>
using is_bitcast_broadcast = std::bool_constant<
    !is_native_broadcast<T>::value && std::is_trivially_copyable<T>::value &&
    (sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8)>;

template <typename T, typename IdT = size_t>
using EnableIfBitcastBroadcast = std::enable_if_t<
    is_bitcast_broadcast<T>::value && std::is_integral<IdT>::value, T>;

template <typename T>
using ConvertToNativeBroadcastType_t = fixed_width_unsigned<sizeof(T)>;

// Generic broadcasts may require multiple calls to SPIR-V GroupBroadcast
// intrinsics, and should use the fewest broadcasts possible
// - Loop over 64-bit chunks until remaining bytes < 64-bit
// - At most one 32-bit, 16-bit and 8-bit chunk left over
template <typename T>
using is_generic_broadcast =
    std::bool_constant<!is_native_broadcast<T>::value &&
                       !is_bitcast_broadcast<T>::value &&
                       std::is_trivially_copyable<T>::value>;

template <typename T, typename IdT = size_t>
using EnableIfGenericBroadcast = std::enable_if_t<
    is_generic_broadcast<T>::value && std::is_integral<IdT>::value, T>;

// FIXME: Disable widening once all backends support all data types.
template <typename T>
using WidenOpenCLTypeTo32_t = std::conditional_t<
    std::is_same<T, opencl::cl_char>() || std::is_same<T, opencl::cl_short>(),
    opencl::cl_int,
    std::conditional_t<std::is_same<T, opencl::cl_uchar>() ||
                           std::is_same<T, opencl::cl_ushort>(),
                       opencl::cl_uint, T>>;

// Broadcast with scalar local index
// Work-group supports any integral type
// Sub-group currently supports only uint32_t
template <typename Group> struct GroupId {
  using type = size_t;
};
template <> struct GroupId<::sycl::ext::oneapi::sub_group> {
  using type = uint32_t;
};
template <> struct GroupId<::sycl::sub_group> {
  using type = uint32_t;
};

// Consolidated function for converting group arguments to OpenCL types.
template <typename Group, typename T, typename IdT>
EnableIfNativeBroadcast<T, IdT> GroupBroadcast(Group, T x, IdT local_id) {
  auto GroupLocalId = static_cast<typename GroupId<Group>::type>(local_id);
  auto OCLX = detail::convertToOpenCLType(x);
  WidenOpenCLTypeTo32_t<decltype(OCLX)> WideOCLX = OCLX;
  auto OCLId = detail::convertToOpenCLType(GroupLocalId);
  return __spirv_GroupBroadcast(group_scope<Group>::value, WideOCLX, OCLId);
}

template <typename ParentGroup, typename T, typename IdT>
EnableIfNativeBroadcast<T, IdT>
GroupBroadcast(sycl::ext::oneapi::experimental::fragment<ParentGroup> g, T x,
               IdT local_id) {
  // Remap local_id to its original numbering in ParentGroup.
  auto LocalId = detail::IdToMaskPosition(g, local_id);

  // TODO: Refactor to avoid duplication after design settles.
  auto GroupLocalId = static_cast<typename GroupId<ParentGroup>::type>(LocalId);
  auto OCLX = detail::convertToOpenCLType(x);
  WidenOpenCLTypeTo32_t<decltype(OCLX)> WideOCLX = OCLX;
  auto OCLId = detail::convertToOpenCLType(GroupLocalId);

  // fragment partitions its parent into two groups (0 and 1)
  // We have to force each group down different control flow
  // Work-items in the "false" group (0) may still be active
  if (g.get_group_id() == 1) {
    return __spirv_GroupNonUniformBroadcast(group_scope<ParentGroup>::value,
                                            WideOCLX, OCLId);
  } else {
    return __spirv_GroupNonUniformBroadcast(group_scope<ParentGroup>::value,
                                            WideOCLX, OCLId);
  }
  // TODO: adding support for fragments have partitioning into more than two
  // groups such as labeled_partition:
  //
  // 1. add size_t framgment::get_group_range() const definition
  // 2. const auto group_range{g.get_group_range()};
  // 3. for (size_t i{}; i != group_range; ++i)
  //      if (g.get_group_id() == i)
  //        return __spirv_GroupNonUniformBroadcast(
  //          group_scope<ParentGroup>::value,
  //          WideOCLX,
  //          OCLId);
}
template <typename ParentGroup, typename T, typename IdT>
EnableIfNativeBroadcast<T, IdT>
GroupBroadcast(ext::oneapi::experimental::tangle<ParentGroup> g, T x,
               IdT local_id) {
  // Remap local_id to its original numbering in ParentGroup.
  auto LocalId = detail::IdToMaskPosition(g, local_id);

  // TODO: Refactor to avoid duplication after design settles.
  auto GroupLocalId = static_cast<typename GroupId<ParentGroup>::type>(LocalId);
  auto OCLX = detail::convertToOpenCLType(x);
  WidenOpenCLTypeTo32_t<decltype(OCLX)> WideOCLX = OCLX;
  auto OCLId = detail::convertToOpenCLType(GroupLocalId);

  return __spirv_GroupNonUniformBroadcast(group_scope<ParentGroup>::value,
                                          WideOCLX, OCLId);
}

template <size_t ChunkSize, typename ParentGroup, typename T, typename IdT>
EnableIfNativeBroadcast<T, IdT>
GroupBroadcast(ext::oneapi::experimental::chunk<ChunkSize, ParentGroup> g, T x,
               IdT local_id) {
  // Remap local_id to its original numbering in ParentGroup
  auto LocalId = g.get_group_linear_id() * ChunkSize + local_id;

  // TODO: Refactor to avoid duplication after design settles.
  auto GroupLocalId = static_cast<typename GroupId<ParentGroup>::type>(LocalId);
  auto OCLX = detail::convertToOpenCLType(x);
  WidenOpenCLTypeTo32_t<decltype(OCLX)> WideOCLX = OCLX;
  auto OCLId = detail::convertToOpenCLType(GroupLocalId);

  // NonUniformBroadcast requires Id to be dynamically uniform, which does not
  // hold here; each partition is broadcasting a separate index.
  // In case of falling back to either NonUniformShuffle or a
  // NonUniformBroadcast per partition, and it's unclear which will be faster in
  // practice.
  return __spirv_GroupNonUniformShuffle(group_scope<ParentGroup>::value,
                                        WideOCLX, OCLId);
}

// opportunistic_group merged into fragment

template <typename Group, typename T, typename IdT>
EnableIfBitcastBroadcast<T, IdT> GroupBroadcast(Group g, T x, IdT local_id) {
  using BroadcastT = ConvertToNativeBroadcastType_t<T>;
  auto BroadcastX = sycl::bit_cast<BroadcastT>(x);
  BroadcastT Result = GroupBroadcast(g, BroadcastX, local_id);
  return sycl::bit_cast<T>(Result);
}
template <typename Group, typename T, typename IdT>
EnableIfGenericBroadcast<T, IdT> GroupBroadcast(Group g, T x, IdT local_id) {
  // Initialize with x to support type T without default constructor
  T Result = x;
  char *XBytes = reinterpret_cast<char *>(&x);
  char *ResultBytes = reinterpret_cast<char *>(&Result);
  auto BroadcastBytes = [=](size_t Offset, size_t Size) {
    uint64_t BroadcastX, BroadcastResult;
    detail::memcpy_no_adl(&BroadcastX, XBytes + Offset, Size);
    BroadcastResult = GroupBroadcast(g, BroadcastX, local_id);
    detail::memcpy_no_adl(ResultBytes + Offset, &BroadcastResult, Size);
  };
  GenericCall<T>(BroadcastBytes);
  return Result;
}

// Broadcast with vector local index
template <typename Group, typename T, int Dimensions>
EnableIfNativeBroadcast<T> GroupBroadcast(Group g, T x,
                                          id<Dimensions> local_id) {
  if (Dimensions == 1) {
    return GroupBroadcast(g, x, local_id[0]);
  }
  using IdT = vec<size_t, Dimensions>;
  IdT VecId;
  for (int i = 0; i < Dimensions; ++i) {
    VecId[i] = local_id[Dimensions - i - 1];
  }
  auto OCLX = detail::convertToOpenCLType(x);
  WidenOpenCLTypeTo32_t<decltype(OCLX)> WideOCLX = OCLX;
  auto OCLId = detail::convertToOpenCLType(VecId);
  return __spirv_GroupBroadcast(group_scope<Group>::value, WideOCLX, OCLId);
}
template <typename ParentGroup, typename T>
EnableIfNativeBroadcast<T>
GroupBroadcast(sycl::ext::oneapi::experimental::fragment<ParentGroup> g, T x,
               id<1> local_id) {
  // Limited to 1D indices for now because ParentGroup must be sub-group.
  return GroupBroadcast(g, x, local_id[0]);
}
template <typename Group, typename T, int Dimensions>
EnableIfBitcastBroadcast<T> GroupBroadcast(Group g, T x,
                                           id<Dimensions> local_id) {
  using BroadcastT = ConvertToNativeBroadcastType_t<T>;
  auto BroadcastX = sycl::bit_cast<BroadcastT>(x);
  BroadcastT Result = GroupBroadcast(g, BroadcastX, local_id);
  return sycl::bit_cast<T>(Result);
}
template <typename Group, typename T, int Dimensions>
EnableIfGenericBroadcast<T> GroupBroadcast(Group g, T x,
                                           id<Dimensions> local_id) {
  if (Dimensions == 1) {
    return GroupBroadcast(g, x, local_id[0]);
  }
  // Initialize with x to support type T without default constructor
  T Result = x;
  char *XBytes = reinterpret_cast<char *>(&x);
  char *ResultBytes = reinterpret_cast<char *>(&Result);
  auto BroadcastBytes = [=](size_t Offset, size_t Size) {
    uint64_t BroadcastX, BroadcastResult;
    detail::memcpy_no_adl(&BroadcastX, XBytes + Offset, Size);
    BroadcastResult = GroupBroadcast(g, BroadcastX, local_id);
    detail::memcpy_no_adl(ResultBytes + Offset, &BroadcastResult, Size);
  };
  GenericCall<T>(BroadcastBytes);
  return Result;
}

// Single happens-before means semantics should always apply to all spaces
// Although consume is unsupported, forwarding to acquire is valid
template <typename T>
static constexpr
    typename std::enable_if<std::is_same<T, sycl::memory_order>::value,
                            __spv::MemorySemanticsMask::Flag>::type
    getMemorySemanticsMask(T Order) {
  __spv::MemorySemanticsMask::Flag SpvOrder = __spv::MemorySemanticsMask::None;
  switch (Order) {
  case T::relaxed:
    SpvOrder = __spv::MemorySemanticsMask::None;
    break;
  case T::__consume_unsupported:
  case T::acquire:
    SpvOrder = __spv::MemorySemanticsMask::Acquire;
    break;
  case T::release:
    SpvOrder = __spv::MemorySemanticsMask::Release;
    break;
  case T::acq_rel:
    SpvOrder = __spv::MemorySemanticsMask::AcquireRelease;
    break;
  case T::seq_cst:
    SpvOrder = __spv::MemorySemanticsMask::SequentiallyConsistent;
    break;
  }
  return static_cast<__spv::MemorySemanticsMask::Flag>(
      SpvOrder | __spv::MemorySemanticsMask::SubgroupMemory |
      __spv::MemorySemanticsMask::WorkgroupMemory |
      __spv::MemorySemanticsMask::CrossWorkgroupMemory);
}

static constexpr __spv::Scope::Flag getScope(memory_scope Scope) {
  switch (Scope) {
  case memory_scope::work_item:
    return __spv::Scope::Invocation;
  case memory_scope::sub_group:
    return __spv::Scope::Subgroup;
  case memory_scope::work_group:
    return __spv::Scope::Workgroup;
  case memory_scope::device:
    return __spv::Scope::Device;
  case memory_scope::system:
    return __spv::Scope::CrossDevice;
  }
}

template <typename T, access::address_space AddressSpace,
          access::decorated IsDecorated>
inline typename std::enable_if_t<std::is_integral<T>::value, T>
AtomicCompareExchange(multi_ptr<T, AddressSpace, IsDecorated> MPtr,
                      memory_scope Scope, memory_order Success,
                      memory_order Failure, T Desired, T Expected) {
  auto SPIRVSuccess = getMemorySemanticsMask(Success);
  auto SPIRVFailure = getMemorySemanticsMask(Failure);
  auto SPIRVScope = getScope(Scope);
  auto *Ptr = GetMultiPtrDecoratedAs<T>(MPtr);
  return __spirv_AtomicCompareExchange(Ptr, SPIRVScope, SPIRVSuccess,
                                       SPIRVFailure, Desired, Expected);
}

template <typename T, access::address_space AddressSpace,
          access::decorated IsDecorated>
inline typename std::enable_if_t<std::is_floating_point<T>::value, T>
AtomicCompareExchange(multi_ptr<T, AddressSpace, IsDecorated> MPtr,
                      memory_scope Scope, memory_order Success,
                      memory_order Failure, T Desired, T Expected) {
  using I = detail::fixed_width_unsigned<sizeof(T)>;
  auto SPIRVSuccess = getMemorySemanticsMask(Success);
  auto SPIRVFailure = getMemorySemanticsMask(Failure);
  auto SPIRVScope = getScope(Scope);
  auto *PtrInt = GetMultiPtrDecoratedAs<I>(MPtr);
  I DesiredInt = sycl::bit_cast<I>(Desired);
  I ExpectedInt = sycl::bit_cast<I>(Expected);
  I ResultInt = __spirv_AtomicCompareExchange(
      PtrInt, SPIRVScope, SPIRVSuccess, SPIRVFailure, DesiredInt, ExpectedInt);
  return sycl::bit_cast<T>(ResultInt);
}

template <typename T, access::address_space AddressSpace,
          access::decorated IsDecorated>
inline typename std::enable_if_t<std::is_integral<T>::value, T>
AtomicLoad(multi_ptr<T, AddressSpace, IsDecorated> MPtr, memory_scope Scope,
           memory_order Order) {
  auto *Ptr = GetMultiPtrDecoratedAs<T>(MPtr);
  auto SPIRVOrder = getMemorySemanticsMask(Order);
  auto SPIRVScope = getScope(Scope);
  return __spirv_AtomicLoad(Ptr, SPIRVScope, SPIRVOrder);
}

template <typename T, access::address_space AddressSpace,
          access::decorated IsDecorated>
inline typename std::enable_if_t<std::is_floating_point<T>::value, T>
AtomicLoad(multi_ptr<T, AddressSpace, IsDecorated> MPtr, memory_scope Scope,
           memory_order Order) {
  using I = detail::fixed_width_unsigned<sizeof(T)>;
  auto *PtrInt = GetMultiPtrDecoratedAs<I>(MPtr);
  auto SPIRVOrder = getMemorySemanticsMask(Order);
  auto SPIRVScope = getScope(Scope);
  I ResultInt = __spirv_AtomicLoad(PtrInt, SPIRVScope, SPIRVOrder);
  return sycl::bit_cast<T>(ResultInt);
}

template <typename T, access::address_space AddressSpace,
          access::decorated IsDecorated>
inline typename std::enable_if_t<std::is_same_v<T, half>, T>
AtomicLoad(multi_ptr<T, AddressSpace, IsDecorated> MPtr, memory_scope Scope,
           memory_order Order) {
  auto *Ptr = GetMultiPtrDecoratedAs<_Float16>(MPtr);
  auto SPIRVOrder = getMemorySemanticsMask(Order);
  auto SPIRVScope = getScope(Scope);
  return __spirv_AtomicLoad(Ptr, SPIRVScope, SPIRVOrder);
}

template <typename T, access::address_space AddressSpace,
          access::decorated IsDecorated>
inline typename std::enable_if_t<std::is_integral<T>::value>
AtomicStore(multi_ptr<T, AddressSpace, IsDecorated> MPtr, memory_scope Scope,
            memory_order Order, T Value) {
  auto *Ptr = GetMultiPtrDecoratedAs<T>(MPtr);
  auto SPIRVOrder = getMemorySemanticsMask(Order);
  auto SPIRVScope = getScope(Scope);
  __spirv_AtomicStore(Ptr, SPIRVScope, SPIRVOrder, Value);
}

template <typename T, access::address_space AddressSpace,
          access::decorated IsDecorated>
inline typename std::enable_if_t<std::is_floating_point<T>::value>
AtomicStore(multi_ptr<T, AddressSpace, IsDecorated> MPtr, memory_scope Scope,
            memory_order Order, T Value) {
  using I = detail::fixed_width_unsigned<sizeof(T)>;
  auto *PtrInt = GetMultiPtrDecoratedAs<I>(MPtr);
  auto SPIRVOrder = getMemorySemanticsMask(Order);
  auto SPIRVScope = getScope(Scope);
  I ValueInt = sycl::bit_cast<I>(Value);
  __spirv_AtomicStore(PtrInt, SPIRVScope, SPIRVOrder, ValueInt);
}

template <typename T, access::address_space AddressSpace,
          access::decorated IsDecorated>
inline typename std::enable_if_t<std::is_same_v<T, half>>
AtomicStore(multi_ptr<T, AddressSpace, IsDecorated> MPtr, memory_scope Scope,
            memory_order Order, T Value) {
  auto *Ptr = GetMultiPtrDecoratedAs<_Float16>(MPtr);
  auto SPIRVOrder = getMemorySemanticsMask(Order);
  auto SPIRVScope = getScope(Scope);
  __spirv_AtomicStore(Ptr, SPIRVScope, SPIRVOrder, Value);
}

template <typename T, access::address_space AddressSpace,
          access::decorated IsDecorated>
inline typename std::enable_if_t<std::is_integral<T>::value, T>
AtomicExchange(multi_ptr<T, AddressSpace, IsDecorated> MPtr, memory_scope Scope,
               memory_order Order, T Value) {
  auto *Ptr = GetMultiPtrDecoratedAs<T>(MPtr);
  auto SPIRVOrder = getMemorySemanticsMask(Order);
  auto SPIRVScope = getScope(Scope);
  return __spirv_AtomicExchange(Ptr, SPIRVScope, SPIRVOrder, Value);
}

template <typename T, access::address_space AddressSpace,
          access::decorated IsDecorated>
inline typename std::enable_if_t<std::is_floating_point<T>::value, T>
AtomicExchange(multi_ptr<T, AddressSpace, IsDecorated> MPtr, memory_scope Scope,
               memory_order Order, T Value) {
  using I = detail::fixed_width_unsigned<sizeof(T)>;
  auto *PtrInt = GetMultiPtrDecoratedAs<I>(MPtr);
  auto SPIRVOrder = getMemorySemanticsMask(Order);
  auto SPIRVScope = getScope(Scope);
  I ValueInt = sycl::bit_cast<I>(Value);
  I ResultInt =
      __spirv_AtomicExchange(PtrInt, SPIRVScope, SPIRVOrder, ValueInt);
  return sycl::bit_cast<T>(ResultInt);
}

template <typename T, access::address_space AddressSpace,
          access::decorated IsDecorated>
inline typename std::enable_if_t<std::is_same_v<half, T>, T>
AtomicExchange(multi_ptr<T, AddressSpace, IsDecorated> MPtr, memory_scope Scope,
               memory_order Order, T Value) {
  auto *Ptr = GetMultiPtrDecoratedAs<_Float16>(MPtr);
  auto SPIRVOrder = getMemorySemanticsMask(Order);
  auto SPIRVScope = getScope(Scope);
  return __spirv_AtomicExchange(Ptr, SPIRVScope, SPIRVOrder, Value);
}

template <typename T, access::address_space AddressSpace,
          access::decorated IsDecorated>
inline typename std::enable_if_t<std::is_integral<T>::value, T>
AtomicIAdd(multi_ptr<T, AddressSpace, IsDecorated> MPtr, memory_scope Scope,
           memory_order Order, T Value) {
  auto *Ptr = GetMultiPtrDecoratedAs<T>(MPtr);
  auto SPIRVOrder = getMemorySemanticsMask(Order);
  auto SPIRVScope = getScope(Scope);
  return __spirv_AtomicIAdd(Ptr, SPIRVScope, SPIRVOrder, Value);
}

template <typename T, access::address_space AddressSpace,
          access::decorated IsDecorated>
inline typename std::enable_if_t<std::is_integral<T>::value, T>
AtomicISub(multi_ptr<T, AddressSpace, IsDecorated> MPtr, memory_scope Scope,
           memory_order Order, T Value) {
  auto *Ptr = GetMultiPtrDecoratedAs<T>(MPtr);
  auto SPIRVOrder = getMemorySemanticsMask(Order);
  auto SPIRVScope = getScope(Scope);
  return __spirv_AtomicISub(Ptr, SPIRVScope, SPIRVOrder, Value);
}

template <typename T, access::address_space AddressSpace,
          access::decorated IsDecorated>
inline typename std::enable_if_t<std::is_floating_point<T>::value, T>
AtomicFAdd(multi_ptr<T, AddressSpace, IsDecorated> MPtr, memory_scope Scope,
           memory_order Order, T Value) {
  auto *Ptr = GetMultiPtrDecoratedAs<T>(MPtr);
  auto SPIRVOrder = getMemorySemanticsMask(Order);
  auto SPIRVScope = getScope(Scope);
  return __spirv_AtomicFAddEXT(Ptr, SPIRVScope, SPIRVOrder, Value);
}

template <typename T, access::address_space AddressSpace,
          access::decorated IsDecorated>
inline typename std::enable_if_t<std::is_same_v<half, T>, T>
AtomicFAdd(multi_ptr<T, AddressSpace, IsDecorated> MPtr, memory_scope Scope,
           memory_order Order, T Value) {
  auto *Ptr = GetMultiPtrDecoratedAs<_Float16>(MPtr);
  auto SPIRVOrder = getMemorySemanticsMask(Order);
  auto SPIRVScope = getScope(Scope);
  return __spirv_AtomicFAddEXT(Ptr, SPIRVScope, SPIRVOrder, Value);
}

template <typename T, access::address_space AddressSpace,
          access::decorated IsDecorated>
inline typename std::enable_if_t<std::is_integral<T>::value, T>
AtomicAnd(multi_ptr<T, AddressSpace, IsDecorated> MPtr, memory_scope Scope,
          memory_order Order, T Value) {
  auto *Ptr = GetMultiPtrDecoratedAs<T>(MPtr);
  auto SPIRVOrder = getMemorySemanticsMask(Order);
  auto SPIRVScope = getScope(Scope);
  return __spirv_AtomicAnd(Ptr, SPIRVScope, SPIRVOrder, Value);
}

template <typename T, access::address_space AddressSpace,
          access::decorated IsDecorated>
inline typename std::enable_if_t<std::is_integral<T>::value, T>
AtomicOr(multi_ptr<T, AddressSpace, IsDecorated> MPtr, memory_scope Scope,
         memory_order Order, T Value) {
  auto *Ptr = GetMultiPtrDecoratedAs<T>(MPtr);
  auto SPIRVOrder = getMemorySemanticsMask(Order);
  auto SPIRVScope = getScope(Scope);
  return __spirv_AtomicOr(Ptr, SPIRVScope, SPIRVOrder, Value);
}

template <typename T, access::address_space AddressSpace,
          access::decorated IsDecorated>
inline typename std::enable_if_t<std::is_integral<T>::value, T>
AtomicXor(multi_ptr<T, AddressSpace, IsDecorated> MPtr, memory_scope Scope,
          memory_order Order, T Value) {
  auto *Ptr = GetMultiPtrDecoratedAs<T>(MPtr);
  auto SPIRVOrder = getMemorySemanticsMask(Order);
  auto SPIRVScope = getScope(Scope);
  return __spirv_AtomicXor(Ptr, SPIRVScope, SPIRVOrder, Value);
}

template <typename T, access::address_space AddressSpace,
          access::decorated IsDecorated>
inline typename std::enable_if_t<std::is_integral<T>::value, T>
AtomicMin(multi_ptr<T, AddressSpace, IsDecorated> MPtr, memory_scope Scope,
          memory_order Order, T Value) {
  auto *Ptr = GetMultiPtrDecoratedAs<T>(MPtr);
  auto SPIRVOrder = getMemorySemanticsMask(Order);
  auto SPIRVScope = getScope(Scope);
  return __spirv_AtomicMin(Ptr, SPIRVScope, SPIRVOrder, Value);
}

template <typename T, access::address_space AddressSpace,
          access::decorated IsDecorated>
inline typename std::enable_if_t<std::is_floating_point<T>::value, T>
AtomicMin(multi_ptr<T, AddressSpace, IsDecorated> MPtr, memory_scope Scope,
          memory_order Order, T Value) {
  auto *Ptr = GetMultiPtrDecoratedAs<T>(MPtr);
  auto SPIRVOrder = getMemorySemanticsMask(Order);
  auto SPIRVScope = getScope(Scope);
  return __spirv_AtomicMin(Ptr, SPIRVScope, SPIRVOrder, Value);
}

template <typename T, access::address_space AddressSpace,
          access::decorated IsDecorated>
inline typename std::enable_if_t<std::is_same_v<half, T>, T>
AtomicMin(multi_ptr<T, AddressSpace, IsDecorated> MPtr, memory_scope Scope,
          memory_order Order, T Value) {
  auto *Ptr = GetMultiPtrDecoratedAs<_Float16>(MPtr);
  auto SPIRVOrder = getMemorySemanticsMask(Order);
  auto SPIRVScope = getScope(Scope);
  return __spirv_AtomicFMinEXT(Ptr, SPIRVScope, SPIRVOrder, Value);
}

template <typename T, access::address_space AddressSpace,
          access::decorated IsDecorated>
inline typename std::enable_if_t<std::is_integral<T>::value, T>
AtomicMax(multi_ptr<T, AddressSpace, IsDecorated> MPtr, memory_scope Scope,
          memory_order Order, T Value) {
  auto *Ptr = GetMultiPtrDecoratedAs<T>(MPtr);
  auto SPIRVOrder = getMemorySemanticsMask(Order);
  auto SPIRVScope = getScope(Scope);
  return __spirv_AtomicMax(Ptr, SPIRVScope, SPIRVOrder, Value);
}

template <typename T, access::address_space AddressSpace,
          access::decorated IsDecorated>
inline typename std::enable_if_t<std::is_floating_point<T>::value, T>
AtomicMax(multi_ptr<T, AddressSpace, IsDecorated> MPtr, memory_scope Scope,
          memory_order Order, T Value) {
  auto *Ptr = GetMultiPtrDecoratedAs<T>(MPtr);
  auto SPIRVOrder = getMemorySemanticsMask(Order);
  auto SPIRVScope = getScope(Scope);
  return __spirv_AtomicMax(Ptr, SPIRVScope, SPIRVOrder, Value);
}

template <typename T, access::address_space AddressSpace,
          access::decorated IsDecorated>
inline typename std::enable_if_t<std::is_same_v<half, T>, T>
AtomicMax(multi_ptr<T, AddressSpace, IsDecorated> MPtr, memory_scope Scope,
          memory_order Order, T Value) {
  auto *Ptr = GetMultiPtrDecoratedAs<_Float16>(MPtr);
  auto SPIRVOrder = getMemorySemanticsMask(Order);
  auto SPIRVScope = getScope(Scope);
  return __spirv_AtomicFMaxEXT(Ptr, SPIRVScope, SPIRVOrder, Value);
}

// Native shuffles map directly to a shuffle intrinsic:
// - The Intel SPIR-V extension natively supports all arithmetic types.
//   However, OpenCL extension natively supports float vectors,
//   integer vectors, half scalar and double scalar.
//   For double, long, long long, unsigned long, unsigned long long
//   and half vectors we perform emulation with scalar version.
// - The CUDA shfl intrinsics do not support vectors, and we use the _i32
//   variants for all scalar types
#ifndef __NVPTX__

template <typename T>
struct VecTypeIsProhibitedForNativeShuffle
    : std::bool_constant<
          (detail::get_vec_size<T>::size > 1) &&
          check_type_in_v<vector_element_t<T>, double, long, long long,
                          unsigned long, unsigned long long, half>> {};

// Native shuffle is supported if T is scalar arithmetic type or vector type
// with arithmetic element type which is not in prohibited list.
// detail::is_arithmetic looks through pointer, marray and vec into the element
// type, so we have to exclude marray and pointer types.
template <typename T>
struct NativeShuffle
    : std::bool_constant<detail::is_arithmetic<T>::value &&
                         !VecTypeIsProhibitedForNativeShuffle<T>::value &&
                         !detail::is_marray_v<T> && !detail::is_pointer_v<T>> {
};

// Non-scalar shuffle (emulation via loop + scalar shuffle) is used if we have a
// vector type for which native shuffle is not supported or for marray type.
template <typename T>
struct NonScalarShuffle
    : std::bool_constant<!NativeShuffle<T>::value &&
                         (detail::is_vec_v<T> || detail::is_marray_v<T>)> {};

template <typename T>
using EnableIfNativeShuffle = std::enable_if_t<NativeShuffle<T>::value, T>;

template <typename T>
using EnableIfNonScalarShuffle =
    std::enable_if_t<NonScalarShuffle<T>::value, T>;

#else  // ifndef __NVPTX__

template <typename T>
using EnableIfNativeShuffle = std::enable_if_t<
    std::is_integral<T>::value && (sizeof(T) <= sizeof(int32_t)), T>;

template <typename T>
using EnableIfNonScalarShuffle =
    std::enable_if_t<detail::is_nonscalar_arithmetic<T>::value, T>;
#endif // ifndef __NVPTX__

// Bitcast shuffles can be implemented using a single SubgroupShuffle
// intrinsic, but require type-punning via an appropriate integer type
#ifndef __NVPTX__

// Use bitcast shuffle for trivially copyable types satisfying size requirements
// that are not handled by native shuffle and non-scalar shuffle.
template <typename T>
struct BitcastShuffle : std::bool_constant<!NativeShuffle<T>::value &&
                                           !NonScalarShuffle<T>::value &&
                                           std::is_trivially_copyable_v<T> &&
                                           (sizeof(T) == 1 || sizeof(T) == 2 ||
                                            sizeof(T) == 4 || sizeof(T) == 8)> {
};

template <typename T>
using EnableIfBitcastShuffle = std::enable_if_t<BitcastShuffle<T>::value, T>;
#else
template <typename T>
using EnableIfBitcastShuffle =
    std::enable_if_t<!(std::is_integral_v<T> &&
                       (sizeof(T) <= sizeof(int32_t))) &&
                         !detail::is_nonscalar_arithmetic<T>::value &&
                         (std::is_trivially_copyable_v<T> &&
                          (sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4)),
                     T>;
#endif // ifndef __NVPTX__

// Generic shuffles may require multiple calls to SubgroupShuffle
// intrinsics, and should use the fewest shuffles possible:
// - Loop over 64-bit chunks until remaining bytes < 64-bit
// - At most one 32-bit, 16-bit and 8-bit chunk left over
#ifndef __NVPTX__
template <typename T>
using EnableIfGenericShuffle =
    std::enable_if_t<!NativeShuffle<T>::value && !NonScalarShuffle<T>::value &&
                         !BitcastShuffle<T>::value,
                     T>;
#else
template <typename T>
using EnableIfGenericShuffle = std::enable_if_t<
    !(std::is_integral<T>::value && (sizeof(T) <= sizeof(int32_t))) &&
        !detail::is_nonscalar_arithmetic<T>::value &&
        !(std::is_trivially_copyable_v<T> &&
          (sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4)),
    T>;
#endif

#ifdef __NVPTX__
inline uint32_t membermask() {
  // use a full mask as sync operations are required to be convergent and exited
  // threads can safely be in the mask
  return 0xFFFFFFFF;
}
#endif

template <typename GroupT>
inline uint32_t MapShuffleID(GroupT g, id<1> local_id) {
  if constexpr (sycl::detail::is_tangle_v<GroupT> ||
                sycl::detail::is_fragment_v<GroupT>)
    return detail::IdToMaskPosition(g, local_id);
  else if constexpr (sycl::detail::is_chunk_v<GroupT>)
    return g.get_group_linear_id() * g.get_local_range().size() + local_id;
  else
    return local_id.get(0);
}

// Forward declarations for template overloadings
template <typename GroupT, typename T>
EnableIfBitcastShuffle<T> Shuffle(GroupT g, T x, id<1> local_id);

template <typename GroupT, typename T>
EnableIfBitcastShuffle<T> ShuffleXor(GroupT g, T x, id<1> local_id);

template <typename GroupT, typename T>
EnableIfBitcastShuffle<T> ShuffleDown(GroupT g, T x, uint32_t delta);

template <typename GroupT, typename T>
EnableIfBitcastShuffle<T> ShuffleUp(GroupT g, T x, uint32_t delta);

template <typename GroupT, typename T>
EnableIfGenericShuffle<T> Shuffle(GroupT g, T x, id<1> local_id);

template <typename GroupT, typename T>
EnableIfGenericShuffle<T> ShuffleXor(GroupT g, T x, id<1> local_id);

template <typename GroupT, typename T>
EnableIfGenericShuffle<T> ShuffleDown(GroupT g, T x, uint32_t delta);

template <typename GroupT, typename T>
EnableIfGenericShuffle<T> ShuffleUp(GroupT g, T x, uint32_t delta);

template <typename GroupT, typename T>
EnableIfNativeShuffle<T> Shuffle(GroupT g, T x, id<1> local_id) {
  uint32_t LocalId = MapShuffleID(g, local_id);
#ifndef __NVPTX__
  std::ignore = g;
  if constexpr (ext::oneapi::experimental::is_user_constructed_group_v<
                    GroupT> &&
                detail::is_vec<T>::value) {
    // Temporary work-around due to a bug in IGC.
    // TODO: Remove when IGC bug is fixed.
    T result;
    for (int s = 0; s < x.size(); ++s)
      result[s] = Shuffle(g, x[s], local_id);
    return result;
  } else if constexpr (ext::oneapi::experimental::is_user_constructed_group_v<
                           GroupT>) {
    return convertFromOpenCLTypeFor<T>(__spirv_GroupNonUniformShuffle(
        group_scope<GroupT>::value, convertToOpenCLType(x), LocalId));
  } else {
    // Subgroup.
    return convertFromOpenCLTypeFor<T>(
        __spirv_SubgroupShuffleINTEL(convertToOpenCLType(x), LocalId));
  }
#else
  if constexpr (ext::oneapi::experimental::is_user_constructed_group_v<
                    GroupT>) {
    return cuda_shfl_sync_idx_i32(detail::ExtractMask(detail::GetMask(g))[0], x,
                                  LocalId, 31);
  } else {
    return cuda_shfl_sync_idx_i32(membermask(), x, LocalId, 31);
  }
#endif
}

template <typename GroupT, typename T>
EnableIfNativeShuffle<T> ShuffleXor(GroupT g, T x, id<1> mask) {
#ifndef __NVPTX__
  std::ignore = g;
  if constexpr (ext::oneapi::experimental::is_user_constructed_group_v<
                    GroupT> &&
                detail::is_vec<T>::value) {
    // Temporary work-around due to a bug in IGC.
    // TODO: Remove when IGC bug is fixed.
    T result;
    for (int s = 0; s < x.size(); ++s)
      result[s] = ShuffleXor(g, x[s], mask);
    return result;
  } else if constexpr (ext::oneapi::experimental::is_user_constructed_group_v<
                           GroupT>) {
    // Since the masks are relative to the groups, we could either try to adjust
    // the mask or simply do the xor ourselves. Latter option is efficient,
    // general, and simple so we go with that.
    id<1> TargetLocalId = g.get_local_id() ^ mask;
    uint32_t TargetId = MapShuffleID(g, TargetLocalId);
    return __spirv_GroupNonUniformShuffle(group_scope<GroupT>::value,
                                          convertToOpenCLType(x), TargetId);
  } else {
    // Subgroup.
    return convertFromOpenCLTypeFor<T>(__spirv_SubgroupShuffleXorINTEL(
        convertToOpenCLType(x), static_cast<uint32_t>(mask.get(0))));
  }
#else
  if constexpr (ext::oneapi::experimental::is_user_constructed_group_v<
                    GroupT>) {
    auto MemberMask = detail::ExtractMask(detail::GetMask(g))[0];
    if constexpr (sycl::detail::is_chunk_v<GroupT>) {
      return cuda_shfl_sync_bfly_i32(MemberMask, x,
                                     static_cast<uint32_t>(mask.get(0)), 0x1f);
    } else {
      int unfoldedSrcSetBit =
          (g.get_local_id()[0] ^ static_cast<uint32_t>(mask.get(0))) + 1;
      return cuda_shfl_sync_idx_i32(
          MemberMask, x, __nvvm_fns(MemberMask, 0, unfoldedSrcSetBit), 31);
    }
  } else {
    return cuda_shfl_sync_bfly_i32(membermask(), x,
                                   static_cast<uint32_t>(mask.get(0)), 0x1f);
  }
#endif
}

template <typename GroupT, typename T>
EnableIfNativeShuffle<T> ShuffleDown(GroupT g, T x, uint32_t delta) {
#ifndef __NVPTX__
  if constexpr (ext::oneapi::experimental::is_user_constructed_group_v<
                    GroupT> &&
                detail::is_vec<T>::value) {
    // Temporary work-around due to a bug in IGC.
    // TODO: Remove when IGC bug is fixed.
    T result;
    for (int s = 0; s < x.size(); ++s)
      result[s] = ShuffleDown(g, x[s], delta);
    return result;
  } else if constexpr (ext::oneapi::experimental::is_user_constructed_group_v<
                           GroupT>) {
    id<1> TargetLocalId = g.get_local_id();
    // ID outside the group range is UB, so we just keep the current item ID
    // unchanged.
    if (TargetLocalId[0] + delta < g.get_local_linear_range())
      TargetLocalId[0] += delta;
    uint32_t TargetId = MapShuffleID(g, TargetLocalId);
    return __spirv_GroupNonUniformShuffle(group_scope<GroupT>::value,
                                          convertToOpenCLType(x), TargetId);
  } else {
    // Subgroup.
    return convertFromOpenCLTypeFor<T>(__spirv_SubgroupShuffleDownINTEL(
        convertToOpenCLType(x), convertToOpenCLType(x), delta));
  }
#else
  if constexpr (ext::oneapi::experimental::is_user_constructed_group_v<
                    GroupT>) {
    auto MemberMask = detail::ExtractMask(detail::GetMask(g))[0];
    if constexpr (sycl::detail::is_chunk_v<GroupT>) {
      return cuda_shfl_sync_down_i32(MemberMask, x, delta, 31);
    } else {
      unsigned localSetBit = g.get_local_id()[0] + 1;
      int unfoldedSrcSetBit = localSetBit + delta;
      return cuda_shfl_sync_idx_i32(
          MemberMask, x, __nvvm_fns(MemberMask, 0, unfoldedSrcSetBit), 31);
    }
  } else {
    return cuda_shfl_sync_down_i32(membermask(), x, delta, 31);
  }
#endif
}

template <typename GroupT, typename T>
EnableIfNativeShuffle<T> ShuffleUp(GroupT g, T x, uint32_t delta) {
#ifndef __NVPTX__
  if constexpr (ext::oneapi::experimental::is_user_constructed_group_v<
                    GroupT> &&
                detail::is_vec<T>::value) {
    // Temporary work-around due to a bug in IGC.
    // TODO: Remove when IGC bug is fixed.
    T result;
    for (int s = 0; s < x.size(); ++s)
      result[s] = ShuffleUp(g, x[s], delta);
    return result;
  } else if constexpr (ext::oneapi::experimental::is_user_constructed_group_v<
                           GroupT>) {
    id<1> TargetLocalId = g.get_local_id();
    // Underflow is UB, so we just keep the current item ID unchanged.
    if (TargetLocalId[0] >= delta)
      TargetLocalId[0] -= delta;
    uint32_t TargetId = MapShuffleID(g, TargetLocalId);
    return __spirv_GroupNonUniformShuffle(group_scope<GroupT>::value,
                                          convertToOpenCLType(x), TargetId);
  } else {
    // Subgroup.
    return convertFromOpenCLTypeFor<T>(__spirv_SubgroupShuffleUpINTEL(
        convertToOpenCLType(x), convertToOpenCLType(x), delta));
  }
#else
  if constexpr (ext::oneapi::experimental::is_user_constructed_group_v<
                    GroupT>) {
    auto MemberMask = detail::ExtractMask(detail::GetMask(g))[0];
    if constexpr (sycl::detail::is_chunk_v<GroupT>) {
      return cuda_shfl_sync_up_i32(MemberMask, x, delta, 0);
    } else {
      unsigned localSetBit = g.get_local_id()[0] + 1;
      int unfoldedSrcSetBit = localSetBit - delta;

      return cuda_shfl_sync_idx_i32(
          MemberMask, x, __nvvm_fns(MemberMask, 0, unfoldedSrcSetBit), 31);
    }
  } else {
    return cuda_shfl_sync_up_i32(membermask(), x, delta, 0);
  }
#endif
}

template <typename GroupT, typename T>
EnableIfNonScalarShuffle<T> Shuffle(GroupT g, T x, id<1> local_id) {
  T result;
  for (int s = 0; s < x.size(); ++s) {
    result[s] = Shuffle(g, x[s], local_id);
  }
  return result;
}

template <typename GroupT, typename T>
EnableIfNonScalarShuffle<T> ShuffleXor(GroupT g, T x, id<1> local_id) {
  T result;
  for (int s = 0; s < x.size(); ++s) {
    result[s] = ShuffleXor(g, x[s], local_id);
  }
  return result;
}

template <typename GroupT, typename T>
EnableIfNonScalarShuffle<T> ShuffleDown(GroupT g, T x, uint32_t delta) {
  T result;
  for (int s = 0; s < x.size(); ++s) {
    result[s] = ShuffleDown(g, x[s], delta);
  }
  return result;
}

template <typename GroupT, typename T>
EnableIfNonScalarShuffle<T> ShuffleUp(GroupT g, T x, uint32_t delta) {
  T result;
  for (int s = 0; s < x.size(); ++s) {
    result[s] = ShuffleUp(g, x[s], delta);
  }
  return result;
}

template <typename T>
using ConvertToNativeShuffleType_t = fixed_width_unsigned<sizeof(T)>;

template <typename GroupT, typename T>
EnableIfBitcastShuffle<T> Shuffle(GroupT g, T x, id<1> local_id) {
  using ShuffleT = ConvertToNativeShuffleType_t<T>;
  auto ShuffleX = sycl::bit_cast<ShuffleT>(x);
  ShuffleT Result = Shuffle(g, ShuffleX, local_id);
  return sycl::bit_cast<T>(Result);
}

template <typename GroupT, typename T>
EnableIfBitcastShuffle<T> ShuffleXor(GroupT g, T x, id<1> local_id) {
  using ShuffleT = ConvertToNativeShuffleType_t<T>;
  auto ShuffleX = sycl::bit_cast<ShuffleT>(x);
  ShuffleT Result = ShuffleXor(g, ShuffleX, local_id);
  return sycl::bit_cast<T>(Result);
}

template <typename GroupT, typename T>
EnableIfBitcastShuffle<T> ShuffleDown(GroupT g, T x, uint32_t delta) {
  using ShuffleT = ConvertToNativeShuffleType_t<T>;
  auto ShuffleX = sycl::bit_cast<ShuffleT>(x);
  ShuffleT Result = ShuffleDown(g, ShuffleX, delta);
  return sycl::bit_cast<T>(Result);
}

template <typename GroupT, typename T>
EnableIfBitcastShuffle<T> ShuffleUp(GroupT g, T x, uint32_t delta) {
  using ShuffleT = ConvertToNativeShuffleType_t<T>;
  auto ShuffleX = sycl::bit_cast<ShuffleT>(x);
  ShuffleT Result = ShuffleUp(g, ShuffleX, delta);
  return sycl::bit_cast<T>(Result);
}

template <typename GroupT, typename T>
EnableIfGenericShuffle<T> Shuffle(GroupT g, T x, id<1> local_id) {
  T Result;
  char *XBytes = reinterpret_cast<char *>(&x);
  char *ResultBytes = reinterpret_cast<char *>(&Result);
  auto ShuffleBytes = [=](size_t Offset, size_t Size) {
    ShuffleChunkT ShuffleX, ShuffleResult;
    detail::memcpy_no_adl(&ShuffleX, XBytes + Offset, Size);
    ShuffleResult = Shuffle(g, ShuffleX, local_id);
    detail::memcpy_no_adl(ResultBytes + Offset, &ShuffleResult, Size);
  };
  GenericCall<T>(ShuffleBytes);
  return Result;
}

template <typename GroupT, typename T>
EnableIfGenericShuffle<T> ShuffleXor(GroupT g, T x, id<1> local_id) {
  T Result;
  char *XBytes = reinterpret_cast<char *>(&x);
  char *ResultBytes = reinterpret_cast<char *>(&Result);
  auto ShuffleBytes = [=](size_t Offset, size_t Size) {
    ShuffleChunkT ShuffleX, ShuffleResult;
    detail::memcpy_no_adl(&ShuffleX, XBytes + Offset, Size);
    ShuffleResult = ShuffleXor(g, ShuffleX, local_id);
    detail::memcpy_no_adl(ResultBytes + Offset, &ShuffleResult, Size);
  };
  GenericCall<T>(ShuffleBytes);
  return Result;
}

template <typename GroupT, typename T>
EnableIfGenericShuffle<T> ShuffleDown(GroupT g, T x, uint32_t delta) {
  T Result;
  char *XBytes = reinterpret_cast<char *>(&x);
  char *ResultBytes = reinterpret_cast<char *>(&Result);
  auto ShuffleBytes = [=](size_t Offset, size_t Size) {
    ShuffleChunkT ShuffleX, ShuffleResult;
    detail::memcpy_no_adl(&ShuffleX, XBytes + Offset, Size);
    ShuffleResult = ShuffleDown(g, ShuffleX, delta);
    detail::memcpy_no_adl(ResultBytes + Offset, &ShuffleResult, Size);
  };
  GenericCall<T>(ShuffleBytes);
  return Result;
}

template <typename GroupT, typename T>
EnableIfGenericShuffle<T> ShuffleUp(GroupT g, T x, uint32_t delta) {
  T Result;
  char *XBytes = reinterpret_cast<char *>(&x);
  char *ResultBytes = reinterpret_cast<char *>(&Result);
  auto ShuffleBytes = [=](size_t Offset, size_t Size) {
    ShuffleChunkT ShuffleX, ShuffleResult;
    detail::memcpy_no_adl(&ShuffleX, XBytes + Offset, Size);
    ShuffleResult = ShuffleUp(g, ShuffleX, delta);
    detail::memcpy_no_adl(ResultBytes + Offset, &ShuffleResult, Size);
  };
  GenericCall<T>(ShuffleBytes);
  return Result;
}

template <typename Group>
typename std::enable_if_t<
    ext::oneapi::experimental::is_fixed_topology_group_v<Group>>
ControlBarrier(Group, memory_scope FenceScope, memory_order Order) {
  __spirv_ControlBarrier(group_scope<Group>::value, getScope(FenceScope),
                         getMemorySemanticsMask(Order) |
                             __spv::MemorySemanticsMask::SubgroupMemory |
                             __spv::MemorySemanticsMask::WorkgroupMemory |
                             __spv::MemorySemanticsMask::CrossWorkgroupMemory);
}

template <typename Group>
typename std::enable_if_t<
    ext::oneapi::experimental::is_user_constructed_group_v<Group>>
ControlBarrier(Group g, memory_scope FenceScope, memory_order Order) {
#if defined(__NVPTX__)
  __nvvm_bar_warp_sync(detail::ExtractMask(detail::GetMask(g))[0]);
#else
  (void)g;
  // SPIR-V does not define an instruction to synchronize partial groups.
  // However, most (possibly all?) of the current SPIR-V targets execute
  // work-items in lockstep, so we can probably get away with a MemoryBarrier.
  // TODO: Replace this if SPIR-V defines a NonUniformControlBarrier
  __spirv_MemoryBarrier(getScope(FenceScope),
                        getMemorySemanticsMask(Order) |
                            __spv::MemorySemanticsMask::SubgroupMemory |
                            __spv::MemorySemanticsMask::WorkgroupMemory |
                            __spv::MemorySemanticsMask::CrossWorkgroupMemory);
#endif
}

// TODO: Refactor to avoid duplication after design settles
#define __SYCL_GROUP_COLLECTIVE_OVERLOAD(Instruction, GroupExt)                \
  template <__spv::GroupOperation Op, typename Group, typename T>              \
  inline typename std::enable_if_t<                                            \
      ext::oneapi::experimental::is_fixed_topology_group_v<Group>, T>          \
  Group##Instruction(Group, T x) {                                             \
    using ConvertedT = detail::ConvertToOpenCLType_t<T>;                       \
                                                                               \
    using OCLT = std::conditional_t<                                           \
        std::is_same<ConvertedT, opencl::cl_char>() ||                         \
            std::is_same<ConvertedT, opencl::cl_short>(),                      \
        opencl::cl_int,                                                        \
        std::conditional_t<std::is_same<ConvertedT, opencl::cl_uchar>() ||     \
                               std::is_same<ConvertedT, opencl::cl_ushort>(),  \
                           opencl::cl_uint, ConvertedT>>;                      \
    OCLT Arg = x;                                                              \
    OCLT Ret = __spirv_Group##Instruction##GroupExt(                           \
        group_scope<Group>::value, static_cast<unsigned int>(Op), Arg);        \
    return Ret;                                                                \
  }

// Define separate macros for each template to avoid nesting issues
#define __SYCL_GROUP_COLLECTIVE_FRAGMENT(Instruction, GroupExt)                \
  template <__spv::GroupOperation Op, typename ParentGroup, typename T>        \
  inline T Group##Instruction(                                                 \
      ext::oneapi::experimental::fragment<ParentGroup> g, T x) {               \
    using ConvertedT = detail::ConvertToOpenCLType_t<T>;                       \
                                                                               \
    using OCLT = std::conditional_t<                                           \
        std::is_same<ConvertedT, opencl::cl_char>() ||                         \
            std::is_same<ConvertedT, opencl::cl_short>(),                      \
        opencl::cl_int,                                                        \
        std::conditional_t<std::is_same<ConvertedT, opencl::cl_uchar>() ||     \
                               std::is_same<ConvertedT, opencl::cl_ushort>(),  \
                           opencl::cl_uint, ConvertedT>>;                      \
    OCLT Arg = x;                                                              \
    /* fragment partitions its parent into two groups (0 and 1) */             \
    /* We have to force each group down different control flow */              \
    /* Work-items in the "false" group (0) may still be active */              \
    constexpr auto Scope = group_scope<ParentGroup>::value;                    \
    constexpr auto OpInt = static_cast<unsigned int>(Op);                      \
    if (g.get_group_id() == 1) {                                               \
      return __spirv_GroupNonUniform##Instruction(Scope, OpInt, Arg);          \
    } else {                                                                   \
      return __spirv_GroupNonUniform##Instruction(Scope, OpInt, Arg);          \
    }                                                                          \
    /* TODO: add support for partitioning into more than two groups */         \
  }

#define __SYCL_GROUP_COLLECTIVE_CHUNK(Instruction, GroupExt)                   \
  template <__spv::GroupOperation Op, size_t ChunkSize, typename ParentGroup,  \
            typename T>                                                        \
  inline T Group##Instruction(                                                 \
      ext::oneapi::experimental::chunk<ChunkSize, ParentGroup> g, T x) {       \
    using ConvertedT = detail::ConvertToOpenCLType_t<T>;                       \
                                                                               \
    using OCLT = std::conditional_t<                                           \
        std::is_same<ConvertedT, opencl::cl_char>() ||                         \
            std::is_same<ConvertedT, opencl::cl_short>(),                      \
        opencl::cl_int,                                                        \
        std::conditional_t<std::is_same<ConvertedT, opencl::cl_uchar>() ||     \
                               std::is_same<ConvertedT, opencl::cl_ushort>(),  \
                           opencl::cl_uint, ConvertedT>>;                      \
    OCLT Arg = x;                                                              \
    constexpr auto Scope = group_scope<ParentGroup>::value;                    \
                                                                               \
    /* SPIR-V only defines a ClusteredReduce, with no equivalents for scan. */ \
    /* Emulate Clustered*Scan using control flow to separate clusters.      */ \
    if constexpr (Op == __spv::GroupOperation::Reduce) {                       \
      constexpr auto OpInt =                                                   \
          static_cast<unsigned int>(__spv::GroupOperation::ClusteredReduce);   \
      return __spirv_GroupNonUniform##Instruction(Scope, OpInt, Arg,           \
                                                  ChunkSize);                  \
    } else {                                                                   \
      T tmp;                                                                   \
      for (size_t Cluster = 0; Cluster < g.get_group_linear_range();           \
           ++Cluster) {                                                        \
        if (Cluster == g.get_group_linear_id()) {                              \
          constexpr auto OpInt = static_cast<unsigned int>(Op);                \
          tmp = __spirv_GroupNonUniform##Instruction(Scope, OpInt, Arg);       \
        }                                                                      \
      }                                                                        \
      return tmp;                                                              \
    }                                                                          \
  }

#define __SYCL_GROUP_COLLECTIVE_TANGLE(Instruction, GroupExt)                  \
  template <__spv::GroupOperation Op, typename Group, typename T>              \
  inline typename std::enable_if_t<sycl::detail::is_tangle_v<Group>, T>        \
  Group##Instruction(Group, T x) {                                             \
    using ConvertedT = detail::ConvertToOpenCLType_t<T>;                       \
                                                                               \
    using OCLT = std::conditional_t<                                           \
        std::is_same<ConvertedT, opencl::cl_char>() ||                         \
            std::is_same<ConvertedT, opencl::cl_short>(),                      \
        opencl::cl_int,                                                        \
        std::conditional_t<std::is_same<ConvertedT, opencl::cl_uchar>() ||     \
                               std::is_same<ConvertedT, opencl::cl_ushort>(),  \
                           opencl::cl_uint, ConvertedT>>;                      \
    OCLT Arg = x;                                                              \
    OCLT Ret = __spirv_GroupNonUniform##Instruction(                           \
        group_scope<Group>::value, static_cast<unsigned int>(Op), Arg);        \
    return Ret;                                                                \
  }

// Now use all 4 macros together for each instruction
#define __SYCL_GROUP_COLLECTIVE_ALL(Instruction, GroupExt)                     \
  __SYCL_GROUP_COLLECTIVE_OVERLOAD(Instruction, GroupExt)                      \
  __SYCL_GROUP_COLLECTIVE_FRAGMENT(Instruction, GroupExt)                      \
  __SYCL_GROUP_COLLECTIVE_CHUNK(Instruction, GroupExt)                         \
  __SYCL_GROUP_COLLECTIVE_TANGLE(Instruction, GroupExt)

__SYCL_GROUP_COLLECTIVE_ALL(SMin, )
__SYCL_GROUP_COLLECTIVE_ALL(UMin, )
__SYCL_GROUP_COLLECTIVE_ALL(FMin, )

__SYCL_GROUP_COLLECTIVE_ALL(SMax, )
__SYCL_GROUP_COLLECTIVE_ALL(UMax, )
__SYCL_GROUP_COLLECTIVE_ALL(FMax, )

__SYCL_GROUP_COLLECTIVE_ALL(IAdd, )
__SYCL_GROUP_COLLECTIVE_ALL(FAdd, )

__SYCL_GROUP_COLLECTIVE_ALL(IMul, KHR)
__SYCL_GROUP_COLLECTIVE_ALL(FMul, KHR)
__SYCL_GROUP_COLLECTIVE_ALL(CMulINTEL, )

__SYCL_GROUP_COLLECTIVE_ALL(BitwiseOr, KHR)
__SYCL_GROUP_COLLECTIVE_ALL(BitwiseXor, KHR)
__SYCL_GROUP_COLLECTIVE_ALL(BitwiseAnd, KHR)

__SYCL_GROUP_COLLECTIVE_ALL(LogicalAnd, KHR)
__SYCL_GROUP_COLLECTIVE_ALL(LogicalOr, KHR)

#undef __SYCL_GROUP_COLLECTIVE_OVERLOAD
#undef __SYCL_GROUP_COLLECTIVE_FRAGMENT
#undef __SYCL_GROUP_COLLECTIVE_CHUNK
#undef __SYCL_GROUP_COLLECTIVE_TANGLE
#undef __SYCL_GROUP_COLLECTIVE_ALL

} // namespace spirv
} // namespace detail
} // namespace _V1
} // namespace sycl
#endif //  __SYCL_DEVICE_ONLY__
