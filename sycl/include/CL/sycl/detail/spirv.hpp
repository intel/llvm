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
#include <CL/sycl/detail/generic_type_traits.hpp>
#include <CL/sycl/detail/type_traits.hpp>
#include <CL/sycl/intel/atomic_enums.hpp>

#ifdef __SYCL_DEVICE_ONLY__
__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace intel {
struct sub_group;
} // namespace intel
namespace detail {
namespace spirv {

template <typename Group> struct group_scope {};

template <int Dimensions> struct group_scope<group<Dimensions>> {
  static constexpr __spv::Scope::Flag value = __spv::Scope::Flag::Workgroup;
};

template <> struct group_scope<::cl::sycl::intel::sub_group> {
  static constexpr __spv::Scope::Flag value = __spv::Scope::Flag::Subgroup;
};

template <typename Group> bool GroupAll(bool pred) {
  return __spirv_GroupAll(group_scope<Group>::value, pred);
}

template <typename Group> bool GroupAny(bool pred) {
  return __spirv_GroupAny(group_scope<Group>::value, pred);
}

// Broadcast with scalar local index
template <typename Group, typename T, typename IdT>
detail::enable_if_t<std::is_integral<IdT>::value, T>
GroupBroadcast(T x, IdT local_id) {
  using OCLT = detail::ConvertToOpenCLType_t<T>;
  using OCLIdT = detail::ConvertToOpenCLType_t<IdT>;
  OCLT ocl_x = detail::convertDataToType<T, OCLT>(x);
  OCLIdT ocl_id = detail::convertDataToType<IdT, OCLIdT>(local_id);
  return __spirv_GroupBroadcast(group_scope<Group>::value, ocl_x, ocl_id);
}

// Broadcast with vector local index
template <typename Group, typename T, int Dimensions>
T GroupBroadcast(T x, id<Dimensions> local_id) {
  if (Dimensions == 1) {
    return GroupBroadcast<Group>(x, local_id[0]);
  }
  using IdT = vec<size_t, Dimensions>;
  using OCLT = detail::ConvertToOpenCLType_t<T>;
  using OCLIdT = detail::ConvertToOpenCLType_t<IdT>;
  IdT vec_id;
  for (int i = 0; i < Dimensions; ++i) {
    vec_id[i] = local_id[Dimensions - i - 1];
  }
  OCLT ocl_x = detail::convertDataToType<T, OCLT>(x);
  OCLIdT ocl_id = detail::convertDataToType<IdT, OCLIdT>(vec_id);
  return __spirv_GroupBroadcast(group_scope<Group>::value, ocl_x, ocl_id);
}

// Single happens-before means semantics should always apply to all spaces
// Although consume is unsupported, forwarding to acquire is valid
static inline constexpr __spv::MemorySemanticsMask::Flag
getMemorySemanticsMask(intel::memory_order Order) {
  __spv::MemorySemanticsMask::Flag SpvOrder = __spv::MemorySemanticsMask::None;
  switch (Order) {
  case intel::memory_order::relaxed:
    SpvOrder = __spv::MemorySemanticsMask::None;
    break;
  case intel::memory_order::__consume_unsupported:
  case intel::memory_order::acquire:
    SpvOrder = __spv::MemorySemanticsMask::Acquire;
    break;
  case intel::memory_order::release:
    SpvOrder = __spv::MemorySemanticsMask::Release;
    break;
  case intel::memory_order::acq_rel:
    SpvOrder = __spv::MemorySemanticsMask::AcquireRelease;
    break;
  case intel::memory_order::seq_cst:
    SpvOrder = __spv::MemorySemanticsMask::SequentiallyConsistent;
    break;
  }
  return static_cast<__spv::MemorySemanticsMask::Flag>(
      SpvOrder | __spv::MemorySemanticsMask::SubgroupMemory |
      __spv::MemorySemanticsMask::WorkgroupMemory |
      __spv::MemorySemanticsMask::CrossWorkgroupMemory);
}

static inline constexpr __spv::Scope::Flag getScope(intel::memory_scope Scope) {
  switch (Scope) {
  case intel::memory_scope::work_item:
    return __spv::Scope::Invocation;
  case intel::memory_scope::sub_group:
    return __spv::Scope::Subgroup;
  case intel::memory_scope::work_group:
    return __spv::Scope::Workgroup;
  case intel::memory_scope::device:
    return __spv::Scope::Device;
  case intel::memory_scope::system:
    return __spv::Scope::CrossDevice;
  }
}

template <typename T, access::address_space AddressSpace>
inline typename detail::enable_if_t<std::is_integral<T>::value, T>
AtomicCompareExchange(multi_ptr<T, AddressSpace> MPtr,
                      intel::memory_scope Scope, intel::memory_order Success,
                      intel::memory_order Failure, T Desired, T Expected) {
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
                      intel::memory_scope Scope, intel::memory_order Success,
                      intel::memory_order Failure, T Desired, T Expected) {
  using I = detail::make_unsinged_integer_t<T>;
  auto SPIRVSuccess = getMemorySemanticsMask(Success);
  auto SPIRVFailure = getMemorySemanticsMask(Failure);
  auto SPIRVScope = getScope(Scope);
  auto *PtrInt =
      reinterpret_cast<typename multi_ptr<I, AddressSpace>::pointer_t>(
          MPtr.get());
  I DesiredInt = detail::bit_cast<I>(Desired);
  I ExpectedInt = detail::bit_cast<I>(Expected);
  I ResultInt = __spirv_AtomicCompareExchange(
      PtrInt, SPIRVScope, SPIRVSuccess, SPIRVFailure, DesiredInt, ExpectedInt);
  return detail::bit_cast<T>(ResultInt);
}

template <typename T, access::address_space AddressSpace>
inline typename detail::enable_if_t<std::is_integral<T>::value, T>
AtomicLoad(multi_ptr<T, AddressSpace> MPtr, intel::memory_scope Scope,
           intel::memory_order Order) {
  auto *Ptr = MPtr.get();
  auto SPIRVOrder = getMemorySemanticsMask(Order);
  auto SPIRVScope = getScope(Scope);
  return __spirv_AtomicLoad(Ptr, SPIRVScope, SPIRVOrder);
}

template <typename T, access::address_space AddressSpace>
inline typename detail::enable_if_t<std::is_floating_point<T>::value, T>
AtomicLoad(multi_ptr<T, AddressSpace> MPtr, intel::memory_scope Scope,
           intel::memory_order Order) {
  using I = detail::make_unsinged_integer_t<T>;
  auto *PtrInt =
      reinterpret_cast<typename multi_ptr<I, AddressSpace>::pointer_t>(
          MPtr.get());
  auto SPIRVOrder = getMemorySemanticsMask(Order);
  auto SPIRVScope = getScope(Scope);
  I ResultInt = __spirv_AtomicLoad(PtrInt, SPIRVScope, SPIRVOrder);
  return detail::bit_cast<T>(ResultInt);
}

template <typename T, access::address_space AddressSpace>
inline typename detail::enable_if_t<std::is_integral<T>::value>
AtomicStore(multi_ptr<T, AddressSpace> MPtr, intel::memory_scope Scope,
            intel::memory_order Order, T Value) {
  auto *Ptr = MPtr.get();
  auto SPIRVOrder = getMemorySemanticsMask(Order);
  auto SPIRVScope = getScope(Scope);
  __spirv_AtomicStore(Ptr, SPIRVScope, SPIRVOrder, Value);
}

template <typename T, access::address_space AddressSpace>
inline typename detail::enable_if_t<std::is_floating_point<T>::value>
AtomicStore(multi_ptr<T, AddressSpace> MPtr, intel::memory_scope Scope,
            intel::memory_order Order, T Value) {
  using I = detail::make_unsinged_integer_t<T>;
  auto *PtrInt =
      reinterpret_cast<typename multi_ptr<I, AddressSpace>::pointer_t>(
          MPtr.get());
  auto SPIRVOrder = getMemorySemanticsMask(Order);
  auto SPIRVScope = getScope(Scope);
  I ValueInt = detail::bit_cast<I>(Value);
  __spirv_AtomicStore(PtrInt, SPIRVScope, SPIRVOrder, ValueInt);
}

template <typename T, access::address_space AddressSpace>
inline typename detail::enable_if_t<std::is_integral<T>::value, T>
AtomicExchange(multi_ptr<T, AddressSpace> MPtr, intel::memory_scope Scope,
               intel::memory_order Order, T Value) {
  auto *Ptr = MPtr.get();
  auto SPIRVOrder = getMemorySemanticsMask(Order);
  auto SPIRVScope = getScope(Scope);
  return __spirv_AtomicExchange(Ptr, SPIRVScope, SPIRVOrder, Value);
}

template <typename T, access::address_space AddressSpace>
inline typename detail::enable_if_t<std::is_floating_point<T>::value, T>
AtomicExchange(multi_ptr<T, AddressSpace> MPtr, intel::memory_scope Scope,
               intel::memory_order Order, T Value) {
  using I = detail::make_unsinged_integer_t<T>;
  auto *PtrInt =
      reinterpret_cast<typename multi_ptr<I, AddressSpace>::pointer_t>(
          MPtr.get());
  auto SPIRVOrder = getMemorySemanticsMask(Order);
  auto SPIRVScope = getScope(Scope);
  I ValueInt = detail::bit_cast<I>(Value);
  I ResultInt =
      __spirv_AtomicExchange(PtrInt, SPIRVScope, SPIRVOrder, ValueInt);
  return detail::bit_cast<T>(ResultInt);
}

template <typename T, access::address_space AddressSpace>
inline typename detail::enable_if_t<std::is_integral<T>::value, T>
AtomicIAdd(multi_ptr<T, AddressSpace> MPtr, intel::memory_scope Scope,
           intel::memory_order Order, T Value) {
  auto *Ptr = MPtr.get();
  auto SPIRVOrder = getMemorySemanticsMask(Order);
  auto SPIRVScope = getScope(Scope);
  return __spirv_AtomicIAdd(Ptr, SPIRVScope, SPIRVOrder, Value);
}

template <typename T, access::address_space AddressSpace>
inline typename detail::enable_if_t<std::is_integral<T>::value, T>
AtomicISub(multi_ptr<T, AddressSpace> MPtr, intel::memory_scope Scope,
           intel::memory_order Order, T Value) {
  auto *Ptr = MPtr.get();
  auto SPIRVOrder = getMemorySemanticsMask(Order);
  auto SPIRVScope = getScope(Scope);
  return __spirv_AtomicISub(Ptr, SPIRVScope, SPIRVOrder, Value);
}

template <typename T, access::address_space AddressSpace>
inline typename detail::enable_if_t<std::is_integral<T>::value, T>
AtomicAnd(multi_ptr<T, AddressSpace> MPtr, intel::memory_scope Scope,
          intel::memory_order Order, T Value) {
  auto *Ptr = MPtr.get();
  auto SPIRVOrder = getMemorySemanticsMask(Order);
  auto SPIRVScope = getScope(Scope);
  return __spirv_AtomicAnd(Ptr, SPIRVScope, SPIRVOrder, Value);
}

template <typename T, access::address_space AddressSpace>
inline typename detail::enable_if_t<std::is_integral<T>::value, T>
AtomicOr(multi_ptr<T, AddressSpace> MPtr, intel::memory_scope Scope,
         intel::memory_order Order, T Value) {
  auto *Ptr = MPtr.get();
  auto SPIRVOrder = getMemorySemanticsMask(Order);
  auto SPIRVScope = getScope(Scope);
  return __spirv_AtomicOr(Ptr, SPIRVScope, SPIRVOrder, Value);
}

template <typename T, access::address_space AddressSpace>
inline typename detail::enable_if_t<std::is_integral<T>::value, T>
AtomicXor(multi_ptr<T, AddressSpace> MPtr, intel::memory_scope Scope,
          intel::memory_order Order, T Value) {
  auto *Ptr = MPtr.get();
  auto SPIRVOrder = getMemorySemanticsMask(Order);
  auto SPIRVScope = getScope(Scope);
  return __spirv_AtomicXor(Ptr, SPIRVScope, SPIRVOrder, Value);
}

template <typename T, access::address_space AddressSpace>
inline typename detail::enable_if_t<std::is_integral<T>::value, T>
AtomicMin(multi_ptr<T, AddressSpace> MPtr, intel::memory_scope Scope,
          intel::memory_order Order, T Value) {
  auto *Ptr = MPtr.get();
  auto SPIRVOrder = getMemorySemanticsMask(Order);
  auto SPIRVScope = getScope(Scope);
  return __spirv_AtomicMin(Ptr, SPIRVScope, SPIRVOrder, Value);
}

template <typename T, access::address_space AddressSpace>
inline typename detail::enable_if_t<std::is_integral<T>::value, T>
AtomicMax(multi_ptr<T, AddressSpace> MPtr, intel::memory_scope Scope,
          intel::memory_order Order, T Value) {
  auto *Ptr = MPtr.get();
  auto SPIRVOrder = getMemorySemanticsMask(Order);
  auto SPIRVScope = getScope(Scope);
  return __spirv_AtomicMax(Ptr, SPIRVScope, SPIRVOrder, Value);
}

} // namespace spirv
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
#endif //  __SYCL_DEVICE_ONLY__
