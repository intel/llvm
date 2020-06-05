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
static inline constexpr __spv::MemorySemanticsMask::Flag
getMemorySemanticsMask(intel::memory_order order) {
  __spv::MemorySemanticsMask::Flag spv_order;
  switch (order) {
  case intel::memory_order::relaxed:
    spv_order = __spv::MemorySemanticsMask::None;
  case intel::memory_order::acquire:
    spv_order = __spv::MemorySemanticsMask::Acquire;
  case intel::memory_order::release:
    spv_order = __spv::MemorySemanticsMask::Release;
  case intel::memory_order::acq_rel:
    spv_order = __spv::MemorySemanticsMask::AcquireRelease;
  case intel::memory_order::seq_cst:
    spv_order = __spv::MemorySemanticsMask::SequentiallyConsistent;
  }
  return static_cast<__spv::MemorySemanticsMask::Flag>(
      spv_order | __spv::MemorySemanticsMask::SubgroupMemory |
      __spv::MemorySemanticsMask::WorkgroupMemory |
      __spv::MemorySemanticsMask::CrossWorkgroupMemory);
}

static inline constexpr __spv::Scope::Flag getScope(intel::memory_scope scope) {
  switch (scope) {
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
AtomicCompareExchange(multi_ptr<T, AddressSpace> mptr,
                      intel::memory_scope scope, intel::memory_order success,
                      intel::memory_order failure, T desired, T expected) {
  auto spirv_success = getMemorySemanticsMask(success);
  auto spirv_failure = getMemorySemanticsMask(failure);
  auto spirv_scope = getScope(scope);
  auto *ptr = mptr.get();
  return __spirv_AtomicCompareExchange(ptr, spirv_scope, spirv_success,
                                       spirv_failure, desired, expected);
}

template <typename T, access::address_space AddressSpace>
inline typename detail::enable_if_t<std::is_floating_point<T>::value, T>
AtomicCompareExchange(multi_ptr<T, AddressSpace> mptr,
                      intel::memory_scope scope, intel::memory_order success,
                      intel::memory_order failure, T desired, T expected) {
  using I = detail::make_unsinged_integer_t<T>;
  auto spirv_success = getMemorySemanticsMask(success);
  auto spirv_failure = getMemorySemanticsMask(failure);
  auto spirv_scope = getScope(scope);
  auto *ptr_int =
      reinterpret_cast<typename multi_ptr<I, AddressSpace>::pointer_t>(
          mptr.get());
  I desired_int = detail::bit_cast<I>(desired);
  I expected_int = detail::bit_cast<I>(expected);
  I result_int =
      __spirv_AtomicCompareExchange(ptr_int, spirv_scope, spirv_success,
                                    spirv_failure, desired_int, expected_int);
  return detail::bit_cast<T>(result_int);
}

template <typename T, access::address_space AddressSpace>
inline typename detail::enable_if_t<std::is_integral<T>::value, T>
AtomicLoad(multi_ptr<T, AddressSpace> mptr, intel::memory_scope scope,
           intel::memory_order order) {
  auto *ptr = mptr.get();
  auto spirv_order = getMemorySemanticsMask(order);
  auto spirv_scope = getScope(scope);
  return __spirv_AtomicLoad(ptr, spirv_scope, spirv_order);
}

template <typename T, access::address_space AddressSpace>
inline typename detail::enable_if_t<std::is_floating_point<T>::value, T>
AtomicLoad(multi_ptr<T, AddressSpace> mptr, intel::memory_scope scope,
           intel::memory_order order) {
  using I = detail::make_unsinged_integer_t<T>;
  auto *ptr_int =
      reinterpret_cast<typename multi_ptr<I, AddressSpace>::pointer_t>(
          mptr.get());
  auto spirv_order = getMemorySemanticsMask(order);
  auto spirv_scope = getScope(scope);
  I result_int = __spirv_AtomicLoad(ptr_int, spirv_scope, spirv_order);
  return detail::bit_cast<T>(result_int);
}

template <typename T, access::address_space AddressSpace>
inline typename detail::enable_if_t<std::is_integral<T>::value>
AtomicStore(multi_ptr<T, AddressSpace> mptr, intel::memory_scope scope,
            intel::memory_order order, T value) {
  auto *ptr = mptr.get();
  auto spirv_order = getMemorySemanticsMask(order);
  auto spirv_scope = getScope(scope);
  __spirv_AtomicStore(ptr, spirv_scope, spirv_order, value);
}

template <typename T, access::address_space AddressSpace>
inline typename detail::enable_if_t<std::is_floating_point<T>::value>
AtomicStore(multi_ptr<T, AddressSpace> mptr, intel::memory_scope scope,
            intel::memory_order order, T value) {
  using I = detail::make_unsinged_integer_t<T>;
  auto *ptr_int =
      reinterpret_cast<typename multi_ptr<I, AddressSpace>::pointer_t>(
          mptr.get());
  auto spirv_order = getMemorySemanticsMask(order);
  auto spirv_scope = getScope(scope);
  I value_int = detail::bit_cast<I>(value);
  __spirv_AtomicStore(ptr_int, spirv_scope, spirv_order, value_int);
}

template <typename T, access::address_space AddressSpace>
inline typename detail::enable_if_t<std::is_integral<T>::value, T>
AtomicExchange(multi_ptr<T, AddressSpace> mptr, intel::memory_scope scope,
               intel::memory_order order, T value) {
  auto *ptr = mptr.get();
  auto spirv_order = getMemorySemanticsMask(order);
  auto spirv_scope = getScope(scope);
  return __spirv_AtomicExchange(ptr, spirv_scope, spirv_order, value);
}

template <typename T, access::address_space AddressSpace>
inline typename detail::enable_if_t<std::is_floating_point<T>::value, T>
AtomicExchange(multi_ptr<T, AddressSpace> mptr, intel::memory_scope scope,
               intel::memory_order order, T value) {
  using I = detail::make_unsinged_integer_t<T>;
  auto *ptr_int =
      reinterpret_cast<typename multi_ptr<I, AddressSpace>::pointer_t>(
          mptr.get());
  auto spirv_order = getMemorySemanticsMask(order);
  auto spirv_scope = getScope(scope);
  I value_int = detail::bit_cast<I>(value);
  I result_int =
      __spirv_AtomicExchange(ptr_int, spirv_scope, spirv_order, value_int);
  return detail::bit_cast<T>(result_int);
}

template <typename T, access::address_space AddressSpace>
inline typename detail::enable_if_t<std::is_integral<T>::value, T>
AtomicIAdd(multi_ptr<T, AddressSpace> mptr, intel::memory_scope scope,
           intel::memory_order order, T value) {
  auto *ptr = mptr.get();
  auto spirv_order = getMemorySemanticsMask(order);
  auto spirv_scope = getScope(scope);
  return __spirv_AtomicIAdd(ptr, spirv_scope, spirv_order, value);
}

template <typename T, access::address_space AddressSpace>
inline typename detail::enable_if_t<std::is_integral<T>::value, T>
AtomicISub(multi_ptr<T, AddressSpace> mptr, intel::memory_scope scope,
           intel::memory_order order, T value) {
  auto *ptr = mptr.get();
  auto spirv_order = getMemorySemanticsMask(order);
  auto spirv_scope = getScope(scope);
  return __spirv_AtomicISub(ptr, spirv_scope, spirv_order, value);
}

template <typename T, access::address_space AddressSpace>
inline typename detail::enable_if_t<std::is_integral<T>::value, T>
AtomicAnd(multi_ptr<T, AddressSpace> mptr, intel::memory_scope scope,
          intel::memory_order order, T value) {
  auto *ptr = mptr.get();
  auto spirv_order = getMemorySemanticsMask(order);
  auto spirv_scope = getScope(scope);
  return __spirv_AtomicAnd(ptr, spirv_scope, spirv_order, value);
}

template <typename T, access::address_space AddressSpace>
inline typename detail::enable_if_t<std::is_integral<T>::value, T>
AtomicOr(multi_ptr<T, AddressSpace> mptr, intel::memory_scope scope,
         intel::memory_order order, T value) {
  auto *ptr = mptr.get();
  auto spirv_order = getMemorySemanticsMask(order);
  auto spirv_scope = getScope(scope);
  return __spirv_AtomicOr(ptr, spirv_scope, spirv_order, value);
}

template <typename T, access::address_space AddressSpace>
inline typename detail::enable_if_t<std::is_integral<T>::value, T>
AtomicXor(multi_ptr<T, AddressSpace> mptr, intel::memory_scope scope,
          intel::memory_order order, T value) {
  auto *ptr = mptr.get();
  auto spirv_order = getMemorySemanticsMask(order);
  auto spirv_scope = getScope(scope);
  return __spirv_AtomicXor(ptr, spirv_scope, spirv_order, value);
}

template <typename T, access::address_space AddressSpace>
inline typename detail::enable_if_t<std::is_integral<T>::value, T>
AtomicMin(multi_ptr<T, AddressSpace> mptr, intel::memory_scope scope,
          intel::memory_order order, T value) {
  auto *ptr = mptr.get();
  auto spirv_order = getMemorySemanticsMask(order);
  auto spirv_scope = getScope(scope);
  return __spirv_AtomicMin(ptr, spirv_scope, spirv_order, value);
}

template <typename T, access::address_space AddressSpace>
inline typename detail::enable_if_t<std::is_integral<T>::value, T>
AtomicMax(multi_ptr<T, AddressSpace> mptr, intel::memory_scope scope,
          intel::memory_order order, T value) {
  auto *ptr = mptr.get();
  auto spirv_order = getMemorySemanticsMask(order);
  auto spirv_scope = getScope(scope);
  return __spirv_AtomicMax(ptr, spirv_scope, spirv_order, value);
}

} // namespace spirv
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
#endif //  __SYCL_DEVICE_ONLY__
