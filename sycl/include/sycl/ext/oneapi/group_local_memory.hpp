//==----- group_local_memory.hpp --- SYCL group local memory extension -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <CL/__spirv/spirv_vars.hpp>
#include <sycl/detail/defines_elementary.hpp>
#include <sycl/detail/sycl_fe_intrins.hpp>
#include <sycl/detail/type_traits.hpp>
#include <sycl/exception.hpp>
#include <sycl/group.hpp>
#include <sycl/multi_ptr.hpp>

#include <cstdint>
#include <type_traits>
#include <utility>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext::oneapi {
template <typename T, typename Group>
std::enable_if_t<
    std::is_trivially_destructible_v<T> && sycl::detail::is_group<Group>::value,
    multi_ptr<T, access::address_space::local_space, access::decorated::legacy>>
    __SYCL_ALWAYS_INLINE group_local_memory_for_overwrite(Group g) {
  (void)g;
#ifdef __SYCL_DEVICE_ONLY__
  __attribute__((opencl_local)) std::uint8_t *AllocatedMem =
      __sycl_allocateLocalMemory(sizeof(T), alignof(T));
  // If the type is non-trivial we need to default initialize it.
  if constexpr (!std::is_trivial_v<T>) {
    if (g.get_local_linear_id() == 0)
      new (AllocatedMem) T; // Default initialize.
    sycl::detail::workGroupBarrier();
  }
  return reinterpret_cast<__attribute__((opencl_local)) T *>(AllocatedMem);
#else
  throw feature_not_supported(
      "sycl_ext_oneapi_local_memory extension is not supported on host device",
      PI_ERROR_INVALID_OPERATION);
#endif
}

template <typename T, typename Group, typename... Args>
std::enable_if_t<
    std::is_trivially_destructible_v<T> && sycl::detail::is_group<Group>::value,
    multi_ptr<T, access::address_space::local_space, access::decorated::legacy>>
    __SYCL_ALWAYS_INLINE group_local_memory(Group g, Args &&...args) {
#ifdef __SYCL_DEVICE_ONLY__
  __attribute__((opencl_local)) std::uint8_t *AllocatedMem =
      __sycl_allocateLocalMemory(sizeof(T), alignof(T));
  if (g.get_local_linear_id() == 0)
    new (AllocatedMem) T{std::forward<Args>(args)...};
  sycl::detail::workGroupBarrier();
  return reinterpret_cast<__attribute__((opencl_local)) T *>(AllocatedMem);
#else
  // Silence unused variable warning
  (void)g;
  [&args...] {}();
  throw feature_not_supported(
      "sycl_ext_oneapi_local_memory extension is not supported on host device",
      PI_ERROR_INVALID_OPERATION);
#endif
}
} // namespace ext::oneapi
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
