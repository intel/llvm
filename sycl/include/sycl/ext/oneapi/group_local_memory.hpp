//==----- group_local_memory.hpp --- SYCL group local memory extension -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <sycl/access/access.hpp>             // for address_space, decorated
#include <sycl/detail/defines_elementary.hpp> // for __SYCL_ALWAYS_INLINE
#include <sycl/detail/type_traits.hpp>        // for is_group
#include <sycl/exception.hpp>                 // for exception
#include <sycl/ext/intel/usm_pointers.hpp>    // for multi_ptr
#include <sycl/group.hpp>                     // for workGroupBarrier

#include <type_traits> // for enable_if_t

#ifdef __SYCL_DEVICE_ONLY__
// Request a fixed-size allocation in local address space at kernel scope.
extern "C" __DPCPP_SYCL_EXTERNAL __attribute__((opencl_local)) std::uint8_t *
__sycl_allocateLocalMemory(std::size_t Size, std::size_t Alignment);
#endif

namespace sycl {
inline namespace _V1 {
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
  throw sycl::exception(
      sycl::errc::feature_not_supported,
      "sycl_ext_oneapi_local_memory extension is not supported on host");
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
  throw sycl::exception(
      sycl::errc::feature_not_supported,
      "sycl_ext_oneapi_local_memory extension is not supported on host");
#endif
}
} // namespace ext::oneapi
} // namespace _V1
} // namespace sycl
