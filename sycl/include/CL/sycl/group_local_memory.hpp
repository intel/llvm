//==----- group_local_memory.hpp --- SYCL group local memory extension -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <CL/__spirv/spirv_vars.hpp>
#include <sycl/ext/oneapi/group_local_memory.hpp>

#include <type_traits>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
template <typename T, typename Group>
__SYCL_DEPRECATED(
    "use sycl::ext::oneapi::group_local_memory_for_overwrite instead")
std::enable_if_t<
    std::is_trivially_destructible<T>::value && detail::is_group<Group>::value,
    multi_ptr<T, access::address_space::local_space>> __SYCL_ALWAYS_INLINE
    group_local_memory_for_overwrite(Group g) {
  return sycl::ext::oneapi::group_local_memory_for_overwrite<T, Group>(g);
}

template <typename T, typename Group, typename... Args>
__SYCL_DEPRECATED("use sycl::ext::oneapi::group_local_memory instead")
std::enable_if_t<
    std::is_trivially_destructible<T>::value && detail::is_group<Group>::value,
    multi_ptr<T, access::address_space::local_space>> __SYCL_ALWAYS_INLINE
    group_local_memory(Group g, Args &&... args) {
  return sycl::ext::oneapi::group_local_memory<T, Group, Args...>(
      g, std::forward<Args>(args)...);
}
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
