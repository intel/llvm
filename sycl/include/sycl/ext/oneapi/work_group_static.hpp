//==----- work_group_static.hpp --- SYCL group local memory extension -----==//
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

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi {
namespace experimental {

namespace detail {

#ifdef __SYCL_DEVICE_ONLY__
#define __SYCL_WG_SCOPE [[__sycl_detail__::wg_scope]]
#else
#define __SYCL_WG_SCOPE
#endif

template <typename T> class __SYCL_WG_SCOPE work_group_static {
public:
  // static_assert(std::is_unbounded_array<T>,
  //               "Use get_dynamic_work_group_memory for dynamic work group "
  //               "memory allocation");

  __SYCL_ALWAYS_INLINE work_group_static() = default;
  work_group_static(const work_group_static &) = delete;
  work_group_static &operator=(const work_group_static &) = delete;

  operator T &() const noexcept { return *getDecorated(); }

  template <class TArg = T>
  typename std::enable_if<!std::is_array_v<TArg>,
                          const work_group_static &>::type
  operator=(const T &value) const noexcept {
    *getDecorated() = value;
    return *this;
  }

  T *operator&() const noexcept { return *getDecorated(); }

private:
  using decorated_pointer =
#ifdef __SYCL_DEVICE_ONLY__
      __attribute__((opencl_local)) T *;
#else
      T *;
#endif

  // Small trick, memcpy of the class is UB so assume this is in the local
  // space. As the address space may get lost, explicitly cast it to this
  // address space to help the optimizer.
  decorated_pointer getDecorated() const { return (decorated_pointer)&data; }

  T data;
};

} // namespace detail

template <typename T> using work_group_static = detail::work_group_static<T>;

template <typename T>
std::enable_if_t<std::is_trivially_destructible_v<T> &&
                     std::is_trivially_constructible_v<T>,
                 multi_ptr<T, access::address_space::local_space,
                           access::decorated::legacy>> __SYCL_ALWAYS_INLINE
get_dynamic_work_group_memory() {
#ifdef __SYCL_DEVICE_ONLY__
  return reinterpret_cast<__attribute__((opencl_local)) T *>(
      __sycl_dynamicLocalMemoryPlaceholder(alignof(T)));
#else
  throw sycl::exception(
      sycl::errc::feature_not_supported,
      "sycl_ext_oneapi_work_group_static extension is not supported on host");
#endif
}

struct work_group_static_size
    : ::sycl::ext::oneapi::experimental::detail::run_time_property_key<
          ::sycl::ext::oneapi::experimental::detail::WorkGroupMem>,
      ::sycl::ext::oneapi::experimental::detail::compile_time_property_key<
          ::sycl::ext::oneapi::experimental::detail::WorkGroupMem>,
      property_value<work_group_static_size> {
  // Compile time property
  using value_t = property_value<work_group_static_size>;
  // Runtime property part
  constexpr work_group_static_size(size_t bytes) : size(bytes) {}

  size_t size;
};

using work_group_static_size_key = work_group_static_size;

// inline constexpr work_group_static_size_key::value_t work_group_static_size;

template <>
struct is_property_key<work_group_static_size_key> : std::true_type {};

template <typename T>
struct is_property_key_of<work_group_static_size_key, T> : std::true_type {};
template <>
struct is_property_value<work_group_static_size_key>
    : is_property_key<work_group_static_size_key> {};

namespace detail {
template <> struct PropertyMetaInfo<work_group_static_size_key> {
  static constexpr const char *name = "work-group-static";
  static constexpr int value = 1;
};

} // namespace detail

inline bool operator==(const work_group_static_size_key &lhs,
                       const work_group_static_size_key &rhs) {
  return lhs.size == rhs.size;
}
inline bool operator!=(const work_group_static_size_key &lhs,
                       const work_group_static_size_key &rhs) {
  return !(lhs == rhs);
}

} // namespace experimental
} // namespace ext::oneapi
} // namespace _V1
} // namespace sycl
