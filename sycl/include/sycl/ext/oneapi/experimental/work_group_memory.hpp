//===-------------------- work_group_memory.hpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <type_traits>

namespace sycl {
inline namespace _V1 {
namespace detail {
template <typename T> struct is_unbounded_array : std::false_type {};

template <typename T> struct is_unbounded_array<T[]> : std::true_type {};

template <typename T>
inline constexpr bool is_unbounded_array_v = is_unbounded_array<T>::value;

class work_group_memory_impl {
public:
  work_group_memory_impl() : buffer_size{0} {}
  work_group_memory_impl(const work_group_memory_impl &rhs) = default;
  work_group_memory_impl &
  operator=(const work_group_memory_impl &rhs) = default;
  work_group_memory_impl(size_t buffer_size) : buffer_size{buffer_size} {}

private:
  size_t buffer_size;
  friend class sycl::handler;
};

} // namespace detail
namespace ext::oneapi::experimental {

template <typename DataT, typename PropertyListT = empty_properties_t>
class __SYCL_SPECIAL_CLASS __SYCL_TYPE(work_group_memory) work_group_memory
    : sycl::detail::work_group_memory_impl {
public:
  using value_type = std::remove_all_extents_t<DataT>;

private:
  using decoratedPtr = typename sycl::detail::DecoratedType<
      value_type, access::address_space::local_space>::type *;

public:
  work_group_memory() = default;
  work_group_memory(const work_group_memory &rhs) = default;
  work_group_memory &operator=(const work_group_memory &rhs) = default;
  template <typename T = DataT,
            typename = std::enable_if_t<!sycl::detail::is_unbounded_array_v<T>>>
  work_group_memory(handler &)
      : sycl::detail::work_group_memory_impl(sizeof(DataT)) {}
  template <typename T = DataT,
            typename = std::enable_if_t<sycl::detail::is_unbounded_array_v<T>>>
  work_group_memory(size_t num, handler &)
      : sycl::detail::work_group_memory_impl(
            num * sizeof(std::remove_extent_t<DataT>)) {}
  template <access::decorated IsDecorated = access::decorated::no>
  multi_ptr<value_type, access::address_space::local_space, IsDecorated>
  get_multi_ptr() const {
    return sycl::address_space_cast<access::address_space::local_space,
                                    IsDecorated, value_type>(ptr);
  }
  DataT *operator&() const { return reinterpret_cast<DataT *>(ptr); }
  operator DataT &() const { return *reinterpret_cast<DataT *>(ptr); }
  template <typename T = DataT,
            typename = std::enable_if_t<!std::is_array_v<T>>>
  const work_group_memory &operator=(const DataT &value) const {
    *ptr = value;
    return *this;
  }
#ifdef __SYCL_DEVICE_ONLY__
  void __init(decoratedPtr ptr) { this->ptr = ptr; }
#endif
private:
  decoratedPtr ptr;
};
} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
