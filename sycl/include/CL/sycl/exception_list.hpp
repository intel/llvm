//==---------------- exception_list.hpp - SYCL exception_list --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

// 4.9.2 Exception Class Interface

#include <CL/sycl/detail/defines.hpp>
#include <CL/sycl/detail/export.hpp>
#include <CL/sycl/stl.hpp>

#include <cstddef>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

// Forward declaration
namespace detail {
class queue_impl;
}

/// A list of asynchronous exceptions.
///
/// \ingroup sycl_api
class __SYCL_EXPORT exception_list {
public:
  using value_type = std::exception_ptr;
  using reference = value_type &;
  using const_reference = const value_type &;
  using size_type = std::size_t;
  using iterator = std::vector<std::exception_ptr>::const_iterator;
  using const_iterator = std::vector<std::exception_ptr>::const_iterator;

  size_type size() const;
  // first asynchronous exception
  iterator begin() const;
  // refer to past-the-end last asynchronous exception
  iterator end() const;

private:
  friend class detail::queue_impl;
  void PushBack(const_reference Value);
  void PushBack(value_type&& Value);
  void Clear() noexcept;
  std::vector<std::exception_ptr> MList;
};

using async_handler = std::function<void(cl::sycl::exception_list)>;

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
