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
#include <CL/sycl/stl.hpp>

#include <cstddef>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

// Forward declaration
namespace detail {
class queue_impl;
}

class exception_list {
public:
  using value_type = exception_ptr_class;
  using reference = value_type &;
  using const_reference = const value_type &;
  using size_type = std::size_t;
  using iterator = vector_class<exception_ptr_class>::const_iterator;
  using const_iterator = vector_class<exception_ptr_class>::const_iterator;

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
  vector_class<exception_ptr_class> MList;
};

using async_handler = function_class<void(cl::sycl::exception_list)>;

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
