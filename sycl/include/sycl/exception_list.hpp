//==---------------- exception_list.hpp - SYCL exception_list --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

// 4.9.2 Exception Class Interface

#include <sycl/detail/defines.hpp>
#include <sycl/detail/export.hpp>
#include <sycl/detail/iostream_proxy.hpp>
#include <sycl/stl.hpp>

#include <cstddef>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {

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
  void PushBack(value_type &&Value);
  void Clear() noexcept;
  std::vector<std::exception_ptr> MList;
};

using async_handler = std::function<void(sycl::exception_list)>;

namespace detail {
// Default implementation of async_handler used by queue and context when no
// user-defined async_handler is specified.
inline void defaultAsyncHandler(exception_list Exceptions) {
  std::cerr << "Default async_handler caught exceptions:";
  for (auto &EIt : Exceptions) {
    try {
      if (EIt) {
        std::rethrow_exception(EIt);
      }
    } catch (const std::exception &E) {
      std::cerr << "\n\t" << E.what();
    }
  }
  std::cerr << std::endl;
  std::terminate();
}
} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
