//==---------------- exception_list.cpp - SYCL exception_list ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// 4.9.2 Exception Class Interface
#include <sycl/exception_list.hpp>

#include <iostream>
#include <sycl/detail/export.hpp>
#include <utility>

namespace sycl {
inline namespace _V1 {

exception_list::size_type exception_list::size() const { return MList.size(); }

exception_list::iterator exception_list::begin() const { return MList.begin(); }

exception_list::iterator exception_list::end() const { return MList.cend(); }

void exception_list::PushBack(const_reference Value) {
  MList.emplace_back(Value);
}

void exception_list::PushBack(value_type &&Value) {
  MList.emplace_back(std::move(Value));
}

void exception_list::Clear() noexcept { MList.clear(); }

namespace detail {
// Default implementation of async_handler used by queue and context when no
// user-defined async_handler is specified.
void defaultAsyncHandler(exception_list Exceptions) {
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

} // namespace _V1
} // namespace sycl
