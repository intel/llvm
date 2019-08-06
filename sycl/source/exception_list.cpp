//==---------------- exception_list.cpp - SYCL exception_list ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// 4.9.2 Exception Class Interface
#include <CL/sycl/exception_list.hpp>

#include <utility>

namespace cl {
namespace sycl {

exception_list::size_type exception_list::size() const { return MList.size(); }

exception_list::iterator exception_list::begin() const { return MList.begin(); }

exception_list::iterator exception_list::end() const { return MList.cend(); }

void exception_list::PushBack(const_reference Value) { MList.emplace_back(Value); }

void exception_list::PushBack(value_type&& Value) { MList.emplace_back(std::move(Value)); }

void exception_list::Clear() noexcept { MList.clear(); }

} // namespace sycl
} // namespace cl
