//==---------------- exception.cpp - SYCL exception ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// 4.9.2 Exception Class Interface
#include <CL/sycl/context.hpp>
#include <CL/sycl/exception.hpp>

#include <utility>

namespace cl {
namespace sycl {

const char *exception::what() const noexcept { return MMsg.c_str(); }

bool exception::has_context() const { return (MContext != nullptr); }

context exception::get_context() const {
  if (!has_context())
    throw invalid_object_error();

  return *MContext;
}

cl_int exception::get_cl_code() const { return MCLErr; }

exception_list::size_type exception_list::size() const { return MList.size(); }

exception_list::iterator exception_list::begin() const { return MList.begin(); }

exception_list::iterator exception_list::end() const { return MList.cend(); }

void exception_list::PushBack(const_reference Value) { MList.emplace_back(Value); }

void exception_list::PushBack(value_type&& Value) { MList.emplace_back(std::move(Value)); }

void exception_list::Clear() noexcept { MList.clear(); }

} // namespace sycl
} // namespace cl
