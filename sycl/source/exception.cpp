//==---------------- exception.cpp - SYCL exception ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// 4.9.2 Exception Class Interface
#include <detail/global_handler.hpp>
#include <sycl/context.hpp>
#include <sycl/exception.hpp>

#include <cstring>
#include <sstream>

namespace sycl {
inline namespace _V1 {

// protected base constructor for all SYCL 2020 constructors
exception::exception(std::error_code EC, std::shared_ptr<context> SharedPtrCtx,
                     const char *WhatArg)
    : MMsg(std::make_shared<detail::string>(WhatArg)),
      MErr(UR_RESULT_ERROR_INVALID_VALUE), MContext(SharedPtrCtx), MErrC(EC) {
  detail::GlobalHandler::instance().TraceEventXPTI(MMsg->c_str());
}

exception::~exception() {}

const std::error_code &exception::code() const noexcept { return MErrC; }

const std::error_category &exception::category() const noexcept {
  return code().category();
}

const char *exception::what() const noexcept { return MMsg->c_str(); }

bool exception::has_context() const noexcept { return (MContext != nullptr); }

context exception::get_context() const {
  if (!has_context())
    throw sycl::exception(sycl::errc::invalid);

  return *MContext;
}

const std::error_category &sycl_category() noexcept {
  static const detail::SYCLCategory SYCLCategoryObj;
  return SYCLCategoryObj;
}

std::error_code make_error_code(sycl::errc Err) noexcept {
  return {static_cast<int>(Err), sycl_category()};
}

} // namespace _V1
} // namespace sycl
