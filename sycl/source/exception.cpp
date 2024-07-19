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

namespace sycl {
inline namespace _V1 {

exception::exception(std::error_code EC, const char *Msg)
    : exception(EC, nullptr, Msg) {}

// new SYCL 2020 constructors
exception::exception(std::error_code EC) : exception(EC, nullptr, "") {}

exception::exception(int EV, const std::error_category &ECat,
                     const char *WhatArg)
    : exception({EV, ECat}, nullptr, std::string(WhatArg)) {}

exception::exception(int EV, const std::error_category &ECat)
    : exception({EV, ECat}, nullptr, "") {}

// protected base constructor for all SYCL 2020 constructors
exception::exception(std::error_code EC, std::shared_ptr<context> SharedPtrCtx,
                     const char *WhatArg)
    : MMsg(std::make_shared<detail::string>(WhatArg)),
      MPIErr(PI_ERROR_INVALID_VALUE), MContext(SharedPtrCtx), MErrC(EC) {
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

namespace detail {
__SYCL_EXPORT const char *stringifyErrorCode(pi_int32 error) {
  switch (error) {
#define _PI_ERRC(NAME, VAL)                                                    \
  case NAME:                                                                   \
    return #NAME;
#define _PI_ERRC_WITH_MSG(NAME, VAL, MSG)                                      \
  case NAME:                                                                   \
    return MSG;
#include <sycl/detail/pi_error.def>
#undef _PI_ERRC
#undef _PI_ERRC_WITH_MSG

  default:
    return "Unknown error code";
  }
}
} // namespace detail

} // namespace _V1
} // namespace sycl
