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

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

// CP
namespace { // anonymous
char reserved_for_errorcode[1 + sizeof(std::error_code)];
}

// upper 6 go here

exception::exception(context ctx, std::error_code ec,
                     const std::string &what_arg)
    : MMsg(what_arg + reserved_for_errorcode), MCLErr(PI_INVALID_VALUE),
      MContext(&ctx) {
  // For compatibility with previous implementation, we are "hiding" the
  // std:::error_code in the MMsg string, behind the null string terminator
  size_t whatLen = what_arg.length();
  char *reservedPtr = &MMsg[whatLen];
  reservedPtr[0] = '\0';
  reservedPtr++;
  std::error_code *ecPtr = reinterpret_cast<std::error_code *>(reservedPtr);
  *ecPtr = ec; //{ev, ecat};
}

exception::exception(context ctx, std::error_code ec, const char *what_arg)
    : exception(ctx, ec, std::string(what_arg)) {}

exception::exception(context ctx, std::error_code ec)
    : exception(ctx, ec, "") {}

exception::exception(context ctx, int ev, const std::error_category &ecat,
                     const char *what_arg)
    : exception(ctx, {ev, ecat}, std::string(what_arg)) {}

exception::exception(context ctx, int ev, const std::error_category &ecat,
                     const std::string &what_arg)
    : exception(ctx, {ev, ecat}, what_arg) {}

exception::exception(context ctx, int ev, const std::error_category &ecat)
    : exception(ctx, ev, ecat, "") {}

const char *exception::what() const noexcept { return MMsg.c_str(); }

bool exception::has_context() const { return (MContext != nullptr); }

context exception::get_context() const {
  if (!has_context())
    throw invalid_object_error();

  return *MContext;
}

cl_int exception::get_cl_code() const { return MCLErr; }

const std::error_category &sycl_category() noexcept {
  static const detail::SYCLCategory SYCLCategoryObj;
  return SYCLCategoryObj;
}

std::error_code make_error_code(sycl::errc Err) noexcept {
  return {static_cast<int>(Err), sycl_category()};
}

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
