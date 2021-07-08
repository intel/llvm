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

#include <cstring>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

namespace { // anonymous
char reserved_for_errorcode[] =
    "0SYCL1234567812345678"; // 17 (string terminator plus 'SYCL' marker plus
                             // error code)
std::error_code sycl121_proxy_errorcode = make_error_code(sycl::errc::invalid);
} // namespace

exception::exception(std::error_code ec, const char *Msg)
    : exception(static_cast<context *>(nullptr), ec, Msg) {}

exception::exception(std::error_code ec, const std::string &Msg)
    : exception(static_cast<context *>(nullptr), ec, Msg) {}

// new SYCL2020 constructors
exception::exception(std::error_code ec)
    : exception(static_cast<context *>(nullptr), ec, "") {}

exception::exception(int ev, const std::error_category &ecat,
                     const std::string &what_arg)
    : exception(static_cast<context *>(nullptr), {ev, ecat}, what_arg) {}

exception::exception(int ev, const std::error_category &ecat,
                     const char *what_arg)
    : exception(static_cast<context *>(nullptr), {ev, ecat},
                std::string(what_arg)) {}

exception::exception(int ev, const std::error_category &ecat)
    : exception(static_cast<context *>(nullptr), {ev, ecat}, "") {}

exception::exception(context ctx, std::error_code ec,
                     const std::string &what_arg)
    : exception(&ctx, ec, what_arg) {}

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

// protected base constructor for all SYCL2020 constructors
exception::exception(context *ctxPtr, std::error_code ec,
                     const std::string &what_arg)
    : MMsg(what_arg + reserved_for_errorcode), MCLErr(PI_INVALID_VALUE),
      MContext(ctxPtr) {
  // For compatibility with previous implementation, we are "hiding" the
  // std:::error_code in the MMsg string, behind the null string terminator
  size_t whatLen = what_arg.length();
  char *reservedPtr = &MMsg[whatLen];
  reservedPtr[0] = '\0';
  reservedPtr++;
  // insert 'SYCL' marker.
  reservedPtr[0] = 'S';
  reservedPtr[1] = 'Y';
  reservedPtr[2] = 'C';
  reservedPtr[3] = 'L';
  reservedPtr += 4;
  // insert error code
  std::error_code *ecPtr = reinterpret_cast<std::error_code *>(reservedPtr);
  *ecPtr = ec;
}

const std::error_code &exception::code() const noexcept {
  const char *whatStr = MMsg.c_str();
  size_t whatLen = strlen(whatStr);
  // advance past null string-terminator
  const char *reservedPtr = &whatStr[whatLen + 1];
  // check for 'SYCL' marker, which denotes a SYCL2020 exception
  if (reservedPtr[0] == 'S' && reservedPtr[1] == 'Y' && reservedPtr[2] == 'C' &&
      reservedPtr[3] == 'L') {
    reservedPtr += 4;
    const std::error_code *ecPtr =
        reinterpret_cast<const std::error_code *>(reservedPtr);
    return *ecPtr;
  } else {
    // the exception originates from some SYCL 1.2.1 source
    return sycl121_proxy_errorcode;
  }
}

const std::error_category &exception::category() const noexcept {
  return code().category();
}

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
