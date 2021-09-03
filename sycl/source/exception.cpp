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
#include <execinfo.h>
#include <sstream>
#include <dlfcn.h>
#include <cxxabi.h>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

namespace { // anonymous
constexpr char ReservedForErrorcode[] =
    "01234567812345678"; // 17 (string terminator plus error code)
std::error_code SYCL121ProxyErrorcode = make_error_code(sycl::errc::invalid);
} // namespace

std::string GetBacktrace(int Skip) {
  static constexpr int BT_SIZE = 32;
  void *BT[BT_SIZE];
  char **Strings;
  char buf[1024];

  int BTSize = backtrace(BT, BT_SIZE);
  Strings = backtrace_symbols(BT, BTSize);
  if (Strings) {
    std::stringstream S;

    for (int Idx = Skip; Idx < BTSize; ++Idx) {
      Dl_info info;
      if (dladdr(BT[Idx], &info)) {
        char *demangled = NULL;
        int status;
        demangled = abi::__cxa_demangle(info.dli_sname, NULL, 0, &status);
        snprintf(buf, sizeof(buf), "%-3d %*p %s + %zd\n",
                 Idx, (int)(2 + sizeof(void*) * 2), BT[Idx],
                 status == 0 ? demangled : info.dli_sname,
                 (char *)BT[BTSize] - (char *)info.dli_saddr);
        free(demangled);
      } else {
        snprintf(buf, sizeof(buf), "%-3d %*p\n",
                 Idx, (int)(2 + sizeof(void*) * 2), BT[Idx]);
      }
      S << buf;

      snprintf(buf, sizeof(buf), "%s\n", Strings[Idx]);
      S << buf;
    }
    free(Strings);

    return S.str();
  }

  return "no bt_symbols got";
}

exception::exception(std::error_code EC, const char *Msg)
    : exception(EC, nullptr, Msg) {}

exception::exception(std::error_code EC, const std::string &Msg)
    : exception(EC, nullptr, Msg) {}

// new SYCL 2020 constructors
exception::exception(std::error_code EC) : exception(EC, nullptr, "") {}

exception::exception(int EV, const std::error_category &ECat,
                     const std::string &WhatArg)
    : exception({EV, ECat}, nullptr, WhatArg) {}

exception::exception(int EV, const std::error_category &ECat,
                     const char *WhatArg)
    : exception({EV, ECat}, nullptr, std::string(WhatArg)) {}

exception::exception(int EV, const std::error_category &ECat)
    : exception({EV, ECat}, nullptr, "") {}

exception::exception(context Ctx, std::error_code EC,
                     const std::string &WhatArg)
    : exception(EC, std::make_shared<context>(Ctx), WhatArg) {}

exception::exception(context Ctx, std::error_code EC, const char *WhatArg)
    : exception(Ctx, EC, std::string(WhatArg)) {}

exception::exception(context Ctx, std::error_code EC)
    : exception(Ctx, EC, "") {}

exception::exception(context Ctx, int EV, const std::error_category &ECat,
                     const char *WhatArg)
    : exception(Ctx, {EV, ECat}, std::string(WhatArg)) {}

exception::exception(context Ctx, int EV, const std::error_category &ECat,
                     const std::string &WhatArg)
    : exception(Ctx, {EV, ECat}, WhatArg) {}

exception::exception(context Ctx, int EV, const std::error_category &ECat)
    : exception(Ctx, EV, ECat, "") {}

// protected base constructor for all SYCL 2020 constructors
exception::exception(std::error_code EC, std::shared_ptr<context> SharedPtrCtx,
                     const std::string &WhatArg)
    : MMsg(WhatArg + "\n" + GetBacktrace() + ReservedForErrorcode), MCLErr(PI_INVALID_VALUE),
      MContext(SharedPtrCtx) {
  // For compatibility with previous implementation, we are "hiding" the
  // std::error_code in the MMsg string, behind the null string terminator
  const int StringTermPoint = MMsg.length() - strlen(ReservedForErrorcode);
  char *ReservedPtr = &MMsg[StringTermPoint];
  ReservedPtr[0] = '\0';
  ReservedPtr++;
  // insert error code
  std::error_code *ECPtr = reinterpret_cast<std::error_code *>(ReservedPtr);
  memcpy(ECPtr, &EC, sizeof(std::error_code));
}

const std::error_code &exception::code() const noexcept {
  const char *WhatStr = MMsg.c_str();
  // advance to inner string-terminator
  int StringTermPoint = MMsg.length() - strlen(ReservedForErrorcode);
  if (StringTermPoint >= 0) {
    const char *ReservedPtr = &WhatStr[StringTermPoint];
    // check for string terminator, which denotes a SYCL 2020 exception
    if (ReservedPtr[0] == '\0') {
      ReservedPtr++;
      const std::error_code *ECPtr =
          reinterpret_cast<const std::error_code *>(ReservedPtr);
      return *ECPtr;
    }
  }
  // else the exception originates from some SYCL 1.2.1 source
  return SYCL121ProxyErrorcode;
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
