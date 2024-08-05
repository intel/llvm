//==---------------- exception.hpp - SYCL exception ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

// 4.9.2 Exception Class Interface

#include <sycl/detail/cl.h>                   // for cl_int
#include <sycl/detail/defines_elementary.hpp> // for __SYCL2020_DEPRECATED
#include <sycl/detail/export.hpp>             // for __SYCL_EXPORT
#include <sycl/detail/string.hpp>

#include <exception>    // for exception
#include <memory>       // for allocator, shared_ptr, make...
#include <string>       // for string, basic_string, opera...
#include <system_error> // for error_code, error_category
#include <type_traits>  // for true_type

namespace sycl {
inline namespace _V1 {

// Forward declaration
class context;
class exception;

enum class errc : unsigned int {
  success = 0,
  runtime = 1,
  kernel = 2,
  accessor = 3,
  nd_range = 4,
  event = 5,
  kernel_argument = 6,
  build = 7,
  invalid = 8,
  memory_allocation = 9,
  platform = 10,
  profiling = 11,
  feature_not_supported = 12,
  kernel_not_supported = 13,
  backend_mismatch = 14,
};

/// Constructs an error code using e and sycl_category()
__SYCL_EXPORT std::error_code make_error_code(sycl::errc E) noexcept;

__SYCL_EXPORT const std::error_category &sycl_category() noexcept;

namespace detail {
__SYCL_EXPORT const char *stringifyErrorCode(int32_t error);

inline std::string codeToString(int32_t code) {
  return std::to_string(code) + " (" + std::string(stringifyErrorCode(code)) + ")";
}

class __SYCL_EXPORT SYCLCategory : public std::error_category {
public:
  const char *name() const noexcept override { return "sycl"; }
  std::string message(int) const override { return "SYCL Error"; }
};

// Forward declare to declare as a friend in sycl::excepton.
int32_t get_ur_error(const exception &e);
exception set_ur_error(exception &&e, int32_t ur_err);
} // namespace detail

// Derive from std::exception so uncaught exceptions are printed in c++ default
// exception handler.
/// \ingroup sycl_api
class __SYCL_EXPORT exception : public virtual std::exception {
public:
  __SYCL2020_DEPRECATED("The version of an exception constructor which takes "
                        "no arguments is deprecated.")
  exception() = default;
  virtual ~exception();

  exception(std::error_code, const char *Msg);

  exception(std::error_code Ec, const std::string &Msg)
      : exception(Ec, nullptr, Msg.c_str()) {}

  // new SYCL 2020 constructors
  exception(std::error_code);
  exception(int EV, const std::error_category &ECat, const std::string &WhatArg)
      : exception(EV, ECat, WhatArg.c_str()) {}
  exception(int, const std::error_category &, const char *);
  exception(int, const std::error_category &);

  // context.hpp depends on exception.hpp but we can't define these ctors in
  // exception.hpp while context is still an incomplete type.
  // So, definition of ctors that require a context parameter are moved to
  // context.hpp.
  exception(context, std::error_code, const std::string &);
  exception(context, std::error_code, const char *);
  exception(context, std::error_code);
  exception(context, int, const std::error_category &, const std::string &);
  exception(context, int, const std::error_category &, const char *);
  exception(context, int, const std::error_category &);

  const std::error_code &code() const noexcept;
  const std::error_category &category() const noexcept;

  const char *what() const noexcept final;

  bool has_context() const noexcept;

  context get_context() const;

private:
  // Exceptions must be noexcept copy constructible, so cannot use std::string
  // directly.
  std::shared_ptr<detail::string> MMsg;
  int32_t MErr = 0;
  std::shared_ptr<context> MContext;
  std::error_code MErrC = make_error_code(sycl::errc::invalid);

protected:
  // base constructors used by SYCL 1.2.1 exception subclasses
  exception(std::error_code Ec, const char *Msg, const int32_t PIErr)
      : exception(Ec, std::string(Msg), PIErr) {}

  exception(std::error_code Ec, const std::string &Msg, const int32_t URErr)
      : exception(Ec, nullptr, Msg + " " + detail::codeToString(URErr)) {
    MErr = URErr;
  }

  // base constructor for all SYCL 2020 constructors
  // exception(context *, std::error_code, const std::string);
  exception(std::error_code Ec, std::shared_ptr<context> SharedPtrCtx,
            const std::string &what_arg)
      : exception(Ec, SharedPtrCtx, what_arg.c_str()) {}
  exception(std::error_code Ec, std::shared_ptr<context> SharedPtrCtx,
            const char *WhatArg);

  friend int32_t detail::get_ur_error(const exception &);
  // To be used like this:
  //   throw/return detail::set_ur_error(exception(...), some_ur_error);
  // *only* when such a error is coming from the UR level. Otherwise it
  // *should* be left unset/default-initialized and exception should be thrown
  // as-is using public ctors.
  friend exception detail::set_ur_error(exception &&e, int32_t ur_err);
};

namespace detail {
// Even though at the moment those functions are only used in library and not
// in headers, they were put here in case we will need them to implement some
// of OpenCL (and other backends) interop APIs to query native backend error
// from an exception.
// And we don't want them to be part of our library ABI, because of future
// underlying changes (PI -> UR -> Offload).
inline int32_t get_ur_error(const exception &e) { return e.MErr; }
inline exception set_ur_error(exception &&e, int32_t ur_err) {
  e.MErr = ur_err;
  return std::move(e);
}
} // namespace detail

} // namespace _V1
} // namespace sycl

namespace std {
template <> struct is_error_code_enum<sycl::errc> : true_type {};
} // namespace std
