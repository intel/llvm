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
#include <sycl/detail/pi.h>                   // for pi_int32
#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
#include <sycl/detail/string.hpp>
#endif

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
__SYCL_EXPORT const char *stringifyErrorCode(pi_int32 error);

inline std::string codeToString(pi_int32 code) {
  return std::string(std::to_string(code) + " (" + stringifyErrorCode(code) +
                     ")");
}

class __SYCL_EXPORT SYCLCategory : public std::error_category {
public:
  const char *name() const noexcept override { return "sycl"; }
  std::string message(int) const override { return "SYCL Error"; }
};

// Forward declare to declare as a friend in sycl::excepton.
__SYCL_EXPORT pi_int32 get_pi_error(const exception &e);
// TODO: Should it be exported at all?
__SYCL_EXPORT exception set_pi_error(exception &&e, pi_int32 pi_err);
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
#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
  std::shared_ptr<detail::string> MMsg;
#else
  std::shared_ptr<std::string> MMsg;
#endif
  pi_int32 MPIErr = 0;
  std::shared_ptr<context> MContext;
  std::error_code MErrC = make_error_code(sycl::errc::invalid);

protected:
  // base constructors used by SYCL 1.2.1 exception subclasses
  exception(std::error_code Ec, const char *Msg, const pi_int32 PIErr)
      : exception(Ec, std::string(Msg), PIErr) {}

  exception(std::error_code Ec, const std::string &Msg, const pi_int32 PIErr)
      : exception(Ec, nullptr, Msg + " " + detail::codeToString(PIErr)) {
    MPIErr = PIErr;
  }

  // base constructor for all SYCL 2020 constructors
  // exception(context *, std::error_code, const std::string);
  exception(std::error_code Ec, std::shared_ptr<context> SharedPtrCtx,
            const std::string &what_arg)
      : exception(Ec, SharedPtrCtx, what_arg.c_str()) {}
  exception(std::error_code Ec, std::shared_ptr<context> SharedPtrCtx,
            const char *WhatArg);

  friend __SYCL_EXPORT pi_int32 detail::get_pi_error(const exception &);
  // To be used like this:
  //   throw/return detail::set_pi_error(exception(...), some_pi_error);
  // *only* when such a error is coming from the PI/UR level. Otherwise it
  // *should be left unset/default-initialized and exception should be thrown
  // as-is using public ctors.
  friend __SYCL_EXPORT exception detail::set_pi_error(exception &&e,
                                                      pi_int32 pi_err);
};

class __SYCL2020_DEPRECATED(
    "use sycl::exception with sycl::errc::runtime instead.") runtime_error
    : public exception {
public:
  runtime_error() : exception(make_error_code(errc::runtime)) {}

  runtime_error(const char *Msg, pi_int32 Err)
      : runtime_error(std::string(Msg), Err) {}

  runtime_error(const std::string &Msg, pi_int32 Err)
      : exception(make_error_code(errc::runtime), Msg, Err) {}

  runtime_error(std::error_code Ec, const std::string &Msg,
                const pi_int32 PIErr)
      : exception(Ec, Msg, PIErr) {}

protected:
  runtime_error(std::error_code Ec) : exception(Ec) {}
};

class __SYCL2020_DEPRECATED(
    "use sycl::exception with sycl::errc::nd_range instead.") nd_range_error
    : public runtime_error {
public:
  nd_range_error() : runtime_error(make_error_code(errc::nd_range)) {}

  nd_range_error(const char *Msg, pi_int32 Err)
      : nd_range_error(std::string(Msg), Err) {}

  nd_range_error(const std::string &Msg, pi_int32 Err)
      : runtime_error(make_error_code(errc::nd_range), Msg, Err) {}
};

class __SYCL2020_DEPRECATED(
    "use sycl::exception with a sycl::errc enum value instead.")
    invalid_parameter_error : public runtime_error {
public:
  invalid_parameter_error()
      : runtime_error(make_error_code(errc::kernel_argument)) {}

  invalid_parameter_error(const char *Msg, pi_int32 Err)
      : invalid_parameter_error(std::string(Msg), Err) {}

  invalid_parameter_error(const std::string &Msg, pi_int32 Err)
      : runtime_error(make_error_code(errc::kernel_argument), Msg, Err) {}
};

class __SYCL2020_DEPRECATED(
    "use sycl::exception with a sycl::errc enum value instead.") device_error
    : public exception {
public:
  device_error() : exception(make_error_code(errc::invalid)) {}

  device_error(const char *Msg, pi_int32 Err)
      : device_error(std::string(Msg), Err) {}

  device_error(const std::string &Msg, pi_int32 Err)
      : exception(make_error_code(errc::invalid), Msg, Err) {}

protected:
  device_error(std::error_code Ec) : exception(Ec) {}

  device_error(std::error_code Ec, const std::string &Msg, const pi_int32 PIErr)
      : exception(Ec, Msg, PIErr) {}
};

class __SYCL2020_DEPRECATED(
    "use sycl::exception with a sycl::errc enum value instead.")
    compile_program_error : public device_error {
public:
  compile_program_error() : device_error(make_error_code(errc::build)) {}

  compile_program_error(const char *Msg, pi_int32 Err)
      : compile_program_error(std::string(Msg), Err) {}

  compile_program_error(const std::string &Msg, pi_int32 Err)
      : device_error(make_error_code(errc::build), Msg, Err) {}
};

class __SYCL2020_DEPRECATED(
    "use sycl::exception with a sycl::errc enum value instead.")
    invalid_object_error : public device_error {
public:
  invalid_object_error() : device_error(make_error_code(errc::invalid)) {}

  invalid_object_error(const char *Msg, pi_int32 Err)
      : invalid_object_error(std::string(Msg), Err) {}

  invalid_object_error(const std::string &Msg, pi_int32 Err)
      : device_error(make_error_code(errc::invalid), Msg, Err) {}
};

} // namespace _V1
} // namespace sycl

namespace std {
template <> struct is_error_code_enum<sycl::errc> : true_type {};
} // namespace std
