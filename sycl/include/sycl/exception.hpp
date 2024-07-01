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

  __SYCL2020_DEPRECATED("use sycl::exception.code() instead.")
  cl_int get_cl_code() const;

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
  exception(std::error_code Ec, const char *Msg, const pi_int32 PIErr,
            std::shared_ptr<context> Context = nullptr)
      : exception(Ec, std::string(Msg), PIErr, Context) {}

  exception(std::error_code Ec, const std::string &Msg, const pi_int32 PIErr,
            std::shared_ptr<context> Context = nullptr)
      : exception(Ec, Context, Msg + " " + detail::codeToString(PIErr)) {
    MPIErr = PIErr;
  }

  exception(const std::string &Msg)
#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
      : MMsg(std::make_shared<detail::string>(Msg)), MContext(nullptr){}
#else
      : MMsg(std::make_shared<std::string>(Msg)), MContext(nullptr) {
  }
#endif

        // base constructor for all SYCL 2020 constructors
        // exception(context *ctxPtr, std::error_code Ec, const std::string
        // &what_arg);
        exception(std::error_code Ec, std::shared_ptr<context> SharedPtrCtx,
                  const std::string &what_arg)
      : exception(Ec, SharedPtrCtx, what_arg.c_str()) {
  }
  exception(std::error_code Ec, std::shared_ptr<context> SharedPtrCtx,
            const char *WhatArg);
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

class __SYCL2020_DEPRECATED("use sycl::exception with sycl::errc::kernel or "
                            "errc::kernel_argument instead.") kernel_error
    : public runtime_error {
public:
  kernel_error() : runtime_error(make_error_code(errc::kernel)) {}

  kernel_error(const char *Msg, pi_int32 Err)
      : kernel_error(std::string(Msg), Err) {}

  kernel_error(const std::string &Msg, pi_int32 Err)
      : runtime_error(make_error_code(errc::kernel), Msg, Err) {}
};

class __SYCL2020_DEPRECATED(
    "use sycl::exception with sycl::errc::accessor instead.") accessor_error
    : public runtime_error {
public:
  accessor_error() : runtime_error(make_error_code(errc::accessor)) {}

  accessor_error(const char *Msg, pi_int32 Err)
      : accessor_error(std::string(Msg), Err) {}

  accessor_error(const std::string &Msg, pi_int32 Err)
      : runtime_error(make_error_code(errc::accessor), Msg, Err) {}
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
    "use sycl::exception with sycl::errc::event instead.") event_error
    : public runtime_error {
public:
  event_error() : runtime_error(make_error_code(errc::event)) {}

  event_error(const char *Msg, pi_int32 Err)
      : event_error(std::string(Msg), Err) {}

  event_error(const std::string &Msg, pi_int32 Err)
      : runtime_error(make_error_code(errc::event), Msg, Err) {}
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
    link_program_error : public device_error {
public:
  link_program_error() : device_error(make_error_code(errc::build)) {}

  link_program_error(const char *Msg, pi_int32 Err)
      : link_program_error(std::string(Msg), Err) {}

  link_program_error(const std::string &Msg, pi_int32 Err)
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

class __SYCL2020_DEPRECATED(
    "use sycl::exception with sycl::errc::memory_allocation instead.")
    memory_allocation_error : public device_error {
public:
  memory_allocation_error()
      : device_error(make_error_code(errc::memory_allocation)) {}

  memory_allocation_error(const char *Msg, pi_int32 Err)
      : memory_allocation_error(std::string(Msg), Err) {}

  memory_allocation_error(const std::string &Msg, pi_int32 Err)
      : device_error(make_error_code(errc::memory_allocation), Msg, Err) {}
};

class __SYCL2020_DEPRECATED(
    "use sycl::exception with sycl::errc::platform instead.") platform_error
    : public device_error {
public:
  platform_error() : device_error(make_error_code(errc::platform)) {}

  platform_error(const char *Msg, pi_int32 Err)
      : platform_error(std::string(Msg), Err) {}

  platform_error(const std::string &Msg, pi_int32 Err)
      : device_error(make_error_code(errc::platform), Msg, Err) {}
};

class __SYCL2020_DEPRECATED(
    "use sycl::exception with sycl::errc::profiling instead.") profiling_error
    : public device_error {
public:
  profiling_error() : device_error(make_error_code(errc::profiling)) {}

  profiling_error(const char *Msg, pi_int32 Err)
      : profiling_error(std::string(Msg), Err) {}

  profiling_error(const std::string &Msg, pi_int32 Err)
      : device_error(make_error_code(errc::profiling), Msg, Err) {}
};

class __SYCL2020_DEPRECATED(
    "use sycl::exception with sycl::errc::feature_not_supported instead.")
    feature_not_supported : public device_error {
public:
  feature_not_supported()
      : device_error(make_error_code(errc::feature_not_supported)) {}

  feature_not_supported(const char *Msg, pi_int32 Err)
      : feature_not_supported(std::string(Msg), Err) {}

  feature_not_supported(const std::string &Msg, pi_int32 Err)
      : device_error(make_error_code(errc::feature_not_supported), Msg, Err) {}
};

} // namespace _V1
} // namespace sycl

namespace std {
template <> struct is_error_code_enum<sycl::errc> : true_type {};
} // namespace std
