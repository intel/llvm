//==---------------- exception.hpp - SYCL exception ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

// 4.9.2 Exception Class Interface

#include <sycl/backend_types.hpp>
#include <sycl/detail/cl.h>
#include <sycl/detail/common.hpp>
#include <sycl/detail/export.hpp>
#include <sycl/detail/pi.h>
#include <sycl/stl.hpp>

#include <exception>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {

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

template <backend B> using errc_for = typename backend_traits<B>::errc;

/// Constructs an error code using e and sycl_category()
__SYCL_EXPORT std::error_code make_error_code(sycl::errc E) noexcept;

__SYCL_EXPORT const std::error_category &sycl_category() noexcept;

namespace detail {
class __SYCL_EXPORT SYCLCategory : public std::error_category {
public:
  const char *name() const noexcept override { return "sycl"; }
  std::string message(int) const override { return "SYCL Error"; }
};
} // namespace detail

// Derive from std::exception so uncaught exceptions are printed in c++ default
// exception handler.
/// \ingroup sycl_api
class __SYCL_EXPORT exception : public std::exception {
public:
  __SYCL2020_DEPRECATED("The version of an exception constructor which takes "
                        "no arguments is deprecated.")
  exception() = default;

  exception(std::error_code, const char *Msg);

  exception(std::error_code, const std::string &Msg);

  // new SYCL 2020 constructors
  exception(std::error_code);
  exception(int, const std::error_category &, const std::string &);
  exception(int, const std::error_category &, const char *);
  exception(int, const std::error_category &);

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
  std::shared_ptr<std::string> MMsg;
  pi_int32 MPIErr;
  std::shared_ptr<context> MContext;
  std::error_code MErrC = make_error_code(sycl::errc::invalid);

protected:
  // these two constructors are no longer used. Kept for ABI compatability.
  exception(const char *Msg, const pi_int32 PIErr,
            std::shared_ptr<context> Context = nullptr)
      : exception(std::string(Msg), PIErr, Context) {}
  exception(const std::string &Msg, const pi_int32 PIErr,
            std::shared_ptr<context> Context = nullptr)
      : MMsg(std::make_shared<std::string>(Msg + " " +
                                           detail::codeToString(PIErr))),
        MPIErr(PIErr), MContext(Context) {}

  // base constructors used by SYCL 1.2.1 exception subclasses
  exception(std::error_code ec, const char *Msg, const pi_int32 PIErr,
            std::shared_ptr<context> Context = nullptr)
      : exception(ec, std::string(Msg), PIErr, Context) {}

  exception(std::error_code ec, const std::string &Msg, const pi_int32 PIErr,
            std::shared_ptr<context> Context = nullptr)
      : exception(ec, Context, Msg + " " + detail::codeToString(PIErr)) {
    MPIErr = PIErr;
  }

  exception(const std::string &Msg)
      : MMsg(std::make_shared<std::string>(Msg)), MContext(nullptr) {}

  // base constructor for all SYCL 2020 constructors
  // exception(context *ctxPtr, std::error_code ec, const std::string
  // &what_arg);
  exception(std::error_code ec, std::shared_ptr<context> SharedPtrCtx,
            const std::string &what_arg);
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

  runtime_error(std::error_code ec, const std::string &Msg,
                const pi_int32 PIErr)
      : exception(ec, Msg, PIErr) {}

protected:
  runtime_error(std::error_code ec) : exception(ec) {}
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
  device_error(std::error_code ec) : exception(ec) {}

  device_error(std::error_code ec, const std::string &Msg, const pi_int32 PIErr)
      : exception(ec, Msg, PIErr) {}
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

} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl

namespace std {
template <> struct is_error_code_enum<sycl::errc> : true_type {};
} // namespace std
