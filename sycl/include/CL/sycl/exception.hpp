//==---------------- exception.hpp - SYCL exception ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

// 4.9.2 Exception Class Interface

#include <CL/sycl/backend_types.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/export.hpp>
#include <CL/sycl/detail/pi.h>
#include <CL/sycl/stl.hpp>

#include <exception>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

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

  bool has_context() const;

  context get_context() const;

  __SYCL2020_DEPRECATED("use sycl::exception.code() instead.")
  cl_int get_cl_code() const;

private:
  std::string MMsg;
  cl_int MCLErr;
  std::shared_ptr<context> MContext;

protected:
  // these two constructors are no longer used. Kept for ABI compatability.
  exception(const char *Msg, const cl_int CLErr,
            std::shared_ptr<context> Context = nullptr)
      : exception(std::string(Msg), CLErr, Context) {}
  exception(const std::string &Msg, const cl_int CLErr,
            std::shared_ptr<context> Context = nullptr)
      : MMsg(Msg + " " + detail::codeToString(CLErr)), MCLErr(CLErr),
        MContext(Context) {}

  // base constructors used by SYCL 1.2.1 exception subclasses
  exception(std::error_code ec, const char *Msg, const cl_int CLErr,
            std::shared_ptr<context> Context = nullptr)
      : exception(ec, std::string(Msg), CLErr, Context) {}

  exception(std::error_code ec, const std::string &Msg, const cl_int CLErr,
            std::shared_ptr<context> Context = nullptr)
      : exception(ec, Context, Msg + " " + detail::codeToString(CLErr)) {
    MCLErr = CLErr;
  }

  exception(const std::string &Msg) : MMsg(Msg), MContext(nullptr) {}

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
  runtime_error() = default;

  runtime_error(const char *Msg, cl_int Err)
      : runtime_error(std::string(Msg), Err) {}

  runtime_error(const std::string &Msg, cl_int Err)
      : exception(make_error_code(errc::runtime), Msg, Err) {}

  runtime_error(std::error_code ec, const std::string &Msg, const cl_int CLErr)
      : exception(ec, Msg, CLErr) {}
};

class __SYCL2020_DEPRECATED("use sycl::exception with sycl::errc::kernel or "
                            "errc::kernel_argument instead.") kernel_error
    : public runtime_error {
public:
  kernel_error() = default;

  kernel_error(const char *Msg, cl_int Err)
      : kernel_error(std::string(Msg), Err) {}

  kernel_error(const std::string &Msg, cl_int Err)
      : runtime_error(make_error_code(errc::kernel), Msg, Err) {}
};

class __SYCL2020_DEPRECATED(
    "use sycl::exception with sycl::errc::accessor instead.") accessor_error
    : public runtime_error {
public:
  accessor_error() = default;

  accessor_error(const char *Msg, cl_int Err)
      : accessor_error(std::string(Msg), Err) {}

  accessor_error(const std::string &Msg, cl_int Err)
      : runtime_error(make_error_code(errc::accessor), Msg, Err) {}
};

class __SYCL2020_DEPRECATED(
    "use sycl::exception with sycl::errc::nd_range instead.") nd_range_error
    : public runtime_error {
public:
  nd_range_error() = default;

  nd_range_error(const char *Msg, cl_int Err)
      : nd_range_error(std::string(Msg), Err) {}

  nd_range_error(const std::string &Msg, cl_int Err)
      : runtime_error(make_error_code(errc::nd_range), Msg, Err) {}
};

class __SYCL2020_DEPRECATED(
    "use sycl::exception with sycl::errc::event instead.") event_error
    : public runtime_error {
public:
  event_error() = default;

  event_error(const char *Msg, cl_int Err)
      : event_error(std::string(Msg), Err) {}

  event_error(const std::string &Msg, cl_int Err)
      : runtime_error(make_error_code(errc::event), Msg, Err) {}
};

class __SYCL2020_DEPRECATED(
    "use sycl::exception with a sycl::errc enum value instead.")
    invalid_parameter_error : public runtime_error {
public:
  invalid_parameter_error() = default;

  invalid_parameter_error(const char *Msg, cl_int Err)
      : invalid_parameter_error(std::string(Msg), Err) {}

  invalid_parameter_error(const std::string &Msg, cl_int Err)
      : runtime_error(make_error_code(errc::kernel_argument), Msg, Err) {}
};

class __SYCL2020_DEPRECATED(
    "use sycl::exception with a sycl::errc enum value instead.") device_error
    : public exception {
public:
  device_error() = default;

  device_error(const char *Msg, cl_int Err)
      : device_error(std::string(Msg), Err) {}

  device_error(const std::string &Msg, cl_int Err)
      : exception(make_error_code(errc::invalid), Msg, Err) {}

protected:
  device_error(std::error_code ec, const std::string &Msg, const cl_int CLErr)
      : exception(ec, Msg, CLErr) {}
};

class __SYCL2020_DEPRECATED(
    "use sycl::exception with a sycl::errc enum value instead.")
    compile_program_error : public device_error {
public:
  compile_program_error() = default;

  compile_program_error(const char *Msg, cl_int Err)
      : compile_program_error(std::string(Msg), Err) {}

  compile_program_error(const std::string &Msg, cl_int Err)
      : device_error(make_error_code(errc::build), Msg, Err) {}
};

class __SYCL2020_DEPRECATED(
    "use sycl::exception with a sycl::errc enum value instead.")
    link_program_error : public device_error {
public:
  link_program_error() = default;

  link_program_error(const char *Msg, cl_int Err)
      : link_program_error(std::string(Msg), Err) {}

  link_program_error(const std::string &Msg, cl_int Err)
      : device_error(make_error_code(errc::build), Msg, Err) {}
};

class __SYCL2020_DEPRECATED(
    "use sycl::exception with a sycl::errc enum value instead.")
    invalid_object_error : public device_error {
public:
  invalid_object_error() = default;

  invalid_object_error(const char *Msg, cl_int Err)
      : invalid_object_error(std::string(Msg), Err) {}

  invalid_object_error(const std::string &Msg, cl_int Err)
      : device_error(make_error_code(errc::invalid), Msg, Err) {}
};

class __SYCL2020_DEPRECATED(
    "use sycl::exception with sycl::errc::memory_allocation instead.")
    memory_allocation_error : public device_error {
public:
  memory_allocation_error() = default;

  memory_allocation_error(const char *Msg, cl_int Err)
      : memory_allocation_error(std::string(Msg), Err) {}

  memory_allocation_error(const std::string &Msg, cl_int Err)
      : device_error(make_error_code(errc::memory_allocation), Msg, Err) {}
};

class __SYCL2020_DEPRECATED(
    "use sycl::exception with sycl::errc::platform instead.") platform_error
    : public device_error {
public:
  platform_error() = default;

  platform_error(const char *Msg, cl_int Err)
      : platform_error(std::string(Msg), Err) {}

  platform_error(const std::string &Msg, cl_int Err)
      : device_error(make_error_code(errc::platform), Msg, Err) {}
};

class __SYCL2020_DEPRECATED(
    "use sycl::exception with sycl::errc::profiling instead.") profiling_error
    : public device_error {
public:
  profiling_error() = default;

  profiling_error(const char *Msg, cl_int Err)
      : profiling_error(std::string(Msg), Err) {}

  profiling_error(const std::string &Msg, cl_int Err)
      : device_error(make_error_code(errc::profiling), Msg, Err) {}
};

class __SYCL2020_DEPRECATED(
    "use sycl::exception with sycl::errc::feature_not_supported instead.")
    feature_not_supported : public device_error {
public:
  feature_not_supported() = default;

  feature_not_supported(const char *Msg, cl_int Err)
      : feature_not_supported(std::string(Msg), Err) {}

  feature_not_supported(const std::string &Msg, cl_int Err)
      : device_error(make_error_code(errc::feature_not_supported), Msg, Err) {}
};

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

namespace std {
template <> struct is_error_code_enum<cl::sycl::errc> : true_type {};
} // namespace std
