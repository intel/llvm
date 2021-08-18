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
  exception(const char *Msg, const cl_int CLErr,
            std::shared_ptr<context> Context = nullptr)
      : exception(std::string(Msg), CLErr, Context) {}

  exception(const std::string &Msg, const cl_int CLErr,
            std::shared_ptr<context> Context = nullptr)
      : MMsg(Msg + " " + detail::codeToString(CLErr)), MCLErr(CLErr),
        MContext(Context) {}

  exception(const string_class &Msg) : MMsg(Msg), MContext(nullptr) {}

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

  runtime_error(const std::string &Msg, cl_int Err) : exception(Msg, Err) {}
};
class __SYCL2020_DEPRECATED("use sycl::exception with sycl::errc::kernel or "
                            "errc::kernel_argument instead.") kernel_error
    : public runtime_error {
  using runtime_error::runtime_error;
};
class __SYCL2020_DEPRECATED(
    "use sycl::exception with sycl::errc::accessor instead.") accessor_error
    : public runtime_error {
  using runtime_error::runtime_error;
};
class __SYCL2020_DEPRECATED(
    "use sycl::exception with sycl::errc::nd_range instead.") nd_range_error
    : public runtime_error {
  using runtime_error::runtime_error;
};
class __SYCL2020_DEPRECATED(
    "use sycl::exception with sycl::errc::event instead.") event_error
    : public runtime_error {
  using runtime_error::runtime_error;
};
class __SYCL2020_DEPRECATED(
    "use sycl::exception with a sycl::errc enum value instead.")
    invalid_parameter_error : public runtime_error {
  using runtime_error::runtime_error;
};
class __SYCL2020_DEPRECATED(
    "use sycl::exception with a sycl::errc enum value instead.") device_error
    : public exception {
public:
  device_error() = default;

  device_error(const char *Msg, cl_int Err)
      : device_error(std::string(Msg), Err) {}

  device_error(const std::string &Msg, cl_int Err) : exception(Msg, Err) {}
};
class __SYCL2020_DEPRECATED(
    "use sycl::exception with a sycl::errc enum value instead.")
    compile_program_error : public device_error {
  using device_error::device_error;
};
class __SYCL2020_DEPRECATED(
    "use sycl::exception with a sycl::errc enum value instead.")
    link_program_error : public device_error {
  using device_error::device_error;
};
class __SYCL2020_DEPRECATED(
    "use sycl::exception with a sycl::errc enum value instead.")
    invalid_object_error : public device_error {
  using device_error::device_error;
};
class __SYCL2020_DEPRECATED(
    "use sycl::exception with sycl::errc::memory_allocation instead.")
    memory_allocation_error : public device_error {
  using device_error::device_error;
};
class __SYCL2020_DEPRECATED(
    "use sycl::exception with sycl::errc::platform instead.") platform_error
    : public device_error {
  using device_error::device_error;
};
class __SYCL2020_DEPRECATED(
    "use sycl::exception with sycl::errc::profiling instead.") profiling_error
    : public device_error {
  using device_error::device_error;
};
class __SYCL2020_DEPRECATED(
    "use sycl::exception with sycl::errc::feature_not_supported instead.")
    feature_not_supported : public device_error {
  using device_error::device_error;
};

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

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

namespace std {
template <> struct is_error_code_enum<cl::sycl::errc> : true_type {};
} // namespace std
