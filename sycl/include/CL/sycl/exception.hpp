//==---------------- exception.hpp - SYCL exception ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

// 4.9.2 Exception Class Interface

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
  exception() = default;

  exception(std::error_code, const char *Msg)
      : exception(Msg, PI_INVALID_VALUE) {}

  exception(std::error_code, const std::string &Msg)
      : exception(Msg, PI_INVALID_VALUE) {}

  const char *what() const noexcept final;

  bool has_context() const;

  context get_context() const;

  cl_int get_cl_code() const;

private:
  string_class MMsg;
  cl_int MCLErr;
  shared_ptr_class<context> MContext;

protected:
  exception(const char *Msg, const cl_int CLErr,
            shared_ptr_class<context> Context = nullptr)
      : exception(string_class(Msg), CLErr, Context) {}

  exception(const string_class &Msg, const cl_int CLErr,
            shared_ptr_class<context> Context = nullptr)
      : MMsg(Msg + " " + detail::codeToString(CLErr)), MCLErr(CLErr),
        MContext(Context) {}

  exception(const string_class &Msg) : MMsg(Msg), MContext(nullptr) {}
};

class runtime_error : public exception {
public:
  runtime_error() = default;

  runtime_error(const char *Msg, cl_int Err)
      : runtime_error(string_class(Msg), Err) {}

  runtime_error(const string_class &Msg, cl_int Err) : exception(Msg, Err) {}
};
class kernel_error : public runtime_error {
  using runtime_error::runtime_error;
};
class accessor_error : public runtime_error {
  using runtime_error::runtime_error;
};
class nd_range_error : public runtime_error {
  using runtime_error::runtime_error;
};
class event_error : public runtime_error {
  using runtime_error::runtime_error;
};
class invalid_parameter_error : public runtime_error {
  using runtime_error::runtime_error;
};
class device_error : public exception {
public:
  device_error() = default;

  device_error(const char *Msg, cl_int Err)
      : device_error(string_class(Msg), Err) {}

  device_error(const string_class &Msg, cl_int Err) : exception(Msg, Err) {}
};
class compile_program_error : public device_error {
  using device_error::device_error;
};
class link_program_error : public device_error {
  using device_error::device_error;
};
class invalid_object_error : public device_error {
  using device_error::device_error;
};
class memory_allocation_error : public device_error {
  using device_error::device_error;
};
class platform_error : public device_error {
  using device_error::device_error;
};
class profiling_error : public device_error {
  using device_error::device_error;
};
class feature_not_supported : public device_error {
  using device_error::device_error;
};

enum class errc : unsigned int {
  runtime = 0,
  kernel = 1,
  accessor = 2,
  nd_range = 3,
  event = 4,
  kernel_argument = 5,
  build = 6,
  invalid = 7,
  memory_allocation = 8,
  platform = 9,
  profiling = 10,
  feature_not_supported = 11,
  kernel_not_supported = 12,
  backend_mismatch = 13,
};

/// Constructs an error code using e and sycl_category()
__SYCL_EXPORT std::error_code make_error_code(sycl::errc E) noexcept;

__SYCL_EXPORT const std::error_category &sycl_category() noexcept;

namespace detail {
class __SYCL_EXPORT SYCLCategory : public std::error_category {
public:
  const char *name() const noexcept override { return "SYCL"; }
  std::string message(int) const override { return "SYCL Error"; }
};
} // namespace detail

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

namespace std {
template <> struct is_error_condition_enum<cl::sycl::errc> : true_type {};
} // namespace std
