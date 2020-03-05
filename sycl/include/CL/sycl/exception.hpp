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
#include <CL/sycl/detail/pi.h>
#include <CL/sycl/stl.hpp>

#include <exception>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

// Forward declaration
class context;

// Derive from std::exception so uncaught exceptions are printed in c++ default
// exception handler.
class exception: public std::exception {
public:
  exception() = default;

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

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
