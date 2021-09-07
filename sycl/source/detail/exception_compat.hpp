//==---------------- exception.hpp - SYCL exception ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/exception.hpp>

#include <string>

namespace cl {
namespace sycl {

class runtime_error : public __sycl_ns::exception {
public:
  runtime_error() = default;

  runtime_error(const char *Msg, cl_int Err)
      : runtime_error(std::string(Msg), Err) {}

  runtime_error(const std::string &Msg, cl_int Err) : __sycl_ns::exception(Msg, Err) {}
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
class device_error : public __sycl_ns::exception {
public:
  device_error() = default;

  device_error(const char *Msg, cl_int Err)
      : device_error(std::string(Msg), Err) {}

  device_error(const std::string &Msg, cl_int Err) : __sycl_ns::exception(Msg, Err) {}
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
} // namespace cl

__SYCL_OPEN_NS() {

class runtime_error_compat : public __sycl_ns::runtime_error,
                             public cl::sycl::runtime_error {
public:
  runtime_error_compat() = default;
  runtime_error_compat(const char *Msg, cl_int Err)
      : runtime_error_compat(std::string{Msg}, Err) {}

  runtime_error_compat(const std::string &Msg, cl_int Err)
      : __sycl_ns::runtime_error(Msg, Err), cl::sycl::runtime_error(Msg, Err) {}
};

//////////////////////////

#define DEFINE_COMPAT_EXCEPTION_TYPE(Name)                                     \
  class Name##_compat : public __sycl_ns::Name, public cl::sycl::Name {        \
  public:                                                                      \
    Name##_compat() = default;                                                 \
                                                                               \
    Name##_compat(const char *Msg, cl_int Err)                                 \
        : Name##_compat(std::string(Msg), Err) {}                              \
                                                                               \
    Name##_compat(const std::string &Msg, cl_int Err)                          \
        : __sycl_ns::Name(Msg, Err), cl::sycl::Name(Msg, Err) {}               \
  };

DEFINE_COMPAT_EXCEPTION_TYPE(kernel_error)
DEFINE_COMPAT_EXCEPTION_TYPE(device_error)
DEFINE_COMPAT_EXCEPTION_TYPE(feature_not_supported)
DEFINE_COMPAT_EXCEPTION_TYPE(invalid_object_error)
DEFINE_COMPAT_EXCEPTION_TYPE(invalid_parameter_error)
DEFINE_COMPAT_EXCEPTION_TYPE(compile_program_error)
DEFINE_COMPAT_EXCEPTION_TYPE(nd_range_error)

} // __SYCL_OPEN_NS()
__SYCL_CLOSE_NS()
