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

#define DEFINE_CL_SYCL_EXCEPTION(Name)                                         \
  class Name : public __sycl_ns::Name {                                        \
    using __sycl_ns::Name::Name;                                               \
  };


DEFINE_CL_SYCL_EXCEPTION(runtime_error)
DEFINE_CL_SYCL_EXCEPTION(kernel_error)
DEFINE_CL_SYCL_EXCEPTION(accessor_error)
DEFINE_CL_SYCL_EXCEPTION(nd_range_error)
DEFINE_CL_SYCL_EXCEPTION(event_error)
DEFINE_CL_SYCL_EXCEPTION(invalid_parameter_error)
DEFINE_CL_SYCL_EXCEPTION(device_error)
DEFINE_CL_SYCL_EXCEPTION(compile_program_error)
DEFINE_CL_SYCL_EXCEPTION(link_program_error)
DEFINE_CL_SYCL_EXCEPTION(invalid_object_error)
DEFINE_CL_SYCL_EXCEPTION(memory_allocation_error)
DEFINE_CL_SYCL_EXCEPTION(platform_error)
DEFINE_CL_SYCL_EXCEPTION(profiling_error)
DEFINE_CL_SYCL_EXCEPTION(feature_not_supported)


//class runtime_error : public __sycl_ns::runtime_error {
  //using __sycl_ns::runtime_error::runtime_error;
//};

//class kernel_error : public runtime_error {
  //using runtime_error::runtime_error;
//};
//class accessor_error : public runtime_error {
  //using runtime_error::runtime_error;
//};
//class nd_range_error : public runtime_error {
  //using runtime_error::runtime_error;
//};
//class event_error : public runtime_error {
  //using runtime_error::runtime_error;
//};

//class invalid_parameter_error : public __sycl_ns::invalid_parameter_error {
  //using __sycl_ns::invalid_parameter_error::invalid_parameter_error;
//};

//class device_error : public __sycl_ns::exception {
//public:
  //device_error() = default;

  //device_error(const char *Msg, cl_int Err)
      //: device_error(std::string(Msg), Err) {}

  //device_error(const std::string &Msg, cl_int Err) : __sycl_ns::exception(Msg, Err) {}
//};
//class compile_program_error : public device_error {
  //using device_error::device_error;
//};
//class link_program_error : public device_error {
  //using device_error::device_error;
//};
//class invalid_object_error : public device_error {
  //using device_error::device_error;
//};
//class memory_allocation_error : public device_error {
  //using device_error::device_error;
//};
//class platform_error : public device_error {
  //using device_error::device_error;
//};
//class profiling_error : public device_error {
  //using device_error::device_error;
//};
//class feature_not_supported : public device_error {
  //using device_error::device_error;
//};
} // namespace sycl
} // namespace cl

__SYCL_OPEN_NS() {

#define DEFINE_COMPAT_EXCEPTION(Name)                                          \
  class Name##_compat : public cl::sycl::Name {                                \
    using cl::sycl::Name::Name;                                                \
  };

DEFINE_COMPAT_EXCEPTION(runtime_error)
DEFINE_COMPAT_EXCEPTION(kernel_error)
DEFINE_COMPAT_EXCEPTION(accessor_error)
DEFINE_COMPAT_EXCEPTION(nd_range_error)
DEFINE_COMPAT_EXCEPTION(event_error)
DEFINE_COMPAT_EXCEPTION(invalid_parameter_error)
DEFINE_COMPAT_EXCEPTION(device_error)
DEFINE_COMPAT_EXCEPTION(compile_program_error)
DEFINE_COMPAT_EXCEPTION(link_program_error)
DEFINE_COMPAT_EXCEPTION(invalid_object_error)
DEFINE_COMPAT_EXCEPTION(memory_allocation_error)
DEFINE_COMPAT_EXCEPTION(platform_error)
DEFINE_COMPAT_EXCEPTION(profiling_error)
DEFINE_COMPAT_EXCEPTION(feature_not_supported)

//class runtime_error_compat : public cl::sycl::runtime_error {
  //using cl::sycl::runtime_error::runtime_error;
//};

//class runtime_error_compat : public __sycl_ns::runtime_error,
                             //public cl::sycl::runtime_error {
//public:
  //runtime_error_compat() = default;
  //runtime_error_compat(const char *Msg, cl_int Err)
      //: runtime_error_compat(std::string{Msg}, Err) {}

  //runtime_error_compat(const std::string &Msg, cl_int Err)
      //: __sycl_ns::runtime_error(Msg, Err), cl::sycl::runtime_error(Msg, Err) {}
//};

//class invalid_parameter_error_compat
    //: public cl::sycl::invalid_parameter_error {
  //using cl::sycl::invalid_parameter_error::invalid_parameter_error;
//};

//class invalid_parameter_error_compat
    //: public __sycl_ns::invalid_parameter_error,
      //public cl::sycl::invalid_parameter_error {
//public:
  //invalid_parameter_error_compat() = default;
  //invalid_parameter_error_compat(const char *Msg, cl_int Err)
      //: invalid_parameter_error_compat(std::string{Msg}, Err) {}

  //invalid_parameter_error_compat(const std::string &Msg, cl_int Err)
      //: __sycl_ns::invalid_parameter_error(Msg, Err), cl::sycl::invalid_parameter_error(Msg, Err) {}
//};

//////////////////////////

//#define DEFINE_COMPAT_EXCEPTION_TYPE(Name)                                     \
  //class Name##_compat : public __sycl_ns::Name, public cl::sycl::Name {        \
  //public:                                                                      \
    //Name##_compat() = default;                                                 \
                                                                               //\
    //Name##_compat(const char *Msg, cl_int Err)                                 \
        //: Name##_compat(std::string(Msg), Err) {}                              \
                                                                               //\
    //Name##_compat(const std::string &Msg, cl_int Err)                          \
        //: __sycl_ns::Name(Msg, Err), cl::sycl::Name(Msg, Err) {}               \
  //};

//DEFINE_COMPAT_EXCEPTION_TYPE(kernel_error)
//DEFINE_COMPAT_EXCEPTION_TYPE(device_error)
//DEFINE_COMPAT_EXCEPTION_TYPE(feature_not_supported)
//DEFINE_COMPAT_EXCEPTION_TYPE(invalid_object_error)
////DEFINE_COMPAT_EXCEPTION_TYPE(invalid_parameter_error)
//DEFINE_COMPAT_EXCEPTION_TYPE(compile_program_error)
//DEFINE_COMPAT_EXCEPTION_TYPE(nd_range_error)

} // __SYCL_OPEN_NS()
__SYCL_CLOSE_NS()
