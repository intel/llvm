// RUN: %clangxx -D__SYCL_USE_VARIADIC_SPIRV_OCL_PRINTF__ -fsycl -fsycl-device-only -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s

// expected-warning@*:* {{__SYCL_USE_VARIADIC_SPIRV_OCL_PRINTF__ is deprecated and will be removed in a future release.}}
#include <sycl/sycl.hpp>

