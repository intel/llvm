//==------- ocloc_queries_negative.cpp -------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Test when we don't have ocloc (eg CUDA, HIP, etc)
// REQUIRES: !opencl && !level_zero

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
namespace syclex = sycl::ext::oneapi::experimental;

int main() {
  sycl::queue q;
  sycl::device d = q.get_device();

  assert(!d.ext_oneapi_can_compile(syclex::source_language::opencl) &&
         "can_compile(opencl) unexpectedly true");

  assert(!d.ext_oneapi_supports_cl_c_version(syclex::opencl_c_1_0) &&
         "supports_cl_version(1.0) unexpectedly true");

  assert(!d.ext_oneapi_supports_cl_c_feature("__opencl_c_int64") &&
         "int64 support unexpectedly true");
  assert(!d.ext_oneapi_supports_cl_c_feature("not_a_real_feature") &&
         "imaginery feature support unexpectedly true");

  syclex::cl_version version{20, 20, 20};
  bool res = d.ext_oneapi_supports_cl_extension("cl_intel_bfloat16_conversions",
                                                &version);
  assert(!res && "ext_oneapi_supports_cl_extension unexpectedly true");

  assert(d.ext_oneapi_cl_profile() == "" && "cl_profile unexpectedly returned");

  return 0;
}
