//==------- ocloc_queries.cpp ----------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: ocloc && (opencl || level_zero)
// UNSUPPORTED: accelerator

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Note: the ocloc queries that are tested here are relatively new ( Dec 2023 )
// if encountering many failures then an outdated ocloc is the likely culprit.

#include <sycl/sycl.hpp>
namespace syclex = sycl::ext::oneapi::experimental;

int main() {
  sycl::queue q;
  sycl::device d = q.get_device();

  assert(d.ext_oneapi_can_compile(syclex::source_language::opencl) &&
         "can_compile(opencl) unexpectedly false");

  assert(d.ext_oneapi_supports_cl_c_version(syclex::opencl_c_1_0) &&
         "supports_cl_version(1.0) unexpectedly false");

  assert(d.ext_oneapi_supports_cl_c_feature("__opencl_c_int64") &&
         "int64 support unexpectedly false");
  assert(!d.ext_oneapi_supports_cl_c_feature("not_a_real_feature") &&
         "imaginery feature support unexpectedly true");

  syclex::cl_version version{20, 20, 20};
  bool res = d.ext_oneapi_supports_cl_extension("cl_intel_bfloat16_conversions",
                                                &version);
  if (res) {
    assert(version.major != 20 && version.minor != 20 && version.patch != 20 &&
           "version not updated");
  }

  // no supported devices support EMBEDDED_PROFILE at this time.
  assert(d.ext_oneapi_cl_profile() == "FULL_PROFILE" &&
         "unexpected cl_profile");

  assert(syclex::opencl_c_1_0.major == 1 && syclex::opencl_c_1_0.minor == 0 && syclex::opencl_c_1_0.patch == 0);
  assert(syclex::opencl_c_1_1.major == 1 && syclex::opencl_c_1_1.minor == 1 && syclex::opencl_c_1_1.patch == 0);
  assert(syclex::opencl_c_1_2.major == 1 && syclex::opencl_c_1_2.minor == 2 && syclex::opencl_c_1_2.patch == 0);
  assert(syclex::opencl_c_2_0.major == 2 && syclex::opencl_c_2_0.minor == 0 && syclex::opencl_c_2_0.patch == 0);
  assert(syclex::opencl_c_3_0.major == 3 && syclex::opencl_c_3_0.minor == 0 && syclex::opencl_c_3_0.patch == 0);

  return 0;
}
