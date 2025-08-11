//==--- sycl_imf.cpp --- kernel_compiler extension imf tests ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: aspect-usm_device_allocations
// REQUIRES: (opencl || level_zero)

// UNSUPPORTED: accelerator
// UNSUPPORTED-INTENDED: while accelerator is AoT only, this cannot run there.

// RUN: %{build} -o %t.out
// RUN: %{l0_leak_check} %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/usm.hpp>

auto constexpr IMFSources = R"===(
#include <sycl/sycl.hpp>
#include <cmath>
#include <sycl/ext/intel/math.hpp>

extern "C" SYCL_EXTERNAL 
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(sycl::ext::oneapi::experimental::single_task_kernel)
void imf_kernel(float *ptr) {
  // cl_intel_devicelib_imf
  ptr[0] = sycl::ext::intel::math::sqrt(ptr[0] * 2);

  // cl_intel_devicelib_imf_bf16
  ptr[1] = sycl::ext::intel::math::float2bfloat16(ptr[1] * 0.5f);
}
)===";

int main() {
#ifdef SYCL_EXT_ONEAPI_KERNEL_COMPILER
  sycl::queue q;
  namespace syclex = sycl::ext::oneapi::experimental;
  using source_kb = sycl::kernel_bundle<sycl::bundle_state::ext_oneapi_source>;
  using exe_kb = sycl::kernel_bundle<sycl::bundle_state::executable>;

  sycl::context ctx = q.get_context();

  source_kb kbSrc = syclex::create_kernel_bundle_from_source(
      ctx, syclex::source_language::sycl, IMFSources);
  exe_kb kbExe = syclex::build(kbSrc);

  sycl::kernel k = kbExe.ext_oneapi_get_kernel("imf_kernel");
  constexpr size_t nElem = 2;
  float *ptr = sycl::malloc_shared<float>(nElem, q);
  for (int i = 0; i < nElem; ++i)
    ptr[i] = 1.0f;

  q.submit([&](sycl::handler &cgh) {
    cgh.set_arg(0, ptr);
    cgh.single_task(k);
  });
  q.wait_and_throw();

  // Check that the kernel was executed. Given the {1.0, 1.0} input,
  // the expected result is approximately {1.41, 0.5}.
  for (unsigned i = 0; i < nElem; ++i) {
    std::cout << ptr[i] << ' ';
    assert(ptr[i] != 1.0f);
  }
  std::cout << std::endl;

  sycl::free(ptr, q);
#else
  static_assert(false, "Kernel Compiler feature test macro undefined");
#endif
  return 0;
}
