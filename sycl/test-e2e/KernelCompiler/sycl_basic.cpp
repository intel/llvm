//==--- sycl_basic.cpp --- kernel_compiler extension tests -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: true
// UNSUPPORTED-INTENDED: sycl-jit is disabled on this branch.

// REQUIRES: (opencl || level_zero)
// REQUIRES: aspect-usm_device_allocations

// UNSUPPORTED: accelerator
// UNSUPPORTED-INTENDED: while accelerator is AoT only, this cannot run there.

// RUN: %{build} -o %t.out
// RUN: %{l0_leak_check} %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/usm.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

static constexpr size_t NUM = 1024;
static constexpr size_t WGSIZE = 16;

int main() {
  sycl::queue q;

  // The source code for a kernel, defined as a SYCL "free function kernel".
  std::string source = R"""(
    #include <sycl/sycl.hpp>
    namespace syclext = sycl::ext::oneapi;
    namespace syclexp = sycl::ext::oneapi::experimental;

    extern "C"
    SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
    void iota(float start, float *ptr) {
      size_t id = syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
      ptr[id] = start + static_cast<float>(id);
    }
  )""";

  // Create a kernel bundle in "source" state.
  sycl::kernel_bundle<sycl::bundle_state::ext_oneapi_source> kb_src =
      syclexp::create_kernel_bundle_from_source(
          q.get_context(), syclexp::source_language::sycl, source);

  // Compile the kernel.  There is no need to use the "registered_names"
  // property because the kernel is declared extern "C".
  sycl::kernel_bundle<sycl::bundle_state::executable> kb_exe =
      syclexp::build(kb_src);

  // Get the kernel via its compiler-generated name.
  sycl::kernel iota = kb_exe.ext_oneapi_get_kernel("iota");

  float *ptr = sycl::malloc_shared<float>(NUM, q);
  q.submit([&](sycl::handler &cgh) {
     // Set the values of the kernel arguments.
     cgh.set_args(3.14f, ptr);

     // Launch the kernel according to its type, in this case an nd-range
     // kernel.
     sycl::nd_range ndr{{NUM}, {WGSIZE}};
     cgh.parallel_for(ndr, iota);
   }).wait();

  constexpr float eps = 0.001;
  for (int i = 0; i < NUM; i++) {
    const float truth = 3.14f + static_cast<float>(i);
    if (std::abs(ptr[i] - truth) > eps) {
      std::cout << "Result: " << ptr[i] << " expected " << i << "\n";
      sycl::free(ptr, q);
      exit(1);
    }
  }
  sycl::free(ptr, q);
}
