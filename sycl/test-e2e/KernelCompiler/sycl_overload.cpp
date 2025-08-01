//==--- sycl_overload.cpp --- kernel_compiler extension tests --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: aspect-usm_device_allocations

// RUN: %{build} -o %t.out
// RUN: %if hip %{ env SYCL_JIT_AMDGCN_PTX_TARGET_CPU=%{amd_arch} %} %{l0_leak_check} %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/usm.hpp>
namespace syclexp = sycl::ext::oneapi::experimental;

static constexpr size_t NUM = 1024;
static constexpr size_t WGSIZE = 16;

int main() {
  sycl::queue q;

  // The source code for two kernels defined as overloaded functions.
  std::string source = R"""(
    #include <sycl/sycl.hpp>
    namespace syclext = sycl::ext::oneapi;
    namespace syclexp = sycl::ext::oneapi::experimental;

    SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
    void iota(float start, float *ptr) {
      size_t id = syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
      ptr[id] = start + static_cast<float>(id);
    }

    SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
    void iota(int start, int *ptr) {
      size_t id = syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
      ptr[id] = start + static_cast<int>(id);
    }
  )""";

  // Create a kernel bundle in "source" state.
  sycl::kernel_bundle<sycl::bundle_state::ext_oneapi_source> kb_src =
      syclexp::create_kernel_bundle_from_source(
          q.get_context(), syclexp::source_language::sycl, source);

  // Compile the kernel.  Because there are two overloads of "iota", we need to
  // use a C++ cast to disambiguate between them.  Here, we are selecting the
  // "int" overload.
  std::string iota_name{"(void(*)(int, int*))iota"};
  sycl::kernel_bundle<sycl::bundle_state::executable> kb_exe = syclexp::build(
      kb_src, syclexp::properties{syclexp::registered_names{iota_name}});

  // Get the kernel by passing the same string we used to construct the
  // "registered_names" property.
  sycl::kernel iota = kb_exe.ext_oneapi_get_kernel(iota_name);

  int *ptr = sycl::malloc_shared<int>(NUM, q);
  q.submit([&](sycl::handler &cgh) {
     // Set the values of the kernel arguments.
     cgh.set_args(3, ptr);

     // Launch the kernel according to its type, in this case an nd-range
     // kernel.
     sycl::nd_range ndr{{NUM}, {WGSIZE}};
     cgh.parallel_for(ndr, iota);
   }).wait();

  for (int i = 0; i < NUM; i++) {
    if (ptr[i] != i + 3) {
      std::cout << "Result: " << ptr[i] << " expected " << i << "\n";
      sycl::free(ptr, q);
      exit(1);
    }
  }
  sycl::free(ptr, q);
}
