// RUN: %{build} -o %t.out
// RUN: %if hip %{ env SYCL_JIT_AMDGCN_PTX_TARGET_CPU=%{amd_arch} %} %{run} %t.out | FileCheck %s

// UNSUPPORTED: target-native_cpu
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/20142

// CHECK-LABEL: Device compilation failed
// CHECK-NEXT:  Detailed information:
// CHECK-NEXT:  	rtc_0.cpp:2:10: fatal error: 'non-existent.hpp' file not found
// CHECK-NEXT:      2 | #include "non-existent.hpp"
// CHECK-NEXT:        |          ^~~~~~~~~~~~~~~~~~

// Make sure that the error is reported properly when using "--auto-pch" option.

#include <sycl/detail/core.hpp>
#include <sycl/kernel_bundle.hpp>

#include <chrono>
#include <iostream>
#include <sstream>
#include <string_view>

using namespace std::string_view_literals;
namespace syclexp = sycl::ext::oneapi::experimental;

int main(int argc, char **argv) {
  std::string src = R"""(
#include "non-existent.hpp"
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/ext/oneapi/kernel_properties/properties.hpp>

namespace syclext = sycl::ext::oneapi;
namespace syclexp = sycl::ext::oneapi::experimental;

extern "C"
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void iota(float start, float *ptr) {
    size_t id = syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
    ptr[id] = start + static_cast<float>(id);
}
)""";

  sycl::queue q;
  sycl::kernel_bundle<sycl::bundle_state::ext_oneapi_source> kb_src =
      syclexp::create_kernel_bundle_from_source(
          q.get_context(), syclexp::source_language::sycl, src);
  try {
    sycl::kernel_bundle<sycl::bundle_state::executable> kb_exe =
        syclexp::build(kb_src, syclexp::properties{syclexp::build_options{
                                   std::vector<std::string>{"--auto-pch"}}}

        );
    return 1;
  } catch (sycl::exception &e) {
    std::cout << e.what() << std::endl;
  }
}
