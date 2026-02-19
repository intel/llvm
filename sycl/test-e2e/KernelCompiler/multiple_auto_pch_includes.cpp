// REQUIRES: sycl-jit

// RUN: %{build} -o %t.out
// RUN: %if hip %{ env SYCL_JIT_AMDGCN_PTX_TARGET_CPU=%{amd_arch} %} %{run} %t.out

// UNSUPPORTED: target-native_cpu
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/20142

// Verify that including multiple utility header files in the kernel code
// is properly compiled with auto pre-compiled headers, for various c++
// standards.

#include <sycl/detail/core.hpp>
#include <sycl/kernel_bundle.hpp>

#include <chrono>
#include <iostream>
#include <sstream>
#include <string_view>

using namespace std::string_view_literals;
namespace syclexp = sycl::ext::oneapi::experimental;

int main(int argc, char **argv) {
  auto Test = [](std::string src) {
    sycl::queue q;
    std::vector<std::string> cpp_standards = {"-std=c++17", "-std=c++20",
                                              "-std=c++23"};
    for (int j = 0; j < 2; j++) {
      // Two iterations to test pch creation/use:
      for (int i = 0; i < 2; ++i) {
        sycl::kernel_bundle<sycl::bundle_state::ext_oneapi_source> kb_src =
            syclexp::create_kernel_bundle_from_source(
                q.get_context(), syclexp::source_language::sycl, src);
        sycl::kernel_bundle<sycl::bundle_state::executable> kb_exe =
            syclexp::build(
                kb_src,
                syclexp::properties{syclexp::build_options{
                    std::vector<std::string>{"--auto-pch", cpp_standards[j]}}}

            );
      }
    }
  };

  Test(R"""(
#include <sycl/builtins.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/ext/oneapi/kernel_properties/properties.hpp>
#include <sycl/group_barrier.hpp>
#include <sycl/half_type.hpp>
#include <sycl/marray.hpp>
#include <sycl/multi_ptr.hpp>
#include <sycl/vector.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

extern "C"
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::single_task_kernel))
void foo(int *p) {
  *p = 42;
}
)""");
}
