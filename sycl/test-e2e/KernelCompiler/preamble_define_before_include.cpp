// RUN: %{build} -o %t.out
// RUN: %if hip %{ env SYCL_JIT_AMDGCN_PTX_TARGET_CPU=%{amd_arch} %} %{run} %t.out

// UNSUPPORTED: target-native_cpu
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/20142

// Verify that having #define before includes works as expected.

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
    // Two iterations to test pch creation/use:
    for (int i = 0; i < 2; ++i) {
      sycl::kernel_bundle<sycl::bundle_state::ext_oneapi_source> kb_src =
          syclexp::create_kernel_bundle_from_source(
              q.get_context(), syclexp::source_language::sycl, src);
      sycl::kernel_bundle<sycl::bundle_state::executable> kb_exe =
          syclexp::build(kb_src, syclexp::properties{syclexp::build_options{
                                     std::vector<std::string>{"--auto-pch"}}}

          );
    }
  };

  Test(R"""(
#define SYCL_SIMPLE_SWIZZLES
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/ext/oneapi/kernel_properties/properties.hpp>
#include <sycl/vector.hpp>

namespace syclext = sycl::ext::oneapi;
namespace syclexp = sycl::ext::oneapi::experimental;

extern "C"
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void iota(sycl::vec<int, 2> *p) {
    size_t id = syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
    p[id] = p[id].xx();
}
)""");
}
