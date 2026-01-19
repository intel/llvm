// RUN: %{build} -o %t.out
// RUN: %if hip %{ env SYCL_JIT_AMDGCN_PTX_TARGET_CPU=%{amd_arch} %} %{run} %t.out | FileCheck %s

// UNSUPPORTED: target-native_cpu
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/20142

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

  sycl::queue q;
  auto props = syclexp::properties{
      syclexp::build_options{std::vector<std::string>{"--auto-pch"}}};

  try {
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

    // Error when generating PCH:
    // CHECK-LABEL: Device compilation failed
    // CHECK-NEXT:  Detailed information:
    // CHECK-NEXT:  	rtc_0.cpp:2:10: fatal error: 'non-existent.hpp' file not found
    // CHECK-NEXT:      2 | #include "non-existent.hpp"
    // CHECK-NEXT:        |          ^~~~~~~~~~~~~~~~~~

    auto kb_src = syclexp::create_kernel_bundle_from_source(
        q.get_context(), syclexp::source_language::sycl, src);
    auto kb_exe = syclexp::build(kb_src, props);
    return 1;
  } catch (sycl::exception &e) {
    std::cout << e.what() << std::endl;
  }

  try {
    auto kb_src = syclexp::create_kernel_bundle_from_source(
        q.get_context(), syclexp::source_language::sycl,
        R"""(#include "a.hpp")""",
        syclexp::properties{
            syclexp::include_files{"a.hpp", "emit_error = 1;"}});
    // Also when generating, but this time not a preprocessor error:
    // CHECK-LABEL: Device compilation failed
    // CHECK-NEXT:  Detailed information:
    // CHECK-NEXT:  	In file included from rtc_1.cpp:1:
    // CHECK-NEXT:    a.hpp:1:1: error: a type specifier is required for all declarations
    // CHECK-NEXT:      1 | emit_error = 1;
    // CHECK-NEXT:        | ^
    auto kb_exe = syclexp::build(kb_src, props);
    return 2;
  } catch (sycl::exception &e) {
    std::cout << e.what() << std::endl;
  }
  try {
    auto kb_src = syclexp::create_kernel_bundle_from_source(
        q.get_context(), syclexp::source_language::sycl,
        R"""(
#include "a.hpp"
void foo() {
 S s{42};
}
)""",
        syclexp::properties{syclexp::include_files{"a.hpp", "struct S{};"}});
    // PCH generation is fine, error is when using it (outside the preamble):
    // CHECK-LABEL: Device compilation failed
    // CHECK-NEXT:  Detailed information:
    // CHECK-NEXT:  	rtc_2.cpp:4:6: error: excess elements in struct initializer
    // CHECK-NEXT:      4 |  S s{42};
    // CHECK-NEXT:        |      ^~
    auto kb_exe = syclexp::build(kb_src, props);
    return 2;
  } catch (sycl::exception &e) {
    std::cout << e.what() << std::endl;
  }
}
