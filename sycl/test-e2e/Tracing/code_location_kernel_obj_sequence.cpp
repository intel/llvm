// REQUIRES: level_zero || opencl
// RUN: %{build} -std=c++20 -o %t.out
// RUN: %{run} sycl-trace --sycl --print-format=verbose %t.out | FileCheck %s

// Tests that nd_launch reports the caller's code location when the arguments of
// a sycl::kernel are passed as a sequence.

#include <span>

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/ext/oneapi/experimental/raw_kernel_arg.hpp>
#include <sycl/kernel_bundle.hpp>

#include <vector>

namespace syclex = sycl::ext::oneapi::experimental;

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclex::nd_range_kernel<1>))
void takeValue(int) {}

int main() {
  sycl::queue Q;
  auto Bundle =
      syclex::get_kernel_bundle<takeValue, sycl::bundle_state::executable>(
          Q.get_context());
  sycl::kernel Kernel = Bundle.ext_oneapi_get_kernel<takeValue>();
  sycl::nd_range<1> Range{1, 1};

  int Value = 1;
  std::vector<syclex::raw_kernel_arg> Args{{&Value, sizeof(Value)}};
  std::span<const syclex::raw_kernel_arg> Span{Args};

  // CHECK: [SYCL] Task begin
  // CHECK-DAG: sym_source_file_name : {{.*}}code_location_kernel_obj_sequence.cpp
  // CHECK-DAG: sym_line_no : [[# @LINE + 2 ]]
  // CHECK: [SYCL] Task end
  syclex::nd_launch(Q, Range, Kernel, Args);
  Q.wait();

  // CHECK: [SYCL] Task begin
  // CHECK-DAG: sym_source_file_name : {{.*}}code_location_kernel_obj_sequence.cpp
  // CHECK-DAG: sym_line_no : [[# @LINE + 2 ]]
  // CHECK: [SYCL] Task end
  syclex::nd_launch(Q, Range, Kernel, Span);
  Q.wait();
}
