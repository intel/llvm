// RUN: %{build} -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %{run} %t.out

#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/get_kernel_info.hpp>
#include <sycl/kernel_bundle.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<2>))
void func_range(float start, float *ptr) {}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::single_task_kernel))
void func_single(float start, float *ptr) {}

int main() {
  sycl::queue q;
  auto bundle_range =
      syclexp::get_kernel_bundle<func_range, sycl::bundle_state::executable>(
          q.get_context());
  try {
    sycl::kernel k = bundle_range.ext_oneapi_get_kernel<func_single>();
  } catch (const sycl::exception &e) {
    assert(e.code() == sycl::errc::invalid);
    return 0;
  }
  auto bundle_single =
      syclexp::get_kernel_bundle<func_single, sycl::bundle_state::executable>(
          q.get_context());
  try {
    sycl::kernel k = bundle_single.ext_oneapi_get_kernel<func_range>();
  } catch (const sycl::exception &e) {
    assert(e.code() == sycl::errc::invalid);
    return 0;
  }
  return 1;
}
