// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/get_kernel_info.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/usm.hpp>

namespace syclext = sycl::ext::oneapi;
namespace syclexp = sycl::ext::oneapi::experimental;

static constexpr size_t NUM = 1024;
static constexpr size_t WGSIZE = 16;
static constexpr auto FFTestMark = "Free function Kernel Test:";

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<2>))
void func_range(float start, float *ptr) {}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::single_task_kernel))
void func_single(float start, float *ptr) {}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::single_task_kernel))
void kernel_func(sycl::item<1> idx, float value, sycl::accessor<int, 1> acc) {}

template <auto *Func>
int test_num_args_free_function_api(sycl::context &ctxt, sycl::device &dev,
                                    const int expected_num_args) {
  const int actual =
      syclexp::get_kernel_info<Func, sycl::info::kernel::num_args>(ctxt, dev);
  const bool res = actual == expected_num_args;
  if (!res)
    std::cout << FFTestMark << "test_num_args failed: expected_num_args "
              << expected_num_args << " actual " << actual << std::endl;
  return res ? 0 : 1;
}

template <auto *Func>
int test_num_args_kernel_api(sycl::context &ctxt, const int expected_num_args) {
  auto bundle =
      syclexp::get_kernel_bundle<Func, sycl::bundle_state::executable>(ctxt);
  const int actual = bundle.template ext_oneapi_get_kernel<Func>()
                         .template get_info<sycl::info::kernel::num_args>();
  const bool res = actual == expected_num_args;
  if (!res)
    std::cout << FFTestMark
              << "test_num_args_kernel_api failed: expected_num_args "
              << expected_num_args << " actual " << actual << std::endl;
  return res ? 0 : 1;
}

template <auto *Func>
int test_num_args_kernel_id(sycl::context &ctxt, const int expected_num_args) {
  auto KernelId = syclexp::get_kernel_id<Func>();
  auto Bundle =
      syclexp::get_kernel_bundle<Func, sycl::bundle_state::executable>(ctxt);
  if (Bundle.has_kernel(KernelId)) {
    sycl::kernel Kernel = Bundle.get_kernel(KernelId);
    unsigned actual = Kernel.get_info<sycl::info::kernel::num_args>();
    const bool res = actual == expected_num_args;
    if (!res)
      std::cout << FFTestMark
                << "test_num_args_kernel_id failed: expected_num_args "
                << expected_num_args << " actual " << actual << std::endl;
    return res ? 0 : 1;
  }
  return 1;
}

int main() {
  sycl::queue q;
  sycl::context ctx = q.get_context();
  sycl::device dev = q.get_device();

  int ret = test_num_args_free_function_api<func_range>(ctx, dev, 2);
  ret |= test_num_args_free_function_api<func_single>(ctx, dev, 2);
  ret |= test_num_args_free_function_api<kernel_func>(ctx, dev, 3);
  ret |= test_num_args_kernel_api<func_range>(ctx, 2);
  ret |= test_num_args_kernel_api<func_single>(ctx, 2);
  ret |= test_num_args_kernel_api<kernel_func>(ctx, 3);
  ret |= test_num_args_kernel_id<func_range>(ctx, 2);
  ret |= test_num_args_kernel_id<func_single>(ctx, 2);
  ret |= test_num_args_kernel_id<kernel_func>(ctx, 3);

  return ret;
}
