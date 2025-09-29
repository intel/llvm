// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/get_kernel_info.hpp>
#include <sycl/kernel_bundle.hpp>

namespace syclext = sycl::ext::oneapi;
namespace syclexp = sycl::ext::oneapi::experimental;

static constexpr size_t NUM = 1024;
static constexpr size_t WGSIZE = 16;
static constexpr auto FFTestMark = "Free function Kernel Test:";

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<2>))
void func_range(float start, float *ptr) {}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::single_task_kernel))
void func_single(float start, float *ptr) {}

template <auto *Func> int test_free_function_kernel_id(sycl::context &ctxt) {
  sycl::kernel_id id = syclexp::get_kernel_id<Func>();
  auto exe_bndl =
      syclexp::get_kernel_bundle<Func, sycl::bundle_state::executable>(ctxt);
  const auto kernel_ids = exe_bndl.get_kernel_ids();
  const bool res =
      std::find(kernel_ids.begin(), kernel_ids.end(), id) == kernel_ids.end();
  if (res)
    std::cout << FFTestMark << "test_kernel_id failed: kernel id is not found"
              << std::endl;
  return res;
}

template <auto *Func>
int test_kernel_bundle_ctxt(sycl::context &ctxt, std::string_view fname) {
  sycl::kernel_id id = syclexp::get_kernel_id<Func>();
  auto exe_bndl =
      syclexp::get_kernel_bundle<Func, sycl::bundle_state::executable>(ctxt);
  const bool res =
      exe_bndl.has_kernel(id) &&
      exe_bndl.get_kernel(id)
              .template get_info<sycl::info::kernel::function_name>() == fname;
  if (!res)
    std::cout
        << FFTestMark
        << "test_kernel_bundle_ctxt failed: bundle does not contain kernel id "
           "or function name "
        << fname << std::endl;
  return res;
}

template <auto *Func>
int test_kernel_bundle_ctxt_dev(sycl::context &ctxt, sycl::device &dev,
                                std::string_view fname) {
  sycl::kernel_id id = syclexp::get_kernel_id<Func>();
  auto exe_bndl =
      syclexp::get_kernel_bundle<Func, sycl::bundle_state::executable>(ctxt,
                                                                       {dev});
  const bool res =
      exe_bndl.has_kernel(id) &&
      exe_bndl.get_kernel(id)
              .template get_info<sycl::info::kernel::function_name>() == fname;
  if (!res)
    std::cout << FFTestMark
              << "test_kernel_bundle_ctxt_dev failed: bundle does not contain "
                 "kernel id "
                 "or function name "
              << fname << std::endl;
  return res;
}

int main() {
  sycl::queue q;
  sycl::context ctxt = q.get_context();
  sycl::device dev = q.get_device();

  int ret = test_free_function_kernel_id<func_range>(ctxt);
  ret |= test_free_function_kernel_id<func_single>(ctxt);
  ret |= test_kernel_bundle_ctxt<func_range>(ctxt, "func_range");
  ret |= test_kernel_bundle_ctxt<func_single>(ctxt, "func_single");
  ret |= test_kernel_bundle_ctxt_dev<func_range>(ctxt, dev, "func_range");
  ret |= test_kernel_bundle_ctxt_dev<func_single>(ctxt, dev, "func_single");
  return ret;
}
