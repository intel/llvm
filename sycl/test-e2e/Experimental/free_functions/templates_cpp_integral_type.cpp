// REQUIRES: aspect-usm_shared_allocations
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// The name mangling for free function kernels currently does not work with PTX.
// UNSUPPORTED: cuda
// UNSUPPORTED-INTENDED: Not implemented yet for Nvidia/AMD backends.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/usm.hpp>
#include <sycl/kernel_bundle.hpp>

namespace syclext = sycl::ext::oneapi;
namespace syclexp = sycl::ext::oneapi::experimental;

static constexpr size_t NUM = 1024;
static constexpr size_t WGSIZE = 16;

template<typename T>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void func_without_ns(T start, T *ptr) {
  size_t id = syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
  ptr[id] = start + static_cast<T>(id);
}

template<typename T, int a, int b>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void func_with_integral_without_ns(T start, T *ptr) {
  size_t id = syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
  ptr[id + a] = start + static_cast<T>(id) + b;
}


namespace free_functions::tests {
  template<typename T>
  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
  void function_in_ns(T start, T *ptr) {
    size_t id = syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
    ptr[id] = start + static_cast<T>(id + 1);
  }

  template<typename T, int a, int b>
  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
  void func_with_integral_with_ns(T start, T *ptr) {
    size_t id = syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
    ptr[id + a] = start + static_cast<T>(id) + b;
  }

} // namespace free_functions::tests

template<typename T, int a, int b = 10>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void func_with_default_template_arg(T start, T *ptr) {
  size_t id = syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
  ptr[id + a] = start + static_cast<T>(id) + b;
}

template<typename T>
static void call_kernel_code(sycl::queue& q, sycl::kernel& kernel) {
  T *ptr = sycl::malloc_shared<T>(NUM, q);
  q.submit([&](sycl::handler &cgh) {
     cgh.set_args(3.14f, ptr);
     sycl::nd_range ndr{{NUM}, {WGSIZE}};
     cgh.parallel_for(ndr, kernel);
   }).wait();
}

template<typename T>
void test_function_without_ns(sycl::queue& q, sycl::context& ctxt)
{
  // Get a kernel bundle that contains the free function kernel "func_without_ns".
  auto exe_bndl =
      syclexp::get_kernel_bundle<func_without_ns<T>, sycl::bundle_state::executable>(ctxt);
  // Get a kernel object for the "func_without_ns" function from that bundle.
  sycl::kernel k_func_without_ns = exe_bndl.template ext_oneapi_get_kernel<func_without_ns<T>>();
  call_kernel_code<T>(q, k_func_without_ns);
}

template<typename T>
void test_function_in_ns(sycl::queue& q, sycl::context& ctxt)
{
  // Get a kernel bundle that contains the free function kernel "function_in_ns".
  auto exe_bndl =
      syclexp::get_kernel_bundle<free_functions::tests::function_in_ns<T>, sycl::bundle_state::executable>(ctxt);
  // Get a kernel object for the "function_in_ns" function from that bundle.
  sycl::kernel k_function_in_ns = exe_bndl.template ext_oneapi_get_kernel<free_functions::tests::function_in_ns<T>>();
  call_kernel_code<T>(q, k_function_in_ns);
}

template<typename T, int a, int b>
void test_function_with_integral_without_ns(sycl::queue& q, sycl::context& ctxt)
{
  // Get a kernel bundle that contains the free function kernel "func_with_integral_without_ns".
  auto exe_bndl =
      syclexp::get_kernel_bundle<func_with_integral_without_ns<T, a, b>, sycl::bundle_state::executable>(ctxt);
  // Get a kernel object for the "func_with_integral_without_ns" function from that bundle.
  sycl::kernel k_func_with_integral_without_ns = exe_bndl.template ext_oneapi_get_kernel<func_with_integral_without_ns<T, a, b>>();
  call_kernel_code<T>(q, k_func_with_integral_without_ns);
}

template<typename T, int a, int b>
void test_function_with_integral_with_ns(sycl::queue& q, sycl::context& ctxt)
{
  // Get a kernel bundle that contains the free function kernel "func_with_integral_with_ns".
  auto exe_bndl =
      syclexp::get_kernel_bundle<free_functions::tests::func_with_integral_with_ns<T, a, b>, sycl::bundle_state::executable>(ctxt);
  // Get a kernel object for the "func_with_integral_with_ns" function from that bundle.
  sycl::kernel k_func_with_integral_with_ns = exe_bndl.template ext_oneapi_get_kernel<free_functions::tests::func_with_integral_with_ns<T, a, b>>();
  call_kernel_code<T>(q, k_func_with_integral_with_ns);
}

template<typename T, int a, int b = 10>
void test_func_with_default_template_arg(sycl::queue& q, sycl::context& ctxt)
{
  // Get a kernel bundle that contains the free function kernel "func_with_integral_with_ns".
  auto exe_bndl =
      syclexp::get_kernel_bundle<free_functions::tests::func_with_integral_with_ns<T, a, b>, sycl::bundle_state::executable>(ctxt);
  // Get a kernel object for the "func_with_integral_with_ns" function from that bundle.
  sycl::kernel k_func_with_integral_with_ns = exe_bndl.template ext_oneapi_get_kernel<free_functions::tests::func_with_integral_with_ns<T, a, b>>();
  call_kernel_code<T>(q, k_func_with_integral_with_ns);
}


int main() {
  sycl::queue q;
  sycl::context ctxt = q.get_context();

  test_function_without_ns<int>(q, ctxt);
  test_function_without_ns<float>(q, ctxt);
  test_function_in_ns<int>(q, ctxt);
  test_function_in_ns<float>(q, ctxt);
  test_function_with_integral_without_ns<int, 0, 1>(q, ctxt);
  test_function_with_integral_without_ns<float, 0, 1>(q, ctxt);
  test_function_with_integral_with_ns<int, 0, 1>(q, ctxt);
  test_function_with_integral_with_ns<float, 0, 1>(q, ctxt);
  test_func_with_default_template_arg<int, 0, 1>(q, ctxt);
  test_func_with_default_template_arg<float, 0>(q, ctxt);

  return 0;
}
