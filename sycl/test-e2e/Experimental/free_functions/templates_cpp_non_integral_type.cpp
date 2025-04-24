// REQUIRES: aspect-usm_shared_allocations
// RUN: %{build} %cxx_std_optionc++20 -o %t.out
// RUN: %{run} %t.out

// The name mangling for free function kernels currently does not work with PTX.
// UNSUPPORTED: cuda
// UNSUPPORTED-INTENDED: Not implemented yet for Nvidia/AMD backends.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/usm.hpp>

namespace syclext = sycl::ext::oneapi;
namespace syclexp = sycl::ext::oneapi::experimental;

static constexpr size_t NUM = 1024;
static constexpr size_t WGSIZE = 16;

class TestClass {};

struct TestStruct {};

namespace free_functions::tests {
class TestClass {};

struct TestStruct {};
} // namespace free_functions::tests

using AliasType = float;

template <typename T>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void tempalted_func(T start, T *ptr) {
  size_t id = syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
}

template <typename T>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void templated_func_uses_id(sycl::id<1> idx, T *ptr) {
  size_t id = syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
  idx[0] = static_cast<int>(id);
}

template <typename T = TestStruct>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void tempalted_func_with_default_type(T start, T *ptr) {
  size_t id = syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
}

template <typename T, class Y>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void tempalted_func_with_different_types(T *ptr1, Y *ptr2) {
  size_t id = syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
}

template <typename T>
static void call_kernel_code(sycl::queue &q, sycl::kernel &kernel) {
  T *ptr = sycl::malloc_shared<T>(NUM, q);
  q.submit([&](sycl::handler &cgh) {
     cgh.set_args(3.14f, ptr);
     sycl::nd_range ndr{{NUM}, {WGSIZE}};
     cgh.parallel_for(ndr, kernel);
   }).wait();
  sycl::free(ptr, q);
}

template <typename T>
static void call_kernel_code_with_id(sycl::queue &q, sycl::kernel &kernel) {
  T *ptr = sycl::malloc_shared<T>(NUM, q);
  q.submit([&](sycl::handler &cgh) {
     cgh.set_args(sycl::id<1>(0), ptr);
     sycl::nd_range ndr{{NUM}, {WGSIZE}};
     cgh.parallel_for(ndr, kernel);
   }).wait();
  sycl::free(ptr, q);
}

template <typename T, class Y>
static void call_kernel_code_with_different_types(sycl::queue &q,
                                                  sycl::kernel &kernel) {
  T *ptr = sycl::malloc_shared<T>(NUM, q);
  Y *ptr2 = sycl::malloc_shared<Y>(NUM, q);
  q.submit([&](sycl::handler &cgh) {
     cgh.set_args(ptr, ptr2);
     sycl::nd_range ndr{{NUM}, {WGSIZE}};
     cgh.parallel_for(ndr, kernel);
   }).wait();
  sycl::free(ptr, q);
  sycl::free(ptr2, q);
}

template <typename T>
void test_tempalted_func(sycl::queue &q, sycl::context &ctxt) {
  // Get a kernel bundle that contains the free function kernel
  // "tempalted_func".
  auto exe_bndl =
      syclexp::get_kernel_bundle<tempalted_func<T>,
                                 sycl::bundle_state::executable>(ctxt);
  // Get a kernel object for the "tempalted_func" function from that bundle.
  sycl::kernel k_tempalted_func =
      exe_bndl.template ext_oneapi_get_kernel<tempalted_func<T>>();
  call_kernel_code<T>(q, k_tempalted_func);
}

template <typename T>
void test_templated_func_with_id(sycl::queue &q, sycl::context &ctxt) {
  auto exe_bndl =
      syclexp::get_kernel_bundle<templated_func_uses_id<T>,
                                 sycl::bundle_state::executable>(ctxt);
  sycl::kernel k_tempalted_func =
      exe_bndl.template ext_oneapi_get_kernel<templated_func_uses_id<T>>();
  call_kernel_code_with_id<T>(q, k_tempalted_func);
}

template <typename T = TestStruct>
void test_templated_func_with_default_type(sycl::queue &q,
                                           sycl::context &ctxt) {
  auto exe_bndl =
      syclexp::get_kernel_bundle<tempalted_func_with_default_type<T>,
                                 sycl::bundle_state::executable>(ctxt);
  sycl::kernel k_tempalted_func = exe_bndl.template ext_oneapi_get_kernel<
      tempalted_func_with_default_type<T>>();
  call_kernel_code<T>(q, k_tempalted_func);
}

template <typename T, class Y>
void test_templated_func_with_different_types(sycl::queue &q,
                                              sycl::context &ctxt) {
  auto exe_bndl =
      syclexp::get_kernel_bundle<tempalted_func_with_different_types<T, Y>,
                                 sycl::bundle_state::executable>(ctxt);
  sycl::kernel k_tempalted_func = exe_bndl.template ext_oneapi_get_kernel<
      tempalted_func_with_different_types<T, Y>>();
  call_kernel_code_with_different_types<T, Y>(q, k_tempalted_func);
}

// TODO: Add tests to check accessors

int main() {
  sycl::queue q;
  sycl::context ctxt = q.get_context();
  test_tempalted_func<TestClass>(q, ctxt);
  test_tempalted_func<TestStruct>(q, ctxt);
  test_tempalted_func<free_functions::tests::TestClass>(q, ctxt);
  test_tempalted_func<free_functions::tests::TestStruct>(q, ctxt);
  test_tempalted_func<AliasType>(q, ctxt);
  test_tempalted_func<sycl::id<1>>(q, ctxt);
  test_tempalted_func<sycl::range<2>>(q, ctxt);
  test_tempalted_func<sycl::marray<int, 4>>(q, ctxt);
  test_tempalted_func<sycl::vec<int, 4>>(q, ctxt);
  test_templated_func_with_default_type<free_functions::tests::TestClass>(q,
                                                                          ctxt);
  test_templated_func_with_default_type(q, ctxt);
  test_templated_func_with_different_types<TestClass, TestStruct>(q, ctxt);
  return 0;
}
