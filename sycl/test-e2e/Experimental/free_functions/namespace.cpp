// REQUIRES: aspect-usm_shared_allocations
// RUN: %{build} -o %t.out
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

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void func(float start, float *ptr) {
  size_t id = syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
  ptr[id] = start + static_cast<float>(id);
}

namespace free_functions::tests {
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void function_in_ns(float start, float *ptr) {
  size_t id = syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
  ptr[id] = start + static_cast<float>(id);
}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void func(float start, float *ptr) {
  size_t id = syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
  ptr[id] = start + static_cast<float>(id);
}
} // namespace free_functions::tests

namespace free_functions::tests {
inline namespace V1 {
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void function_in_inline_ns(float start, float *ptr) {
  size_t id = syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
  ptr[id] = start + static_cast<float>(id);
}
} // namespace V1
} // namespace free_functions::tests

namespace {
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void function_in_anonymous_ns(float start, float *ptr) {
  size_t id = syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
  ptr[id] = start + static_cast<float>(id);
}
} // namespace

struct TestClass {
  float data;
  TestClass(float d) : data(d) {}
};

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void function_with_test_class(float start, TestClass *ptr) {
  size_t id = syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
  ptr[id].data = start + static_cast<float>(id);
}

template <typename T> void check_result(T *ptr) {
  for (size_t i = 0; i < NUM; ++i) {
    const float expected = 3.14f + static_cast<float>(i);
    assert(ptr[i] == expected &&
           "Kernel execution did not produce the expected result");
  }
}

template <> void check_result(TestClass *ptr) {
  for (size_t i = 0; i < NUM; ++i) {
    const float expected = 3.14f + static_cast<float>(i);
    assert(ptr[i].data == expected &&
           "Kernel execution did not produce the expected result");
  }
}

template <typename T>
static void call_kernel_code(sycl::queue &q, sycl::kernel &kernel) {
  T *ptr = sycl::malloc_shared<T>(NUM, q);
  q.submit([&](sycl::handler &cgh) {
     cgh.set_args(3.14f, ptr);
     sycl::nd_range ndr{{NUM}, {WGSIZE}};
     cgh.parallel_for(ndr, kernel);
   }).wait();
  check_result(ptr);
}

void test_function_without_ns(sycl::queue &q, sycl::context &ctxt) {
  // Get a kernel bundle that contains the free function kernel
  // "func".
  auto exe_bndl =
      syclexp::get_kernel_bundle<func, sycl::bundle_state::executable>(ctxt);
  // Get a kernel object for the "func" function from that bundle.
  sycl::kernel k_func = exe_bndl.ext_oneapi_get_kernel<func>();
  call_kernel_code<float>(q, k_func);
}

void test_function_in_ns(sycl::queue &q, sycl::context &ctxt) {
  // Get a kernel bundle that contains the free function kernel
  // "function_in_ns".
  auto exe_bndl =
      syclexp::get_kernel_bundle<free_functions::tests::function_in_ns,
                                 sycl::bundle_state::executable>(ctxt);

  // Get a kernel object for the "function_in_ns" function from that bundle.
  sycl::kernel k_function_in_ns =
      exe_bndl.ext_oneapi_get_kernel<free_functions::tests::function_in_ns>();
  call_kernel_code<float>(q, k_function_in_ns);
}

void test_func_in_ns_with_same_name(sycl::queue &q, sycl::context &ctxt) {
  // Get a kernel bundle that contains the free function kernel
  // "func".
  auto exe_bndl =
      syclexp::get_kernel_bundle<free_functions::tests::func,
                                 sycl::bundle_state::executable>(ctxt);

  // Get a kernel object for the "func" function from that bundle.
  sycl::kernel k_func_in_ns =
      exe_bndl.ext_oneapi_get_kernel<free_functions::tests::func>();
  call_kernel_code<float>(q, k_func_in_ns);
}

void test_function_in_inline_ns(sycl::queue &q, sycl::context &ctxt) {
  // Get a kernel bundle that contains the free function kernel
  // "function_in_inline_ns".
  auto exe_bndl =
      syclexp::get_kernel_bundle<free_functions::tests::function_in_inline_ns,
                                 sycl::bundle_state::executable>(ctxt);

  // Get a kernel object for the "function_in_inline_ns" function from that
  // bundle.
  sycl::kernel k_function_in_inline_ns = exe_bndl.ext_oneapi_get_kernel<
      free_functions::tests::function_in_inline_ns>();
  call_kernel_code<float>(q, k_function_in_inline_ns);
}

void test_function_in_anonymous_ns(sycl::queue &q, sycl::context &ctxt) {
  // Get a kernel bundle that contains the free function kernel
  // "function_in_anonymous_ns".
  auto exe_bndl =
      syclexp::get_kernel_bundle<function_in_anonymous_ns,
                                 sycl::bundle_state::executable>(ctxt);

  // Get a kernel object for the "function_in_anonymous_ns" function from that
  // bundle.
  sycl::kernel k_function_in_anonymous_ns =
      exe_bndl.ext_oneapi_get_kernel<function_in_anonymous_ns>();
  call_kernel_code<float>(q, k_function_in_anonymous_ns);
}

void test_function_with_class(sycl::queue &q, sycl::context &ctxt) {
  // Get a kernel bundle that contains the free function kernel
  // "function_with_test_class".
  auto exe_bndl =
      syclexp::get_kernel_bundle<function_with_test_class,
                                 sycl::bundle_state::executable>(ctxt);

  // Get a kernel object for the "function_with_test_class" function from that
  // bundle.
  sycl::kernel k_function_with_test_class =
      exe_bndl.template ext_oneapi_get_kernel<function_with_test_class>();
  // call_kernel_code_with_class(q, k_function_with_test_class);
  call_kernel_code<TestClass>(q, k_function_with_test_class);
}

int main() {
  sycl::queue q;
  sycl::context ctxt = q.get_context();

  test_function_without_ns(q, ctxt);
  test_function_in_ns(q, ctxt);
  test_function_in_inline_ns(q, ctxt);
  test_function_in_anonymous_ns(q, ctxt);
  test_func_in_ns_with_same_name(q, ctxt);
  test_function_with_class(q, ctxt);
  return 0;
}
