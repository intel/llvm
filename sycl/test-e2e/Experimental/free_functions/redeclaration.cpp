// REQUIRES: aspect-usm_shared_allocations
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// The name mangling for free function kernels currently does not work with PTX.
// UNSUPPORTED: cuda, hip
// UNSUPPORTED-INTENDED: Not implemented yet for Nvidia/AMD backends.

#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/usm.hpp>

namespace syclext = sycl::ext::oneapi;
namespace syclexp = sycl::ext::oneapi::experimental;

static constexpr size_t NUM = 1024;
static constexpr size_t WGSIZE = 16;

int check_result(int *ptr) {
  for (size_t i = 0; i < NUM; ++i) {
    const float expected = 3 + static_cast<int>(i);
    if (ptr[i] != expected) {
      std::cout << "Kernel execution did not produce the expected result\n";
      return 1;
    }
  }
  return 0;
}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void free_func(int start, int *ptr);

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void free_func(int start, int *ptr);

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void free_func1(int start, int *ptr);

void free_func1(int start, int *ptr);

static int call_kernel_code(sycl::queue &q, sycl::kernel &kernel) {
  int *ptr = sycl::malloc_shared<int>(NUM, q);
  q.submit([&](sycl::handler &cgh) {
     cgh.set_args(3, ptr);
     sycl::nd_range ndr{{NUM}, {WGSIZE}};
     cgh.parallel_for(ndr, kernel);
   }).wait();
  const int ret = check_result(ptr);
  sycl::free(ptr, q);
  return ret;
}

template <auto Func>
int test_declarations(sycl::queue &q, sycl::context &ctxt) {
  auto exe_bndl =
      syclexp::get_kernel_bundle<Func, sycl::bundle_state::executable>(ctxt);
  sycl::kernel k_func = exe_bndl.template ext_oneapi_get_kernel<Func>();
  return call_kernel_code(q, k_func);
}

#define KERNEL_CODE(start, ptr)                                                \
  size_t id =                                                                  \
      syclext::this_work_item::get_nd_item<1>().get_global_linear_id();        \
  ptr[id] = start + static_cast<int>(id);

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void free_func(int start, int *ptr) { KERNEL_CODE(start, ptr); }

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void free_func1(int start, int *ptr) { KERNEL_CODE(start, ptr); }

int main() {
  sycl::queue q;
  sycl::context ctxt = q.get_context();

  int result{0};
  result |= test_declarations<free_func>(q, ctxt);
  result |= test_declarations<free_func1>(q, ctxt);
  return result;
}
