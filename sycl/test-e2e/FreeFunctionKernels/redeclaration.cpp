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

template <typename T> int check_result(T *ptr, T value) {
  for (size_t i = 0; i < NUM; ++i) {
    const T expected = value + static_cast<T>(i);
    if (ptr[i] != expected) {
      std::cout << "Kernel execution did not produce the expected result\n";
      return 1;
    }
  }
  return 0;
}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void free_func(int *ptr, int start);

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void free_func(int *ptr, int start);

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void free_func1(int *ptr, int start);

void free_func1(int *ptr, int start);

template <typename T>
static int call_kernel_code(sycl::queue &q, sycl::kernel &kernel, T value) {
  T *ptr = sycl::malloc_shared<T>(NUM, q);
  q.submit([&](sycl::handler &cgh) {
     if (value == 0)
       cgh.set_args(ptr);
     else
       cgh.set_args(ptr, value);
     sycl::nd_range ndr{{NUM}, {WGSIZE}};
     cgh.parallel_for(ndr, kernel);
   }).wait();
  const int ret = check_result(ptr, value);
  sycl::free(ptr, q);
  return ret;
}

#define KERNEL_CODE(start, ptr, type)                                          \
  size_t id =                                                                  \
      syclext::this_work_item::get_nd_item<1>().get_global_linear_id();        \
  ptr[id] = static_cast<type>(start) + static_cast<type>(id);

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void free_func(int *ptr, int start) { KERNEL_CODE(start, ptr, int); }

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void free_func1(int *ptr, int start) { KERNEL_CODE(start, ptr, int); }

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void free_func2(int *ptr, int start) { KERNEL_CODE(start, ptr, int); }

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void free_func2(float *ptr, float start) { KERNEL_CODE(start, ptr, float); }

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void free_func2(int *ptr) { KERNEL_CODE(0, ptr, int); }

template <auto Func, typename T>
int test_declarations(sycl::queue &q, sycl::context &ctxt, T value) {
  auto exe_bndl =
      syclexp::get_kernel_bundle<Func, sycl::bundle_state::executable>(ctxt);
  sycl::kernel k_func = exe_bndl.template ext_oneapi_get_kernel<Func>();
  return call_kernel_code<T>(q, k_func, value);
}

int main() {
  sycl::queue q;
  sycl::context ctxt = q.get_context();

  int result{0};
  result |= test_declarations<free_func, int>(q, ctxt, 3);
  result |= test_declarations<free_func1, int>(q, ctxt, 3);
  result |=
      test_declarations<static_cast<void (*)(int *, int)>(free_func2), int>(
          q, ctxt, 3);
  result |= test_declarations<static_cast<void (*)(float *, float)>(free_func2),
                              float>(q, ctxt, 3.14f);
  result |= test_declarations<static_cast<void (*)(int *)>(free_func2), int>(
      q, ctxt, 0);
  return result;
}
