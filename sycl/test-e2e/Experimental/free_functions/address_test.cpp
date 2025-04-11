// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// The name mangling for free function kernels currently does not work with PTX.
// UNSUPPORTED: cuda
// UNSUPPORTED-INTENDED: Not implemented yet for Nvidia/AMD backends.

#include <sycl/sycl.hpp>
namespace syclexp = sycl::ext::oneapi::experimental;

template <typename T>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void iota(T start, T *ptr) {
  // ...
}

template void iota<float>(float start, float *ptr);
template void iota<int>(int start, int *ptr);

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::single_task_kernel))
void ping(float *x) {
  // ...
}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::single_task_kernel))
void ping(int *x) {
  // ...
}

template <typename T> void test_iota_kernel_id(sycl::context &ctxt) {
  sycl::kernel_id id = syclexp::get_kernel_id<iota<T>>();
  auto exe_bndl =
      syclexp::get_kernel_bundle<iota<T>, sycl::bundle_state::executable>(ctxt);
  assert(exe_bndl.has_kernel(id) &&
         "Kernel bundle does not contain the expected kernel");
}

template <typename T> void test_ping_kernel_id(sycl::context &ctxt) {
  sycl::kernel_id id = syclexp::get_kernel_id<(void (*)(T *))ping>();
  auto exe_bndl =
      syclexp::get_kernel_bundle<(void (*)(T *))ping,
                                 sycl::bundle_state::executable>(ctxt);
  assert(exe_bndl.has_kernel(id) &&
         "Kernel bundle does not contain the expected kernel");
}

int main() {
  sycl::queue q;
  sycl::context ctxt = q.get_context();

  test_iota_kernel_id<float>(ctxt);
  test_iota_kernel_id<int>(ctxt);
  test_ping_kernel_id<float>(ctxt);
  test_ping_kernel_id<int>(ctxt);
  return 0;
}
