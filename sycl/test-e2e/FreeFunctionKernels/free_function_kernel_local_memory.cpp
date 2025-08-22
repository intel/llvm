// REQUIRES: aspect-usm_shared_allocations

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test verifies that we can compile, run and get correct results when
// using a free function kernel that allocates shared local memory in a kernel
// either by way of th work group scratch memory extension or the work group
// static memory extension.

#include <sycl/ext/oneapi/work_group_static.hpp>

#include "helpers.hpp"
#include <cassert>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/group_barrier.hpp>
#include <sycl/usm.hpp>

namespace syclext = sycl::ext::oneapi;
namespace syclexp = sycl::ext::oneapi::experimental;

constexpr int SIZE = 16;

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void double_kernel(float *src, float *dst) {
  size_t lid = syclext::this_work_item::get_nd_item<1>().get_local_linear_id();
  float *local_mem = (float *)syclexp::get_work_group_scratch_memory();
  local_mem[lid] = 2 * src[lid];
  dst[lid] = local_mem[lid];
}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void square_kernel(float *src, float *dst) {
  sycl::nd_item<1> item = syclext::this_work_item::get_nd_item<1>();
  size_t lid = item.get_local_linear_id();
  syclexp::work_group_static<float[SIZE]> local_mem;
  local_mem[lid] = src[lid] * src[lid];
  sycl::group_barrier(item.get_group());
  if (item.get_group().leader()) { // Check that memory is indeed shared between
                                   // the work group
    for (int i = 0; i < SIZE; ++i)
      assert(local_mem[i] == src[i] * src[i]);
  }
  dst[lid] = local_mem[lid];
}

int main() {
  sycl::queue q;
  float *src = sycl::malloc_shared<float>(SIZE, q);
  float *dst = sycl::malloc_shared<float>(SIZE, q);

  for (int i = 1; i < SIZE; i++) {
    src[i] = i;
  }

  auto kbndl =
      syclexp::get_kernel_bundle<double_kernel, sycl::bundle_state::executable>(
          q.get_context());
  sycl::kernel DoubleKernel =
      kbndl.template ext_oneapi_get_kernel<double_kernel>();
  sycl::kernel SquareKernel =
      kbndl.template ext_oneapi_get_kernel<square_kernel>();
  syclexp::launch_config DoubleKernelcfg{
      ::sycl::nd_range<1>(::sycl::range<1>(SIZE), ::sycl::range<1>(SIZE)),
      syclexp::properties{
          syclexp::work_group_scratch_size(SIZE * sizeof(float))}};
  syclexp::launch_config SquareKernelcfg{
      ::sycl::nd_range<1>(::sycl::range<1>(SIZE), ::sycl::range<1>(SIZE))};

  syclexp::nd_launch(q, DoubleKernelcfg, DoubleKernel, src, dst);
  q.wait();
  for (int i = 0; i < SIZE; i++) {
    assert(dst[i] == 2 * src[i]);
  }

  syclexp::nd_launch(q, SquareKernelcfg, SquareKernel, src, dst);
  q.wait();
  for (int i = 0; i < SIZE; i++) {
    assert(dst[i] == src[i] * src[i]);
  }

  sycl::free(src, q);
  sycl::free(dst, q);
  return 0;
}
