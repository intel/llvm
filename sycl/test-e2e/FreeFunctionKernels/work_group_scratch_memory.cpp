// REQUIRES: aspect-usm_shared_allocations

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test verifies that we can compile, run and get correct results when
// using a free function kernel that uses the work group scratch memory feature.

#include <sycl/ext/oneapi/work_group_static.hpp>

#include "helpers.hpp"
#include <cassert>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/usm.hpp>

namespace syclext = sycl::ext::oneapi;
namespace syclexp = sycl::ext::oneapi::experimental;

constexpr int SIZE = 16;

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void double_kernel(float *src, float *dst) {
  size_t lid = syclext::this_work_item::get_nd_item<1>().get_local_linear_id();

  float *local_mem = (float *)syclexp::get_work_group_scratch_memory();

  for (int i = 0; i < SIZE; i++) {
    local_mem[lid] = 2 * src[i];
    dst[i] = local_mem[i];
  }
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
  sycl::kernel k = kbndl.template ext_oneapi_get_kernel<double_kernel>();

  syclexp::launch_config cfg{
      ::sycl::nd_range<1>(::sycl::range<1>(SIZE), ::sycl::range<1>(SIZE)),
      syclexp::properties{
          syclexp::work_group_scratch_size(SIZE * sizeof(float))}};

  syclexp::nd_launch(q, cfg, k, src, dst);
  q.wait();

  for (int i = 0; i < SIZE; i++) {
    assert(dst[i] == 2 * src[i]);
  }

  sycl::free(src, q);
  sycl::free(dst, q);
  return 0;
}
