// REQUIRES: aspect-usm_shared_allocations
// UNSUPPORTED: target-amd
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/16072

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test verifies that we can compile, run and get correct results when
// using a free function kernel that allocates shared local memory in a kernel
// either by way of the work group scratch memory extension or the work group
// static memory extension.

#include "helpers.hpp"

#include <cassert>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/ext/oneapi/work_group_static.hpp>
#include <sycl/group_barrier.hpp>
#include <sycl/usm.hpp>

namespace syclext = sycl::ext::oneapi;
namespace syclexp = sycl::ext::oneapi::experimental;

constexpr int SIZE = 16;

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void scratchKernel(float *src, float *dst) {
  size_t lid = syclext::this_work_item::get_nd_item<1>().get_local_linear_id();
  float *localMem =
      reinterpret_cast<float *>(syclexp::get_work_group_scratch_memory());
  localMem[lid] = 2 * src[lid];
  dst[lid] = localMem[lid];
}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void staticKernel(float *src, float *dst) {
  sycl::nd_item<1> item = syclext::this_work_item::get_nd_item<1>();
  size_t lid = item.get_local_linear_id();
  syclexp::work_group_static<float[SIZE]> localMem;
  localMem[lid] = src[lid] * src[lid];
  sycl::group_barrier(item.get_group());
  if (item.get_group().leader()) { // Check that memory is indeed shared between
                                   // the work group.
    for (int i = 0; i < SIZE; ++i)
      assert(localMem[i] == src[i] * src[i]);
  }
  dst[lid] = localMem[lid];
}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void scratchStaticKernel(float *src, float *dst) {
  size_t lid = syclext::this_work_item::get_nd_item<1>().get_local_linear_id();
  float *scratchMem =
      reinterpret_cast<float *>(syclexp::get_work_group_scratch_memory());
  syclexp::work_group_static<float[SIZE]> staticMem;
  scratchMem[lid] = src[lid];
  staticMem[lid] = src[lid];
  dst[lid] = scratchMem[lid] + staticMem[lid];
}

int main() {
  sycl::queue q;
  float *src = sycl::malloc_shared<float>(SIZE, q);
  float *dst = sycl::malloc_shared<float>(SIZE, q);

  for (int i = 0; i < SIZE; i++) {
    src[i] = i;
  }

  auto scratchBndl =
      syclexp::get_kernel_bundle<scratchKernel, sycl::bundle_state::executable>(
          q.get_context());
  auto staticBndl =
      syclexp::get_kernel_bundle<staticKernel, sycl::bundle_state::executable>(
          q.get_context());
  auto scratchStaticBndl = syclexp::get_kernel_bundle<
      scratchStaticKernel, sycl::bundle_state::executable>(q.get_context());

  sycl::kernel scratchKrn =
      scratchBndl.template ext_oneapi_get_kernel<scratchKernel>();
  sycl::kernel staticKrn =
      staticBndl.template ext_oneapi_get_kernel<staticKernel>();
  sycl::kernel scratchStaticKrn =
      scratchStaticBndl.template ext_oneapi_get_kernel<scratchStaticKernel>();
  syclexp::launch_config scratchKernelcfg{
      ::sycl::nd_range<1>(::sycl::range<1>(SIZE), ::sycl::range<1>(SIZE)),
      syclexp::properties{
          syclexp::work_group_scratch_size(SIZE * sizeof(float))}};
  syclexp::launch_config staticKernelcfg{
      ::sycl::nd_range<1>(::sycl::range<1>(SIZE), ::sycl::range<1>(SIZE))};

  syclexp::nd_launch(q, scratchKernelcfg, scratchKrn, src, dst);
  q.wait();
  for (int i = 0; i < SIZE; i++) {
    assert(dst[i] == 2 * src[i]);
  }

  syclexp::nd_launch(q, staticKernelcfg, staticKrn, src, dst);
  q.wait();
  for (int i = 0; i < SIZE; i++) {
    assert(dst[i] == src[i] * src[i]);
  }

  syclexp::nd_launch(q, scratchKernelcfg, scratchStaticKrn, src, dst);
  q.wait();
  for (int i = 0; i < SIZE; i++) {
    assert(dst[i] == 2 * src[i]);
  }

  sycl::free(src, q);
  sycl::free(dst, q);
  return 0;
}
