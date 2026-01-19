// REQUIRES: aspect-usm_shared_allocations
// UNSUPPORTED: target-amd
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/16072

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// XFAIL: target-native_cpu
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/20142

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
void scratchKernel(float *Src, float *Dst) {
  size_t Lid = syclext::this_work_item::get_nd_item<1>().get_local_linear_id();
  float *LocalMem =
      reinterpret_cast<float *>(syclexp::get_work_group_scratch_memory());
  LocalMem[Lid] = 2 * Src[Lid];
  Dst[Lid] = LocalMem[Lid];
}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void staticKernel(float *Src, float *Dst) {
  sycl::nd_item<1> Item = syclext::this_work_item::get_nd_item<1>();
  size_t Lid = Item.get_local_linear_id();
  syclexp::work_group_static<float[SIZE]> LocalMem;
  LocalMem[Lid] = Src[Lid] * Src[Lid];
  sycl::group_barrier(Item.get_group());
  if (Item.get_group().leader()) { // Check that memory is indeed shared between
                                   // the work group.
    for (int I = 0; I < SIZE; ++I)
      assert(LocalMem[I] == Src[I] * Src[I]);
  }
  Dst[Lid] = LocalMem[Lid];
}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void scratchStaticKernel(float *Src, float *Dst) {
  size_t Lid = syclext::this_work_item::get_nd_item<1>().get_local_linear_id();
  float *ScratchMem =
      reinterpret_cast<float *>(syclexp::get_work_group_scratch_memory());
  syclexp::work_group_static<float[SIZE]> StaticMem;
  ScratchMem[Lid] = Src[Lid];
  StaticMem[Lid] = Src[Lid];
  Dst[Lid] = ScratchMem[Lid] + StaticMem[Lid];
}

int main() {
  sycl::queue Q;
  float *Src = sycl::malloc_shared<float>(SIZE, Q);
  float *Dst = sycl::malloc_shared<float>(SIZE, Q);

  for (int I = 0; I < SIZE; I++) {
    Src[I] = I;
  }

  auto ScratchBndl =
      syclexp::get_kernel_bundle<scratchKernel, sycl::bundle_state::executable>(
          Q.get_context());
  auto StaticBndl =
      syclexp::get_kernel_bundle<staticKernel, sycl::bundle_state::executable>(
          Q.get_context());
  auto ScratchStaticBndl = syclexp::get_kernel_bundle<
      scratchStaticKernel, sycl::bundle_state::executable>(Q.get_context());

  sycl::kernel ScratchKrn =
      ScratchBndl.template ext_oneapi_get_kernel<scratchKernel>();
  sycl::kernel StaticKrn =
      StaticBndl.template ext_oneapi_get_kernel<staticKernel>();
  sycl::kernel ScratchStaticKrn =
      ScratchStaticBndl.template ext_oneapi_get_kernel<scratchStaticKernel>();
  syclexp::launch_config ScratchKernelcfg{
      ::sycl::nd_range<1>(::sycl::range<1>(SIZE), ::sycl::range<1>(SIZE)),
      syclexp::properties{
          syclexp::work_group_scratch_size(SIZE * sizeof(float))}};
  syclexp::launch_config StaticKernelcfg{
      ::sycl::nd_range<1>(::sycl::range<1>(SIZE), ::sycl::range<1>(SIZE))};

  syclexp::nd_launch(Q, ScratchKernelcfg, ScratchKrn, Src, Dst);
  Q.wait();
  for (int I = 0; I < SIZE; I++) {
    assert(Dst[I] == 2 * Src[I]);
  }

  syclexp::nd_launch(Q, StaticKernelcfg, StaticKrn, Src, Dst);
  Q.wait();
  for (int I = 0; I < SIZE; I++) {
    assert(Dst[I] == Src[I] * Src[I]);
  }

  syclexp::nd_launch(Q, ScratchKernelcfg, ScratchStaticKrn, Src, Dst);
  Q.wait();
  for (int I = 0; I < SIZE; I++) {
    assert(Dst[I] == 2 * Src[I]);
  }

  sycl::free(Src, Q);
  sycl::free(Dst, Q);
  return 0;
}
