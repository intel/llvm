// REQUIRES: aspect-usm_shared_allocations
// UNSUPPORTED: target-amd
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/16072

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test checks the interaction between a launch_config carrying a
// work_group_scratch_size property and free function kernels launched through
// nd_launch. A launch_config is often shared across several kernels, some of
// which do not use get_work_group_scratch_memory(). Requesting scratch memory
// for such a kernel must be a no-op rather than an error: the kernel has no
// implicit local memory argument to bind the size to, so backends that do not
// support the driver-side work group memory property (e.g. Level Zero) would
// otherwise reject the launch. See intel/llvm#22706.

#include <cassert>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/ext/oneapi/experimental/free_function_traits.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/ext/oneapi/work_group_scratch_memory.hpp>
#include <sycl/usm.hpp>

namespace syclext = sycl::ext::oneapi;
namespace syclexp = sycl::ext::oneapi::experimental;

constexpr int SIZE = 16;

// Does NOT use get_work_group_scratch_memory(): no implicit local arg is
// generated for this kernel.
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void squareNoScratch(int *src, int *dst) {
  size_t Gid = syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
  dst[Gid] = src[Gid] * src[Gid];
}

// Uses get_work_group_scratch_memory(): an implicit local arg is generated and
// the requested scratch size is bound to it.
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void squareWithScratch(int *src, int *dst) {
  size_t Lid = syclext::this_work_item::get_nd_item<1>().get_local_linear_id();
  size_t Gid = syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
  int *Scratch =
      reinterpret_cast<int *>(syclexp::get_work_group_scratch_memory());
  Scratch[Lid] = src[Gid] * src[Gid];
  dst[Gid] = Scratch[Lid];
}

int main() {
  sycl::queue Q;
  int *Src = sycl::malloc_shared<int>(SIZE, Q);
  int *Dst = sycl::malloc_shared<int>(SIZE, Q);
  for (int I = 0; I < SIZE; ++I) {
    Src[I] = I;
    Dst[I] = 0;
  }

  // A launch_config carrying a scratch size, reused for both kernels below.
  syclexp::launch_config Config{
      sycl::nd_range<1>{sycl::range<1>{SIZE}, sycl::range<1>{SIZE}},
      syclexp::properties{
          syclexp::work_group_scratch_size(SIZE * sizeof(int))}};

  // Kernel that does not use scratch: requesting scratch must be a no-op, not
  // an error. Exercise both the queue and handler nd_launch paths.
  syclexp::nd_launch(Q, Config, syclexp::kernel_function<squareNoScratch>, Src,
                     Dst);
  Q.wait();
  for (int I = 0; I < SIZE; ++I)
    assert(Dst[I] == Src[I] * Src[I]);

  for (int I = 0; I < SIZE; ++I)
    Dst[I] = 0;
  Q.submit([&](sycl::handler &CGH) {
     syclexp::nd_launch(CGH, Config, syclexp::kernel_function<squareNoScratch>,
                        Src, Dst);
   }).wait();
  for (int I = 0; I < SIZE; ++I)
    assert(Dst[I] == Src[I] * Src[I]);

  // Kernel that does use scratch: the requested size must be honored.
  for (int I = 0; I < SIZE; ++I)
    Dst[I] = 0;
  syclexp::nd_launch(Q, Config, syclexp::kernel_function<squareWithScratch>, Src,
                     Dst);
  Q.wait();
  for (int I = 0; I < SIZE; ++I)
    assert(Dst[I] == Src[I] * Src[I]);

  sycl::free(Src, Q);
  sycl::free(Dst, Q);
  return 0;
}
