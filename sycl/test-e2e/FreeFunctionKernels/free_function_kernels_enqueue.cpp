// REQUIRES: aspect-usm_shared_allocations

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// XFAIL: target-native_cpu
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/20142

// This test checks that free function kernels can be submitted using the
// enqueued functions defined in the free function kernel extension, namely the
// single_task and the nd_launch functions that take a queue/handler as an
// argument. These were added in https://github.com/intel/llvm/pull/19995.

#include <cassert>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/ext/oneapi/experimental/free_function_traits.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/ext/oneapi/work_group_static.hpp>
#include <sycl/usm.hpp>

namespace syclext = sycl::ext::oneapi;
namespace syclexp = sycl::ext::oneapi::experimental;

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::single_task_kernel))
void successor(int *src, int *dst) { *dst = *src + 1; }

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void square(int *src, int *dst) {
  size_t Lid = syclext::this_work_item::get_nd_item<1>().get_local_linear_id();
  dst[Lid] = src[Lid] * src[Lid];
}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void squareWithScratchMemory(int *src, int *dst) {
  size_t Lid = syclext::this_work_item::get_nd_item<1>().get_local_linear_id();
  int *LocalMem =
      reinterpret_cast<int *>(syclexp::get_work_group_scratch_memory());
  LocalMem[Lid] = src[Lid] * src[Lid];
  dst[Lid] = LocalMem[Lid];
}

constexpr int SIZE = 16;

int main() {
  sycl::queue Q;
  int *Src = sycl::malloc_shared<int>(SIZE, Q);
  int *Dst = sycl::malloc_shared<int>(SIZE, Q);

  for (int I = 0; I < SIZE; I++) {
    Src[I] = I;
  }

  syclexp::launch_config Config{
      ::sycl::nd_range<1>(::sycl::range<1>(SIZE), ::sycl::range<1>(SIZE)),
      syclexp::properties{
          syclexp::work_group_scratch_size(SIZE * sizeof(int))}};

  static_assert(
      std::is_same_v<decltype(syclexp::nd_launch(
                         Q, Config,
                         syclexp::kernel_function<squareWithScratchMemory>, Src,
                         Dst)),
                     void>);

  syclexp::nd_launch(
      Q, Config, syclexp::kernel_function<squareWithScratchMemory>, Src, Dst);
  Q.wait();

  for (int I = 0; I < SIZE; I++) {
    assert(Dst[I] == Src[I] * Src[I]);
  }

  syclexp::nd_launch(
      Q, ::sycl::nd_range<1>(::sycl::range<1>(SIZE), ::sycl::range<1>(SIZE)),
      syclexp::kernel_function<square>, Src, Dst);
  Q.wait();

  for (int I = 0; I < SIZE; I++) {
    assert(Dst[I] == Src[I] * Src[I]);
  }

  static_assert(
      std::is_same_v<decltype(syclexp::single_task(
                         Q, syclexp::kernel_function<successor>, Src, Dst)),
                     void>);
  syclexp::single_task(Q, syclexp::kernel_function<successor>, Src, Dst);
  Q.wait();

  assert(Dst[0] == Src[0] + 1);

  Q.submit([&](sycl::handler &CGH) {
     static_assert(
         std::is_same_v<decltype(syclexp::nd_launch(
                            CGH, Config,
                            syclexp::kernel_function<squareWithScratchMemory>,
                            Src, Dst)),
                        void>);
     syclexp::nd_launch(CGH, Config,
                        syclexp::kernel_function<squareWithScratchMemory>, Src,
                        Dst);
   }).wait();

  for (int I = 0; I < SIZE; I++) {
    assert(Dst[I] == Src[I] * Src[I]);
  }

  Q.submit([&](sycl::handler &CGH) {
     static_assert(
         std::is_same_v<decltype(syclexp::nd_launch(
                            CGH,
                            ::sycl::nd_range<1>(::sycl::range<1>(SIZE),
                                                ::sycl::range<1>(SIZE)),
                            syclexp::kernel_function<square>, Src, Dst)),
                        void>);

     syclexp::nd_launch(
         CGH,
         ::sycl::nd_range<1>(::sycl::range<1>(SIZE), ::sycl::range<1>(SIZE)),
         syclexp::kernel_function<square>, Src, Dst);
   }).wait();

  for (int I = 0; I < SIZE; I++) {
    assert(Dst[I] == Src[I] * Src[I]);
  }

  Q.submit([&](sycl::handler &CGH) {
     static_assert(std::is_same_v<decltype(syclexp::single_task(
                                      CGH, syclexp::kernel_function<successor>,
                                      Src, Dst)),
                                  void>);
     syclexp::single_task(CGH, syclexp::kernel_function<successor>, Src, Dst);
   }).wait();

  assert(Dst[0] == Src[0] + 1);
  return 0;
}
