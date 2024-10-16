// Tests whether or not cluster launch was successful, with the correct ranges
// that were passed via enqueue functions extension
// REQUIRES: aspect-ext_oneapi_cuda_cluster_group
// RUN: %{build} -Xsycl-target-backend --cuda-gpu-arch=sm_90 -o %t.out
// RUN: %{run} %t.out

#include <sycl/ext/oneapi/experimental/cluster_group_prop.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/queue.hpp>
#include <sycl/usm.hpp>

#include <string>

template <int Dim>
int test_cluster_launch_enqueue_functions(sycl::queue &Queue,
                                          sycl::range<Dim> GlobalRange,
                                          sycl::range<Dim> LocalRange,
                                          sycl::range<Dim> ClusterRange) {
  using namespace sycl::ext::oneapi::experimental;

  cuda::cluster_size ClusterDims(ClusterRange);
  properties ClusterLaunchProperty{ClusterDims};

  int *CorrectResultFlag = sycl::malloc_device<int>(1, Queue);
  Queue.memset(CorrectResultFlag, 0, sizeof(int)).wait();

  submit_with_event(Queue, [&](sycl::handler &CGH) {
    nd_launch(CGH,
              launch_config(sycl::nd_range<Dim>(GlobalRange, LocalRange),
                            ClusterLaunchProperty),
              [=](sycl::nd_item<Dim> It) {
                uint32_t ClusterDimX, ClusterDimY, ClusterDimZ;
// Temporary solution till cluster group class is implemented
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SYCL_CUDA_ARCH__) &&            \
    (__SYCL_CUDA_ARCH__ >= 900)
                asm volatile("\n\t"
                             "mov.u32 %0, %%cluster_nctaid.x; \n\t"
                             "mov.u32 %1, %%cluster_nctaid.y; \n\t"
                             "mov.u32 %2, %%cluster_nctaid.z; \n\t"
                             : "=r"(ClusterDimZ), "=r"(ClusterDimY),
                               "=r"(ClusterDimX));
#endif
                if constexpr (Dim == 1) {
                  if (ClusterDimZ == ClusterRange[0] && ClusterDimY == 1 &&
                      ClusterDimX == 1) {
                    *CorrectResultFlag = 1;
                  }
                } else if constexpr (Dim == 2) {
                  if (ClusterDimZ == ClusterRange[1] &&
                      ClusterDimY == ClusterRange[0] && ClusterDimX == 1) {
                    *CorrectResultFlag = 1;
                  }
                } else {
                  if (ClusterDimZ == ClusterRange[2] &&
                      ClusterDimY == ClusterRange[1] &&
                      ClusterDimX == ClusterRange[0]) {
                    *CorrectResultFlag = 1;
                  }
                }
              });
  }).wait_and_throw();

  int CorrectResultFlagHost = 0;
  Queue.copy(CorrectResultFlag, &CorrectResultFlagHost, 1).wait();
  return CorrectResultFlagHost;
}

int main() {

  sycl::queue Queue;

  int HostCorrectFlag =
      test_cluster_launch_enqueue_functions(Queue, sycl::range{128, 128, 128},
                                            sycl::range{16, 16, 2},
                                            sycl::range{2, 4, 1}) &&
      test_cluster_launch_enqueue_functions(Queue, sycl::range{512, 1024},
                                            sycl::range{32, 32},
                                            sycl::range{4, 2}) &&
      test_cluster_launch_enqueue_functions(Queue, sycl::range{128},
                                            sycl::range{32}, sycl::range{2}) &&
      test_cluster_launch_enqueue_functions(Queue, sycl::range{16384},
                                            sycl::range{32}, sycl::range{16});

  return !HostCorrectFlag;
}
