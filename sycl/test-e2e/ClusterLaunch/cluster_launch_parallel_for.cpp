// Tests whether or not cluster launch was successful, with the correct ranges
// that were passed via parallel for overload
// REQUIRES: aspect-ext_oneapi_cuda_cluster_group
// RUN: %{build} -Xsycl-target-backend --cuda-gpu-arch=sm_90 -o %t.out
// RUN: %{run} %t.out

#include <sycl/ext/oneapi/experimental/cluster_group_prop.hpp>
#include <sycl/queue.hpp>
#include <sycl/usm.hpp>

#include <string>

template <int Dim, typename T> struct KernelFunctor {
  int *mCorrectResultFlag;
  T mClusterLaunchProperty;
  sycl::range<Dim> mClusterRange;
  KernelFunctor(int *CorrectResultFlag, T ClusterLaunchProperty,
                sycl::range<Dim> ClusterRange)
      : mCorrectResultFlag(CorrectResultFlag),
        mClusterLaunchProperty(ClusterLaunchProperty),
        mClusterRange(ClusterRange) {}

  void operator()(sycl::nd_item<Dim> It) const {
    uint32_t ClusterDimX, ClusterDimY, ClusterDimZ;
// Temporary solution till cluster group class is implemented
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SYCL_CUDA_ARCH__) &&            \
    (__SYCL_CUDA_ARCH__ >= 900)
    asm volatile("\n\t"
                 "mov.u32 %0, %%cluster_nctaid.x; \n\t"
                 "mov.u32 %1, %%cluster_nctaid.y; \n\t"
                 "mov.u32 %2, %%cluster_nctaid.z; \n\t"
                 : "=r"(ClusterDimZ), "=r"(ClusterDimY), "=r"(ClusterDimX));
#endif
    if constexpr (Dim == 1) {
      if (ClusterDimZ == mClusterRange[0] && ClusterDimY == 1 &&
          ClusterDimX == 1) {
        *mCorrectResultFlag = 1;
      }
    } else if constexpr (Dim == 2) {
      if (ClusterDimZ == mClusterRange[1] && ClusterDimY == mClusterRange[0] &&
          ClusterDimX == 1) {
        *mCorrectResultFlag = 1;
      }
    } else {
      if (ClusterDimZ == mClusterRange[2] && ClusterDimY == mClusterRange[1] &&
          ClusterDimX == mClusterRange[0]) {
        *mCorrectResultFlag = 1;
      }
    }
  }
  auto get(sycl::ext::oneapi::experimental::properties_tag) const {
    return mClusterLaunchProperty;
  }
};

template <int Dim>
int test_cluster_launch_parallel_for(sycl::queue &Queue,
                                     sycl::range<Dim> GlobalRange,
                                     sycl::range<Dim> LocalRange,
                                     sycl::range<Dim> ClusterRange) {
  using namespace sycl::ext::oneapi::experimental;

  cuda::cluster_size ClusterDims(ClusterRange);
  properties ClusterLaunchProperty{ClusterDims};

  int *CorrectResultFlag = sycl::malloc_device<int>(1, Queue);
  Queue.memset(CorrectResultFlag, 0, sizeof(int)).wait();

  Queue
      .submit([&](sycl::handler &CGH) {
        CGH.parallel_for(
            sycl::nd_range<Dim>(GlobalRange, LocalRange),
            KernelFunctor<Dim, decltype(ClusterLaunchProperty)>(
                CorrectResultFlag, ClusterLaunchProperty, ClusterRange));
      })
      .wait_and_throw();

  int CorrectResultFlagHost = 0;
  Queue.copy(CorrectResultFlag, &CorrectResultFlagHost, 1).wait();
  return CorrectResultFlagHost;
}

int main() {

  sycl::queue Queue;

  int HostCorrectFlag =
      test_cluster_launch_parallel_for(Queue, sycl::range{128, 128, 128},
                                       sycl::range{16, 16, 2},
                                       sycl::range{2, 4, 1}) &&
      test_cluster_launch_parallel_for(Queue, sycl::range{512, 1024},
                                       sycl::range{32, 32},
                                       sycl::range{4, 2}) &&
      test_cluster_launch_parallel_for(Queue, sycl::range{128}, sycl::range{32},
                                       sycl::range{2}) &&
      test_cluster_launch_parallel_for(Queue, sycl::range{16384},
                                       sycl::range{32}, sycl::range{16});

  return !HostCorrectFlag;
}
