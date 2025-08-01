// Checks whether or not event Dependencies are honored by
// urEnqueueKernelLaunchWithArgsExp with cluster dimensions
// REQUIRES: target-nvidia, aspect-ext_oneapi_cuda_cluster_group
// RUN: %{build} -Xsycl-target-backend --cuda-gpu-arch=sm_90 -o %t.out
// RUN: %{run} %t.out

#include <sycl/accessor.hpp>
#include <sycl/buffer.hpp>
#include <sycl/ext/oneapi/experimental/cluster_group_prop.hpp>
#include <sycl/queue.hpp>
#include <sycl/usm.hpp>

#include <vector>

template <typename T> void dummy_kernel(T *Input, int N, sycl::nd_item<1> It) {
#if defined(__SYCL_CUDA_ARCH__) && (__SYCL_CUDA_ARCH__ >= 900)
  auto ID = It.get_global_linear_id();
  uint32_t ClusterDim;
  asm volatile("mov.u32 %0, %%cluster_nctaid.x;" : "=r"(ClusterDim));

  if (ID < N) {
    Input[ID] += static_cast<T>(ClusterDim);
  }
#endif
}

template <typename T1, typename T2> struct KernelFunctor {
  T1 mAcc;
  T2 mClusterLaunchProperty;
  KernelFunctor(T2 ClusterLaunchProperty, T1 Acc)
      : mClusterLaunchProperty(ClusterLaunchProperty), mAcc(Acc) {}

  void operator()(sycl::nd_item<1> It) const {
    dummy_kernel(
        mAcc.template get_multi_ptr<sycl::access::decorated::yes>().get(), 4096,
        It);
  }
  auto get(sycl::ext::oneapi::experimental::properties_tag) const {
    return mClusterLaunchProperty;
  }
};

int main() {

  std::vector<int> HostArray(4096, -20);
  sycl::queue Queue;

  if (!Queue.get_device().has(sycl::aspect::ext_oneapi_cuda_cluster_group)) {
    printf("Cluster group not supported on this arch, exiting...\n");
    return 0;
  }

  {
    sycl::buffer<int, 1> Buff(HostArray.data(), 4096);
    Queue.submit([&](sycl::handler &CGH) {
      auto Acc = Buff.template get_access<sycl::access::mode::write>(CGH);
      CGH.parallel_for(4096, [=](auto i) { Acc[i] = 1; });
    });
    Queue.submit([&](sycl::handler &CGH) {
      using namespace sycl::ext::oneapi::experimental;

      cuda::cluster_size ClusterDims(sycl::range{2});
      properties ClusterLaunchProperty{ClusterDims};
      auto Acc = Buff.template get_access<sycl::access::mode::read_write>(CGH);
      CGH.parallel_for(sycl::nd_range({4096}, {32}),
                       KernelFunctor(ClusterLaunchProperty, Acc));
    });
    Queue.submit([&](sycl::handler &CGH) {
      auto Acc = Buff.template get_access<sycl::access::mode::read_write>(CGH);
      CGH.parallel_for(4096, [=](auto i) { Acc[i] *= 5; });
    });
  }

  for (const auto &V : HostArray) {
    if (V != 15) {
      return 1;
    }
  }
  return 0;
}
