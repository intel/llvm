// REQUIRES: cuda
// RUN: %{build} -Xsycl-target-backend --cuda-gpu-arch=sm_90 -o %t.out
// RUN: %{run} %t.out

#include <sycl/ext/oneapi/experimental/cluster_group_prop.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/sycl.hpp>

#include <string>

int main() {
  using namespace sycl::ext::oneapi::experimental;

  sycl::queue queue;
  auto computeCapability =
      std::stof(queue.get_device().get_info<sycl::info::device::backend_version>());

  if (computeCapability >= 9.0) {
  cluster_size cluster_dims(sycl::range<3>(2, 2, 1));
  properties cluster_launch_property{cluster_dims};

  int *correct_result_flag = sycl::malloc_device<int>(1, queue);
  queue.memset(correct_result_flag, 1, sizeof(int)).wait();

  submit_with_event(queue, [&](sycl::handler &cgh) {

        nd_launch(cgh, launch_config(sycl::nd_range<3>({64, 64, 1}, {32, 32, 1}),
                         cluster_launch_property), [=](sycl::nd_item<3> it) {
                           uint32_t cluster_dim_x, cluster_dim_y, cluster_dim_z;
// Temporary solution till cluster group class is implemented
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SYCL_CUDA_ARCH__) && (__SYCL_CUDA_ARCH__ >= 900)
                           asm volatile("\n\t"
                                        "mov.u32 %0, %%cluster_nctaid.x; \n\t"
                                        "mov.u32 %1, %%cluster_nctaid.y; \n\t"
                                        "mov.u32 %2, %%cluster_nctaid.z; \n\t"
                                        : "=r"(cluster_dim_z),
                                          "=r"(cluster_dim_y),
                                          "=r"(cluster_dim_x));
#endif

                           if (cluster_dim_z != 1 || cluster_dim_y != 2 ||
                               cluster_dim_x != 2) {
                             *correct_result_flag = 0;
                           }
                         });
      })
      .wait_and_throw();

  int correct_result_flag_host = 1;
  queue.copy(correct_result_flag, &correct_result_flag_host, 1).wait();

  if (!correct_result_flag_host) {
    std::cerr << "Cluster Dimensions did not match " << std::endl;
  }

  return !correct_result_flag_host;
  }
  return 0;
}
