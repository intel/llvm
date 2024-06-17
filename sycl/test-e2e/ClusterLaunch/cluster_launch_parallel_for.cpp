// Tests whether or not cluster launch was successful, with the correct ranges
// that were passed via parallel for overload
// REQUIRES: cuda
// RUN: %{build} -Xsycl-target-backend --cuda-gpu-arch=sm_90 -o %t.out
// RUN: %{run} %t.out

#include <sycl/ext/oneapi/experimental/cluster_group_prop.hpp>
#include <sycl/queue.hpp>
#include <sycl/usm.hpp>

#include <string>

template <int Dim>
int test_cluster_launch_parallel_for(sycl::queue &queue,
                                     sycl::range<Dim> global_range,
                                     sycl::range<Dim> local_range,
                                     sycl::range<Dim> cluster_range) {
  using namespace sycl::ext::oneapi::experimental;

  cuda::cluster_size cluster_dims(cluster_range);
  properties cluster_launch_property{cluster_dims};

  int *correct_result_flag = sycl::malloc_device<int>(1, queue);
  queue.memset(correct_result_flag, 0, sizeof(int)).wait();

  queue
      .submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl::nd_range<Dim>(global_range, local_range),
            cluster_launch_property, [=](sycl::nd_item<Dim> it) {
              uint32_t cluster_dim_x, cluster_dim_y, cluster_dim_z;
// Temporary solution till cluster group class is implemented
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SYCL_CUDA_ARCH__) &&            \
    (__SYCL_CUDA_ARCH__ >= 900)
              asm volatile("\n\t"
                           "mov.u32 %0, %%cluster_nctaid.x; \n\t"
                           "mov.u32 %1, %%cluster_nctaid.y; \n\t"
                           "mov.u32 %2, %%cluster_nctaid.z; \n\t"
                           : "=r"(cluster_dim_z), "=r"(cluster_dim_y),
                             "=r"(cluster_dim_x));
#endif
              if constexpr (Dim == 1) {
                if (cluster_dim_z == cluster_range[0] && cluster_dim_y == 1 &&
                    cluster_dim_x == 1) {
                  *correct_result_flag = 1;
                }
              } else if constexpr (Dim == 2) {
                if (cluster_dim_z == cluster_range[1] &&
                    cluster_dim_y == cluster_range[0] && cluster_dim_x == 1) {
                  *correct_result_flag = 1;
                }
              } else {
                if (cluster_dim_z == cluster_range[2] &&
                    cluster_dim_y == cluster_range[1] &&
                    cluster_dim_x == cluster_range[0]) {
                  *correct_result_flag = 1;
                }
              }
            });
      })
      .wait_and_throw();

  int correct_result_flag_host = 0;
  queue.copy(correct_result_flag, &correct_result_flag_host, 1).wait();
  return correct_result_flag_host;
}

int main() {

  sycl::queue queue;
  auto computeCapability = std::stof(
      queue.get_device().get_info<sycl::info::device::backend_version>());

  if (computeCapability < 9.0) {
    printf("Cluster group not supported on this arch, exiting...\n");
    return 0;
  }

  int host_correct_flag =
      test_cluster_launch_parallel_for(queue, sycl::range{128, 128, 128},
                                       sycl::range{16, 16, 2},
                                       sycl::range{2, 4, 1}) &&
      test_cluster_launch_parallel_for(queue, sycl::range{512, 1024},
                                       sycl::range{32, 32},
                                       sycl::range{4, 2}) &&
      test_cluster_launch_parallel_for(queue, sycl::range{128}, sycl::range{32},
                                       sycl::range{2}) &&
      test_cluster_launch_parallel_for(queue, sycl::range{16384},
                                       sycl::range{32}, sycl::range{16});

  return !host_correct_flag;
}
