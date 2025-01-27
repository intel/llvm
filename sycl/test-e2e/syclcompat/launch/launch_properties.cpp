/***************************************************************************
 *
 *  Copyright (C) Codeplay Software Ltd.
 *
 *  Part of the LLVM Project, under the Apache License v2.0 with LLVM
 *  Exceptions. See https://llvm.org/LICENSE.txt for license information.
 *  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  SYCLcompat API
 *
 *  launch_properties.cpp
 *
 *  Description:
 *     launch<F> with launch properties tests - test cluster_dims passed
 *     correctly. Adapted from
 *     sycl/test-e2e/ClusterLaunch/cluster_launch_parallel_for.cpp
 **************************************************************************/

// REQUIRES: aspect-ext_oneapi_cuda_cluster_group
// REQUIRES: build-and-run-mode
// RUN: %{build} -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_90 -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/cluster_group_prop.hpp>
#include <sycl/ext/oneapi/properties/properties.hpp>

#include <syclcompat/launch.hpp>
#include <syclcompat/memory.hpp>

namespace compat_exp = syclcompat::experimental;
namespace sycl_exp = sycl::ext::oneapi::experimental;

template <int Dim>
void cluster_launch_kernel(sycl::range<Dim> cluster_range,
                           int *correct_result_flag) {
  uint32_t cluster_dim_x, cluster_dim_y, cluster_dim_z;
// Temporary solution till cluster group class is implemented
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SYCL_CUDA_ARCH__) &&            \
    (__SYCL_CUDA_ARCH__ >= 900)
  asm volatile("\n\t"
               "mov.u32 %0, %%cluster_nctaid.x; \n\t"
               "mov.u32 %1, %%cluster_nctaid.y; \n\t"
               "mov.u32 %2, %%cluster_nctaid.z; \n\t"
               : "=r"(cluster_dim_z), "=r"(cluster_dim_y), "=r"(cluster_dim_x));
#endif
  if constexpr (Dim == 1) {
    if (cluster_dim_z == cluster_range[0] && cluster_dim_y == 1 &&
        cluster_dim_x == 1) {
      *correct_result_flag = 1;
    }
  } else if constexpr (Dim == 2) {
    if (cluster_dim_z == cluster_range[1] && cluster_dim_y == cluster_range[0] &&
        cluster_dim_x == 1) {
      *correct_result_flag = 1;
    }
  } else {
    if (cluster_dim_z == cluster_range[2] && cluster_dim_y == cluster_range[1] &&
        cluster_dim_x == cluster_range[0]) {
      *correct_result_flag = 1;
    }
  }
};

template <int Dim>
int test_cluster_launch_parallel_for(sycl::range<Dim> global_range,
                                     sycl::range<Dim> local_range,
                                     sycl::range<Dim> cluster_range) {

  sycl_exp::cuda::cluster_size cluster_dims(cluster_range);

  int *correct_result_flag = syclcompat::malloc<int>(1);
  syclcompat::memset(correct_result_flag, 0, sizeof(int));

  compat_exp::launch_policy policy{global_range, local_range,
                                   compat_exp::launch_properties{cluster_dims}};
  compat_exp::launch<cluster_launch_kernel<Dim>>(policy, cluster_range,
                                                 correct_result_flag);

  int correct_result_flag_host = 0;
  syclcompat::memcpy<int>(&correct_result_flag_host, correct_result_flag, 1);
  return correct_result_flag_host;
}

int main() {

  sycl::queue Queue;

  int host_correct_flag =
      test_cluster_launch_parallel_for(sycl::range{128, 128, 128},
                                       sycl::range{16, 16, 2},
                                       sycl::range{2, 4, 1}) &&
      test_cluster_launch_parallel_for(
          sycl::range{512, 1024}, sycl::range{32, 32}, sycl::range{4, 2}) &&
      test_cluster_launch_parallel_for(sycl::range{128}, sycl::range{32},
                                       sycl::range{2}) &&
      test_cluster_launch_parallel_for(sycl::range{16384}, sycl::range{32},
                                       sycl::range{16});

  return !host_correct_flag;
}
