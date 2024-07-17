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
 *     launch<F> with launch properties tests - test ClusterDims passed
 *     correctly. Adapted from
 *     sycl/test-e2e/ClusterLaunch/cluster_launch_parallel_for.cpp
 **************************************************************************/

// REQUIRES: aspect-ext_oneapi_cuda_cluster_group
// RUN: %clangxx -std=c++20 -fsycl -fsycl-targets=%{sycl_triple} -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_90 %s -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/cluster_group_prop.hpp>
#include <sycl/ext/oneapi/properties/properties.hpp>

#include <syclcompat/launch.hpp>
#include <syclcompat/memory.hpp>

namespace compat_exp = syclcompat::experimental;
namespace sycl_exp = sycl::ext::oneapi::experimental;

template <int Dim>
void cluster_launch_kernel(sycl::range<Dim> ClusterRange,
                           int *CorrectResultFlag) {
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
    if (ClusterDimZ == ClusterRange[0] && ClusterDimY == 1 &&
        ClusterDimX == 1) {
      *CorrectResultFlag = 1;
    }
  } else if constexpr (Dim == 2) {
    if (ClusterDimZ == ClusterRange[1] && ClusterDimY == ClusterRange[0] &&
        ClusterDimX == 1) {
      *CorrectResultFlag = 1;
    }
  } else {
    if (ClusterDimZ == ClusterRange[2] && ClusterDimY == ClusterRange[1] &&
        ClusterDimX == ClusterRange[0]) {
      *CorrectResultFlag = 1;
    }
  }
};

template <int Dim>
int test_cluster_launch_parallel_for(sycl::range<Dim> GlobalRange,
                                     sycl::range<Dim> LocalRange,
                                     sycl::range<Dim> ClusterRange) {

  sycl_exp::cuda::cluster_size ClusterDims(ClusterRange);
  sycl_exp::properties ClusterLaunchProperty{ClusterDims};

  int *CorrectResultFlag = syclcompat::malloc<int>(1);
  syclcompat::memset(CorrectResultFlag, 0, sizeof(int));

  compat_exp::launch_policy policy{GlobalRange, LocalRange,
                                   compat_exp::launch_properties{ClusterDims}};
  compat_exp::launch<cluster_launch_kernel<Dim>>(policy, ClusterRange,
                                                 CorrectResultFlag);

  int CorrectResultFlagHost = 0;
  syclcompat::memcpy<int>(&CorrectResultFlagHost, CorrectResultFlag, 1);
  return CorrectResultFlagHost;
}

int main() {

  sycl::queue Queue;

  int HostCorrectFlag =
      test_cluster_launch_parallel_for(sycl::range{128, 128, 128},
                                       sycl::range{16, 16, 2},
                                       sycl::range{2, 4, 1}) &&
      test_cluster_launch_parallel_for(
          sycl::range{512, 1024}, sycl::range{32, 32}, sycl::range{4, 2}) &&
      test_cluster_launch_parallel_for(sycl::range{128}, sycl::range{32},
                                       sycl::range{2}) &&
      test_cluster_launch_parallel_for(sycl::range{16384}, sycl::range{32},
                                       sycl::range{16});

  return !HostCorrectFlag;
}
