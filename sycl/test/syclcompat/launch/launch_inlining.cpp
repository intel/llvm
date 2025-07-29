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
 *  launch_inlining.cpp
 *
 *  Description:
 *    Ensure kernels are inlined
 **************************************************************************/
// RUN: %clangxx -fsycl -fgpu-inline-threshold=0 %if cl_options %{/clang:-S /clang:-emit-llvm%} %else %{-S -emit-llvm%} %s -o - | FileCheck %s
// We set -fgpu-inline-threshold=0 to disable heuristic inlining for the
// purposes of the test
#include <sycl/detail/core.hpp>
#include <sycl/group_barrier.hpp>
#include <syclcompat/launch.hpp>
#include <syclcompat/memory.hpp>

namespace compat_exp = syclcompat::experimental;
namespace sycl_exp = sycl::ext::oneapi::experimental;
namespace sycl_intel_exp = sycl::ext::intel::experimental;

static constexpr int LOCAL_MEM_SIZE = 1024;

// CHECK: define {{.*}}spir_kernel{{.*}}write_mem_kernel{{.*}} {
// CHECK-NOT: call {{.*}}write_mem_kernel
// CHECK: }

template <typename T> void write_mem_kernel(T *data, int num_elements) {
  const int id =
      sycl::ext::oneapi::this_work_item::get_nd_item<1>().get_global_id(0);
  if (id < num_elements) {
    data[id] = static_cast<T>(id);
  }
};

// CHECK: define {{.*}}spir_kernel{{.*}}dynamic_local_mem_typed_kernel{{.*}} {
// CHECK-NOT: call {{.*}}dynamic_local_mem_typed_kernel
// CHECK: }
template <typename T>
void dynamic_local_mem_typed_kernel(T *data, char *local_mem) {
  constexpr size_t num_elements = LOCAL_MEM_SIZE / sizeof(T);
  T *typed_local_mem = reinterpret_cast<T *>(local_mem);

  const int id =
      sycl::ext::oneapi::this_work_item::get_nd_item<1>().get_global_id(0);
  if (id < num_elements) {
    typed_local_mem[id] = static_cast<T>(id);
  }
  sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_work_group<1>());
  if (id < num_elements) {
    data[id] = typed_local_mem[num_elements - id - 1];
  }
};

int test_write_mem() {
  compat_exp::launch_policy my_dim3_config(syclcompat::dim3{32},
                                           syclcompat::dim3{32});

  const int memsize = 1024;
  int *d_a = (int *)syclcompat::malloc(memsize);
  compat_exp::launch<write_mem_kernel<int>>(my_dim3_config, d_a,
                                            memsize / sizeof(int))
      .wait();

  syclcompat::free(d_a);
  return 0;
}

int test_lmem_launch() {
  int local_mem_size = LOCAL_MEM_SIZE;

  size_t num_elements = local_mem_size / sizeof(int);
  int *d_a = (int *)syclcompat::malloc(local_mem_size);

  compat_exp::launch_policy my_config(
      sycl::nd_range<1>{{256}, {256}},
      compat_exp::local_mem_size(local_mem_size));

  compat_exp::launch<dynamic_local_mem_typed_kernel<int>>(my_config, d_a)
      .wait();

  syclcompat::free(d_a);

  return 0;
}
