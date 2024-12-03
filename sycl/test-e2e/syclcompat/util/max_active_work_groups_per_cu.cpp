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
 *  SYCLcompat
 *
 *  max_active_work_groups_per_cu.cpp
 *
 *  Description:
 *    Test the syclcompat::max_active_work_groups_per_cu API
 **************************************************************************/
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "sycl/accessor.hpp"
#include <sycl/detail/core.hpp>
#include <syclcompat/util.hpp>

template <class T, size_t Dim>
using sycl_global_accessor =
    sycl::accessor<T, Dim, sycl::access::mode::read_write,
                   sycl::access::target::global_buffer>;

using value_type = int;

template <int RangeDim> struct MyKernel {
  MyKernel(sycl_global_accessor<value_type, RangeDim> acc) : acc_{acc} {}
  void operator()(sycl::nd_item<RangeDim> item) const {
    auto gid = item.get_global_id();
    acc_[gid] = item.get_global_linear_id();
  }
  sycl_global_accessor<value_type, RangeDim> acc_;
  static constexpr bool has_local_mem = false;
};

template <int RangeDim> struct MyLocalMemKernel {
  MyLocalMemKernel(sycl_global_accessor<value_type, RangeDim> acc,
                   sycl::local_accessor<value_type, RangeDim> lacc)
      : acc_{acc}, lacc_{lacc} {}
  void operator()(sycl::nd_item<RangeDim> item) const {
    auto gid = item.get_global_id();
    acc_[gid] = item.get_global_linear_id();
    auto lid = item.get_local_id();
    lacc_[lid] = item.get_global_linear_id();
  }
  sycl_global_accessor<value_type, RangeDim> acc_;
  sycl::local_accessor<value_type, RangeDim> lacc_;
  static constexpr bool has_local_mem = true;
};

template <template <int> class KernelName, int RangeDim>
void test_max_active_work_groups_per_cu(sycl::queue q,
                                        sycl::range<RangeDim> wg_range,
                                        size_t local_mem_size = 0) {
  if constexpr (!KernelName<RangeDim>::has_local_mem)
    assert(local_mem_size == 0 && "Bad test setup");

  size_t max_per_cu = syclcompat::max_active_work_groups_per_cu<KernelName<RangeDim>>(
      wg_range, local_mem_size, q);
 
  // Check we get the same result passing equivalent dim3
  syclcompat::dim3 wg_dim3{wg_range};
  size_t max_per_cu_dim3 = syclcompat::max_active_work_groups_per_cu<KernelName<RangeDim>>(
      wg_dim3, local_mem_size, q);
  assert(max_per_cu == max_per_cu_dim3);

  // Compare w/ reference impl
  size_t max_compute_units =
      q.get_device().get_info<sycl::info::device::max_compute_units>();
  namespace syclex = sycl::ext::oneapi::experimental;
  auto ctx = q.get_context();
  auto bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(ctx);
  auto kernel = bundle.template get_kernel<KernelName<RangeDim>>();
  size_t max_wgs = kernel.template ext_oneapi_get_info<
      syclex::info::kernel_queue_specific::max_num_work_groups>(
      q, sycl::range<3>{syclcompat::dim3{wg_range}}, local_mem_size);
  assert(max_per_cu == max_wgs / max_compute_units);

  // We aren't interested in the launch, it's here to define the kernel
  if (false) {
    sycl::range<RangeDim> global_range = wg_range;
    if(max_per_cu > 0)
      global_range[0] = global_range[0] * max_per_cu * max_compute_units;
    sycl::nd_range<RangeDim> my_range{global_range, wg_range};
    sycl::buffer<value_type, RangeDim> buf{global_range};

    q.submit([&](sycl::handler &cgh) {
      auto acc = buf.template get_access<sycl::access::mode::read_write>(cgh);
      if constexpr (KernelName<RangeDim>::has_local_mem) {
        sycl::local_accessor<value_type, RangeDim> lacc(
            my_range.get_local_range(), cgh);
        cgh.parallel_for(my_range, KernelName<RangeDim>{acc, lacc});
      } else {
        cgh.parallel_for(my_range, KernelName<RangeDim>{acc});
      }
    });
  }
}

int main() {
  sycl::queue q{};
  sycl::range<1> range_1d{32};
  sycl::range<2> range_2d{1, 32};
  sycl::range<3> range_3d{1, 1, 32};
  syclcompat::dim3 wg_dim3{32, 1, 1};

  size_t lmem_size_small = sizeof(value_type) * 32;
  size_t lmem_size_medium = lmem_size_small * 32;
  size_t lmem_size_large = lmem_size_medium * 32;

  test_max_active_work_groups_per_cu<MyKernel, 3>(q, range_3d);
  test_max_active_work_groups_per_cu<MyKernel, 2>(q, range_2d);
  test_max_active_work_groups_per_cu<MyKernel, 1>(q, range_1d);
  test_max_active_work_groups_per_cu<MyLocalMemKernel, 3>(q, range_3d,
                                                          lmem_size_small);
  test_max_active_work_groups_per_cu<MyLocalMemKernel, 3>(q, range_3d,
                                                          lmem_size_medium);
  test_max_active_work_groups_per_cu<MyLocalMemKernel, 3>(q, range_3d,
                                                          lmem_size_large);
  test_max_active_work_groups_per_cu<MyLocalMemKernel, 1>(q, range_1d,
                                                          lmem_size_large);
  return 0;
}
