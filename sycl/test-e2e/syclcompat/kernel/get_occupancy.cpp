
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

#include <sycl/detail/core.hpp>
#include "sycl/accessor.hpp"
#include <syclcompat/kernel.hpp>

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

  auto ctx = q.get_context();
  auto bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(ctx);
  auto kernel = bundle.template get_kernel<KernelName<RangeDim>>();

  size_t max_per_cu = syclcompat::max_active_work_groups_per_cu(
      kernel, q, wg_range, local_mem_size);
  size_t max_compute_units =
      q.get_device().get_info<sycl::info::device::max_compute_units>();

  std::cout << "max_per_cu: " << max_per_cu << std::endl;
  std::cout << "compute_units: " << max_compute_units << std::endl;

  // We aren't interested in the launch here, it's here to define the kernel
  sycl::range<RangeDim> global_range = wg_range;
  global_range[0] = global_range[0] * max_per_cu * max_compute_units;
  sycl::nd_range<RangeDim> my_range{global_range, wg_range};
  sycl::buffer<value_type, RangeDim> buf{global_range};

  if (false) {
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

  test_max_active_work_groups_per_cu<MyKernel, 3>(q, {32, 1, 1});
  test_max_active_work_groups_per_cu<MyLocalMemKernel, 3>(
      q, {32, 1, 1}, 32 * sizeof(value_type));
  test_max_active_work_groups_per_cu<MyLocalMemKernel, 2>(
      q, {32, 1}, 32 * 200 * sizeof(value_type));
  // test_max_active_work_groups_per_cu<MyKernel<2>, 2>(q); //TODO: template arg
  // here is a mess
  // TODO: What tests cases do we want here?
  // Regular
  // Local mem
  // range dim
  assert(false);
  return 0;
}
