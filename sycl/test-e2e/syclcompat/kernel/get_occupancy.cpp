
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
#include <syclcompat/kernel.hpp>

template <class T, size_t Dim>
using sycl_global_accessor =
    sycl::accessor<T, Dim, sycl::access::mode::read_write,
                   sycl::access::target::global_buffer>;

using value_type = int;

struct MyKernel {
  MyKernel(sycl_global_accessor<value_type, 1> acc) : acc_{acc} {}
  void operator()(sycl::nd_item<3> item) const {
    auto gid = item.get_global_linear_id();
    acc_[gid] = gid;
  }
  sycl_global_accessor<value_type, 1> acc_;
};

// struct MyLocalMemKernel {
//   MyKernel(sycl_global_accessor<value_type, 1> acc) : acc_{acc} {}
//   void operator()(sycl::nd_item<3> item) const {
//     auto gid = item.get_global_linear_id();
//     acc_[gid] = gid;
//   }
//   sycl_global_accessor<value_type, 1> acc_;
// };
// TODO: local mem kernel

template <class KernelName> void test_max_active_work_groups_per_cu(sycl::queue q) {
  auto ctx = q.get_context();
  auto bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(ctx);
  auto kernel = bundle.template get_kernel<KernelName>();

  sycl::range<3> wg_range{1, 1, 32}; //TODO: do we need to support 1, 2 D?
  size_t num_work_items = 32; // TODO(joe)
  sycl::buffer<value_type, 1> buf{sycl::range<1>{num_work_items}};
  
 //TODO: this is a test 
  namespace syclex = sycl::ext::oneapi::experimental;
  auto maxWGs = kernel.template ext_oneapi_get_info<
      syclex::info::kernel_queue_specific::max_num_work_groups>(
      q, sycl::range<1>{32}, 0);
  std::cout << "maxWGs: " << maxWGs << std::endl;
  size_t max_per_cu = syclcompat::max_active_work_groups_per_cu(kernel, q, wg_range, 0);

  std::cout << "max_per_cu: " << max_per_cu << std::endl;
  assert(false);
  sycl::nd_range<3> my_range{wg_range*20, wg_range};//TODO:
  q.submit([&](sycl::handler &cgh) {
    auto acc = buf.get_access<sycl::access::mode::read_write>(cgh);
    cgh.parallel_for(my_range, MyKernel{acc});
  });
}

int main() {
  sycl::queue q{};

  test_max_active_work_groups_per_cu<MyKernel>(q);
  //TODO: What tests cases do we want here?
  //Regular
  //Local mem
  //range dim
}
