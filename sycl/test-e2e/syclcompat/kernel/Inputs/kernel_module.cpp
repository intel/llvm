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
 *  kernel_module.cpp
 *
 *  Description:
 *    function implementation used in kernel_function header API tests
 **************************************************************************/

// The original source was under the license below:
// ====------ kernel_module_lin.cpp------------------------ -*- C++ -* ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

#include <sycl/detail/core.hpp>

#include <syclcompat/defs.hpp>

void foo(int *k, sycl::nd_item<3> item_ct1, uint8_t *local_mem) {
  k[item_ct1.get_global_linear_id()] = item_ct1.get_global_linear_id();
}

extern "C" {
SYCLCOMPAT_EXPORT void foo_wrapper(sycl::queue &queue,
                                   const sycl::nd_range<3> &nr,
                                   unsigned int local_mem_size,
                                   void **kernel_params, void **extra) {
  int *k;
  k = (int *)kernel_params[0];
  queue.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<uint8_t, 1> local_acc_ct1(
        sycl::range<1>(local_mem_size), cgh);
    cgh.parallel_for(nr, [=](sycl::nd_item<3> item_ct1) {
      foo(k, item_ct1,
          local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
    });
  });
}
}
