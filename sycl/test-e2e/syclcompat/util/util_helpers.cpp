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
 *  util_helpers.cpp
 *
 *  Description:
 *    generic utility helpers tests
 **************************************************************************/

// The original source was under the license below:
// ====------ UtilSelectFromSubGroup.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

// RUN: %clangxx -fsycl -fsycl-targets=%{sycl_triple} %s -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include <syclcompat/util.hpp>

void test_reinterpreted_queue_ptr() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  sycl::queue q;
  sycl::queue *q_ptr = &q;
  uintptr_t reinterpreted_q = reinterpret_cast<uintptr_t>(q_ptr);
  assert(q_ptr == syclcompat::int_as_queue_ptr(reinterpreted_q));
}

void test_default_queue_from_int_as_queue_ptr() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  // Check that int_as_queue_ptr with x < 2 maps to the default queue.
  // Queue addresses may not be equal, but the queues should have the same
  // device.
  auto default_name = syclcompat::get_default_queue()
                          .get_device()
                          .get_info<sycl::info::device::name>();
  auto int_as_queue_name = syclcompat::int_as_queue_ptr(1)
                               ->get_device()
                               .get_info<sycl::info::device::name>();

  assert(default_name == int_as_queue_name);
}

void foo(sycl::float2 *x, int n, sycl::nd_item<3> item_ct1, float f = .1) {}

void test_args_selector() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  int n = 2;
  sycl::float2 *a = syclcompat::malloc_host<sycl::float2>(n);
  a[0] = {1.0, 2.0};
  a[1] = {3.0, 4.0};
  float f = .1;

  void *kernelParams[3] = {
      static_cast<void *>(&a),
      static_cast<void *>(&n),
      static_cast<void *>(&f),
  };

  syclcompat::args_selector<2, 1, decltype(foo)> selector(kernelParams,
                                                          nullptr);
  auto &a_ref = selector.get<0>();
  auto &n_ref = selector.get<1>();
  auto &f_ref = selector.get<2>();

  assert(a_ref[0][0] == 1.0);
  assert(a_ref[0][1] == 2.0);
  assert(a_ref[1][0] == 3.0);
  assert(a_ref[1][1] == 4.0);
  assert(n_ref == 2);
  assert(f_ref == .1f);

  syclcompat::free(a);
}

int main() {
  test_reinterpreted_queue_ptr();
  test_default_queue_from_int_as_queue_ptr();
  test_args_selector();
  return 0;
}
