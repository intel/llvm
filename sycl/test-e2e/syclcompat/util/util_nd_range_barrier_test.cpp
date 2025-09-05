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
 *  util_nd_range_barrier_test.cpp
 *
 *  Description:
 *    nd_range_barrier tests
 **************************************************************************/

// The original source was under the license below:
// ====------ UtilNdRangeBarrierTest.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <cstring>

#include <iostream>
#include <stdio.h>
#include <sycl/detail/core.hpp>
#include <syclcompat.hpp>

void kernel_1(sycl::nd_item<3> item_ct1,
              sycl::atomic_ref<
                  unsigned int, syclcompat::experimental::barrier_memory_order,
                  sycl::memory_scope::device,
                  sycl::access::address_space::global_space> &sync_ct1) {
  syclcompat::experimental::nd_range_barrier(item_ct1, sync_ct1);
}

void kernel_2(sycl::nd_item<3> item_ct1,
              sycl::atomic_ref<
                  unsigned int, syclcompat::experimental::barrier_memory_order,
                  sycl::memory_scope::device,
                  sycl::access::address_space::global_space> &sync_ct1) {
  syclcompat::experimental::nd_range_barrier(item_ct1, sync_ct1);

  syclcompat::experimental::nd_range_barrier(item_ct1, sync_ct1);
}

void test_nd_range_barrier_dim3() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  syclcompat::device_ext &dev_ct1 = syclcompat::get_current_device();
  sycl::queue *q_ct1 = dev_ct1.default_queue();
  {
    syclcompat::global_memory<unsigned int, 0> d_sync_ct1;
    unsigned *sync_ct1 = d_sync_ct1.get_ptr(syclcompat::get_default_queue());
    syclcompat::get_default_queue().memset(sync_ct1, 0, sizeof(int)).wait();

    q_ct1
        ->submit([&](sycl::handler &cgh) {
          cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 4) *
                                                 sycl::range<3>(1, 1, 4),
                                             sycl::range<3>(1, 1, 4)),
                           [=](sycl::nd_item<3> item_ct1) {
                             auto atm_sync_ct1 = sycl::atomic_ref<
                                 unsigned int,
                                 syclcompat::experimental::barrier_memory_order,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>(
                                 sync_ct1[0]);
                             kernel_1(item_ct1, atm_sync_ct1);
                           });
        })
        .wait();
  }
  dev_ct1.queues_wait_and_throw();

  {

    syclcompat::global_memory<unsigned int, 0> d_sync_ct1;
    unsigned *sync_ct1 = d_sync_ct1.get_ptr(syclcompat::get_default_queue());
    syclcompat::get_default_queue().memset(sync_ct1, 0, sizeof(int)).wait();

    q_ct1
        ->submit([&](sycl::handler &cgh) {
          cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 4) *
                                                 sycl::range<3>(1, 1, 4),
                                             sycl::range<3>(1, 1, 4)),
                           [=](sycl::nd_item<3> item_ct1) {
                             auto atm_sync_ct1 = sycl::atomic_ref<
                                 unsigned int,
                                 syclcompat::experimental::barrier_memory_order,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>(
                                 sync_ct1[0]);
                             kernel_2(item_ct1, atm_sync_ct1);
                           });
        })
        .wait();
  }
  dev_ct1.queues_wait_and_throw();
}

void kernel_1(sycl::nd_item<1> item_ct1,
              sycl::atomic_ref<
                  unsigned int, syclcompat::experimental::barrier_memory_order,
                  sycl::memory_scope::device,
                  sycl::access::address_space::global_space> &sync_ct1) {
  syclcompat::experimental::nd_range_barrier(item_ct1, sync_ct1);
}

void kernel_2(sycl::nd_item<1> item_ct1,
              sycl::atomic_ref<
                  unsigned int, syclcompat::experimental::barrier_memory_order,
                  sycl::memory_scope::device,
                  sycl::access::address_space::global_space> &sync_ct1) {
  syclcompat::experimental::nd_range_barrier(item_ct1, sync_ct1);

  syclcompat::experimental::nd_range_barrier(item_ct1, sync_ct1);
}

void test_nd_range_barrier_dim1() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  syclcompat::device_ext &dev_ct1 = syclcompat::get_current_device();
  sycl::queue *q_ct1 = dev_ct1.default_queue();

  {
    syclcompat::global_memory<unsigned int, 0> d_sync_ct1;
    unsigned *sync_ct1 = d_sync_ct1.get_ptr(syclcompat::get_default_queue());
    syclcompat::get_default_queue().memset(sync_ct1, 0, sizeof(int)).wait();

    q_ct1
        ->submit([&](sycl::handler &cgh) {
          cgh.parallel_for(
              sycl::nd_range<1>(sycl::range<1>(4) * sycl::range<1>(4),
                                sycl::range<1>(4)),
              [=](sycl::nd_item<1> item_ct1) {
                auto atm_sync_ct1 = sycl::atomic_ref<
                    unsigned int,
                    syclcompat::experimental::barrier_memory_order,
                    sycl::memory_scope::device,
                    sycl::access::address_space::global_space>(sync_ct1[0]);
                kernel_1(item_ct1, atm_sync_ct1);
              });
        })
        .wait();
  }

  dev_ct1.queues_wait_and_throw();
  {
    syclcompat::global_memory<unsigned int, 0> d_sync_ct1;
    unsigned *sync_ct1 = d_sync_ct1.get_ptr(syclcompat::get_default_queue());
    syclcompat::get_default_queue().memset(sync_ct1, 0, sizeof(int)).wait();

    q_ct1
        ->submit([&](sycl::handler &cgh) {
          cgh.parallel_for(
              sycl::nd_range<1>(sycl::range<1>(4) * sycl::range<1>(4),
                                sycl::range<1>(4)),
              [=](sycl::nd_item<1> item_ct1) {
                auto atm_sync_ct1 = sycl::atomic_ref<
                    unsigned int,
                    syclcompat::experimental::barrier_memory_order,
                    sycl::memory_scope::device,
                    sycl::access::address_space::global_space>(sync_ct1[0]);
                kernel_2(item_ct1, atm_sync_ct1);
              });
        })
        .wait();
  }
  dev_ct1.queues_wait_and_throw();
}

int main() {
  test_nd_range_barrier_dim1();
  test_nd_range_barrier_dim3();

  return 0;
}
