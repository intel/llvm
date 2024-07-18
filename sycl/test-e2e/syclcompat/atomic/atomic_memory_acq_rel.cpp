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
 *  atomic_memory_acq_rel.cpp
 *
 *  Description:
 *    Tests fetch_add for acquire and release memory ordering
 **************************************************************************/

// The original source was under the license below:
// ====-------------------------------------------------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

// UNSUPPORTED: hip

// RUN: %clangxx -fsycl -fsycl-targets=%{sycl_triple} %if any-device-is-cuda %{ -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_70 %} %s -o %t.out
// RUN: %{run} %t.out

#include <iostream>
#include <numeric>

#include <sycl/detail/core.hpp>

#include <syclcompat/atomic.hpp>

#include "atomic_fixt.hpp"

using namespace sycl;

using address_space = sycl::access::address_space;

template <memory_order order> void test_acquire_global() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  const size_t N_items = 256;
  const size_t N_iters = 1000;

  int error = 0;
  int val[] = {0, 0};

  queue q;
  {
    buffer<int> error_buf(&error, 1);
    buffer<int> val_buf(val, 2);

    q.submit([&](handler &cgh) {
       auto error =
           error_buf.template get_access<access::mode::read_write>(cgh);
       auto val = val_buf.template get_access<access::mode::read_write>(cgh);
       cgh.parallel_for(range<1>(N_items), [=](item<1> it) {
         volatile int *val_p = val.get_pointer();
         auto atm0 =
             atomic_ref<int, memory_order::relaxed, memory_scope::device,
                        address_space::global_space>(val[0]);
         auto atm1 =
             atomic_ref<int, memory_order::relaxed, memory_scope::device,
                        address_space::global_space>(val[1]);
         for (int i = 0; i < N_iters; i++) {
           if (it.get_id(0) == 0) {

             syclcompat::atomic_fetch_add<address_space::global_space, order>(
                 &val[0], 1);
             val_p[1]++;
           } else {
             // syclcompat:: doesn't offer load/store so using sycl::atomic_ref
             // here
             int tmp1 = atm1.load(memory_order::acquire);
             int tmp0 = atm0.load(memory_order::relaxed);
             if (tmp0 < tmp1) {
               error[0] = 1;
             }
           }
         }
       });
     }).wait_and_throw();
  }
  assert(error == 0);
}

template <memory_order order> void test_acquire_local() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  const size_t local_size = 256;
  const size_t N_wgs = 16;
  const size_t global_size = local_size * N_wgs;
  const size_t N_iters = 1000;

  int error = 0;
  int val[] = {0, 0};

  queue q;
  {
    buffer<int> error_buf(&error, 1);
    buffer<int> val_buf(val, 2);

    q.submit([&](handler &cgh) {
       auto error =
           error_buf.template get_access<access::mode::read_write>(cgh);
       local_accessor<int, 1> val(2, cgh);
       cgh.parallel_for(nd_range<1>(global_size, local_size), [=](nd_item<1>
                                                                      it) {
         size_t lid = it.get_local_id(0);
         val[0] = 0;
         val[1] = 0;
         it.barrier(access::fence_space::local_space);
         volatile int *val_p = val.get_pointer();
         auto atm0 =
             atomic_ref<int, memory_order::relaxed, memory_scope::device,
                        address_space::local_space>(val[0]);
         auto atm1 =
             atomic_ref<int, memory_order::relaxed, memory_scope::device,
                        address_space::local_space>(val[1]);
         for (int i = 0; i < N_iters; i++) {
           if (it.get_local_id(0) == 0) {
             syclcompat::atomic_fetch_add<address_space::local_space, order>(
                 &val[0], 1);
             val_p[1]++;
           } else {
             // syclcompat:: doesn't offer load/store so using
             // sycl::atomic_ref here
             int tmp1 = atm1.load(memory_order::acquire);
             int tmp0 = atm0.load(memory_order::relaxed);
             if (tmp0 < tmp1) {
               error[0] = 1;
             }
           }
         }
       });
     }).wait_and_throw();
  }
  assert(error == 0);
}

template <memory_order order> void test_release_global() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  const size_t N_items = 256;
  const size_t N_iters = 1000;

  int error = 0;
  int val[] = {0, 0};

  queue q;
  {
    buffer<int> error_buf(&error, 1);
    buffer<int> val_buf(val, 2);

    q.submit([&](handler &cgh) {
       auto error =
           error_buf.template get_access<access::mode::read_write>(cgh);
       auto val = val_buf.template get_access<access::mode::read_write>(cgh);
       cgh.parallel_for(range<1>(N_items), [=](item<1> it) {
         volatile int *val_p = val.get_pointer();
         auto atm0 =
             atomic_ref<int, memory_order::relaxed, memory_scope::device,
                        address_space::global_space>(val[0]);
         auto atm1 =
             atomic_ref<int, memory_order::relaxed, memory_scope::device,
                        address_space::global_space>(val[1]);
         for (int i = 0; i < N_iters; i++) {
           if (it.get_id(0) == 0) {
             val_p[0]++;
             syclcompat::atomic_fetch_add<address_space::global_space, order>(
                 &val[1], 1);
           } else {
             // syclcompat:: doesn't offer load/store so using sycl::atomic_ref
             // here
             int tmp1 = atm1.load(memory_order::acquire);
             int tmp0 = atm0.load(memory_order::relaxed);
             if (tmp0 < tmp1) {
               error[0] = 1;
             }
           }
         }
       });
     }).wait_and_throw();
  }
  assert(error == 0);
}

template <memory_order order> void test_release_local() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  const size_t local_size = 256;
  const size_t N_wgs = 16;
  const size_t global_size = local_size * N_wgs;
  const size_t N_iters = 1000;

  int error = 0;
  int val[] = {0, 0};

  queue q;
  {
    buffer<int> error_buf(&error, 1);
    buffer<int> val_buf(val, 2);

    q.submit([&](handler &cgh) {
       auto error =
           error_buf.template get_access<access::mode::read_write>(cgh);
       local_accessor<int, 1> val(2, cgh);
       cgh.parallel_for(nd_range<1>(global_size, local_size), [=](nd_item<1>
                                                                      it) {
         size_t lid = it.get_local_id(0);
         val[0] = 0;
         val[1] = 0;
         it.barrier(access::fence_space::local_space);
         volatile int *val_p = val.get_pointer();
         auto atm0 =
             atomic_ref<int, memory_order::relaxed, memory_scope::device,
                        address_space::local_space>(val[0]);
         auto atm1 =
             atomic_ref<int, memory_order::relaxed, memory_scope::device,
                        address_space::local_space>(val[1]);
         for (int i = 0; i < N_iters; i++) {
           if (it.get_local_id(0) == 0) {
             val_p[0]++;
             syclcompat::atomic_fetch_add<address_space::local_space, order>(
                 &val[1], 1);
           } else {
             // syclcompat:: doesn't offer load/store so using
             // sycl::atomic_ref here
             int tmp1 = atm1.load(memory_order::acquire);
             int tmp0 = atm0.load(memory_order::relaxed);
             if (tmp0 < tmp1) {
               error[0] = 1;
             }
           }
         }
       });
     }).wait_and_throw();
  }
  assert(error == 0);
}

int main() {
  queue q;
  std::vector<memory_order> supported_memory_orders =
      q.get_device()
          .get_info<sycl::info::device::atomic_memory_order_capabilities>();

  if (is_supported(supported_memory_orders, memory_order::acq_rel)) {
    // Acquire-release memory order must also support both acquire and release
    // orderings.
    assert(is_supported(supported_memory_orders, memory_order::acquire) &&
           is_supported(supported_memory_orders, memory_order::release));
    test_acquire_global<memory_order::acq_rel>();
    test_acquire_local<memory_order::acq_rel>();
    test_release_global<memory_order::acq_rel>();
    test_release_local<memory_order::acq_rel>();
  }

  if (is_supported(supported_memory_orders, memory_order::seq_cst)) {
    test_acquire_global<memory_order::seq_cst>();
    test_acquire_local<memory_order::seq_cst>();
    test_release_global<memory_order::seq_cst>();
    test_release_local<memory_order::seq_cst>();
  }

  return 0;
}
