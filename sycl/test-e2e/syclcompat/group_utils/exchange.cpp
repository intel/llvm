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
 *  exchange.cpp
 *
 *  Description:
 *    Group exchange API tests
 **************************************************************************/

// ===------- exchange.cpp---------------------- *- C++ -* ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <iostream>

#include <sycl/detail/core.hpp>

#include <syclcompat/device.hpp>
#include <syclcompat/group_utils.hpp>

void StripedToBlockedKernel(int *d_data, const sycl::nd_item<3> &item_ct1,
                            uint8_t *load_temp_storage,
                            uint8_t *store_temp_storage,
                            uint8_t *temp_storage) {
  using BlockLoadT = syclcompat::group::group_load<
      int, 4, syclcompat::group::group_load_algorithm::striped>;
  using BlockStoreT = syclcompat::group::group_store<
      int, 4, syclcompat::group::group_store_algorithm::striped>;
  typedef syclcompat::group::exchange<int, 4> BlockExchange;

  int thread_data[4];
  BlockLoadT(load_temp_storage).load(item_ct1, d_data, thread_data);
  BlockExchange(temp_storage)
      .striped_to_blocked(item_ct1, thread_data, thread_data);
  BlockStoreT(store_temp_storage).store(item_ct1, d_data, thread_data);
}

void BlockedToStripedKernel(int *d_data, const sycl::nd_item<3> &item_ct1,
                            uint8_t *temp_storage) {

  typedef syclcompat::group::exchange<int, 4> BlockExchange;

  int thread_data[4];
  syclcompat::group::load_direct_striped(item_ct1, d_data, thread_data);
  BlockExchange(temp_storage)
      .blocked_to_striped(item_ct1, thread_data, thread_data);
  syclcompat::group::store_direct_striped(item_ct1, d_data, thread_data);
}

void ScatterToBlockedKernel(int *d_data, int *d_rank,
                            const sycl::nd_item<3> &item_ct1,
                            uint8_t *temp_storage) {

  using BlockExchange = syclcompat::group::exchange<int, 4>;

  int thread_data[4], thread_rank[4];
  syclcompat::group::load_direct_striped(item_ct1, d_data, thread_data);
  syclcompat::group::load_direct_striped(item_ct1, d_rank, thread_rank);
  BlockExchange(temp_storage)
      .scatter_to_blocked(item_ct1, thread_data, thread_rank);
  syclcompat::group::store_direct_striped(item_ct1, d_data, thread_data);
}

void ScatterToStripedKernel(int *d_data, int *d_rank,
                            const sycl::nd_item<3> &item_ct1,
                            uint8_t *temp_storage) {

  using BlockExchange = syclcompat::group::exchange<int, 4>;

  int thread_data[4], thread_rank[4];
  syclcompat::group::load_direct_striped(item_ct1, d_data, thread_data);
  syclcompat::group::load_direct_striped(item_ct1, d_rank, thread_rank);
  BlockExchange(temp_storage)
      .scatter_to_striped(item_ct1, thread_data, thread_rank);
  syclcompat::group::store_direct_striped(item_ct1, d_data, thread_data);
}

bool test_striped_to_blocked() {
  syclcompat::device_ext &dev_ct1 = syclcompat::get_current_device();
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  int *d_data;
  d_data = sycl::malloc_shared<int>(512, q_ct1);
  for (int i = 0; i < 128; i++) {
    d_data[4 * i + 0] = i;
    d_data[4 * i + 1] = i + 1 * 128;
    d_data[4 * i + 2] = i + 2 * 128;
    d_data[4 * i + 3] = i + 3 * 128;
  }

  q_ct1.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<uint8_t, 1> load_temp_storage_acc(
        syclcompat::group::group_load<int, 4>::get_local_memory_size(
            sycl::range<3>(1, 1, 128).size()),
        cgh);
    sycl::local_accessor<uint8_t, 1> store_temp_storage_acc(
        syclcompat::group::group_store<int, 4>::get_local_memory_size(
            sycl::range<3>(1, 1, 128).size()),
        cgh);
    sycl::local_accessor<uint8_t, 1> temp_storage_acc(
        syclcompat::group::exchange<int, 4>::get_local_memory_size(
            sycl::range<3>(1, 1, 128).size()),
        cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
        [=](sycl::nd_item<3> item_ct1) {
          StripedToBlockedKernel(d_data, item_ct1, &load_temp_storage_acc[0],
                                 &store_temp_storage_acc[0],
                                 &temp_storage_acc[0]);
        });
  });
  dev_ct1.queues_wait_and_throw();

  for (int i = 0; i < 512; ++i) {
    if (d_data[i] != i) {
      std::cout << "test_striped_to_blocked failed\n";
      std::ostream_iterator<int> Iter(std::cout, ", ");
      std::copy(d_data, d_data + 512, Iter);
      std::cout << std::endl;
      sycl::free(d_data, q_ct1);
      return false;
    }
  }
  sycl::free(d_data, q_ct1);
  std::cout << "test_striped_to_blocked pass\n";
  return true;
}

bool test_blocked_to_striped() {
  syclcompat::device_ext &dev_ct1 = syclcompat::get_current_device();
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  int *d_data, expected[512];
  d_data = sycl::malloc_shared<int>(512, q_ct1);
  for (int i = 0; i < 512; ++i)
    d_data[i] = i;

  q_ct1.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<uint8_t, 1> temp_storage_acc(
        syclcompat::group::exchange<int, 4>::get_local_memory_size(
            sycl::range<3>(1, 1, 128).size()),
        cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
        [=](sycl::nd_item<3> item_ct1) {
          BlockedToStripedKernel(d_data, item_ct1, &temp_storage_acc[0]);
        });
  });
  dev_ct1.queues_wait_and_throw();

  for (int i = 0; i < 128; i++) {
    expected[4 * i + 0] = i;
    expected[4 * i + 1] = i + 1 * 128;
    expected[4 * i + 2] = i + 2 * 128;
    expected[4 * i + 3] = i + 3 * 128;
  }

  for (int i = 0; i < 512; ++i) {
    if (expected[i] != d_data[i]) {
      std::cout << "test_blocked_to_striped failed\n";
      std::ostream_iterator<int> Iter(std::cout, ", ");
      std::copy(d_data, d_data + 512, Iter);
      std::cout << std::endl;
      sycl::free(d_data, q_ct1);
      return false;
    }
  }
  sycl::free(d_data, q_ct1);
  std::cout << "test_blocked_to_striped pass\n";
  return true;
}

bool test_scatter_to_blocked() {
  syclcompat::device_ext &dev_ct1 = syclcompat::get_current_device();
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  int *d_data, *d_rank;
  d_data = sycl::malloc_shared<int>(512, q_ct1);
  d_rank = sycl::malloc_shared<int>(512, q_ct1);
  for (int i = 0; i < 128; i++) {
    d_data[4 * i + 0] = i;
    d_data[4 * i + 1] = i + 1 * 128;
    d_data[4 * i + 2] = i + 2 * 128;
    d_data[4 * i + 3] = i + 3 * 128;
    d_rank[4 * i + 0] = i * 4 + 0;
    d_rank[4 * i + 1] = i * 4 + 1;
    d_rank[4 * i + 2] = i * 4 + 2;
    d_rank[4 * i + 3] = i * 4 + 3;
  }

  q_ct1.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<uint8_t, 1> temp_storage_acc(
        syclcompat::group::exchange<int, 4>::get_local_memory_size(
            sycl::range<3>(1, 1, 128).size()),
        cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
        [=](sycl::nd_item<3> item_ct1) {
          ScatterToBlockedKernel(d_data, d_rank, item_ct1,
                                 &temp_storage_acc[0]);
        });
  });
  dev_ct1.queues_wait_and_throw();

  for (int i = 0; i < 512; ++i) {
    if (d_data[i] != i) {
      std::cout << "test_scatter_to_blocked failed\n";
      std::ostream_iterator<int> Iter(std::cout, ", ");
      std::copy(d_data, d_data + 512, Iter);
      std::cout << std::endl;
      sycl::free(d_data, q_ct1);
      sycl::free(d_rank, q_ct1);
      return false;
    }
  }
  sycl::free(d_data, q_ct1);
  sycl::free(d_rank, q_ct1);
  std::cout << "test_scatter_to_blocked pass\n";
  return true;
}

bool test_scatter_to_striped() {
  syclcompat::device_ext &dev_ct1 = syclcompat::get_current_device();
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  int *d_data, *d_rank, expected[512];
  d_data = sycl::malloc_shared<int>(512, q_ct1);
  d_rank = sycl::malloc_shared<int>(512, q_ct1);
  for (int i = 0; i < 512; ++i)
    d_data[i] = i;

  d_rank[0] = 0;
  d_rank[128] = 1;
  d_rank[256] = 2;
  d_rank[384] = 3;
  for (int i = 1; i < 128; i++) {
    d_rank[0 * 128 + i] = d_rank[0 * 128 + i - 1] + 4;
    d_rank[1 * 128 + i] = d_rank[1 * 128 + i - 1] + 4;
    d_rank[2 * 128 + i] = d_rank[2 * 128 + i - 1] + 4;
    d_rank[3 * 128 + i] = d_rank[3 * 128 + i - 1] + 4;
  }

  q_ct1.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<uint8_t, 1> temp_storage_acc(
        syclcompat::group::exchange<int, 4>::get_local_memory_size(
            sycl::range<3>(1, 1, 128).size()),
        cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
        [=](sycl::nd_item<3> item_ct1) {
          ScatterToStripedKernel(d_data, d_rank, item_ct1,
                                 &temp_storage_acc[0]);
        });
  });
  dev_ct1.queues_wait_and_throw();

  for (int i = 0; i < 128; i++) {
    expected[4 * i + 0] = i + 0 * 128;
    expected[4 * i + 1] = i + 1 * 128;
    expected[4 * i + 2] = i + 2 * 128;
    expected[4 * i + 3] = i + 3 * 128;
  }

  for (int i = 0; i < 512; ++i) {
    if (expected[i] != d_data[i]) {
      std::cout << "test_blocked_to_striped failed\n";
      std::ostream_iterator<int> Iter(std::cout, ", ");
      std::copy(d_data, d_data + 512, Iter);
      std::cout << std::endl;
      sycl::free(d_data, q_ct1);
      sycl::free(d_rank, q_ct1);
      return false;
    }
  }
  sycl::free(d_data, q_ct1);
  sycl::free(d_rank, q_ct1);
  std::cout << "test_blocked_to_striped pass\n";
  return true;
}

int main() {
  return !(test_blocked_to_striped() && test_striped_to_blocked() &&
           test_scatter_to_blocked() && test_scatter_to_striped());
}
