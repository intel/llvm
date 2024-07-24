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
 *  util_permute_sub_group_by_xor.cpp
 *
 *  Description:
 *    permute_sub_group_by_xor tests
 **************************************************************************/

// The original source was under the license below:
// ====------ UtilPermuteSubGroupByXor.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

// REQUIRES: sg-32
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include <syclcompat.hpp>

#define WARP_SIZE 32
#define DATA_NUM 128

using namespace sycl::ext::oneapi::experimental;

template <typename T = int> void init_data(T *data, int num) {
  for (int i = 0; i < num; i++)
    data[i] = i;
}

template <typename T = int>
void verify_data(T *data, T *expect, int num, int step = 1) {
  for (int i = 0; i < num; i = i + step) {
    assert(data[i] == expect[i]);
  }
}

void permute_sub_group_by_xor1(unsigned int *data, sycl::nd_item<3> item_ct1) {
  int threadid = item_ct1.get_local_id(2) +
                 item_ct1.get_local_id(1) * item_ct1.get_local_range(2) +
                 item_ct1.get_local_id(0) * item_ct1.get_local_range(2) *
                     item_ct1.get_local_range(1) +
                 item_ct1.get_group(2) * item_ct1.get_local_range(2) *
                     item_ct1.get_local_range(1) * item_ct1.get_local_range(0);
  int output = 0;
  output = syclcompat::permute_sub_group_by_xor(item_ct1.get_sub_group(),
                                                threadid, 2);
  data[threadid] = output;
}

void permute_sub_group_by_xor2(unsigned int *data, sycl::nd_item<3> item_ct1) {
  int threadid = item_ct1.get_local_id(2) +
                 item_ct1.get_local_id(1) * item_ct1.get_local_range(2) +
                 item_ct1.get_local_id(0) * item_ct1.get_local_range(2) *
                     item_ct1.get_local_range(1) +
                 item_ct1.get_group(2) * item_ct1.get_local_range(2) *
                     item_ct1.get_local_range(1) * item_ct1.get_local_range(0);
  int output = 0;
  output = syclcompat::permute_sub_group_by_xor(item_ct1.get_sub_group(),
                                                threadid, 1, 8);
  data[threadid] = output;
}

void test_permute_sub_group_by_xor() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  syclcompat::device_ext &dev_ct1 = syclcompat::get_current_device();
  sycl::queue *q_ct1 = dev_ct1.default_queue();
  bool Result = true;
  int *dev_data = nullptr;
  unsigned int *dev_data_u = nullptr;
  sycl::range<3> GridSize(1, 1, 1);
  sycl::range<3> BlockSize(1, 1, 1);
  dev_data = sycl::malloc_device<int>(DATA_NUM, *q_ct1);
  dev_data_u = sycl::malloc_device<unsigned int>(DATA_NUM, *q_ct1);

  GridSize = sycl::range<3>(1, 1, 2);
  BlockSize = sycl::range<3>(1, 2, 32);
  unsigned int expect1[DATA_NUM] = {
      2,   3,   0,   1,   6,   7,   4,   5,   10,  11,  8,   9,   14,  15,  12,
      13,  18,  19,  16,  17,  22,  23,  20,  21,  26,  27,  24,  25,  30,  31,
      28,  29,  34,  35,  32,  33,  38,  39,  36,  37,  42,  43,  40,  41,  46,
      47,  44,  45,  50,  51,  48,  49,  54,  55,  52,  53,  58,  59,  56,  57,
      62,  63,  60,  61,  66,  67,  64,  65,  70,  71,  68,  69,  74,  75,  72,
      73,  78,  79,  76,  77,  82,  83,  80,  81,  86,  87,  84,  85,  90,  91,
      88,  89,  94,  95,  92,  93,  98,  99,  96,  97,  102, 103, 100, 101, 106,
      107, 104, 105, 110, 111, 108, 109, 114, 115, 112, 113, 118, 119, 116, 117,
      122, 123, 120, 121, 126, 127, 124, 125};
  unsigned int host_dev_data_u[DATA_NUM];
  init_data<unsigned int>(host_dev_data_u, DATA_NUM);
  q_ct1->memcpy(dev_data_u, host_dev_data_u, DATA_NUM * sizeof(unsigned int))
      .wait();

  q_ct1->parallel_for(sycl::nd_range<3>(GridSize * BlockSize, BlockSize),
                      [=](sycl::nd_item<3> item_ct1)
                          [[intel::reqd_sub_group_size(32)]] {
                            permute_sub_group_by_xor1(dev_data_u, item_ct1);
                          });

  dev_ct1.queues_wait_and_throw();
  q_ct1->memcpy(host_dev_data_u, dev_data_u, DATA_NUM * sizeof(unsigned int))
      .wait();
  verify_data<unsigned int>(host_dev_data_u, expect1, DATA_NUM);

  GridSize = sycl::range<3>(1, 1, 2);
  BlockSize = sycl::range<3>(1, 2, 32);
  unsigned int expect2[DATA_NUM] = {
      1,   0,   3,   2,   5,   4,   7,   6,   9,   8,   11,  10,  13,  12,  15,
      14,  17,  16,  19,  18,  21,  20,  23,  22,  25,  24,  27,  26,  29,  28,
      31,  30,  33,  32,  35,  34,  37,  36,  39,  38,  41,  40,  43,  42,  45,
      44,  47,  46,  49,  48,  51,  50,  53,  52,  55,  54,  57,  56,  59,  58,
      61,  60,  63,  62,  65,  64,  67,  66,  69,  68,  71,  70,  73,  72,  75,
      74,  77,  76,  79,  78,  81,  80,  83,  82,  85,  84,  87,  86,  89,  88,
      91,  90,  93,  92,  95,  94,  97,  96,  99,  98,  101, 100, 103, 102, 105,
      104, 107, 106, 109, 108, 111, 110, 113, 112, 115, 114, 117, 116, 119, 118,
      121, 120, 123, 122, 125, 124, 127, 126};
  init_data<unsigned int>(host_dev_data_u, DATA_NUM);

  q_ct1->memcpy(dev_data_u, host_dev_data_u, DATA_NUM * sizeof(unsigned int))
      .wait();
  q_ct1->parallel_for(sycl::nd_range<3>(GridSize * BlockSize, BlockSize),
                      [=](sycl::nd_item<3> item_ct1)
                          [[intel::reqd_sub_group_size(32)]] {
                            permute_sub_group_by_xor2(dev_data_u, item_ct1);
                          });

  dev_ct1.queues_wait_and_throw();
  q_ct1->memcpy(host_dev_data_u, dev_data_u, DATA_NUM * sizeof(unsigned int))
      .wait();
  verify_data<unsigned int>(host_dev_data_u, expect2, DATA_NUM);

  sycl::free(dev_data, *q_ct1);
  sycl::free(dev_data_u, *q_ct1);
}

int main() {
  test_permute_sub_group_by_xor();

  return 0;
}
