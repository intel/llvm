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
 *  math_vectorized_isgreater_test.cpp
 *
 *  Description:
 *    vectorized_isgreater tests
 **************************************************************************/

// The original source was under the license below:
// ====---- UtilVectorizedIsgreaterTest.cpp----------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include <syclcompat.hpp>

void test_kernel_vect_is_greater_1(unsigned int vect_count,
                                   unsigned int *input_1, unsigned int *input_2,
                                   unsigned int *output,
                                   sycl::nd_item<3> item_ct1) {

  int index = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) +
              item_ct1.get_local_id(2);

  if (index < vect_count) {
    output[index] =
        syclcompat::vectorized_isgreater<sycl::ushort2, unsigned int>(
            input_1[index], input_2[index]);
  }
}

void test_vec_gt_1() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  syclcompat::device_ext &dev_ct1 = syclcompat::get_current_device();
  sycl::queue *q_ct1 = dev_ct1.default_queue();

  const unsigned int num_data = 7;
  unsigned int mem_size = sizeof(unsigned int) * num_data;

  unsigned int *h_out_data = (unsigned int *)malloc(mem_size);
  unsigned int *h_data = (unsigned int *)malloc(mem_size);

  for (unsigned int i = 0; i < num_data; i++)
    h_out_data[i] = 0;

  unsigned int *d_out_data;
  d_out_data = (unsigned int *)sycl::malloc_device(mem_size, *q_ct1);

  unsigned int *d_in_data_1;
  d_in_data_1 = (unsigned int *)sycl::malloc_device(mem_size, *q_ct1);

  unsigned int *d_in_data_2;
  d_in_data_2 = (unsigned int *)sycl::malloc_device(mem_size, *q_ct1);

  for (unsigned int i = 0; i < num_data; i++)
    h_data[i] = i;
  q_ct1->memcpy(d_in_data_1, h_data, mem_size).wait();

  for (unsigned int i = 0; i < num_data; i++)
    h_data[i] = num_data - 1 - i;
  q_ct1->memcpy(d_in_data_2, h_data, mem_size).wait();

  q_ct1->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 3) * sycl::range<3>(1, 1, 3),
                          sycl::range<3>(1, 1, 3)),
        [=](sycl::nd_item<3> item_ct1) {
          test_kernel_vect_is_greater_1(num_data, d_in_data_1, d_in_data_2,
                                        d_out_data, item_ct1);
        });
  });
  dev_ct1.queues_wait_and_throw();

  q_ct1->memcpy(h_out_data, d_out_data, mem_size).wait();

  unsigned int ref_data[num_data] = {0, 0, 0, 0, 1, 1, 1};
  for (unsigned int i = 0; i < num_data; i++) {
    if (h_out_data[i] != ref_data[i]) {
      printf("vec_max_test_1 failed!\n");
      exit(-1);
    }
  }

  free(h_out_data);
  free(h_data);
  sycl::free(d_out_data, *q_ct1);
  sycl::free(d_in_data_1, *q_ct1);
  sycl::free(d_in_data_2, *q_ct1);
}

void test_kernel_vect_is_greater_2(unsigned int vect_count,
                                   unsigned int *input_1, unsigned int *input_2,
                                   unsigned int *output,
                                   sycl::nd_item<3> item_ct1) {

  int index = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) +
              item_ct1.get_local_id(2);

  if (index < vect_count) {
    output[index] = syclcompat::vectorized_isgreater<sycl::half2, unsigned int>(
        input_1[index], input_2[index]);
  }
}

void test_vec_gt_2() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  syclcompat::device_ext &dev_ct1 = syclcompat::get_current_device();
  sycl::queue *q_ct1 = dev_ct1.default_queue();

  // This test uses vector of sycl::half, which requires support for fp16 on
  // the corresponding device.
  if (!dev_ct1.has(sycl::aspect::fp16)) {
    std::cout
        << "Test case skipped as the device does not support aspect::fp16."
        << std::endl;
    return;
  }

  const unsigned int num_data = 7;
  unsigned int mem_size = sizeof(unsigned int) * num_data;

  unsigned int *h_out_data = (unsigned int *)malloc(mem_size);
  unsigned int *h_data = (unsigned int *)malloc(mem_size);
  unsigned int a_array[num_data] = {6 + 65535, 1 + 65535, 2 + 65535, 3,
                                    4,         5,         6};
  unsigned int b_array[num_data] = {0 + 65535, 5 + 65535, 4 + 65535, 3,
                                    2,         1,         0};

  for (unsigned int i = 0; i < num_data; i++)
    h_out_data[i] = 0;

  unsigned int *d_out_data;
  d_out_data = (unsigned int *)sycl::malloc_device(mem_size, *q_ct1);

  unsigned int *d_in_data_1;
  d_in_data_1 = (unsigned int *)sycl::malloc_device(mem_size, *q_ct1);

  unsigned int *d_in_data_2;
  d_in_data_2 = (unsigned int *)sycl::malloc_device(mem_size, *q_ct1);

  for (unsigned int i = 0; i < num_data; i++)
    h_data[i] = a_array[i];

  q_ct1->memcpy(d_in_data_1, h_data, mem_size).wait();

  for (unsigned int i = 0; i < num_data; i++)
    h_data[i] = b_array[i];

  q_ct1->memcpy(d_in_data_2, h_data, mem_size).wait();

  q_ct1->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 3) * sycl::range<3>(1, 1, 3),
                          sycl::range<3>(1, 1, 3)),
        [=](sycl::nd_item<3> item_ct1) {
          test_kernel_vect_is_greater_2(num_data, d_in_data_1, d_in_data_2,
                                        d_out_data, item_ct1);
        });
  });
  dev_ct1.queues_wait_and_throw();

  q_ct1->memcpy(h_out_data, d_out_data, mem_size).wait();

  unsigned int ref_data[num_data] = {0xffff0000, 0x0,    0x0,   0x0,
                                     0xffff,     0xffff, 0xffff};
  for (unsigned int i = 0; i < num_data; i++) {
    if (h_out_data[i] != ref_data[i]) {
      printf("vec_max_test_2 failed!\n");
      exit(-1);
    }
  }

  free(h_out_data);
  free(h_data);
  sycl::free(d_out_data, *q_ct1);
  sycl::free(d_in_data_1, *q_ct1);
  sycl::free(d_in_data_2, *q_ct1);
}

int main() {
  test_vec_gt_1();
  test_vec_gt_2();

  return 0;
}
