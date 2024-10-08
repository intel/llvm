// ====------ shared_memory.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#define COMPAT_USM_LEVEL_NONE
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <string.h>

#define M 4
#define N 8

dpct::shared_memory<float, 1> array(N);
dpct::shared_memory<float, 1> result(M*N);

void my_kernel(float* array, float* result,
               sycl::nd_item<3> item_ct1,
               float *resultInGroup)
{


  array[item_ct1.get_local_id(2)] = item_ct1.get_local_id(2);
  resultInGroup[item_ct1.get_local_id(2)] = item_ct1.get_group(2);

  item_ct1.barrier();

  if (item_ct1.get_local_id(2) == 0) {
    memcpy(&result[item_ct1.get_group(2)*N], resultInGroup, sizeof(float)*N);
  }
}


int main () {
  {
    std::pair<dpct::buffer_t, size_t> array_buf_ct0 = dpct::get_buffer_and_offset(array.get_ptr());
    size_t array_offset_ct0 = array_buf_ct0.second;
    std::pair<dpct::buffer_t, size_t> result_buf_ct1 = dpct::get_buffer_and_offset(result.get_ptr());
    size_t result_offset_ct1 = result_buf_ct1.second;
    dpct::get_default_queue().submit(
      [&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> resultInGroup_acc_ct1(sycl::range<1>(8), cgh);
        auto array_acc_ct0 = array_buf_ct0.first.get_access<sycl::access::mode::read_write>(cgh);
        auto result_acc_ct1 = result_buf_ct1.first.get_access<sycl::access::mode::read_write>(cgh);

        cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, M) * sycl::range<3>(1, 1, N), sycl::range<3>(1, 1, N)), 
          [=](sycl::nd_item<3> item_ct1) {
            float *array_ct0 = (float *)(&array_acc_ct0[0] + array_offset_ct0);
            float *result_ct1 = (float *)(&result_acc_ct1[0] + result_offset_ct1);
            my_kernel(array_ct0, result_ct1, item_ct1, resultInGroup_acc_ct1.get_pointer());
          });
      });
  }

  dpct::get_current_device().queues_wait_and_throw();
  for(int j = 0; j < M; j++) {
    for (int i = 0; i < N; i++) {
      printf("%f ", result[j*N + i]);
    }
    printf("\n");
  }

  return 0;
}

