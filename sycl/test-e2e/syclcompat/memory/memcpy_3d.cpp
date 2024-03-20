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
 *  memcpy_3d.cpp
 *
 *  Description:
 *    3D memory copy tests
 **************************************************************************/

// The original source was under the license below:
// ====------ memcpy_3d.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

// RUN: %clangxx -fsycl -fsycl-targets=%{sycl_triple} %s -o %t.out
// RUN: %{run} %t.out

#include <malloc.h>
#include <stdio.h>
#include <sycl/detail/core.hpp>

#include <syclcompat/memory.hpp>

#include "memory_common.hpp"

void test_memcpy3D_memset() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  size_t width = 6;
  size_t height = 8;
  size_t depth = 10;
  float *h_data;
  float *h_ref;

  syclcompat::pitched_data d_data;
  sycl::range<3> extent = sycl::range<3>(sizeof(float) * 1, 1, 1);
  syclcompat::pitched_data cpyParm_from_data_ct1, cpyParm_to_data_ct1;
  sycl::id<3> cpyParm_from_pos_ct1(0, 0, 0), cpyParm_to_pos_ct1(0, 0, 0);
  sycl::range<3> cpyParm_size_ct1(0, 0, 0);

  h_data =
      (float *)syclcompat::malloc_host(sizeof(float) * width * height * depth);
  for (int i = 0; i < width * height * depth; i++)
    h_data[i] = (float)i;

  h_ref =
      (float *)syclcompat::malloc_host(sizeof(float) * width * height * depth);
  for (int i = 0; i < width * height * depth; i++)
    h_ref[i] = (float)i;

  // alloc memory.
  extent = sycl::range<3>(sizeof(float) * width, height, depth);
  d_data = (syclcompat::pitched_data)syclcompat::malloc(extent);

  // copy to Device.
  cpyParm_from_data_ct1 = syclcompat::pitched_data(
      (void *)h_data, sizeof(float) * width, width, height);
  cpyParm_to_data_ct1 = d_data;
  cpyParm_size_ct1 = extent;
  syclcompat::memcpy(cpyParm_to_data_ct1, cpyParm_to_pos_ct1,
                     cpyParm_from_data_ct1, cpyParm_from_pos_ct1,
                     cpyParm_size_ct1);

  // copy back to host.
  cpyParm_from_data_ct1 = d_data;
  cpyParm_to_data_ct1 = syclcompat::pitched_data(
      (void *)h_data, sizeof(float) * width, width, height);
  cpyParm_size_ct1 = extent;
  syclcompat::memcpy(cpyParm_to_data_ct1, cpyParm_to_pos_ct1,
                     cpyParm_from_data_ct1, cpyParm_from_pos_ct1,
                     cpyParm_size_ct1);

  check(h_data, h_ref, width * height * depth);
  // memset device data.
  syclcompat::memset(d_data, 0x1, extent);

  // copy back to host
  cpyParm_from_data_ct1 = d_data;
  cpyParm_to_data_ct1 = syclcompat::pitched_data(
      (void *)h_data, sizeof(float) * width, width, height);
  cpyParm_size_ct1 = extent;
  syclcompat::memcpy(cpyParm_to_data_ct1, cpyParm_to_pos_ct1,
                     cpyParm_from_data_ct1, cpyParm_from_pos_ct1,
                     cpyParm_size_ct1);
  // memset reference data.
  memset(h_ref, 0x1, width * height * depth * sizeof(float));
  check(h_data, h_ref, width * height * depth);

  syclcompat::free(h_data);
  syclcompat::free(h_ref);
  sycl::free(d_data.get_data_ptr(), syclcompat::get_default_context());
}

void test_memcpy3D_memset_q() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  sycl::queue q{{sycl::property::queue::in_order()}};
  size_t width = 6;
  size_t height = 8;
  size_t depth = 10;
  float *h_data;
  float *h_ref;

  syclcompat::pitched_data d_data;
  sycl::range<3> extent = sycl::range<3>(sizeof(float) * 1, 1, 1);
  syclcompat::pitched_data cpyParm_from_data_ct1, cpyParm_to_data_ct1;
  sycl::id<3> cpyParm_from_pos_ct1(0, 0, 0), cpyParm_to_pos_ct1(0, 0, 0);
  sycl::range<3> cpyParm_size_ct1(0, 0, 0);

  h_data = (float *)syclcompat::malloc_host(
      sizeof(float) * width * height * depth, q);
  for (int i = 0; i < width * height * depth; i++)
    h_data[i] = (float)i;

  h_ref = (float *)syclcompat::malloc_host(
      sizeof(float) * width * height * depth, q);
  for (int i = 0; i < width * height * depth; i++)
    h_ref[i] = (float)i;

  // alloc memory.
  extent = sycl::range<3>(sizeof(float) * width, height, depth);
  d_data = (syclcompat::pitched_data)syclcompat::malloc(extent, q);

  // copy to Device.
  cpyParm_from_data_ct1 = syclcompat::pitched_data(
      (void *)h_data, sizeof(float) * width, width, height);
  cpyParm_to_data_ct1 = d_data;
  cpyParm_size_ct1 = extent;
  syclcompat::memcpy(cpyParm_to_data_ct1, cpyParm_to_pos_ct1,
                     cpyParm_from_data_ct1, cpyParm_from_pos_ct1,
                     cpyParm_size_ct1, q);

  // copy back to host.
  cpyParm_from_data_ct1 = d_data;
  cpyParm_to_data_ct1 = syclcompat::pitched_data(
      (void *)h_data, sizeof(float) * width, width, height);
  cpyParm_size_ct1 = extent;
  syclcompat::memcpy(cpyParm_to_data_ct1, cpyParm_to_pos_ct1,
                     cpyParm_from_data_ct1, cpyParm_from_pos_ct1,
                     cpyParm_size_ct1, q);

  check(h_data, h_ref, width * height * depth);
  // memset device data.
  syclcompat::memset(d_data, 0x1, extent, q);

  // copy back to host
  cpyParm_from_data_ct1 = d_data;
  cpyParm_to_data_ct1 = syclcompat::pitched_data(
      (void *)h_data, sizeof(float) * width, width, height);
  cpyParm_size_ct1 = extent;
  syclcompat::memcpy(cpyParm_to_data_ct1, cpyParm_to_pos_ct1,
                     cpyParm_from_data_ct1, cpyParm_from_pos_ct1,
                     cpyParm_size_ct1, q);
  // memset reference data.
  memset(h_ref, 0x1, width * height * depth * sizeof(float));
  check(h_data, h_ref, width * height * depth);

  syclcompat::free(h_data, q);
  syclcompat::free(h_ref, q);
  syclcompat::free(d_data.get_data_ptr(), q);
}

void test_memcpy3D_offset() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  size_t width = 4;
  size_t height = 4;
  size_t depth = 5;
  float *h_data;

  syclcompat::pitched_data d_data;
  sycl::range<3> extent = sycl::range<3>(sizeof(float) * 1, 1, 1);
  syclcompat::pitched_data cpyParm_from_data_ct1, cpyParm_to_data_ct1;
  sycl::id<3> cpyParm_from_pos_ct1(0, 0, 0), cpyParm_to_pos_ct1(0, 0, 0);
  sycl::range<3> cpyParm_size_ct1(0, 0, 0);

  h_data =
      (float *)syclcompat::malloc_host(sizeof(float) * width * height * depth);
  /*
    0.000000        1.000000        2.000000        3.000000
    4.000000        5.000000        6.000000        7.000000
    8.000000        9.000000        10.000000       11.000000
    12.000000       13.000000       14.000000       15.000000

    16.000000       17.000000       18.000000       19.000000
    20.000000       21.000000       22.000000       23.000000
    24.000000       25.000000       26.000000       27.000000
    28.000000       29.000000       30.000000       31.000000

    32.000000       33.000000       34.000000       35.000000
    36.000000       37.000000       38.000000       39.000000
    40.000000       41.000000       42.000000       43.000000
    44.000000       45.000000       46.000000       47.000000

    48.000000       49.000000       50.000000       51.000000
    52.000000       53.000000       54.000000       55.000000
    56.000000       57.000000       58.000000       59.000000
    60.000000       61.000000       62.000000       63.000000

    64.000000       65.000000       66.000000       67.000000
    68.000000       69.000000       70.000000       71.000000
    72.000000       73.000000       74.000000       75.000000
    76.000000       77.000000       78.000000       79.000000
  */
  for (int i = 0; i < width * height * depth; i++)
    h_data[i] = (float)i;

  /*
    5.000000        6.000000
    9.000000        10.000000

    21.000000       22.000000
    25.000000       26.000000

    37.000000       38.000000
    41.000000       42.000000
  */
  float Ref[12] = {5, 6, 9, 10, 21, 22, 25, 26, 37, 38, 41, 42};

  size_t out_width = 2;
  size_t out_height = 2;
  size_t out_depth = 3;

  // alloc memory.
  extent = sycl::range<3>(sizeof(float) * width, height, depth);
  d_data = (syclcompat::pitched_data)syclcompat::malloc(extent);

  // copy to Device.
  cpyParm_from_data_ct1 = syclcompat::pitched_data(
      (void *)h_data, sizeof(float) * width, width, height);
  cpyParm_to_data_ct1 = d_data;
  cpyParm_size_ct1 = extent;
  syclcompat::memcpy(cpyParm_to_data_ct1, cpyParm_to_pos_ct1,
                     cpyParm_from_data_ct1, cpyParm_from_pos_ct1,
                     cpyParm_size_ct1);

  cpyParm_from_pos_ct1 = {1 * sizeof(float), 1, 0}; // set offset on x/y/z.
  cpyParm_size_ct1 = {out_width * sizeof(float), out_height, out_depth};

  for (int i = 0; i < out_width * out_height * out_depth; i++)
    h_data[i] = -1;
  // copy back to host.
  cpyParm_from_data_ct1 = d_data;
  cpyParm_to_data_ct1 = syclcompat::pitched_data(
      (void *)h_data, sizeof(float) * out_width, out_width, out_height);

  syclcompat::memcpy(cpyParm_to_data_ct1, cpyParm_to_pos_ct1,
                     cpyParm_from_data_ct1, cpyParm_from_pos_ct1,
                     cpyParm_size_ct1);

  // Copy back to host data.
  check(h_data, Ref, out_width * out_height * out_depth);
  syclcompat::free(h_data);
  sycl::free(d_data.get_data_ptr(), syclcompat::get_default_context());
}

void test_memcpy3D_offset_q() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  sycl::queue q{{sycl::property::queue::in_order()}};
  size_t width = 4;
  size_t height = 4;
  size_t depth = 5;
  float *h_data;

  syclcompat::pitched_data d_data;
  sycl::range<3> extent = sycl::range<3>(sizeof(float) * 1, 1, 1);
  syclcompat::pitched_data cpyParm_from_data_ct1, cpyParm_to_data_ct1;
  sycl::id<3> cpyParm_from_pos_ct1(0, 0, 0), cpyParm_to_pos_ct1(0, 0, 0);
  sycl::range<3> cpyParm_size_ct1(0, 0, 0);

  h_data = (float *)syclcompat::malloc_host(
      sizeof(float) * width * height * depth, q);
  /*
    0.000000        1.000000        2.000000        3.000000
    4.000000        5.000000        6.000000        7.000000
    8.000000        9.000000        10.000000       11.000000
    12.000000       13.000000       14.000000       15.000000

    16.000000       17.000000       18.000000       19.000000
    20.000000       21.000000       22.000000       23.000000
    24.000000       25.000000       26.000000       27.000000
    28.000000       29.000000       30.000000       31.000000

    32.000000       33.000000       34.000000       35.000000
    36.000000       37.000000       38.000000       39.000000
    40.000000       41.000000       42.000000       43.000000
    44.000000       45.000000       46.000000       47.000000

    48.000000       49.000000       50.000000       51.000000
    52.000000       53.000000       54.000000       55.000000
    56.000000       57.000000       58.000000       59.000000
    60.000000       61.000000       62.000000       63.000000

    64.000000       65.000000       66.000000       67.000000
    68.000000       69.000000       70.000000       71.000000
    72.000000       73.000000       74.000000       75.000000
    76.000000       77.000000       78.000000       79.000000
  */
  for (int i = 0; i < width * height * depth; i++)
    h_data[i] = (float)i;

  /*
    5.000000        6.000000
    9.000000        10.000000

    21.000000       22.000000
    25.000000       26.000000

    37.000000       38.000000
    41.000000       42.000000
  */
  float Ref[12] = {5, 6, 9, 10, 21, 22, 25, 26, 37, 38, 41, 42};

  size_t out_width = 2;
  size_t out_height = 2;
  size_t out_depth = 3;

  // alloc memory.
  extent = sycl::range<3>(sizeof(float) * width, height, depth);
  d_data = (syclcompat::pitched_data)syclcompat::malloc(extent, q);

  // copy to Device.
  cpyParm_from_data_ct1 = syclcompat::pitched_data(
      (void *)h_data, sizeof(float) * width, width, height);
  cpyParm_to_data_ct1 = d_data;
  cpyParm_size_ct1 = extent;
  syclcompat::memcpy(cpyParm_to_data_ct1, cpyParm_to_pos_ct1,
                     cpyParm_from_data_ct1, cpyParm_from_pos_ct1,
                     cpyParm_size_ct1, q);

  cpyParm_from_pos_ct1 = {1 * sizeof(float), 1, 0}; // set offset on x/y/z.
  cpyParm_size_ct1 = {out_width * sizeof(float), out_height, out_depth};

  for (int i = 0; i < out_width * out_height * out_depth; i++)
    h_data[i] = -1;
  // copy back to host.
  cpyParm_from_data_ct1 = d_data;
  cpyParm_to_data_ct1 = syclcompat::pitched_data(
      (void *)h_data, sizeof(float) * out_width, out_width, out_height);

  syclcompat::memcpy(cpyParm_to_data_ct1, cpyParm_to_pos_ct1,
                     cpyParm_from_data_ct1, cpyParm_from_pos_ct1,
                     cpyParm_size_ct1, q);
  // Copy back to host data.
  check(h_data, Ref, out_width * out_height * out_depth);
  syclcompat::free(h_data, q);
  syclcompat::free(d_data.get_data_ptr(), q);
}

void test_memcpy3D_offsetZ() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  size_t width = 4;
  size_t height = 4;
  size_t depth = 5;
  float *h_data;

  syclcompat::pitched_data d_data;
  sycl::range<3> extent = sycl::range<3>(sizeof(float) * 1, 1, 1);
  syclcompat::pitched_data cpyParm_from_data_ct1, cpyParm_to_data_ct1;
  sycl::id<3> cpyParm_from_pos_ct1(0, 0, 0), cpyParm_to_pos_ct1(0, 0, 0);
  sycl::range<3> cpyParm_size_ct1(0, 0, 0);

  h_data =
      (float *)syclcompat::malloc_host(sizeof(float) * width * height * depth);
  /*
    0.000000        1.000000        2.000000        3.000000
    4.000000        5.000000        6.000000        7.000000
    8.000000        9.000000        10.000000       11.000000
    12.000000       13.000000       14.000000       15.000000

    16.000000       17.000000       18.000000       19.000000
    20.000000       21.000000       22.000000       23.000000
    24.000000       25.000000       26.000000       27.000000
    28.000000       29.000000       30.000000       31.000000

    32.000000       33.000000       34.000000       35.000000
    36.000000       37.000000       38.000000       39.000000
    40.000000       41.000000       42.000000       43.000000
    44.000000       45.000000       46.000000       47.000000

    48.000000       49.000000       50.000000       51.000000
    52.000000       53.000000       54.000000       55.000000
    56.000000       57.000000       58.000000       59.000000
    60.000000       61.000000       62.000000       63.000000

    64.000000       65.000000       66.000000       67.000000
    68.000000       69.000000       70.000000       71.000000
    72.000000       73.000000       74.000000       75.000000
    76.000000       77.000000       78.000000       79.000000
  */
  for (int i = 0; i < width * height * depth; i++)
    h_data[i] = (float)i;

  /*
    21.000000       22.000000
    25.000000       26.000000

    37.000000       38.000000
    41.000000       42.000000

    53.000000       54.000000
    57.000000       58.000000
  */
  float Ref[12] = {21, 22, 25, 26, 37, 38, 41, 42, 53, 54, 57, 58};

  size_t out_width = 2;
  size_t out_height = 2;
  size_t out_depth = 3;

  // alloc memory.
  extent = sycl::range<3>(sizeof(float) * width, height, depth);
  d_data = (syclcompat::pitched_data)syclcompat::malloc(extent);

  // copy to Device.
  cpyParm_from_data_ct1 = syclcompat::pitched_data(
      (void *)h_data, sizeof(float) * width, width, height);
  cpyParm_to_data_ct1 = d_data;
  cpyParm_size_ct1 = extent;
  syclcompat::memcpy(cpyParm_to_data_ct1, cpyParm_to_pos_ct1,
                     cpyParm_from_data_ct1, cpyParm_from_pos_ct1,
                     cpyParm_size_ct1);

  cpyParm_from_pos_ct1 = {1 * sizeof(float), 1, 1}; // set offset on x/y/z.
  cpyParm_size_ct1 = {out_width * sizeof(float), out_height, out_depth};

  for (int i = 0; i < out_width * out_height * out_depth; i++)
    h_data[i] = -1;
  // copy back to host.
  cpyParm_from_data_ct1 = d_data;
  cpyParm_to_data_ct1 = syclcompat::pitched_data(
      (void *)h_data, sizeof(float) * out_width, out_width, out_height);

  syclcompat::memcpy(cpyParm_to_data_ct1, cpyParm_to_pos_ct1,
                     cpyParm_from_data_ct1, cpyParm_from_pos_ct1,
                     cpyParm_size_ct1);

  // Copy back to host data.
  check(h_data, Ref, out_width * out_height * out_depth);
  syclcompat::free(h_data);
  sycl::free(d_data.get_data_ptr(), syclcompat::get_default_context());
}

void test_memcpy3D_offsetZ_q() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  sycl::queue q{{sycl::property::queue::in_order()}};
  size_t width = 4;
  size_t height = 4;
  size_t depth = 5;
  float *h_data;

  syclcompat::pitched_data d_data;
  sycl::range<3> extent = sycl::range<3>(sizeof(float) * 1, 1, 1);
  syclcompat::pitched_data cpyParm_from_data_ct1, cpyParm_to_data_ct1;
  sycl::id<3> cpyParm_from_pos_ct1(0, 0, 0), cpyParm_to_pos_ct1(0, 0, 0);
  sycl::range<3> cpyParm_size_ct1(0, 0, 0);

  h_data = (float *)syclcompat::malloc_host(
      sizeof(float) * width * height * depth, q);
  /*
    0.000000        1.000000        2.000000        3.000000
    4.000000        5.000000        6.000000        7.000000
    8.000000        9.000000        10.000000       11.000000
    12.000000       13.000000       14.000000       15.000000

    16.000000       17.000000       18.000000       19.000000
    20.000000       21.000000       22.000000       23.000000
    24.000000       25.000000       26.000000       27.000000
    28.000000       29.000000       30.000000       31.000000

    32.000000       33.000000       34.000000       35.000000
    36.000000       37.000000       38.000000       39.000000
    40.000000       41.000000       42.000000       43.000000
    44.000000       45.000000       46.000000       47.000000

    48.000000       49.000000       50.000000       51.000000
    52.000000       53.000000       54.000000       55.000000
    56.000000       57.000000       58.000000       59.000000
    60.000000       61.000000       62.000000       63.000000

    64.000000       65.000000       66.000000       67.000000
    68.000000       69.000000       70.000000       71.000000
    72.000000       73.000000       74.000000       75.000000
    76.000000       77.000000       78.000000       79.000000
  */
  for (int i = 0; i < width * height * depth; i++)
    h_data[i] = (float)i;

  /*
    21.000000       22.000000
    25.000000       26.000000

    37.000000       38.000000
    41.000000       42.000000

    53.000000       54.000000
    57.000000       58.000000
  */
  float Ref[12] = {21, 22, 25, 26, 37, 38, 41, 42, 53, 54, 57, 58};

  size_t out_width = 2;
  size_t out_height = 2;
  size_t out_depth = 3;

  // alloc memory.
  extent = sycl::range<3>(sizeof(float) * width, height, depth);
  d_data = (syclcompat::pitched_data)syclcompat::malloc(extent, q);

  // copy to Device.
  cpyParm_from_data_ct1 = syclcompat::pitched_data(
      (void *)h_data, sizeof(float) * width, width, height);
  cpyParm_to_data_ct1 = d_data;
  cpyParm_size_ct1 = extent;
  syclcompat::memcpy(cpyParm_to_data_ct1, cpyParm_to_pos_ct1,
                     cpyParm_from_data_ct1, cpyParm_from_pos_ct1,
                     cpyParm_size_ct1, q);

  cpyParm_from_pos_ct1 = {1 * sizeof(float), 1, 1}; // set offset on x/y/z.
  cpyParm_size_ct1 = {out_width * sizeof(float), out_height, out_depth};

  for (int i = 0; i < out_width * out_height * out_depth; i++)
    h_data[i] = -1;
  // copy back to host.
  cpyParm_from_data_ct1 = d_data;
  cpyParm_to_data_ct1 = syclcompat::pitched_data(
      (void *)h_data, sizeof(float) * out_width, out_width, out_height);

  syclcompat::memcpy(cpyParm_to_data_ct1, cpyParm_to_pos_ct1,
                     cpyParm_from_data_ct1, cpyParm_from_pos_ct1,
                     cpyParm_size_ct1, q);

  // Copy back to host data.
  check(h_data, Ref, out_width * out_height * out_depth);
  syclcompat::free(h_data, q);
  syclcompat::free(d_data.get_data_ptr(), q);
}

// short path1
// test copy 3D data special case
// for continuous plane, we can copy it as linear data
void test_memcpy3D_plane() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  size_t width = 4;
  size_t height = 4;
  size_t depth = 5;
  float *h_data;

  syclcompat::pitched_data d_data;
  sycl::range<3> extent = sycl::range<3>(sizeof(float) * 1, 1, 1);
  syclcompat::pitched_data cpyParm_from_data_ct1, cpyParm_to_data_ct1;
  sycl::id<3> cpyParm_from_pos_ct1(0, 0, 0), cpyParm_to_pos_ct1(0, 0, 0);
  sycl::range<3> cpyParm_size_ct1(0, 0, 0);

  h_data =
      (float *)syclcompat::malloc_host(sizeof(float) * width * height * depth);
  /*
    0.000000        1.000000        2.000000        3.000000
    4.000000        5.000000        6.000000        7.000000
    8.000000        9.000000        10.000000       11.000000
    12.000000       13.000000       14.000000       15.000000

    16.000000       17.000000       18.000000       19.000000
    20.000000       21.000000       22.000000       23.000000
    24.000000       25.000000       26.000000       27.000000
    28.000000       29.000000       30.000000       31.000000

    32.000000       33.000000       34.000000       35.000000
    36.000000       37.000000       38.000000       39.000000
    40.000000       41.000000       42.000000       43.000000
    44.000000       45.000000       46.000000       47.000000

    48.000000       49.000000       50.000000       51.000000
    52.000000       53.000000       54.000000       55.000000
    56.000000       57.000000       58.000000       59.000000
    60.000000       61.000000       62.000000       63.000000

    64.000000       65.000000       66.000000       67.000000
    68.000000       69.000000       70.000000       71.000000
    72.000000       73.000000       74.000000       75.000000
    76.000000       77.000000       78.000000       79.000000
  */
  for (int i = 0; i < width * height * depth; i++)
    h_data[i] = (float)i;

  /*
    0.000000        1.000000        2.000000        3.000000
    4.000000        5.000000        6.000000        7.000000
    8.000000        9.000000        10.000000       11.000000
    12.000000       13.000000       14.000000       15.000000

    16.000000       17.000000       18.000000       19.000000
    20.000000       21.000000       22.000000       23.000000
    24.000000       25.000000       26.000000       27.000000
    28.000000       29.000000       30.000000       31.000000
  */
  float Ref[32] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                   11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                   22, 23, 24, 25, 26, 27, 28, 29, 30, 31};

  size_t out_width = 4;
  size_t out_height = 4;
  size_t out_depth = 2;

  // alloc memory.
  extent = sycl::range<3>(sizeof(float) * width, height, depth);
  d_data = (syclcompat::pitched_data)syclcompat::malloc(extent);

  // copy to Device.
  cpyParm_from_data_ct1 = syclcompat::pitched_data(
      (void *)h_data, sizeof(float) * width, width, height);
  cpyParm_to_data_ct1 = d_data;
  cpyParm_size_ct1 = extent;
  syclcompat::memcpy(cpyParm_to_data_ct1, cpyParm_to_pos_ct1,
                     cpyParm_from_data_ct1, cpyParm_from_pos_ct1,
                     cpyParm_size_ct1);

  cpyParm_from_pos_ct1 = {0, 0, 0}; // set offset on x/y/z.
  cpyParm_size_ct1 = {out_width * sizeof(float), out_height, out_depth};

  for (int i = 0; i < out_width * out_height * out_depth; i++)
    h_data[i] = -1;
  // copy back to host.
  cpyParm_from_data_ct1 = d_data;
  cpyParm_to_data_ct1 = syclcompat::pitched_data(
      (void *)h_data, sizeof(float) * out_width, out_width, out_height);
  syclcompat::memcpy(cpyParm_to_data_ct1, cpyParm_to_pos_ct1,
                     cpyParm_from_data_ct1, cpyParm_from_pos_ct1,
                     cpyParm_size_ct1);

  // Copy back to host data.
  check(h_data, Ref, out_width * out_height * out_depth);
  syclcompat::free(h_data);
  sycl::free(d_data.get_data_ptr(), syclcompat::get_default_context());
}

// short path2
// test copy 3D data special case
// for continuous row, we can copy it as linear data
void test_memcpy3D_row() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  size_t width = 4;
  size_t height = 4;
  size_t depth = 5;
  float *h_data;

  syclcompat::pitched_data d_data;
  sycl::range<3> extent = sycl::range<3>(sizeof(float) * 1, 1, 1);
  syclcompat::pitched_data cpyParm_from_data_ct1, cpyParm_to_data_ct1;
  sycl::id<3> cpyParm_from_pos_ct1(0, 0, 0), cpyParm_to_pos_ct1(0, 0, 0);
  sycl::range<3> cpyParm_size_ct1(0, 0, 0);

  h_data =
      (float *)syclcompat::malloc_host(sizeof(float) * width * height * depth);
  /*
    0.000000        1.000000        2.000000        3.000000
    4.000000        5.000000        6.000000        7.000000
    8.000000        9.000000        10.000000       11.000000
    12.000000       13.000000       14.000000       15.000000

    16.000000       17.000000       18.000000       19.000000
    20.000000       21.000000       22.000000       23.000000
    24.000000       25.000000       26.000000       27.000000
    28.000000       29.000000       30.000000       31.000000

    32.000000       33.000000       34.000000       35.000000
    36.000000       37.000000       38.000000       39.000000
    40.000000       41.000000       42.000000       43.000000
    44.000000       45.000000       46.000000       47.000000

    48.000000       49.000000       50.000000       51.000000
    52.000000       53.000000       54.000000       55.000000
    56.000000       57.000000       58.000000       59.000000
    60.000000       61.000000       62.000000       63.000000

    64.000000       65.000000       66.000000       67.000000
    68.000000       69.000000       70.000000       71.000000
    72.000000       73.000000       74.000000       75.000000
    76.000000       77.000000       78.000000       79.000000
  */
  for (int i = 0; i < width * height * depth; i++)
    h_data[i] = (float)i;

  /*
    0.000000        1.000000        2.000000        3.000000
    4.000000        5.000000        6.000000        7.000000

    16.000000       17.000000       18.000000       19.000000
    20.000000       21.000000       22.000000       23.000000
  */
  float Ref[16] = {0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23};

  size_t out_width = 4;
  size_t out_height = 2;
  size_t out_depth = 2;

  // alloc memory.
  extent = sycl::range<3>(sizeof(float) * width, height, depth);
  d_data = (syclcompat::pitched_data)syclcompat::malloc(extent);

  // copy to Device.
  cpyParm_from_data_ct1 = syclcompat::pitched_data(
      (void *)h_data, sizeof(float) * width, width, height);
  cpyParm_to_data_ct1 = d_data;
  cpyParm_size_ct1 = extent;
  syclcompat::memcpy(cpyParm_to_data_ct1, cpyParm_to_pos_ct1,
                     cpyParm_from_data_ct1, cpyParm_from_pos_ct1,
                     cpyParm_size_ct1);

  cpyParm_from_pos_ct1 = {0, 0, 0}; // set offset on x/y/z.
  cpyParm_size_ct1 = {out_width * sizeof(float), out_height, out_depth};

  for (int i = 0; i < out_width * out_height * out_depth; i++)
    h_data[i] = -1;
  // copy back to host.
  cpyParm_from_data_ct1 = d_data;
  cpyParm_to_data_ct1 = syclcompat::pitched_data(
      (void *)h_data, sizeof(float) * out_width, out_width, out_height);
  syclcompat::memcpy(cpyParm_to_data_ct1, cpyParm_to_pos_ct1,
                     cpyParm_from_data_ct1, cpyParm_from_pos_ct1,
                     cpyParm_size_ct1);

  // Copy back to host data.
  check(h_data, Ref, out_width * out_height * out_depth);
  syclcompat::free(h_data);
  sycl::free(d_data.get_data_ptr(), syclcompat::get_default_context());
}

int main() {
  test_memcpy3D_memset();
  test_memcpy3D_memset_q();
  test_memcpy3D_offset();
  test_memcpy3D_offset_q();
  test_memcpy3D_offsetZ();
  test_memcpy3D_offsetZ_q();
  test_memcpy3D_plane();
  test_memcpy3D_row();

  return 0;
}
