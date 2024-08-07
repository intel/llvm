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
 *  util_cast_value_test.cpp
 *
 *  Description:
 *    cast_value tests
 **************************************************************************/

// The original source was under the license below:
// ====------ UtilCastValueTest.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

// REQUIRES: aspect-fp64

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

#include <syclcompat/device.hpp>
#include <syclcompat/util.hpp>

double cast_value(const double &val) {
  int lo = syclcompat::cast_double_to_int(val, false);
  int hi = syclcompat::cast_double_to_int(val);
  return syclcompat::cast_ints_to_double(hi, lo);
}

void test_kernel_cast_value(double *g_odata) {
  double a = 1.12123515e-25f;
  g_odata[0] = cast_value(a);

  a = 0.000000000000000000000000112123515f;
  g_odata[1] = cast_value(a);

  a = 3.1415926f;
  g_odata[2] = cast_value(a);
}

void test_cast_value() {
  sycl::queue q = syclcompat::get_default_queue();

  unsigned int num_data = 3;
  unsigned int mem_size = sizeof(double) * num_data;

  double *h_out_data = (double *)malloc(mem_size);

  for (unsigned int i = 0; i < num_data; i++)
    h_out_data[i] = 0;

  double *d_out_data;
  d_out_data = (double *)sycl::malloc_device(mem_size, q);
  q.memcpy(d_out_data, h_out_data, mem_size).wait();

  q.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
      [=](sycl::nd_item<3> item_ct1) { test_kernel_cast_value(d_out_data); });

  q.memcpy(h_out_data, d_out_data, mem_size).wait();

  assert(h_out_data[0] == 1.12123515e-25f);
  assert(h_out_data[1] == 0.000000000000000000000000112123515f);
  assert(h_out_data[2] == 3.1415926f);

  free(h_out_data);
  sycl::free(d_out_data, q);
}

int main() {
  test_cast_value();

  return 0;
}
