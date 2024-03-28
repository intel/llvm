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
 *  util_perm_byte_test.cpp
 *
 *  Description:
 *    byte_level_permute tests
 **************************************************************************/

// The original source was under the license below:
// ====------ UtilPermByteTest.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

// RUN: %clangxx -fsycl -fsycl-targets=%{sycl_triple} %s -o %t.out
// RUN: %{run} %t.out

#include <stdio.h>
#include <stdlib.h>
#include <sycl/detail/core.hpp>
#include <syclcompat.hpp>

void byte_perm_ref(unsigned int *d_data) {

  unsigned int lo;
  unsigned int hi;

  lo = 0x33221100;
  hi = 0x77665544;

  lo = 0x33221100;
  hi = 0x77665544;

  for (int i = 0; i < 17; i++)
    d_data[i] = syclcompat::byte_level_permute(lo, hi, 0x1111 * i);

  d_data[17] = syclcompat::byte_level_permute(lo, 0, 0x0123);
  d_data[18] = syclcompat::byte_level_permute(lo, hi, 0x7531);
  d_data[19] = syclcompat::byte_level_permute(lo, hi, 0x6420);
}

void test_byte_level_permute() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  const int N = 20;
  unsigned int refer[N] = {0x0,        0x11111111, 0x22222222, 0x33333333,
                           0x44444444, 0x55555555, 0x66666666, 0x77777777,
                           0x0,        0x11111111, 0x22222222, 0x33333333,
                           0x44444444, 0x55555555, 0x66666666, 0x77777777,
                           0x11111100, 0x112233,   0x77553311, 0x66442200};
  unsigned int data[N];

  byte_perm_ref(data);

  for (int i = 0; i < N; i++) {
    assert(refer[i] == data[i]);
  }
}

int main() {
  test_byte_level_permute();

  return 0;
}
