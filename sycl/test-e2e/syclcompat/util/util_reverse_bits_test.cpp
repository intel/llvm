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
 *  util_reverse_bits_test.cpp
 *
 *  Description:
 *    reverse_bits tests
 **************************************************************************/

// The original source was under the license below:
// ====------ UtilReverseBitsTest.cpp---------- -*- C++ -* ----===////
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
#include <syclcompat.hpp>

void test_reverse_bits() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  unsigned int a = 1;
  unsigned int b = syclcompat::reverse_bits(a);
  assert(b == 0x80000000);

  a = 0x12345678;
  b = syclcompat::reverse_bits(a);
  assert(b == 0x1e6a2c48);
}

int main() {
  test_reverse_bits();

  return 0;
}
