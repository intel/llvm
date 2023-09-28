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
 *  SYCLcompat
 *
 *  Defs.cpp
 *
 *  Description:
 *     __syclcompat_align__ tests
 **************************************************************************/

// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %{run} %t.out

#include <cassert>
#include <iostream>
#include <syclcompat/defs.hpp>

void test_align() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  constexpr std::size_t expected_size = 16;
  struct __syclcompat_align__(expected_size) {
    int a;
    char c;
  }
  s;
  assert(sizeof(s) == expected_size);
}

void test_check_error() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  assert(syclcompat::error_code::SUCCESS == SYCLCOMPAT_CHECK_ERROR(0));
  assert(syclcompat::error_code::DEFAULT_ERROR ==
         SYCLCOMPAT_CHECK_ERROR(throw std::runtime_error(
             "Expected exception in test_check_error")));
}

int main() {
  test_align();
  test_check_error();

  return 0;
}
