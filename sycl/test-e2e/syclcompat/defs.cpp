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
 *     Syclcompat macros tests
 **************************************************************************/

// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %{run} %t.out

#include <cassert>
#include <iostream>

#include <sycl/detail/core.hpp>

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

  auto sycl_error_throw = []() {
    throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                          "Expected invalid exception in test_check_error");
  };

  auto runtime_error_throw = []() {
    throw std::runtime_error("Expected invalid exception in test_check_error");
  };

  assert(syclcompat::error_code::SUCCESS == SYCLCOMPAT_CHECK_ERROR());
  assert(syclcompat::error_code::BACKEND_ERROR ==
         SYCLCOMPAT_CHECK_ERROR(sycl_error_throw()));
  assert(syclcompat::error_code::DEFAULT_ERROR ==
         SYCLCOMPAT_CHECK_ERROR(runtime_error_throw()));
}

void test_version() {
  // Check the composition of the version int
  assert(SYCLCOMPAT_MAKE_VERSION(1, 1, 1) == 1001001);
  assert(SYCLCOMPAT_MAKE_VERSION(9, 0, 0) == 9000000);

  // Check some inequalities
  assert(SYCLCOMPAT_MAKE_VERSION(0, 1, 1) > SYCLCOMPAT_MAKE_VERSION(0, 1, 0));
  assert(SYCLCOMPAT_MAKE_VERSION(1, 0, 0) > SYCLCOMPAT_MAKE_VERSION(0, 9, 0));
}

int main() {
  test_align();
  test_check_error();
  test_version();
  return 0;
}
