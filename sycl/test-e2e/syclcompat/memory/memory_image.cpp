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
 *  memory_async.cpp
 *
 *  Description:
 *    Asynchronous memory operations event dependency tests
 **************************************************************************/

// The original source was under the license below:
// ====------ memory_async.cpp------------------- -*- C++ -* ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

// RUN: %clangxx -std=c++20 -fsycl -fsycl-targets=%{sycl_triple} %s -o %t.out
// RUN: %{run} %t.out

// Tests for the sycl::events returned from syclcompat::*Async API calls

#include "sycl/exception.hpp"
#include <stdexcept>
#include <stdio.h>

#include <sycl/detail/core.hpp>

#include <syclcompat/memory.hpp>

// Check these APIs raise std::runtime_error.

void test_memcpy_parameter_async() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  try{
    syclcompat::experimental::memcpy_async(
        syclcompat::experimental::memcpy_parameter{});
    assert(false);
  } catch (std::runtime_error) {}

}

void test_memcpy_parameter() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  try{
    syclcompat::experimental::memcpy(
        syclcompat::experimental::memcpy_parameter{});
    assert(false);
  }catch (std::runtime_error) {}
}
int main() {
  test_memcpy_parameter();
  test_memcpy_parameter_async();
  return 0;
}
