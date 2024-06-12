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

void test_memcpy_parameter_async(
    syclcompat::experimental::memcpy_parameter param, bool xpass) {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  try {
    syclcompat::experimental::memcpy_async(param);
    assert(xpass);
  } catch (std::runtime_error &) {
    assert(!xpass);
  }
}

void test_memcpy_parameter(syclcompat::experimental::memcpy_parameter param,
                           bool xpass) {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  try {
    syclcompat::experimental::memcpy(param);
    assert(xpass);
  } catch (std::runtime_error &) {
    assert(!xpass);
  }
}

// Check (most) memcpy_parameter APIs raise std::runtime_error.
void test_memcpy_parameter_xfails() {

  {
    // Empty `memcpy_params` passes in no bindless_image
    // or image pointers. This is the code path that ought to pass.
    syclcompat::experimental::memcpy_parameter params;
    test_memcpy_parameter(params, true);
    test_memcpy_parameter_async(params, true);
  }

  {
    // Mimick passing a bindless image for source
    syclcompat::experimental::memcpy_parameter params;
    params.from.image_bindless =
        reinterpret_cast<syclcompat::experimental::image_mem_wrapper *>(1);
    test_memcpy_parameter(params, false);
    test_memcpy_parameter_async(params, false);
  }

  {
    // Mimick passing a bindless image for dest
    syclcompat::experimental::memcpy_parameter params;
    params.to.image_bindless =
        reinterpret_cast<syclcompat::experimental::image_mem_wrapper *>(1);
    test_memcpy_parameter(params, false);
    test_memcpy_parameter_async(params, false);
  }

  {
    // Mimick passing a bindless image for source & dest
    syclcompat::experimental::memcpy_parameter params;
    params.from.image_bindless =
        reinterpret_cast<syclcompat::experimental::image_mem_wrapper *>(1);
    params.to.image_bindless =
        reinterpret_cast<syclcompat::experimental::image_mem_wrapper *>(1);
    test_memcpy_parameter(params, false);
    test_memcpy_parameter_async(params, false);
  }

  {
    // Mimick passing an image for source
    syclcompat::experimental::memcpy_parameter params;
    params.from.image =
        reinterpret_cast<syclcompat::experimental::image_matrix *>(1);
    test_memcpy_parameter(params, false);
    test_memcpy_parameter_async(params, false);
  }

  {
    // Mimick passing an image for dest
    syclcompat::experimental::memcpy_parameter params;
    params.to.image =
        reinterpret_cast<syclcompat::experimental::image_matrix *>(1);
    test_memcpy_parameter(params, false);
    test_memcpy_parameter_async(params, false);
  }

  {
    // Mimick passing an image for source & dest
    syclcompat::experimental::memcpy_parameter params;
    params.from.image =
        reinterpret_cast<syclcompat::experimental::image_matrix *>(1);
    params.to.image =
        reinterpret_cast<syclcompat::experimental::image_matrix *>(1);
    test_memcpy_parameter(params, false);
    test_memcpy_parameter_async(params, false);
  }
}

int main() {
  test_memcpy_parameter_xfails();
  return 0;
}
