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
 *  memory_management_shared.cpp
 *
 *  Description:
 *    memory operations tests with shared memory
 **************************************************************************/

// The original source was under the license below:
// ====------ memory_management_test2.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

// REQUIRES: aspect-usm_shared_allocations
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

#include <syclcompat/memory.hpp>

#include "../common.hpp"
#include "memory_common.hpp"

constexpr size_t DataW = 100;
constexpr size_t DataH = 100;

void test_shared_memory() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  syclcompat::shared_memory<float, 1> s_A(DataW);
  syclcompat::shared_memory<float, 1> s_B(DataW);
  syclcompat::shared_memory<float, 1> s_C(DataW);

  s_A.init();
  s_B.init();
  s_C.init();

  for (int i = 0; i < DataW; i++) {
    s_A[i] = 1.0f;
    s_B[i] = 2.0f;
  }

  {
    syclcompat::get_default_queue().submit([&](sycl::handler &cgh) {
      float *d_A = s_A.get_ptr();
      float *d_B = s_B.get_ptr();
      float *d_C = s_C.get_ptr();
      cgh.parallel_for(sycl::range<1>(DataW), [=](sycl::id<1> id) {
        int i = id[0];
        float *A = d_A;
        float *B = d_B;
        float *C = d_C;
        C[i] = A[i] + B[i];
      });
    });
    syclcompat::get_default_queue().wait_and_throw();
  }

  // verify hostD
  for (int i = 0; i < DataW; i++) {
    for (int j = 0; j < DataH; j++) {
      assert(fabs(s_C[i] - s_A[i] - s_B[i]) <= 1e-5);
    }
  }
}

int main() {
  test_shared_memory();

  return 0;
}
