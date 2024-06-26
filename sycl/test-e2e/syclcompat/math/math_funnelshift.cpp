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
 *  math_funnelshift.cpp
 *
 *  Description:
 *    math funnel helpers tests
 **************************************************************************/

// RUN: %clangxx -fsycl -fsycl-targets=%{sycl_triple} %s -o %t.out
// RUN: %{run} %t.out

#include <syclcompat/device.hpp>
#include <syclcompat/math.hpp>
#include <syclcompat/memory.hpp>

void testFunnelShiftKernel(int *const TestResults) {
  TestResults[0] = (syclcompat::funnelshift_l(0xAA000000, 0xBB, 8) == 0xBBAA);
  TestResults[1] =
      (syclcompat::funnelshift_lc(0xAA000000, 0xBB, 16) == 0xBBAA00);
  TestResults[2] = (syclcompat::funnelshift_r(0xAA00, 0xBB, 8) == 0xBB0000AA);
  TestResults[3] = (syclcompat::funnelshift_rc(0xAA0000, 0xBB, 16) == 0xBB00AA);
}

int main() {
  constexpr int nTests = 4;

  sycl::queue q = syclcompat::get_default_queue();
  int *testResults = syclcompat::malloc<int>(nTests, q);
  int *testResultsHost = syclcompat::malloc_host<int>(nTests, q);
  syclcompat::fill<int>(testResults, 0, nTests, q);

  q.submit([&](sycl::handler &cgh) {
     cgh.parallel_for(
         1, [=](sycl::item<1> it) { testFunnelShiftKernel(testResults); });
   }).wait_and_throw();

  syclcompat::memcpy<int>(testResultsHost, testResults, nTests, q);

  for (int i = 0; i < nTests; i++) {
    if (testResultsHost[i] == 0) {
      std::cerr << "funnelshift test " << i << " failed" << std::endl;
      return 1;
    }
  }
  syclcompat::free(testResults, q);
  syclcompat::free(testResultsHost, q);

  return 0;
}
