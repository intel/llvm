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
 *  helloworld.cpp
 *
 *  Description:
 *    Checks that the SYCLcompat example program compiles and runs
 **************************************************************************/

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/sycl.hpp>
#include <syclcompat/syclcompat.hpp>

#include <cstdlib>
#include <iostream>

#define CHECK_MEMORY(ptr) \
  if ((ptr) == nullptr) { \
    std::cerr << "Failed to allocate memory: " << ( #ptr ) << "\n"; \
    exit(EXIT_FAILURE); \
  }

/**
 * Slope intercept form of a straight line equation: Y = m * X + b
 */
template <int BLOCK_SIZE>
void slope_intercept(float *Y, float *X, float m, float b, size_t n) {

  // Block index
  size_t bx = syclcompat::work_group_id::x();
  // Thread index
  size_t tx = syclcompat::local_id::x();

  size_t i = bx * BLOCK_SIZE + tx;
  // or  i = syclcompat::global_id::x();
  if (i < n)
    Y[i] = m * X[i] + b;
}

/**
 * Program main
 */
int main(int argc, char **argv) {
  std::cout << "Simple Kernel example" << "\n";

  constexpr size_t n_points = 32;
  constexpr float m = 1.5f;
  constexpr float b = 0.5f;

  int block_size = 32;
  if (block_size > syclcompat::get_current_device()
                       .get_info<sycl::info::device::max_work_group_size>()) {
    block_size = 16;
  }

  std::cout << "block_size = " << block_size << ", n_points = " << n_points
            << "\n";

  // Allocate host memory for vectors X and Y
  size_t mem_size = n_points * sizeof(float);
  float *h_X = (float *)syclcompat::malloc_host(mem_size);
  float *h_Y = (float *)syclcompat::malloc_host(mem_size);
  CHECK_MEMORY(h_X);
  CHECK_MEMORY(h_Y);

  // Alternative templated allocation for the expected output
  float *h_expected = syclcompat::malloc_host<float>(n_points);
  CHECK_MEMORY(h_expected);

  // Initialize host memory & expected output
  for (size_t i = 0; i < n_points; i++) {
    h_X[i] = i + 1;
    h_expected[i] = m * h_X[i] + b;
  }

  // Allocate device memory
  float *d_X = (float *)syclcompat::malloc(mem_size);
  float *d_Y = (float *)syclcompat::malloc(mem_size);
  CHECK_MEMORY(d_X);
  CHECK_MEMORY(d_Y);

  // copy host memory to device
  syclcompat::memcpy(d_X, h_X, mem_size);

  size_t threads = block_size;
  size_t grid = n_points / block_size;

  std::cout << "Computing result using SYCL Kernel... ";
  if (block_size == 16) {
    syclcompat::launch<slope_intercept<16>>(grid, threads, d_Y, d_X, m, b,
                                        n_points);
  } else {
    syclcompat::launch<slope_intercept<32>>(grid, threads, d_Y, d_X, m, b,
                                        n_points);
  }
  syclcompat::wait();
  std::cout << "DONE" << "\n";

  // Async copy result from device to host
  syclcompat::memcpy_async(h_Y, d_Y, mem_size).wait();

  // Check output
  for (size_t i = 0; i < n_points; i++) {
    if (std::abs(h_Y[i] - h_expected[i]) >= 1e-6) {
      std::cerr << "Mismatch at index " << i << ": expected " << h_expected[i]
          << ", but got " << h_Y[i] << "\n";
      exit(EXIT_FAILURE);
    }
  }

  // Clean up memory
  syclcompat::free(h_X);
  syclcompat::free(h_Y);
  syclcompat::free(h_expected);
  syclcompat::free(d_X);
  syclcompat::free(d_Y);

  return EXIT_SUCCESS;
}
