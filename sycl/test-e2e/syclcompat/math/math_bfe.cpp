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
 *  math_bfe.cpp
 *
 *  Description:
 *    math bitfield extract tests
 **************************************************************************/

// ===----------- math_bfe.cpp ------------------ -*- C++ -* --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <bitset>
#include <chrono>
#include <iostream>
#include <limits.h>
#include <random>
#include <stdint.h>
#include <sycl/detail/core.hpp>
#include <syclcompat/math.hpp>
#include <type_traits>
#include <vector>

template <typename T>
inline std::enable_if_t<std::is_integral_v<T>, T>
bfe_slow(const T source, const uint32_t bit_start, const uint32_t num_bits) {
  const uint32_t msb =
      std::numeric_limits<unsigned char>::digits * sizeof(T) - 1;
  const uint32_t pos = bit_start;
  const uint32_t len = num_bits;

  // If the requested bit field length is zero, the result is zero.
  if (num_bits == 0)
    return 0ULL;

  T sbit;
  std::bitset<std::numeric_limits<unsigned char>::digits * sizeof(T)>
      source_bitset(source);
  if (std::is_unsigned_v<T> || len == 0)
    sbit = 0;
  else
    sbit = source_bitset[std::min(pos + len - 1, msb)];

  // If the start position is beyond the msb of the input, the destination d is
  // filled with the replicated sign bit of the extracted field.
  // -1 is 1111...
  if (bit_start > msb)
    return -sbit;

  std::bitset<std::numeric_limits<unsigned char>::digits * sizeof(T)>
      result_bitset;
  for (uint8_t i = 0; i <= msb; ++i)
    result_bitset[i] =
        (i < len && pos + i <= msb) ? source_bitset[pos + i] : sbit;
  return result_bitset.to_ullong();
}

template <typename T> bool test(const char *Msg, int N) {
  uint32_t bit_width = std::numeric_limits<unsigned char>::digits * sizeof(T);
  T min_value = std::numeric_limits<T>::lowest();
  T max_value = std::numeric_limits<T>::max();
  std::random_device rd;
  std::mt19937::result_type seed =
      rd() ^
      ((std::mt19937::result_type)
           std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch())
               .count() +
       (std::mt19937::result_type)
           std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::high_resolution_clock::now().time_since_epoch())
               .count());

  std::mt19937 gen(seed);
  // Support for char type with uniform_int_distribution isn't universal
  using RandomDataT = std::conditional_t<sizeof(T) == 1, int, T>;
  std::uniform_int_distribution<RandomDataT> rd_source(min_value, max_value);

  // Define a small overshoot so that we adequately test out-of-range cases
  // without sacrificing depth of testing of valid start+length combinations
  constexpr uint32_t overshoot = 2;
  std::uniform_int_distribution<uint32_t> rd_start(0, bit_width + overshoot);
  std::uniform_int_distribution<uint32_t> rd_length(0, bit_width + overshoot);

  std::vector<T> sources(N, 0);
  std::vector<T> compat_results(N, 0);
  std::vector<T> slow_results(N, 0);
  std::vector<uint32_t> starts(N, 0);
  std::vector<uint32_t> lengths(N, 0);
  for (int i = 0; i < N; ++i) {
    sources[i] = static_cast<T>(rd_source(gen));
    starts[i] = rd_start(gen);
    lengths[i] = rd_length(gen);
  }

  sycl::buffer<T, 1> source_buffer(sources.data(), N);
  sycl::buffer<T, 1> compat_results_buffer(compat_results.data(), N);
  sycl::buffer<T, 1> slow_results_buffer(slow_results.data(), N);
  sycl::buffer<uint32_t, 1> starts_buffer(starts.data(), N);
  sycl::buffer<uint32_t, 1> lengths_buffer(lengths.data(), N);

  sycl::queue que;
  que.submit([&](sycl::handler &handler) {
    sycl::accessor source_accessor(source_buffer, handler, sycl::read_only);
    sycl::accessor start_accessor(starts_buffer, handler, sycl::read_only);
    sycl::accessor length_accessor(lengths_buffer, handler, sycl::read_only);
    sycl::accessor compat_result_accessor(compat_results_buffer, handler,
                                          sycl::write_only);
    handler.parallel_for(N, [=](sycl::id<1> i) {
      compat_result_accessor[i] = syclcompat::bfe_safe<T>(
          source_accessor[i], start_accessor[i], length_accessor[i]);
    });
  });

  que.submit([&](sycl::handler &handler) {
    sycl::accessor source_accessor(source_buffer, handler, sycl::read_only);
    sycl::accessor start_accessor(starts_buffer, handler, sycl::read_only);
    sycl::accessor length_accessor(lengths_buffer, handler, sycl::read_only);
    sycl::accessor slow_result_accessor(slow_results_buffer, handler,
                                        sycl::write_only);
    handler.parallel_for(N, [=](sycl::id<1> i) {
      slow_result_accessor[i] = bfe_slow<T>(
          source_accessor[i], start_accessor[i], length_accessor[i]);
    });
  });

  que.wait_and_throw();
  sycl::host_accessor source_accessor(source_buffer, sycl::read_only);
  sycl::host_accessor start_accessor(starts_buffer, sycl::read_only);
  sycl::host_accessor length_accessor(lengths_buffer, sycl::read_only);
  sycl::host_accessor compat_result_accessor(compat_results_buffer,
                                             sycl::read_only);
  sycl::host_accessor slow_result_accessor(slow_results_buffer,
                                           sycl::read_only);

  int failed = 0;
  for (int i = 0; i < N; ++i) {
    if (compat_result_accessor[i] != slow_result_accessor[i]) {
      failed++;
      std::cout << "[source = " << source_accessor[i]
                << ", bit_start = " << start_accessor[i]
                << ", num_bits = " << length_accessor[i] << "] failed, expect "
                << slow_result_accessor[i] << " but got "
                << compat_result_accessor[i] << std::endl;
    }
  }
  std::cout << "===============" << std::endl;
  std::cout << "Test: " << Msg << std::endl;
  std::cout << "Total: " << N << std::endl;
  std::cout << "Success: " << N - failed << std::endl;
  std::cout << "Failed: " << failed << std::endl;
  std::cout << "===============" << std::endl;
  return !failed;
}

int main() {
  const int N = 1000;
  assert(test<int8_t>("int8", N));
  assert(test<uint8_t>("uint8", N));
  assert(test<int16_t>("int16", N));
  assert(test<uint16_t>("uint16", N));
  assert(test<int32_t>("int32", N));
  assert(test<uint32_t>("uint32", N));
  assert(test<int64_t>("int64", N));
  assert(test<uint64_t>("uint64", N));
  return 0;
}
