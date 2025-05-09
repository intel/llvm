// RUN: %{build} -o %t1.out
// RUN: %{run} %t1.out
// clang-format off
// UNSUPPORTED: opencl
// UNSUPPORTED-TRACKER: https://github.com/oneapi-src/unified-runtime/issues/2440
// clang-format on

/**
 * Test of the queue::fill interface with a range of pattern sizes and values.
 *
 * Loops over pattern sizes from 1 to MaxPatternSize bytes and calls queue::fill
 * with std::array<uint8_t,Size> for the pattern. Two pattern values are tested,
 * all zeros and value=index+42. The output is copied back to host and
 * validated.
 */

#include <array>
#include <cstdio>
#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

constexpr size_t MaxPatternSize{32}; // Bytes.
constexpr size_t NumElements{10};
constexpr bool verbose{false};

template <size_t PatternSize, bool SameValue>
int test(sycl::queue &q, uint8_t firstValue = 0) {
  using T = std::array<uint8_t, PatternSize>;
  T value{};

  // Initialize the pattern value on host.
  for (size_t i{0}; i < PatternSize; ++i) {
    if constexpr (SameValue) {
      value[i] = firstValue;
    } else {
      value[i] = firstValue + i;
    }
  }

  // Allocate memory on the device.
  T *dptr{sycl::malloc_device<T>(NumElements, q)};

  // Fill the device memory with the pattern.
  q.fill(dptr, value, NumElements).wait();

  // Copy back the filled memory to host.
  std::array<T, NumElements> host{};
  q.copy<T>(dptr, host.data(), NumElements).wait();

  // Validate whether the filled memory contains the expected values.
  bool pass{true};
  for (size_t i{0}; i < NumElements; ++i) {
    for (size_t j{0}; j < PatternSize; ++j) {
      if (host[i][j] != value[j]) {
        pass = false;
      }
    }
  }

  // Release the device memory allocation.
  sycl::free(dptr, q);

  // Print info on failure or in verbose mode.
  if (!pass || verbose) {
    printf("Pattern size %3zu bytes, %s values (initial %3u) %s\n", PatternSize,
           (SameValue ? " equal" : "varied"), firstValue,
           (pass ? "== PASS ==" : "== FAIL =="));
  }

  return !pass;
}

template <size_t Size> int testOneSize(sycl::queue &q) {
  return test<Size, true>(q, 0) + test<Size, false>(q, 42);
}

template <size_t... Sizes>
int testSizes(sycl::queue &q, std::index_sequence<Sizes...>) {
  return (testOneSize<1u + Sizes>(q) + ...);
}

int main() {
  sycl::queue q{};
  int failures = testSizes(q, std::make_index_sequence<MaxPatternSize>{});
  if (failures > 0) {
    printf("%d / %zu tests failed\n", failures, 2u * MaxPatternSize);
  } else {
    printf("All %zu tests passed\n", 2u * MaxPatternSize);
  }
  return failures;
}
