// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Test bitwise NOT operator (~) on marray<bool>
//
// Per the SYCL spec, marray operators mirror the equivalent std::valarray
// operator. For bool, ~ promotes to int, so ~v is always non-zero and casting
// back to bool yields true for every element (matches ~std::valarray<bool>).

#include <sycl/detail/core.hpp>
#include <sycl/marray.hpp>

#include <cassert>
#include <iostream>

using namespace sycl;

template <size_t N>
bool test_bitwise_not_bool(queue &q, const marray<bool, N> &input,
                           const marray<bool, N> &expected) {
  bool result[N];
  {
    buffer<bool> out_buf(result, N);
    q.submit([&](handler &h) {
       accessor out_acc(out_buf, h, write_only);
       h.single_task([=]() {
         marray<bool, N> res = ~input;
         for (size_t i = 0; i < N; ++i) {
           out_acc[i] = res[i];
         }
       });
     }).wait();
  }

  // Verify results
  for (size_t i = 0; i < N; ++i) {
    if (result[i] != expected[i]) {
      std::cout << "FAILED at index " << i << ": input=" << input[i]
                << ", expected=" << expected[i] << ", got=" << result[i]
                << std::endl;
      return false;
    }
  }
  return true;
}

int main() {
  queue q;

  std::cout << "Testing bitwise NOT (~) on marray<bool>\n";

  // Test case 1: Size 2
  {
    marray<bool, 2> input{true, false};
    marray<bool, 2> expected{true, true};
    assert(test_bitwise_not_bool(q, input, expected) &&
           "Test failed for size 2");
  }

  // Test case 2: Size 4
  {
    marray<bool, 4> input{true, false, true, false};
    marray<bool, 4> expected{true, true, true, true};
    assert(test_bitwise_not_bool(q, input, expected) &&
           "Test failed for size 4");
  }

  // Test case 3: Size 3 (no padding, different code path)
  {
    marray<bool, 3> input{false, true, false};
    marray<bool, 3> expected{true, true, true};
    assert(test_bitwise_not_bool(q, input, expected) &&
           "Test failed for size 3");
  }

  // Test case 4: Size 8
  {
    marray<bool, 8> input{true, true, false, false, true, false, true, false};
    marray<bool, 8> expected{true, true, true, true, true, true, true, true};
    assert(test_bitwise_not_bool(q, input, expected) &&
           "Test failed for size 8");
  }

  // Test case 5: All true
  {
    marray<bool, 4> input{true, true, true, true};
    marray<bool, 4> expected{true, true, true, true};
    assert(test_bitwise_not_bool(q, input, expected) &&
           "Test failed for all true");
  }

  // Test case 6: All false
  {
    marray<bool, 4> input{false, false, false, false};
    marray<bool, 4> expected{true, true, true, true};
    assert(test_bitwise_not_bool(q, input, expected) &&
           "Test failed for all false");
  }

  std::cout << "All tests passed!\n";
  return 0;
}
