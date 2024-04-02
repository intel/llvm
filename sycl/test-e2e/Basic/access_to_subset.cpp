// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

//==---------- access_to_subset.cpp --- access to subset of buffer test ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <iostream>
#include <sycl/detail/core.hpp>

using namespace sycl;
using acc_w = accessor<int, 2, access::mode::write, access::target::device>;

int main() {

  const int M = 6;
  const int N = 7;
  int result[M][N] = {0};
  bool failed = false;
  {
    auto origRange = range<2>(M, N);
    buffer<int, 2> Buffer(origRange);
    Buffer.set_final_data((int *)result);
    auto offset = id<2>(1, 1);
    auto subRange = range<2>(M - 2, N - 2);
    queue myQueue;
    myQueue.submit([&](handler &cgh) {
      acc_w B(Buffer, cgh);
      cgh.parallel_for<class bufferByRange2_init>(
          origRange, [=](id<2> index) { B[index] = 0; });
    });
    myQueue.submit([&](handler &cgh) {
      acc_w B(Buffer, cgh, subRange, offset);
      cgh.parallel_for<class bufferByRange2>(
          subRange, [=](id<2> index) { B[index] = 1; });
    });
  }

  // Check that we filled correct subset of buffer:
  // 0000000     0000000
  // 0000000     0111110
  // 0000000 --> 0111110
  // 0000000     0111110
  // 0000000     0111110
  // 0000000     0000000

  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      size_t expected =
          ((i == 0) || (i == M - 1) || (j == 0) || (j == N - 1)) ? 0 : 1;
      if (result[i][j] != expected) {
        std::cout << "line: " << __LINE__ << " result[" << i << "][" << j
                  << "] is " << result[i][j] << " expected " << expected
                  << std::endl;
        failed = true;
      }
    }
  }
  return failed;
}
