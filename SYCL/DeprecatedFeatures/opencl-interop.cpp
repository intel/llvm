// REQUIRES: opencl

// RUN: %clangxx -fsycl %s -D__SYCL_INTERNAL_API -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

//==---------------- opencl-interop.cpp - SYCL linear id test --------------==//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <sycl/sycl.hpp>

using namespace cl::sycl;

int main(int argc, char *argv[]) {
  queue q;
  if (q.is_host()) {
    std::cout << "Skipping test\n";
    return 0;
  }

  // Compute expected answer.
  const uint32_t dimA = 2;
  const uint32_t dimB = 8;
  std::vector<int> input(dimA * dimB), output(dimA), expected(dimA);
  std::iota(input.begin(), input.end(), 0);
  for (int j = 0; j < dimA; ++j) {
    int sum = 0;
    for (int i = 0; i < dimB; ++i) {
      sum += input[j * dimB + i];
    }
    expected[j] = sum;
  }

  // Compute sum using one work-group per element of dimA
  program prog(q.get_context(), {q.get_device()});
  prog.build_with_source("__kernel void sum(__global const int* input, "
                         "__global int* output, const int dimA, const int dimB)"
                         "{"
                         "  int j = get_global_id(1);"
                         "  int i = get_global_id(0);"
                         "  int sum = work_group_reduce_add(input[j*dimB+i]);"
                         "  if (get_local_id(0) == 0)"
                         "  {"
                         "    output[j] = sum;"
                         "  }"
                         "}",
                         "-cl-std=CL2.0");
  kernel sum = prog.get_kernel("sum");
  {
    buffer<int, 2> input_buf(input.data(), range<2>(dimA, dimB));
    buffer<int, 1> output_buf(output.data(), range<1>(dimA));
    q.submit([&](handler &cgh) {
      auto input = input_buf.get_access<access::mode::read>(cgh);
      auto output = output_buf.get_access<access::mode::discard_write>(cgh);
      cgh.set_args(input, output, dimA, dimB);
      cgh.parallel_for(nd_range<2>(range<2>(dimA, dimB), range<2>(1, dimB)),
                       sum);
    });
  }

  // Compare with expected result
  for (int j = 0; j < dimA; ++j) {
    assert(output[j] == expected[j]);
  }
  std::cout << "Test passed.\n";
  return 0;
}
