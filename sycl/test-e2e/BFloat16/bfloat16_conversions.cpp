// RUN: %{build} -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %{run} %t.out

// Currently the feature is supported only on CPU and GPU, natively or by
// software emulation.
// UNSUPPORTED: accelerator

// FIXME: enable opaque pointers support on CPU.
// UNSUPPORTED: cpu

//==---------- bfloat16_conversions.cpp - SYCL bfloat16 type test ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include <iostream>
#include <sycl/detail/core.hpp>

using namespace sycl;

template <typename T> T calculate(T a, T b) {
  sycl::ext::oneapi::bfloat16 x = -a;
  sycl::ext::oneapi::bfloat16 y = b;
  sycl::ext::oneapi::bfloat16 z = x + y;
  T result = z;
  return result;
}

template <typename T> int test_device(queue Q) {
  T data[3] = {-7.0f, 8.1f, 0.0f};

  buffer<T, 1> buf{data, 3};
  Q.submit([&](handler &cgh) {
    accessor numbers{buf, cgh, read_write};
    cgh.single_task([=]() { numbers[2] = calculate(numbers[0], numbers[1]); });
  });

  host_accessor hostOutAcc{buf, read_only};
  std::cout << "Device Result = " << hostOutAcc[2] << std::endl;
  if (hostOutAcc[2] == 15.125f)
    return 0;
  return 1;
}

template <typename T> int test_host() {
  T a{-5.6f};
  T b{-1.1f};
  T result = calculate(a, b);
  std::cout << "Host Result = " << result << std::endl;
  if (result == 4.5f)
    return 0;
  return 1;
}

int main() {
  queue Q;
  int result;
  result = test_host<sycl::half>();
  result |= test_host<float>();
  if (Q.get_device().has(aspect::fp16))
    result |= test_device<sycl::half>(Q);
  result |= test_device<float>(Q);
  if (result)
    std::cout << "FAIL\n";
  else
    std::cout << "PASS\n";

  return result;
}
