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

#include <sycl/ext/oneapi/bfloat16.hpp>

using namespace sycl;
using bfloat16 = sycl::ext::oneapi::bfloat16;

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

int test_host_vector_conversions() {
  bool Passed = true;
  std::cout << "float[4] -> bfloat16[4] -> float[4] conversion on host..."
            << std::flush;

  float FloatArray[4] = {1.0f, 2.0f, 3.0f, 4.0f};

  // float[4] -> bfloat16[4]
  bfloat16 BFloatArray[4];
  sycl::ext::oneapi::detail::FloatVecToBF16Vec<4>(FloatArray, BFloatArray);

  // bfloat16[4] -> float[4]
  float NewFloatArray[4];
  sycl::ext::oneapi::detail::BF16VecToFloatVec<4>(BFloatArray, NewFloatArray);

  // Check results.
  for (int i = 0; i < 4; ++i)
    Passed &= (FloatArray[i] == NewFloatArray[i]);

  if (Passed)
    std::cout << "passed\n";
  else
    std::cout << "failed\n";

  return !Passed;
}

int test_device_vector_conversions(queue Q) {
  int err = 0;
  buffer<int> err_buf(&err, 1);

  std::cout << "float[4] -> bfloat16[4] conversion on device..." << std::flush;
  // Convert float array to bfloat16 array
  Q.submit([&](handler &CGH) {
     accessor<int, 1, access::mode::write, target::device> ERR(err_buf, CGH);
     CGH.single_task([=]() {
       float FloatArray[4] = {1.0f, -1.0f, 0.0f, 2.0f};
       bfloat16 BF16Array[4];
       sycl::ext::oneapi::detail::FloatVecToBF16Vec<4>(FloatArray, BF16Array);
       for (int i = 0; i < 4; i++) {
         if (FloatArray[i] != (float)BF16Array[i]) {
           ERR[0] = 1;
         }
       }
     });
   }).wait();

  if (err)
    std::cout << "failed\n";
  else
    std::cout << "passed\n";

  std::cout << "bfloat16[4] -> float[4] conversion on device..." << std::flush;
  // Convert bfloat16 array back to float array
  Q.submit([&](handler &CGH) {
     accessor<int, 1, access::mode::write, target::device> ERR(err_buf, CGH);
     CGH.single_task([=]() {
       bfloat16 BF16Array[3] = {1.0f, 0.0f, -1.0f};
       float FloatArray[3];
       sycl::ext::oneapi::detail::BF16VecToFloatVec<4>(BF16Array, FloatArray);
       for (int i = 0; i < 3; i++) {
         if (FloatArray[i] != (float)BF16Array[i]) {
           ERR[0] = 1;
         }
       }
     });
   }).wait();

  if (err)
    std::cout << "failed\n";
  else
    std::cout << "passed\n";

  return err;
}

int main() {
  queue Q;
  int result;
  result = test_host<sycl::half>();
  result |= test_host<float>();
  if (Q.get_device().has(aspect::fp16))
    result |= test_device<sycl::half>(Q);
  result |= test_device<float>(Q);

  // Test vector BF16 -> float conversion and vice versa.
  result |= test_host_vector_conversions();
  result |= test_device_vector_conversions(Q);

  if (result)
    std::cout << "FAIL\n";
  else
    std::cout << "PASS\n";

  return result;
}
