// ====------ util_fast_length_test.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#define SYCLCOMPAT_USM_LEVEL_NONE
#include <sycl/detail/core.hpp>
#include <syclcompat/syclcompat.hpp>

void fast_length_test() {

  {
    float inputData_0(0.8970062715);

    sycl::range<1> ndRng(1);
    float *kernelResult = new float[1];
    auto testQueue = syclcompat::get_default_queue();
    {
      sycl::buffer<float, 1> buffer(kernelResult, ndRng);

      testQueue.submit([&](sycl::handler &h) {
        auto resultPtr =
            buffer.template get_access<sycl::access::mode::write>(h);

        h.single_task(
            [=]() { resultPtr[0] = syclcompat::fast_length(&inputData_0, 1); });
      });
    }
    testQueue.wait_and_throw();
    if (fabs(inputData_0 - *kernelResult) > 1e-5) {
      printf("fast_length_test 1 failed\n");
      exit(-1);
    }
    delete[] kernelResult;
  }

  {
    float inputData_0[2] = {0.8335529744, 0.7346600673};

    sycl::range<1> ndRng(1);
    float *kernelResult = new float[1];
    auto testQueue = syclcompat::get_default_queue();
    {
      sycl::buffer<float, 1> buffer(kernelResult, ndRng);

      testQueue.submit([&](sycl::handler &h) {
        auto resultPtr =
            buffer.template get_access<sycl::access::mode::write>(h);

        h.single_task(
            [=]() { resultPtr[0] = syclcompat::fast_length(&inputData_0[0], 2); });
      });
    }
    testQueue.wait_and_throw();

    if (fabs(sqrtf(0.8335529744 * 0.8335529744 + 0.7346600673 * 0.7346600673) -
             *kernelResult) > 1e-5) {
      printf("fast_length_test 2 failed\n");
      exit(-1);
    }

    delete[] kernelResult;
  }

  {
    float inputData_0[3] = {0.1658983906, 0.590226484, 0.4891553616};

    sycl::range<1> ndRng(1);
    float *kernelResult = new float[1];
    auto testQueue = syclcompat::get_default_queue();
    {
      sycl::buffer<float, 1> buffer(kernelResult, ndRng);

      testQueue.submit([&](sycl::handler &h) {
        auto resultPtr =
            buffer.template get_access<sycl::access::mode::write>(h);

        h.single_task(
            [=]() { resultPtr[0] = syclcompat::fast_length(&inputData_0[0], 3); });
      });
    }
    testQueue.wait_and_throw();

    if (fabs(sqrtf(0.1658983906 * 0.1658983906 + 0.590226484 * 0.590226484 +
                   0.4891553616 * 0.4891553616) -
             *kernelResult) > 1e-5) {
      printf("fast_length_test 3 failed\n");
      exit(-1);
    }

    delete[] kernelResult;
  }

  {
    float inputData_0[4] = {0.6041178723, 0.7760620605, 0.2944284976,
                            0.6851913766};

    sycl::range<1> ndRng(1);
    float *kernelResult = new float[1];
    auto testQueue = syclcompat::get_default_queue();
    {
      sycl::buffer<float, 1> buffer(kernelResult, ndRng);

      testQueue.submit([&](sycl::handler &h) {
        auto resultPtr =
            buffer.template get_access<sycl::access::mode::write>(h);

        h.single_task(
            [=]() { resultPtr[0] = syclcompat::fast_length(&inputData_0[0], 4); });
      });
    }
    testQueue.wait_and_throw();

    if (fabs(sqrtf(0.6041178723 * 0.6041178723 + 0.7760620605 * 0.7760620605 +
                   0.2944284976 * 0.2944284976 + 0.6851913766 * 0.6851913766) -
             *kernelResult) > 1e-5) {
      printf("fast_length_test 4 failed\n");
      exit(-1);
    }

    delete[] kernelResult;
  }

  {
    float inputData_0[5] = {0.6041178723, 0.7760620605, 0.2944284976,
                            0.6851913766, 0.6851913766};

    sycl::range<1> ndRng(1);
    float *kernelResult = new float[1];
    auto testQueue = syclcompat::get_default_queue();
    {
      sycl::buffer<float, 1> buffer(kernelResult, ndRng);

      testQueue.submit([&](sycl::handler &h) {
        auto resultPtr =
            buffer.template get_access<sycl::access::mode::write>(h);

        h.single_task(
            [=]() { resultPtr[0] = syclcompat::fast_length(&inputData_0[0], 5); });
      });
    }
    testQueue.wait_and_throw();

    if (fabs(sqrtf(0.6041178723 * 0.6041178723 + 0.7760620605 * 0.7760620605 +
                   0.2944284976 * 0.2944284976 + 0.6851913766 * 0.6851913766 +
                   0.6851913766 * 0.6851913766) -
             *kernelResult) > 1e-5) {
      printf("fast_length_test 5 failed\n");
      exit(-1);
    }

    delete[] kernelResult;
  }
  printf("fast_length test is passed!\n");
}

int main() {

  fast_length_test();

  return 0;
}
