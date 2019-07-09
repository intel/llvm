// RUN: %clang -std=c++11 -g -fsycl %s -o %t.out -lstdc++ -lOpenCL -lsycl
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//==------------------- buffer.cpp - SYCL buffer basic test ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <CL/sycl.hpp>
#include <cassert>
#include <memory>

using namespace cl::sycl;

const std::string clKernelSourceCodeSimple = "\
__kernel void test(__global int* a, __global int* b) {\
    int i = get_global_id(0);\
    if (i < 256) { \
      b[i] = 0; \
    } \
}";

const std::string clKernelSourceCodeNonOverlap = "\
__kernel void test(__global int* a, __global int* b, __global int *c) {\
    int i = get_global_id(0);\
    if (i < 128) { \
      b[i] = 0; \
    } \
    if (i >= 256 && i < 384) { \
      c[i - 256] = 0; \
    } \
}";

const std::string clKernelSourceCodeOverlap = "\
__kernel void test1(__global int* a, __global int* b) {\
    int i = get_global_id(0);\
    if (i < 256) { \
      b[i] = 0; \
    } \
} \
__kernel void test2(__global int* a, __global int* b) {\
    int i = get_global_id(0);\
    if (i < 128) { \
      b[i] = 1; \
    } \
}";

int main() {
  bool Failed = false;
  {
    constexpr size_t n_size = ((1U << 10) / sizeof(int)) * 2;
    cl_int Error = CL_SUCCESS;
    int a[n_size];
    for (size_t i = 0; i < n_size; i++) {
      a[i] = i;
    }

    try {
      queue MyQueue;

      const char *src[] = {clKernelSourceCodeSimple.c_str()};
      const size_t src_size = clKernelSourceCodeSimple.size();

      cl_context clContext = MyQueue.get_context().get();
      cl_device_id clDevice = MyQueue.get_device().get();

      cl_program clProgram =
          clCreateProgramWithSource(clContext, 1, src, &src_size, &Error);

      CHECK_OCL_CODE(Error);

      Error = clBuildProgram(clProgram, 1, &clDevice, NULL, NULL, NULL);

      CHECK_OCL_CODE(Error);

      cl_kernel clKernel = clCreateKernel(clProgram, "test", &Error);

      CHECK_OCL_CODE(Error);

      buffer<int, 1> BufA(a, cl::sycl::range<1>(n_size));
      buffer<int, 1> BufB(BufA, n_size / 2, n_size / 2);

      kernel MyKernel(clKernel, MyQueue.get_context());

      MyQueue.submit([&](handler &cgh) {
        auto a_acc = BufA.get_access<access::mode::read>(cgh);
        auto b_acc = BufB.get_access<access::mode::write>(cgh);
        cgh.set_arg(0, a_acc);
        cgh.set_arg(1, b_acc);

        cgh.parallel_for(range<1>(n_size), MyKernel);
      });

      clReleaseKernel(clKernel);
      clReleaseProgram(clProgram);
    } catch (exception &ex) {
      std::cout << ex.what() << std::endl;
    }

    for (int i = 0; i < n_size; ++i) {
      if (i < n_size / 2 && a[i] != i) {
        std::cout << " array[" << i << "] is " << a[i] << " expected " << i
                  << std::endl;
        assert(false);
        Failed = true;
      } else if (i >= n_size / 2 && a[i] != 0) {
        std::cout << " array[" << i << "] is " << a[i] << " expected " << 0
                  << std::endl;
        assert(false);
        Failed = true;
      }
    }
  }
  {
    constexpr size_t n_size = ((1U << 10) / sizeof(int)) * 2;
    cl_int Error = CL_SUCCESS;
    int a[n_size];
    for (size_t i = 0; i < n_size; i++) {
      a[i] = i;
    }

    try {
      queue MyQueue;

      const char *src[] = {clKernelSourceCodeNonOverlap.c_str()};
      const size_t src_size = clKernelSourceCodeNonOverlap.size();

      cl_context clContext = MyQueue.get_context().get();
      cl_device_id clDevice = MyQueue.get_device().get();

      cl_program clProgram =
          clCreateProgramWithSource(clContext, 1, src, &src_size, &Error);

      CHECK_OCL_CODE(Error);

      Error = clBuildProgram(clProgram, 1, &clDevice, NULL, NULL, NULL);

      CHECK_OCL_CODE(Error);

      cl_kernel clKernel = clCreateKernel(clProgram, "test", &Error);

      CHECK_OCL_CODE(Error);

      buffer<int, 1> BufA(a, cl::sycl::range<1>(n_size));
      buffer<int, 1> BufB(BufA, 0, n_size / 4);
      buffer<int, 1> BufC(BufA, 2 * n_size / 4, n_size / 4);

      kernel MyKernel(clKernel, MyQueue.get_context());

      MyQueue.submit([&](handler &cgh) {
        auto a_acc = BufA.get_access<access::mode::read>(cgh);
        auto b_acc = BufB.get_access<access::mode::write>(cgh);
        auto c_acc = BufC.get_access<access::mode::write>(cgh);
        cgh.set_arg(0, a_acc);
        cgh.set_arg(1, b_acc);
        cgh.set_arg(2, c_acc);

        cgh.parallel_for(range<1>(n_size), MyKernel);
      });

      clReleaseKernel(clKernel);
      clReleaseProgram(clProgram);
    } catch (exception &ex) {
      std::cout << ex.what() << std::endl;
    }

    for (int i = 0; i < n_size; ++i) {
      if (i < n_size / 4 && a[i] != 0) {
        std::cout << " array[" << i << "] is " << a[i] << " expected " << 0
                  << std::endl;
        assert(false);
        Failed = true;
      } else if (i >= n_size / 4 && i < 2 * n_size / 4 && a[i] != i) {
        std::cout << " array[" << i << "] is " << a[i] << " expected " << i
                  << std::endl;
        assert(false);
        Failed = true;
      } else if (i >= 2 * n_size / 4 && i < 3 * n_size / 4 && a[i] != 0) {
        std::cout << " array[" << i << "] is " << a[i] << " expected " << 0
                  << std::endl;
        assert(false);
        Failed = true;
      } else if (i >= 3 * n_size / 4 && a[i] != i) {
        std::cout << " array[" << i << "] is " << a[i] << " expected " << i
                  << std::endl;
        assert(false);
        Failed = true;
      }
    }
  }
  {
    constexpr size_t n_size = ((1U << 10) / sizeof(int)) * 2;
    cl_int Error = CL_SUCCESS;
    int a[n_size];
    for (size_t i = 0; i < n_size; i++) {
      a[i] = i;
    }

    try {
      queue MyQueue;

      const char *src[] = {clKernelSourceCodeOverlap.c_str()};
      const size_t src_size = clKernelSourceCodeOverlap.size();

      cl_context clContext = MyQueue.get_context().get();
      cl_device_id clDevice = MyQueue.get_device().get();

      cl_program clProgram =
          clCreateProgramWithSource(clContext, 1, src, &src_size, &Error);

      CHECK_OCL_CODE(Error);

      Error = clBuildProgram(clProgram, 1, &clDevice, NULL, NULL, NULL);

      CHECK_OCL_CODE(Error);

      cl_kernel clKernel1 = clCreateKernel(clProgram, "test1", &Error);
      CHECK_OCL_CODE(Error);

      cl_kernel clKernel2 = clCreateKernel(clProgram, "test2", &Error);
      CHECK_OCL_CODE(Error);

      buffer<int, 1> BufA(a, cl::sycl::range<1>(n_size));
      buffer<int, 1> BufB(BufA, 0, n_size / 2);
      buffer<int, 1> BufC(BufA, n_size / 4, n_size / 4);

      kernel MyKernel1(clKernel1, MyQueue.get_context());
      kernel MyKernel2(clKernel2, MyQueue.get_context());

      MyQueue.submit([&](handler &cgh) {
        auto a_acc = BufA.get_access<access::mode::read>(cgh);
        auto b_acc = BufB.get_access<access::mode::write>(cgh);
        cgh.set_arg(0, a_acc);
        cgh.set_arg(1, b_acc);

        cgh.parallel_for(range<1>(n_size), MyKernel1);
      });

      MyQueue.submit([&](handler &cgh) {
        auto a_acc = BufA.get_access<access::mode::read>(cgh);
        auto c_acc = BufC.get_access<access::mode::write>(cgh);
        cgh.set_arg(0, a_acc);
        cgh.set_arg(1, c_acc);

        cgh.parallel_for(range<1>(n_size), MyKernel2);
      });

      clReleaseKernel(clKernel1);
      clReleaseKernel(clKernel2);
      clReleaseProgram(clProgram);
    } catch (exception &ex) {
      std::cout << ex.what() << std::endl;
    }

    for (int i = 0; i < n_size; ++i) {
      if (i < n_size / 4 && a[i] != 0) {
        std::cout << " array[" << i << "] is " << a[i] << " expected " << 0
                  << std::endl;
        assert(false);
        Failed = true;
      } else if (i >= n_size / 4 && i < 2 * n_size / 4 && a[i] != 1) {
        std::cout << " array[" << i << "] is " << a[i] << " expected " << i
                  << std::endl;
        assert(false);
        Failed = true;
      } else if (i >= 2 * n_size / 4 && a[i] != i) {
        std::cout << " array[" << i << "] is " << a[i] << " expected " << i
                  << std::endl;
        assert(false);
        Failed = true;
      }
    }
  }
  return Failed;
}
