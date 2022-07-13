// REQUIRES: opencl, opencl_icd

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -D__SYCL_INTERNAL_API %s -o %t.out %opencl_lib
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

//==------------ subbuffer_interop.cpp - SYCL buffer basic test ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

#include <cassert>
#include <memory>
#include <numeric>

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
    // Test if we can write into subbufer from OpenCL code.
    const size_t NSize = 512;
    cl_int Error = CL_SUCCESS;
    int AMem[NSize];
    for (size_t i = 0; i < NSize; i++) {
      AMem[i] = i;
    }

    try {
      queue TestQueue;

      const char *SrcString[] = {clKernelSourceCodeSimple.c_str()};
      const size_t SrcStringSize = clKernelSourceCodeSimple.size();

      cl_context clContext = TestQueue.get_context().get();
      cl_device_id clDevice = TestQueue.get_device().get();

      cl_program clProgram = clCreateProgramWithSource(clContext, 1, SrcString,
                                                       &SrcStringSize, &Error);

      assert(Error == CL_SUCCESS);

      Error = clBuildProgram(clProgram, 1, &clDevice, NULL, NULL, NULL);

      assert(Error == CL_SUCCESS);

      cl_kernel clKernel = clCreateKernel(clProgram, "test", &Error);

      assert(Error == CL_SUCCESS);

      buffer<int, 1> BufA(AMem, cl::sycl::range<1>(NSize));
      buffer<int, 1> BufB(BufA, NSize / 2, NSize / 2);

      kernel TestKernel(clKernel, TestQueue.get_context());

      TestQueue.submit([&](handler &cgh) {
        auto a_acc = BufA.get_access<access::mode::read>(cgh);
        auto b_acc = BufB.get_access<access::mode::write>(cgh);
        cgh.set_arg(0, a_acc);
        cgh.set_arg(1, b_acc);

        cgh.parallel_for(range<1>(NSize), TestKernel);
      });

      clReleaseKernel(clKernel);
      clReleaseProgram(clProgram);
    } catch (exception &ex) {
      std::cout << ex.what() << std::endl;
    }

    for (int i = 0; i < NSize; ++i) {
      if (i < NSize / 2 && AMem[i] != i) {
        std::cout << " array[" << i << "] is " << AMem[i] << " expected " << i
                  << std::endl;
        assert(false);
        Failed = true;
      } else if (i >= NSize / 2 && AMem[i] != 0) {
        std::cout << " array[" << i << "] is " << AMem[i] << " expected " << 0
                  << std::endl;
        assert(false);
        Failed = true;
      }
    }
  }
  {
    // Test if we can use two sub buffers, pointing to one buffer, from OpenCL.
    const size_t NSize = 512;
    cl_int Error = CL_SUCCESS;
    int AMem[NSize];
    for (size_t i = 0; i < NSize; i++) {
      AMem[i] = i;
    }

    try {
      queue TestQueue;

      const char *SrcString[] = {clKernelSourceCodeNonOverlap.c_str()};
      const size_t SrcStringSize = clKernelSourceCodeNonOverlap.size();

      cl_context clContext = TestQueue.get_context().get();
      cl_device_id clDevice = TestQueue.get_device().get();

      cl_program clProgram = clCreateProgramWithSource(clContext, 1, SrcString,
                                                       &SrcStringSize, &Error);

      assert(Error == CL_SUCCESS);

      Error = clBuildProgram(clProgram, 1, &clDevice, NULL, NULL, NULL);

      assert(Error == CL_SUCCESS);

      cl_kernel clKernel = clCreateKernel(clProgram, "test", &Error);

      assert(Error == CL_SUCCESS);

      buffer<int, 1> BufA(AMem, cl::sycl::range<1>(NSize));
      buffer<int, 1> BufB(BufA, 0, NSize / 4);
      buffer<int, 1> BufC(BufA, 2 * NSize / 4, NSize / 4);

      kernel TestKernel(clKernel, TestQueue.get_context());

      TestQueue.submit([&](handler &cgh) {
        auto a_acc = BufA.get_access<access::mode::read>(cgh);
        auto b_acc = BufB.get_access<access::mode::write>(cgh);
        auto c_acc = BufC.get_access<access::mode::write>(cgh);
        cgh.set_arg(0, a_acc);
        cgh.set_arg(1, b_acc);
        cgh.set_arg(2, c_acc);

        cgh.parallel_for(range<1>(NSize), TestKernel);
      });

      clReleaseKernel(clKernel);
      clReleaseProgram(clProgram);
    } catch (exception &ex) {
      std::cout << ex.what() << std::endl;
    }

    for (int i = 0; i < NSize; ++i) {
      if (i < NSize / 4 && AMem[i] != 0) {
        std::cout << " array[" << i << "] is " << AMem[i] << " expected " << 0
                  << std::endl;
        assert(false);
        Failed = true;
      } else if (i >= NSize / 4 && i < 2 * NSize / 4 && AMem[i] != i) {
        std::cout << " array[" << i << "] is " << AMem[i] << " expected " << i
                  << std::endl;
        assert(false);
        Failed = true;
      } else if (i >= 2 * NSize / 4 && i < 3 * NSize / 4 && AMem[i] != 0) {
        std::cout << " array[" << i << "] is " << AMem[i] << " expected " << 0
                  << std::endl;
        assert(false);
        Failed = true;
      } else if (i >= 3 * NSize / 4 && AMem[i] != i) {
        std::cout << " array[" << i << "] is " << AMem[i] << " expected " << i
                  << std::endl;
        assert(false);
        Failed = true;
      }
    }
  }
  {
    // Test if we can use two sub buffers, pointing to one buffer, from
    // two different OpenCL kernels.
    const size_t NSize = 512;
    cl_int Error = CL_SUCCESS;
    int AMem[NSize];
    for (size_t i = 0; i < NSize; i++) {
      AMem[i] = i;
    }

    try {
      queue TestQueue;

      const char *SrcString[] = {clKernelSourceCodeOverlap.c_str()};
      const size_t SrcStringSize = clKernelSourceCodeOverlap.size();

      cl_context clContext = TestQueue.get_context().get();
      cl_device_id clDevice = TestQueue.get_device().get();

      cl_program clProgram = clCreateProgramWithSource(clContext, 1, SrcString,
                                                       &SrcStringSize, &Error);

      assert(Error == CL_SUCCESS);

      Error = clBuildProgram(clProgram, 1, &clDevice, NULL, NULL, NULL);

      assert(Error == CL_SUCCESS);

      cl_kernel clKernel1 = clCreateKernel(clProgram, "test1", &Error);
      assert(Error == CL_SUCCESS);

      cl_kernel clKernel2 = clCreateKernel(clProgram, "test2", &Error);
      assert(Error == CL_SUCCESS);

      buffer<int, 1> BufA(AMem, cl::sycl::range<1>(NSize));
      buffer<int, 1> BufB(BufA, 0, NSize / 2);
      buffer<int, 1> BufC(BufA, NSize / 4, NSize / 4);

      kernel TestKernel1(clKernel1, TestQueue.get_context());
      kernel TestKernel2(clKernel2, TestQueue.get_context());

      TestQueue.submit([&](handler &cgh) {
        auto a_acc = BufA.get_access<access::mode::read>(cgh);
        auto b_acc = BufB.get_access<access::mode::write>(cgh);
        cgh.set_arg(0, a_acc);
        cgh.set_arg(1, b_acc);

        cgh.parallel_for(range<1>(NSize), TestKernel1);
      });

      TestQueue.submit([&](handler &cgh) {
        auto a_acc = BufA.get_access<access::mode::read>(cgh);
        auto c_acc = BufC.get_access<access::mode::write>(cgh);
        cgh.set_arg(0, a_acc);
        cgh.set_arg(1, c_acc);

        cgh.parallel_for(range<1>(NSize), TestKernel2);
      });

      clReleaseKernel(clKernel1);
      clReleaseKernel(clKernel2);
      clReleaseProgram(clProgram);
    } catch (exception &ex) {
      std::cout << ex.what() << std::endl;
    }

    for (int i = 0; i < NSize; ++i) {
      if (i < NSize / 4 && AMem[i] != 0) {
        std::cout << " array[" << i << "] is " << AMem[i] << " expected " << 0
                  << std::endl;
        assert(false);
        Failed = true;
      } else if (i >= NSize / 4 && i < 2 * NSize / 4 && AMem[i] != 1) {
        std::cout << " array[" << i << "] is " << AMem[i] << " expected " << i
                  << std::endl;
        assert(false);
        Failed = true;
      } else if (i >= 2 * NSize / 4 && AMem[i] != i) {
        std::cout << " array[" << i << "] is " << AMem[i] << " expected " << i
                  << std::endl;
        assert(false);
        Failed = true;
      }
    }
  }

  const char *cl_src = "kernel void test(global int *p) { "
                       "        printf(\"offset on device = \%d\\n\", *p);"
                       "        if (p) *p *= 3;"
                       "}";

  {
    cl::sycl::queue Q;

    // Create OpenCL program
    cl_int err;
    auto context_cl = Q.get_context().get();
    auto device_cl = Q.get_device().get();
    cl_program program_cl =
        clCreateProgramWithSource(context_cl, 1, &cl_src, nullptr, &err);
    err = clBuildProgram(program_cl, 1, &device_cl, nullptr, nullptr, nullptr);
    cl_kernel kernel_cl = clCreateKernel(program_cl, "test", &err);
    cl::sycl::kernel kernel_sycl(kernel_cl, Q.get_context());

    // Create buffer
    constexpr int N = 256;
    std::vector<int> v(2 * N);
    std::iota(v.begin(), v.end(), 0);
    cl::sycl::buffer<int, 1> buf(v.data(), v.size());
    cl::sycl::buffer<int, 1> subbuf(buf, N, N);

    auto subbuf_copy =
        new cl::sycl::buffer<int, 1>(subbuf.reinterpret<int, 1>(N));

    // Test offsets
    {
      auto host_acc = subbuf_copy->get_access<cl::sycl::access::mode::read>();
      std::cout << "On host: offset = " << host_acc[0] << std::endl;
      assert(host_acc[0] == 256 && "Invalid subbuffer origin");
    }

    Q.submit([&](cl::sycl::handler &cgh) {
      auto acc = subbuf_copy->get_access<cl::sycl::access::mode::write>(cgh);
      cgh.set_args(acc);
      cgh.single_task(kernel_sycl);
    });

    Q.wait_and_throw();

    {
      auto host_acc = subbuf_copy->get_access<cl::sycl::access::mode::read>();
      std::cout << "On host: offset = " << host_acc[0] << std::endl;
      assert(host_acc[0] == 256 * 3 && "Invalid subbuffer origin");
    }
  }

  return Failed;
}
