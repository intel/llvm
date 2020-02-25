// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out -L %opencl_libs_dir -lOpenCL
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// REQUIRES: opencl

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

int main() {
  bool Failed = false;
  {
    constexpr size_t Size = 32;
    int Init[Size] = {5};
    cl_int Error = CL_SUCCESS;
    cl::sycl::range<1> InteropRange{Size};
    size_t InteropSize = Size * sizeof(int);

    queue MyQueue;

    cl_mem OpenCLBuffer = clCreateBuffer(
        MyQueue.get_context().get(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        Size * sizeof(int), Init, &Error);
    CHECK_OCL_CODE(Error);
    buffer<int, 1> Buffer{OpenCLBuffer, MyQueue.get_context()};

    if (Buffer.get_range() != InteropRange) {
          assert(false);
          Failed = true;
    }
    if (Buffer.get_size() != InteropSize) {
          assert(false);
          Failed = true;
    }
    if (Buffer.get_count() != Size) {
          assert(false);
          Failed = true;
    }

    MyQueue.submit([&](handler &CGH) {
      auto B = Buffer.get_access<access::mode::write>(CGH);
      CGH.parallel_for<class BufferInterop>(
          range<1>{Size}, [=](id<1> Index) { B[Index] = 10; });
    });

    int Data[Size] = {10};
    std::vector<int> Result(Size, 0);
    {
      buffer<int, 1> BufferData{Data, range<1>{Size},
                                {property::buffer::use_host_ptr()}};
      BufferData.set_final_data(Result.begin());
      MyQueue.submit([&](handler &CGH) {
        auto Data = BufferData.get_access<access::mode::write>(CGH);
        auto CLData = Buffer.get_access<access::mode::read>(CGH);
        CGH.parallel_for<class UseMemContent>(range<1>{Size}, [=](id<1> Index) {
          Data[Index] = 2 * CLData[Index];
        });
      });
    }

    Error = clReleaseMemObject(OpenCLBuffer);
    CHECK_OCL_CODE(Error);

    for (size_t i = 0; i < Size; ++i) {
      if (Result[i] != 20) {
        std::cout << " array[" << i << "] is " << Result[i] << " expected "
                  << 20 << std::endl;
        assert(false);
        Failed = true;
      }
    }
  }
  // Check set_final_data
  {
    constexpr size_t Size = 32;
    int Init[Size] = {5};
    int Result[Size] = {5};
    cl_int Error = CL_SUCCESS;

    queue MyQueue;

    cl_mem OpenCLBuffer = clCreateBuffer(
        MyQueue.get_context().get(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        Size * sizeof(int), Init, &Error);
    CHECK_OCL_CODE(Error);
    {
      buffer<int, 1> Buffer{OpenCLBuffer, MyQueue.get_context()};
      Buffer.set_final_data(Result);

      MyQueue.submit([&](handler &CGH) {
        auto B = Buffer.get_access<access::mode::write>(CGH);
        CGH.parallel_for<class FinalData>(
            range<1>{Size}, [=](id<1> Index) { B[Index] = 10; });
      });
    }
    Error = clReleaseMemObject(OpenCLBuffer);
    CHECK_OCL_CODE(Error);
    for (size_t i = 0; i < Size; ++i) {
      if (Result[i] != 10) {
        std::cout << " array[" << i << "] is " << Result[i] << " expected "
                  << 10 << std::endl;
        assert(false);
        Failed = true;
      }
    }
  }
  // Check host accessor
  {
    constexpr size_t Size = 32;
    int Init[Size] = {5};
    cl_int Error = CL_SUCCESS;

    queue MyQueue;

    cl_mem OpenCLBuffer = clCreateBuffer(
        MyQueue.get_context().get(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        Size * sizeof(int), Init, &Error);
    CHECK_OCL_CODE(Error);
    buffer<int, 1> Buffer{OpenCLBuffer, MyQueue.get_context()};

    MyQueue.submit([&](handler &CGH) {
      auto B = Buffer.get_access<access::mode::write>(CGH);
      CGH.parallel_for<class HostAccess>(range<1>{Size},
                                        [=](id<1> Index) { B[Index] = 10; });
    });
    auto Acc = Buffer.get_access<cl::sycl::access::mode::read>();
    for (size_t i = 0; i < Size; ++i) {
      if (Acc[i] != 10) {
        std::cout << " array[" << i << "] is " << Acc[i] << " expected "
                  << 10 << std::endl;
        assert(false);
        Failed = true;
      }
    }
    Error = clReleaseMemObject(OpenCLBuffer);
    CHECK_OCL_CODE(Error);
  }
  // Check interop constructor event
  {
    // Checks that the cl_event is not deleted on memory object destruction
    queue MyQueue;
    cl_context OpenCLContext = MyQueue.get_context().get();

    int Val;
    cl_int Error = CL_SUCCESS;
    cl_mem OpenCLBuffer =
        clCreateBuffer(OpenCLContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                       sizeof(int), &Val, &Error);
    CHECK_OCL_CODE(Error);
    cl_event OpenCLEvent = clCreateUserEvent(OpenCLContext, &Error);
    CHECK_OCL_CODE(Error);
    CHECK_OCL_CODE(clSetUserEventStatus(OpenCLEvent, CL_COMPLETE));

    {
      event Event(OpenCLEvent, OpenCLContext);
      buffer<int, 1> Buffer{OpenCLBuffer, MyQueue.get_context(), Event};

      MyQueue.submit([&](handler &Cgh) {
        auto Acc = Buffer.get_access<access::mode::write>(Cgh);
        Cgh.single_task<class TestEvent>([=]() { Acc[0] = 42; });
      });

      auto Acc = Buffer.get_access<access::mode::read>();
      if (42 != Acc[0]) {
        assert(false);
        Failed = true;
      }
    }

    CHECK_OCL_CODE(clReleaseMemObject(OpenCLBuffer));
    CHECK_OCL_CODE(clReleaseContext(OpenCLContext));
    CHECK_OCL_CODE(clReleaseEvent(OpenCLEvent));
  }

  return Failed;
}
