// REQUIRES: opencl, opencl_icd

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -D__SYCL_INTERNAL_API -o %t.out %opencl_lib
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

//==---------------- buffer_interop.cpp - SYCL buffer basic test -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <sycl/sycl.hpp>

#include <cassert>
#include <iostream>
#include <memory>

using namespace sycl;

int main() {
  bool Failed = false;
  {
    sycl::context Ctx;
    cl_context OCLCtx = Ctx.get();

    cl_int Error = CL_SUCCESS;
    cl_mem OCLBuf =
        clCreateBuffer(OCLCtx, CL_MEM_READ_WRITE, sizeof(int), nullptr, &Error);
    assert(Error == CL_SUCCESS);
    Error = clReleaseContext(OCLCtx);
    assert(Error == CL_SUCCESS);

    sycl::buffer<int, 1> Buf{OCLBuf, Ctx};

    sycl::queue Q;

    if (Ctx == Q.get_context()) {
      std::cerr << "Expected different contexts" << std::endl;
      Failed = true;
    }

    Q.submit([&](sycl::handler &CGH) {
      auto Acc = Buf.get_access<access::mode::write>(CGH);
      CGH.single_task<class BufferInterop_DifferentContext>(
          [=]() { Acc[0] = 42; });
    });

    auto Acc = Buf.get_access<sycl::access::mode::read>();
    if (Acc[0] != 42) {
      std::cerr << "Result is incorrect" << std::endl;
      Failed = true;
    }
  }
  {
    constexpr size_t Size = 32;
    int Init[Size] = {5};
    cl_int Error = CL_SUCCESS;
    sycl::range<1> InteropRange{Size};
    size_t InteropSize = Size * sizeof(int);

    queue MyQueue;

    cl_context OCLCtx = MyQueue.get_context().get();

    cl_mem OpenCLBuffer =
        clCreateBuffer(OCLCtx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                       Size * sizeof(int), Init, &Error);
    assert(Error == CL_SUCCESS);
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
      buffer<int, 1> BufferData{
          Data, range<1>{Size}, {property::buffer::use_host_ptr()}};
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
    assert(Error == CL_SUCCESS);

    for (size_t i = 0; i < Size; ++i) {
      if (Result[i] != 20) {
        std::cout << " array[" << i << "] is " << Result[i] << " expected "
                  << 20 << std::endl;
        assert(false);
        Failed = true;
      }
    }
    Error = clReleaseContext(OCLCtx);
    assert(Error == CL_SUCCESS);
  }
  // Check set_final_data
  {
    constexpr size_t Size = 32;
    int Init[Size] = {5};
    int Result[Size] = {5};
    cl_int Error = CL_SUCCESS;

    queue MyQueue;
    cl_context OCLCtx = MyQueue.get_context().get();

    cl_mem OpenCLBuffer =
        clCreateBuffer(OCLCtx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                       Size * sizeof(int), Init, &Error);
    assert(Error == CL_SUCCESS);
    {
      buffer<int, 1> Buffer{OpenCLBuffer, MyQueue.get_context()};
      Buffer.set_final_data(Result);

      MyQueue.submit([&](handler &CGH) {
        auto B = Buffer.get_access<access::mode::write>(CGH);
        CGH.parallel_for<class FinalData>(range<1>{Size},
                                          [=](id<1> Index) { B[Index] = 10; });
      });
    }
    Error = clReleaseMemObject(OpenCLBuffer);
    assert(Error == CL_SUCCESS);
    for (size_t i = 0; i < Size; ++i) {
      if (Result[i] != 10) {
        std::cout << " array[" << i << "] is " << Result[i] << " expected "
                  << 10 << std::endl;
        assert(false);
        Failed = true;
      }
    }
    Error = clReleaseContext(OCLCtx);
    assert(Error == CL_SUCCESS);
  }
  // Check host accessor
  {
    constexpr size_t Size = 32;
    int Init[Size] = {5};
    cl_int Error = CL_SUCCESS;

    queue MyQueue;
    cl_context OCLCtx = MyQueue.get_context().get();

    cl_mem OpenCLBuffer =
        clCreateBuffer(OCLCtx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                       Size * sizeof(int), Init, &Error);
    assert(Error == CL_SUCCESS);
    buffer<int, 1> Buffer{OpenCLBuffer, MyQueue.get_context()};

    MyQueue.submit([&](handler &CGH) {
      auto B = Buffer.get_access<access::mode::write>(CGH);
      CGH.parallel_for<class HostAccess>(range<1>{Size},
                                         [=](id<1> Index) { B[Index] = 10; });
    });
    auto Acc = Buffer.get_access<sycl::access::mode::read>();
    for (size_t i = 0; i < Size; ++i) {
      if (Acc[i] != 10) {
        std::cout << " array[" << i << "] is " << Acc[i] << " expected " << 10
                  << std::endl;
        assert(false);
        Failed = true;
      }
    }
    Error = clReleaseMemObject(OpenCLBuffer);
    assert(Error == CL_SUCCESS);
    Error = clReleaseContext(OCLCtx);
    assert(Error == CL_SUCCESS);
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
    assert(Error == CL_SUCCESS);
    cl_event OpenCLEvent = clCreateUserEvent(OpenCLContext, &Error);
    assert(Error == CL_SUCCESS);
    assert(clSetUserEventStatus(OpenCLEvent, CL_COMPLETE) == CL_SUCCESS);

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

    assert(clReleaseMemObject(OpenCLBuffer) == CL_SUCCESS);
    assert(clReleaseContext(OpenCLContext) == CL_SUCCESS);
    assert(clReleaseEvent(OpenCLEvent) == CL_SUCCESS);
  }

  {
    queue Queue;
    if (!Queue.is_host()) {
      std::vector<int> Data1(10, -1);
      std::vector<int> Data2(10, -2);
      {
        buffer<int, 1> BufferA(Data1.data(), range<1>(10));
        buffer<int, 1> BufferB(Data2);

        program Program(Queue.get_context());
        Program.build_with_source(
            "kernel void override_source(global int* Acc) "
            "{Acc[get_global_id(0)] = 0; }\n");
        sycl::kernel Kernel = Program.get_kernel("override_source");
        Queue.submit([&](handler &CGH) {
          auto AccA = BufferA.get_access<access::mode::read_write>(CGH);
          CGH.set_arg(0, AccA);
          auto AccB = BufferB.get_access<access::mode::read_write>(CGH);
          CGH.parallel_for(sycl::range<1>(10), Kernel);
        });
      } // Data is copied back
      for (int i = 0; i < 10; i++) {
        if (Data2[i] != -2) {
          std::cout << " Data2[" << i << "] is " << Data2[i] << " expected "
                    << -2 << std::endl;
          assert(false);
          Failed = true;
        }
      }
      for (int i = 0; i < 10; i++) {
        if (Data1[i] != 0) {
          std::cout << " Data1[" << i << "] is " << Data1[i] << " expected "
                    << 0 << std::endl;
          assert(false);
          Failed = true;
        }
      }
    }
  }

  {
    queue Queue;
    if (!Queue.is_host()) {
      std::vector<int> Data1(10, -1);
      std::vector<int> Data2(10, -2);
      {
        buffer<int, 1> BufferA(Data1.data(), range<1>(10));
        buffer<int, 1> BufferB(Data2);
        accessor<int, 1, access::mode::read_write, access::target::device,
                 access::placeholder::true_t>
            AccA(BufferA);
        accessor<int, 1, access::mode::read_write, access::target::device,
                 access::placeholder::true_t>
            AccB(BufferB);

        program Program(Queue.get_context());
        Program.build_with_source(
            "kernel void override_source_placeholder(global "
            "int* Acc) {Acc[get_global_id(0)] = 0; }\n");
        sycl::kernel Kernel = Program.get_kernel("override_source_placeholder");

        Queue.submit([&](handler &CGH) {
          CGH.require(AccA);
          CGH.set_arg(0, AccA);
          CGH.require(AccB);
          CGH.parallel_for(sycl::range<1>(10), Kernel);
        });
      } // Data is copied back
      for (int i = 0; i < 10; i++) {
        if (Data2[i] != -2) {
          std::cout << " Data2[" << i << "] is " << Data2[i] << " expected "
                    << -2 << std::endl;
          assert(false);
          Failed = true;
        }
      }
      for (int i = 0; i < 10; i++) {
        if (Data1[i] != 0) {
          std::cout << " Data1[" << i << "] is " << Data1[i] << " expected "
                    << 0 << std::endl;
          assert(false);
          Failed = true;
        }
      }
    }
  }

  return Failed;
}
