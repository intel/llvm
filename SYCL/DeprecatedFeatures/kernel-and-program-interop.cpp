// REQUIRES: opencl, opencl_icd

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -D__SYCL_INTERNAL_API -o %t.out %opencl_lib
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// No ACC device because Images are not supported there.

//==--- kernel-and-program.cpp - SYCL kernel/program test ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

#include <iostream>
#include <numeric>
#include <string>
#include <utility>

int main() {
  // Single task invocation methods
  {
    sycl::queue q;
    int data = 0;

    // OpenCL interoperability kernel invocation
    if (!q.is_host()) {
      {
        cl_int err;
        sycl::context ctx = q.get_context();
        cl_context clCtx = ctx.get();
        cl_command_queue clQ = q.get();
        cl_mem clBuffer =
            clCreateBuffer(clCtx, CL_MEM_WRITE_ONLY, sizeof(int), NULL, NULL);
        err = clEnqueueWriteBuffer(clQ, clBuffer, CL_TRUE, 0, sizeof(int),
                                   &data, 0, NULL, NULL);
        assert(err == CL_SUCCESS);
        clFinish(clQ);
        sycl::program prog(ctx);
        prog.build_with_source(
            "kernel void SingleTask(global int* a) {*a+=1; }\n");
        q.submit([&](sycl::handler &cgh) {
          cgh.set_args(clBuffer);
          cgh.single_task(prog.get_kernel("SingleTask"));
        });
        q.wait();
        err = clEnqueueReadBuffer(clQ, clBuffer, CL_TRUE, 0, sizeof(int), &data,
                                  0, NULL, NULL);
        clReleaseCommandQueue(clQ);
        clReleaseContext(clCtx);
        assert(err == CL_SUCCESS);
        assert(data == 1);
      }
      {
        sycl::queue sycl_queue;
        sycl::program prog(sycl_queue.get_context());
        prog.build_with_source("kernel void foo(global int* a, global int* b, "
                               "global int* c) {*a=*b+*c; }\n");
        int a = 13, b = 14, c = 15;
        {
          sycl::buffer<int, 1> bufa(&a, sycl::range<1>(1));
          sycl::buffer<int, 1> bufb(&b, sycl::range<1>(1));
          sycl::buffer<int, 1> bufc(&c, sycl::range<1>(1));
          sycl_queue.submit([&](sycl::handler &cgh) {
            auto A = bufa.get_access<sycl::access::mode::write>(cgh);
            auto B = bufb.get_access<sycl::access::mode::read>(cgh);
            auto C = bufc.get_access<sycl::access::mode::read>(cgh);
            cgh.set_args(A, B, C);
            cgh.single_task(prog.get_kernel("foo"));
          });
        }
        assert(a == b + c);
      }
    }
    {
      sycl::queue Queue;
      if (!Queue.is_host()) {
        sycl::sampler first(sycl::coordinate_normalization_mode::normalized,
                            sycl::addressing_mode::clamp,
                            sycl::filtering_mode::linear);
        sycl::sampler second(sycl::coordinate_normalization_mode::unnormalized,
                             sycl::addressing_mode::clamp_to_edge,
                             sycl::filtering_mode::nearest);
        sycl::program prog(Queue.get_context());
        prog.build_with_source(
            "kernel void sampler_args(int a, sampler_t first, "
            "int b, sampler_t second, int c) {}\n");
        sycl::kernel krn = prog.get_kernel("sampler_args");

        Queue.submit([&](sycl::handler &cgh) {
          cgh.set_args(0, first, 2, second, 3);
          cgh.single_task(krn);
        });
      }
    }
  }
  // Parallel for with range
  {
    sycl::queue q;
    std::vector<int> dataVec(10);
    std::iota(dataVec.begin(), dataVec.end(), 0);

    if (!q.is_host()) {
      cl_int err;
      {
        sycl::context ctx = q.get_context();
        cl_context clCtx = ctx.get();
        cl_command_queue clQ = q.get();
        cl_mem clBuffer = clCreateBuffer(
            clCtx, CL_MEM_WRITE_ONLY, sizeof(int) * dataVec.size(), NULL, NULL);
        err = clEnqueueWriteBuffer(clQ, clBuffer, CL_TRUE, 0,
                                   sizeof(int) * dataVec.size(), dataVec.data(),
                                   0, NULL, NULL);
        assert(err == CL_SUCCESS);

        sycl::program prog(ctx);
        prog.build_with_source(
            "kernel void ParallelFor(__global int* a, int v, __local int *l) "
            "{ size_t index = get_global_id(0); l[index] = a[index];"
            " l[index] += v; a[index] = l[index]; }\n");

        q.submit([&](sycl::handler &cgh) {
          const int value = 1;
          auto local_acc =
              sycl::accessor<int, 1, sycl::access::mode::read_write,
                             sycl::access::target::local>(sycl::range<1>(10),
                                                          cgh);
          cgh.set_args(clBuffer, value, local_acc);
          cgh.parallel_for(sycl::range<1>(10), prog.get_kernel("ParallelFor"));
        });

        q.wait();
        err = clEnqueueReadBuffer(clQ, clBuffer, CL_TRUE, 0,
                                  sizeof(int) * dataVec.size(), dataVec.data(),
                                  0, NULL, NULL);
        clReleaseCommandQueue(clQ);
        clReleaseContext(clCtx);
        assert(err == CL_SUCCESS);
        for (size_t i = 0; i < dataVec.size(); ++i) {
          assert(dataVec[i] == i + 1);
        }
      }
    }
  }

  // Parallel for with nd_range
  {
    sycl::queue q;
    std::vector<int> dataVec(10);
    std::iota(dataVec.begin(), dataVec.end(), 0);

    if (!q.is_host()) {
      cl_int err;
      {
        sycl::context ctx = q.get_context();
        cl_context clCtx = ctx.get();
        cl_command_queue clQ = q.get();
        cl_mem clBuffer = clCreateBuffer(
            clCtx, CL_MEM_WRITE_ONLY, sizeof(int) * dataVec.size(), NULL, NULL);
        err = clEnqueueWriteBuffer(clQ, clBuffer, CL_TRUE, 0,
                                   sizeof(int) * dataVec.size(), dataVec.data(),
                                   0, NULL, NULL);
        assert(err == CL_SUCCESS);

        sycl::program prog(ctx);
        prog.build_with_source(
            "kernel void ParallelForND( local int* l,global int* a)"
            "{  size_t idx = get_global_id(0);"
            "  int pos = idx & 1;"
            "  int opp = pos ^ 1;"
            "  l[pos] = a[get_global_id(0)];"
            "  barrier(CLK_LOCAL_MEM_FENCE);"
            "  a[idx]=l[opp]; }");

        // TODO is there no way to set local memory size via interoperability?
        sycl::kernel krn = prog.get_kernel("ParallelForND");
        clSetKernelArg(krn.get(), 0, sizeof(int) * 2, NULL);

        q.submit([&](sycl::handler &cgh) {
          cgh.set_arg(1, clBuffer);
          cgh.parallel_for(
              sycl::nd_range<1>(sycl::range<1>(10), sycl::range<1>(2)), krn);
        });

        q.wait();
        err = clEnqueueReadBuffer(clQ, clBuffer, CL_TRUE, 0,
                                  sizeof(int) * dataVec.size(), dataVec.data(),
                                  0, NULL, NULL);
        clReleaseCommandQueue(clQ);
        clReleaseContext(clCtx);
        assert(err == CL_SUCCESS);
      }
      for (size_t i = 0; i < dataVec.size(); ++i) {
        assert(dataVec[i] == (i ^ 1));
      }
    }
  }
}
