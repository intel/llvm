// REQUIRES: opencl, opencl_icd

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -D__SYCL_INTERNAL_API %s -o %t.out %opencl_lib
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

//==--------------- kernel_interop.cpp - SYCL kernel ocl interop test ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <sycl/sycl.hpp>

#include <cassert>

using namespace sycl;

// This test checks that SYCL kernel interoperabitily constructor is implemented
// in accordance with SYCL spec:
// - It throws an exception when passed SYCL context doesn't represent the same
//   underlying OpenCL context associated with passed cl_kernel
// - It retains passed cl_kernel so releasing kernel won't produce errors.

int main() {
  queue Queue;
  if (Queue.is_host())
    return 0;

  context Context = Queue.get_context();

  cl_context ClContext = Context.get();

  const size_t CountSources = 1;
  const char *Sources[CountSources] = {
      "kernel void foo1(global float* Array, global int* Value) { *Array = "
      "42; *Value = 1; }\n",
  };

  cl_int Err;
  cl_program ClProgram = clCreateProgramWithSource(ClContext, CountSources,
                                                   Sources, nullptr, &Err);
  assert(Err == CL_SUCCESS);

  Err = clBuildProgram(ClProgram, 0, nullptr, nullptr, nullptr, nullptr);
  assert(Err == CL_SUCCESS);

  cl_kernel ClKernel = clCreateKernel(ClProgram, "foo1", &Err);
  assert(Err == CL_SUCCESS);

  // Try to create kernel with another context
  bool Pass = false;
  context OtherContext{Context.get_devices()[0]};
  try {
    kernel Kernel(ClKernel, OtherContext);
  } catch (sycl::invalid_parameter_error e) {
    Pass = true;
  }
  assert(Pass);

  kernel Kernel(ClKernel, Context);

  assert(clReleaseKernel(ClKernel) == CL_SUCCESS);
  assert(clReleaseContext(ClContext) == CL_SUCCESS);
  assert(clReleaseProgram(ClProgram) == CL_SUCCESS);

  return 0;
}
