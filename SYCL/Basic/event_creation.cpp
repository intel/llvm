// REQUIRES: opencl, opencl_icd

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out %opencl_lib
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

//==--------------- event.cpp - SYCL event test ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <CL/sycl.hpp>
#include <iostream>

int main() {
  try {
    std::cout << "Create default event" << std::endl;
    cl::sycl::event e;
  } catch (cl::sycl::device_error e) {
    std::cout << "Failed to create device for event" << std::endl;
  }
  try {
    std::cout << "Try create OpenCL event" << std::endl;
    cl::sycl::context c;
    if (!c.is_host()) {
      ::cl_int error;
      cl_event u_e = clCreateUserEvent(c.get(), &error);
      cl::sycl::event cl_e(u_e, c);
      std::cout << "OpenCL event: " << std::hex << cl_e.get()
                << ((cl_e.get() == u_e) ? " matches " : " does not match ")
                << u_e << std::endl;

    } else {
      std::cout << "Failed to create OpenCL context" << std::endl;
    }
  } catch (cl::sycl::device_error e) {
    std::cout << "Failed to create device for context" << std::endl;
  }
}
