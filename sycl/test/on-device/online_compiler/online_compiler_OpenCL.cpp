// REQUIRES: opencl

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -lOpenCL -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/online_compiler.hpp>

#include <vector>

using byte = unsigned char;

sycl::kernel getSYCLKernelWithIL(sycl::context &Context,
                                 const std::vector<byte> &IL) {
  cl_int Err;
  cl_program ClProgram =
      clCreateProgramWithIL(Context.get(), IL.data(), IL.size(), &Err);
  if (Err != CL_SUCCESS)
    throw sycl::compile_program_error();

  Err = clBuildProgram(ClProgram, 0, nullptr, nullptr, nullptr, nullptr);
  if (Err != CL_SUCCESS)
    throw sycl::runtime_error();

  cl_kernel ClKernel = clCreateKernel(ClProgram, "my_kernel", &Err);
  if (Err != CL_SUCCESS)
    throw sycl::runtime_error();

  return sycl::kernel(ClKernel, Context);
}

#include "online_compiler_common.hpp"
