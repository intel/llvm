// REQUIRES: opencl, opencl_icd
// RUN: %{build} -o %t.out %opencl_lib
// RUN: %{run} %t.out

#include <CL/opencl.h>
#include <sycl/backend.hpp>
#include <sycl/backend/opencl.hpp>
#include <sycl/detail/core.hpp>

using namespace sycl;

int main() {
  queue Queue{};

  const char KernelCode[] = "__kernel void foo() { }\n";
  const size_t KernelCodeSize = sizeof(KernelCode);
  const char *CLCode[1] = {KernelCode};

  auto Context = Queue.get_info<info::queue::context>();
  auto Device = Queue.get_info<info::queue::device>();
  cl_context CLContext = get_native<backend::opencl>(Context);
  cl_device_id CLDevice = get_native<backend::opencl>(Device);

  cl_int Err;

  cl_program CLProgram =
      clCreateProgramWithSource(CLContext, 1, CLCode, &KernelCodeSize, &Err);
  assert(Err == CL_SUCCESS);
  Err = clBuildProgram(CLProgram, 1, &CLDevice, "", nullptr, nullptr);
  assert(Err == CL_SUCCESS);

  cl_kernel CLKernel = clCreateKernel(CLProgram, "foo", &Err);
  assert(Err == CL_SUCCESS);
  kernel SYCLKernel = sycl::make_kernel<backend::opencl>(CLKernel, Context);

  Queue.submit(
      [&](handler &commandgroup) { commandgroup.single_task(SYCLKernel); });
  return 0;
}
