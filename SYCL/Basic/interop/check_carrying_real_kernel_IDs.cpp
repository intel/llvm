// UNSUPPORTED: windows || linux
//   temporarily disabled

// REQUIRES: opencl, opencl_icd
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out %opencl_lib
// RUN: %ACC_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include <CL/opencl.h>
#include <sycl/backend/opencl.hpp>
#include <sycl/sycl.hpp>

using namespace cl::sycl;

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
