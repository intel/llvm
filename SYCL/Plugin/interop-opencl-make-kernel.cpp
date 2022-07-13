// REQUIRES: opencl, opencl_icd

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out %opencl_lib
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// Test for OpenCL make_kernel

#include <CL/opencl.h>
#include <sycl/backend/opencl.hpp>
#include <sycl/sycl.hpp>

using namespace cl::sycl;

class DummyKernel;

constexpr size_t N = 1024;

const char OpenCLIotaKernelCode[] = "__kernel void iota(__global int *a) {\
        int i = get_global_id(0);\
        a[i] = i;\
    }";
const size_t OpenCLIotaKernelCodeSize = sizeof(OpenCLIotaKernelCode);
const char *OpenCLCode[1] = {OpenCLIotaKernelCode};

int main() {
  queue Queue{};
  auto Context = Queue.get_info<info::queue::context>();
  auto Device = Queue.get_info<info::queue::device>();

  cl_context OpenCLContext = get_native<backend::opencl>(Context);
  cl_device_id OpenCLDevice = get_native<backend::opencl>(Device);

  cl_int OpenCLError;
  cl_program OpenCLProgram = clCreateProgramWithSource(
      OpenCLContext, 1, OpenCLCode, &OpenCLIotaKernelCodeSize, &OpenCLError);
  assert(OpenCLError == CL_SUCCESS);

  OpenCLError = clBuildProgram(OpenCLProgram, 1, &OpenCLDevice, nullptr,
                               nullptr, nullptr);
  assert(OpenCLError == CL_SUCCESS);

  cl_kernel OpenCLKernel = clCreateKernel(OpenCLProgram, "iota", &OpenCLError);
  assert(OpenCLError == CL_SUCCESS);

  kernel Kernel = make_kernel<backend::opencl>(OpenCLKernel, Context);

  // The associated kernel bundle should not contain the dummy-kernel.
  assert(!Kernel.get_kernel_bundle().has_kernel(get_kernel_id<DummyKernel>()));

  int *IotaResult = malloc_shared<int>(N, Device, Context);
  Queue
      .submit([&](sycl::handler &CGH) {
        CGH.set_arg(0, IotaResult);
        CGH.parallel_for(N, Kernel);
      })
      .wait();

  for (int i = 0; i < N; ++i)
    assert(IotaResult[i] == i);

  free(IotaResult, Context);

  Queue
      .submit(
          [&](sycl::handler &CGH) { CGH.single_task<DummyKernel>([=]() {}); })
      .wait();

  return 0;
}
