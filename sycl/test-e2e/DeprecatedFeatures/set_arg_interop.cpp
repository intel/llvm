// REQUIRES: opencl, opencl_icd

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -D__SYCL_INTERNAL_API -o %t.out %opencl_lib -O3
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <sycl/sycl.hpp>

#include <cassert>

using namespace sycl;

int main() {
  queue Queue;
  context Context = Queue.get_context();

  cl_context ClContext = Context.get();

  const size_t CountSources = 3;
  const char *Sources[CountSources] = {
      "kernel void foo1(global float* Array, global int* Value) { *Array = "
      "42; *Value = 1; }\n",
      "kernel void foo2(global float* Array) { int id = get_global_id(0); "
      "Array[id] = id; }\n",
      "kernel void foo3(global float* Array, local float* LocalArray) { "
      "(void)LocalArray; (void)Array; }\n",
  };

  cl_int Err;
  cl_program ClProgram = clCreateProgramWithSource(ClContext, CountSources,
                                                   Sources, nullptr, &Err);
  assert(Err == CL_SUCCESS);

  Err = clBuildProgram(ClProgram, 0, nullptr, nullptr, nullptr, nullptr);
  assert(Err == CL_SUCCESS);

  cl_kernel FirstCLKernel = clCreateKernel(ClProgram, "foo1", &Err);
  assert(Err == CL_SUCCESS);

  cl_kernel SecondCLKernel = clCreateKernel(ClProgram, "foo2", &Err);
  assert(Err == CL_SUCCESS);

  cl_kernel ThirdCLKernel = clCreateKernel(ClProgram, "foo3", &Err);
  assert(Err == CL_SUCCESS);

  const size_t Count = 100;
  float Array[Count];

  kernel FirstKernel(FirstCLKernel, Context);
  kernel SecondKernel(SecondCLKernel, Context);
  kernel ThirdKernel(ThirdCLKernel, Context);
  int Value;
  {
    buffer<float, 1> FirstBuffer(Array, range<1>(1));
    buffer<int, 1> SecondBuffer(&Value, range<1>(1));
    Queue.submit([&](handler &CGH) {
      CGH.set_arg(0, FirstBuffer.get_access<access::mode::write>(CGH));
      CGH.set_arg(1, SecondBuffer.get_access<access::mode::write>(CGH));
      CGH.single_task(FirstKernel);
    });
  }
  Queue.wait_and_throw();

  assert(Array[0] == 42);
  assert(Value == 1);

  {
    buffer<float, 1> FirstBuffer(Array, range<1>(Count));
    Queue.submit([&](handler &CGH) {
      auto Acc = FirstBuffer.get_access<access::mode::read_write>(CGH);
      CGH.set_arg(0, FirstBuffer.get_access<access::mode::read_write>(CGH));
      CGH.parallel_for(range<1>{Count}, SecondKernel);
    });
  }
  Queue.wait_and_throw();

  for (size_t I = 0; I < Count; ++I) {
    assert(Array[I] == I);
  }

  {
    auto dev = Queue.get_device();
    auto ctxt = Queue.get_context();
    if (dev.get_info<info::device::usm_shared_allocations>()) {
      float *data =
          static_cast<float *>(malloc_shared(Count * sizeof(float), dev, ctxt));

      Queue.submit([&](handler &CGH) {
        CGH.set_arg(0, data);
        CGH.parallel_for(range<1>{Count}, SecondKernel);
      });
      Queue.wait_and_throw();

      for (size_t I = 0; I < Count; ++I) {
        assert(data[I] == I);
      }
      free(data, ctxt);
    }
  }

  {
    buffer<float, 1> FirstBuffer(Array, range<1>(Count));
    Queue.submit([&](handler &CGH) {
      auto Acc = FirstBuffer.get_access<access::mode::read_write>(CGH);
      CGH.set_arg(0, FirstBuffer.get_access<access::mode::read_write>(CGH));
      CGH.set_arg(1, sycl::accessor<float, 1, sycl::access::mode::read_write,
                                    sycl::access::target::local>(
                         sycl::range<1>(Count), CGH));
      CGH.parallel_for(range<1>{Count}, ThirdKernel);
    });
  }
  Queue.wait_and_throw();

  clReleaseContext(ClContext);
  clReleaseKernel(FirstCLKernel);
  clReleaseKernel(SecondCLKernel);
  clReleaseKernel(ThirdCLKernel);
  clReleaseProgram(ClProgram);
  return 0;
}
