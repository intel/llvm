// RUN: %clangxx -fsycl -sycl-std=2020 -Xclang -verify -Xclang -verify-ignore-unexpected=note %s -o %t.out
// RUN: %clangxx -fsycl -Xclang -verify -Xclang -verify-ignore-unexpected=note %s -o %t.out
// RUN: %clangxx -fsycl -sycl-std=2017 -Werror %s -o %t.out
// RUN: %clangxx -fsycl -sycl-std=1.2.1 -Werror %s -o %t.out

#include <CL/sycl.hpp>

int main() {
  cl_context ClCtx;
  // expected-warning@+1 {{'context' is deprecated: OpenCL interop APIs are deprecated}}
  sycl::context Ctx{ClCtx};
  // expected-warning@+1 {{'get' is deprecated: OpenCL interop APIs are deprecated}}
  (void)Ctx.get();

  cl_mem Mem;
  // expected-warning@+1 {{'buffer' is deprecated: OpenCL interop APIs are deprecated}}
  sycl::buffer<int, 1> Buf{Mem, Ctx};
  (void)Buf;

  cl_device_id DevId;
  // expected-warning@+1 {{'device' is deprecated: OpenCL interop APIs are deprecated}}
  sycl::device Device{DevId};
  // expected-warning@+1 {{'get' is deprecated: OpenCL interop APIs are deprecated}}
  (void)Device.get();

  cl_event ClEvent;
  // expected-warning@+1 {{'event' is deprecated: OpenCL interop APIs are deprecated}}
  sycl::event Evt{ClEvent, Ctx};
  // expected-warning@+1 {{'get' is deprecated: OpenCL interop APIs are deprecated}}
  (void)Evt.get();

  // expected-warning@+1 {{'image' is deprecated: OpenCL interop APIs are deprecated}}
  sycl::image<1> Img{Mem, Ctx};
  (void)Img;

  cl_platform_id ClPlatform;
  // expected-warning@+1 {{'platform' is deprecated: OpenCL interop APIs are deprecated}}
  sycl::platform Platform{ClPlatform};
  // expected-warning@+1 {{'get' is deprecated: OpenCL interop APIs are deprecated}}
  (void)Platform.get();

  cl_command_queue ClQueue;
  // expected-warning@+1 {{'queue' is deprecated: OpenCL interop APIs are deprecated}}
  sycl::queue Queue{ClQueue, Ctx};
  // expected-warning@+1 {{'get' is deprecated: OpenCL interop APIs are deprecated}}
  (void)Queue.get();

  cl_sampler ClSampler;
  // expected-warning@+1 {{'sampler' is deprecated: OpenCL interop APIs are deprecated}}
  sycl::sampler Sampler{ClSampler, Ctx};
  (void)Sampler;

  cl_kernel ClKernel;
  // expected-warning@+1 {{'kernel' is deprecated: OpenCL interop constructors are deprecated, use make_kernel() instead}}
  sycl::kernel Kernel{ClKernel, Ctx};
  // expected-warning@+1 {{'get' is deprecated: OpenCL interop get() functions are deprecated, use get_native() instead}}
  (void)Kernel.get();

  // expected-warning@+1 {{'program' is deprecated: program class is deprecated, use kernel_bundle instead}}
  sycl::program Prog{Ctx};

  return 0;
}
