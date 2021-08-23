// RUN: %clangxx %fsycl-host-only -fsyntax-only -sycl-std=2020 -Xclang -verify -Xclang -verify-ignore-unexpected=note %s -o %t.out
// RUN: %clangxx %fsycl-host-only -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s -o %t.out
// RUN: %clangxx %fsycl-host-only -fsyntax-only -sycl-std=2017 -Xclang -verify -Xclang -verify-ignore-unexpected=note %s -o %t.out
// RUN: %clangxx %fsycl-host-only -fsyntax-only -sycl-std=1.2.1 -Xclang -verify -Xclang -verify-ignore-unexpected=note %s -o %t.out

#include <CL/sycl.hpp>
#include <CL/sycl/ONEAPI/atomic_fence.hpp>
#include <CL/sycl/INTEL/online_compiler.hpp>

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
  // expected-warning@+1 {{'has_extension' is deprecated: use device::has() function with aspects APIs instead}}
  (void)Device.has_extension("abc");

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
  // expected-warning@+1 {{'has_extension' is deprecated: use platform::has() function with aspects APIs instead}}
  (void)Platform.has_extension("abc");

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

  sycl::buffer<int, 1> Buffer(4);
  // expected-warning@+1{{'get_count' is deprecated: get_count() is deprecated, please use size() instead}}
  size_t BufferGetCount = Buffer.get_count();
  size_t BufferSize = Buffer.size();
  // expected-warning@+1 {{'get_size' is deprecated: get_size() is deprecated, please use byte_size() instead}}
  size_t BufferGetSize = Buffer.get_size();

  sycl::vec<int, 2> Vec(1, 2);
  // expected-warning@+1{{'get_count' is deprecated: get_count() is deprecated, please use size() instead}}
  size_t VecGetCount = Vec.get_count();
  // expected-warning@+1 {{'get_size' is deprecated: get_size() is deprecated, please use byte_size() instead}}
  size_t VecGetSize = Vec.get_size();

  // expected-warning@+1 {{'runtime_error' is deprecated: use sycl::exception with sycl::errc::runtime instead.}}
  sycl::runtime_error re;
  // expected-warning@+1 {{'kernel_error' is deprecated: use sycl::exception with sycl::errc::kernel or errc::kernel_argument instead.}}
  sycl::kernel_error ke;
  // expected-warning@+1 {{'accessor_error' is deprecated: use sycl::exception with sycl::errc::accessor instead.}}
  sycl::accessor_error ae;
  // expected-warning@+1 {{'nd_range_error' is deprecated: use sycl::exception with sycl::errc::nd_range instead.}}
  sycl::nd_range_error ne;
  // expected-warning@+1 {{'event_error' is deprecated: use sycl::exception with sycl::errc::event instead.}}
  sycl::event_error ee;
  // expected-warning@+1 {{'invalid_parameter_error' is deprecated: use sycl::exception with a sycl::errc enum value instead.}}
  sycl::invalid_parameter_error ipe;
  // expected-warning@+1 {{'device_error' is deprecated: use sycl::exception with a sycl::errc enum value instead.}}
  sycl::device_error de;
  // expected-warning@+1 {{'compile_program_error' is deprecated: use sycl::exception with a sycl::errc enum value instead.}}
  sycl::compile_program_error cpe;
  // expected-warning@+1 {{'link_program_error' is deprecated: use sycl::exception with a sycl::errc enum value instead.}}
  sycl::link_program_error lpe;
  // expected-warning@+1 {{'invalid_object_error' is deprecated: use sycl::exception with a sycl::errc enum value instead.}}
  sycl::invalid_object_error ioe;
  // expected-warning@+1 {{'memory_allocation_error' is deprecated: use sycl::exception with sycl::errc::memory_allocation instead.}}
  sycl::memory_allocation_error mae;
  // expected-warning@+1 {{'platform_error' is deprecated: use sycl::exception with sycl::errc::platform instead.}}
  sycl::platform_error ple;
  // expected-warning@+1 {{'profiling_error' is deprecated: use sycl::exception with sycl::errc::profiling instead.}}
  sycl::profiling_error pre;
  // expected-warning@+1 {{'feature_not_supported' is deprecated: use sycl::exception with sycl::errc::feature_not_supported instead.}}
  sycl::feature_not_supported fns;
  // expected-warning@+1{{'string_class' is deprecated: use STL classes directly}}
  sycl::string_class Str = "abc";
  (void)Str;
  // expected-warning@+1{{'mutex_class' is deprecated: use STL classes directly}}
  sycl::mutex_class Mtx;
  (void)Mtx;
  // expected-warning@+1{{'exception' is deprecated: The version of an exception constructor which takes no arguments is deprecated.}}
  sycl::exception ex;
  // expected-warning@+1{{'get_cl_code' is deprecated: use sycl::exception.code() instead.}}
  ex.get_cl_code();
  (void)ex;

  Queue.submit([](sycl::handler &CGH) {
    // expected-warning@+3{{'nd_range' is deprecated: offsets are deprecated in SYCL2020}}
    // expected-warning@+2{{'nd_range' is deprecated: offsets are deprecated in SYCL2020}}
    CGH.parallel_for<class Test>(
        sycl::nd_range<1>{sycl::range{10}, sycl::range{10}, sycl::range{1}},
        [](sycl::nd_item<1> it) {
          // expected-warning@+2{{'mem_fence' is deprecated: use sycl::atomic_fence() free function instead}}
          // expected-warning@+1{{'mem_fence<sycl::access::mode::read_write>' is deprecated: use sycl::atomic_fence() free function instead}}
          it.mem_fence();
        });
  });

  // expected-warning@+1{{'byte' is deprecated: use std::byte instead}}
  sycl::byte B;
  (void)B;

  // expected-warning@+1{{'max_constant_buffer_size' is deprecated: max_constant_buffer_size is deprecated}}
  auto MCBS = sycl::info::device::max_constant_buffer_size;
  (void)MCBS;
  // expected-warning@+1{{'max_constant_args' is deprecated: max_constant_args is deprecated}}
  auto MCA = sycl::info::device::max_constant_args;
  (void)MCA;

  // expected-warning@+4{{'ONEAPI' is deprecated: use 'ext::oneapi' instead}}
  // expected-warning@+3{{'atomic_fence' is deprecated: use sycl::atomic_fence instead}}
  // expected-warning@+2{{'ONEAPI' is deprecated: use 'ext::oneapi' instead}}
  // expected-warning@+2{{'ONEAPI' is deprecated: use 'ext::oneapi' instead}}
  sycl::ONEAPI::atomic_fence(sycl::ONEAPI::memory_order::relaxed,
                             sycl::ONEAPI::memory_scope::work_group);

  // expected-warning@+1{{'INTEL' is deprecated: use 'ext::intel' instead}}
  auto SL = sycl::INTEL::source_language::opencl_c;
  (void)SL;

  return 0;
}
