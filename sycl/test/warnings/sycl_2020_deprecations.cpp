// RUN: %clangxx %fsycl-host-only -fsyntax-only -sycl-std=2020 -Xclang -verify -Xclang -verify-ignore-unexpected=note %s -o %t.out

#include <CL/sycl.hpp>
#include <sycl/ext/intel/online_compiler.hpp>

int main() {
  cl_context ClCtx;
  // expected-error@+1 {{no matching constructor for initialization of 'sycl::context'}}
  sycl::context Ctx{ClCtx};
  // expected-error@+1 {{no member named 'get' in 'sycl::context'}}
  (void)Ctx.get();

  cl_mem Mem;
  // expected-error@+1 {{no matching constructor for initialization of 'sycl::buffer<int, 1>'}}
  sycl::buffer<int, 1> Buf{Mem, Ctx};
  (void)Buf;

  cl_device_id DevId;
  // expected-error@+1 {{no matching constructor for initialization of 'sycl::device'}}
  sycl::device Device{DevId};
  // expected-error@+1 {{no member named 'get' in 'sycl::device'}}
  (void)Device.get();
  // expected-warning@+1 {{'has_extension' is deprecated: use device::has() function with aspects APIs instead}}
  (void)Device.has_extension("abc");

  cl_event ClEvent;
  // expected-error@+1 {{no matching constructor for initialization of 'sycl::event'}}
  sycl::event Evt{ClEvent, Ctx};
  // expected-error@+1 {{no member named 'get' in 'sycl::event'}}
  (void)Evt.get();

  // expected-error@+1 {{no matching constructor for initialization of 'sycl::image<1>'}}
  sycl::image<1> Img{Mem, Ctx};
  (void)Img;

  cl_platform_id ClPlatform;
  // expected-error@+1 {{no matching constructor for initialization of 'sycl::platform'}}
  sycl::platform Platform{ClPlatform};
  // expected-error@+1 {{no member named 'get' in 'sycl::platform'}}
  (void)Platform.get();
  // expected-warning@+1 {{'has_extension' is deprecated: use platform::has() function with aspects APIs instead}}
  (void)Platform.has_extension("abc");

  cl_command_queue ClQueue;
  // expected-error@+1 {{no matching constructor for initialization of 'sycl::queue'}}
  sycl::queue Queue{ClQueue, Ctx};
  // expected-error@+1 {{no member named 'get' in 'sycl::queue'}}
  (void)Queue.get();

  cl_sampler ClSampler;
  // expected-error@+1 {{no matching constructor for initialization of 'sycl::sampler'}}
  sycl::sampler Sampler{ClSampler, Ctx};
  (void)Sampler;

  cl_kernel ClKernel;
  // expected-error@+1 {{no matching constructor for initialization of 'sycl::kernel'}}
  sycl::kernel Kernel{ClKernel, Ctx};
  // expected-error@+1 {{no member named 'get' in 'sycl::kernel'}}
  (void)Kernel.get();

  // expected-error@+1 {{no type named 'program' in namespace 'sycl'}}
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

  // expected-warning@+1{{'built_in_kernels' is deprecated: use built_in_kernel_ids instead}}
  auto BIK = sycl::info::device::built_in_kernels;
  (void)BIK;

  // expected-warning@+1{{'extensions' is deprecated: platform::extensions is deprecated, use device::get_info() with info::device::aspects instead.}}
  auto PE = sycl::info::platform::extensions;

  // expected-warning@+1{{'extensions' is deprecated: device::extensions is deprecated, use info::device::aspects instead.}}
  auto DE = sycl::info::device::extensions;

  // expected-warning@+3{{'atomic_fence' is deprecated: use sycl::atomic_fence instead}}
  // expected-error@+2{{no member named 'ONEAPI' in namespace 'sycl'}}
  // expected-error@+2{{no member named 'ONEAPI' in namespace 'sycl'}}
  sycl::ext::oneapi::atomic_fence(sycl::ONEAPI::memory_order::relaxed,
                             sycl::ONEAPI::memory_scope::work_group);

  // expected-error@+1{{no member named 'INTEL' in namespace 'sycl'}}
  auto SL = sycl::INTEL::source_language::opencl_c;
  (void)SL;

  // expected-warning@+1{{'intel' is deprecated: use 'ext::intel::experimental' instead}}
  auto SLExtIntel = sycl::ext::intel::source_language::opencl_c;
  (void)SLExtIntel;

  // expected-warning@+1{{'level_zero' is deprecated: use 'ext_oneapi_level_zero' instead}}
  auto LevelZeroBackend = sycl::backend::level_zero;
  (void)LevelZeroBackend;

  // expected-warning@+1{{'esimd_cpu' is deprecated: use 'ext_oneapi_esimd_emulator' instead}}
  auto ESIMDCPUBackend = sycl::backend::esimd_cpu;
  (void)ESIMDCPUBackend;

  sycl::half Val = 1.0f;
  // expected-warning@+1{{'bit_cast<unsigned short, sycl::detail::half_impl::half>' is deprecated: use 'sycl::bit_cast' instead}}
  auto BitCastRes = sycl::detail::bit_cast<unsigned short>(Val);
  (void)BitCastRes;

  // expected-warning@+1{{'submit_barrier' is deprecated: use 'ext_oneapi_submit_barrier' instead}}
  Queue.submit_barrier();

  // expected-warning@+1{{'barrier' is deprecated: use 'ext_oneapi_barrier' instead}}
  Queue.submit([&](sycl::handler &CGH) { CGH.barrier(); });

  cl::sycl::multi_ptr<int, cl::sycl::access::address_space::global_space> a(
      nullptr);
  // expected-warning@+1 {{'atomic<int, sycl::access::address_space::global_space>' is deprecated: sycl::atomic is deprecated since SYCL 2020}}
  cl::sycl::atomic<int> b(a);

  return 0;
}
