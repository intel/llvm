// RUN: %clangxx %fsycl-host-only -fsyntax-only -ferror-limit=0 -sycl-std=2020 -Xclang -verify -Xclang -verify-ignore-unexpected=note %s

#include <CL/sycl.hpp>
#include <sycl/ext/intel/experimental/online_compiler.hpp>

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
  // expected-warning@+1{{'host' is deprecated: removed in SYCL 2020, 'host' device has been removed}}
  (void)Device.has(sycl::aspect::host);

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

  sycl::buffer<int, 1> Buffer(4);
  // expected-warning@+1{{'get_count' is deprecated: get_count() is deprecated, please use size() instead}}
  size_t BufferGetCount = Buffer.get_count();
  size_t BufferSize = Buffer.size();
  // expected-warning@+1 {{'get_size' is deprecated: get_size() is deprecated, please use byte_size() instead}}
  size_t BufferGetSize = Buffer.get_size();
  {
    // expected-warning@+2 {{'get_access' is deprecated: get_access for host_accessor is deprecated, please use get_host_access instead}}
    // expected-warning@+1 {{'get_access<sycl::access::mode::read_write>' is deprecated: get_access for host_accessor is deprecated, please use get_host_access instead}}
    auto acc = Buffer.get_access<sycl::access_mode::read_write>();
  }
  {
    // expected-warning@+3 {{'get_access' is deprecated: get_access for host_accessor is deprecated, please use get_host_access instead}}
    // expected-warning@+2 {{'get_access<sycl::access::mode::read_write>' is deprecated: get_access for host_accessor is deprecated, please use get_host_access instead}}
    auto acc =
        Buffer.get_access<sycl::access_mode::read_write>(sycl::range<1>(0));
  }

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

  // expected-warning@+1{{'online_compiler<sycl::ext::intel::experimental::source_language::opencl_c>' is deprecated}}
  sycl::ext::intel::experimental::online_compiler<
      sycl::ext::intel::experimental::source_language::opencl_c>
      oc(Device);

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

  // expected-warning@+1{{abs for floating point types is non-standard and has been deprecated. Please use fabs instead.}}
  sycl::abs(0.0f);

  // expected-warning@+1{{'image_support' is deprecated: deprecated in SYCL 2020, use device::has(aspect::ext_intel_legacy_image) to query for SYCL 1.2.1 image support}}
  using IS = sycl::info::device::image_support;
  // expected-warning@+1{{'max_constant_buffer_size' is deprecated: deprecated in SYCL 2020}}
  using MCBS = sycl::info::device::max_constant_buffer_size;
  // expected-warning@+1{{'max_constant_args' is deprecated: deprecated in SYCL 2020}}
  using MCA = sycl::info::device::max_constant_args;
  // expected-warning@+1{{'host_unified_memory' is deprecated: deprecated in SYCL 2020, use device::has() with one of the aspect::usm_* aspects instead}}
  using HUM = sycl::info::device::host_unified_memory;
  // expected-warning@+1{{'is_endian_little' is deprecated: deprecated in SYCL 2020, check the byte order of the host system instead, the host and the device are required to have the same byte order}}
  using IEL = sycl::info::device::is_endian_little;
  // expected-warning@+1{{'is_compiler_available' is deprecated: deprecated in SYCL 2020, use device::has(aspect::online_compiler) instead}}
  using ICA = sycl::info::device::is_compiler_available;
  // expected-warning@+1{{'is_linker_available' is deprecated: deprecated in SYCL 2020, use device::has(aspect::online_linker) instead}}
  using ILA = sycl::info::device::is_linker_available;
  // expected-warning@+1{{'queue_profiling' is deprecated: deprecated in SYCL 2020, use device::has(aspect::queue_profiling) instead}}
  using QP = sycl::info::device::queue_profiling;
  // expected-warning@+1{{'built_in_kernels' is deprecated: deprecated in SYCL 2020, use info::device::built_in_kernel_ids instead}}
  using BIK = sycl::info::device::built_in_kernels;
  // expected-warning@+1{{'profile' is deprecated: deprecated in SYCL 2020}}
  using DP = sycl::info::device::profile;
  // expected-warning@+1{{'extensions' is deprecated: deprecated in SYCL 2020, use info::device::aspects instead}}
  using DE = sycl::info::device::extensions;
  // expected-warning@+1{{'printf_buffer_size' is deprecated: deprecated in SYCL 2020}}
  using PBS = sycl::info::device::printf_buffer_size;
  // expected-warning@+1{{'preferred_interop_user_sync' is deprecated: deprecated in SYCL 2020}}
  using PIUS = sycl::info::device::preferred_interop_user_sync;
  // expected-warning@+1{{'usm_system_allocator' is deprecated: use info::device::usm_system_allocations instead}}
  using USA = sycl::info::device::usm_system_allocator;

  // expected-warning@+1{{'extensions' is deprecated: deprecated in SYCL 2020, use device::get_info() with info::device::aspects instead}}
  using PE = sycl::info::platform::extensions;

  // expected-error@+1{{no member named 'INTEL' in namespace 'sycl'}}
  auto SL = sycl::INTEL::source_language::opencl_c;
  (void)SL;

  sycl::multi_ptr<int, sycl::access::address_space::global_space,
                  sycl::access::decorated::yes>
      a(nullptr);
  // expected-warning@+1 {{'atomic<int, sycl::access::address_space::global_space>' is deprecated: sycl::atomic is deprecated since SYCL 2020}}
  sycl::atomic<int> b(a);

  sycl::group<1> group = sycl::detail::Builder::createGroup<1>({8}, {4}, {1});
  // expected-warning@+1{{'get_id' is deprecated: use sycl::group::get_group_id() instead}}
  group.get_id();
  // expected-warning@+1{{'get_id' is deprecated: use sycl::group::get_group_id() instead}}
  group.get_id(1);
  // expected-warning@+1{{'get_linear_id' is deprecated: use sycl::group::get_group_linear_id() instead}}
  group.get_linear_id();

  // expected-warning@+1{{'default_selector' is deprecated: Use the callable sycl::default_selector_v instead.}}
  sycl::default_selector ds;
  // expected-warning@+1{{'cpu_selector' is deprecated: Use the callable sycl::cpu_selector_v instead.}}
  sycl::cpu_selector cs;
  // expected-warning@+1{{'gpu_selector' is deprecated: Use the callable sycl::gpu_selector_v instead.}}
  sycl::gpu_selector gs;
  // expected-warning@+1{{'accelerator_selector' is deprecated: Use the callable sycl::accelerator_selector_v instead.}}
  sycl::accelerator_selector as;
  // expected-warning@+1{{'host_selector' is deprecated: Host device is no longer supported.}}
  sycl::host_selector hs;

  // expected-warning@+1{{Use SYCL 2020 callable device selectors instead.}}
  class user_defined_device_selector : public sycl::device_selector {
  public:
    int operator()(const sycl::device &) const override { return 100; }
  } uds;

  // expected-warning@+1{{SYCL 1.2.1 device selectors are deprecated. Please use SYCL 2020 device selectors instead.}}
  sycl::device dd{ds};
  // expected-warning@+1{{SYCL 1.2.1 device selectors are deprecated. Please use SYCL 2020 device selectors instead.}}
  sycl::device cd{cs};
  // expected-warning@+1{{SYCL 1.2.1 device selectors are deprecated. Please use SYCL 2020 device selectors instead.}}
  sycl::device gd{gs};
  // expected-warning@+1{{SYCL 1.2.1 device selectors are deprecated. Please use SYCL 2020 device selectors instead.}}
  sycl::device ad{as};
  // expected-warning@+1{{SYCL 1.2.1 device selectors are deprecated. Please use SYCL 2020 device selectors instead.}}
  sycl::device hd{hs};
  // expected-warning@+1{{SYCL 1.2.1 device selectors are deprecated. Please use SYCL 2020 device selectors instead.}}
  sycl::device udd{uds};

  // expected-warning@+1{{SYCL 1.2.1 device selectors are deprecated. Please use SYCL 2020 device selectors instead.}}
  sycl::platform dp{ds};
  // expected-warning@+1{{SYCL 1.2.1 device selectors are deprecated. Please use SYCL 2020 device selectors instead.}}
  sycl::platform cp{cs};
  // expected-warning@+1{{SYCL 1.2.1 device selectors are deprecated. Please use SYCL 2020 device selectors instead.}}
  sycl::platform gp{gs};
  // expected-warning@+1{{SYCL 1.2.1 device selectors are deprecated. Please use SYCL 2020 device selectors instead.}}
  sycl::platform ap{as};
  // expected-warning@+1{{SYCL 1.2.1 device selectors are deprecated. Please use SYCL 2020 device selectors instead.}}
  sycl::platform hp{hs};
  // expected-warning@+1{{SYCL 1.2.1 device selectors are deprecated. Please use SYCL 2020 device selectors instead.}}
  sycl::platform udp{uds};

  // expected-warning@+1{{SYCL 1.2.1 device selectors are deprecated. Please use SYCL 2020 device selectors instead.}}
  sycl::queue dq1{ds};
  // expected-warning@+1{{SYCL 1.2.1 device selectors are deprecated. Please use SYCL 2020 device selectors instead.}}
  sycl::queue cq1{cs};
  // expected-warning@+1{{SYCL 1.2.1 device selectors are deprecated. Please use SYCL 2020 device selectors instead.}}
  sycl::queue gq1{gs};
  // expected-warning@+1{{SYCL 1.2.1 device selectors are deprecated. Please use SYCL 2020 device selectors instead.}}
  sycl::queue aq1{as};
  // expected-warning@+1{{SYCL 1.2.1 device selectors are deprecated. Please use SYCL 2020 device selectors instead.}}
  sycl::queue hq1{hs};
  // expected-warning@+1{{SYCL 1.2.1 device selectors are deprecated. Please use SYCL 2020 device selectors instead.}}
  sycl::queue udq1{uds};

  sycl::context ctx;

  // expected-warning@+1{{SYCL 1.2.1 device selectors are deprecated. Please use SYCL 2020 device selectors instead.}}
  sycl::queue dq2{ctx, ds};
  // expected-warning@+1{{SYCL 1.2.1 device selectors are deprecated. Please use SYCL 2020 device selectors instead.}}
  sycl::queue cq2{ctx, cs};
  // expected-warning@+1{{SYCL 1.2.1 device selectors are deprecated. Please use SYCL 2020 device selectors instead.}}
  sycl::queue gq2{ctx, gs};
  // expected-warning@+1{{SYCL 1.2.1 device selectors are deprecated. Please use SYCL 2020 device selectors instead.}}
  sycl::queue aq2{ctx, as};
  // expected-warning@+1{{SYCL 1.2.1 device selectors are deprecated. Please use SYCL 2020 device selectors instead.}}
  sycl::queue hq2{ctx, hs};
  // expected-warning@+1{{SYCL 1.2.1 device selectors are deprecated. Please use SYCL 2020 device selectors instead.}}
  sycl::queue udq2{ctx, uds};

  auto ah = [](sycl::exception_list) {};

  // expected-warning@+1{{SYCL 1.2.1 device selectors are deprecated. Please use SYCL 2020 device selectors instead.}}
  sycl::queue dq3{ds, ah};
  // expected-warning@+1{{SYCL 1.2.1 device selectors are deprecated. Please use SYCL 2020 device selectors instead.}}
  sycl::queue cq3{cs, ah};
  // expected-warning@+1{{SYCL 1.2.1 device selectors are deprecated. Please use SYCL 2020 device selectors instead.}}
  sycl::queue gq3{gs, ah};
  // expected-warning@+1{{SYCL 1.2.1 device selectors are deprecated. Please use SYCL 2020 device selectors instead.}}
  sycl::queue aq3{as, ah};
  // expected-warning@+1{{SYCL 1.2.1 device selectors are deprecated. Please use SYCL 2020 device selectors instead.}}
  sycl::queue hq3{hs, ah};
  // expected-warning@+1{{SYCL 1.2.1 device selectors are deprecated. Please use SYCL 2020 device selectors instead.}}
  sycl::queue udq3{uds, ah};

  // expected-warning@+1{{SYCL 1.2.1 device selectors are deprecated. Please use SYCL 2020 device selectors instead.}}
  sycl::queue dq4{ctx, ds, ah};
  // expected-warning@+1{{SYCL 1.2.1 device selectors are deprecated. Please use SYCL 2020 device selectors instead.}}
  sycl::queue cq4{ctx, cs, ah};
  // expected-warning@+1{{SYCL 1.2.1 device selectors are deprecated. Please use SYCL 2020 device selectors instead.}}
  sycl::queue gq4{ctx, gs, ah};
  // expected-warning@+1{{SYCL 1.2.1 device selectors are deprecated. Please use SYCL 2020 device selectors instead.}}
  sycl::queue aq4{ctx, as, ah};
  // expected-warning@+1{{SYCL 1.2.1 device selectors are deprecated. Please use SYCL 2020 device selectors instead.}}
  sycl::queue hq4{ctx, hs, ah};
  // expected-warning@+1{{SYCL 1.2.1 device selectors are deprecated. Please use SYCL 2020 device selectors instead.}}
  sycl::queue udq4{ctx, uds, ah};

  Queue.submit([&](sycl::handler &CGH) {
    // expected-warning@+1{{'local' is deprecated: use `local_accessor` instead}}
    sycl::accessor<int, 1, sycl::access::mode::read_write, sycl::target::local>
        LocalAcc(sycl::range<1>(1), CGH);
    // expected-warning@+3{{'multi_ptr' is deprecated: multi_ptr construction using target::local specialized accessor is deprecated since SYCL 2020}}
    sycl::multi_ptr<int, sycl::access::address_space::local_space,
                    sycl::access::decorated::no>
        LocalMptr(LocalAcc);
    // expected-warning@+3{{'multi_ptr' is deprecated: multi_ptr construction using target::local specialized accessor is deprecated since SYCL 2020}}
    sycl::multi_ptr<int, sycl::access::address_space::generic_space,
                    sycl::access::decorated::no>
        GenericMptr(LocalAcc);
    // expected-warning@+1{{'local' is deprecated: use `local_accessor` instead}}
    sycl::accessor<const int, 1, sycl::access::mode::read, sycl::target::local>
        LocalConstAcc(sycl::range<1>(1), CGH);
    // expected-warning@+3{{'multi_ptr' is deprecated: multi_ptr construction using target::local specialized accessor is deprecated since SYCL 2020}}
    sycl::multi_ptr<const int, sycl::access::address_space::local_space,
                    sycl::access::decorated::no>
        LocalConstMptr(LocalConstAcc);
    // expected-warning@+3{{'multi_ptr' is deprecated: multi_ptr construction using target::local specialized accessor is deprecated since SYCL 2020}}
    sycl::multi_ptr<const int, sycl::access::address_space::generic_space,
                    sycl::access::decorated::no>
        GenericConstMptr(LocalConstAcc);
    // expected-warning@+3{{'multi_ptr' is deprecated: multi_ptr construction using target::local specialized accessor is deprecated since SYCL 2020}}
    sycl::multi_ptr<const void, sycl::access::address_space::local_space,
                    sycl::access::decorated::no>
        LocalConstVoidMptr(LocalConstAcc);
    // expected-warning@+3{{'multi_ptr' is deprecated: multi_ptr construction using target::local specialized accessor is deprecated since SYCL 2020}}
    sycl::multi_ptr<void, sycl::access::address_space::local_space,
                    sycl::access::decorated::no>
        LocalVoidMptr(LocalAcc);
  });

  Queue.submit([&](sycl::handler &CGH) {
    sycl::accessor GlobalAcc{Buffer, CGH, sycl::write_only};
    sycl::local_accessor<int, 1> LocalAcc{1, CGH};
    CGH.parallel_for(
        sycl::nd_range<1>{sycl::range<1>{1}, sycl::range<1>{1}},
        [=](sycl::nd_item<1> Idx) {
          int PrivateVal = 0;

          // expected-warning@+6{{'legacy' is deprecated: sycl::access::decorated::legacy is deprecated since SYCL 2020}}
          // expected-warning@+8{{'legacy' is deprecated: sycl::access::decorated::legacy is deprecated since SYCL 2020}}
          // expected-warning@+8{{'get_pointer' is deprecated: accessor::get_pointer() is deprecated, please use get_multi_ptr()}}
          // expected-warning@+7{{'get_pointer<sycl::access::target::global_buffer, void>' is deprecated: accessor::get_pointer() is deprecated, please use get_multi_ptr()}}
          // expected-warning@+4{{'make_ptr<int, sycl::access::address_space::global_space, sycl::access::decorated::legacy, void>' is deprecated: make_ptr is deprecated since SYCL 2020. Please use address_space_cast instead.}}
          sycl::multi_ptr<int, sycl::access::address_space::global_space,
                          sycl::access::decorated::legacy>
              LegacyGlobalMptr =
                  sycl::make_ptr<int, sycl::access::address_space::global_space,
                                 sycl::access::decorated::legacy>(
                      GlobalAcc.get_pointer());
          // expected-warning@+5{{'legacy' is deprecated: sycl::access::decorated::legacy is deprecated since SYCL 2020}}
          // expected-warning@+7{{'legacy' is deprecated: sycl::access::decorated::legacy is deprecated since SYCL 2020}}
          // expected-warning@+7{{'get_pointer' is deprecated: local_accessor::get_pointer() is deprecated, please use get_multi_ptr()}}
          // expected-warning@+4{{'make_ptr<int, sycl::access::address_space::local_space, sycl::access::decorated::legacy, void>' is deprecated: make_ptr is deprecated since SYCL 2020. Please use address_space_cast instead.}}
          sycl::multi_ptr<int, sycl::access::address_space::local_space,
                          sycl::access::decorated::legacy>
              LegacyLocalMptr =
                  sycl::make_ptr<int, sycl::access::address_space::local_space,
                                 sycl::access::decorated::legacy>(
                      LocalAcc.get_pointer());

          // expected-warning@+4{{'legacy' is deprecated: sycl::access::decorated::legacy is deprecated since SYCL 2020}}
          // expected-warning@+5{{'make_ptr<int, sycl::access::address_space::private_space, sycl::access::decorated::legacy, void>' is deprecated: make_ptr is deprecated since SYCL 2020. Please use address_space_cast instead.}}
          // expected-warning@+6{{'legacy' is deprecated: sycl::access::decorated::legacy is deprecated since SYCL 2020}}
          sycl::multi_ptr<int, sycl::access::address_space::private_space,
                          sycl::access::decorated::legacy>
              LegacyPrivateMptr =
                  sycl::make_ptr<int,
                                 sycl::access::address_space::private_space,
                                 sycl::access::decorated::legacy>(&PrivateVal);

          sycl::multi_ptr<int, sycl::access::address_space::global_space,
                          sycl::access::decorated::yes>
              DecoratedGlobalMptr{GlobalAcc};
          sycl::multi_ptr<int, sycl::access::address_space::local_space,
                          sycl::access::decorated::yes>
              DecoratedLocalMptr{LocalAcc};
          sycl::multi_ptr<int, sycl::access::address_space::private_space,
                          sycl::access::decorated::yes>
              DecoratedPrivateMptr = sycl::address_space_cast<
                  sycl::access::address_space::private_space,
                  sycl::access::decorated::yes>(&PrivateVal);

          sycl::multi_ptr<int, sycl::access::address_space::global_space,
                          sycl::access::decorated::yes>
              UndecoratedGlobalMptr = DecoratedGlobalMptr;
          sycl::multi_ptr<int, sycl::access::address_space::local_space,
                          sycl::access::decorated::yes>
              UndecoratedLocalMptr = DecoratedLocalMptr;
          sycl::multi_ptr<int, sycl::access::address_space::private_space,
                          sycl::access::decorated::yes>
              UndecoratedPrivateMptr = DecoratedPrivateMptr;

          // expected-warning@+2{{'operator int *' is deprecated: Conversion to pointer type is deprecated since SYCL 2020. Please use get() instead.}}
          auto DecoratedGlobalPtr =
              static_cast<typename decltype(DecoratedGlobalMptr)::pointer>(
                  DecoratedGlobalMptr);
          // expected-warning@+2{{'operator int *' is deprecated: Conversion to pointer type is deprecated since SYCL 2020. Please use get() instead.}}
          auto DecoratedLocalPtr =
              static_cast<typename decltype(DecoratedLocalMptr)::pointer>(
                  DecoratedLocalMptr);
          // expected-warning@+2{{'operator int *' is deprecated: Conversion to pointer type is deprecated since SYCL 2020. Please use get() instead.}}
          auto DecoratedPrivatePtr =
              static_cast<typename decltype(DecoratedPrivateMptr)::pointer>(
                  DecoratedPrivateMptr);
          // expected-warning@+2{{'operator int *' is deprecated: Conversion to pointer type is deprecated since SYCL 2020. Please use get() instead.}}
          auto UndecoratedGlobalPtr =
              static_cast<typename decltype(UndecoratedGlobalMptr)::pointer>(
                  UndecoratedGlobalMptr);
          // expected-warning@+2{{'operator int *' is deprecated: Conversion to pointer type is deprecated since SYCL 2020. Please use get() instead.}}
          auto UndecoratedLocalPtr =
              static_cast<typename decltype(UndecoratedLocalMptr)::pointer>(
                  UndecoratedLocalMptr);
          // expected-warning@+2{{'operator int *' is deprecated: Conversion to pointer type is deprecated since SYCL 2020. Please use get() instead.}}
          auto UndecoratedPrivatePtr =
              static_cast<typename decltype(UndecoratedPrivateMptr)::pointer>(
                  UndecoratedPrivateMptr);

          // expected-warning@+2{{'async_work_group_copy' is deprecated: Use decorated multi_ptr arguments instead}}
          // expected-warning@+1{{'async_work_group_copy<int>' is deprecated: Use decorated multi_ptr arguments instead}}
          Idx.async_work_group_copy(LegacyGlobalMptr, LegacyLocalMptr, 10);
          // expected-warning@+2{{'async_work_group_copy' is deprecated: Use decorated multi_ptr arguments instead}}
          // expected-warning@+1{{'async_work_group_copy<int>' is deprecated: Use decorated multi_ptr arguments instead}}
          Idx.async_work_group_copy(LegacyLocalMptr, LegacyGlobalMptr, 10);
          // expected-warning@+2{{'async_work_group_copy' is deprecated: Use decorated multi_ptr arguments instead}}
          // expected-warning@+1{{'async_work_group_copy<int>' is deprecated: Use decorated multi_ptr arguments instead}}
          Idx.async_work_group_copy(LegacyGlobalMptr, LegacyLocalMptr, 10, 2);
          // expected-warning@+2{{'async_work_group_copy' is deprecated: Use decorated multi_ptr arguments instead}}
          // expected-warning@+1{{'async_work_group_copy<int>' is deprecated: Use decorated multi_ptr arguments instead}}
          Idx.async_work_group_copy(LegacyLocalMptr, LegacyGlobalMptr, 10, 2);

          auto Group = Idx.get_group();
          // expected-warning@+2{{'async_work_group_copy' is deprecated: Use decorated multi_ptr arguments instead}}
          // expected-warning@+1{{'async_work_group_copy<int>' is deprecated: Use decorated multi_ptr arguments instead}}
          Group.async_work_group_copy(LegacyGlobalMptr, LegacyLocalMptr, 10);
          // expected-warning@+2{{'async_work_group_copy' is deprecated: Use decorated multi_ptr arguments instead}}
          // expected-warning@+1{{'async_work_group_copy<int>' is deprecated: Use decorated multi_ptr arguments instead}}
          Group.async_work_group_copy(LegacyLocalMptr, LegacyGlobalMptr, 10);
          // expected-warning@+2{{'async_work_group_copy' is deprecated: Use decorated multi_ptr arguments instead}}
          // expected-warning@+1{{'async_work_group_copy<int>' is deprecated: Use decorated multi_ptr arguments instead}}
          Group.async_work_group_copy(LegacyGlobalMptr, LegacyLocalMptr, 10, 2);
          // expected-warning@+2{{'async_work_group_copy' is deprecated: Use decorated multi_ptr arguments instead}}
          // expected-warning@+1{{'async_work_group_copy<int>' is deprecated: Use decorated multi_ptr arguments instead}}
          Group.async_work_group_copy(LegacyLocalMptr, LegacyGlobalMptr, 10, 2);
        });
  });

  Queue.submit([&](sycl::handler &CGH) {
    sycl::stream Stream(1024, 80, CGH);
    // expected-warning@+1{{'get_size' is deprecated: get_size() is deprecated since SYCL 2020. Please use size() instead.}}
    size_t StreamSize = Stream.get_size();
    // expected-warning@+1{{'get_max_statement_size' is deprecated: get_max_statement_size() is deprecated since SYCL 2020. Please use get_work_item_buffer_size() instead.}}
    size_t StreamMaxStatementSize = Stream.get_max_statement_size();
  });

  // expected-warning@+1 {{'fast_distance<double, double>' is deprecated: fast_distance for double precision types is non-standard and has been deprecated}}
  std::ignore = sycl::fast_distance(double{1.0}, double{2.0});
  // expected-warning@+2 {{'fast_distance<sycl::vec<double, 2>, sycl::vec<double, 2>>' is deprecated: fast_distance for double precision types is non-standard and has been deprecated}}
  std::ignore =
      sycl::fast_distance(sycl::vec<double, 2>{0.0}, sycl::vec<double, 2>{1.0});

  // clang-format off
  // SYCL 2020, revision 9 uses fixed-width integer type, one of these has to be
  // deprecated.
  // expected-warning-re@+1 {{'nan<{{.*}}>' is deprecated: This is a deprecated argument type for SYCL nan built-in function.}}
  std::ignore = (sycl::nan((unsigned long){0}), sycl::nan((unsigned long long){0}));
  // expected-warning-re@+1 {{'nan<sycl::vec<{{.*}}, 2>>' is deprecated: This is a deprecated argument type for SYCL nan built-in function.}}
  std::ignore = (sycl::nan(sycl::vec<unsigned long, 2>{0}), sycl::nan(sycl::vec<unsigned long long, 2>{0}));
  // clang-format on

  return 0;
}
