// RUN: %clangxx %fsycl-host-only -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s -o %t.out

// Test for SYCL-2020 Level Zero interop API

#include <sycl/sycl.hpp>
// clang-format off
#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
// clang-format on

using namespace sycl;

//
// 4.5.1 SYCL application interoperability may be provided for
// platform,
// device,
// context,
// queue,
// event,
// kernel_bundle,
// kernel,
// TODO:
// buffer,
// device_image,
// sampled_image,
// unsampled_image.

int main() {

  // Create SYCL objects
  device Device;
  platform Platform = Device.get_info<info::device::platform>();
  context Context(Device);
  queue Queue(Device);
  event Event;
  kernel_bundle<bundle_state::executable> KernelBundle =
      get_kernel_bundle<bundle_state::executable>(Context);
  kernel Kernel = KernelBundle.get_kernel(get_kernel_ids().front());

  // 4.5.1.1 For each SYCL runtime class T which supports SYCL application
  // interoperability with the SYCL backend, a specialization of return_type
  // must be defined as the type of SYCL application interoperability native
  // backend object associated with T for the SYCL backend, specified in the
  // SYCL backend specification.
  //
  // return_type is used when retrieving the backend specific native object from
  // a SYCL object. See the relevant backend specification for details.

  backend_traits<backend::ext_oneapi_level_zero>::return_type<platform>
      ZeDriver;
  backend_traits<backend::ext_oneapi_level_zero>::return_type<device> ZeDevice;
  backend_traits<backend::ext_oneapi_level_zero>::return_type<context>
      ZeContext;
  backend_traits<backend::ext_oneapi_level_zero>::return_type<queue> ZeQueue;
  backend_traits<backend::ext_oneapi_level_zero>::return_type<event> ZeEvent;
  backend_traits<backend::ext_oneapi_level_zero>::return_type<
      kernel_bundle<bundle_state::executable>>
      ZeKernelBundle;
  backend_traits<backend::ext_oneapi_level_zero>::return_type<kernel> ZeKernel;

  // 4.5.1.2 For each SYCL runtime class T which supports SYCL application
  // interoperability, a specialization of get_native must be defined, which
  // takes an instance of T and returns a SYCL application interoperability
  // native backend object associated with syclObject which can be used for SYCL
  // application interoperability. The lifetime of the object returned are
  // backend-defined and specified in the backend specification.

  ZeDriver = get_native<backend::ext_oneapi_level_zero>(Platform);
  ZeDevice = get_native<backend::ext_oneapi_level_zero>(Device);
  ZeContext = get_native<backend::ext_oneapi_level_zero>(Context);
  ZeQueue = get_native<backend::ext_oneapi_level_zero>(Queue);
  ZeEvent = get_native<backend::ext_oneapi_level_zero>(Event);
  ZeKernelBundle = get_native<backend::ext_oneapi_level_zero>(KernelBundle);
  ZeKernel = get_native<backend::ext_oneapi_level_zero>(Kernel);

  // Check deprecated
  // expected-warning@+2 {{'get_native' is deprecated: Use SYCL 2020 sycl::get_native free function}}
  // expected-warning@+1 {{'get_native<sycl::backend::ext_oneapi_level_zero>' is deprecated: Use SYCL 2020 sycl::get_native free function}}
  ZeDriver = Platform.get_native<backend::ext_oneapi_level_zero>();
  // expected-warning@+2 {{'get_native' is deprecated: Use SYCL 2020 sycl::get_native free function}}
  // expected-warning@+1 {{'get_native<sycl::backend::ext_oneapi_level_zero>' is deprecated: Use SYCL 2020 sycl::get_native free function}}
  ZeDevice = Device.get_native<backend::ext_oneapi_level_zero>();
  // expected-warning@+2 {{'get_native' is deprecated: Use SYCL 2020 sycl::get_native free function}}
  // expected-warning@+1 {{'get_native<sycl::backend::ext_oneapi_level_zero>' is deprecated: Use SYCL 2020 sycl::get_native free function}}
  ZeContext = Context.get_native<backend::ext_oneapi_level_zero>();
  // expected-warning@+2 {{'get_native' is deprecated: Use SYCL 2020 sycl::get_native free function}}
  // expected-warning@+1 {{'get_native<sycl::backend::ext_oneapi_level_zero>' is deprecated: Use SYCL 2020 sycl::get_native free function}}
  ZeQueue = Queue.get_native<backend::ext_oneapi_level_zero>();
  // expected-warning@+2 {{'get_native' is deprecated: Use SYCL 2020 sycl::get_native free function}}
  // expected-warning@+1 {{'get_native<sycl::backend::ext_oneapi_level_zero>' is deprecated: Use SYCL 2020 sycl::get_native free function}}
  ZeEvent = Event.get_native<backend::ext_oneapi_level_zero>();
  // expected-warning@+3 {{'get_native' is deprecated: Use SYCL 2020 sycl::get_native free function}}
  // expected-warning@+2 {{'get_native<sycl::backend::ext_oneapi_level_zero>' is deprecated: Use SYCL 2020 sycl::get_native free function}}
  /*ZeKernelBundle*/ (
      void)KernelBundle.get_native<backend::ext_oneapi_level_zero>();
  // expected-warning@+2 {{'get_native' is deprecated: Use SYCL 2020 sycl::get_native free function}}
  // expected-warning@+1 {{'get_native<sycl::backend::ext_oneapi_level_zero>' is deprecated: Use SYCL 2020 sycl::get_native free function}}
  ZeKernel = Kernel.get_native<backend::ext_oneapi_level_zero>();

  // 4.5.1.1 For each SYCL runtime class T which supports SYCL application
  // interoperability with the SYCL backend, a specialization of input_type must
  // be defined as the type of SYCL application interoperability native backend
  // object associated with T for the SYCL backend, specified in the SYCL
  // backend specification. input_type is used when constructing SYCL objects
  // from backend specific native objects. See the relevant backend
  // specification for details.

  // 4.5.1.3 For each SYCL runtime class T which supports SYCL application
  // interoperability, a specialization of the appropriate template function
  // make_{sycl_class} where {sycl_class} is the class name of T, must be
  // defined, which takes a SYCL application interoperability native backend
  // object and constructs and returns an instance of T. The availability and
  // behavior of these template functions is defined by the SYCL backend
  // specification document.

  backend_input_t<backend::ext_oneapi_level_zero, platform>
      InteropPlatformInput{ZeDriver};
  platform InteropPlatform =
      make_platform<backend::ext_oneapi_level_zero>(InteropPlatformInput);

  backend_input_t<backend::ext_oneapi_level_zero, device> InteropDeviceInput{
      ZeDevice};
  device InteropDevice =
      make_device<backend::ext_oneapi_level_zero>(InteropDeviceInput);

  backend_input_t<backend::ext_oneapi_level_zero, context> InteropContextInput{
      ZeContext, std::vector<device>(1, InteropDevice),
      ext::oneapi::level_zero::ownership::keep};
  context InteropContext =
      make_context<backend::ext_oneapi_level_zero>(InteropContextInput);

  queue InteropQueue = make_queue<backend::ext_oneapi_level_zero>(
      {ZeQueue, ext::oneapi::level_zero::ownership::keep}, Context);
  event InteropEvent = make_event<backend::ext_oneapi_level_zero>(
      {ZeEvent, ext::oneapi::level_zero::ownership::keep}, Context);
  kernel_bundle<bundle_state::executable> InteropKernelBundle =
      make_kernel_bundle<backend::ext_oneapi_level_zero,
                         bundle_state::executable>(
          {ZeKernelBundle.front(), ext::oneapi::level_zero::ownership::keep},
          Context);
  kernel InteropKernel = make_kernel<backend::ext_oneapi_level_zero>(
      {KernelBundle, ZeKernel, ext::oneapi::level_zero::ownership::keep},
      Context);

  // Check deprecated
  // expected-warning@+1 {{'make<sycl::platform, nullptr>' is deprecated: Use SYCL 2020 sycl::make_platform free function}}
  auto P = ext::oneapi::level_zero::make<platform>(ZeDriver);
  // expected-warning@+1 {{'make<sycl::device, nullptr>' is deprecated: Use SYCL 2020 sycl::make_device free function}}
  auto D = ext::oneapi::level_zero::make<device>(P, ZeDevice);
  // expected-warning@+1 {{'make<sycl::context, nullptr>' is deprecated: Use SYCL 2020 sycl::make_context free function}}
  auto C = ext::oneapi::level_zero::make<context>(
      std::vector<device>(1, D), ZeContext,
      ext::oneapi::level_zero::ownership::keep);
  // expected-warning@+1 {{'make<sycl::queue, nullptr>' is deprecated: Use SYCL 2020 sycl::make_queue free function}}
  auto Q = ext::oneapi::level_zero::make<queue>(
      Context, ZeQueue, ext::oneapi::level_zero::ownership::keep);
  // expected-warning@+1 {{'make<sycl::event, nullptr>' is deprecated: Use SYCL 2020 sycl::make_event free function}}
  auto E = ext::oneapi::level_zero::make<event>(
      Context, ZeEvent, ext::oneapi::level_zero::ownership::keep);

  return 0;
}
