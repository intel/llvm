// REQUIRES: cuda
// RUN: %clangxx %fsycl-host-only -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s -o %t.out
// RUN: %clangxx %fsycl-host-only -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note -D__SYCL_INTERNAL_API %s -o %t.out

// Test for experimental CUDA interop API

#define SYCL_EXT_ONEAPI_BACKEND_CUDA_EXPERIMENTAL 1
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/backend/cuda.hpp>

#include <cuda.h>

using namespace sycl;

//
// 4.5.1 SYCL application interoperability may be provided for
// platform,
// device,
// context,
// queue,
// event,
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

  // 4.5.1.1 For each SYCL runtime class T which supports SYCL application
  // interoperability with the SYCL backend, a specialization of return_type
  // must be defined as the type of SYCL application interoperability native
  // backend object associated with T for the SYCL backend, specified in the
  // SYCL backend specification.
  //
  // return_type is used when retrieving the backend specific native object from
  // a SYCL object. See the relevant backend specification for details.

  backend_traits<backend::ext_oneapi_cuda>::return_type<platform> cu_platform;
  backend_traits<backend::ext_oneapi_cuda>::return_type<device> cu_device;
  backend_traits<backend::ext_oneapi_cuda>::return_type<context> cu_context;
  backend_traits<backend::ext_oneapi_cuda>::return_type<queue> cu_queue;
  backend_traits<backend::ext_oneapi_cuda>::return_type<event> cu_event;

  // 4.5.1.2 For each SYCL runtime class T which supports SYCL application
  // interoperability, a specialization of get_native must be defined, which
  // takes an instance of T and returns a SYCL application interoperability
  // native backend object associated with syclObject which can be used for SYCL
  // application interoperability. The lifetime of the object returned are
  // backend-defined and specified in the backend specification.

  cu_platform = get_native<backend::ext_oneapi_cuda>(Platform);
  cu_device = get_native<backend::ext_oneapi_cuda>(Device);
  cu_context = get_native<backend::ext_oneapi_cuda>(Context);
  cu_queue = get_native<backend::ext_oneapi_cuda>(Queue);
  cu_event = get_native<backend::ext_oneapi_cuda>(Event);

  // Check deprecated
  // expected-warning@+2 {{'get_native' is deprecated: Use SYCL 2020 sycl::get_native free function}}
  // expected-warning@+1 {{'get_native<sycl::backend::ext_oneapi_cuda>' is deprecated: Use SYCL 2020 sycl::get_native free function}}
  cu_platform = Platform.get_native<backend::ext_oneapi_cuda>();
  // expected-warning@+2 {{'get_native' is deprecated: Use SYCL 2020 sycl::get_native free function}}
  // expected-warning@+1 {{'get_native<sycl::backend::ext_oneapi_cuda>' is deprecated: Use SYCL 2020 sycl::get_native free function}}
  cu_device = Device.get_native<backend::ext_oneapi_cuda>();
  // expected-warning@+2 {{'get_native' is deprecated: Use SYCL 2020 sycl::get_native free function}}
  // expected-warning@+1 {{'get_native<sycl::backend::ext_oneapi_cuda>' is deprecated: Use SYCL 2020 sycl::get_native free function}}
  cu_context = Context.get_native<backend::ext_oneapi_cuda>();
  // expected-warning@+2 {{'get_native' is deprecated: Use SYCL 2020 sycl::get_native free function}}
  // expected-warning@+1 {{'get_native<sycl::backend::ext_oneapi_cuda>' is deprecated: Use SYCL 2020 sycl::get_native free function}}
  cu_queue = Queue.get_native<backend::ext_oneapi_cuda>();
  // expected-warning@+2 {{'get_native' is deprecated: Use SYCL 2020 sycl::get_native free function}}
  // expected-warning@+1 {{'get_native<sycl::backend::ext_oneapi_cuda>' is deprecated: Use SYCL 2020 sycl::get_native free function}}
  cu_event = Event.get_native<backend::ext_oneapi_cuda>();

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

  backend_input_t<backend::ext_oneapi_cuda, platform> InteropPlatformInput{cu_platform};
  platform InteropPlatform = make_platform<backend::ext_oneapi_cuda>(InteropPlatformInput);

  backend_input_t<backend::ext_oneapi_cuda, device> InteropDeviceInput{cu_device};
  device InteropDevice = make_device<backend::ext_oneapi_cuda>(InteropDeviceInput);

  backend_input_t<backend::ext_oneapi_cuda, context> InteropContextInput{cu_context[0]};
  context InteropContext = make_context<backend::ext_oneapi_cuda>(InteropContextInput);

  queue InteropQueue = make_queue<backend::ext_oneapi_cuda>(cu_queue, Context);
  event InteropEvent = make_event<backend::ext_oneapi_cuda>(cu_event, Context);

  return 0;
}
