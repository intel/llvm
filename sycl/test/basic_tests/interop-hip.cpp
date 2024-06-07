// REQUIRES: hip
// RUN: %clangxx %fsycl-host-only -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s
// RUN: %clangxx %fsycl-host-only -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note -D__SYCL_INTERNAL_API %s

// Test for HIP interop API

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/hip.hpp>

using namespace sycl;

//
// 4.5.1 SYCL application interoperability may be provided for
// platform,
// device,
// context,
// queue,
// event,
// buffer,
// device_image,
// sampled_image,
// unsampled_image.

int main() {

  // Create SYCL objects
  device Device;
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

  backend_traits<backend::ext_oneapi_hip>::return_type<device> hip_device;
  backend_traits<backend::ext_oneapi_hip>::return_type<context> hip_context;
  backend_traits<backend::ext_oneapi_hip>::return_type<event> hip_event;
  backend_traits<backend::ext_oneapi_hip>::return_type<queue> hip_queue;

  // 4.5.1.2 For each SYCL runtime class T which supports SYCL application
  // interoperability, a specialization of get_native must be defined, which
  // takes an instance of T and returns a SYCL application interoperability
  // native backend object associated with syclObject which can be used for SYCL
  // application interoperability. The lifetime of the object returned are
  // backend-defined and specified in the backend specification.

  hip_device = get_native<backend::ext_oneapi_hip>(Device);
  // expected-warning@+1{{'get_native<sycl::backend::ext_oneapi_hip, sycl::context>' is deprecated: Context interop is deprecated for HIP. If a native context is required, use hipDevicePrimaryCtxRetain with a native device}}
  hip_context = get_native<backend::ext_oneapi_hip>(Context);
  hip_event = get_native<backend::ext_oneapi_hip>(Event);
  hip_queue = get_native<backend::ext_oneapi_hip>(Queue);

  device InteropDevice = make_device<backend::ext_oneapi_hip>(hip_device);
  event InteropEvent = make_event<backend::ext_oneapi_hip>(hip_event, Context);
  queue InteropQueue = make_queue<backend::ext_oneapi_hip>(hip_queue, Context);

  return 0;
}
