// REQUIRES: hip
// RUN: %clangxx %fsycl-host-only -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s -o %t.out
// RUN: %clangxx %fsycl-host-only -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note -D__SYCL_INTERNAL_API %s -o %t.out
//
/// Also test the experimental HIP interop interface
// RUN: %clangxx %fsycl-host-only -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note -DSYCL_EXT_ONEAPI_BACKEND_HIP_EXPERIMENTAL %s -o %t.out
// RUN: %clangxx %fsycl-host-only -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note -D__SYCL_INTERNAL_API -DSYCL_EXT_ONEAPI_BACKEND_HIP_EXPERIMENTAL %s -o %t.out
// expected-no-diagnostics

// Test for legacy and experimental HIP interop API

#ifdef SYCL_EXT_ONEAPI_BACKEND_HIP_EXPERIMENTAL
#include <sycl/ext/oneapi/experimental/backend/hip.hpp>
#endif

#include <sycl/sycl.hpp>

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
  hip_context = get_native<backend::ext_oneapi_hip>(Context);
  hip_event = get_native<backend::ext_oneapi_hip>(Event);
  hip_queue = get_native<backend::ext_oneapi_hip>(Queue);

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

#ifdef SYCL_EXT_ONEAPI_BACKEND_HIP_EXPERIMENTAL
  backend_input_t<backend::ext_oneapi_hip, device> InteropDeviceInput{
      hip_device};
  device InteropDevice =
      make_device<backend::ext_oneapi_hip>(InteropDeviceInput);

  backend_input_t<backend::ext_oneapi_hip, context> InteropContextInput{
      hip_context[0]};
  context InteropContext =
      make_context<backend::ext_oneapi_hip>(InteropContextInput);
  event InteropEvent = make_event<backend::ext_oneapi_hip>(hip_event, Context);

  queue InteropQueue = make_queue<backend::ext_oneapi_hip>(hip_queue, Context);
#endif

  return 0;
}
