#include <CL/sycl/detail/pi.hpp>
#include "CL/opencl.h"

namespace cl {
namespace sycl {
namespace detail {

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// Convinience macro makes source code search easier
#define OCL(pi_api) ocl_##pi_api

// Example of a PI interface that does not map exactly to an OpenCL one.
pi_result OCL(piPlatformsGet)(pi_uint32      num_entries,
                              pi_platform *  platforms,
                              pi_uint32 *    num_platforms) {
  cl_int result =
    clGetPlatformIDs(pi_cast<cl_uint>           (num_entries),
                     pi_cast<cl_platform_id *>  (platforms),
                     pi_cast<cl_uint *>         (num_platforms));

  // Absorb the CL_PLATFORM_NOT_FOUND_KHR and just return 0 in num_platforms
  if (result == CL_PLATFORM_NOT_FOUND_KHR) {
    pi_assert(num_platforms != 0);
    *num_platforms = 0;
    result = CL_SUCCESS;
  }
  return pi_cast<pi_result>(result);
}


// Example of a PI interface that does not map exactly to an OpenCL one.
pi_result OCL(piDevicesGet)(pi_platform      platform,
                            pi_device_type   device_type,
                            pi_uint32        num_entries,
                            pi_device *      devices,
                            pi_uint32 *      num_devices) {
  cl_int result =
    clGetDeviceIDs(pi_cast<cl_platform_id> (platform),
                   pi_cast<cl_device_type> (device_type),
                   pi_cast<cl_uint>        (num_entries),
                   pi_cast<cl_device_id *> (devices),
                   pi_cast<cl_uint *>      (num_devices));

  // Absorb the CL_DEVICE_NOT_FOUND and just return 0 in num_devices
  if (result == CL_DEVICE_NOT_FOUND) {
    pi_assert(num_devices != 0);
    *num_devices = 0;
    result = CL_SUCCESS;
  }
  return pi_cast<pi_result>(result);
}

pi_result OCL(piextDeviceSelectBinary)(
  pi_device           device, // TODO: does this need to be context?
  pi_device_binary *  images,
  pi_uint32           num_images,
  pi_device_binary *  selected_image) {

  // TODO dummy implementation.
  // Real implementaion will use the same mechanism OpenCL ICD dispatcher
  // uses. Somthing like:
  //   PI_VALIDATE_HANDLE_RETURN_HANDLE(ctx, PI_INVALID_CONTEXT);
  //     return context->dispatch->piextDeviceSelectIR(
  //       ctx, images, num_images, selected_image);
  // where context->dispatch is set to the dispatch table provided by PI
  // plugin for platform/device the ctx was created for.

  *selected_image = num_images > 0 ? images[0] : nullptr;
  return PI_SUCCESS;
}

// TODO: implement portable call forwarding (ifunc is a GNU extension).
// TODO: reuse same PI -> OCL mapping in pi_opencl.hpp, or maybe just
//       wait until that one is completely removed.
//
#define PI_ALIAS(pi_api, ocl_api)           \
static void *__resolve_##pi_api(void) {       \
  return (void*) (ocl_api);                 \
}                                           \
decltype(ocl_api) OCL(pi_api) __attribute__((ifunc ("__resolve_" #pi_api)));

// Platform
PI_ALIAS(piPlatformGetInfo,   clGetPlatformInfo)
// Device
PI_ALIAS(piDeviceRetain,      clRetainDevice)
PI_ALIAS(piDeviceRelease,     clReleaseDevice)
PI_ALIAS(piDevicePartition,   clCreateSubDevices)
PI_ALIAS(piDeviceGetInfo,     clGetDeviceInfo)

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

} // namespace detail
} // namespace sycl
} // namespace cl

