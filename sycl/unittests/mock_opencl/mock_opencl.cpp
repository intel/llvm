#include <cstddef>

static size_t kernelRetains = 0;
static size_t queueRetains = 0;
static size_t memObjectRetains = 0;
static size_t contextRetains = 0;
static size_t deviceRetains = 0;
static size_t eventRetains = 0;

#if _MSC_VER
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

extern "C" {
using handle = size_t *;
using cl_int = int;

EXPORT int clRetainKernel(handle) {
  kernelRetains++;
  return 0;
}
EXPORT int clRetainCommandQueue(handle) {
  queueRetains++;
  return 0;
}
EXPORT int clRetainMemObject(handle) {
  memObjectRetains++;
  return 0;
}
EXPORT int clRetainContext(handle) {
  contextRetains++;
  return 0;
}
EXPORT int clRetainDevice(handle) {
  deviceRetains++;
  return 0;
}
EXPORT int clRetainEvent(handle) {
  eventRetains++;
  return 0;
}
}

// This function is a no-op and can be used to force the fake OpenCL library to
// be linked by calling it
namespace sycl {
namespace _V1 {
namespace unittest {
EXPORT void loadMockOpenCL() {}
} // namespace unittest
} // namespace _V1
} // namespace sycl

EXPORT size_t mockOpenCLNumKernelRetains() { return kernelRetains; }

EXPORT size_t mockOpenCLNumQueueRetains() { return queueRetains; }

EXPORT size_t mockOpenCLNumMemObjectRetains() { return memObjectRetains; }

EXPORT size_t mockOpenCLNumContextRetains() { return contextRetains; }

EXPORT size_t mockOpenCLNumDeviceRetains() { return deviceRetains; }

EXPORT size_t mockOpenCLNumEventRetains() { return deviceRetains; }
