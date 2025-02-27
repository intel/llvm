#include <cstddef>

static size_t kernelRetains = 0;
static size_t queueRetains = 0;
static size_t memObjectRetains = 0;
static size_t contextRetains = 0;
static size_t deviceRetains = 0;
static size_t eventRetains = 0;

extern "C" {
using handle = size_t *;
using cl_int = int;

int clRetainKernel(handle) {
  kernelRetains++;
  return 0;
}
int clRetainCommandQueue(handle) {
  queueRetains++;
  return 0;
}
int clRetainMemObject(handle) {
  memObjectRetains++;
  return 0;
}
int clRetainContext(handle) {
  contextRetains++;
  return 0;
}
int clRetainDevice(handle) {
  deviceRetains++;
  return 0;
}
int clRetainEvent(handle) {
  eventRetains++;
  return 0;
}
}

// This function is a no-op and can be used to force the fake OpenCL library to
// be linked by calling it
namespace sycl {
namespace _V1 {
namespace unittest {
void loadMockOpenCL() {}
} // namespace unittest
} // namespace _V1
} // namespace sycl

size_t mockOpenCLNumKernelRetains() { return kernelRetains; }

size_t mockOpenCLNumQueueRetains() { return queueRetains; }

size_t mockOpenCLNumMemObjectRetains() { return memObjectRetains; }

size_t mockOpenCLNumContextRetains() { return contextRetains; }

size_t mockOpenCLNumDeviceRetains() { return deviceRetains; }

size_t mockOpenCLNumEventRetains() { return deviceRetains; }
