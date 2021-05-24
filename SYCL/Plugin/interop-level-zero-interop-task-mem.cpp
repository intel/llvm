// REQUIRES: level_zero, level_zero_dev_kit
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %level_zero_options %s -o %t.out
// RUN: env SYCL_BE=PI_LEVEL_ZERO %GPU_RUN_PLACEHOLDER %t.out

// Test fails on Level Zero on Linux
// UNSUPPORTED: level_zero && linux

// Test for Level Zero interop_task

#include <CL/sycl.hpp>
// clang-format off
#include <level_zero/ze_api.h>
#include <CL/sycl/backend/level_zero.hpp>
// clang-format on

class my_selector : public cl::sycl::device_selector {
public:
  int operator()(const cl::sycl::device &dev) const override {
    return (dev.get_platform().get_backend() == cl::sycl::backend::level_zero)
               ? 1
               : 0;
  }
};

int main() {
  sycl::queue sycl_queue = sycl::queue(my_selector());

  ze_context_handle_t ze_context =
      sycl_queue.get_context().get_native<sycl::backend::level_zero>();
  std::cout << "zeContextGetStatus = " << zeContextGetStatus(ze_context)
            << std::endl;

  auto buf = cl::sycl::buffer<uint8_t, 1>(1024);
  sycl_queue.submit([&](cl::sycl::handler &cgh) {
    auto acc = buf.get_access<cl::sycl::access::mode::write>(cgh);
    cgh.interop_task([&](const cl::sycl::interop_handler &ih) {
      void *device_ptr = ih.get_mem<sycl::backend::level_zero>(acc);
      ze_memory_allocation_properties_t memAllocProperties{};
      zeMemGetAllocProperties(ze_context, device_ptr, &memAllocProperties,
                              nullptr);
      std::cout << "Memory type = " << memAllocProperties.type << std::endl;
    });
  });

  return 0;
}
