// REQUIRES: level_zero, level_zero_dev_kit

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %level_zero_options %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include <CL/sycl.hpp>
#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>

int main() {
  using namespace sycl;

  queue Q{gpu_selector_v};

  auto nativeQ = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(Q);

  backend_input_t<backend::ext_oneapi_level_zero, queue> QueueInteropInput = {
      nativeQ, Q.get_device(), ext::oneapi::level_zero::ownership::keep};
  queue Q2 = make_queue<backend::ext_oneapi_level_zero>(QueueInteropInput,
                                                        Q.get_context());

  // Command submission works fine
  Q2.submit([&](handler &cgh) { cgh.single_task<class K>([]() {}); });

  // Check that copy submission also works
  int *hostMem = (int *)malloc_host(sizeof(int), Q2);
  int *devMem = malloc_device<int>(1, Q2);
  Q2.memcpy(hostMem, devMem, sizeof(int));
}
