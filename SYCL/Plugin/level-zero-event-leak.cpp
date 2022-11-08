// REQUIRES: level_zero, level_zero_dev_kit
// TODO: ZE_DEBUG=4 produces no output on Windows. Enable when fixed.
// UNSUPPORTED: windows
//
// RUN: %clangxx -fsycl -fsycl-unnamed-lambda -fsycl-targets=%sycl_triple %level_zero_options %s -o %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR="level_zero:*" ZE_DEBUG=4 %GPU_RUN_PLACEHOLDER %t.out wait 2>&1 %GPU_CHECK_PLACEHOLDER
// RUN: env ONEAPI_DEVICE_SELECTOR="level_zero:*" ZE_DEBUG=4 %GPU_RUN_PLACEHOLDER %t.out nowait 2>&1 %GPU_CHECK_PLACEHOLDER
//
// RUN: %clangxx -fsycl -fsycl-unnamed-lambda -fsycl-targets=%sycl_triple %level_zero_options %s -DCHECK_INORDER -o %t.inorder.out
// RUN: env ONEAPI_DEVICE_SELECTOR="level_zero:*" ZE_DEBUG=4 %GPU_RUN_PLACEHOLDER %t.inorder.out wait 2>&1 %GPU_CHECK_PLACEHOLDER
// RUN: env ONEAPI_DEVICE_SELECTOR="level_zero:*" ZE_DEBUG=4 %GPU_RUN_PLACEHOLDER %t.inorder.out nowait 2>&1 %GPU_CHECK_PLACEHOLDER
//
// CHECK-NOT: LEAK

// The test is to check that there are no leaks reported with the embedded
// ZE_DEBUG=4 testing capability. Example of a leak reported is this:
//
// clang-format off
// ZE_DEBUG=4: check balance of create/destroy calls
// ----------------------------------------------------------
//               zeContextCreate = 1     \--->              zeContextDestroy = 1
//          zeCommandQueueCreate = 1     \--->         zeCommandQueueDestroy = 0     ---> LEAK = 1
//                zeModuleCreate = 1     \--->               zeModuleDestroy = 0     ---> LEAK = 1
//                zeKernelCreate = 1     \--->               zeKernelDestroy = 0     ---> LEAK = 1
//             zeEventPoolCreate = 1     \--->            zeEventPoolDestroy = 1
//  zeCommandListCreateImmediate = 1     |
//           zeCommandListCreate = 2     \--->          zeCommandListDestroy = 2     ---> LEAK = 1
//                 zeEventCreate = 129   \--->                zeEventDestroy = 1     ---> LEAK = 128
//                 zeFenceCreate = 2     \--->                zeFenceDestroy = 0     ---> LEAK = 2
//                 zeImageCreate = 0     \--->                zeImageDestroy = 0
//               zeSamplerCreate = 0     \--->              zeSamplerDestroy = 0
//              zeMemAllocDevice = 0     |
//                zeMemAllocHost = 0     |
//              zeMemAllocShared = 0     \--->                     zeMemFree = 0
//
// clang-format on
//
// NOTE: The 1000 value below is to be larger than the "128" heuristic in
// queue_impl::addSharedEvent.

#include <sycl/sycl.hpp>

int main(int argc, char **argv) {
  assert(argc == 2 && "Invalid number of arguments");
  std::string use_queue_finish(argv[1]);

  bool Use = false;
  if (use_queue_finish == "wait") {
    Use = true;
    std::cerr << "Use queue::wait" << std::endl;
  } else if (use_queue_finish == "nowait") {
    std::cerr << "No wait. Ensure resources are released anyway" << std::endl;
  } else {
    assert(0 && "Unsupported parameter value");
  }

#ifdef CHECK_INORDER
  sycl::queue Q({sycl::property::queue::in_order{}});
#else
  sycl::queue Q;
#endif

  const unsigned n_chunk = 1000;
  for (int i = 0; i < n_chunk; i++)
    Q.single_task([=]() {});

  if (Use)
    Q.wait();

  return 0;
}
