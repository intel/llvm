// REQUIRES: level_zero, level_zero_dev_kit
//
// RUN: %{build} %level_zero_options -o %t.out
// RUN: %{l0_leak_check} %{run} %t.out wait 2>&1 | FileCheck %s
// RUN: %{l0_leak_check} %{run} %t.out nowait 2>&1 | FileCheck %s
//
// RUN: %{build} %level_zero_options -DCHECK_INORDER -o %t.inorder.out
// RUN: %{l0_leak_check} %{run} %t.inorder.out wait 2>&1 | FileCheck %s
// RUN: %{l0_leak_check} %{run} %t.inorder.out nowait 2>&1 | FileCheck %s
//
// CHECK-NOT: LEAK

// The test is to check that there are no leaks reported with the embedded
// UR_L0_LEAKS_DEBUG=1 testing capability. Example of a leak reported is this:
//
// clang-format off
// Check balance of create/destroy calls
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
