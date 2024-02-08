// REQUIRES: level_zero
//
// RUN: %{build} -o %t.out
//
// RUN: env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 SYCL_PI_LEVEL_ZERO_BATCH_SIZE=4 ONEAPI_DEVICE_SELECTOR='level_zero:*' UR_L0_LEAKS_DEBUG=1 %{run} %t.out wait  2>&1 | FileCheck %s
// RUN: env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 SYCL_PI_LEVEL_ZERO_BATCH_SIZE=4 ONEAPI_DEVICE_SELECTOR='level_zero:*' UR_L0_LEAKS_DEBUG=1 %{run} %t.out nowait 2>&1 | FileCheck %s
//
// CHECK-NOT: LEAK
//
// The test is to check that there are no leaks reported with the embedded
// UR_L0_LEAKS_DEBUG=1 testing capability.
// In addition to general leak checking, especially for discard_events, the test
// checks that piKernelRelease to be executed for each kernel call, and
// EventRelease for events, that are used for dependencies between
// command-lists.

#include <CL/sycl.hpp>
int main(int argc, char *argv[]) {
  assert(argc == 2 && "Invalid number of arguments");
  std::string use_queue_finish(argv[1]);

  bool use = false;
  if (use_queue_finish == "wait") {
    use = true;
    std::cerr << "Use queue::wait" << std::endl;
  } else if (use_queue_finish == "nowait") {
    std::cerr << "No wait. Ensure resources are released anyway" << std::endl;
  } else {
    assert(0 && "Unsupported parameter value");
  }

  sycl::property_list Props{
      sycl::property::queue::in_order{},
      sycl::ext::oneapi::property::queue::discard_events{}};
  sycl::queue q(Props);

  // test has multiple command-lists thanks to this loop and fixed batch size.
  for (size_t i = 0; i < 100; ++i)
    q.single_task<class test>([]() {});

  if (use)
    q.wait();

  std::cout << "The test passed." << std::endl;
  return 0;
}
