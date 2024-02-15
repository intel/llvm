// REQUIRES: gpu, level_zero

// Flaky failure on windows
// UNSUPPORTED: windows

// RUN: %{build} %level_zero_options %threads_lib -o %t.out
// RUN: env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck --check-prefixes=CHECK-ONE-CMDLIST %s
// RUN: env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=2 SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck --check-prefixes=CHECK-PER-THREAD-CMDLIST %s

// The test checks that immediate commandlists are created per-thread.
// One immediate commandlist is created for device init, the rest for the queue.

// CHECK-ONE-CMDLIST: zeCommandListCreateImmediate = 2
// CHECK-PER-THREAD-CMDLIST: zeCommandListCreateImmediate = 4

#include <sycl/sycl.hpp>
#include <thread>

using namespace sycl;

bool results[3];

bool run_sample_kernel(queue Queue, int n) {
  // Creating buffer of 4 ints to be used inside the kernel code
  buffer<cl_int, 1> Buffer(4);

  // Size of index space for kernel
  range<1> NumOfWorkItems{Buffer.size()};

  // Submitting command group(work) to queue
  Queue.submit([&](handler &cgh) {
    // Getting write only access to the buffer on a device
    accessor Accessor = {Buffer, cgh, write_only};
    // Executing kernel
    cgh.parallel_for<class FillBuffer>(NumOfWorkItems, [=](id<1> WIid) {
      // Fill buffer with indexes
      Accessor[WIid] = (cl_int)WIid.get(0);
    });
  });

  // Getting read only access to the buffer on the host.
  // Implicit barrier waiting for queue to complete the work.
  const host_accessor HostAccessor = {Buffer, read_only};

  // Check the results
  bool MismatchFound = false;
  for (size_t I = 0; I < Buffer.size(); ++I) {
    if (HostAccessor[I] != I) {
      std::cout << "The result is incorrect for element: " << I
                << " , expected: " << I << " , got: " << HostAccessor[I]
                << std::endl;
      MismatchFound = true;
    }
  }

  if (!MismatchFound) {
    std::cout << "The results are correct!" << std::endl;
  }

  return MismatchFound;
}

void run_sample(queue Queue, int n) {
  results[n] = false;
  for (int i = 0; i < 5; i++)
    results[n] |= run_sample_kernel(Queue, n);
}

int main() {

  // Creating SYCL queue
  queue Queue;

  // Create one queue
  auto D = Queue.get_device();
  const char *devType = D.is_cpu() ? "CPU" : "GPU";
  std::string pluginName = D.get_platform().get_info<info::platform::name>();
  std::cout << "Running on device " << devType << " ("
            << D.get_info<info::device::name>() << ") " << pluginName
            << " plugin\n";

  // Use queue in multiple threads
  std::thread T1(run_sample, Queue, 0);
  std::thread T2(run_sample, Queue, 1);
  std::thread T3(run_sample, Queue, 2);

  T1.join();
  T2.join();
  T3.join();

  return (results[0] || results[1] || results[2]) ? 1 : 0;
}
