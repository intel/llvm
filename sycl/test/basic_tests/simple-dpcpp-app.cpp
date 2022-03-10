// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %RUN_ON_HOST %t.out

// Simple DPCPP application example
#include <CL/sycl.hpp>

using namespace cl::sycl;

int main() {
  // Create a buffer of 4 ints to be used inside the kernel code.
  buffer<int, 1> Buffer(4);

  // Create a simple asynchronous exception handler.
  auto AsyncHandler = [](exception_list ExceptionList) {
    for (auto &Exception : ExceptionList) {
      std::rethrow_exception(Exception);
    }
  };

  // Create a SYCL queue.
  queue Queue(AsyncHandler);

  // Size of index space for kernel.
  range<1> NumOfWorkItems{Buffer.size()};

  // Submit command group(work) to queue.
  Queue.submit([&](handler &cgh) {
    // Get write only access to the buffer on a device.
    auto Accessor = Buffer.get_access<access::mode::write>(cgh);
    // Execute kernel.
    cgh.parallel_for<class FillBuffer>(NumOfWorkItems, [=](id<1> WIid) {
      // Fill buffer with indices.
      Accessor[WIid] = static_cast<int>(WIid.get(0));
    });
  });

  // Get read only access to the buffer on the host.
  // This introduces an implicit barrier which blocks execution until the
  // command group above completes.
  const auto HostAccessor = Buffer.get_access<access::mode::read>();

  // Check the results.
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
