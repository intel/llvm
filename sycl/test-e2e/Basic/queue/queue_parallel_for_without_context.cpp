// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/usm.hpp>

using namespace sycl; // (optional) avoids need for "sycl::" before SYCL names

int main() {
  //  Create a default queue to enqueue work to the default device
  queue myQueue;

  if (!(myQueue.get_device())
           .get_info<sycl::info::device::usm_shared_allocations>()) {
    return 0;
  }

  // Allocate shared memory bound to the device and context associated to the
  // queue.
  int *data = sycl::malloc_shared<int>(1024, myQueue);

  myQueue.parallel_for(1024, [=](id<1> idx) {
    // Initialize each buffer element with its own rank number starting at 0
    data[idx] = idx;
  }); // End of the kernel function

  // Explicitly wait for kernel execution since there is no accessor involved
  myQueue.wait();

  for (int i = 0; i < 1024; i++)
    assert(data[i] == i);

  sycl::free(data, myQueue);

  return 0;
}
