// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test checks thread-safety of sycl::event's native handle data member.
// To do that we create a host task and a kernel task which depends on the host
// task. After submissions we yield in the main thread to let the host task to
// work and result in creation of kernel event's handle and start checking the
// status of the kernel event in a loop to catch the moment when handle is
// modified. If read and modification of sycl::event's handle is not thread-safe
// then this results in a segfault.

#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>
#include <thread>

int main() {
  // Create a SYCL queue
  sycl::queue queue;
  if (!queue.get_device().has(sycl::aspect::usm_shared_allocations))
    return 0;

  // Define the size of the buffers
  static constexpr size_t size = 1024;

  // Allocate USM memory for source and destination buffers
  int *src = sycl::malloc_shared<int>(size, queue);
  int *dst = sycl::malloc_shared<int>(size, queue);

  // Initialize the source buffer with some data
  for (size_t i = 0; i < size; ++i) {
    src[i] = i;
  }

  auto host_task_event = queue.submit([&](sycl::handler &cgh) {
    cgh.host_task([=]() {
      // Do some work in the host task
      std::cout << "Host task is executing." << std::endl;
      memcpy(dst, src, size * sizeof(int));
      std::cout << "Host task completed." << std::endl;
    });
  });

  sycl::event kernel_event = queue.submit([&](sycl::handler &cgh) {
    cgh.depends_on(host_task_event);
    cgh.memcpy(dst, src, size * sizeof(int));
  });

  // Let host task thread to work which will result in kernel_event's handle to
  // be created at some random moment.
  std::this_thread::yield();
  // Use number of iterations large enough to catch the moment when handle is
  // modifed.
  for (int i = 0; i < 100000; i++) {
    std::ignore =
        kernel_event.get_info<sycl::info::event::command_execution_status>();
  }

  kernel_event.wait();

  // Free the USM memory
  sycl::free(src, queue);
  sycl::free(dst, queue);
  return 0;
}
