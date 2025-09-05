#include "inc.hpp"

void gpu_foo(sycl::queue &queue, int *buf) {
  queue.submit([&](sycl::handler &h) {
    h.single_task<KernelTest1>([=]() { buf[0] = buf[1] - 3; });
  });
}
