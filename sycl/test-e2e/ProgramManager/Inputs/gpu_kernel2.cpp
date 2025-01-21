#include "inc.hpp"

void gpu_bar(sycl::queue &queue, int *buf) {
  queue.submit([&](sycl::handler &h) {
    h.single_task<KernelTest2>([=]() { buf[0] = buf[1] + 9; });
  });
}
