// REQUIRES: cpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

using namespace sycl;
using namespace sycl::access;

void test() {
  queue Q;
  std::shared_ptr<std::mutex> Mutex(new std::mutex());
  Mutex->lock();
  {
    auto E = Q.submit([&](handler &CGH) {
      CGH.host_task([=]() { std::lock_guard<std::mutex> Guard(*Mutex); });
    });
    // Host task should block kernel enqueue but not cause dead lock here
    Q.submit([&](handler &CGH) {
      CGH.depends_on(E);
      CGH.single_task([=]() {});
    });
  }
  // Unblock kernel execution
  Mutex->unlock();
  Q.wait();
}

int main() {
  test();
  return 0;
}
