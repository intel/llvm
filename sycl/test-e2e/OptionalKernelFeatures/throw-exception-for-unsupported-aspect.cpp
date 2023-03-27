// REQUIRES: cpu

// RUN: %clangxx -fsycl -O0 %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out

#include <sycl/sycl.hpp>

[[sycl::device_has(sycl::aspect::gpu)]] void foo() {}

int main() {
  sycl::queue q;
  try {
    q.submit([&](sycl::handler &h) { h.single_task([=]() { foo(); }); });
    q.wait();
  } catch (sycl::exception &e) {
    const char *ErrMsg = "Required aspect gpu is not supported on the device";
    if (std::string(e.what()).find(ErrMsg) != std::string::npos) {
      return 0;
    }
  }
  return 1;
}
