// UNSUPPORTED: target-nvidia,cuda
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/20109

// RUN: %{build} %O0 -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

#include <sycl/stream.hpp>

int main() {
  sycl::queue q;
  q.submit([&](sycl::handler &cgh) {
    sycl::stream os(1024, 256, cgh);
    cgh.single_task([=]() { os << "test"; });
  });
  q.wait();
  return 0;
}
