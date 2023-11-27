#ifndef KERNEL_BINARY_CACHING
#define KERNEL_BINARY_CACHING

#include <sycl/sycl.hpp>

using namespace sycl;

constexpr size_t dataSize = 512;

class Kernel0;
class Kernel1;
class Kernel2;

void performFusion(queue &q) {
  int a;
  int b;
  int c = 0;
  {
    sycl::buffer<int> a_buf(&a, 1);
    sycl::buffer<int> b_buf(&b, 1);
    sycl::buffer<int> c_buf(&c, 1);
    ext::codeplay::experimental::fusion_wrapper fw{q};
    fw.start_fusion();
    q.submit([&](handler &cgh) {
      accessor a(a_buf, cgh, write_only, no_init);
      cgh.parallel_for<Kernel0>(1, [=](id<1>) { a[0] = 1; });
    });
    q.submit([&](handler &cgh) {
      accessor b(b_buf, cgh, write_only, no_init);
      cgh.parallel_for<Kernel1>(1, [=](id<1>) { b[0] = 1; });
    });
    q.submit([&](handler &cgh) {
      accessor a(a_buf, cgh, read_only);
      accessor b(b_buf, cgh, read_only);
      accessor c(c_buf, cgh, write_only, no_init);
      cgh.parallel_for<Kernel2>(1, [=](id<1>) { c[0] = a[0] + b[0]; });
    });
    fw.complete_fusion();
  }
  assert(a == 1 && b == 1 && c == 2 && "COMPUTATION ERROR");
}

int main() {
  queue cpu_queue{
      cpu_selector_v,
      ext::codeplay::experimental::property::queue::enable_fusion{}};
  queue gpu_queue{
      gpu_selector_v,
      ext::codeplay::experimental::property::queue::enable_fusion{}};

  // Initial invocation for CPU device
  performFusion(cpu_queue);

  // Identical invocation, should lead to kernel binary cache hit.
  performFusion(cpu_queue);

  // Initial invocation for GPU device
  performFusion(gpu_queue);

  // Identical invocation, should lead to kernel binary cache hit.
  performFusion(gpu_queue);

  return 0;
}

#endif // KERNEL_BINARY_CACHING
