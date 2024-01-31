// Test caching for JIT fused kernels when different devices are involved.

#include <sycl/sycl.hpp>

using namespace sycl;

constexpr inline std::size_t size(1024);

class Kernel0;
class Kernel1;

void performFusion(queue q, std::size_t *a, std::size_t *b) {
  {
    buffer<std::size_t> a_buf(a, size);
    buffer<std::size_t> b_buf(b, size);

    ext::codeplay::experimental::fusion_wrapper fw{q};
    fw.start_fusion();
    q.submit([&](handler &cgh) {
      accessor a(a_buf, cgh, write_only, no_init);
      cgh.parallel_for<Kernel0>(size, [=](id<1> i) { a[i] = i; });
    });
    q.submit([&](handler &cgh) {
      accessor a(a_buf, cgh, read_only);
      accessor b(b_buf, cgh, write_only, no_init);
      cgh.parallel_for<Kernel1>(size, [=](id<1> i) { b[i] = a[i] * 2; });
    });
    fw.complete_fusion();
  }
  for (std::size_t i = 0; i < size; ++i) {
    assert(a[i] == i && "WRONG a VALUE");
    assert(b[i] == i * 2 && "WRONG b VALUE");
  }
}

int main() {
  queue q{gpu_selector_v,
          ext::codeplay::experimental::property::queue::enable_fusion{}};
  queue q_cpu{cpu_selector_v,
              ext::codeplay::experimental::property::queue::enable_fusion{}};

  std::vector<std::size_t> a(size);
  std::vector<std::size_t> b(size);

  // Initial invocation
  performFusion(q, a.data(), b.data());

  // Identical invocation, should lead to JIT cache hit.
  performFusion(q, a.data(), b.data());

  // Invocation on CPU device.
  performFusion(q_cpu, a.data(), b.data());

  return 0;
}
