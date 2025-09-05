// RUN: %clangxx -fsycl -fsycl-device-only -S %s -o /dev/null
// Test that the ESIMD Verifier doesn't error on locally defined types
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl::ext::intel::esimd;

constexpr size_t n = 64;

constexpr size_t size = 4;

class functor {
public:
  template <typename T> sycl::event operator()(sycl::queue &q, T *x) {
    return q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(n / size, [=](auto item) SYCL_ESIMD_KERNEL {
        size_t offset = item.get_id(0);
        simd<T, size> vec(offset);
        block_store(x + offset * size, vec);
      });
    });
  }
};

template <typename T> sycl::event func(sycl::queue &q, T *x) {
  return q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(n / size, [=](auto item) SYCL_ESIMD_KERNEL {
      size_t offset = item.get_id(0);
      simd<T, size> vec(offset);
      block_store(x + offset * size, vec);
    });
  });
}

int main() {
  sycl::queue q;
  float *x = sycl::malloc_shared<float>(n, q);
  auto event = func(q, x);
  event.wait();
  functor f;
  auto event2 = f(q, x);
  event2.wait();
  sycl::free(x, q);
  return 0;
}
