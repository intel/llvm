// Source for saxpy.spv
// Compiled using dpcpp: clang++ saxpy.cpp -fsycl -o saxpy.cpp.out
// Extracted using: clang-offload-extract saxpy.cpp.out

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

int main() {
  size_t array_size = 16;

  sycl::queue sycl_queue;
  uint32_t *X = sycl::malloc_device<uint32_t>(array_size, sycl_queue);
  uint32_t *Z = sycl::malloc_device<uint32_t>(array_size, sycl_queue);

  sycl_queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<class saxpy>(sycl::range<1>{array_size},
                                  [=](sycl::item<1> itemId) {
                                    constexpr uint32_t A = 2;
                                    Z[itemId] = X[itemId] * A + Z[itemId];
                                  });
  });
  return 0;
}
