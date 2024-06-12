// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// UNSUPPORTED: cuda || hip

#include <sycl/detail/core.hpp>

#include <cstdlib>
#include <iostream>
#include <vector>

static uint64_t RandNext[64];

int simple_rand_host(size_t rand_idx) {
  uint64_t x = RandNext[rand_idx];
  x ^= x >> 12;
  x ^= x << 25;
  x ^= x >> 27;
  RandNext[rand_idx] = x;
  return static_cast<int>((x * 0x2545F4914F6CDD1Dul) >> 32) & 0x7fffffff;
}

int main() {
  // Size of the vectors
  const size_t N = 64;

  std::vector<int[4]> C(N);

  // Create a SYCL queue to enqueue work to
  sycl::queue Q;
  {
    // Create buffers for the vectors
    sycl::buffer<int[4], 1> bufferC(C.data(), sycl::range<1>(N));

    // Submit a command group to the queue
    Q.submit([&](sycl::handler &cgh) {
      // Get access to the buffers
      auto accessC = bufferC.get_access<sycl::access::mode::write>(cgh);

      // Execute the kernel
      cgh.parallel_for<class vector_addition>(
          sycl::range<1>(N), [=](sycl::id<1> idx) {
            srand(idx[0] + 1);
            for (size_t jdx = 0; jdx < 4; ++jdx)
              accessC[idx][jdx] = rand();
          });
    });
  }
  // Wait for the queue to finish
  Q.wait();

  for (size_t idx = 0; idx < N; ++idx)
    RandNext[idx] = 1 + idx;

  for (size_t idx = 0; idx < N; ++idx) {
    for (size_t jdx = 0; jdx < 4; ++jdx) {
      if (C[idx][jdx] != simple_rand_host(idx)) {
        std::cout << "work item: " << idx << " failed at iter: " << jdx
                  << std::endl;
        abort();
      }
    }
  }

  std::cout << "Pass!" << std::endl;
  return 0;
}
