// REQUIRES: arch-intel_gpu_mtl_h || arch-intel_gpu_mtl_u || arch-intel_gpu_lnl_m || arch-intel_gpu_ptl_h || arch-intel_gpu_ptl_u
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %if level_zero %{ env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out %}

#include <sycl/detail/core.hpp>

#include <cassert>
#include <cstdlib>

static constexpr int Iterations = 250;

static constexpr int Sizes[] = {1325, 246800}; // in number of floats
static constexpr int Delay = 2000;

static void runTest(sycl::queue &Q, int N) {
  float *x = static_cast<float *>(std::malloc(N * sizeof(float)));
  float *y = static_cast<float *>(std::malloc(N * sizeof(float)));
  assert(x && y);

  for (int iter = 0; iter < Iterations; ++iter) {
    for (int i = 0; i < N; ++i) {
      x[i] = static_cast<float>(iter * N + i);
      y[i] = 0.0f;
    }

    {
      sycl::buffer<float, 1> xBuf(x, sycl::range<1>(N));
      sycl::buffer<float, 1> yBuf(y, sycl::range<1>(N));

      // Kernel: y[i] = x[i].  A spin loop widens the race window between
      // this kernel and the map copy triggered by host_accessor below.
      Q.submit([&](sycl::handler &cgh) {
        sycl::accessor xAcc(xBuf, cgh, sycl::read_only);
        sycl::accessor yAcc(yBuf, cgh, sycl::write_only, sycl::no_init);
        cgh.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
          // Delay loop - avoid optimizing out by the compiler.
          float s = 1.0f + static_cast<float>(i[0] & 0x3ff);
          for (int d = 0; d < Delay; ++d)
            s = s * 0.5f + 0.5f;
          yAcc[i] = xAcc[i] + (s > 0.5f ? 0.0f : 1.0f);
        });
      });

      // host_accessor triggers appendMemBufferMap with the kernel event
      // as a dependency.  Without either fix the map copy races the kernel.
      sycl::host_accessor yHost(yBuf, sycl::read_only);
      for (int i = 0; i < N; ++i) {
        assert(yHost[i] == x[i] &&
               "host_accessor read stale data — map did not wait for kernel");
      }
    }

    // After the buffer destructor writes back to y[], verify the write-back
    // path (appendMemUnmap) also correctly ordered the copy.
    for (int i = 0; i < N; ++i) {
      assert(y[i] == x[i] &&
             "write-back to host ptr is stale — unmap did not wait");
    }
  }

  std::free(x);
  std::free(y);
}

int main() {
  sycl::queue Q;
  for (int N : Sizes)
    runTest(Q, N);
  return 0;
}
