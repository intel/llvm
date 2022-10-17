// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.event_list.out
// RUN: %CPU_RUN_PLACEHOLDER %t.event_list.out
// RUN: env SYCL_PI_LEVEL_ZERO_USE_MULTIPLE_COMMANDLIST_BARRIERS=1 %GPU_RUN_PLACEHOLDER %t.event_list.out
// RUN: %ACC_RUN_PLACEHOLDER %t.event_list.out
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -DUSE_QUEUE_WIDE_BARRIER %s -o %t.queue_wide.out
// RUN: %CPU_RUN_PLACEHOLDER %t.queue_wide.out
// RUN: env SYCL_PI_LEVEL_ZERO_USE_MULTIPLE_COMMANDLIST_BARRIERS=1 %GPU_RUN_PLACEHOLDER %t.queue_wide.out
// RUN: %ACC_RUN_PLACEHOLDER %t.queue_wide.out
//
// Tests that barriers block all following execution on queues with active work.
// For L0 we currently need to set
// SYCL_PI_LEVEL_ZERO_USE_MULTIPLE_COMMANDLIST_BARRIERS to enable fix on certain
// hardware.

#include <sycl/sycl.hpp>

constexpr size_t NElemsPerSplit = 1000;
constexpr size_t Splits = 8;
constexpr size_t NElems = NElemsPerSplit * Splits;
constexpr size_t InnerIters = 90000;

int main(int argc, char **argv) {
  sycl::queue Q;

  float *ThrowawayDevMem = sycl::malloc_device<float>(NElems, Q);
  int *DevMem = sycl::malloc_device<int>(NElems, Q);
  int *HostMem = sycl::malloc_host<int>(NElems, Q);
  Q.fill<int>(DevMem, 0, NElems);
  Q.fill<int>(HostMem, 0, NElems);
  Q.wait();

  std::vector<sycl::event> Events(Splits);
  for (size_t I = 0; I < Splits; ++I) {
    Events[I] = Q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(NElemsPerSplit, [=](sycl::id<1> Idx) {
        int X = 0;
        float Y = 1.0f;
        while (X < InnerIters) {
          ++X;
          Y = sycl::tan(Y);
        }
        int AdjustedIdx = Idx + I * NElemsPerSplit;
        ThrowawayDevMem[AdjustedIdx] = Y;
        DevMem[AdjustedIdx] = X + AdjustedIdx;
      });
    });
  }

#ifdef USE_QUEUE_WIDE_BARRIER
  Q.ext_oneapi_submit_barrier();
#else
  Q.ext_oneapi_submit_barrier(Events);
#endif

  Q.memcpy(HostMem, DevMem, NElems * sizeof(int));
  Q.wait();

  int Result = 0;
  for (int I = 0; I < NElems; I++) {
    if (HostMem[I] != InnerIters + I) {
      std::cout << "Failed at " << I << " with " << HostMem[I] << std::endl;
      Result = 1;
      break;
    }
  }

  sycl::free(ThrowawayDevMem, Q);
  sycl::free(DevMem, Q);
  sycl::free(HostMem, Q);

  return Result;
}
