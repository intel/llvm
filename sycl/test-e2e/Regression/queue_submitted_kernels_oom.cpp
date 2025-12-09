// Tests for memory use of kernel submission. Should not grow
// unbounded even with thousands of kernel submissions.
// Only intended for the new L0v2 adapter.
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// REQUIRES: linux && level_zero_v2_adapter

#include <array>
#include <cassert>
#include <cstdint>
#include <sys/resource.h>
#include <thread>
#include <vector>

#include <sycl/atomic_ref.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

static long getRusageKbs() {
  struct rusage r_usage;
  if (getrusage(RUSAGE_SELF, &r_usage) == 0) {
    return r_usage.ru_maxrss;
  }
  return -1;
}

// There's some variability in memory usage on the various
// platforms when running kernels.
static constexpr long MarginKb = 400;

static bool withinMargin(long base, long current) {
  if (base < 0 || current < 0)
    return false;
  // theoretically, memory use can shrink after e.g., wait()...
  long diff = (current > base) ? (current - base) : (base - current);
  return diff <= MarginKb;
}

static constexpr size_t UniqueKernels = 256;
static constexpr size_t ConsecutiveDupSubmissions =
    100000;                                         // same kernel over and over
static constexpr size_t CyclicSubmissions = 100000; // cycle over small subset
static constexpr size_t CyclicSubset = 16;          // cycle kernel subset
static constexpr size_t AllKernelsSubmissions = 100000; // running all kernel

template <size_t ID> struct KernelTag;

template <size_t ID> static void submitIncrement(sycl::queue &Q, int *accum) {
  Q.submit([&](sycl::handler &CGH) {
    CGH.single_task<KernelTag<ID>>([=]() {
      // atomic_ref to avoid data races while we spam submissions.
      sycl::atomic_ref<int, sycl::memory_order::relaxed,
                       sycl::memory_scope::device>
          ref(accum[ID]);
      ref.fetch_add(1);
    });
  });
}

using SubmitFn = void (*)(sycl::queue &, int *);

template <std::size_t... Is>
static auto makeFnTable(std::index_sequence<Is...>) {
  return std::array<SubmitFn, UniqueKernels>{&submitIncrement<Is>...};
}

int main() {
  bool rusageUnsupported = getRusageKbs() == -1;
  if (rusageUnsupported) {
    return 1; // can't collect mem statistics, no point in running the test.
  }

  sycl::queue Q;

  int *accum = sycl::malloc_shared<int>(UniqueKernels, Q);
  assert(accum && "USM alloc failed");
  for (std::size_t i = 0; i < UniqueKernels; ++i)
    accum[i] = 0;

  std::vector<std::size_t> expected(UniqueKernels, 0);

  auto fns = makeFnTable(std::make_index_sequence<UniqueKernels>{});

  // Submit the same kernel over and over again. The submitted kernel
  // vector shouldn't grow at all, since we do a lookback over
  // a few previous kernels.
  auto runDuplicates = [&]() {
    for (size_t i = 0; i < ConsecutiveDupSubmissions; ++i) {
      fns[0](Q, accum);
      expected[0]++;
    }
  };

  // Run a small subset of kernels in a loop. Likely the most realistic
  // scenario. Should be mostly absorbed by loopback duplicate search, and,
  // possibliy, compaction.
  auto runCyclical = [&]() {
    for (size_t i = 0; i < CyclicSubmissions; ++i) {
      size_t id = i % CyclicSubset;
      fns[id](Q, accum);
      expected[id]++;
    }
  };

  // Run all kernels in the loop. Should dynamically adjust the
  // threshold for submitted kernels.
  auto runAll = [&]() {
    for (size_t i = 0; i < AllKernelsSubmissions; ++i) {
      size_t id = i % UniqueKernels;
      fns[id](Q, accum);
      expected[id]++;
    }
  };

  runAll();
  Q.wait(); // first run all the kernels, just to get all the caches warm.

  long baseMemUsage = getRusageKbs();

  // Run from small kernel variety, to large, to small, to test dynamic
  // threshold changes.
  runDuplicates();
  runCyclical();
  runAll();

  long afterRampup = getRusageKbs();

  assert(withinMargin(baseMemUsage, afterRampup));

  Q.wait(); // this clears the submitted kernels list, allowing the threshold to
            // lower.
  runAll();
  runCyclical();
  runDuplicates();

  long afterRampdown = getRusageKbs();
  assert(withinMargin(baseMemUsage, afterRampdown));

  Q.wait(); // this clears vector again. But memory usage should stay the same.
  long afterCleanup = getRusageKbs();
  assert(withinMargin(baseMemUsage, afterCleanup));

  int ret = 0;
  for (std::size_t i = 0; i < UniqueKernels; ++i) {
    if (static_cast<std::size_t>(accum[i]) != expected[i]) {
      ret = 0;
      std::cout << "fail: " << accum[i] << " != " << expected[i] << "\n";
    }
  }

  sycl::free(accum, Q);
  return ret;
}
