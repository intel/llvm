
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <array>
#include <cassert>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <vector>

static constexpr std::size_t kUniqueKernels = 256;
static constexpr std::size_t kConsecutiveDupSubmissions =
    5000; // same kernel over and over
static constexpr std::size_t kCyclicSubmissions =
    8000;                                        // cycle over small subset
static constexpr std::size_t kCyclicSubset = 16; // cycle kernel subset
static constexpr std::size_t kAllKernelsSubmissions =
    10000; // running all kernel

template <int ID> struct KernelTag;

template <int ID> static void submit_increment(sycl::queue &Q, int *accum) {
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
static auto make_fn_table(std::index_sequence<Is...>) {
  return std::array<SubmitFn, kUniqueKernels>{
      &submit_increment<static_cast<int>(Is)>...};
}

int main() {
  sycl::queue Q;

  int *accum = sycl::malloc_shared<int>(kUniqueKernels, Q);
  assert(accum && "USM alloc failed");
  for (std::size_t i = 0; i < kUniqueKernels; ++i)
    accum[i] = 0;

  std::vector<std::size_t> expected(kUniqueKernels, 0);

  auto fns = make_fn_table(std::make_index_sequence<kUniqueKernels>{});

  // Submit the same kernel over and over again. The submitted kernel
  // vector shouldn't grow at all, since we do a lookback over
  // a few previous kernels.
  auto runDuplicates = [&]() {
    for (size_t i = 0; i < kConsecutiveDupSubmissions; ++i) {
      fns[0](Q, accum);
      expected[0]++;
    }
  };

  // Run a small subset of kernels in a loop. Likely the most realistic
  // scenario. Should be mostly absorbed by loopback duplicate search, and,
  // possibliy, compaction.
  auto runCyclical = [&]() {
    for (size_t i = 0; i < kCyclicSubmissions; ++i) {
      size_t id = i % kCyclicSubset;
      fns[id](Q, accum);
      expected[id]++;
    }
  };

  // Run all kernels in the loop. Should dynamically adjust the
  // threshold for submitted kernels.
  auto runAll = [&]() {
    for (size_t i = 0; i < kAllKernelsSubmissions; ++i) {
      size_t id = i % kUniqueKernels;
      fns[id](Q, accum);
      expected[id]++;
    }
  };

  // Run from small kernel variety, to large, to small, to test dynamic
  // threshold changes.
  runDuplicates();
  runCyclical();
  runAll();
  Q.wait(); // this clears the submitted kernels list, allowing the threshold to
            // lower.
  runCyclical();
  runDuplicates();

  Q.wait();

  int ret = 0;
  for (std::size_t i = 0; i < kUniqueKernels; ++i) {
    if (static_cast<std::size_t>(accum[i]) != expected[i]) {
      ret = 0;
      std::cout << "fail: " << accum[i] << " != " << expected[i] << "\n";
    }
  }

  sycl::free(accum, Q);
  return ret;
}
