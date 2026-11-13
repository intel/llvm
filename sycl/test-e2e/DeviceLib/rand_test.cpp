// XFAIL: new-offload-model && target-amd
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/20797

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// XFAIL: target-native_cpu
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/20142

#include <sycl/builtins.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

#include <cstdlib>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

static constexpr int SEQUENCE_LEN = 32;
static constexpr unsigned int SEED_A = 42u;
static constexpr unsigned int SEED_B = 123u;

bool test_different_seeds(sycl::queue &q) {
  int *val_a = sycl::malloc_shared<int>(1, q);
  int *val_b = sycl::malloc_shared<int>(1, q);

  if (!val_a || !val_b) {
    std::cerr << "  USM allocation failed.\n";
    sycl::free(val_a, q);
    sycl::free(val_b, q);
    return false;
  }

  *val_a = 0;
  *val_b = 0;

  q.single_task([=]() {
     std::srand(SEED_A);
     *val_a = std::rand();
   }).wait();

  q.single_task([=]() {
     std::srand(SEED_B);
     *val_b = std::rand();
   }).wait();

  bool differ = (*val_a != *val_b);
  std::cout << "  seed=" << SEED_A << ", rand()=" << *val_a << "\n";
  std::cout << "  seed=" << SEED_B << ", rand()=" << *val_b << "\n";
  if (!differ)
    std::cerr << "  Seeds " << SEED_A << " and " << SEED_B
              << " produced the same first value: " << *val_a << "\n";

  sycl::free(val_a, q);
  sycl::free(val_b, q);
  return differ;
}

bool test_sequence_varies(sycl::queue &q) {
  int *seq = sycl::malloc_shared<int>(SEQUENCE_LEN, q);

  if (!seq) {
    std::cerr << "  USM allocation failed.\n";
    return false;
  }

  for (int i = 0; i < SEQUENCE_LEN; ++i)
    seq[i] = 0;

  q.single_task([=]() {
     std::srand(SEED_A);
     for (int i = 0; i < SEQUENCE_LEN; ++i)
       seq[i] = std::rand();
   }).wait();

  std::cout << "  First 8 values: ";
  for (int i = 0; i < 8; ++i)
    std::cout << seq[i] << " ";
  std::cout << "\n";

  int first = seq[0];
  bool varies = false;
  for (int i = 1; i < SEQUENCE_LEN; ++i) {
    if (seq[i] != first) {
      varies = true;
      break;
    }
  }
  if (!varies)
    std::cerr << "  All " << SEQUENCE_LEN << " values are identical (" << first
              << ").\n";

  sycl::free(seq, q);
  return varies;
}

int main() {
  sycl::queue q;
  try {
    q = sycl::queue(sycl::gpu_selector_v, [](sycl::exception_list el) {
      for (auto &e : el)
        std::rethrow_exception(e);
    });
  } catch (...) {
    std::cout << "GPU not available – falling back to default selector.\n";
    q = sycl::queue(sycl::default_selector_v);
  }

  auto dev = q.get_device();
  if (!dev.has(sycl::aspect::usm_shared_allocations)) {
    std::cerr << "Device does not support USM shared allocations. Aborting.\n";
    return EXIT_FAILURE;
  }

  int passed = 0, failed = 0;

  struct TestCase {
    std::string name;
    std::function<bool(sycl::queue &)> fn;
  };

  const std::vector<TestCase> tests = {
      {"srand() different seeds leading to different first values",
       test_different_seeds},
      {"rand() sequence is non-constant", test_sequence_varies},
  };

  for (auto &tc : tests) {
    std::cout << "--- " << tc.name << " ---\n";
    bool ok = false;
    try {
      ok = tc.fn(q);
    } catch (const sycl::exception &e) {
      std::cerr << "  SYCL exception: " << e.what() << "\n";
    } catch (const std::exception &e) {
      std::cerr << "  std::exception: " << e.what() << "\n";
    }
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << tc.name << "\n";
    ok ? ++passed : ++failed;
    std::cout << "\n";
  }

  std::cout << "========================================\n";
  std::cout << "Results: " << passed << " passed, " << failed << " failed"
            << " (out of " << tests.size() << " tests)\n";
  std::cout << "========================================\n";

  return failed;
}
