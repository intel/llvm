// RUN: %{build} -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %{run} %t.out
//
// The test checks functionality of the slm_gather/slm_scatter ESIMD APIs.

#include "../esimd_test_utils.hpp"

using namespace sycl;

template <typename T, unsigned VL, unsigned STRIDE> struct Kernel {
  static constexpr int MASKED_LANE = VL - 1;
  T *buf;
  Kernel(T *buf) : buf(buf) {}

  void operator()(id<1> i) const SYCL_ESIMD_KERNEL {
    using namespace sycl::ext::intel::esimd;

    // In this test, we have a single workitem. No barriers required.
    slm_init<VL * STRIDE *
             sizeof(typename sycl::ext::intel::esimd::detail::dword_type<
                    T>::type)>();

    simd<T, VL> valsIn;
    valsIn.copy_from(buf);

    simd<uint32_t, VL> offsets(0, STRIDE * sizeof(T));
    simd_mask<VL> pred = 1;
    if constexpr (MASKED_LANE > 0) {
      simd<T, 1> V1(static_cast<T>(-1));
      simd<uint32_t, 1> Offsets1(MASKED_LANE * STRIDE * sizeof(T));
      slm_scatter<T, 1>(Offsets1, V1);
      pred[MASKED_LANE] = 0; // mask out the last lane if not the only lane
    }
    slm_scatter<T, VL>(offsets, valsIn, pred);

    simd<T, VL> valsOut = slm_gather<T, VL>(offsets);

    valsOut.copy_to(buf);
  }
};

template <typename T, unsigned VL, unsigned STRIDE> bool test(queue q) {
  using namespace sycl::ext::intel::esimd;
  constexpr size_t size = VL;
  constexpr int MASKED_LANE = VL - 1;

  std::cout << "Testing T=" << typeid(T).name() << " VL=" << VL
            << " STRIDE=" << STRIDE << "...\n";

  T *A = malloc_shared<T>(size, q);
  T *gold = new T[size];

  for (int i = 0; i < size; ++i) {
    A[i] = (T)i + 1;
    gold[i] = (T)i + 1;
  }

  try {
    range<1> glob_range{1};

    q.submit([&](handler &cgh) {
       Kernel<T, VL, STRIDE> kernel(A);
       cgh.parallel_for(glob_range, kernel);
     }).wait();
  } catch (sycl::exception const &e) {
    std::cerr << "SYCL exception caught: " << e.what() << '\n';
    free(A, q);
    delete[] gold;
    return false;
  }

  int err_cnt = 0;
  for (unsigned i = 0; i < size; ++i) {
    T GoldVal = (i == MASKED_LANE && i > 0) ? static_cast<T>(-1) : gold[i];
    if (A[i] != GoldVal && ++err_cnt < VL) {
      std::cout << "failed at index " << i << ": " << A[i] << " != " << GoldVal
                << " (gold)\n";
    }
  }

  if (err_cnt > 0) {
    std::cout << "  pass rate: "
              << ((float)(size - err_cnt) / (float)size) * 100.0f << "% ("
              << (size - err_cnt) << "/" << size << ")\n";
  }

  free(A, q);
  delete[] gold;

  std::cout << (err_cnt > 0 ? "  FAILED\n" : "  Passed\n");
  return err_cnt == 0;
}

template <typename T, unsigned VL> bool test(queue q) {
  bool passed = true;
  passed &= test<T, VL, 1>(q);
  passed &= test<T, VL, 2>(q);
  passed &= test<T, VL, 3>(q);
  passed &= test<T, VL, 4>(q);
  return passed;
}

int main(void) {
  queue q(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler());
  auto dev = q.get_device();
  esimd_test::printTestLabel(q);

  bool passed = true;

  passed &= test<char, 16>(q);
  passed &= test<char, 32>(q);
  passed &= test<short, 16>(q);
  passed &= test<short, 32>(q);

  passed &= test<int, 16>(q);
  passed &= test<int, 32>(q);
  passed &= test<float, 16>(q);
  passed &= test<float, 32>(q);

  if (dev.has(aspect::fp16)) {
    passed &= test<half, 16>(q);
    passed &= test<half, 32>(q);
  }

  passed &= test<int64_t, 16>(q);
  passed &= test<int64_t, 32>(q);

  if (dev.has(sycl::aspect::fp64)) {
    passed &= test<double, 16>(q);
    passed &= test<double, 32>(q);
  }

  return passed ? 0 : 1;
}
