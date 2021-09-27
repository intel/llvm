// REQUIRES: gpu
// UNSUPPORTED: cuda
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
//
// The test checks functionality of the slm_gather/slm_scatter ESIMD APIs.

// TODO: Enable the test for 1 and 2 byte types when the implementation is fixed
#define SKIP_ONE_AND_TWO_BYTE_BLOCKS 1

#include "../esimd_test_utils.hpp"

#include <CL/sycl.hpp>
#include <iostream>
#include <sycl/ext/intel/experimental/esimd.hpp>

using namespace cl::sycl;

template <typename T, unsigned VL, unsigned STRIDE> struct Kernel {
  T *buf;
  Kernel(T *buf) : buf(buf) {}

  void operator()(id<1> i) const SYCL_ESIMD_KERNEL {
    using namespace sycl::ext::intel::experimental::esimd;

    // In this test, we have a single workitem. No barriers required.
    slm_init(VL * STRIDE *
             sizeof(typename sycl::ext::intel::experimental::esimd::detail::
                        dword_type<T>::type));

    simd<T, VL> valsIn;
    valsIn.copy_from(buf);

    simd<uint32_t, VL> offsets(0, STRIDE * sizeof(T));
    slm_scatter<T, VL>(valsIn, offsets);

    simd_mask<VL> pred = 1;
    pred[VL - 1] = 0; // mask out the last lane
    simd<T, VL> valsOut = slm_gather<T, VL>(offsets, pred);
    valsOut *= -1;

    valsOut.copy_to(buf);
  }
};

template <typename T, unsigned VL, unsigned STRIDE> bool test(queue q) {
  using namespace sycl::ext::intel::experimental::esimd;
  constexpr size_t size = VL;

  std::cout << "Testing T=" << typeid(T).name() << " VL=" << VL
            << " STRIDE=" << STRIDE << "...\n";

  auto dev = q.get_device();
  auto ctxt = q.get_context();
  T *A = static_cast<T *>(malloc_shared(size * sizeof(T), dev, ctxt));
  T *gold = new T[size];

  for (int i = 0; i < size; ++i) {
    A[i] = (T)-i;
    gold[i] = (T)i;
  }

  // Account for masked out last lane (with pred argument to slm_gather).
  gold[(VL - 1)] = (T)0;

  try {
    range<1> glob_range{1};

    auto e = q.submit([&](handler &cgh) {
      Kernel<T, VL, STRIDE> kernel(A);
      cgh.parallel_for(glob_range, kernel);
    });
    e.wait();
  } catch (sycl::exception const &e) {
    std::cerr << "SYCL exception caught: " << e.what() << '\n';
    free(A, ctxt);
    delete[] gold;
    return static_cast<bool>(e.code());
  }

  int err_cnt = 0;
  for (unsigned i = 0; i < size; ++i) {
    if (A[i] != gold[i]) {
      if (++err_cnt < VL) {
        std::cout << "failed at index " << i << ": " << A[i]
                  << " != " << gold[i] << " (gold)\n";
      }
    }
  }

  if (err_cnt > 0) {
    std::cout << "  pass rate: "
              << ((float)(size - err_cnt) / (float)size) * 100.0f << "% ("
              << (size - err_cnt) << "/" << size << ")\n";
  }

  free(A, ctxt);
  delete[] gold;

  std::cout << (err_cnt > 0 ? "  FAILED\n" : "  Passed\n");
  return err_cnt > 0 ? false : true;
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
  queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler());

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

  bool passed = true;

#ifndef SKIP_ONE_AND_TWO_BYTE_BLOCKS
  passed &= test<char, 16>(q);
  passed &= test<char, 32>(q);
  passed &= test<short, 16>(q);
  passed &= test<short, 32>(q);
#endif
  passed &= test<int, 16>(q);
  passed &= test<int, 32>(q);
  passed &= test<float, 16>(q);
  passed &= test<float, 32>(q);

  return passed ? 0 : 1;
}
