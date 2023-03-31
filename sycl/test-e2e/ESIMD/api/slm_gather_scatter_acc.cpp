// REQUIRES: gpu
// UNSUPPORTED: gpu-intel-gen9 && windows
// UNSUPPORTED: cuda || hip

// RUN: %clangxx -fsycl-device-code-split=per_kernel -fsycl %s -o %t1.out
// RUN: %GPU_RUN_PLACEHOLDER %t1.out

// RUN: %clangxx -fsycl-device-code-split=per_kernel -fsycl -DACC_TO_GATHER %s -o %t2.out
// RUN: %GPU_RUN_PLACEHOLDER %t2.out

// The test checks functionality of the slm_gather/slm_scatter ESIMD APIs.

#include "../esimd_test_utils.hpp"

#include <iostream>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::intel::esimd;

template <typename T, unsigned VL, unsigned STRIDE> bool test(queue q) {

  uint32_t local_range = 8;
  nd_range<1> ndr{range<1>{local_range}, range<1>{local_range}};

  size_t size = VL * ndr.get_global_range().size();
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
    q.submit([&](handler &cgh) {
       uint32_t promo_type_coeff = sizeof(T) < 4 ? 4 / sizeof(T) : 1;
       uint32_t elems_per_wi = VL * STRIDE * promo_type_coeff;
       local_accessor<T, 1> local_acc(local_range * elems_per_wi, cgh);

       cgh.parallel_for(ndr, [=](nd_item<1> item) SYCL_ESIMD_KERNEL {
         using namespace sycl::ext::intel::esimd;

         constexpr uint32_t promo_type_size = sizeof(T) <= 4 ? 4 : sizeof(T);
         uint32_t mem_per_wi = VL * STRIDE * promo_type_size;
         uint32_t group_size = item.get_local_range(0);

         uint32_t gid = item.get_global_id(0);
         uint32_t lid = item.get_local_id(0);

         simd<T, VL> vals(A + gid * VL);
         vals = vals + static_cast<T>(1);
         simd<uint32_t, VL> offsets(lid * VL * STRIDE * promo_type_size,
                                    STRIDE * promo_type_size);
         slm_scatter<T, VL>(offsets, vals);

         simd_mask<VL> pred = 1;
         pred[VL - 1] = 0; // mask out the last lane
         simd<T, VL> out = slm_gather<T, VL>(offsets, pred);
         out.copy_to(A + gid * VL);
       });
     }).wait();
  } catch (sycl::exception const &e) {
    std::cerr << "SYCL exception caught: " << e.what() << '\n';
    free(A, q);
    delete[] gold;
    return static_cast<bool>(e.code());
  }

  int err_cnt = 0;
  int masked_err_cnt = 0;
  for (unsigned i = 0; i < size; ++i) {
    T ExpectedLoaded = gold[i] + static_cast<T>(1);
    if ((i % VL) == MASKED_LANE) {
      if (sizeof(T) >= 2) {
        // Value in this lane should be overwritten by whatever value was
        // returned by the masked gather in this lane. If, when verifying, we
        // see the original value stored in memory at this index this 99.999%
        // likely means the read hasn't been masked. There is non-zero chance
        // masked read will spontaneously return the same value we wrote here
        // and we'll get false negative. But this is the best we can do to test
        // masked reads. Disable this for 1/2-byte elements, where probability
        // of false negative increases.
        if (A[i] == ExpectedLoaded) {
          if (++masked_err_cnt < VL) {
            std::cout << "masking failed at index " << i << "\n";
          }
        }
      }
    } else if (A[i] != ExpectedLoaded) {
      if (++err_cnt < VL) {
        std::cout << "failed at index " << i << ": " << A[i]
                  << " != " << ExpectedLoaded << " (gold)\n";
      }
    }
  }

  // The comment above for masked elements gives impression that
  // chance of false-error is still high. Consider error on
  // masked elements as a true error if the masking was ignored
  // for all gathers.
  if (masked_err_cnt == ndr.get_global_range().size())
    err_cnt++;

  if (err_cnt > 0) {
    std::cout << "  pass rate: "
              << ((float)(size - err_cnt) / (float)size) * 100.0f << "% ("
              << (size - err_cnt) << "/" << size << ")\n";
  }

  free(A, q);
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
  queue q(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler());

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<sycl::info::device::name>()
            << "\n";

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
  return passed ? 0 : 1;
}
