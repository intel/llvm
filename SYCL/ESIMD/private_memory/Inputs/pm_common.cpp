//==--------------- pm_common.cpp - DPC++ ESIMD on-device test ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Since in ESIMD a single WI occupies entire underlying H/W thread, SYCL
// private memory maps to what's known as 'thread private memory' in CM.
// This test is intended to check TPM support implementation in ESIMD
// backend. In order to force using of TPM need to allocate 96x32 bytes or more.

#include "esimd_test_utils.hpp"

#include <iostream>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;

constexpr unsigned VL = 8;
constexpr unsigned SZ = 800; // big enough to use TPM

template <int CASE_NUM>
ESIMD_INLINE void work(int *o, int offx1, int offx2, int offy1, int offy2,
                       int offz, int base1, int base2, int divisor) {
  int x1[SZ];
  for (int j = 0; j < SZ; ++j) {
    int idx = (j + offx1) % SZ;
    x1[idx] = (idx % 2) == 0 ? j : base1;
  }

  int x2[SZ];
  for (int j = 0; j < SZ; ++j) {
    int idx = (j + offx2) % SZ;
    x2[idx] = base2 << (j % 32);
  }

  // some work with X1
  for (int j = 1; j < SZ; ++j) {
    if ((x1[j] + j) > base1)
      x1[j] = (j * (x1[j] + x1[j - 1]) / divisor) - base2;
  }

  // some work with X2
  for (int j = 1; j < SZ; ++j) {
    if ((x2[j] + j) < base2)
      x2[j] = (divisor * (x2[j] - x2[j - 1]) / j) + base1;
  }

  if constexpr (CASE_NUM == 1) {
    for (int j = 0; j < SZ; ++j)
      o[j % VL] += x1[j] - x2[j];
  } else {
    int *y1[SZ];
    for (int j = 0; j < SZ; ++j) {
      int idx = (j + offy1) % SZ;
      y1[j] = j % 6 == 0 ? x1 + idx : x2 + idx;
    }

    int *y2[SZ];
    for (int j = 0; j < SZ; ++j) {
      int idx = (j + offy2) % SZ;
      y2[j] = j % 2 == 0 ? x2 + idx : x1 + idx;
    }

    // some work with Y1
    for (int j = 0; j < SZ; j += 2) {
      if (*(y1[j]) > *(y1[j + 1]))
        *(y1[j]) = *(y1[j + 1]) - *(y1[j]);
    }

    // some work with Y2
    for (int j = 1; j < SZ - 1; j += 2) {
      if ((*(y2[j]) <= *(y2[j + 1]))) {
        auto temp = y2[j];
        y2[j] = y2[j + 1];
        y2[j + 1] = temp;
      }
    }

    if constexpr (CASE_NUM == 2) {
      for (int j = 0; j < SZ; ++j)
        o[j % VL] += *(y1[j]) - *(y2[j]);
    } else {
      static_assert(CASE_NUM == 3, "invalid CASE_NUM");

      int **z[SZ];
      for (int j = 0; j < SZ; ++j) {
        int idx = (j + offz) % SZ;
        z[j] = y1 + idx;
      }

      // some work with Z
      for (int j = 0; j < SZ - 1; ++j) {
        if (*(*(z[j])) < *(*(z[j + 1])))
          z[j] = y2 + j;
        if (j % 18 == 0)
          (*(*(z[j])))++;
      }

      for (int j = 0; j < SZ; ++j)
        o[j % VL] += *(*(z[j]));
    }
  }
}

template <int CASE_NUM> class KernelID;

template <int CASE_NUM> int test() {
  queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler());

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";
  auto ctx = q.get_context();

  int *output =
      static_cast<int *>(sycl::malloc_shared(VL * sizeof(int), dev, ctx));
  memset(output, 0, VL * sizeof(int));

  int offx1 = 111;
  int offx2 = 55;
  int offy1 = 499;
  int offy2 = 223;
  int offz = 99;
  int base1 = 500;
  int base2 = 100;
  int divisor = 4;

  {
    auto e = q.submit([&](handler &cgh) {
      cgh.parallel_for<KernelID<CASE_NUM>>(
          sycl::range<1>{1}, [=](id<1> i) SYCL_ESIMD_KERNEL {
            using namespace sycl::ext::intel::esimd;

            int o[VL] = {0};

            work<CASE_NUM>(o, offx1, offx2, offy1, offy2, offz, base1, base2,
                           divisor);

            simd<int, VL> val(0);
            for (int j = 0; j < VL; j++)
              val.select<1, 1>(j) += o[j];

            val.copy_to(output);
          });
    });
    e.wait();
  }

  int o[VL] = {0};

  work<CASE_NUM>(o, offx1, offx2, offy1, offy2, offz, base1, base2, divisor);

  int err_cnt = 0;
  for (int j = 0; j < VL; ++j)
    if (output[j] != o[j])
      err_cnt += 1;

  sycl::free(output, ctx);

  if (err_cnt > 0) {
    std::cout << "FAILED.\n";
    return 1;
  }

  std::cout << "Passed\n";
  return 0;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cout << "Skipped! Specify case number" << std::endl;
    return 1;
  }

  int case_num = atoi(argv[1]);
  switch (case_num) {
  case 1:
    return test<1>();
  case 2:
    return test<2>();
  case 3:
    return test<3>();
  }
  std::cerr << "Invalid case number: " << case_num << "\n";
  return 1;
}
