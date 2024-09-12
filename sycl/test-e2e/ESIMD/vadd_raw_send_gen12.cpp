//==-- vadd_raw_send_gen12.cpp  - DPC++ ESIMD on-device test --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------===//
// REQUIRES: gpu-intel-gen12
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "esimd_test_utils.hpp"

using namespace sycl;
using namespace sycl::ext::intel::esimd;

simd<int, 16> gather(int *addr) {
  uint32_t desc = 0x022D0BFF;
  simd<int, 16> ret;
  simd<uint64_t, 16> addrs(reinterpret_cast<uint64_t>(addr), sizeof(int));
  ret = raw_send<4, 12, 1, 2>(ret, addrs, 0, desc);
  return ret;
}

void scatter(simd<int, 16> &vec, int *addr) {
  uint32_t desc = 0x080691FF;
  simd<uint64_t, 16> addrs(reinterpret_cast<uint64_t>(addr), sizeof(int));
  raw_sends<4, 12, 4, 2>(addrs, vec, 0, desc);
}

int main(void) {
  constexpr unsigned Size = 16;
  constexpr unsigned VL = 16;

  queue q(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler());

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

  int *A = malloc_shared<int>(Size, q);
  int *B = malloc_shared<int>(Size, q);
  int *C = malloc_shared<int>(Size, q);

  for (unsigned i = 0; i < Size; ++i) {
    A[i] = B[i] = i;
  }

  try {
    auto e = q.submit([&](handler &cgh) {
      cgh.single_task([=]() SYCL_ESIMD_KERNEL {
        simd<int, VL> va = gather(A);
        simd<int, VL> vb = gather(B);
        simd<int, VL> vc = va + vb;
        scatter(vc, C);
      });
    });
    e.wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    free(A, q);
    free(B, q);
    free(C, q);
    return 1;
  }

  int err_cnt = 0;

  for (unsigned i = 0; i < Size; ++i) {
    if (A[i] + B[i] != C[i]) {
      if (++err_cnt < 10) {
        std::cout << "failed at index " << i << ", " << C[i] << " != " << A[i]
                  << " + " << B[i] << "\n";
      }
    }
  }
  if (err_cnt > 0) {
    std::cout << "  pass rate: "
              << ((float)(Size - err_cnt) / (float)Size) * 100.0f << "% ("
              << (Size - err_cnt) << "/" << Size << ")\n";
  }

  free(A, q);
  free(B, q);
  free(C, q);
  std::cout << (err_cnt > 0 ? "FAILED\n" : "Passed\n");
  return err_cnt > 0 ? 1 : 0;
}
