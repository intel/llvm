//===------ noinline_args_size_common.hpp  - DPC++ ESIMD on-device test ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The test checks that ESIMD kernels support call of noinline function from
// main function with different total arguments size and retval size. Cases:
//   Total arguments size < %arg register size (32 GRFs)
//   Total arguments size == %arg register size
//   Total arguments size > %arg register size (i.e. stack mem is required)
//   Return value size < %retval register size (12 GRFs)
//   Return value size == %retval register size
//   Return value size > %retval register size

#include "esimd_test_utils.hpp"

#include <CL/sycl.hpp>
#include <iostream>
#include <sycl/ext/intel/esimd.hpp>

static_assert(SIZE >= VL, "Size must greater than or equal to VL");
static_assert(SIZE % VL == 0, "Size must be multiple of VL");
constexpr unsigned ROWS = SIZE / VL;

using namespace cl::sycl;

class KernelID;

template <typename TA, typename TB, typename TC>
ESIMD_NOINLINE TC add(TA A, TB B) {
  return (TC)A + (TC)B;
}

int main(void) {
  queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler());

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

  a_data_t *A = sycl::malloc_shared<a_data_t>(SIZE, q);
  for (int i = 0; i < SIZE; i++)
    A[i] = (a_data_t)1;

  b_data_t *B = sycl::malloc_shared<b_data_t>(SIZE, q);
  for (int i = 0; i < SIZE; i++)
    B[i] = (b_data_t)i;

  c_data_t *C = sycl::malloc_shared<c_data_t>(SIZE, q);
  memset(C, 0, SIZE * sizeof(c_data_t));

  try {
    auto qq = q.submit([&](handler &cgh) {
      cgh.parallel_for<KernelID>(
          sycl::range<1>{1}, [=](id<1> i) SYCL_ESIMD_KERNEL {
            using namespace sycl::ext::intel::esimd;

            simd<a_data_t, SIZE> va(0);
            simd<b_data_t, SIZE> vb(0);
            for (int j = 0; j < ROWS; j++) {
              simd<a_data_t, VL> a_data;
              a_data.copy_from(A + j * VL);
              va.select<VL, 1>(j * VL) = a_data;
              simd<b_data_t, VL> b_data;
              b_data.copy_from(B + j * VL);
              vb.select<VL, 1>(j * VL) = b_data;
            }

            auto vc = add<simd<a_data_t, SIZE>, simd<b_data_t, SIZE>,
                          simd<c_data_t, SIZE>>(va, vb);

            for (int j = 0; j < ROWS; j++) {
              simd<c_data_t, VL> vals = vc.select<VL, 1>(j * VL);
              vals.copy_to(C + j * VL);
            }
          });
    });

    qq.wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << std::endl;
    sycl::free(A, q);
    sycl::free(B, q);
    sycl::free(C, q);
    return 1;
  }

  unsigned err_cnt = 0;
  for (int i = 0; i < SIZE; i++)
    if (C[i] != A[i] + B[i])
      err_cnt++;

  sycl::free(A, q);
  sycl::free(B, q);
  sycl::free(C, q);

  if (err_cnt > 0) {
    std::cout << "FAILED" << std::endl;
    return 1;
  }

  std::cout << "passed" << std::endl;
  return 0;
}
