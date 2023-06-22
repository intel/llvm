//==---------------- mask_math.cpp  - DPC++ ESIMD on-device test -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This test checks basic arithmetic operations between simd and simd_mask
//===----------------------------------------------------------------------===//
// UNSUPPORTED: esimd_emulator, windows
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "../esimd_test_utils.hpp"
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

namespace esimd = sycl::ext::intel::esimd;
// Result type of a scalar binary Op
template <class T1, class T2, class OpClass>
using scalar_comp_t =
    std::conditional_t<std::is_same_v<OpClass, esimd_test::CmpOp>,
                       typename esimd::simd_mask<1>::element_type,
                       __ESIMD_DNS::computation_type_t<T1, T2>>;
// Result type of a vector binary Op
template <class T1, class T2, class OpClass, int N = 0>
using comp_t = std::conditional_t<
    N == 0, scalar_comp_t<T1, T2, OpClass>,
    std::conditional_t<
        std::is_same_v<OpClass, esimd_test::CmpOp>, esimd::simd_mask<N>,
        esimd::simd<__ESIMD_DNS::computation_type_t<T1, T2>, N>>>;

// Number of test cases for every operator type
constexpr int TestCaseNumber = 7;

template <class T, class OpClass, unsigned Size, class Ops>
bool test(Ops ops, queue &q) {
  T *Input = sycl::malloc_shared<T>(Size, q);
  constexpr int NumOps = (int)Ops::size;
  int CSize = NumOps * Size;
  T *Output = sycl::malloc_shared<T>(CSize * TestCaseNumber, q);
  for (int i = 0; i < Size; ++i)
    Input[i] = i;
  constexpr T TestValue = 2;
  std::vector<T> expectedValues;

  // esimd_test::ArithBinaryOps has 4 members for general arithmetic
  // operations: +, -, *, /.
  if constexpr (Size == 1) {
    if constexpr (std::is_same_v<OpClass, esimd_test::BinaryOp> &&
                  Ops::size == 4) {
      if constexpr (__ESIMD_DNS::is_generic_floating_point_v<T>) {
        expectedValues = {2, 2,  0, std::numeric_limits<T>::infinity(),
                          2, -2, 0, 0,
                          2, 2,  0, std::numeric_limits<T>::infinity(),
                          2, -2, 0, 0,
                          0, 0,  0, 1,
                          2, 2,  0, std::numeric_limits<T>::infinity(),
                          2, -2, 0, 0};
      } else {
        expectedValues = {2, 2, 0, 0,          2, 4294967294, 0, 0, 2, 2,
                          0, 0, 2, 4294967294, 0, 0,          0, 0, 0, 0,
                          2, 2, 0, 0,          2, 4294967294, 0, 0};
      }

    } else if constexpr (std::is_same_v<OpClass, esimd_test::CmpOp>) {
      expectedValues = {0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0,
                        0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0,
                        1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0};
    } else {
      // esimd_test::IntBinaryOpsNoShiftNoDivRem has 10 members for integer
      // arithmetic operations: +, -, *, |, & ^, <<, >>, %, /
      expectedValues = {2, 2, 0, 0, 0, 2, 2, 2,  0, 2, 2, -2, 0, 0,  0, 0, 0, 2,
                        0, 2, 2, 2, 0, 0, 0, 2,  2, 2, 0, 2,  2, -2, 0, 0, 0, 0,
                        0, 2, 0, 2, 0, 0, 0, 0,  0, 0, 0, 0,  0, 0,  2, 2, 0, 0,
                        0, 2, 2, 2, 0, 2, 2, -2, 0, 0, 0, 0,  0, 2,  0, 2};
    }
  } else {
    if constexpr (std::is_same_v<OpClass, esimd_test::BinaryOp> &&
                  Ops::size == 4) {
      if constexpr (__ESIMD_DNS::is_generic_floating_point_v<T>) {
        expectedValues = {
            3, 2, 1,  2,  2, 0, 2,   std::numeric_limits<T>::infinity(),
            3, 2, -1, -2, 2, 0, 0.5, 0,
            3, 2, 1,  2,  2, 0, 2,   std::numeric_limits<T>::infinity(),
            3, 2, -1, -2, 2, 0, 0.5, 0,
            2, 0, 0,  0,  1, 0, 1,   1,
            3, 2, 1,  2,  2, 0, 2,   std::numeric_limits<T>::infinity(),
            3, 2, -1, -2, 2, 0, 0.5, 0};
      } else {
        expectedValues = {3, 2, 1,          2,          2, 0, 2, 2,
                          3, 2, 4294967295, 4294967294, 2, 0, 0, 0,
                          3, 2, 1,          2,          2, 0, 2, 2,
                          3, 2, 4294967295, 4294967294, 2, 0, 0, 0,
                          2, 0, 0,          0,          1, 0, 1, 1,
                          3, 2, 1,          2,          2, 0, 2, 2,
                          3, 2, 4294967295, 4294967294, 2, 0, 0, 0};
      }
    } else if constexpr (std::is_same_v<OpClass, esimd_test::CmpOp>) {
      expectedValues = {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
                        0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1,
                        1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0};
    } else {
      // esimd_test::IntBinaryOpsNoShiftNoDivRem has 10 members for integer
      // arithmetic operations: +, -, *, |, & ^, <<, >>, %, /
      expectedValues = {
          3, 2, 1,  2,  2, 0, 2, 2, 0, 0, 4, 2, 1, 2, 3, 2, 0, 0, 3, 2,
          3, 2, -1, -2, 2, 0, 0, 0, 1, 0, 4, 0, 0, 0, 3, 2, 0, 0, 3, 2,
          3, 2, 1,  2,  2, 0, 2, 2, 0, 0, 4, 2, 1, 2, 3, 2, 0, 0, 3, 2,
          3, 2, -1, -2, 2, 0, 0, 0, 1, 0, 4, 0, 0, 0, 3, 2, 0, 0, 3, 2,
          2, 0, 0,  0,  1, 0, 1, 1, 0, 0, 2, 0, 0, 0, 1, 0, 1, 0, 0, 0,
          3, 2, 1,  2,  2, 0, 2, 2, 0, 0, 4, 2, 1, 2, 3, 2, 0, 0, 3, 2,
          3, 2, -1, -2, 2, 0, 0, 0, 1, 0, 4, 0, 0, 0, 3, 2, 0, 0, 3, 2};
    }
  }

  try {
    q.submit([&](sycl::handler &cgh) {
       cgh.single_task([=]() SYCL_ESIMD_KERNEL {
         esimd::simd<T, Size> x = TestValue;
         esimd::simd<T, Size> InputVector(Input);

         esimd_test::apply_ops(
             ops, x, InputVector < (Size / 2),
             [=](comp_t<T, T, OpClass, Size> res, OpClass op, unsigned op_num) {
               unsigned res_off = op_num * Size;
               res.copy_to(Output + res_off);
             });
         esimd_test::apply_ops(
             ops, InputVector < (Size / 2), x,
             [=](comp_t<T, T, OpClass, Size> res, OpClass op, unsigned op_num) {
               unsigned res_off = CSize + op_num * Size;
               res.copy_to(Output + res_off);
             });
         esimd_test::apply_ops(
             ops, TestValue, InputVector < (Size / 2),
             [=](comp_t<T, T, OpClass, Size> res, OpClass op, unsigned op_num) {
               unsigned res_off = 2 * CSize + op_num * Size;
               res.copy_to(Output + res_off);
             });
         esimd_test::apply_ops(
             ops, InputVector < (Size / 2), TestValue,
             [=](comp_t<T, T, OpClass, Size> res, OpClass op, unsigned op_num) {
               unsigned res_off = 3 * CSize + op_num * Size;
               res.copy_to(Output + res_off);
             });
         esimd_test::apply_ops(
             ops, InputVector < (Size / 2), InputVector < (Size / 2),
             [=](comp_t<T, T, OpClass, Size> res, OpClass op, unsigned op_num) {
               unsigned res_off = 4 * CSize + op_num * Size;
               res.copy_to(Output + res_off);
             });
         esimd_test::apply_ops(
             ops, x.template bit_cast_view<T>(), InputVector < (Size / 2),
             [=](comp_t<T, T, OpClass, Size> res, OpClass op, unsigned op_num) {
               unsigned res_off = 5 * CSize + op_num * Size;
               res.copy_to(Output + res_off);
             });
         esimd_test::apply_ops(
             ops, InputVector < (Size / 2), x.template bit_cast_view<T>(),
             [=](comp_t<T, T, OpClass, Size> res, OpClass op, unsigned op_num) {
               unsigned res_off = 6 * CSize + op_num * Size;
               res.copy_to(Output + res_off);
             });
       });
     }).wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    sycl::free(Input, q);
    sycl::free(Output, q);
    return false;
  }
  bool Result = true;
  int err_cnt = 0;

  for (int i = 0; i < CSize * TestCaseNumber; ++i) {
    if (Output[i] != expectedValues[i]) {
      Result = false;
      if (++err_cnt < 10) {
        std::cout << "  failed at index " << i
                  << " Expected Value: " << expectedValues[i]
                  << " Result: " << Output[i] << ".\n";
      }
    }
  }
  sycl::free(Input, q);
  sycl::free(Output, q);
  return Result;
}

int main(int, char **) {
  sycl::queue q;
  auto dev = q.get_device();
  bool Passed = true;
  auto arith_ops = esimd_test::ArithBinaryOps;
  Passed &= test<float, esimd_test::BinaryOp, 2>(arith_ops, q);
  Passed &= test<float, esimd_test::BinaryOp, 1>(arith_ops, q);
  Passed &= test<uint32_t, esimd_test::BinaryOp, 2>(arith_ops, q);
  Passed &= test<uint32_t, esimd_test::BinaryOp, 1>(arith_ops, q);
  if (dev.has(aspect::fp16)) {
    Passed &= test<half, esimd_test::BinaryOp, 2>(arith_ops, q);
  }

  auto int_arith_ops = esimd_test::IntBinaryOps;
  Passed &= test<int32_t, esimd_test::BinaryOp, 2>(int_arith_ops, q);

  auto cmp_ops = esimd_test::CmpOps;
  Passed &= test<uint16_t, esimd_test::CmpOp, 2>(cmp_ops, q);
  Passed &= test<uint16_t, esimd_test::CmpOp, 1>(cmp_ops, q);

  if (!Passed) {
    std::cout << "Test failed." << std::endl;
  } else {
    std::cout << "Test passed." << std::endl;
  }

  return Passed ? 0 : 1;
}
