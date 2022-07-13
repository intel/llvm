//==--------------- bin_un_cmp_ops_heavy.cpp  - DPC++ ESIMD on-device test -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu
// UNSUPPORTED: cuda || hip
// TODO: esimd_emulator fails due to unimplemented 'half' type
// XFAIL: esimd_emulator
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Tests various binary operations applied to simd objects.

// TODO
// Arithmetic operations behaviour depends on Gen's control regiter's rounding
// mode, which is RTNE by default:
//    cr0.5:4 is 00b = Round to Nearest or Even (RTNE)
// For half this leads to divergence between Gen and host (emulated) results
// larger than certain threshold. Might need to tune the cr0 once this feature
// is available in ESIMD.
//

#include "../esimd_test_utils.hpp"

#include <iostream>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace cl::sycl;
using namespace sycl::ext::intel::esimd;

template <class T1, class T2, int VL, class OpClass, class Ops> class TestID;

// Result type of a scalar binary Op
template <class T1, class T2, class OpClass>
using scalar_comp_t =
    std::conditional_t<std::is_same_v<OpClass, esimd_test::CmpOp>,
                       typename simd_mask<8>::element_type,
                       __ESIMD_DNS::computation_type_t<T1, T2>>;

// Result type of a vector binary Op
template <class T1, class T2, class OpClass, int N = 0>
using comp_t = std::conditional_t<
    N == 0, scalar_comp_t<T1, T2, OpClass>,
    std::conditional_t<std::is_same_v<OpClass, esimd_test::CmpOp>, simd_mask<N>,
                       simd<__ESIMD_DNS::computation_type_t<T1, T2>, N>>>;

// Helpers for printing
template <class T> auto cast(T val) { return val; }
template <> auto cast<char>(char val) { return (int)val; }
template <> auto cast<unsigned char>(unsigned char val) {
  return (unsigned int)val;
}
#ifdef __SYCL_DEVICE_ONLY__
template <> auto cast<_Float16>(_Float16 val) { return (float)val; }
#endif

// Main test function.
// T1, T2 - operand types,
// VL - vector length,
// OpClass - binary or comparison operations,
// VerifyF and InitF - verification and initialization function types
//   (instantiated within the test function),
// Ops - a compile-time sequence of operations to test.
//
template <class T1, class T2, int VL, class OpClass,
          template <class, class, class> class VerifyF,
          template <class, class, class> class InitF, class Ops>
bool test(Ops ops, queue &q, comp_t<T1, T2, OpClass> epsilon = 0) {
  // Log test case info
  std::cout << "Testing T1=" << typeid(T1).name() << " T2=" << typeid(T2).name()
            << ", VL=" << VL << " ...\n";
  std::cout << "Operations:";
  esimd_test::iterate_ops(ops, [=](OpClass op) {
    std::cout << " '" << esimd_test::Op2Str(op) << "'";
  });
  std::cout << "\n";

  // initialize test data
  constexpr int Size = 1024 * 7;
  T1 *A = sycl::malloc_shared<T1>(Size, q);
  T2 *B = sycl::malloc_shared<T2>(Size, q);
  constexpr int NumOps = (int)Ops::size;
  int CSize = NumOps * Size;
  using T = comp_t<T1, T2, OpClass>;
  // Result array. For each pair of A[i] and B[i] elements it reserves NumOps
  // elements to store result of all operations under test applied to the A[i]
  // and B[i]
  T *C = sycl::malloc_shared<T>(CSize, q);
  InitF<T1, T2, OpClass> init;

  for (int i = 0; i < Size; ++i) {
    init(A, B, C, i);
  }

  // submit the kernel
  try {
    auto e = q.submit([&](handler &cgh) {
      cgh.parallel_for<TestID<T1, T2, VL, OpClass, Ops>>(
          Size / VL, [=](id<1> i) SYCL_ESIMD_KERNEL {
            unsigned off = i * VL;
            simd<T1, VL> va(A + off, vector_aligned_tag{});
            simd<T2, VL> vb(B + off, vector_aligned_tag{});

            // applies each of the input operations to the va and vb vectors,
            // then invokes the lambda below, passing the result of the
            // operation, its ID and sequential number within the input sequence
            esimd_test::apply_ops(
                ops, va, vb,
                [=](comp_t<T1, T2, OpClass, VL> res, OpClass op,
                    unsigned op_num) {
                  unsigned res_off = off * NumOps + op_num * VL;
                  res.copy_to(C + res_off, vector_aligned_tag{});
                });
          });
    });
    e.wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    sycl::free(A, q);
    sycl::free(B, q);
    sycl::free(C, q);
    return false;
  }

  int err_cnt = 0;

  // now verify the results using provided verification function type
  for (unsigned i = 0; i < Size / VL; ++i) {
    unsigned off = i * VL;

    for (int j = 0; j < VL; ++j) {
      T1 a = A[off + j];
      T2 b = B[off + j];

      esimd_test::apply_ops(
          ops, a, b, [&](T Gold, OpClass op, unsigned op_num) {
            unsigned res_off = off * NumOps + op_num * VL;
            T Res = C[res_off + j];
            using Tint = esimd_test::int_type_t<sizeof(T)>;
            Tint ResBits = *(Tint *)&Res;
            Tint GoldBits = *(Tint *)&Gold;
            VerifyF<T1, T2, OpClass> verify_f(epsilon);

            if (!verify_f(Gold, Res, op)) {
              if (++err_cnt < 10) {
                std::cout << "  failed at index " << (res_off + j) << ", op "
                          << esimd_test::Op2Str(op) << ": " << cast(Res)
                          << "(0x" << std::hex << ResBits << ")"
                          << " != " << std::dec << cast(Gold) << "(0x"
                          << std::hex << GoldBits << ") [" << std::dec
                          << cast(a) << " " << esimd_test::Op2Str(op) << " "
                          << cast(b) << "]\n";
              }
            }
          });
    }
  }
  if (err_cnt > 0) {
    auto Size1 = NumOps * Size;
    std::cout << "  pass rate: "
              << ((float)(Size1 - err_cnt) / (float)Size1) * 100.0f << "% ("
              << (Size1 - err_cnt) << "/" << Size1 << ")\n";
  }

  free(A, q);
  free(B, q);
  free(C, q);
  std::cout << (err_cnt > 0 ? "  FAILED\n" : "  Passed\n");
  return err_cnt == 0;
}

// Flavours of verification function types.

template <class T1, class T2, class OpClass> struct verify_strict {
  using T = comp_t<T1, T2, OpClass>;

  verify_strict(T) {}

  bool operator()(T res, T gold, OpClass op) { return res == gold; }
};

#define EQ(x, y, epsilon)                                                      \
  ((x) > (y) ? (x) - (y) <= epsilon : (y) - (x) <= epsilon)

template <class T1, class T2, class OpClass> struct verify_epsilon {
  using T = comp_t<T1, T2, OpClass>;
  T epsilon;
  verify_epsilon(T epsilon) : epsilon(epsilon) {}

  bool operator()(T res, T gold, OpClass op) {
    if constexpr (std::is_same_v<OpClass, esimd_test::BinaryOp>) {
      if (op == esimd_test::BinaryOp::div) {
        return EQ(res, gold, epsilon);
      }
    }
    return res == gold;
  }
};

template <class T1, class T2, class OpClass> struct verify_n {
  using T = comp_t<T1, T2, OpClass>;
  int n;
  verify_n(int n) : n(n) {}

  bool operator()(T res, T gold, OpClass op) {
    using Tint = esimd_test::int_type_t<sizeof(T)>;
    Tint res_bits = *(Tint *)&res;
    Tint gold_bits = *(Tint *)&gold;
    return (abs(gold_bits - res_bits) > n) ? false : true;
  }
};

// Flavours of initialization function types.

template <class T1, class T2, class OpClass> struct init_default {
  using T = comp_t<T1, T2, OpClass>;

  void operator()(T1 *A, T2 *B, T *C, int i) {
    A[i] = (i % 3) * 90 + 10; /*10, 100, 190, 10, ...*/
    if constexpr (std::is_unsigned_v<T2>) {
      B[i] = (i % 3) * 99 + 1 /*1, 100, 199, 1, ...*/;
    } else {
      B[i] = (i % 4) * 180 - 170; /*-170, 10, 190, 370, -170,...*/
    }
    C[i] = 0;
  }
};

template <class T1, class T2, class OpClass> struct init_for_shift {
  using T = comp_t<T1, T2, OpClass>;

  void operator()(T1 *A, T2 *B, T *C, int i) {
    if constexpr (std::is_unsigned_v<T1>) {
      A[i] = (i % 3) + 100; /*100, 101, 102, 100, ...*/
    } else {
      A[i] = (i % 4) * 100 - 150; /*-150, -50, 50, 150, -150, ...*/
    }
    B[i] = (i % 3);
    C[i] = 0;
  }
};

// shortcuts for less clutter
template <class T1, class T2, class C> using VSf = verify_strict<T1, T2, C>;
template <class T1, class T2, class C> using VEf = verify_epsilon<T1, T2, C>;
template <class T1, class T2, class C> using VNf = verify_n<T1, T2, C>;
template <class T1, class T2, class C> using IDf = init_default<T1, T2, C>;
template <class T1, class T2, class C> using ISf = init_for_shift<T1, T2, C>;

int main(void) {
  queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler());

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";
  bool passed = true;
  using BinOp = esimd_test::BinaryOp;

  auto arith_ops = esimd_test::ArithBinaryOps;
  passed &= test<unsigned char, int, 1, BinOp, VSf, IDf>(arith_ops, q);
  passed &= test<char, float, 7, BinOp, VEf, IDf>(arith_ops, q, 0.000001f);
  passed &= test<short, double, 7, BinOp, VSf, IDf>(arith_ops, q);
  passed &= test<float, float, 32, BinOp, VEf, IDf>(arith_ops, q, 0.000001f);
  passed &= test<half, char, 1, BinOp, verify_n, IDf>(arith_ops, q, 1);
  passed &= test<half, unsigned int, 32, BinOp, VSf, IDf>(arith_ops, q, 1);
  passed &= test<double, half, 7, BinOp, VSf, IDf>(arith_ops, q);
  passed &= test<short, uint64_t, 7, BinOp, VSf, IDf>(arith_ops, q);

  auto int_ops =
      esimd_test::IntBinaryOpsNoShift; // different data needed for shift
  passed &= test<unsigned char, unsigned int, 1, BinOp, VSf, IDf>(int_ops, q);
  passed &= test<char, uint64_t, 1, BinOp, VSf, IDf>(int_ops, q);
  passed &= test<uint64_t, char, 32, BinOp, VSf, IDf>(int_ops, q);
  passed &= test<int, short, 1, BinOp, VSf, IDf>(int_ops, q);
  passed &= test<short, int, 8, BinOp, VSf, IDf>(int_ops, q);
  passed &= test<int, int, 7, BinOp, VSf, IDf>(int_ops, q);

  auto sh_ops = esimd_test::BinaryOpSeq<BinOp::shl, BinOp::shr>{};
  passed &= test<unsigned char, unsigned int, 1, BinOp, VSf, ISf>(sh_ops, q);
  passed &= test<char, int64_t, 1, BinOp, VSf, ISf>(sh_ops, q);
  passed &= test<uint64_t, char, 32, BinOp, VSf, ISf>(sh_ops, q);
  passed &= test<int, short, 1, BinOp, VSf, ISf>(sh_ops, q);
  passed &= test<short, int, 8, BinOp, VSf, ISf>(sh_ops, q);
  passed &= test<int, int, 7, BinOp, VSf, ISf>(sh_ops, q);

  using CmpOp = esimd_test::CmpOp;
  auto cmp_ops = esimd_test::CmpOps;
  passed &= test<unsigned char, int, 1, CmpOp, VSf, IDf>(cmp_ops, q);
  passed &= test<char, float, 7, CmpOp, VSf, IDf>(cmp_ops, q);
  passed &= test<short, double, 7, CmpOp, VSf, IDf>(cmp_ops, q);
  passed &= test<float, float, 32, CmpOp, VSf, IDf>(cmp_ops, q);
  passed &= test<half, char, 1, CmpOp, VSf, IDf>(cmp_ops, q, 1);
  passed &= test<half, unsigned int, 32, CmpOp, VSf, IDf>(cmp_ops, q, 1);
  passed &= test<double, half, 7, CmpOp, VSf, IDf>(cmp_ops, q);
  passed &= test<short, uint64_t, 7, CmpOp, VSf, IDf>(cmp_ops, q);

  std::cout << (passed ? "Test PASSED\n" : "Test FAILED\n");
  return passed ? 0 : 1;
}
