//==--------------- unary_ops_heavy.cpp  - DPC++ ESIMD on-device test ------==//
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

// Tests various unary operations applied to simd objects.

// TODO
// Arithmetic operations behaviour depends on Gen's control regiter's rounding
// mode, which is RTNE by default:
//    cr0.5:4 is 00b = Round to Nearest or Even (RTNE)
// For half this leads to divergence between Gen and host (emulated) results
// larger than certain threshold. Might need to tune the cr0 once this feature
// is available in ESIMD.
//

#include "../esimd_test_utils.hpp"

#include <CL/sycl.hpp>
#include <iostream>
#include <sycl/ext/intel/esimd.hpp>

using namespace cl::sycl;
using namespace sycl::ext::intel::esimd;

template <class T, int VL, class Ops> class TestID;

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
// T - operand type,
// VL - vector length,
// Ops - a compile-time sequence of operations to test.
//
template <class T, int VL, class Ops, template <class, int> class SimdT = simd>
bool test(Ops ops, queue &q) {
  using OpClass = esimd_test::UnaryOp;
  // Log test case info
  std::cout << "Testing T=" << typeid(T).name() << ", VL=" << VL << " ...\n";
  std::cout << "Operations:";
  esimd_test::iterate_ops(ops, [=](OpClass op) {
    std::cout << " '" << esimd_test::Op2Str(op) << "'";
  });
  std::cout << "\n";

  // initialize test data
  constexpr int Size = 1024 * 7;
  T *A = sycl::malloc_shared<T>(Size, q);
  constexpr int NumOps = (int)Ops::size;
  int CSize = NumOps * Size;
  T *C = sycl::malloc_shared<T>(CSize, q);

  for (int i = 0; i < Size; ++i) {
    if constexpr (std::is_unsigned_v<T>) {
      A[i] = i;
    } else {
      A[i] = i - Size / 2;
    }
    C[i] = 0;
  }

  // submit the kernel
  try {
    auto e = q.submit([&](handler &cgh) {
      cgh.parallel_for<TestID<T, VL, Ops>>(
          Size / VL, [=](id<1> i) SYCL_ESIMD_KERNEL {
            unsigned off = i * VL;
            SimdT<T, VL> va(A + off);
            // applies each of the input operations to the va,
            // then invokes the lambda below, passing the result of the
            // operation, its ID and sequential number within the input sequence
            esimd_test::apply_unary_ops(
                ops, va, [=](SimdT<T, VL> res, OpClass op, unsigned op_num) {
                  unsigned res_off = off * NumOps + op_num * VL;
                  res.copy_to(C + res_off);
                });
          });
    });
    e.wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    sycl::free(A, q);
    sycl::free(C, q);
    return false;
  }

  int err_cnt = 0;

  // now verify the results using provided verification function type
  for (unsigned i = 0; i < Size / VL; ++i) {
    unsigned off = i * VL;

    for (int j = 0; j < VL; ++j) {
      T a = A[off + j];

      esimd_test::apply_unary_ops(
          ops, a, [&](T Gold, OpClass op, unsigned op_num) {
            unsigned res_off = off * NumOps + op_num * VL;
            T Res = C[res_off + j];
            using Tint = esimd_test::int_type_t<sizeof(T)>;
            Tint ResBits = *(Tint *)&Res;
            Tint GoldBits = *(Tint *)&Gold;
            // allow 1 bit discrepancy for half on modifying op
            int delta = ((int)op >= (int)OpClass::minus_minus_pref) &&
                                ((int)op <= (int)OpClass::plus_plus_inf) &&
                                std::is_same_v<T, half>
                            ? 1
                            : 0;

            if ((Gold != Res) && (abs(ResBits - GoldBits) > delta)) {
              if (++err_cnt < 10) {
                std::cout << "  failed at index " << (res_off + j) << ", op "
                          << esimd_test::Op2Str(op) << ": " << cast(Res)
                          << "(0x" << std::hex << ResBits << ")"
                          << " != " << cast(Gold) << "(0x" << std::hex
                          << GoldBits << ") [" << esimd_test::Op2Str(op) << " "
                          << std::dec << cast(a) << "]\n";
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
  free(C, q);
  std::cout << (err_cnt > 0 ? "  FAILED\n" : "  Passed\n");
  return err_cnt == 0;
}

int main(void) {
  queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler());

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";
  bool passed = true;
  using UnOp = esimd_test::UnaryOp;

  auto mod_ops =
      esimd_test::OpSeq<UnOp, UnOp::minus_minus_pref, UnOp::minus_minus_inf,
                        UnOp::plus_plus_pref, UnOp::plus_plus_inf>{};
  passed &= test<char, 7>(mod_ops, q);
  passed &= test<unsigned char, 1>(mod_ops, q);
  passed &= test<short, 7>(mod_ops, q);
  passed &= test<unsigned short, 7>(mod_ops, q);
  passed &= test<int, 16>(mod_ops, q);
  passed &= test<unsigned int, 8>(mod_ops, q);
  passed &= test<int64_t, 16>(mod_ops, q);
  passed &= test<uint64_t, 1>(mod_ops, q);
  passed &= test<half, 1>(mod_ops, q);
  passed &= test<half, 32>(mod_ops, q);
  passed &= test<float, 32>(mod_ops, q);
  passed &= test<double, 7>(mod_ops, q);

  auto singed_ops = esimd_test::OpSeq<UnOp, UnOp::minus, UnOp::plus>{};
  passed &= test<char, 7>(singed_ops, q);
  passed &= test<short, 7>(singed_ops, q);
  passed &= test<int, 16>(singed_ops, q);
  passed &= test<int64_t, 16>(singed_ops, q);
  passed &= test<half, 16>(singed_ops, q);
  passed &= test<float, 16>(singed_ops, q);
  passed &= test<double, 16>(singed_ops, q);

  auto bit_ops = esimd_test::OpSeq<UnOp, UnOp::bit_not>{};
  passed &= test<char, 7>(bit_ops, q);
  passed &= test<unsigned char, 1>(bit_ops, q);
  passed &= test<short, 7>(bit_ops, q);
  passed &= test<unsigned short, 7>(bit_ops, q);
  passed &= test<int, 16>(bit_ops, q);
  passed &= test<unsigned int, 8>(bit_ops, q);
  passed &= test<int64_t, 16>(bit_ops, q);
  passed &= test<uint64_t, 1>(bit_ops, q);

  auto not_ops = esimd_test::OpSeq<UnOp, UnOp::log_not, UnOp::bit_not>{};
  passed &= test<simd_mask<1>::element_type, 7, decltype(not_ops),
                 __ESIMD_DNS::simd_mask_impl>(not_ops, q);

  std::cout << (passed ? "Test passed\n" : "Test FAILED\n");
  return passed ? 0 : 1;
}
