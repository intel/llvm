// RUN: %clangxx -fsycl -fsycl-device-only -Xclang -emit-llvm -o %t.comp.ll %s
// RUN: sycl-post-link -ir-output-only -lower-esimd -S %t.comp.ll -o %t.out.ll
// RUN: FileCheck --input-file=%t.out.ll %s

// This test verifies that calls of ext::intel::experimental::esimd::wait()
// are lowered properly, not deleted as unused and they do not let delete
// the argument of wait() function.

#include <iostream>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::intel::esimd;
namespace iesimd = sycl::ext::intel::experimental::esimd;

SYCL_ESIMD_FUNCTION SYCL_EXTERNAL void func(int IArg) {
  // Test case 1: check wait() with esimd::simd argument.
  {
    simd<int, 16> A = IArg;
    simd<int, 16> B = A * A;
    iesimd::wait(B);
    // CHECK: mul <16 x i32>
    // CHECK: llvm.genx.dummy.mov
  }

  // Test case 2: check wait() with esimd::simd_view argument.
  {
    simd<int, 16> A = IArg;
    simd<int, 16> B = A * 17;
    auto BView = B.select<8, 2>(0);
    BView += 2;
    iesimd::wait(BView);
    // CHECK: mul <16 x i32>
    // CHECK: add <8 x i32>
    // CHECK: llvm.genx.dummy.mov
  }

  // Test case 3: check wait() that preserves one simd and lets
  // optimize away the other/unused one.
  {
    simd<uint64_t, 8> A = IArg;
    auto B = A * 17;
    iesimd::wait(B);
    auto C = B * 17;
    // CHECK: mul <8 x i64>
    // CHECK-NOT: add <8 x i64>
    // CHECK: llvm.genx.dummy.mov
    // CHECK-NEXT: ret void
  }
}
