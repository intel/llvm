// RUN: clang++ -Xcgeist --use-opaque-pointers=1 -fsycl -fsycl-targets=spir64-unknown-unknown-syclmlir -O0 -w %s -o %t.O0.out 2>&1 | FileCheck %s --allow-empty --implicit-check-not="{{error|Error}}:"
// RUN: clang++ -Xcgeist --use-opaque-pointers=1 -fsycl -fsycl-targets=spir64-unknown-unknown-syclmlir -O1 -w %s -o %t.O1.out 2>&1 | FileCheck %s --allow-empty --implicit-check-not="{{error|Error}}:"
// RUN: clang++ -Xcgeist --use-opaque-pointers=1 -fsycl -fsycl-targets=spir64-unknown-unknown-syclmlir -O2 -w %s -o %t.O2.out 2>&1 | FileCheck %s --allow-empty --implicit-check-not="{{error|Error}}:"
// RUN: clang++ -Xcgeist --use-opaque-pointers=1 -fsycl -fsycl-targets=spir64-unknown-unknown-syclmlir -O3 -w %s -o %t.O3.out 2>&1 | FileCheck %s --allow-empty --implicit-check-not="{{error|Error}}:"
// RUN: clang++ -Xcgeist --use-opaque-pointers=1 -fsycl -fsycl-targets=spir64-unknown-unknown-syclmlir -Ofast -w %s -o %t.Ofast.out 2>&1 | FileCheck %s --allow-empty --implicit-check-not="{{error|Error}}:"

// RUN: clang++ -Xcgeist --use-opaque-pointers=1 -fsycl -fsycl-device-only -O0 -w -emit-llvm -fsycl-targets=spir64-unknown-unknown-syclmlir %s -o %t.bc

// Test that the LLVMIR generated is verifiable.
// RUN: opt -passes=verify -disable-output < %t.bc

// Verify that LLVMIR generated is translatable to SPIRV.
// RUN: llvm-spirv %t.bc

// Test that the SYCL math functions are lowered to llvm intrinsics.
// RUN: llvm-dis %t.bc
// RUN: cat %t.ll | FileCheck %s --check-prefix=LLVM

#include <climits>
#include <cmath>

#include <sycl/sycl.hpp>
using namespace sycl;

void host_math_funcs(std::array<float, 1> &A) {
  auto q = queue{};
  device d = q.get_device();
  std::cout << "Using " << d.get_info<info::device::name>() << "\n";

  {
    auto buf = buffer<float, 1>{A.data(), 1};
    q.submit([&](handler &cgh) {
      auto A = buf.get_access<access::mode::read_write>(cgh);
      cgh.single_task<class kernel_math_funcs>([=]() {
        // LLVM: define internal spir_func void @_ZZZ15host_math_funcsRSt5arrayIfLm1EEENKUlRN4sycl3_V17handlerEE_clES5_ENKUlvE_clEv
        float a = A[0], b, c;
        // LLVM: call float @llvm.ceil.f32
        a += b = sycl::ceil(a);
        // LLVM: call float @llvm.copysign.f32
        a += c = sycl::copysign(a, b);
        // LLVM: call float @llvm.cos.f32
        a += sycl::cos(a);
        // LLVM: call float @llvm.exp.f32
        a += sycl::exp(a);
        // LLVM: call float @llvm.exp2.f32
        a += sycl::exp2(a);
        // LLVM: %[[exp:.*]] = call float @llvm.exp.f32
        // LLVM-NEXT: fsub float %[[exp]], 1.000000e+00
        a += sycl::expm1(a);
        // LLVM: call float @llvm.fabs.f32
        a += sycl::fabs(a);
        // LLVM: call float @llvm.floor.f32
        a += sycl::floor(a);
        // LLVM: call float @llvm.fma.f32
        a += sycl::fma(a, b, c);
        // LLVM: call float @llvm.log.f32
        a += sycl::log(a);
        // LLVM: call float @llvm.log10.f32
        a += sycl::log10(a);
        // LLVM: call float @llvm.log2.f32
        a += sycl::log2(a);
        // LLVM: call float @llvm.pow.f32
        a += sycl::pow(a, a);
        // LLVM: call float @llvm.round.f32
        a += sycl::round(a);
        // LLVM: %[[sqrt:.*]] = call float @llvm.sqrt.f32
        // LLVM-NEXT: fdiv float 1.000000e+00, %[[sqrt]]
        a += sycl::rsqrt(a);
        // LLVM: call float @llvm.sin.f32
        a += sycl::sin(a);
        // LLVM: call float @llvm.sqrt.f32
        a += sycl::sqrt(a);
        // LLVM: call float @llvm.trunc.f32
        a += sycl::trunc(a);
        A[0] = a;
      });
    });
  }
}

int main() {
  std::array<float, 1> A = {0.0f};
  host_math_funcs(A);
  assert(std::fabs(A[0] - std::sqrt(2.0f)) < std::numeric_limits<float>::epsilon() );
  std::cout << "Test passed" << std::endl;
}
