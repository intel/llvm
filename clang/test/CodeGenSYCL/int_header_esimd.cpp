// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -fsycl-int-header=%t.h %s
// RUN: FileCheck -input-file=%t.h %s

// This test checks that
// 1) New isESIMD() member is generated into the integration header
// 2) It returns 1 for ESIMD kernels and 0 - for non-ESIMD.

#include "sycl.hpp"

using namespace cl::sycl;

// -- ESIMD Lambda kernel.

void testA() {
  queue q;
  q.submit([&](handler &h) {
    h.single_task<class KernelA>([=]() __attribute__((sycl_explicit_simd)){});
  });
}
// CHECK-LABEL: template <> struct KernelInfo<class KernelA> {
// CHECK:   static constexpr bool isESIMD() { return 1; }

// --  ESIMD Functor object kernel.

struct KernelFunctor {
  void operator()() const __attribute__((sycl_explicit_simd)) {}
};

void testB() {
  queue q;
  q.submit([&](handler &h) {
    h.single_task(KernelFunctor{});
  });
}
// CHECK-LABEL: template <> struct KernelInfo<::KernelFunctor> {
// CHECK:   static constexpr bool isESIMD() { return 1; }

// -- Non-ESIMD Lambda kernel.

void testNA() {
  queue q;
  q.submit([&](handler &h) {
    h.single_task<class KernelNA>([=]() {});
  });
}
// CHECK-LABEL: template <> struct KernelInfo<class KernelNA> {
// CHECK:   static constexpr bool isESIMD() { return 0; }

// --  Non-ESIMD Functor object kernel.

struct KernelNonESIMDFunctor {
  void operator()() const {}
};

void testNB() {
  queue q;
  q.submit([&](handler &h) {
    h.single_task(KernelNonESIMDFunctor{});
  });
}
// CHECK-LABEL: template <> struct KernelInfo<::KernelNonESIMDFunctor> {
// CHECK:   static constexpr bool isESIMD() { return 0; }
