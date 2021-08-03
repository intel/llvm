// RUN: %clang_cc1 -fsycl-is-device -fno-sycl-unnamed-lambda -internal-isystem %S/Inputs -fsycl-int-header=%t.h %s
// RUN: FileCheck -input-file=%t.h %s --check-prefixes=CHECK,NUL
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -fsycl-int-header=%t.h %s
// RUN: FileCheck -input-file=%t.h %s --check-prefixes=CHECK,UL

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
// CHECK-LABEL: template <> struct KernelInfo<KernelA> {
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
// NUL-LABEL: template <> struct KernelInfo<::KernelFunctor> {
// UL-LABEL: template <> struct KernelInfoData<'_', 'Z', 'T', 'S', '1', '3', 'K', 'e', 'r', 'n', 'e', 'l', 'F', 'u', 'n', 'c', 't', 'o', 'r'> {
// CHECK:   static constexpr bool isESIMD() { return 1; }

// -- Non-ESIMD Lambda kernel.

void testNA() {
  queue q;
  q.submit([&](handler &h) {
    h.single_task<class KernelNA>([=]() {});
  });
}
// CHECK-LABEL: template <> struct KernelInfo<KernelNA> {
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
// NUL-LABEL: template <> struct KernelInfo<::KernelNonESIMDFunctor> {
// UL-LABEL: template <> struct KernelInfoData<'_', 'Z', 'T', 'S', '2', '1', 'K', 'e', 'r', 'n', 'e', 'l', 'N', 'o', 'n', 'E', 'S', 'I', 'M', 'D', 'F', 'u', 'n', 'c', 't', 'o', 'r'> {
// CHECK:   static constexpr bool isESIMD() { return 0; }
