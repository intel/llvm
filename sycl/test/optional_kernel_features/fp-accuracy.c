// Tests that kernels which use different fp-accuracy level end up in different
// device images.

// 1. Accuracy is specified for particular math functions.
// RUN: %clangxx %s -o %test_func.bc -ffp-accuracy=high:sin,sqrt -ffp-accuracy=medium:cos -ffp-accuracy=low:tan -ffp-accuracy=cuda:exp,acos -ffp-accuracy=sycl:log,asin  -fno-math-errno  -fsycl -fsycl-device-only
// RUN: sycl-post-link -properties -split=auto -symbols %test_func.bc -o %test_func.table
// RUN: FileCheck %s -input-file=%test_func.table --check-prefixes CHECK-FUNC-TABLE
// RUN: FileCheck %s -input-file=%test_func_0.sym --check-prefixes CHECK-FUNC-M0-SYMS
// RUN: FileCheck %s -input-file=%test_func_1.sym --check-prefixes CHECK-FUNC-M1-SYMS
// RUN: FileCheck %s -input-file=%test_func_2.sym --check-prefixes CHECK-FUNC-M2-SYMS
// RUN: FileCheck %s -input-file=%test_func_3.sym --check-prefixes CHECK-FUNC-M3-SYMS
// RUN: FileCheck %s -input-file=%test_func_4.sym --check-prefixes CHECK-FUNC-M4-SYMS
// RUN: FileCheck %s -input-file=%test_func_5.sym --check-prefixes CHECK-FUNC-M5-SYMS

// 2. Accuracy is specified for TU.
// RUN: %clangxx %s -o %test_tu.bc -ffp-accuracy=high -fno-math-errno -fsycl -fsycl-device-only
// RUN: sycl-post-link -properties -split=auto -symbols %test_tu.bc -o %test_tu.table
// RUN: FileCheck %s -input-file=%test_tu.table --check-prefixes CHECK-TU-TABLE
// RUN: FileCheck %s -input-file=%test_tu_0.sym --check-prefixes CHECK-TU-M0-SYMS
// RUN: FileCheck %s -input-file=%test_tu_1.sym --check-prefixes CHECK-TU-M1-SYMS

// 3. Mixed case.
// RUN: %clangxx %s -o %test_mix.bc -ffp-accuracy=medium -ffp-accuracy=high:sin,sqrt -ffp-accuracy=medium:cos -ffp-accuracy=cuda:exp -ffp-accuracy=sycl:log  -fno-math-errno  -fsycl -fsycl-device-only
// RUN: sycl-post-link -properties -split=auto -symbols %test_mix.bc -o %test_mix.table
// RUN: FileCheck %s -input-file=%test_mix.table --check-prefixes CHECK-MIX-TABLE
// RUN: FileCheck %s -input-file=%test_mix_0.sym --check-prefixes CHECK-MIX-M0-SYMS
// RUN: FileCheck %s -input-file=%test_mix_1.sym --check-prefixes CHECK-MIX-M1-SYMS
// RUN: FileCheck %s -input-file=%test_mix_2.sym --check-prefixes CHECK-MIX-M2-SYMS
// RUN: FileCheck %s -input-file=%test_mix_3.sym --check-prefixes CHECK-MIX-M3-SYMS

// CHECK-FUNC-TABLE: Code
// CHECK-FUNC-TABLE-NEXT: _0.sym
// CHECK-FUNC-TABLE-NEXT: _1.sym
// CHECK-FUNC-TABLE-NEXT: _2.sym
// CHECK-FUNC-TABLE-NEXT: _3.sym
// CHECK-FUNC-TABLE-NEXT: _4.sym
// CHECK-FUNC-TABLE-NEXT: _5.sym
// CHECK-FUNC-TABLE-NEXT: _6.sym
// CHECK-FUNC-TABLE-EMPTY:

// CHECK-TU-TABLE: Code
// CHECK-TU-TABLE-NEXT: _0.sym
// CHECK-TU-TABLE-NEXT: _1.sym
// CHECK-TU-TABLE-EMPTY:

// CHECK-MIX-TABLE: Code
// CHECK-MIX-TABLE-NEXT: _0.sym
// CHECK-MIX-TABLE-NEXT: _1.sym
// CHECK-MIX-TABLE-NEXT: _2.sym
// CHECK-MIX-TABLE-NEXT: _3.sym
// CHECK-MIX-TABLE-EMPTY:

// CHECK-FUNC-M0-SYMS: __pf_kernel_wrapper{{.*}}Kernel1
// CHECK-FUNC-M0-SYMS-NEXT: Kernel1
// CHECK-FUNC-M0-SYMS-NEXT: __pf_kernel_wrapper{{.*}}Kernel7
// CHECK-FUNC-M0-SYMS-NEXT: Kernel7
// CHECK-FUNC-M0-SYMS-EMPTY:

// CHECK-FUNC-M1-SYMS: __pf_kernel_wrapper{{.*}}Kernel2
// CHECK-FUNC-M1-SYMS-NEXT: Kernel2
// CHECK-FUNC-M1-SYMS-EMPTY:

// CHECK-FUNC-M2-SYMS: __pf_kernel_wrapper{{.*}}Kernel3
// CHECK-FUNC-M2-SYMS-NEXT: Kernel3
// CHECK-FUNC-M2-SYMS-EMPTY:

// CHECK-FUNC-M3-SYMS: __pf_kernel_wrapper{{.*}}Kernel6
// CHECK-FUNC-M3-SYMS-NEXT: Kernel6
// CHECK-FUNC-M3-SYMS-EMPTY:

// CHECK-FUNC-M4-SYMS: __pf_kernel_wrapper{{.*}}Kernel4
// CHECK-FUNC-M4-SYMS-NEXT: Kernel4
// CHECK-FUNC-M4-SYMS-EMPTY:

// CHECK-FUNC-M5-SYMS: __pf_kernel_wrapper{{.*}}Kernel5
// CHECK-FUNC-M5-SYMS-NEXT: Kernel5
// CHECK-FUNC-M5-SYMS-EMPTY:

// CHECK-FUNC-M6-SYMS: __pf_kernel_wrapper{{.*}}Kernel0
// CHECK-FUNC-M6-SYMS-NEXT: Kernel0
// CHECK-FUNC-M6-SYMS-EMPTY:

// CHECK-TU-M0-SYMS: __pf_kernel_wrapper{{.*}}Kernel1
// CHECK-TU-M0-SYMS-NEXT: Kernel1
// CHECK-TU-M0-SYMS-NEXT: __pf_kernel_wrapper{{.*}}Kernel2
// CHECK-TU-M0-SYMS-NEXT: Kernel2
// CHECK-TU-M0-SYMS-NEXT: __pf_kernel_wrapper{{.*}}Kernel3
// CHECK-TU-M0-SYMS-NEXT: Kernel3
// CHECK-TU-M0-SYMS-NEXT: __pf_kernel_wrapper{{.*}}Kernel4
// CHECK-TU-M0-SYMS-NEXT: Kernel4
// CHECK-TU-M0-SYMS-NEXT: __pf_kernel_wrapper{{.*}}Kernel5
// CHECK-TU-M0-SYMS-NEXT: Kernel5
// CHECK-TU-M0-SYMS-NEXT: __pf_kernel_wrapper{{.*}}Kernel6
// CHECK-TU-M0-SYMS-NEXT: Kernel6
// CHECK-TU-M0-SYMS-NEXT: __pf_kernel_wrapper{{.*}}Kernel7
// CHECK-TU-M0-SYMS-NEXT: Kernel7
// CHECK-TU-M6-SYMS-EMPTY:

// CHECK-TU-M1-SYMS: __pf_kernel_wrapper{{.*}}Kernel0
// CHECK-TU-M1-SYMS-NEXT: Kernel0
// CHECK-TU-M1-SYMS-EMPTY:

// CHECK-MIX-M0-SYMS: __pf_kernel_wrapper{{.*}}Kernel1
// CHECK-MIX-M0-SYMS-NEXT: Kernel1
// CHECK-MIX-M0-SYMS-NEXT: __pf_kernel_wrapper{{.*}}Kernel7
// CHECK-MIX-M0-SYMS-NEXT: Kernel7
// CHECK-MIX-M0-SYMS-EMPTY:

// CHECK-MIX-M1-SYMS: __pf_kernel_wrapper{{.*}}Kernel2
// CHECK-MIX-M1-SYMS-NEXT: Kernel2
// CHECK-MIX-M1-SYMS-NEXT: __pf_kernel_wrapper{{.*}}Kernel3
// CHECK-MIX-M1-SYMS-NEXT: Kernel3
// CHECK-MIX-M1-SYMS-NEXT: __pf_kernel_wrapper{{.*}}Kernel5
// CHECK-MIX-M1-SYMS-NEXT: Kernel5
// CHECK-MIX-M1-SYMS-NEXT: __pf_kernel_wrapper{{.*}}Kernel6
// CHECK-MIX-M1-SYMS-NEXT: Kernel6
// CHECK-MIX-M1-SYMS-EMPTY:

// CHECK-MIX-M2-SYMS: __pf_kernel_wrapper{{.*}}Kernel4
// CHECK-MIX-M2-SYMS-NEXT: Kernel4
// CHECK-MIX-M2-SYMS-EMPTY:

// CHECK-MIX-M3-SYMS: __pf_kernel_wrapper{{.*}}Kernel0
// CHECK-MIX-M3-SYMS-NEXT: Kernel0
// CHECK-MIX-M3-SYMS-EMPTY:

#include <array>
#include <cmath>
#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;

constexpr access::mode sycl_read = access::mode::read;
constexpr access::mode sycl_write = access::mode::write;

int main() {
  const size_t array_size = 4;
  std::array<double, array_size> D = {{1., 2., 3., 4.}}, E;
  queue deviceQueue;
  range<1> numOfItems{array_size};
  double Value = 5.;
  buffer<double, 1> bufferOut(E.data(), numOfItems);

  // Kernel0 doesn't use math functions.
  deviceQueue.submit([&](handler &cgh) {
    auto accessorOut = bufferOut.template get_access<sycl_write>(cgh);

    cgh.parallel_for<class Kernel0>(
        numOfItems, [=](id<1> wiID) { accessorOut[wiID] = Value; });
  });

  // Kernel1 uses high-accuracy sin.
  deviceQueue.submit([&](handler &cgh) {
    auto accessorOut = bufferOut.template get_access<sycl_write>(cgh);

    cgh.parallel_for<class Kernel1>(
        numOfItems, [=](id<1> wiID) { accessorOut[wiID] = std::sin(Value); });
  });

  // Kernel2 uses:
  // 1. medium-accuracy cos
  // 2. high-accuracy cos
  // 3. medium-accuracy cos
  deviceQueue.submit([&](handler &cgh) {
    auto accessorOut = bufferOut.template get_access<sycl_write>(cgh);

    cgh.parallel_for<class Kernel2>(
        numOfItems, [=](id<1> wiID) { accessorOut[wiID] = std::cos(Value); });
  });

  // Kernel3 uses:
  // 1. low-accuracy tan
  // 2. high-accuracy tan
  // 3. medium-accuracy tan.
  deviceQueue.submit([&](handler &cgh) {
    auto accessorOut = bufferOut.template get_access<sycl_write>(cgh);

    cgh.parallel_for<class Kernel3>(
        numOfItems, [=](id<1> wiID) { accessorOut[wiID] = std::tan(Value); });
  });

  // Kernel4 uses:
  // 1. cuda-accuracy exp and sycl-accuracy log.
  // 2. high-accuracy exp and high-accuracy log.
  // 3. cuda-accuracy exp and sycl-accuracy log.
  deviceQueue.submit([&](handler &cgh) {
    auto accessorOut = bufferOut.template get_access<sycl_write>(cgh);

    cgh.parallel_for<class Kernel4>(numOfItems, [=](id<1> wiID) {
      accessorOut[wiID] = std::log(std::exp(Value));
    });
  });

  // Kernel5 uses:
  // 1. cuda-accuracy acos.
  // 1. high-accuracy acos.
  // 1. medium-accuracy acos.
  deviceQueue.submit([&](handler &cgh) {
    auto accessorOut = bufferOut.template get_access<sycl_write>(cgh);

    cgh.parallel_for<class Kernel5>(
        numOfItems, [=](id<1> wiID) { accessorOut[wiID] = std::acos(Value); });
  });

  // Kernel6 uses:
  // 1. sycl-accuracy acos.
  // 1. high-accuracy acos.
  // 1. medium-accuracy acos.
  deviceQueue.submit([&](handler &cgh) {
    auto accessorOut = bufferOut.template get_access<sycl_write>(cgh);

    cgh.parallel_for<class Kernel6>(
        numOfItems, [=](id<1> wiID) { accessorOut[wiID] = std::asin(Value); });
  });

  // Kernel7 uses high-accuracy sqrt.
  deviceQueue.submit([&](handler &cgh) {
    auto accessorOut = bufferOut.template get_access<sycl_write>(cgh);

    cgh.parallel_for<class Kernel7>(
        numOfItems, [=](id<1> wiID) { accessorOut[wiID] = std::sqrt(Value); });
  });

  return 0;
}
