// RUN: %clangxx %s -o %test.bc -ffp-accuracy=high:sin,sqrt -ffp-accuracy=medium:cos -ffp-accuracy=low:tan -ffp-accuracy=cuda:exp,acos -ffp-accuracy=sycl:log,asin  -fno-math-errno  -fsycl -fsycl-device-only
// RUN: sycl-post-link -split=auto -symbols %test.bc -o %test.table
// RUN: FileCheck %s -input-file=%test.table --check-prefixes CHECK-TABLE
// RUN: FileCheck %s -input-file=%test_0.sym --check-prefixes CHECK-M0-SYMS
// RUN: FileCheck %s -input-file=%test_1.sym --check-prefixes CHECK-M1-SYMS
// RUN: FileCheck %s -input-file=%test_2.sym --check-prefixes CHECK-M2-SYMS
// RUN: FileCheck %s -input-file=%test_3.sym --check-prefixes CHECK-M3-SYMS
// RUN: FileCheck %s -input-file=%test_4.sym --check-prefixes CHECK-M4-SYMS
// RUN: FileCheck %s -input-file=%test_5.sym --check-prefixes CHECK-M5-SYMS

// Tests that kernels which use different fp-accuracy level end up in different
// device images.

// CHECK-TABLE: Code
// CHECK-TABLE-NEXT: _0.sym
// CHECK-TABLE-NEXT: _1.sym
// CHECK-TABLE-NEXT: _2.sym
// CHECK-TABLE-NEXT: _3.sym
// CHECK-TABLE-NEXT: _4.sym
// CHECK-TABLE-NEXT: _5.sym
// CHECK-TABLE-NEXT: _6.sym
// CHECK-TABLE-EMPTY:

// CHECK-M0-SYMS: __pf_kernel_wrapper{{.*}}Kernel1
// CHECK-M0-SYMS-NEXT: Kernel1
// CHECK-M0-SYMS-NEXT: __pf_kernel_wrapper{{.*}}Kernel7
// CHECK-M0-SYMS-NEXT: Kernel7
// CHECK-M0-SYMS-EMPTY:

// CHECK-M1-SYMS: __pf_kernel_wrapper{{.*}}Kernel2
// CHECK-M1-SYMS-NEXT: Kernel2
// CHECK-M1-SYMS-EMPTY:

// CHECK-M2-SYMS: __pf_kernel_wrapper{{.*}}Kernel3
// CHECK-M2-SYMS-NEXT: Kernel3
// CHECK-M2-SYMS-EMPTY:

// CHECK-M3-SYMS: __pf_kernel_wrapper{{.*}}Kernel6
// CHECK-M3-SYMS-NEXT: Kernel6
// CHECK-M3-SYMS-EMPTY:

// CHECK-M4-SYMS: __pf_kernel_wrapper{{.*}}Kernel4
// CHECK-M4-SYMS-NEXT: Kernel4
// CHECK-M4-SYMS-EMPTY:

// CHECK-M5-SYMS: __pf_kernel_wrapper{{.*}}Kernel5
// CHECK-M5-SYMS-NEXT: Kernel5
// CHECK-M5-SYMS-EMPTY:

// CHECK-M6-SYMS: __pf_kernel_wrapper{{.*}}Kernel0
// CHECK-M6-SYMS-NEXT: Kernel0
// CHECK-M6-SYMS-EMPTY:

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

  // Kernel2 uses medium-accuracy cos.
  deviceQueue.submit([&](handler &cgh) {
    auto accessorOut = bufferOut.template get_access<sycl_write>(cgh);

    cgh.parallel_for<class Kernel2>(
        numOfItems, [=](id<1> wiID) { accessorOut[wiID] = std::cos(Value); });
  });

  // Kernel3 uses low-accuracy tan.
  deviceQueue.submit([&](handler &cgh) {
    auto accessorOut = bufferOut.template get_access<sycl_write>(cgh);

    cgh.parallel_for<class Kernel3>(
        numOfItems, [=](id<1> wiID) { accessorOut[wiID] = std::tan(Value); });
  });

  // Kernel4 uses cuda-accuracy exp and sycl-accuracy log.
  deviceQueue.submit([&](handler &cgh) {
    auto accessorOut = bufferOut.template get_access<sycl_write>(cgh);

    cgh.parallel_for<class Kernel4>(numOfItems, [=](id<1> wiID) {
      accessorOut[wiID] = std::log(std::exp(Value));
    });
  });

  // Kernel5 uses cuda-accuracy acos.
  deviceQueue.submit([&](handler &cgh) {
    auto accessorOut = bufferOut.template get_access<sycl_write>(cgh);

    cgh.parallel_for<class Kernel5>(
        numOfItems, [=](id<1> wiID) { accessorOut[wiID] = std::acos(Value); });
  });

  // Kernel6 uses sycl-accuracy asin.
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
