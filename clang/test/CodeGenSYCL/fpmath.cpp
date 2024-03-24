// RUN: %clang_cc1 -triple spir64-unknown-unknown -fsycl-is-device -emit-llvm %s -o - | FileCheck --check-prefix=CHECK-FPMATH %s
// RUN: %clang_cc1 -cl-fp32-correctly-rounded-divide-sqrt -triple spir64-unknown-unknown -fsycl-is-device -emit-llvm %s -o - | FileCheck %s --implicit-check-not='!fpmath'

#include "Inputs/sycl.hpp"

using namespace sycl;

SYCL_EXTERNAL float sqrt(float);
SYCL_EXTERNAL double sqrt(double);

int main() {
  queue q;
  range<1> numOfItems{1};
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<class Kernel0>(numOfItems,[=](id<1> wiID) {
    float a = 0;
    float b = 1;
    // CHECK-FPMATH: fdiv float {{[^)]+}}, {{[^)]+}}, !fpmath ![[#DivMD:]]
    float c = a/b;
    (void) c;
    double d = (double)a/b;
    // CHECK-FPMATH-NOT: fdiv double{{.*}}!fpmath
    (void) d;
    });
  });

   q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<class Kernel1>(numOfItems,[=](id<1> wiID) {

    float a = 1;
    // CHECK-FPMATH: call spir_func noundef float @_Z4sqrtf(float noundef {{[^)]+}}) #{{[0-9]+}}, !fpmath ![[#SqrtMD:]]
    float b = sqrt(a);
    (void) b;
    // CHECK-FPMATH-NOT: call spir_func noundef double @_Z4sqrtd{{.*}}!fpmath
    sqrt((double)a);
    });
  });
}

// CHECK-FPMATH: ![[#DivMD]] = !{float 2.500000e+00}
// CHECK-FPMATH: ![[#SqrtMD]] = !{float 3.000000e+00}
