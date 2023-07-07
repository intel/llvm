// RUN: %clang_cc1  -fsycl-is-device -ffp-builtin-accuracy=high:sin,sqrt -ffp-builtin-accuracy=medium:cos -ffp-builtin-accuracy=low:tan -ffp-builtin-accuracy=cuda:exp,acos -ffp-builtin-accuracy=sycl:log,asin -emit-llvm -triple spir64-unknown-unknown -disable-llvm-passes %s -o - | FileCheck %s

// Tests that sycl_used_aspects metadata is attached to the fpbuiltin call based on -ffp-accuracy option.

#include "Inputs/sycl.hpp"

extern "C" SYCL_EXTERNAL double sin(double);
extern "C" SYCL_EXTERNAL double cos(double);
extern "C" SYCL_EXTERNAL double tan(double);
extern "C" SYCL_EXTERNAL double log(double);
extern "C" SYCL_EXTERNAL double exp(double);
extern "C" SYCL_EXTERNAL double acos(double);
extern "C" SYCL_EXTERNAL double asin(double);
extern "C" SYCL_EXTERNAL double sqrt(double);

using namespace sycl;

int main() {
  const unsigned array_size = 4;
  double Value = .5;
  queue deviceQueue;
  range<1> numOfItems{array_size};

  // Kernel0 doesn't use math functions.
  deviceQueue.submit([&](handler& cgh) {
    cgh.parallel_for<class Kernel0>(numOfItems,
    [=](id<1> wiID) {
      (void)Value;
    });
  });

  // Kernel1 uses high-accuracy sin.
  deviceQueue.submit([&](handler& cgh) {
    cgh.parallel_for<class Kernel1>(numOfItems,
    [=](id<1> wiID) {
// CHECK: call double @llvm.fpbuiltin.sin.f64(double {{.*}}) #[[ATTR:[0-9]+]], !sycl_used_aspects ![[ASPECT1:[0-9]+]]
      (void)sin(Value);
    });
  });

  deviceQueue.submit([&](handler& cgh) {
    cgh.parallel_for<class Kernel2>(numOfItems,
    [=](id<1> wiID) {
// CHECK: call double @llvm.fpbuiltin.cos.f64(double {{.*}}) #[[ATTR:[0-9]+]], !sycl_used_aspects ![[ASPECT2:[0-9]+]]
      (void)cos(Value);
    });
  });

  // Kernel3 uses low-accuracy tan.
  deviceQueue.submit([&](handler& cgh) {
    cgh.parallel_for<class Kernel3>(numOfItems,
    [=](id<1> wiID) {
// CHECK: call double @llvm.fpbuiltin.tan.f64(double {{.*}}) #[[ATTR:[0-9]+]], !sycl_used_aspects ![[ASPECT3:[0-9]+]]
      (void)tan(Value);
    });
  });

  // Kernel4 uses cuda-accuracy exp and sycl-accuracy log.
  deviceQueue.submit([&](handler& cgh) {
    cgh.parallel_for<class Kernel4>(numOfItems,
    [=](id<1> wiID) {
// CHECK: call double @llvm.fpbuiltin.exp.f64(double {{.*}}) #[[ATTR:[0-9]+]], !sycl_used_aspects ![[ASPECT4:[0-9]+]]
// CHECK: call double @llvm.fpbuiltin.log.f64(double {{.*}}) #[[ATTR:[0-9]+]], !sycl_used_aspects ![[ASPECT5:[0-9]+]]
      (void)log(exp(Value));
    });
  });
  deviceQueue.wait();

  // Kernel5 uses cuda-accuracy acos.
  deviceQueue.submit([&](handler& cgh) {
    cgh.parallel_for<class Kernel5>(numOfItems,
    [=](id<1> wiID) {
// CHECK: call double @llvm.fpbuiltin.acos.f64(double {{.*}}) #[[ATTR:[0-9]+]], !sycl_used_aspects ![[ASPECT4:[0-9]+]]
      (void)acos(Value);
    });
  });

  // Kernel6 uses sycl-accuracy asin.
  deviceQueue.submit([&](handler& cgh) {
    cgh.parallel_for<class Kernel6>(numOfItems,
    [=](id<1> wiID) {
// CHECK: call double @llvm.fpbuiltin.asin.f64(double {{.*}}) #[[ATTR:[0-9]+]], !sycl_used_aspects ![[ASPECT5:[0-9]+]]
      (void)asin(Value);
    });
  });

  // Kernel7 uses high-accuracy sqrt.
  deviceQueue.submit([&](handler& cgh) {
    cgh.parallel_for<class Kernel7>(numOfItems,
    [=](id<1> wiID) {
// CHECK: call double @llvm.fpbuiltin.sqrt.f64(double {{.*}}) #[[ATTR:[0-9]+]], !sycl_used_aspects ![[ASPECT1:[0-9]+]]
      (void)sqrt(Value);
    });
  });
  return 0;
}

// CHECK: [[ASPECT1]] = !{i32 -1}
// CHECK: [[ASPECT2]] = !{i32 -2}
// CHECK: [[ASPECT3]] = !{i32 -3}
// CHECK: [[ASPECT4]] = !{i32 -5}
// CHECK: [[ASPECT5]] = !{i32 -4}
