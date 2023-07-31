// RUN: %clang_cc1 -internal-isystem %S/Inputs -fsycl-is-device -ffp-builtin-accuracy=high:sin,sqrt -ffp-builtin-accuracy=medium:cos -ffp-builtin-accuracy=low:tan -ffp-builtin-accuracy=cuda:exp,acos -ffp-builtin-accuracy=sycl:log,asin -emit-llvm -triple spir64-unknown-unknown %s -o - | FileCheck --check-prefix CHECK-FUNC %s
// RUN: %clang_cc1 -internal-isystem %S/Inputs -fsycl-is-device -ffp-builtin-accuracy=high -emit-llvm -triple spir64-unknown-unknown %s -o - | FileCheck --check-prefix CHECK-TU %s
// RUN: %clang_cc1 -internal-isystem %S/Inputs -fsycl-is-device -ffp-builtin-accuracy=medium -ffp-builtin-accuracy=high:sin,sqrt -ffp-builtin-accuracy=medium:cos -ffp-builtin-accuracy=cuda:exp -ffp-builtin-accuracy=sycl:log -emit-llvm -triple spir64-unknown-unknown %s -o - | FileCheck --check-prefix CHECK-MIX %s

// Tests that sycl_used_aspects metadata is attached to the fpbuiltin call based on -ffp-accuracy option.

#include "sycl.hpp"

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
// CHECK-FUNC: call double @llvm.fpbuiltin.sin.f64(double {{.*}}) #[[ATTR:[0-9]+]], !sycl_used_aspects ![[HIGH_ACC:[0-9]+]]
// CHECK-TU: call double @llvm.fpbuiltin.sin.f64(double {{.*}}) #[[ATTR:[0-9]+]], !sycl_used_aspects ![[HIGH_ACC:[0-9]+]]
// CHECK-MIX: call double @llvm.fpbuiltin.sin.f64(double {{.*}}) #[[ATTR:[0-9]+]], !sycl_used_aspects ![[HIGH_ACC:[0-9]+]]
      (void)sin(Value);
    });
  });

  deviceQueue.submit([&](handler& cgh) {
    cgh.parallel_for<class Kernel2>(numOfItems,
    [=](id<1> wiID) {
// CHECK-FUNC: call double @llvm.fpbuiltin.cos.f64(double {{.*}}) #[[ATTR:[0-9]+]], !sycl_used_aspects ![[MEDIUM_ACC:[0-9]+]]
// CHECK-TU: call double @llvm.fpbuiltin.cos.f64(double {{.*}}) #[[ATTR:[0-9]+]], !sycl_used_aspects ![[HIGH_ACC]]
// CHECK-MIX: call double @llvm.fpbuiltin.cos.f64(double {{.*}}) #[[ATTR:[0-9]+]], !sycl_used_aspects ![[MEDIUM_ACC:[0-9]+]]
      (void)cos(Value);
    });
  });

  // Kernel3 uses low-accuracy tan.
  deviceQueue.submit([&](handler& cgh) {
    cgh.parallel_for<class Kernel3>(numOfItems,
    [=](id<1> wiID) {
// CHECK-FUNC: call double @llvm.fpbuiltin.tan.f64(double {{.*}}) #[[ATTR:[0-9]+]], !sycl_used_aspects ![[LOW_ACC:[0-9]+]]
// CHECK-TU: call double @llvm.fpbuiltin.tan.f64(double {{.*}}) #[[ATTR:[0-9]+]], !sycl_used_aspects ![[HIGH_ACC]]
// CHECK-MIX: call double @llvm.fpbuiltin.tan.f64(double {{.*}}) #[[ATTR:[0-9]+]], !sycl_used_aspects ![[MEDIUM_ACC]]
      (void)tan(Value);
    });
  });

  // Kernel4 uses cuda-accuracy exp and sycl-accuracy log.
  deviceQueue.submit([&](handler& cgh) {
    cgh.parallel_for<class Kernel4>(numOfItems,
    [=](id<1> wiID) {
// CHECK-FUNC: call double @llvm.fpbuiltin.exp.f64(double {{.*}}) #[[ATTR:[0-9]+]], !sycl_used_aspects ![[CUDA_ACC:[0-9]+]]
// CHECK-FUNC: call double @llvm.fpbuiltin.log.f64(double {{.*}}) #[[ATTR:[0-9]+]], !sycl_used_aspects ![[SYCL_ACC:[0-9]+]]
// CHECK-TU: call double @llvm.fpbuiltin.exp.f64(double {{.*}}) #[[ATTR:[0-9]+]], !sycl_used_aspects ![[HIGH_ACC]]
// CHECK-TU: call double @llvm.fpbuiltin.log.f64(double {{.*}}) #[[ATTR:[0-9]+]], !sycl_used_aspects ![[HIGH_ACC]]
// CHECK-MIX: call double @llvm.fpbuiltin.exp.f64(double {{.*}}) #[[ATTR:[0-9]+]], !sycl_used_aspects ![[CUDA_ACC:[0-9]+]]
// CHECK-MIX: call double @llvm.fpbuiltin.log.f64(double {{.*}}) #[[ATTR:[0-9]+]], !sycl_used_aspects ![[SYCL_ACC:[0-9]+]]
      (void)log(exp(Value));
    });
  });
  deviceQueue.wait();

  // Kernel5 uses cuda-accuracy acos.
  deviceQueue.submit([&](handler& cgh) {
    cgh.parallel_for<class Kernel5>(numOfItems,
    [=](id<1> wiID) {
// CHECK-FUNC: call double @llvm.fpbuiltin.acos.f64(double {{.*}}) #[[ATTR:[0-9]+]], !sycl_used_aspects ![[CUDA_ACC]]
// CHECK-TU: call double @llvm.fpbuiltin.acos.f64(double {{.*}}) #[[ATTR:[0-9]+]], !sycl_used_aspects ![[HIGH_ACC]]
// CHECK-MIX: call double @llvm.fpbuiltin.acos.f64(double {{.*}}) #[[ATTR:[0-9]+]], !sycl_used_aspects ![[MEDIUM_ACC]]
      (void)acos(Value);
    });
  });

  // Kernel6 uses sycl-accuracy asin.
  deviceQueue.submit([&](handler& cgh) {
    cgh.parallel_for<class Kernel6>(numOfItems,
    [=](id<1> wiID) {
// CHECK-FUNC: call double @llvm.fpbuiltin.asin.f64(double {{.*}}) #[[ATTR:[0-9]+]], !sycl_used_aspects ![[SYCL_ACC]]
// CHECK-TU: call double @llvm.fpbuiltin.asin.f64(double {{.*}}) #[[ATTR:[0-9]+]], !sycl_used_aspects ![[HIGH_ACC]]
// CHECK-MIX: call double @llvm.fpbuiltin.asin.f64(double {{.*}}) #[[ATTR:[0-9]+]], !sycl_used_aspects ![[MEDIUM_ACC]]
      (void)asin(Value);
    });
  });

  // Kernel7 uses high-accuracy sqrt.
  deviceQueue.submit([&](handler& cgh) {
    cgh.parallel_for<class Kernel7>(numOfItems,
    [=](id<1> wiID) {
// CHECK-FUNC: call double @llvm.fpbuiltin.sqrt.f64(double {{.*}}) #[[ATTR:[0-9]+]], !sycl_used_aspects ![[HIGH_ACC]]
// CHECK-TU: call double @llvm.fpbuiltin.sqrt.f64(double {{.*}}) #[[ATTR:[0-9]+]], !sycl_used_aspects ![[HIGH_ACC]]
// CHECK-MIX: call double @llvm.fpbuiltin.sqrt.f64(double {{.*}}) #[[ATTR:[0-9]+]], !sycl_used_aspects ![[HIGH_ACC]]
      (void)sqrt(Value);
    });
  });
  return 0;
}

// CHECK-FUNC: [[HIGH_ACC]] = !{i32 -1}
// CHECK-FUNC: [[MEDIUM_ACC]] = !{i32 -2}
// CHECK-FUNC: [[LOW_ACC]] = !{i32 -3}
// CHECK-FUNC: [[CUDA_ACC]] = !{i32 -5}
// CHECK-FUNC: [[SYCL_ACC]] = !{i32 -4}

// CHECK-TU: [[HIGH_ACC]] = !{i32 -1}

// CHECK-MIX: [[HIGH_ACC]] = !{i32 -1}
// CHECK-MIX: [[MEDIUM_ACC]] = !{i32 -2}
// CHECK-MIX: [[CUDA_ACC]] = !{i32 -5}
// CHECK-MIX: [[SYCL_ACC]] = !{i32 -4}
