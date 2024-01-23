// DEFINE: %{common_opts} = -fsycl -fsycl-device-only -fno-math-errno \
// DEFINE: -ffp-accuracy=high -S -emit-llvm -o - %s

// RUN: %clangxx %{common_opts} | FileCheck %s

// RUN: %clangxx %{common_opts} -ffp-accuracy=low:exp \
// RUN: | FileCheck %s --check-prefix=CHECK-LOW-EXP

#include <sycl/sycl.hpp>

SYCL_EXTERNAL auto foo(double x) {
  using namespace sycl;
  return cos(exp(log(x)));
}

// CHECK-LABEL: define {{.*}}food
// CHECK: tail call double @llvm.fpbuiltin.log.f64(double {{.*}}) #[[ATTR_HIGH:[0-9]+]]
// CHECK: tail call double @llvm.fpbuiltin.exp.f64(double {{.*}}) #[[ATTR_HIGH]]
// CHECK: tail call double @llvm.fpbuiltin.cos.f64(double {{.*}}) #[[ATTR_HIGH]]

// CHECK: attributes #[[ATTR_HIGH]] = {{.*}}"fpbuiltin-max-error"="1.0"

// CHECK-LOW-EXP-LABEL: define {{.*}}food
// CHECK-LOW-EXP: tail call double @llvm.fpbuiltin.log.f64(double {{.*}}) #[[ATTR_F1_HIGH:[0-9]+]]
// CHECK-LOW-EXP: tail call double @llvm.fpbuiltin.exp.f64(double {{.*}}) #[[ATTR_F1_LOW:[0-9]+]]
// CHECK-LOW-EXP: tail call double @llvm.fpbuiltin.cos.f64(double {{.*}}) #[[ATTR_F1_HIGH]]

// CHECK-F1: attributes #[[ATTR_F1_HIGH]] = {{.*}}"fpbuiltin-max-error"="1.0"
// CHECK-F1: attributes #[[ATTR_F1_LOW]] = {{.*}}"fpbuiltin-max-error"="67108864.0"
