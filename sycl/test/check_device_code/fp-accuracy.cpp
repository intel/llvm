// RUN: %clangxx -fsycl %s -ffp-accuracy=high -fno-math-errno \
// RUN: -S -emit-llvm -o- | FileCheck %s

// RUN: %clangxx -fsycl %s -ffp-accuracy=high -ffp-accuracy=low:exp \
// RUN: -fno-math-errno -S -emit-llvm -o- | FileCheck --check-prefix=CHECK-F1 %s

#include <sycl/sycl.hpp>
using namespace sycl;

#define sz 2
double input[sz];
double output[sz];

int main() {
  queue deviceQueue;
  double Value = 5.;

  range<1> size{sz};

  buffer<double, 1> in(input, size);
  buffer<double, 1> out(output, size);

  deviceQueue.submit([&](handler &cgh) {
    accessor in_vals{in, cgh, read_only};
    accessor out_vals{out, cgh, write_only};
    cgh.single_task<class Kernel1>([=]() {
      for (int i = 0; i < sz; i++)
        out_vals[i] = cos(exp(log(in_vals[i])));
    });
  });
  return 0;
}

// CHECK-LABEL: define {{.*}}spir_kernel void @{{.*}}Kernel1
// CHECK: tail call double @llvm.fpbuiltin.log.f64(double %2) #[[ATTR_HIGH:[0-9]+]]
// CHECK: tail call double @llvm.fpbuiltin.exp.f64(double %3) #[[ATTR_HIGH]]
// CHECK: tail call double @llvm.fpbuiltin.cos.f64(double %4) #[[ATTR_HIGH]]

// CHECK: attributes #[[ATTR_HIGH]] = {{.*}}"fpbuiltin-max-error"="1.0"

// CHECK-F1-LABEL: define {{.*}}spir_kernel void @{{.*}}Kernel1
// CHECK-F1: tail call double @llvm.fpbuiltin.log.f64(double {{.*}}) #[[ATTR_F1_HIGH:[0-9]+]]
// CHECK-F1: tail call double @llvm.fpbuiltin.exp.f64(double {{.*}}) #[[ATTR_F1_LOW:[0-9]+]]
// CHECK-F1: tail call double @llvm.fpbuiltin.cos.f64(double {{.*}}) #[[ATTR_F1_HIGH]]

// CHECK-F1: attributes #[[ATTR_F1_HIGH]] = {{.*}}"fpbuiltin-max-error"="1.0"
// CHECK-F1: attributes #[[ATTR_F1_LOW]] = {{.*}}"fpbuiltin-max-error"="67108864.0"
