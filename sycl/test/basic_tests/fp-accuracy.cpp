// Checks that the test is passing.
// RUN: %clangxx -%fsycl-host-only -c -ffp-accuracy=high \
// RUN: -faltmathlib=SVMLAltMathLibrary -fno-math-errno %s
//
// Checks that the attribute 'builtin-max-error' is generated.
// RUN: %clangxx -c -fsycl -ffp-accuracy=high -faltmathlib=SVMLAltMathLibrary \
// RUN: -fno-math-errno -ffp-model=precise -S -emit-llvm -o - %s | FileCheck %s

#include <sycl/sycl.hpp>
using namespace sycl;

float res[] = {};

constexpr access::mode sycl_write = access::mode::write;

int main() {
  queue deviceQueue;
  double Value = 5.;

  float input;

  range<1> Length{1};

  buffer<float, 1> out(res, 1);

  deviceQueue.submit([&](handler &cgh) {
    cgh.single_task<class Kernel0>([=]() { double res = std::sin(Value); });
  });

  deviceQueue.submit([&](handler &cgh) {
    cgh.single_task<class Kernel1>([=]() { double res = sycl::sin(Value); });
  });

  deviceQueue.submit([&](handler &cgh) {
    auto output = out.template get_access<sycl_write>(cgh);

    cgh.single_task<class Kernel2>([=]() {
      for (int i = 0; i < 1; i++)
        output[i] = sycl::sin(input);
    });
  });
  // CHECK-LABEL: define {{.*}}spir_kernel void {{.*}}Kernel2
  // CHECK: tail call {{.*}} float @llvm.fpbuiltin.spirv.ocl.sin.f32(float {{.*}}) #[[ATTR_HIGH:[0-9]+]]
  // CHECK: attributes #[[ATTR_HIGH]] = {{.*}}"fpbuiltin-max-error"="1.0"

  return 0;
}
