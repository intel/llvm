// DEFINE: %{common_opts} = -internal-isystem %S/Inputs -fsycl-is-device \
// DEFINE: -emit-llvm -triple spir64-unknown-unknown

// RUN: %clang_cc1 %{common_opts} %s -o - \
// RUN: | FileCheck --check-prefix PREC-SQRT %s

// RUN: %clang_cc1 %{common_opts} -ftarget-prec-sqrt %s -o - \
// RUN: | FileCheck --check-prefix PREC-SQRT %s

// RUN: %clang_cc1 %{common_opts} -fno-target-prec-sqrt %s -o - \
// RUN: | FileCheck --check-prefix ROUNDED-SQRT %s

// RUN: %clang_cc1 %{common_opts} -ftarget-prec-div %s -o - \
// RUN: | FileCheck --check-prefix PREC-DIV %s

// RUN: %clang_cc1 %{common_opts} -fno-target-prec-div %s -o - \
// RUN: | FileCheck --check-prefix ROUNDED-DIV %s


#include "sycl.hpp"

extern "C" SYCL_EXTERNAL float sqrt(float);
extern "C" SYCL_EXTERNAL float fdiv(float, float);

using namespace sycl;

int main() {
  const unsigned array_size = 4;
  range<1> numOfItems{array_size};
  float Value1 = .5f;
  float Value2 = .9f;
  queue deviceQueue;

  deviceQueue.submit([&](handler& cgh) {
    cgh.parallel_for<class KernelSqrt>(numOfItems,
    [=](id<1> wiID) {
      // PREC-SQRT: call spir_func float @sqrt(float noundef {{.*}})
      // ROUNDED-SQRT: call float @llvm.fpbuiltin.sqrt.f32(float {{.*}}) #[[ATTR_SQRT:[0-9]+]]
      // ROUNDED-DIV:  call spir_func float @sqrt(float noundef {{.*}})
      (void)sqrt(Value1);
    });
  });

  deviceQueue.submit([&](handler& cgh) {
    cgh.parallel_for<class KernelFdiv>(numOfItems,
    [=](id<1> wiID) {
      // PREC-SQRT: call spir_func float @fdiv(float noundef {{.*}}, float noundef {{.*}})
      // ROUNDED-SQRT: call spir_func float @fdiv(float noundef {{.*}}, float noundef {{.*}})
      // PREC-DIV: call spir_func float @fdiv(float noundef {{.*}}, float noundef {{.*}})
      // ROUNDED-DIV: call float @llvm.fpbuiltin.fdiv.f32(float {{.*}}, float {{.*}}) #[[ATTR_DIV:[0-9]+]]

      (void)fdiv(Value1, Value1);
    });
  });

return 0;
}

// ROUNDED-SQRT: attributes #[[ATTR_SQRT]] = {{.*}}"fpbuiltin-max-error"="3.0"
// ROUNDED-DIV: attributes #[[ATTR_DIV]] = {{.*}}"fpbuiltin-max-error"="2.5"
