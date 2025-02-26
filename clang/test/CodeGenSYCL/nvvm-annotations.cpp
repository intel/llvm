// RUN: %clang_cc1 -fno-sycl-force-inline-kernel-lambda -fno-sycl-decompose-functor -fsycl-is-device -internal-isystem %S/Inputs -triple nvptx-nvidia-cuda -target-cpu sm_70 -disable-llvm-passes -sycl-std=2020 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,GRIDCONST
// RUN: %clang_cc1 -fno-sycl-force-inline-kernel-lambda -fno-sycl-decompose-functor -fsycl-is-device -internal-isystem %S/Inputs -triple nvptx64-nvidia-cuda -target-cpu sm_70 -disable-llvm-passes -sycl-std=2020 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,GRIDCONST

// RUN: %clang_cc1 -fno-sycl-force-inline-kernel-lambda -fno-sycl-decompose-functor -fsycl-is-device -internal-isystem %S/Inputs -triple nvptx-nvidia-cuda -target-cpu sm_60 -disable-llvm-passes -sycl-std=2020 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,NOGRIDCONST
// RUN: %clang_cc1 -fno-sycl-force-inline-kernel-lambda -fno-sycl-decompose-functor -fsycl-is-device -internal-isystem %S/Inputs -triple nvptx64-nvidia-cuda -target-cpu sm_60 -disable-llvm-passes -sycl-std=2020 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,NOGRIDCONST

// Tests that certain SYCL kernel parameters are annotated with "grid_constant" for supported microarchitectures.

#include "sycl.hpp"

using namespace sycl;

int main() {
  queue q;

  struct S {
    int a;
  } s;

  q.submit([&](handler &h) {
    // CHECK: define{{.*}} void @[[FUNC1:.*kernel_grid_const_params]](ptr noundef byval(%class.anon) align 4 %_arg__sycl_functor)
    h.single_task<class kernel_grid_const_params>([=]() { (void) s;});
  });

  return 0;
}

// Don't emit grid_constant annotations for older architectures.
// NOGRIDCONST-NOT: "grid_constant"

// This isn't stable in general, as it depends on the order of the captured
// parameters, but in this case there's only one parameter so we know it's 1.
// GRIDCONST-DAG: = !{ptr @[[FUNC1]], !"grid_constant", [[MD:\![0-9]+]]}
// GRIDCONST-DAG: [[MD]] = !{i32 1}
