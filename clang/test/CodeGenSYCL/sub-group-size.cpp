// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -sycl-std=2020 -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefix=NONE
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -fsycl-default-sub-group-size=primary -sycl-std=2020 -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefix=PRIM_DEF
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -fsycl-default-sub-group-size=10 -sycl-std=2020 -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefix=TEN_DEF

#include "Inputs/sycl.hpp"
using namespace cl::sycl;

void default_behavior() {
  kernel_single_task<class Kernel1>([]() {
  });
}
