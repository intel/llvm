// RUN: %clangxx -O0 -fsycl-device-only -S -Xclang -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK-IR
// RUN: %clangxx -O0 -fsycl-device-only -Xclang -verify %s
// expected-no-diagnostics
// Tests for propagation of compile options

#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental;

int main() {
  queue Q;
  // CHECK-IR: spir_kernel void @{{.*}}Kernel0(){{.*}} #[[COAttr1:[0-9]+]]
  Q.single_task<class Kernel0>([]() {});
}
