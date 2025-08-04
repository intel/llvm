// RUN: %clang_cc1 -triple native_cpu -aux-triple x86_64-unknown-linux-gnu -fsycl-is-device -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,X86,NOAVX
// RUN: %clang_cc1 -triple native_cpu -aux-triple x86_64-unknown-linux-gnu -aux-target-cpu skylake -fsycl-is-device -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,X86,AVX
// RUN: %clang_cc1 -triple native_cpu -aux-triple x86_64-unknown-linux-gnu -aux-target-feature +avx -fsycl-is-device -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,X86,AVX
//
// This is not sensible but check that we do not crash.
// RUN: %clang_cc1 -triple native_cpu -aux-triple native_cpu -fsycl-is-device -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,NOX86,NOAVX

#include "Inputs/sycl.hpp"
using namespace sycl;

class Test;
int main() {
  sycl::queue deviceQueue;
  deviceQueue.submit([&](handler &h) { h.single_task<Test>([=] {}); });
}

// X86: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// NOX86: target datalayout = "e"

// CHECK: void @_ZTS4Test() [[ATTRS:#[0-9]+]]
// CHECK: [[ATTRS]] = {
// NOAVX-NOT: "target-features"="{{[^"]*}}+avx{{[^"]*}}"
// AVX-SAME: "target-features"="{{[^"]*}}+avx{{[^"]*}}"
