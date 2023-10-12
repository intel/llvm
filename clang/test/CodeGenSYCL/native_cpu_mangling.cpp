// This test ensures the native-cpu device generates the expected kernel names,
// and that the MS mangler doesn't assert on the code below.

// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc   -aux-triple x86_64-pc-windows-msvc   -I %S/Inputs -fsycl-is-device -fsycl-is-native-cpu -emit-llvm -o - -x c++ %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -aux-triple x86_64-unknown-linux-gnu -I %S/Inputs -fsycl-is-device -fsycl-is-native-cpu -emit-llvm -o - -x c++ %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -aux-triple x86_64-pc-windows-msvc   -I %S/Inputs -fsycl-is-device -fsycl-is-native-cpu -emit-llvm -o - -x c++ %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc   -aux-triple x86_64-unknown-linux-gnu -I %S/Inputs -fsycl-is-device -fsycl-is-native-cpu -emit-llvm -o - -x c++ %s | FileCheck %s
// Todo: check other cpus

#include "sycl.hpp"

struct name1;

void test(sycl::handler &h) {
  h.parallel_for_work_group<name1>(sycl::range<1>(2),sycl::range<1>(1), [=](sycl::group<1> G) {});
}    

// CHECK: void @_ZTS5name1(
