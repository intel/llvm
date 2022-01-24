// RUN: %clangxx -fsycl -S -emit-llvm %s -o - | FileCheck %s

#include <sycl/sycl.hpp>

void kernel(int *data) {
  long long int a = -9223372034707292160ll;
  data[0] = sycl::ctz(a);
}

int main() { return 0; }

// CHECK: call i64 {{.*}}@_ZN2cl4sycl3ctz{{.*}}_(i64 %0)
