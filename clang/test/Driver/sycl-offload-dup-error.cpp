// Tests to make sure that there is no duplicate error messaging for host
// REQUIRES: x86-registered-target
// RUN: not %clangxx -c -target x86_64-unknown-linux-gnu -fsycl %s \
// RUN: 2>&1 | FileCheck %s

void foo() {
  foobar s;
}

// CHECK: error: unknown type name 'foobar'
// CHECK-NOT: error: unknown type name 'foobar'

