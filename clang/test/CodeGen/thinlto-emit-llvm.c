// XFAIL: target={{.*}}-aix{{.*}}

// Test to ensure -emit-llvm and -emit-llvm-bc work when invoking the
// ThinLTO backend path.
// RUN: %clang -O2 %s -flto=thin -c -o %t.o
// RUN: llvm-lto -thinlto -o %t %t.o
// RUN: %clang_cc1 -O2 -x ir %t.o -fthinlto-index=%t.thinlto.bc -emit-llvm -o - | FileCheck %s
// Added -opaque-pointers.
// FIXME: Align with the community code when project is ready to enable opaque
// pointers by default
// RUN: %clang_cc1 -O2 -x ir %t.o -fthinlto-index=%t.thinlto.bc -emit-llvm-bc -o - | llvm-dis -opaque-pointers -o - | FileCheck %s

// CHECK: define{{.*}} void @foo()
void foo(void) {
}
