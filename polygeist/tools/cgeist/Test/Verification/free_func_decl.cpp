// RUN: cgeist %s -O0 --function=* -S -emit-llvm | FileCheck %s

struct foo;

// CHECK-LABEL:   declare void @free(ptr)
extern "C" void free(void *);

// CHECK-LABEL:   define void @_Z2f0Pi(
// CHECK-SAME:                         ptr %[[VAL_0:.*]]) {
// CHECK:           call void @free(ptr %[[VAL_0]])
// CHECK:           ret void
void f0(int *ptr) { delete ptr; }

// CHECK-LABEL:   define void @_Z2f1P3foo(
// CHECK-SAME:                            ptr %[[VAL_0:.*]]) {
// CHECK:           call void @free(ptr %[[VAL_0]])
// CHECK:           ret void
void f1(foo *ptr) { free(ptr); }
