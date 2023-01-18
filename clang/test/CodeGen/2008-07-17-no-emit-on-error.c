// RUN: rm -f %t1.bc
// Added -opaque-pointers.
// FIXME: Align with the community code when project is ready to enable opaque
// pointers by default
// RUN: %clang_cc1 -mllvm -opaque-pointers -DPASS %s -emit-llvm-bc -o %t1.bc
// Added -opaque-pointers.
// FIXME: Align with the community code when project is ready to enable opaque
// pointers by default
// RUN: opt -opaque-pointers %t1.bc -disable-output
// RUN: rm -f %t1.bc
// Added -opaque-pointers.
// FIXME: Align with the community code when project is ready to enable opaque
// pointers by default
// RUN: not %clang_cc1 -mllvm -opaque-pointers %s -emit-llvm-bc -o %t1.bc
// Added -opaque-pointers.
// FIXME: Align with the community code when project is ready to enable opaque
// pointers by default
// RUN: not opt -opaque-pointers %t1.bc -disable-output

void f(void) {
}

#ifndef PASS
void g(void) {
  *10;
}
#endif
