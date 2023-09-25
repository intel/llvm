// RUN: not cgeist %s -O0 --function=* -S -emit-llvm 2>&1 | FileCheck %s

// CHECK: error: 'free' declared with conflicting signature w.r.t. stdlib function

struct foo;

extern "C" int free(foo *);

void f0(int *ptr) { delete ptr; }
void f1(foo *ptr) { free(ptr); }
