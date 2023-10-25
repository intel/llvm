// RUN: cgeist %s --function=* -S | FileCheck %s

// XFAIL: *

// COM: Failing test tracked by https://github.com/intel/llvm/issues/11656

void f0() { int *x = {}; }
void f1() { int *x{}; }
