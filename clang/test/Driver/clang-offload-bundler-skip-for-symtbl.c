// RUN: llvm-as %s.ll -o %t-in.bc
// RUN: %clang -c -o %t-in.o %s
// RUN: clang-offload-bundler -type=o -targets=openmp-spir64_gen,host-x86_64-unknown-linux-gnu -input=%t-in.bc -input=%t-in.o -output=%t-out.o
// RUN: llvm-readobj --string-dump=.tgtsym %t-out.o | FileCheck %s.ll

int main() {return 0;}
