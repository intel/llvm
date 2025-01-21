// Target "host-x86_64-unknown-linux-gnu" only works on Linux
// REQUIRES: system-linux

// Test archive unbundling with multiple files with the same target
// One of the files is a bundled BC file and the other file is a bundled object file

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Create BC file to be bundled
// RUN: %clangxx -emit-llvm -c %s -o %t.bc
// Create object file to be bundled
// RUN: %clangxx            -c %s -o %t.o

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Bundle BC and object file to same target
// RUN: clang-offload-bundler -type=bc -targets=host-x86_64-unknown-linux-gnu -input=%t.bc -output=%t_bundled.bc
// RUN: clang-offload-bundler -type=o  -targets=host-x86_64-unknown-linux-gnu -input=%t.o  -output=%t_bundled.o

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Create unbundled BC and object file to use as reference
// RUN: clang-offload-bundler -type=bc -targets=host-x86_64-unknown-linux-gnu -input=%t_bundled.bc -unbundle -output=%t_unbundled.bc
// RUN: clang-offload-bundler -type=o  -targets=host-x86_64-unknown-linux-gnu -input=%t_bundled.o  -unbundle -output=%t_unbundled.o

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Make archive with bundled BC and o files
// RUN: rm -f %t_bundled.a
// RUN: llvm-ar cr %t_bundled.a %t_bundled.bc %t_bundled.o

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Test archive unbundling with multiple files with the same target
// RUN: clang-offload-bundler -unbundle --targets=host-x86_64-unknown-linux-gnu -type=aoo -input=%t_bundled.a -output=%t_list.txt

// RUN: wc %t_list.txt -l | FileCheck --check-prefixes=CHECK-LIST-LENGTH %s

// CHECK-LIST-LENGTH: 2

// RUN: cmp %t_unbundled.bc `grep .bc$ %t_list.txt`
// RUN: cmp %t_unbundled.o  `grep .o$  %t_list.txt`

void foo() {
}
