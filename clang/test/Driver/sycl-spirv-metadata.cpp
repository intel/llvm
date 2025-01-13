///
/// Tests for -fsycl-preserve-device-nonsemantic-metadata
///

// RUN: touch %tfoo.o
// RUN: %clangxx -fsycl --offload-new-driver -fsycl-preserve-device-nonsemantic-metadata -### %tfoo.o 2>&1 | \
// RUN:  FileCheck -check-prefix CHECK-WITH %s
// RUN: %clangxx -fsycl --offload-new-driver -### %tfoo.o 2>&1 | \
// RUN:  FileCheck -check-prefix CHECK-WITHOUT %s

// CHECK-WITH: clang-linker-wrapper{{.*}} "--llvm-spirv-options={{.*}} --spirv-preserve-auxdata
// CHECK-WITHOUT-NOT: --spirv-preserve-auxdata
