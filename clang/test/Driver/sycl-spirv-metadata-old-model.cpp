///
/// Tests for -fsycl-preserve-device-nonsemantic-metadata
///

// RUN: touch %tfoo.o
// RUN: %clangxx -fsycl --no-offload-new-driver -fsycl-preserve-device-nonsemantic-metadata -### %tfoo.o 2>&1 | \
// RUN:  FileCheck -check-prefix CHECK-WITH %s
// RUN: %clangxx -fsycl --no-offload-new-driver -### %tfoo.o 2>&1 | \
// RUN:  FileCheck -check-prefix CHECK-WITHOUT %s

// CHECK-WITH: llvm-spirv{{.*}} "--spirv-preserve-auxdata"
// CHECK-WITH-SAME: "-spirv-ext=-all,{{.*}},+SPV_KHR_cooperative_matrix"

// CHECK-WITHOUT: "{{.*}}llvm-spirv"
// CHECK-WITHOUT-NOT: --spirv-preserve-auxdata
