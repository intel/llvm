// TODO: Remove the -fsycl-llc-options option once https://github.com/intel/llvm/issues/14139 is fixed.
// RUN: %clang -fsycl -fsycl-llc-options="-foo -bar" -### 2>&1 %s | FileCheck %s
// CHECK: clang-offload-wrapper
// CHECK: llc{{.*}}-foo{{.*}}-bar
