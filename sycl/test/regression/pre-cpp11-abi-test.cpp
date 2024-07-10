// RUN: not grep __cxx11 %sycl_source_dir/../test/abi/sycl_symbols_linux.dump | FileCheck %s --allow-empty
// CHECK-NOT: {{.}}
// REQUIRES: linux
