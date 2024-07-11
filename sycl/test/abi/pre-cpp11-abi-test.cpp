// REQUIRES: linux
// At first exclude known symbols which needs to be fixed and then check that cxx11 is not matched.
// RUN: grep -v -f %S/cxx11_abi_exclude_list.txt %S/sycl_symbols_linux.dump | FileCheck %s
// CHECK-NOT: cxx11

