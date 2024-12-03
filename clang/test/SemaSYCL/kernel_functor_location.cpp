// RUN: %clang_cc1 -fsycl-is-device -ast-dump %s | FileCheck %s
//
// Checks that the compound statement of the implicitly generated kernel body
// has a valid source location (containing "line"). Previously this location
// was invalid containing "<<invalid sloc>>" which causes asserts in the
// llvm profiling tools.

#include "Inputs/sycl.hpp"

struct Functor {
  void operator()() const {}
};

// CHECK: FunctionDecl {{.*}} _ZTS7Functor 'void ()'
// CHECK-NEXT: |-CompoundStmt {{.*}} <{{.*}}line{{.*}}>

int main() {
  
  sycl::queue().submit([&](sycl::handler &cgh) {
    cgh.single_task(Functor{});
  });
}
