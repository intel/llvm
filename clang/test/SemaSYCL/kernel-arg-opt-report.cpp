// RUN: %clang_cc1 -triple spir64-unknown-unknown-sycldevice -fsycl-is-device \
// RUN: -Wno-sycl-2017-compat -emit-llvm-bc %s -o %t-host.bc -opt-record-file %t-host.yaml
// RUN: FileCheck -check-prefix=CHECK --input-file %t-host.yaml %s
// The test generates remarks about the kernel argument, their location and type
// in the resulting yaml file.

#include "Inputs/sycl.hpp"

class second_base {
public:
  int *e;
};

class InnerFieldBase {
public:
  int d;
};
class InnerField : public InnerFieldBase {
  int c;
};

struct base {
public:
  int b;
  InnerField obj;
};

//CHECK: --- !Passed
//CHECK: Pass:{{.*}}sycl
//CHECK: Name:{{.*}}Region
//CHECK: DebugLoc:{{.*}} { File: '{{.*}}kernel-arg-opt-report.cpp',
//CHECK: Line: 85, Column: 18 }
//CHECK: Function: _ZTS7derived
//CHECK: Args:
//CHECK-NEXT: String:   'Argument '
//CHECK-NEXT: Argument: '0'
//CHECK-NEXT: String:   ' for function kernel: '
//CHECK-NEXT: String:   '&'
//CHECK-NEXT: String:   ' '
//CHECK-NEXT: String:   _ZTS7derived
//CHECK-NEXT: String:   .
//CHECK-NEXT: String:   ' '
//CHECK-NEXT: String:   '('
//CHECK-NEXT: String:   struct base
//CHECK-NEXT: String:   ')'

//CHECK: --- !Passed
//CHECK: Pass:{{.*}}sycl
//CHECK: Name:{{.*}}Region
//CHECK: DebugLoc:{{.*}} { File: '{{.*}}kernel-arg-opt-report.cpp',
//CHECK: Line: 11, Column: 8 }
//CHECK: Function: _ZTS7derived
//CHECK: Args:
//CHECK-NEXT: String:  'Argument '
//CHECK-NEXT: Argument: '1'
//CHECK-NEXT: String:   ' for function kernel: '
//CHECK-NEXT: String:   ''
//CHECK-NEXT: String:   ' '
//CHECK-NEXT: String:   _ZTS7derived
//CHECK-NEXT: String:   .
//CHECK-NEXT: String:   e
//CHECK-NEXT: String:   '('
//CHECK-NEXT: String:   struct __wrapper_class
//CHECK-NEXT: String:   ')'

//CHECK: --- !Passed
//CHECK: Pass:{{.*}}sycl
//CHECK: Name:{{.*}}Region
//CHECK: DebugLoc:{{.*}} { File: '{{.*}}kernel-arg-opt-report.cpp',
//CHECK: Line: 86, Column: 7 }
//CHECK: Function:  _ZTS7derived
//CHECK: Args:
//CHECK-NEXT: String:   'Argument '
//CHECK-NEXT: Argument: '2'
//CHECK-NEXT: String:   ' for function kernel: '
//CHECK-NEXT: String:   ''
//CHECK-NEXT: String:   ' '
//CHECK-NEXT: String:   _ZTS7derived
//CHECK-NEXT: String:   .
//CHECK-NEXT: String:   a
//CHECK-NEXT: String:   '('
//CHECK-NEXT: String:   int
//CHECK-NEXT: String:   ')'

struct derived : base, second_base {
  int a;

  void operator()() const {
  }
};

int main() {
  sycl::queue q;

  q.submit([&](cl::sycl::handler &cgh) {
    derived f{};
    cgh.single_task(f);
  });

  return 0;
}
