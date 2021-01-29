// RUN: %clang_cc1 -fsycl -fsycl-is-device -internal-isystem %S/Inputs -Wno-return-type -Wno-sycl-2017-compat -fcxx-exceptions -fsyntax-only -ast-dump -verify -pedantic %s | FileCheck %s

#include "sycl.hpp"

using namespace cl::sycl;
queue q;

struct FuncObj {
  //expected-warning@+2 {{attribute 'intelfpga::no_global_work_offset' is deprecated}}
  //expected-note@+1 {{did you mean to use 'intel::no_global_work_offset' instead?}}
  [[intelfpga::no_global_work_offset]] void operator()() const {}
};

int main() {
  q.submit([&](handler &h) {
    // CHECK:       SYCLIntelNoGlobalWorkOffsetAttr {{.*}}
    // CHECK-NEXT:  IntegerLiteral{{.*}}1{{$}}
    h.single_task<class test_kernel1>(FuncObj());

    // CHECK: SYCLIntelNoGlobalWorkOffsetAttr {{.*}}
    // CHECK-NEXT:  IntegerLiteral{{.*}}0{{$}}
    h.single_task<class test_kernel2>(
        []() [[intel::no_global_work_offset(0)]]{});

    // CHECK: SYCLIntelNoGlobalWorkOffsetAttr{{.*}}
    // CHECK-NEXT:  IntegerLiteral{{.*}}42{{$}}
    h.single_task<class test_kernel3>(
        []() [[intel::no_global_work_offset(42)]]{});

    // CHECK: SYCLIntelNoGlobalWorkOffsetAttr{{.*}}
    // CHECK-NEXT: UnaryOperator{{.*}} 'int' prefix '-'
    // CHECK-NEXT-NEXT: IntegerLiteral{{.*}}1{{$}}
    h.single_task<class test_kernel4>(
        []() [[intel::no_global_work_offset(-1)]]{});

    // expected-error@+2{{'no_global_work_offset' attribute requires an integer constant}}
    h.single_task<class test_kernel5>(
        []() [[intel::no_global_work_offset("foo")]]{});

    h.single_task<class test_kernel6>([]() {
      // expected-error@+1{{'no_global_work_offset' attribute only applies to functions}}
      [[intel::no_global_work_offset(1)]] int a;
    });

    // expected-warning@+2{{attribute 'no_global_work_offset' is already applied}}
    h.single_task<class test_kernel7>(
        []() [[intel::no_global_work_offset(0), intel::no_global_work_offset(1)]]{});
  });
  return 0;
}
