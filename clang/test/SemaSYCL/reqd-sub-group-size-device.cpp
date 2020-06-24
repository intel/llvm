// RUN: %clang_cc1 -fsycl -fsycl-is-device -fsyntax-only -verify -DTRIGGER_ERROR %s
// RUN: %clang_cc1 -fsycl -fsycl-is-device -ast-dump %s | FileCheck %s

[[cl::intel_reqd_sub_group_size(4)]] void foo() {} // expected-note {{conflicting attribute is here}}
// expected-note@-1 {{conflicting attribute is here}}
[[cl::intel_reqd_sub_group_size(32)]] void baz() {} // expected-note {{conflicting attribute is here}}

class Functor16 {
public:
  [[cl::intel_reqd_sub_group_size(16)]] void operator()() {}
};

class Functor8 { // expected-error {{conflicting attributes applied to a SYCL kernel}}
public:
  [[cl::intel_reqd_sub_group_size(8)]] void operator()() { // expected-note {{conflicting attribute is here}}
    foo();
  }
};

class Functor {
public:
  void operator()() {
    foo();
  }
};

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
  kernelFunc();
}

void bar() {
  Functor16 f16;
  kernel<class kernel_name1>(f16);

  Functor f;
  kernel<class kernel_name2>(f);

#ifdef TRIGGER_ERROR
  Functor8 f8;
  kernel<class kernel_name3>(f8);

  kernel<class kernel_name4>([]() { // expected-error {{conflicting attributes applied to a SYCL kernel}}
    foo();
    baz();
  });
#endif

  kernel<class kernel_name5>([]() [[cl::intel_reqd_sub_group_size(2)]] { });
  kernel<class kernel_name6>([]() [[cl::intel_reqd_sub_group_size(4)]] { foo(); });
}

[[cl::intel_reqd_sub_group_size(16)]] SYCL_EXTERNAL void B();
[[cl::intel_reqd_sub_group_size(16)]] void A() {
}
[[cl::intel_reqd_sub_group_size(16)]] SYCL_EXTERNAL void B() {
  A();
}

#ifdef TRIGGER_ERROR
// expected-note@+1 {{conflicting attribute is here}}
[[cl::intel_reqd_sub_group_size(2)]] void sg_size2() {}

// expected-note@+2 {{conflicting attribute is here}}
// expected-error@+1 {{conflicting attributes applied to a SYCL kernel}}
[[cl::intel_reqd_sub_group_size(4)]] __attribute__((sycl_device)) void sg_size4() {
  sg_size2();
}
#endif

// CHECK: FunctionDecl {{.*}} {{.*}}kernel_name1
// CHECK: IntelReqdSubGroupSizeAttr {{.*}}
// CHECK-NEXT: IntegerLiteral{{.*}}16{{$}}
// CHECK: FunctionDecl {{.*}} {{.*}}kernel_name2
// CHECK: IntelReqdSubGroupSizeAttr {{.*}}
// CHECK-NEXT: IntegerLiteral{{.*}}4{{$}}
// CHECK: FunctionDecl {{.*}} {{.*}}kernel_name5
// CHECK: IntelReqdSubGroupSizeAttr {{.*}}
// CHECK-NEXT: IntegerLiteral{{.*}}2{{$}}
