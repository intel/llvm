// RUN: %clang_cc1 -fsycl -fsycl-is-device -fsyntax-only -verify -DTRIGGER_ERROR %s
// RUN: %clang_cc1 -fsycl -fsycl-is-device -ast-dump %s | FileCheck %s

[[cl::reqd_work_group_size(4, 1, 1)]] void f4x1x1() {} // expected-note {{conflicting attribute is here}}
// expected-note@-1 {{conflicting attribute is here}}
[[cl::reqd_work_group_size(32, 1, 1)]] void f32x1x1() {} // expected-note {{conflicting attribute is here}}

[[cl::reqd_work_group_size(16, 1, 1)]] void f16x1x1() {}   // expected-note {{conflicting attribute is here}}
[[cl::reqd_work_group_size(16, 16, 1)]] void f16x16x1() {} // expected-note {{conflicting attribute is here}}

[[cl::reqd_work_group_size(32, 32, 1)]] void f32x32x1() {}   // expected-note {{conflicting attribute is here}}
[[cl::reqd_work_group_size(32, 32, 32)]] void f32x32x32() {} // expected-note {{conflicting attribute is here}}

class Functor16 {
public:
  [[cl::reqd_work_group_size(16, 1, 1)]] [[cl::reqd_work_group_size(16, 1, 1)]] void operator()() {}
};

#ifdef TRIGGER_ERROR
class Functor32 {
public:
  //expected-warning@+2{{attribute 'reqd_work_group_size' is already applied with different parameters}}
  // expected-error@+1{{'reqd_work_group_size' attribute conflicts with 'reqd_work_group_size' attribute}}
  [[cl::reqd_work_group_size(32, 1, 1)]] [[cl::reqd_work_group_size(1, 1, 32)]] void operator()() {}
};
#endif
class Functor16x16x16 {
public:
  [[cl::reqd_work_group_size(16, 16, 16)]] void operator()() {}
};

class Functor8 { // expected-error {{conflicting attributes applied to a SYCL kernel}}
public:
  [[cl::reqd_work_group_size(1, 1, 8)]] void operator()() { // expected-note {{conflicting attribute is here}}
    f4x1x1();
  }
};

class Functor {
public:
  void operator()() {
    f4x1x1();
  }
};

class FunctorAttr {
public:
  __attribute__((reqd_work_group_size(128, 128, 128))) void operator()() {}
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

  Functor16x16x16 f16x16x16;
  kernel<class kernel_name3>(f16x16x16);

  FunctorAttr fattr;
  kernel<class kernel_name4>(fattr);

  kernel<class kernel_name5>([]() [[cl::reqd_work_group_size(32, 32, 32), cl::reqd_work_group_size(32, 32, 32)]] {
    f32x32x32();
  });

#ifdef TRIGGER_ERROR
  Functor8 f8;
  kernel<class kernel_name6>(f8);

  Functor32 f32;
  kernel<class kernel_name1>(f32);

  kernel<class kernel_name7>([]() { // expected-error {{conflicting attributes applied to a SYCL kernel}}
    f4x1x1();
    f32x1x1();
  });

  kernel<class kernel_name8>([]() { // expected-error {{conflicting attributes applied to a SYCL kernel}}
    f16x1x1();
    f16x16x1();
  });

  kernel<class kernel_name9>([]() { // expected-error {{conflicting attributes applied to a SYCL kernel}}
    f32x32x32();
    f32x32x1();
  });

  // expected-error@+1 {{expected variable name or 'this' in lambda capture list}}
  kernel<class kernel_name10>([[cl::reqd_work_group_size(32, 32, 32)]][]() {
    f32x32x32();
  });

#endif
}

// CHECK: FunctionDecl {{.*}} {{.*}}kernel_name1
// CHECK: ReqdWorkGroupSizeAttr {{.*}} 1 1 16
// CHECK: FunctionDecl {{.*}} {{.*}}kernel_name2
// CHECK: ReqdWorkGroupSizeAttr {{.*}} 1 1 4
// CHECK: FunctionDecl {{.*}} {{.*}}kernel_name3
// CHECK: ReqdWorkGroupSizeAttr {{.*}} 16 16 16
// CHECK: FunctionDecl {{.*}} {{.*}}kernel_name4
// CHECK: ReqdWorkGroupSizeAttr {{.*}} 128 128 128
// CHECK: FunctionDecl {{.*}} {{.*}}kernel_name5
// CHECK: ReqdWorkGroupSizeAttr {{.*}} 32 32 32
