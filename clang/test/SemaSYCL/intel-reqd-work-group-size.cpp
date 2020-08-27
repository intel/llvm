// RUN: %clang_cc1 -fsycl -fsycl-is-device -Wno-sycl-2017-compat -fsyntax-only -verify -DTRIGGER_ERROR %s
// RUN: %clang_cc1 -fsycl -fsycl-is-device -Wno-sycl-2017-compat -ast-dump %s | FileCheck %s
// RUN: %clang_cc1 -fsycl -fsycl-is-host -Wno-sycl-2017-compat -fsyntax-only -verify %s

#ifndef __SYCL_DEVICE_ONLY__
// expected-no-diagnostics
class Functor {
public:
  [[intel::reqd_work_group_size(4)]] void operator()() const {}
};

template <typename name, typename Func>
void kernel(const Func &kernelFunc) {
  kernelFunc();
}

void bar() {
  Functor f;
  kernel<class kernel_name>(f);
}
#else
[[intel::reqd_work_group_size(4)]] void f4x1x1() {} // expected-note {{conflicting attribute is here}}
// expected-note@-1 {{conflicting attribute is here}}
[[intel::reqd_work_group_size(32)]] void f32x1x1() {} // expected-note {{conflicting attribute is here}}

[[intel::reqd_work_group_size(16)]] void f16x1x1() {}      // expected-note {{conflicting attribute is here}}
[[intel::reqd_work_group_size(16, 16)]] void f16x16x1() {} // expected-note {{conflicting attribute is here}}

[[intel::reqd_work_group_size(32, 32)]] void f32x32x1() {}      // expected-note {{conflicting attribute is here}}
[[intel::reqd_work_group_size(32, 32, 32)]] void f32x32x32() {} // expected-note {{conflicting attribute is here}}

#ifdef TRIGGER_ERROR
class Functor32 {
public:
  [[cl::reqd_work_group_size(32)]] void operator()() const {} // expected-error {{'reqd_work_group_size' attribute requires exactly 3 arguments}}
};
class Functor33 {
public:
  [[intel::reqd_work_group_size(32, -4)]] void operator()() const {} // expected-error {{'reqd_work_group_size' attribute requires a non-negative integral compile time constant expression}}
};
#endif // TRIGGER_ERROR

class Functor16 {
public:
  [[intel::reqd_work_group_size(16)]] void operator()() const {}
};

class Functor64 {
public:
  [[intel::reqd_work_group_size(64, 64)]] void operator()() const {}
};

class Functor16x16x16 {
public:
  [[intel::reqd_work_group_size(16, 16, 16)]] void operator()() const {}
};

class Functor8 { // expected-error {{conflicting attributes applied to a SYCL kernel}}
public:
  [[intel::reqd_work_group_size(8)]] void operator()() const { // expected-note {{conflicting attribute is here}}
    f4x1x1();
  }
};

class Functor {
public:
  void operator()() const {
    f4x1x1();
  }
};

class FunctorAttr {
public:
  __attribute__((reqd_work_group_size(128, 128, 128))) void operator()() const {}
};

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(const Func &kernelFunc) {
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

  kernel<class kernel_name5>([]() [[intel::reqd_work_group_size(32, 32, 32)]] {
    f32x32x32();
  });

#ifdef TRIGGER_ERROR
  Functor8 f8;
  kernel<class kernel_name6>(f8);

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
  kernel<class kernel_name10>([[intel::reqd_work_group_size(32, 32, 32)]][]() {
    f32x32x32();
  });

#endif // TRIGGER_ERROR
}

// CHECK: FunctionDecl {{.*}} {{.*}}kernel_name1
// CHECK: ReqdWorkGroupSizeAttr {{.*}}  1 1 16
// CHECK: FunctionDecl {{.*}} {{.*}}kernel_name2
// CHECK: ReqdWorkGroupSizeAttr {{.*}} 1 1 4
// CHECK: FunctionDecl {{.*}} {{.*}}kernel_name3
// CHECK: ReqdWorkGroupSizeAttr {{.*}} 16 16 16
// CHECK: FunctionDecl {{.*}} {{.*}}kernel_name4
// CHECK: ReqdWorkGroupSizeAttr {{.*}} 128 128 128
// CHECK: FunctionDecl {{.*}} {{.*}}kernel_name5
// CHECK: ReqdWorkGroupSizeAttr {{.*}} 32 32 32
#endif // __SYCL_DEVICE_ONLY__
