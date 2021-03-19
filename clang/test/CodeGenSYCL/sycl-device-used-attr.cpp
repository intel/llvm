// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes %s -emit-llvm -o -  | FileCheck %s
// Test that the 'used' attribute does not force the emission of globals on sycl device

// CHECK-NOT: @_ZN1hI1aE1iE
// CHECK-NOT: @_ZGVN1hI1aE1iE
// CHECK-NOT: @__dso_handle
// CHECK-NOT: @llvm.global_ctors
// CHECK-NOT: @llvm.used

// CHECK-NOT: {{.*}} void @_ZN1fI1aEC2Ev(%struct._ZTS1fI1aE.f* %this)

// CHECK-NOT: {{.*}} void @__cxx_global_var_init()
// CHECK-NOT: {{.*}} void @_ZN1gD1Ev(%class._ZTS1g.g*) unnamed_addr #2
// CHECK-NOT: {{.*}} i32 @__cxa_atexit(void (i8*)*, i8*, i8*)
// CHECK-NOT: {{.*}} void @_ZN1fI1aEC1Ev(%struct._ZTS1fI1aE.f* %this)

struct a;
class g {
public:
  int c;
  ~g();
};
template <class>
class h {
public:
  static const void k();
  static g i;
};
template <class j>
const void h<j>::k() { i.c = 0; }
template <class j>
g h<j>::i;
template <class>
struct f { f() __attribute__((used)); };
template <class j>
f<j>::f() { h<j>::k(); }
template struct f<a>;
