// XFAIL: *
// RUN: %clang_cc1 -triple spir64-unknown-unknown-sycldevice -fms-extensions \
// RUN: -aux-triple x86_64-unknown-linux-gnu -fsycl-is-device \
// RUN: -fsyntax-only -Wno-sycl-2017-compat -DWARNCHECK %s -o /dev/null 2>&1 | FileCheck %s
// check random triple aux-triple with sycl-device

// RUN: %clang_cc1 -triple spir64-unknown-windows-sycldevice -Wno-sycl-2017-compat -fsyntax-only \
// RUN: -fms-extensions -DWARNCHECK %s -o /dev/null 2>&1 | FileCheck %s
// check without -aux-triple but sycl-device

// RUN: %clang_cc1 -triple spir64-unknown-windows-sycldevice \
// RUN: -fsycl-is-device -aux-triple x86_64-pc-windows-msvc -fms-extensions \
// RUN: -fsyntax-only -Wno-sycl-2017-compat -DWARNCHECK %s -o /dev/null 2>&1 | \
// RUN: FileCheck %s --check-prefixes CHECKALL
// check -aux-tripe without sycl-device

// RUN: %clang_cc1 -triple spir64-unknown-windows-sycldevice -Wno-sycl-2017-compat -fsyntax-only \
// RUN: -aux-triple x86_64-pc-windows-msvc -fsycl-is-device \
// RUN: -fms-extensions -verify  %s
// check error message when dllimport function gets called in sycl-kernel code

#if defined(WARNCHECK)
// CHECK: warning: __declspec attribute 'dllexport' is not supported
int __declspec(dllexport) foo(int a) {
  return a;
}

// CHECK: warning: __declspec attribute 'dllimport' is not supported
int __declspec(dllimport) bar();


// CHECK: warning: unknown attribute 'dllimport' ignored
int [[dllimport]]xoo();

// CHECKALL: warning: unknown attribute 'dllimport' ignored
int zoo() __attribute__((dllimport));

#else

// emit error if dllimport function is called in sycl kernel
int  __declspec(dllexport) foo(int a) {
   return a;
}
// expected-note@+1 {{'bar' declared here}}
SYCL_EXTERNAL int __declspec(dllimport) bar();
// expected-note@+2 {{previous attribute is here}}
// expected-note@+1 {{previous declaration is here}}
int __declspec(dllimport) foobar();
int foobar()  // expected-warning {{'foobar' redeclared without 'dllimport' attribute: previous 'dllimport' ignored}}
{
  return 10;
}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(const Func &kernelFunc) {
  kernelFunc();
}

int main() {
  foo(10);  // expected-no-error
  bar();  // expected-no-error
  kernel_single_task<class fake_kernel>([]() {
    foo(10);// expected-no-error
    bar(); // expected-error {{SYCL kernel cannot call a dllimport function}}
    foobar(); // expected-no-error
  });
  bar();  // expected-no-error
  return 0;
}
#endif
