// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-windows-sycldevice -aux-triple x86_64-pc-windows-msvc -fsyntax-only -Wno-sycl-2017-compat -verify %s

// expected-no-warning@+1
__inline __cdecl int printf(char const* const _Format, ...) { return 0; }
// expected-no-warning@+1
__inline __cdecl __attribute__((sycl_device)) int foo() { return 0; }
// expected-no-warning@+1
__inline __cdecl int moo() { return 0; }

void bar() {
  printf("hello\n"); // expected-no-error
}

template <typename name, typename Func>
// expected-no-warning@+1
__cdecl __attribute__((sycl_kernel)) void kernel_single_task(const Func &kernelFunc) {
  // expected-error@+1{{SYCL kernel cannot call a variadic function}}
  printf("cannot call from here\n");
  // expected-no-error@+1
  moo();
  // expected-note@+1{{called by}}
  kernelFunc();
}

int main() {
  //expected-note@+2 {{in instantiation of}}
  //expected-error@+1 {{SYCL kernel cannot call a variadic function}}
  kernel_single_task<class fake_kernel>([]() { printf("world\n"); });
  bar();
  return 0;
}
