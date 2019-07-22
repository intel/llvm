// RUN: %clang_cc1 -triple spir64-unknown-windows-sycldevice -aux-triple x86_64-pc-windows-msvc -fsycl-is-device -fsyntax-only -verify %s

//expected-warning@+1 {{'__cdecl' calling convention is not supported for this target}}
__inline __cdecl int printf(char const* const _Format, ...) { return 0; }

void bar() {
  printf("hello\n"); // expected-no-error
}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  kernelFunc();
}

int main() {
  //expected-error@+1 {{SYCL kernel cannot call a variadic function}}
  kernel_single_task<class fake_kernel>([]() { printf("world\n"); });
  bar();
  return 0;
}
