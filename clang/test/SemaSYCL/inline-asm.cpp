// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -verify %s -DLINUX_ASM
// RUN: %clang_cc1 -triple x86_64-windows -fsycl-is-device -fsyntax-only -fasm-blocks -verify %s

void foo() {
  int a;
#ifdef LINUX_ASM
  __asm__("int3");
#else
  __asm int 3
#endif // LINUX_ASM
}

void bar() {
  int a;
#ifdef LINUX_ASM
   __asm__("int3");  // expected-error {{SYCL kernel cannot use inline assembly}}
#else
   __asm int 3 // expected-error {{SYCL kernel cannot use inline assembly}}
#endif // LINUX_ASM
}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  // expected-note@+1 {{called by 'kernel_single_task<fake_kernel, (lambda}}
  kernelFunc();
}

int main() {
  foo();
  // expected-note@+1 {{called by 'operator()'}}
  kernel_single_task<class fake_kernel>([]() { bar(); });
  return 0;
}
