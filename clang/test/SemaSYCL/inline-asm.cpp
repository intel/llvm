// RUN: %clang_cc1 -fsycl -fsycl-is-device -fsyntax-only -verify %s -DLINUX_ASM
// RUN: %clang_cc1 -fsycl -fsycl-is-device -fsyntax-only -verify -triple x86_64-windows -fasm-blocks %s

// expected-no-diagnostics

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
  __asm__("int3");
#else
  __asm int 3
#endif // LINUX_ASM
}

template <typename Name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  kernelFunc();
#ifdef LINUX_ASM
  __asm__("int3");
#else
  __asm int 3
#endif // LINUX_ASM
}

int main() {
  foo();
  kernel_single_task<class fake_kernel>([]() { bar(); });
  return 0;
}
