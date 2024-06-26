// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -verify %s -DLINUX_ASM
// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -verify %s -DLINUX_ASM -DSPIR_CHECK -triple spir64-unknown-unknown
// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -verify -triple x86_64-windows -fasm-blocks %s

#ifndef SPIR_CHECK
//expected-no-diagnostics
#endif // SPIR_CHECK

static __inline unsigned int
asm_func(unsigned int __leaf, unsigned long __d[]) {
  unsigned int __result;
#ifdef SPIR_CHECK
  __asm__("enclu"
          : "=a"(__result), "=b"(__d[0]), "=c"(__d[1]), "=d"(__d[2])
          : "a"(__leaf), "b"(__d[0]), "c"(__d[1]), "d"(__d[2])
          : "cc");
#endif // SPIR_CHECK
  return __result;
}

static __inline unsigned int
asm_func_2(unsigned int __leaf, unsigned long __d[]) {
  unsigned int __result;
#ifdef SPIR_CHECK
  // expected-error@+2 {{invalid output constraint '=a' in asm}}
  __asm__("enclu"
          : "=a"(__result), "=b"(__d[0]), "=c"(__d[1]), "=d"(__d[2])
          : "a"(__leaf), "b"(__d[0]), "c"(__d[1]), "d"(__d[2])
          : "cc");
#endif // SPIR_CHECK
  return __result;
}

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
__attribute__((sycl_kernel)) void kernel_single_task(const Func &kernelFunc) {
  kernelFunc();
#ifdef LINUX_ASM
  __asm__("int3");

#ifdef SPIR_CHECK
  unsigned int i = 3;
  unsigned long d[4];
  // expected-note@+1 {{called by 'kernel_single_task}}
  asm_func_2(i, d);
#endif // SPIR_CHECK
#else
  __asm int 3
#endif // LINUX_ASM
}

int main() {
  foo();
  kernel_single_task<class fake_kernel>([]() { bar(); });
  return 0;
}
