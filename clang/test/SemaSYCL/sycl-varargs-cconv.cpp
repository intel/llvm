// The following runs use -Wno-sycl-strict to bypass SYCL_EXTERNAL applied to
// funtion with raw pointer parameter
// RUN: %clang_cc1 -fsycl-is-device -verify -Wno-sycl-strict -fsyntax-only %s
// RUN: %clang_cc1 -fsycl-is-device -verify -Wno-sycl-strict -fsyntax-only -DPRINTF_INVALID_DEF %s
// RUN: %clang_cc1 -fsycl-is-device -verify -fsyntax-only -Wno-sycl-strict -DPRINTF_INVALID_DECL %s
// RUN: %clang_cc1 -fsycl-is-device -verify -fsyntax-only -Wno-sycl-strict -DPRINTF_VALID1 %s
// RUN: %clang_cc1 -fsycl-is-device -verify -fsyntax-only -Wno-sycl-strict -DPRINTF_VALID2 %s

#if defined(PRINTF_INVALID_DECL)
extern "C" SYCL_EXTERNAL int __spirv_ocl_printf(const char *__format, ...);
namespace A {
SYCL_EXTERNAL int __spirv_ocl_printf(const char *__format, ...);
}
#elif defined(PRINTF_INVALID_DEF)
int __spirv_ocl_printf(const char *__format, ...) {
  return 42;
}
#elif defined(PRINTF_VALID1)
class A {
  friend int __spirv_ocl_printf(const char *__format, ...);
};
SYCL_EXTERNAL
int __spirv_ocl_printf(const char *__format, ...);
#elif defined(PRINTF_VALID2)
extern "C" {
extern "C++" {
SYCL_EXTERNAL
int __spirv_ocl_printf(const char *__format, ...);
}
}
#else
SYCL_EXTERNAL
int __spirv_ocl_printf(const char *, ...);
#endif

SYCL_EXTERNAL int __cdecl foo(int, ...); // expected-no-error

float bar(float f, ...) { return ++f; } // expected-no-error

void bar() {
  foo(5);    // expected-no-error
  bar(7.0f); // expected-no-error
}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(const Func &kernelFunc) {
  kernelFunc(); //expected-note 2+ {{called by 'kernel_single_task}}
}

int main() {
  //expected-error@+1 {{SYCL kernel cannot call a variadic function}}
  kernel_single_task<class fake_kernel>([]() { foo(6); });
  //expected-error@+1 {{SYCL kernel cannot call a variadic function}}
  kernel_single_task<class fake_kernel>([]() { bar(9.0); });

#if defined(PRINTF_INVALID_DECL)
  //expected-error@+1 {{SYCL kernel cannot call a variadic function}}
  kernel_single_task<class fake_kernel>([]() { A::__spirv_ocl_printf("Hello world! %d%d\n", 4, 2); });
  //expected-error@+1 {{SYCL kernel cannot call a variadic function}}
  kernel_single_task<class fake_kernel>([]() { __spirv_ocl_printf("Hello world! %d%d\n", 4, 2); });
#elif defined(PRINTF_INVALID_DEF)
  //expected-error@+1 {{SYCL kernel cannot call a variadic function}}
  kernel_single_task<class fake_kernel>([]() { __spirv_ocl_printf("Hello world! %d%d\n", 4, 2); });
#else
  kernel_single_task<class fake_kernel>([]() { __spirv_ocl_printf("Hello world! %d%d\n", 4, 2); });
#endif

  bar();
  return 0;
}
