// RUN: %clang_cc1 -fsycl-is-device -verify -fsyntax-only -x c++ %s
// RUN: %clang_cc1 -fsycl-is-device -verify -fsyntax-only -DPRINTF_INVALID_DEF -x c++ %s
// RUN: %clang_cc1 -fsycl-is-device -verify -fsyntax-only -DPRINTF_INVALID_DECL -x c++ %s
// RUN: %clang_cc1 -fsycl-is-device -verify -fsyntax-only -DPRINTF_VALID1 -x c++ %s
// RUN: %clang_cc1 -fsycl-is-device -verify -fsyntax-only -DPRINTF_VALID2 -x c++ %s

#if defined(PRINTF_INVALID_DECL)
extern "C" int __spirv_ocl_printf(const char *__format, ...);
namespace A {
  int __spirv_ocl_printf(const char *__format, ...);
}
#elif defined(PRINTF_INVALID_DEF)
int __spirv_ocl_printf(const char *__format, ...) {
  return 42;
}
#elif defined(PRINTF_VALID1)
class A {
  friend int __spirv_ocl_printf(const char *__format, ...);
};
int __spirv_ocl_printf(const char *__format, ...);
#elif defined(PRINTF_VALID2)
extern "C" {
  extern "C++" {
    int __spirv_ocl_printf(const char *__format, ...);
  }
}
#else
int __spirv_ocl_printf(const char *__format, ...);
#endif

int __cdecl foo(int, ...); // expected-no-error

float bar(float f, ...) { return ++f; } // expected-no-error

void bar() {
  foo(5); // expected-no-error
  bar(7.0f); // expected-no-error
}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  kernelFunc();
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
