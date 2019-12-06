// RUN: %clang_cc1 -fsycl-is-device -verify -fsyntax-only -x c++ %s
// RUN: %clang_cc1 -fsycl-is-device -verify -fsyntax-only -DGOOD_PRINTF -x c++ %s

#ifdef GOOD_PRINTF
int __spirv_ocl_printf(const char *__format, ...);
#else
extern "C" int __spirv_ocl_printf(const char *__format, ...);
namespace A {
  int __spirv_ocl_printf(const char *__format, ...);
}
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

#ifdef GOOD_PRINTF
  kernel_single_task<class fake_kernel>([]() { __spirv_ocl_printf("Hello world! %d%d\n", 4, 2); });
#else
  //expected-error@+1 {{SYCL kernel cannot call a variadic function}}
  kernel_single_task<class fake_kernel>([]() { A::__spirv_ocl_printf("Hello world! %d%d\n", 4, 2); });
  //expected-error@+1 {{SYCL kernel cannot call a variadic function}}
  kernel_single_task<class fake_kernel>([]() { __spirv_ocl_printf("Hello world! %d%d\n", 4, 2); });
#endif

  bar();
  return 0;
}
