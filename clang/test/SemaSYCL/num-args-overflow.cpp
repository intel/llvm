// RUN: %clang_cc1 -fsycl -triple spir64_gen -DGPU -fsycl-is-device -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsycl -triple spir64 -fsycl-is-device -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsycl -triple spir64_gen -Wno-sycl-strict -fsycl-is-device -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsycl -triple spir64_gen -Werror=sycl-strict -DERROR -fsycl-is-device -fsyntax-only -verify %s

template <typename Name, typename F>
__attribute__((sycl_kernel)) void kernel(F kernelFunc) {
  kernelFunc();
}

void use() {
  int Arr[2001];
#ifdef GPU
  // expected-warning@+6 {{resulting number of kernel arguments 2001 is greater than maximum supported on GPU device - 2000}}
#elif ERROR
  // expected-error@+4 {{resulting number of kernel arguments 2001 is greater than maximum supported on GPU device - 2000}}
#else
  // expected-no-diagnostics
#endif
  kernel<class Foo>([=]() { (void)Arr[0]; });
}
