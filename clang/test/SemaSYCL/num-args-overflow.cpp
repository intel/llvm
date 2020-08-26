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
  // expected-warning@+8 {{kernel argument count (2001) exceeds supported maximum of 2000 on GPU}}
  // expected-note@+7 {{array elements and fields of a class/struct may be counted separately}}
#elif ERROR
  // expected-error@+5 {{kernel argument count (2001) exceeds supported maximum of 2000 on GPU}}
  // expected-note@+4 {{array elements and fields of a class/struct may be counted separately}}
#else
  // expected-no-diagnostics
#endif
  kernel<class Foo>([=]() { (void)Arr[0]; });
}
