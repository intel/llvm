// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -Wno-sycl-2017-compat -verify %s
template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(const Func &kernelFunc) {
  kernelFunc();
}

int main() {
  kernel<class kernel_function>([=]() {
    int acc[1] = {5};
    acc[0] *= 2;
    acc[0] += (int)__builtin_fabsl(-1.0);      // expected-error{{builtin is not supported on this target}}
    acc[0] += (int)__builtin_cosl(-1.0);       // expected-error{{builtin is not supported on this target}}
    acc[0] += (int)__builtin_powl(-1.0, 10.0); // expected-error{{builtin is not supported on this target}}
  });
  return 0;
}
