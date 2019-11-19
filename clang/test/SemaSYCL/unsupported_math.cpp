// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -verify %s

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
  kernelFunc();
}

int main() {
  kernel<class kernel_function>([=]() {
    int acc[1] = {5};
    acc[0] *= 2;
    acc[0] += (int)__builtin_fabsf(-1.0f);       // expected-error{{builtin is not supported on this target}}
    acc[0] += (int)__builtin_cosf(-1.0f);        // expected-error{{builtin is not supported on this target}}
    acc[0] += (int)__builtin_powf(-1.0f, 10.0f); // expected-error{{builtin is not supported on this target}}
  });
  return 0;
}
