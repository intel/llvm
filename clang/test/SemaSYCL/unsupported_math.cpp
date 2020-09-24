// RUN: %clang_cc1 -fsycl -fsycl-is-device -fsyntax-only -Wno-sycl-2017-compat -verify %s
extern "C" float  sinf(float);
extern "C" float  cosf(float);
extern "C" float  logf(float);
extern "C" double sin(double);
extern "C" double cos(double);
extern "C" double log(double);
template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(const Func &kernelFunc) {
  kernelFunc();
}

int main() {
  kernel<class kernel_function>([=]() {
    int acc[1] = {5};
    acc[0] *= 2;
    acc[0] += (int)sinf(1.0f);                  // expected-no-error
    acc[0] += (int)sin(1.0);                    // expected-no-error
    acc[0] += (int)__builtin_sinf(1.0f);        // expected-no-error
    acc[0] += (int)__builtin_sin(1.0);          // expected-no-error
    acc[0] += (int)cosf(1.0f);                  // expected-no-error
    acc[0] += (int)cos(1.0);                    // expected-no-error
    acc[0] += (int)__builtin_cosf(1.0f);        // expected-no-error
    acc[0] += (int)__builtin_cos(1.0);          // expected-no-error
    acc[0] += (int)logf(1.0f);                  // expected-no-error
    acc[0] += (int)log(1.0);                    // expected-no-error
    acc[0] += (int)__builtin_logf(1.0f);        // expected-no-error
    acc[0] += (int)__builtin_log(1.0);          // expected-no-error
    acc[0] += (int)__builtin_fabsl(-1.0);       // expected-error{{builtin is not supported on this target}}
    acc[0] += (int)__builtin_cosl(-1.0);        // expected-error{{builtin is not supported on this target}}
    acc[0] += (int)__builtin_powl(-1.0, 10.0);  // expected-error{{builtin is not supported on this target}}
  });
  return 0;
}
