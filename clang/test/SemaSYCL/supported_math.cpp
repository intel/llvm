// RUN: %clang_cc1 -fsycl -fsycl-is-device -fsyntax-only -Wno-sycl-strict -verify %s
extern "C" float sinf(float);
extern "C" float cosf(float);
extern "C" float floorf(float);
extern "C" float logf(float);
extern "C" float nearbyintf(float);
extern "C" float rintf(float);
extern "C" float roundf(float);
extern "C" float truncf(float);
extern "C" float copysignf(float, float);
extern "C" double sin(double);
extern "C" double cos(double);
extern "C" double floor(double);
extern "C" double log(double);
extern "C" double nearbyint(double);
extern "C" double rint(double);
extern "C" double round(double);
extern "C" double trunc(double);
extern "C" double copysign(double, double);
template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(const Func &kernelFunc) {
  kernelFunc();
}

int main() {
  kernel<class kernel_function>([=]() {
    int acc[1] = {5};
    acc[0] *= 2;
    acc[0] += (int)truncf(1.0f);                     // expected-no-diagnostics
    acc[0] += (int)trunc(1.0);                       // expected-no-diagnostics
    acc[0] += (int)roundf(1.0f);                     // expected-no-diagnostics
    acc[0] += (int)round(1.0);                       // expected-no-diagnostics
    acc[0] += (int)rintf(1.0f);                      // expected-no-diagnostics
    acc[0] += (int)rint(1.0);                        // expected-no-diagnostics
    acc[0] += (int)nearbyintf(0.5f);                 // expected-no-diagnostics
    acc[0] += (int)nearbyint(0.5);                   // expected-no-diagnostics
    acc[0] += (int)floorf(0.5f);                     // expected-no-diagnostics
    acc[0] += (int)floor(0.5);                       // expected-no-diagnostics
    acc[0] += (int)copysignf(1.0f, -0.5f);           // expected-no-diagnostics
    acc[0] += (int)copysign(1.0, -0.5);              // expected-no-diagnostics
    acc[0] += (int)sinf(1.0f);                       // expected-no-diagnostics
    acc[0] += (int)sin(1.0);                         // expected-no-diagnostics
    acc[0] += (int)__builtin_sinf(1.0f);             // expected-no-diagnostics
    acc[0] += (int)__builtin_sin(1.0);               // expected-no-diagnostics
    acc[0] += (int)cosf(1.0f);                       // expected-no-diagnostics
    acc[0] += (int)cos(1.0);                         // expected-no-diagnostics
    acc[0] += (int)__builtin_cosf(1.0f);             // expected-no-diagnostics
    acc[0] += (int)__builtin_cos(1.0);               // expected-no-diagnostics
    acc[0] += (int)logf(1.0f);                       // expected-no-diagnostics
    acc[0] += (int)log(1.0);                         // expected-no-diagnostics
    acc[0] += (int)__builtin_truncf(1.0f);           // expected-no-diagnostics
    acc[0] += (int)__builtin_trunc(1.0);             // expected-no-diagnostics
    acc[0] += (int)__builtin_rintf(1.0f);            // expected-no-diagnostics
    acc[0] += (int)__builtin_rint(1.0);              // expected-no-diagnostics
    acc[0] += (int)__builtin_nearbyintf(0.5f);       // expected-no-diagnostics
    acc[0] += (int)__builtin_nearbyint(0.5);         // expected-no-diagnostics
    acc[0] += (int)__builtin_floorf(0.5f);           // expected-no-diagnostics
    acc[0] += (int)__builtin_floor(0.5);             // expected-no-diagnostics
    acc[0] += (int)__builtin_copysignf(1.0f, -0.5f); // expected-no-diagnostics
    acc[0] += (int)__builtin_logf(1.0f);             // expected-no-diagnostics
    acc[0] += (int)__builtin_log(1.0);               // expected-no-diagnostics
  });
  return 0;
}
