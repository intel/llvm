// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -Wno-sycl-strict -verify %s
// expected-no-diagnostics
extern "C" float sinf(float);
extern "C" float cosf(float);
extern "C" float floorf(float);
extern "C" float logf(float);
extern "C" float nearbyintf(float);
extern "C" float rintf(float);
extern "C" float roundf(float);
extern "C" float truncf(float);
extern "C" float copysignf(float, float);
extern "C" float fminf(float, float);
extern "C" float fmaxf(float, float);
extern "C" double sin(double);
extern "C" double cos(double);
extern "C" double floor(double);
extern "C" double log(double);
extern "C" double nearbyint(double);
extern "C" double rint(double);
extern "C" double round(double);
extern "C" double trunc(double);
extern "C" double copysign(double, double);
extern "C" double fmin(double, double);
extern "C" double fmax(double, double);
template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(const Func &kernelFunc) {
  kernelFunc();
}

int main() {
  kernel<class kernel_function>([=]() {
    int acc[1] = {5};
    acc[0] *= 2;
    acc[0] += (int)truncf(1.0f);
    acc[0] += (int)trunc(1.0);
    acc[0] += (int)roundf(1.0f);
    acc[0] += (int)round(1.0);
    acc[0] += (int)rintf(1.0f);
    acc[0] += (int)rint(1.0);
    acc[0] += (int)nearbyintf(0.5f);
    acc[0] += (int)nearbyint(0.5);
    acc[0] += (int)floorf(0.5f);
    acc[0] += (int)floor(0.5);
    acc[0] += (int)copysignf(1.0f, -0.5f);
    acc[0] += (int)copysign(1.0, -0.5);
    acc[0] += (int)fminf(1.5f, 0.5f);
    acc[0] += (int)fmin(1.5, 0.5);
    acc[0] += (int)fmaxf(1.5f, 0.5f);
    acc[0] += (int)fmax(1.5, 0.5);
    acc[0] += (int)sinf(1.0f);
    acc[0] += (int)sin(1.0);
    acc[0] += (int)__builtin_sinf(1.0f);
    acc[0] += (int)__builtin_sin(1.0);
    acc[0] += (int)cosf(1.0f);
    acc[0] += (int)cos(1.0);
    acc[0] += (int)__builtin_cosf(1.0f);
    acc[0] += (int)__builtin_cos(1.0);
    acc[0] += (int)logf(1.0f);
    acc[0] += (int)log(1.0);
    acc[0] += (int)__builtin_truncf(1.0f);
    acc[0] += (int)__builtin_trunc(1.0);
    acc[0] += (int)__builtin_rintf(1.0f);
    acc[0] += (int)__builtin_rint(1.0);
    acc[0] += (int)__builtin_nearbyintf(0.5f);
    acc[0] += (int)__builtin_nearbyint(0.5);
    acc[0] += (int)__builtin_floorf(0.5f);
    acc[0] += (int)__builtin_floor(0.5);
    acc[0] += (int)__builtin_copysignf(1.0f, -0.5f);
    acc[0] += (int)__builtin_fminf(1.5f, 0.5f);
    acc[0] += (int)__builtin_fmin(1.5, 0.5);
    acc[0] += (int)__builtin_fmaxf(1.5f, 0.5f);
    acc[0] += (int)__builtin_fmax(1.5, 0.5);
    acc[0] += (int)__builtin_logf(1.0f);
    acc[0] += (int)__builtin_log(1.0);
    acc[0] += __builtin_isinf(1.0);
    acc[0] += __builtin_isfinite(1.0);
    acc[0] += __builtin_isnormal(1.0);
    acc[0] += __builtin_fpclassify(0, 1, 4, 3, 2, 1.0);
  });
  return 0;
}
