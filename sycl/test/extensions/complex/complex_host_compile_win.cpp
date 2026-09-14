// Regression test for CMPLRLLVM-77544. The SYCL complex experimental
// extension's inline-visibility macro previously expanded to an unguarded
// __attribute__((visibility(...), always_inline)), which cl.exe cannot parse.
// That surfaced when icx/clang++ drives cl.exe as the SYCL host compiler
// (e.g. torch-xpu-ops builds on Windows). The header now guards the macro
// with _MSC_VER; this test ensures the partial specialization of complex<T>
// keeps parsing under MSVC host compilation.

// RUN: %clangxx -fsycl -fsycl-host-compiler=cl -fsycl-host-compiler-options='/std:c++20 /MD /EHsc /permissive- /DSYCL_DISABLE_FSYCL_SYCLHPP_WARNING' %s -c -o %t.o
// REQUIRES: windows

#define SYCL_EXT_ONEAPI_COMPLEX
#include <sycl/ext/oneapi/experimental/complex/complex.hpp>
#include <sycl/sycl.hpp>

namespace syclex = sycl::ext::oneapi::experimental;

// Force instantiation of the complex<T, enable_if<is_genfloat<T>>> partial
// specialization for each supported floating-point type. Prior to the fix,
// simply parsing the class body under cl.exe hit C2059 on the first member
// decorated with _SYCL_EXT_CPLX_INLINE_VISIBILITY.
void force_instantiate() {
  syclex::complex<float> zf(1.0f, 2.0f);
  syclex::complex<double> zd(1.0, 2.0);
  syclex::complex<sycl::half> zh(sycl::half{1.0f}, sycl::half{2.0f});
  (void)syclex::exp(zf).real();
  (void)syclex::exp(zd).real();
  (void)syclex::exp(zh).real();
}
