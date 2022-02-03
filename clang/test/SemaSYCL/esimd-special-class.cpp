// RUN: %clang_cc1 -fsycl-is-device -verify -internal-isystem %S/Inputs -fsyntax-only %s

// The test is to ensure that the use of sycl_explicit_simd attribute doesn't
// crash when used with sampler or stream. Currently samplers/stream are not
// supported in esimd.

#include "sycl.hpp"
using namespace cl::sycl;
void test() {

  queue q;

  q.submit([&](handler &h) {
    cl::sycl::sampler Smplr;
    cl::sycl::stream Stream(1024, 128, h);
    // expected-note@+1{{in instantiation of function template specialization}}
    h.single_task<class SamplerTester>(
        // expected-error@+1{{type 'sampler' is not supported in ESIMD context}}
        [=]() [[intel::sycl_explicit_simd]] { Smplr.use(); });

    // expected-note@+1{{in instantiation of function template specialization}}
    h.single_task<class StreamTester>(
        // expected-error@+1{{type 'stream' is not supported in ESIMD context}}
        [=]() [[intel::sycl_explicit_simd]] { Stream.use(); });
  });
}
