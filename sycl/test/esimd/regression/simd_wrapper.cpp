// RUN: %clangxx -fsycl -fsyntax-only -fsycl-device-only -Xclang -verify %s

#include <limits>
#include <sycl/ext/intel/esimd.hpp>
#include <utility>

// This is a regression test for simd object being non-trivial copy
// constructible. In order to fix it, you need to provide copy constructor for
// SimdWrapper, e.g.:
//   SimdWrapper (const SimdWrapper& rhs) : v1(rhs.v1) {}

using namespace sycl::ext::intel::esimd;

struct SimdWrapper {
  union {
    // expected-note@+1 {{copy constructor of 'SimdWrapper' is implicitly deleted because variant field '' has a non-trivial copy constructor}}
    struct {
      simd<int, 4> v1;
    };
  };
  SimdWrapper() {}
};

void encapsulate_simd() SYCL_ESIMD_FUNCTION {
  SimdWrapper s1;
  // expected-error@+1 {{call to implicitly-deleted copy constructor of 'SimdWrapper'}}
  SimdWrapper s2(s1);
}
