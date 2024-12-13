// RUN: %clangxx -fsycl %s -o %t_default.out
// RUN: %t_default.out

// FIXME: Everything should compile cleanly.
// RUN: %clangxx -fsycl -fsycl-device-only -DCHECK_ERRORS -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note,error %s

#include <sycl/vector.hpp>

int main() {
  sycl::vec<int, 4> v{1, 2, 3, 4};
  auto sw = v.swizzle<1, 2>();
  assert(sw.lo()[0] == 2);
  assert(sw.hi()[0] == 3);

  // FIXME: Should be "4":
  assert((sw + sw).lo()[0] == 2);

  // FIXME: The below should compile.
#if CHECK_ERRORS
  // expected-error-re@+1 {{no template named 'swizzle' in {{.*}}}}
  assert(sw.swizzle<0>()[0] == 2);
  // expected-error-re@+1 {{no template named 'swizzle' in {{.*}}}}
  assert(sw.swizzle<1>()[0] == 3);

  {
    // expected-error-re@+1 {{no template named 'swizzle' in {{.*}}}}
    auto tmp = sw.swizzle<1, 0>();
    assert(tmp[0] == 3);
    assert(tmp[1] == 2);
  }

  {
    // expected-error-re@+1 {{no template named 'swizzle' in {{.*}}}}
    auto tmp = (sw + sw).swizzle<1, 0>();

    assert(tmp[0] == 6);
    assert(tmp[1] == 4);
  }
#endif

  return 0;
}
