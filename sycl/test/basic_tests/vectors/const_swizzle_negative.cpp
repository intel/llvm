// REQUIRES: preview-breaking-changes-supported
// RUN: %clangxx -fsycl-device-only -ferror-limit=0 -Xclang -fsycl-is-device -fsyntax-only -fpreview-breaking-changes -Xclang -verify -Xclang -verify-ignore-unexpected=note %s

#include <sycl/sycl.hpp>

int main() {
  sycl::queue Q;
  Q.single_task([]() {
    const sycl::vec<int, 4> X{1};

    // expected-error@+1 {{no viable overloaded '='}}
    X.swizzle<0>() = 1;
    // expected-error@+1 {{no viable overloaded '='}}
    X.swizzle<0>() = sycl::vec<int, 1>{1};
    // expected-error@+1 {{no viable overloaded '='}}
    X.swizzle<0, 2>() = sycl::vec<int, 2>{1};

    // expected-error@+1 {{no viable overloaded '+='}}
    X.swizzle<0>() += 1;
    // expected-error@+1 {{no viable overloaded '-='}}
    X.swizzle<0>() -= 1;
    // expected-error@+1 {{no viable overloaded '*='}}
    X.swizzle<0>() *= 1;
    // expected-error@+1 {{no viable overloaded '/='}}
    X.swizzle<0>() /= 1;
    // expected-error@+1 {{no viable overloaded '%='}}
    X.swizzle<0>() %= 1;
    // expected-error@+1 {{no viable overloaded '&='}}
    X.swizzle<0>() &= 1;
    // expected-error@+1 {{no viable overloaded '|='}}
    X.swizzle<0>() |= 1;
    // expected-error@+1 {{no viable overloaded '^='}}
    X.swizzle<0>() ^= 1;
    // expected-error@+1 {{no viable overloaded '>>='}}
    X.swizzle<0>() >>= 1;
    // expected-error@+1 {{no viable overloaded '<<='}}
    X.swizzle<0>() <<= 1;

    // expected-error@+1 {{no viable overloaded '+='}}
    X.swizzle<0>() += sycl::vec<int, 1>{1};
    // expected-error@+1 {{no viable overloaded '-='}}
    X.swizzle<0>() -= sycl::vec<int, 1>{1};
    // expected-error@+1 {{no viable overloaded '*='}}
    X.swizzle<0>() *= sycl::vec<int, 1>{1};
    // expected-error@+1 {{no viable overloaded '/='}}
    X.swizzle<0>() /= sycl::vec<int, 1>{1};
    // expected-error@+1 {{no viable overloaded '%='}}
    X.swizzle<0>() %= sycl::vec<int, 1>{1};
    // expected-error@+1 {{no viable overloaded '&='}}
    X.swizzle<0>() &= sycl::vec<int, 1>{1};
    // expected-error@+1 {{no viable overloaded '|='}}
    X.swizzle<0>() |= sycl::vec<int, 1>{1};
    // expected-error@+1 {{no viable overloaded '^='}}
    X.swizzle<0>() ^= sycl::vec<int, 1>{1};
    // expected-error@+1 {{no viable overloaded '>>='}}
    X.swizzle<0>() >>= sycl::vec<int, 1>{1};
    // expected-error@+1 {{no viable overloaded '<<='}}
    X.swizzle<0>() <<= sycl::vec<int, 1>{1};

    // expected-error@+1 {{no viable overloaded '+='}}
    X.swizzle<0>() += X.swizzle<1>();
    // expected-error@+1 {{no viable overloaded '-='}}
    X.swizzle<0>() -= X.swizzle<1>();
    // expected-error@+1 {{no viable overloaded '*='}}
    X.swizzle<0>() *= X.swizzle<1>();
    // expected-error@+1 {{no viable overloaded '/='}}
    X.swizzle<0>() /= X.swizzle<1>();
    // expected-error@+1 {{no viable overloaded '%='}}
    X.swizzle<0>() %= X.swizzle<1>();
    // expected-error@+1 {{no viable overloaded '&='}}
    X.swizzle<0>() &= X.swizzle<1>();
    // expected-error@+1 {{no viable overloaded '|='}}
    X.swizzle<0>() |= X.swizzle<1>();
    // expected-error@+1 {{no viable overloaded '^='}}
    X.swizzle<0>() ^= X.swizzle<1>();
    // expected-error@+1 {{no viable overloaded '>>='}}
    X.swizzle<0>() >>= X.swizzle<1>();
    // expected-error@+1 {{no viable overloaded '<<='}}
    X.swizzle<0>() <<= X.swizzle<1>();

    // expected-error@+1 {{cannot increment value of type}}
    X.swizzle<0>()++;
    // expected-error@+1 {{cannot increment value of type}}
    ++X.swizzle<0>();
    // expected-error@+1 {{cannot decrement value of type}}
    X.swizzle<0>()--;
    // expected-error@+1 {{cannot decrement value of type}}
    --X.swizzle<0>();

    int I = 1;
    // expected-error@+1 {{no matching member function for call to 'load'}}
    X.load(0,
           sycl::address_space_cast<sycl::access::address_space::private_space,
                                    sycl::access::decorated::no>(&I));
  });
  return 0;
}
