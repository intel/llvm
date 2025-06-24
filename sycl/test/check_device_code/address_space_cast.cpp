// This test checks that the dynamic address casts can be optimized out
// (via LowerSYCLAddressSpaceCasts) when the address spaces are known at compile
// time by checking that only one of the from {foo, bar, baz} are called. Note:
// this can only happen if dispatch is inlined, which is why we use
// __attribute__((always_inline)) on it.

// XFAIL: *
// RUN: %clangxx -O1 -DADDR_SPACE=global_space -S -emit-llvm -fsycl -fsycl-device-only %s -o - \
// RUN: | FileCheck %s --implicit-check-not=bar --implicit-check-not=baz
// RUN: %clangxx -O1 -DADDR_SPACE=local_space -S -emit-llvm -fsycl -fsycl-device-only %s -o - \
// RUN: | FileCheck %s --implicit-check-not=foo --implicit-check-not=baz
// RUN: %clangxx -O1 -DADDR_SPACE=private_space -S -emit-llvm -fsycl -fsycl-device-only %s -o - \
// RUN: | FileCheck %s --implicit-check-not=foo --implicit-check-not=bar

#include <sycl/ext/oneapi/experimental/address_cast.hpp>
#include <sycl/multi_ptr.hpp>

SYCL_EXTERNAL float foo(float x);
SYCL_EXTERNAL float bar(float x);
SYCL_EXTERNAL float baz(float x);

using namespace sycl::ext::oneapi::experimental;
using namespace sycl::access;

__attribute__((always_inline)) float dispatch(float *x) {
  if (dynamic_address_cast<address_space::global_space>(x)
          .get_decorated() /*!=nullptr*/) {
    return foo(*x);
  } else if (dynamic_address_cast<address_space::local_space>(x)
                 .get_decorated() /*!=nullptr*/) {
    return bar(*x);
  } else if (dynamic_address_cast<address_space::private_space>(x)
                 .get_decorated() /*!=nullptr*/) {
    return baz(*x);
  }

  return -1;
}

SYCL_EXTERNAL float
fun(sycl::multi_ptr<float, address_space::ADDR_SPACE, decorated::no> x) {
  return dispatch(x.get());
}
