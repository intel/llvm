// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics

#include <sycl/multi_ptr.hpp>

SYCL_EXTERNAL void
foo(sycl::multi_ptr<int, sycl::access::address_space::global_space> mptr) {
  mptr.prefetch(0);
}

SYCL_EXTERNAL void
bar(sycl::multi_ptr<int, sycl::access::address_space::global_space,
                    sycl::access::decorated::legacy>
        mptr) {
  mptr.prefetch(0);
}
