// RUN: %clang_cc1 -fsycl-is-device -std=c++17 -sycl-std=2020 -verify %s
#include "Inputs/sycl.hpp"

// Tests that SYCL_EXTERNAL can be applied to device_global variables, and cannot be applied to other variables.
using namespace sycl::ext::oneapi;

SYCL_EXTERNAL device_global<int> glob;
// expected-error@+1{{'sycl_device' attribute cannot be applied to a variable without external linkage}}
SYCL_EXTERNAL static device_global<float> static_glob;

namespace foo {
SYCL_EXTERNAL device_global<int> same_name;
}

struct RandomStruct {
  int M;
};

// expected-error@+1{{'sycl_device' attribute can only be applied to 'device_global' variables}}
SYCL_EXTERNAL RandomStruct S;

namespace {
// expected-error@+1{{'sycl_device' attribute cannot be applied to a variable without external linkage}}
SYCL_EXTERNAL device_global<int> same_name;

struct UnnX {};
} // namespace

// expected-error@+1{{'sycl_device' attribute cannot be applied to a variable without external linkage}}
SYCL_EXTERNAL device_global<UnnX> dg_x;

// expected-error@+1{{'sycl_device' attribute can only be applied to 'device_global' variables}}
SYCL_EXTERNAL int AAA;

struct B {
  SYCL_EXTERNAL static device_global<int> Member;
};

void foofoo() {
  // expected-warning@+1{{'sycl_device' attribute only applies to functions and global variables}}
  SYCL_EXTERNAL RandomStruct S;
  // expected-warning@+1{{'sycl_device' attribute only applies to functions and global variables}}
  SYCL_EXTERNAL int A;
}

template <typename T> struct NonDevGlob {
};

template <typename T> struct TS {
  SYCL_EXTERNAL static device_global<T> D;
  // expected-error@+1{{'sycl_device' attribute can only be applied to 'device_global' variables}}
  SYCL_EXTERNAL static NonDevGlob<T> ND;
};

// expected-note@+1 {{in instantiation of template class 'TS<int>' requested here}}
TS<int> A;

struct [[__sycl_detail__::global_variable_allowed]] GlobAllowedOnly {
};

// expected-error@+1{{'sycl_device' attribute can only be applied to 'device_global' variables}}
SYCL_EXTERNAL GlobAllowedOnly GAO;


SYCL_EXTERNAL extern device_global<int> Good;
extern device_global<int> Bad;

int main() {
  sycl::kernel_single_task<class KernelName1>([=]() {
    Good.get();
    // expected-error@+1 {{invalid reference to 'device_global' variable; external 'device_global' variable must be marked with SYCL_EXTERNAL macro}}
    Bad.get();

    (void)GAO;
  });
  return 0;
}
