// RUN: %clang_cc1 -fsycl-is-device -std=c++17 -sycl-std=2020 -verify %s
#include "Inputs/sycl.hpp"

// Tests that [[intel::numbanks()]] can be applied to device_global variables.
using namespace sycl::ext::oneapi;

[[intel::numbanks(2)]] device_global<int> dev_glob; // OK
[[intel::numbanks(4)]] static device_global<float> static_dev_glob; // OK

// expected-error@+1{{'numbanks' attribute only applies to constant variables, local variables, static variables, agent memory arguments, non-static data members, and non-constant device_global variables}}
[[intel::numbanks(2)]] int K;

struct bar {
  [[intel::numbanks(2)]] /*const*/ device_global<int> const_glob3; // OK
  [[intel::numbanks(2)]] const device_global<int> const_glob4; // OK
};

void foo() {
  [[intel::numbanks(2)]] int A1; // OK
}

struct [[__sycl_detail__::global_variable_allowed]] GlobAllowedVarOnly {
};

// expected-error@+1{{'numbanks' attribute only applies to constant variables, local variables, static variables, agent memory arguments, non-static data members, and non-constant device_global variables}}
[[intel::numbanks(2)]] GlobAllowedVarOnly GAVO;

[[intel::numbanks(4)]] /*const*/ device_global<int> Good;
[[intel::numbanks(4)]] extern device_global<int> Bad;

int main() {
  sycl::kernel_single_task<class KernelName1>([=]() {
    Good.get();
    // expected-error@+1 {{invalid reference to 'device_global' variable; external 'device_global' variable must be marked with SYCL_EXTERNAL macro}}
    Bad.get();
    (void)GAVO;
  });
  return 0;
}

//expected-error@+1{{'numbanks' attribute only applies to constant variables, local variables, static variables, agent memory arguments, non-static data members, and non-constant device_global variables}}
[[intel::numbanks(2)]]
__attribute__((opencl_global)) unsigned int ocl_glob_num_p2d[64] = {1, 2, 3};

