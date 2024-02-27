// RUN: %clang_cc1 -fsycl-is-device -std=c++17 -sycl-std=2020 -verify %s
#include "Inputs/sycl.hpp"

// Tests that [[intel::numbanks()]] only applies to constant variables, local variables, static variables, agent memory arguments, non-static data members and device_global variables.

using namespace sycl::ext::oneapi;

[[intel::numbanks(2)]] device_global<int> nonconst_dev_glob; // OK
[[intel::numbanks(8)]] const device_global<int> constdev_glob; // OK
[[intel::numbanks(4)]] static device_global<float> static_dev_glob; // OK

// expected-error@+1{{'numbanks' attribute only applies to constant variables, local variables, static variables, agent memory arguments, non-static data members and device_global variables}}
[[intel::numbanks(2)]] int K;

struct bar {
  [[intel::numbanks(2)]] device_global<int> nonconst_glob3; // OK
  [[intel::numbanks(2)]] const device_global<int> const_glob4; // OK
  [[intel::numbanks(8)]] unsigned int numbanks[64];
};

void foo() {
  [[intel::numbanks(2)]] int A1; // OK
  [[intel::numbanks(4)]] static unsigned int ext_five[64]; // OK
}

void attr_on_const_no_error()
{
  //expected-no-error@+1
  [[intel::numbanks(16)]] const int const_var[64] = {0, 1};
}

//expected-no-error@+1
void attr_on_func_arg([[intel::numbanks(8)]] int pc) {}

struct [[__sycl_detail__::global_variable_allowed]] GlobAllowedVarOnly {
};

// expected-error@+1{{'numbanks' attribute only applies to constant variables, local variables, static variables, agent memory arguments, non-static data members and device_global variables}}
[[intel::numbanks(2)]] GlobAllowedVarOnly GAVO;

[[intel::numbanks(4)]] device_global<int> Good;
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

//expected-error@+1{{'numbanks' attribute only applies to constant variables, local variables, static variables, agent memory arguments, non-static data members and device_global variables}}
[[intel::numbanks(2)]]
__attribute__((opencl_global)) unsigned int ocl_glob_num_p2d[64] = {1, 2, 3};

[[intel::numbanks(8)]]
__attribute__((opencl_constant)) unsigned int const_var[64] = {1, 2, 3}; // OK
