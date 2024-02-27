// RUN: %clang_cc1 -fsycl-is-device -std=c++17 -sycl-std=2020 -verify %s
#include "Inputs/sycl.hpp"

// Tests that [[intel::numbanks()]], [[intel::force_pow2_depth()]], [[intel::max_replicates()]] only applies to constant variables, local variables, static variables, agent memory arguments, non-static data members and device_global variables.

using namespace sycl::ext::oneapi;

[[intel::numbanks(4)]] static device_global<float> static_dev_glob;
[[intel::max_replicates(12)]] static device_global<float> static_dev_glob1;
[[intel::force_pow2_depth(1)]] static device_global<float> static_dev_glob2;

// expected-error@+1{{'numbanks' attribute only applies to constant variables, local variables, static variables, agent memory arguments, non-static data members and device_global variables}}
[[intel::numbanks(2)]] int K;

// expected-error@+1{{'max_replicates' attribute only applies to constant variables, local variables, static variables, agent memory arguments, non-static data members and device_global variables}}
[[intel::max_replicates(10)]] int K1;

// expected-error@+1{{'force_pow2_depth' attribute only applies to constant variables, local variables, static variables, agent memory arguments, non-static data members and device_global variables}}
[[intel::force_pow2_depth(1)]] int K2;

struct bar {
  [[intel::numbanks(2)]] device_global<int> nonconst_glob;
  [[intel::numbanks(4)]] const device_global<int> const_glob;
  [[intel::numbanks(8)]] unsigned int numbanks[64];

  [[intel::max_replicates(2)]] device_global<int> nonconst_glob1;
  [[intel::max_replicates(4)]] const device_global<int> const_glob1;
  [[intel::max_replicates(8)]] unsigned int max_rep[64];

  [[intel::force_pow2_depth(0)]] device_global<int> nonconst_glob2;
  [[intel::force_pow2_depth(0)]] const device_global<int> const_glob2;
  [[intel::force_pow2_depth(1)]] unsigned int force_dep[64];
};

void foo() {
  [[intel::numbanks(2)]] int A;
  [[intel::numbanks(4)]] static unsigned int ext_five[64];
  [[intel::max_replicates(2)]] int A1;
  [[intel::max_replicates(4)]] static unsigned int ext_five1[64];
  [[intel::force_pow2_depth(0)]] int A2;
  [[intel::force_pow2_depth(1)]] static unsigned int ext_five2[64];
}

void attr_on_const_no_error()
{
  [[intel::numbanks(16)]] const int const_var[64] = {0, 1};
  [[intel::max_replicates(16)]] const int const_var_max[64] = {0, 1};
  [[intel::force_pow2_depth(1)]] const int const_var_force[64] = {0, 1};
}

void attr_on_func_arg([[intel::numbanks(8)]] int pc) {}
void attr_on_func_arg1([[intel::max_replicates(8)]] int pc1) {}
void attr_on_func_arg2([[intel::force_pow2_depth(1)]] int pc2) {}

struct [[__sycl_detail__::global_variable_allowed]] GlobAllowedVarOnly {
};

// expected-error@+1{{'numbanks' attribute only applies to constant variables, local variables, static variables, agent memory arguments, non-static data members and device_global variables}}
[[intel::numbanks(2)]] GlobAllowedVarOnly GAVO;

// expected-error@+1{{'max_replicates' attribute only applies to constant variables, local variables, static variables, agent memory arguments, non-static data members and device_global variables}}
[[intel::max_replicates(20)]] GlobAllowedVarOnly GAVO1;

// expected-error@+1{{'force_pow2_depth' attribute only applies to constant variables, local variables, static variables, agent memory arguments, non-static data members and device_global variables}}
[[intel::force_pow2_depth(0)]] GlobAllowedVarOnly GAVO2;

[[intel::numbanks(4)]] device_global<int> Good;
[[intel::numbanks(4)]] extern device_global<int> Bad;

[[intel::max_replicates(8)]] device_global<int> Good1;
[[intel::max_replicates(10)]] extern device_global<int> Bad1;

[[intel::force_pow2_depth(0)]] device_global<int> Good2;
[[intel::force_pow2_depth(0)]] extern device_global<int> Bad2;

int main() {
  sycl::kernel_single_task<class KernelName1>([=]() {
    Good.get();
    Good1.get();
    Good2.get();
    // expected-error@+1 {{invalid reference to 'device_global' variable; external 'device_global' variable must be marked with SYCL_EXTERNAL macro}}
    Bad.get();
    // expected-error@+1 {{invalid reference to 'device_global' variable; external 'device_global' variable must be marked with SYCL_EXTERNAL macro}}
    Bad1.get();
    // expected-error@+1 {{invalid reference to 'device_global' variable; external 'device_global' variable must be marked with SYCL_EXTERNAL macro}}
    Bad2.get();
    (void)GAVO;
    (void)GAVO1;
    (void)GAVO2;
  });
  return 0;
}

//expected-error@+1{{'numbanks' attribute only applies to constant variables, local variables, static variables, agent memory arguments, non-static data members and device_global variables}}
[[intel::numbanks(2)]]
__attribute__((opencl_global)) unsigned int ocl_glob_num_p2d[64] = {1, 2, 3};

//expected-error@+1{{'max_replicates' attribute only applies to constant variables, local variables, static variables, agent memory arguments, non-static data members and device_global variables}}
[[intel::max_replicates(20)]]
__attribute__((opencl_global)) unsigned int ocl_glob_max_p2d[64] = {1, 2, 3};

[[intel::numbanks(8)]]
__attribute__((opencl_constant)) unsigned int const_var[64] = {1, 2, 3};

[[intel::max_replicates(16)]]
__attribute__((opencl_constant)) unsigned int const_var_max_rep[64] = {1, 2, 3};

[[intel::force_pow2_depth(0)]]
__attribute__((opencl_constant)) unsigned int const_force_var[64] = {1, 2, 3};
