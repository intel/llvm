// RUN: %clang_cc1 -fsycl-is-device -std=c++17 -sycl-std=2020 -verify %s
#include "Inputs/sycl.hpp"

// Tests that [[intel::numbanks()]], [[intel::fpga_register]], [[intel::private_copies()]], [[intel::doublepump]], [[intel::singlepump]], [[intel::merge()]], [[intel::fpga_memory()]], [[intel::bank_bits()]], [[intel::force_pow2_depth()]], [[intel::max_replicates()]], [[intel::bankwidth()]], [[intel::simple_dual_port]] can be applied to device_global variables as well as constant variables, local variables, static variables, agent memory arguments, non-static data members.

using namespace sycl::ext::oneapi;

[[intel::numbanks(4)]] static device_global<float> static_dev_glob;
[[intel::max_replicates(12)]] static device_global<float> static_dev_glob1;
[[intel::force_pow2_depth(1)]] static device_global<float> static_dev_glob2;
[[intel::bankwidth(4)]] static device_global<float> static_dev_glob3;
[[intel::simple_dual_port]] static device_global<float> static_dev_glob4;
[[intel::fpga_memory]] static device_global<float> static_dev_glob5;
[[intel::bank_bits(3, 4)]] static device_global<float> static_dev_glob6;
[[intel::fpga_register]] static device_global<float> static_dev_glob7;
[[intel::doublepump]] static device_global<float> static_dev_glob8;
[[intel::singlepump]] static device_global<float> static_dev_glob9;
[[intel::merge("mrg5", "width")]] static device_global<float> static_dev_glob10;

// expected-error@+1{{'numbanks' attribute only applies to constant variables, local variables, static variables, agent memory arguments, non-static data members and device_global variables}}
[[intel::numbanks(2)]] int K;

// expected-error@+1{{'max_replicates' attribute only applies to constant variables, local variables, static variables, agent memory arguments, non-static data members and device_global variables}}
[[intel::max_replicates(10)]] int K1;

// expected-error@+1{{'force_pow2_depth' attribute only applies to constant variables, local variables, static variables, agent memory arguments, non-static data members and device_global variables}}
[[intel::force_pow2_depth(1)]] int K2;

// expected-error@+1{{'bankwidth' attribute only applies to constant variables, local variables, static variables, agent memory arguments, non-static data members and device_global variables}}
[[intel::bankwidth(8)]] int K3;

// expected-error@+1{{'simple_dual_port' attribute only applies to constant variables, local variables, static variables, agent memory arguments, non-static data members and device_global variables}}
[[intel::simple_dual_port]] int K4;

// expected-error@+1{{'fpga_memory' attribute only applies to constant variables, local variables, static variables, agent memory arguments, non-static data members and device_global variables}}
[[intel::fpga_memory]] int K5;

// expected-error@+1{{'bank_bits' attribute only applies to constant variables, local variables, static variables, agent memory arguments, non-static data members and device_global variables}}
[[intel::bank_bits(3, 4)]] int K6;

// expected-error@+1{{'fpga_register' attribute only applies to constant variables, local variables, static variables, non-static data members and device_global variables}}
[[intel::fpga_register]] int K7;

// expected-error@+1{{'doublepump' attribute only applies to constant variables, local variables, static variables, non-static data members and device_global variables}}
[[intel::doublepump]] int K8;

// expected-error@+1{{'singlepump' attribute only applies to constant variables, local variables, static variables, non-static data members and device_global variables}}
[[intel::singlepump]] int K9;

// expected-error@+1{{'merge' attribute only applies to constant variables, local variables, static variables, non-static data members and device_global variables}}
[[intel::merge("mrg3", "width")]] int K10;

//expected-error@+1{{'private_copies' attribute only applies to const variables, local variables, non-static data members and device_global variables}}
[[intel::private_copies(16)]] int K12;

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

  [[intel::bankwidth(2)]] device_global<int> nonconst_glob3;
  [[intel::bankwidth(4)]] const device_global<int> const_glob3;
  [[intel::bankwidth(16)]] unsigned int bankw[64];

  [[intel::simple_dual_port]] device_global<int> nonconst_glob4;
  [[intel::simple_dual_port]] const device_global<int> const_glob4;
  [[intel::simple_dual_port]] unsigned int simple[64];

  [[intel::fpga_memory]] device_global<int> nonconst_glob5;
  [[intel::fpga_memory("MLAB")]] const device_global<int> const_glob5;
  [[intel::fpga_memory("BLOCK_RAM")]] unsigned int mem_block_ram[32];

  [[intel::bank_bits(3, 4)]] device_global<int> nonconst_glob6;
  [[intel::bank_bits(4, 5)]] const device_global<int> const_glob6;
  [[intel::bank_bits(3, 4)]] unsigned int mem_block_bits[32];

  [[intel::fpga_register]] device_global<int> nonconst_glob7;
  [[intel::fpga_register]] const device_global<int> const_glob7;
  [[intel::fpga_register]] unsigned int reg;

  [[intel::singlepump]] device_global<int> nonconst_glob8;
  [[intel::singlepump]] const device_global<int> const_glob8;
  [[intel::singlepump]] unsigned int spump;

  [[intel::doublepump]] device_global<int> nonconst_glob9;
  [[intel::doublepump]] const device_global<int> const_glob9;
  [[intel::doublepump]] unsigned int dpump;

  [[intel::merge("mrg6", "depth")]] device_global<int> nonconst_glob10;
  [[intel::merge("mrg6", "depth")]] const device_global<int> const_glob10;
  [[intel::merge("mrg6", "width")]] unsigned int mergewidth;

  [[intel::private_copies(32)]] device_global<int> nonconst_glob11;
  [[intel::private_copies(8)]] const device_global<int> const_glob11;
  [[intel::private_copies(8)]] unsigned int pc;
};

struct RandomStruct {
  int M;
};

// expected-error@+1{{'numbanks' attribute only applies to constant variables, local variables, static variables, agent memory arguments, non-static data members and device_global variables}}
[[intel::numbanks(4)]] RandomStruct S;
// expected-error@+1{{'bankwidth' attribute only applies to constant variables, local variables, static variables, agent memory arguments, non-static data members and device_global variables}}
[[intel::bankwidth(4)]] RandomStruct S1;
// expected-error@+1{{'force_pow2_depth' attribute only applies to constant variables, local variables, static variables, agent memory arguments, non-static data members and device_global variables}}
[[intel::force_pow2_depth(1)]] RandomStruct S2;
// expected-error@+1{{'max_replicates' attribute only applies to constant variables, local variables, static variables, agent memory arguments, non-static data members and device_global variables}}
[[intel::max_replicates(8)]] RandomStruct S3;
// expected-error@+1{{'simple_dual_port' attribute only applies to constant variables, local variables, static variables, agent memory arguments, non-static data members and device_global variables}}
[[intel::simple_dual_port]] RandomStruct S4;

// expected-error@+1{{'fpga_memory' attribute only applies to constant variables, local variables, static variables, agent memory arguments, non-static data members and device_global variables}}
[[intel::fpga_memory]] RandomStruct S5;

// expected-error@+1{{'bank_bits' attribute only applies to constant variables, local variables, static variables, agent memory arguments, non-static data members and device_global variables}}
[[intel::bank_bits(4, 5)]] RandomStruct S6;

// expected-error@+1{{'fpga_register' attribute only applies to constant variables, local variables, static variables, non-static data members and device_global variables}}
[[intel::fpga_register]] RandomStruct S7;

// expected-error@+1{{'singlepump' attribute only applies to constant variables, local variables, static variables, non-static data members and device_global variables}}
[[intel::singlepump]] RandomStruct S8;

// expected-error@+1{{'doublepump' attribute only applies to constant variables, local variables, static variables, non-static data members and device_global variables}}
[[intel::doublepump]] RandomStruct S9;

// expected-error@+1{{'merge' attribute only applies to constant variables, local variables, static variables, non-static data members and device_global variables}}
[[intel::merge("mrg1", "width")]] RandomStruct S10;

//expected-error@+1{{'private_copies' attribute only applies to const variables, local variables, non-static data members and device_global variables}}
[[intel::private_copies(32)]] RandomStruct S11;

void foo() {
  [[intel::numbanks(2)]] int A;
  [[intel::numbanks(4)]] static unsigned int ext_five[64];
  [[intel::numbanks(8)]] RandomStruct S;

  [[intel::max_replicates(2)]] int A1;
  [[intel::max_replicates(4)]] static unsigned int ext_five1[64];
  [[intel::max_replicates(24)]] RandomStruct S1;

  [[intel::force_pow2_depth(0)]] int A2;
  [[intel::force_pow2_depth(1)]] static unsigned int ext_five2[64];
  [[intel::force_pow2_depth(0)]] RandomStruct S2;

  [[intel::bankwidth(2)]] int A3;
  [[intel::bankwidth(4)]] static unsigned int ext_five3[64];
  [[intel::bankwidth(8)]] RandomStruct S3;

  [[intel::simple_dual_port]] int A4;
  [[intel::simple_dual_port]] static unsigned int ext_five4[64];
  [[intel::simple_dual_port]] RandomStruct S4;

  [[intel::fpga_memory("BLOCK_RAM")]] int A5;
  [[intel::fpga_memory("MLAB")]] static unsigned int ext_five5[64];
  [[intel::fpga_memory]] RandomStruct S5;

  [[intel::bank_bits(6, 7)]] int A6;
  [[intel::bank_bits(9, 10)]] static unsigned int ext_five6[64];
  [[intel::bank_bits(4, 5)]] RandomStruct S6;

  [[intel::fpga_register]] int A7;
  [[intel::fpga_register]] static unsigned int ext_five7[64];
  [[intel::fpga_register]] RandomStruct S7;

  [[intel::singlepump]] int A8;
  [[intel::singlepump]] static unsigned int ext_five8[64];
  [[intel::singlepump]] RandomStruct S8;

  [[intel::doublepump]] int A9;
  [[intel::doublepump]] static unsigned int ext_five9[64];
  [[intel::doublepump]] RandomStruct S9;

  [[intel::merge("mrg1", "depth")]] int A10;
  [[intel::merge("mrg1", "width")]] static unsigned int ext_five10[64];
  [[intel::merge("mrg1", "width")]] RandomStruct S10;

  [[intel::private_copies(8)]] int A11;
  //expected-error@+1{{'private_copies' attribute only applies to const variables, local variables, non-static data members and device_global variables}}
  [[intel::private_copies(16)]] static unsigned int ext_five11[64];
  [[intel::private_copies(32)]] RandomStruct S11;
}

void attr_on_const_no_error()
{
  [[intel::numbanks(16)]] const int const_var[64] = {0, 1};
  [[intel::max_replicates(16)]] const int const_var_max[64] = {0, 1};
  [[intel::force_pow2_depth(1)]] const int const_var_force[64] = {0, 1};
  [[intel::bankwidth(16)]] const int const_var_bankw[64] = {0, 1};
  [[intel::simple_dual_port]] const int const_var_simple_dual[64] = {0, 1};
  [[intel::fpga_memory]] const int const_var_mem[64] = {0, 1};
  [[intel::bank_bits(6, 7)]] const int const_var_bits[64] = {0, 1};
  [[intel::fpga_register]] const int const_var_regis[64] = {0, 1};
  [[intel::singlepump]] const int const_var_spump[64] = {0, 1};
  [[intel::doublepump]] const int const_var_dpump[64] = {0, 1};
  [[intel::merge("mrg6", "width")]] const int const_var_mergewid[64] = {0, 1};
}

void attr_on_func_arg([[intel::numbanks(8)]] int pc) {}
void attr_on_func_arg1([[intel::max_replicates(8)]] int pc1) {}
void attr_on_func_arg2([[intel::force_pow2_depth(1)]] int pc2) {}
void attr_on_func_arg3([[intel::bankwidth(8)]] int pc3) {}
void attr_on_func_arg4([[intel::simple_dual_port]] int pc4) {}
void attr_on_func_arg5([[intel::fpga_memory]] int pc5) {}
void attr_on_func_arg6([[intel::bank_bits(7, 8)]] int pc6) {}
// expected-error@+1{{'singlepump' attribute only applies to constant variables, local variables, static variables, non-static data members and device_global variables}}
void attr_on_func_arg7([[intel::singlepump]] int pc7) {}
// expected-error@+1{{'doublepump' attribute only applies to constant variables, local variables, static variables, non-static data members and device_global variables}}
void attr_on_func_arg8([[intel::doublepump]] int pc8) {}
// expected-error@+1{{'fpga_register' attribute only applies to constant variables, local variables, static variables, non-static data members and device_global variables}}
void attr_on_func_arg9([[intel::fpga_register]] int pc9) {}
// expected-error@+1{{'merge' attribute only applies to constant variables, local variables, static variables, non-static data members and device_global variables}}
void attr_on_func_arg10([[intel::merge("mrg1", "width")]] int pc10) {}

struct [[__sycl_detail__::global_variable_allowed]] GlobAllowedVarOnly {
};

// expected-error@+1{{'numbanks' attribute only applies to constant variables, local variables, static variables, agent memory arguments, non-static data members and device_global variables}}
[[intel::numbanks(2)]] GlobAllowedVarOnly GAVO;

// expected-error@+1{{'max_replicates' attribute only applies to constant variables, local variables, static variables, agent memory arguments, non-static data members and device_global variables}}
[[intel::max_replicates(20)]] GlobAllowedVarOnly GAVO1;

// expected-error@+1{{'force_pow2_depth' attribute only applies to constant variables, local variables, static variables, agent memory arguments, non-static data members and device_global variables}}
[[intel::force_pow2_depth(0)]] GlobAllowedVarOnly GAVO2;

// expected-error@+1{{'bankwidth' attribute only applies to constant variables, local variables, static variables, agent memory arguments, non-static data members and device_global variables}}
[[intel::bankwidth(16)]] GlobAllowedVarOnly GAVO3;

// expected-error@+1{{'simple_dual_port' attribute only applies to constant variables, local variables, static variables, agent memory arguments, non-static data members and device_global variables}}
[[intel::simple_dual_port]] GlobAllowedVarOnly GAVO4;

// expected-error@+1{{'fpga_memory' attribute only applies to constant variables, local variables, static variables, agent memory arguments, non-static data members and device_global variables}}
[[intel::fpga_memory]] GlobAllowedVarOnly GAVO5;

// expected-error@+1{{'bank_bits' attribute only applies to constant variables, local variables, static variables, agent memory arguments, non-static data members and device_global variables}}
[[intel::bank_bits(6, 7)]] GlobAllowedVarOnly GAVO6;

// expected-error@+1{{'fpga_register' attribute only applies to constant variables, local variables, static variables, non-static data members and device_global variables}}
[[intel::fpga_register]] GlobAllowedVarOnly GAVO7;

// expected-error@+1{{'singlepump' attribute only applies to constant variables, local variables, static variables, non-static data members and device_global variables}}
[[intel::singlepump]] GlobAllowedVarOnly GAVO8;

// expected-error@+1{{'doublepump' attribute only applies to constant variables, local variables, static variables, non-static data members and device_global variables}}
[[intel::doublepump]] GlobAllowedVarOnly GAVO9;

// expected-error@+1{{'merge' attribute only applies to constant variables, local variables, static variables, non-static data members and device_global variables}}
[[intel::merge("mrg5", "width")]] GlobAllowedVarOnly GAVO10;

//expected-error@+1{{'private_copies' attribute only applies to const variables, local variables, non-static data members and device_global variables}}
[[intel::private_copies(16)]] GlobAllowedVarOnly GAVO11;

[[intel::numbanks(4)]] device_global<int> Good;
[[intel::numbanks(4)]] extern device_global<int> Bad;

[[intel::max_replicates(8)]] device_global<int> Good1;
[[intel::max_replicates(10)]] extern device_global<int> Bad1;

[[intel::force_pow2_depth(0)]] device_global<int> Good2;
[[intel::force_pow2_depth(0)]] extern device_global<int> Bad2;

[[intel::bankwidth(2)]] device_global<int> Good3;
[[intel::bankwidth(2)]] extern device_global<int> Bad3;

[[intel::simple_dual_port]] device_global<int> Good4;
[[intel::simple_dual_port]] extern device_global<int> Bad4;

[[intel::fpga_memory("MLAB")]] device_global<int> Good5;
[[intel::fpga_memory("BLOCK_RAM")]] extern device_global<int> Bad5;

[[intel::bank_bits(6, 7)]] device_global<int> Good6;
[[intel::bank_bits(7, 8)]] extern device_global<int> Bad6;

[[intel::fpga_register]] device_global<int> Good7;
[[intel::fpga_register]] extern device_global<int> Bad7;

[[intel::doublepump]] device_global<int> Good8;
[[intel::doublepump]] extern device_global<int> Bad8;

[[intel::singlepump]] device_global<int> Good9;
[[intel::singlepump]] extern device_global<int> Bad9;

[[intel::merge("mrg1", "depth")]] device_global<int> Good10;
[[intel::merge("mrg1", "depth")]] extern device_global<int> Bad10;

[[intel::private_copies(16)]] device_global<int> Good11;
[[intel::private_copies(16)]] extern device_global<int> Bad11;

int main() {
  sycl::kernel_single_task<class KernelName1>([=]() {
    Good.get();
    Good1.get();
    Good2.get();
    Good3.get();
    Good4.get();
    Good5.get();
    Good6.get();
    Good7.get();
    Good8.get();
    Good9.get();
    Good10.get();
    Good11.get();

    // expected-error@+1 {{invalid reference to 'device_global' variable; external 'device_global' variable must be marked with SYCL_EXTERNAL macro}}
    Bad.get();
    // expected-error@+1 {{invalid reference to 'device_global' variable; external 'device_global' variable must be marked with SYCL_EXTERNAL macro}}
    Bad1.get();
    // expected-error@+1 {{invalid reference to 'device_global' variable; external 'device_global' variable must be marked with SYCL_EXTERNAL macro}}
    Bad2.get();
    // expected-error@+1 {{invalid reference to 'device_global' variable; external 'device_global' variable must be marked with SYCL_EXTERNAL macro}}
    Bad3.get();
    // expected-error@+1 {{invalid reference to 'device_global' variable; external 'device_global' variable must be marked with SYCL_EXTERNAL macro}}
    Bad4.get();
    // expected-error@+1 {{invalid reference to 'device_global' variable; external 'device_global' variable must be marked with SYCL_EXTERNAL macro}}
    Bad5.get();
    // expected-error@+1 {{invalid reference to 'device_global' variable; external 'device_global' variable must be marked with SYCL_EXTERNAL macro}}
    Bad6.get();
    // expected-error@+1 {{invalid reference to 'device_global' variable; external 'device_global' variable must be marked with SYCL_EXTERNAL macro}}
    Bad7.get();
    // expected-error@+1 {{invalid reference to 'device_global' variable; external 'device_global' variable must be marked with SYCL_EXTERNAL macro}}
    Bad8.get();
    // expected-error@+1 {{invalid reference to 'device_global' variable; external 'device_global' variable must be marked with SYCL_EXTERNAL macro}}
    Bad9.get();
    // expected-error@+1 {{invalid reference to 'device_global' variable; external 'device_global' variable must be marked with SYCL_EXTERNAL macro}}
    Bad10.get();
    // expected-error@+1 {{invalid reference to 'device_global' variable; external 'device_global' variable must be marked with SYCL_EXTERNAL macro}}
    Bad11.get();

    (void)GAVO;
    (void)GAVO1;
    (void)GAVO2;
    (void)GAVO3;
    (void)GAVO4;
    (void)GAVO5;
    (void)GAVO6;
    (void)GAVO7;
    (void)GAVO8;
    (void)GAVO9;
    (void)GAVO10;
    (void)GAVO11;
  });
  return 0;
}

//expected-error@+1{{'numbanks' attribute only applies to constant variables, local variables, static variables, agent memory arguments, non-static data members and device_global variables}}
[[intel::numbanks(2)]]
__attribute__((opencl_global)) unsigned int ocl_glob_num_p2d[64] = {1, 2, 3};

//expected-error@+1{{'max_replicates' attribute only applies to constant variables, local variables, static variables, agent memory arguments, non-static data members and device_global variables}}
[[intel::max_replicates(20)]]
__attribute__((opencl_global)) unsigned int ocl_glob_max_p2d[64] = {1, 2, 3};

//expected-error@+1{{'bankwidth' attribute only applies to constant variables, local variables, static variables, agent memory arguments, non-static data members and device_global variables}}
[[intel::bankwidth(32)]]
__attribute__((opencl_global)) unsigned int ocl_glob_bankw_p2d[64] = {1, 2, 3};

//expected-error@+1{{'simple_dual_port' attribute only applies to constant variables, local variables, static variables, agent memory arguments, non-static data members and device_global variables}}
[[intel::simple_dual_port]]
__attribute__((opencl_global)) unsigned int ocl_glob_simple_p2d[64] = {1, 2, 3};

//expected-error@+1{{'fpga_memory' attribute only applies to constant variables, local variables, static variables, agent memory arguments, non-static data members and device_global variables}}
[[intel::fpga_memory("MLAB")]]
__attribute__((opencl_global)) unsigned int ocl_glob_memory_p2d[64] = {1, 2, 3};

//expected-error@+1{{'bank_bits' attribute only applies to constant variables, local variables, static variables, agent memory arguments, non-static data members and device_global variables}}
[[intel::bank_bits(7, 8)]]
__attribute__((opencl_global)) unsigned int ocl_glob_bank_bits_p2d[64] = {1, 2, 3};

//expected-error@+1{{'fpga_register' attribute only applies to constant variables, local variables, static variables, non-static data members and device_global variables}}
[[intel::fpga_register]]
__attribute__((opencl_global)) unsigned int ocl_glob_reg_p2d[64] = {1, 2, 3};

//expected-error@+1{{'doublepump' attribute only applies to constant variables, local variables, static variables, non-static data members and device_global variables}}
[[intel::doublepump]]
__attribute__((opencl_global)) unsigned int ocl_glob_dpump_p2d[64] = {1, 2, 3};

//expected-error@+1{{'singlepump' attribute only applies to constant variables, local variables, static variables, non-static data members and device_global variables}}
[[intel::singlepump]]
__attribute__((opencl_global)) unsigned int ocl_glob_spump_p2d[64] = {1, 2, 3};

//expected-error@+1{{'merge' attribute only applies to constant variables, local variables, static variables, non-static data members and device_global variables}}
[[intel::merge("mrg4", "depth")]]
__attribute__((opencl_global)) unsigned int ocl_glob_mer_p2d[64] = {1, 2, 3};

//expected-error@+1{{'private_copies' attribute only applies to const variables, local variables, non-static data members and device_global variables}}
[[intel::private_copies(8)]]
__attribute__((opencl_global)) unsigned int ocl_glob_pc_p2d[64] = {1, 2, 3};

//expected-error@+1{{'private_copies' attribute only applies to const variables, local variables, non-static data members and device_global variables}}
[[intel::private_copies(8)]]
__attribute__((opencl_constant)) unsigned int const_var_private_copies[64] = {1, 2, 3};

[[intel::merge("mrg5", "width")]]
__attribute__((opencl_constant)) unsigned int const_var_merge[64] = {1, 2, 3};

[[intel::fpga_register]]
__attribute__((opencl_constant)) unsigned int const_var_fpga_register[64] = {1, 2, 3};

[[intel::fpga_memory]]
__attribute__((opencl_constant)) unsigned int const_var_fpga_memory[64] = {1, 2, 3};

[[intel::bank_bits(2, 3)]]
__attribute__((opencl_constant)) unsigned int const_var_bank_bits[64] = {1, 2, 3};

[[intel::numbanks(8)]]
__attribute__((opencl_constant)) unsigned int const_var[64] = {1, 2, 3};

[[intel::max_replicates(16)]]
__attribute__((opencl_constant)) unsigned int const_var_max_rep[64] = {1, 2, 3};

[[intel::force_pow2_depth(0)]]
__attribute__((opencl_constant)) unsigned int const_force_var[64] = {1, 2, 3};

[[intel::bankwidth(32)]]
__attribute__((opencl_constant)) unsigned int const_bankw_var[64] = {1, 2, 3};

[[intel::simple_dual_port]]
__attribute__((opencl_constant)) unsigned int const_simple_var[64] = {1, 2, 3};

[[intel::doublepump]]
__attribute__((opencl_constant)) unsigned int const_dpump_var[64] = {1, 2, 3};

[[intel::singlepump]]
__attribute__((opencl_constant)) unsigned int const_spump_var[64] = {1, 2, 3};
