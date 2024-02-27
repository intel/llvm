// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -sycl-std=2020 -fsyntax-only -verify %s

// Tests diagnostics for Intel FPGA memory attributes.

#include "sycl.hpp"

sycl::queue deviceQueue;

void diagnostics()
{
  // **doublepump
  //expected-warning@+1 {{unknown attribute 'doublepump' ignored}}
  [[intelfpga::doublepump]] unsigned int doublepump_var[64];

  //expected-error@+2{{attributes are not compatible}}
  [[intel::doublepump]]
  [[intel::singlepump]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int dpump_spump[64];

  //expected-warning@+2{{attribute 'doublepump' is already applied}}
  [[intel::doublepump]] //expected-note{{previous attribute is here}}
  [[intel::doublepump]] unsigned int dpump[64];

  //expected-error@+2{{attributes are not compatible}}
  [[intel::doublepump]]
  [[intel::fpga_register]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int dpump_reg[64];

  // **singlepump
  //expected-warning@+1 {{unknown attribute 'singlepump' ignored}}
  [[intelfpga::singlepump]] unsigned int singlepump_var[64];

  //expected-error@+1{{attributes are not compatible}}
  [[intel::singlepump, intel::doublepump]]
  //expected-note@-1 {{conflicting attribute is here}}
  unsigned int spump_dpump[64];

  //expected-warning@+2{{attribute 'singlepump' is already applied}}
  [[intel::singlepump]] //expected-note{{previous attribute is here}}
  [[intel::singlepump]] unsigned int spump[64];

  //expected-error@+2{{attributes are not compatible}}
  [[intel::singlepump]]
  [[intel::fpga_register]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int spump_reg[64];

  // **fpga_register
  //expected-warning@+1 {{unknown attribute 'register' ignored}}
  [[intelfpga::register]] unsigned int reg_var[64];

  //expected-warning@+2{{attribute 'fpga_register' is already applied}}
  [[intel::fpga_register]] //expected-note{{previous attribute is here}}
  [[intel::fpga_register]] unsigned int reg_reg[64];

  //expected-error@+2{{attributes are not compatible}}
  [[intel::fpga_register]]
  [[intel::singlepump]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int reg_spump[64];

  //expected-error@+2{{attributes are not compatible}}
  [[intel::fpga_register]]
  [[intel::doublepump]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int reg_dpump[64];

  //expected-error@+2{{attributes are not compatible}}
  [[intel::fpga_register]]
  [[intel::fpga_memory]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int reg_memory[64];

  //expected-error@+2{{'bank_bits' and 'fpga_register' attributes are not compatible}}
  [[intel::fpga_register]]
  [[intel::bank_bits(4, 5)]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int reg_bankbits[64];

  // Checking of incompatible attributes.
  //expected-error@+2{{'bankwidth' and 'fpga_register' attributes are not compatible}}
  [[intel::fpga_register]]
  [[intel::bankwidth(16)]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int reg_bankwidth[64];

  //expected-error@+2{{attributes are not compatible}}
  [[intel::fpga_register]]
  [[intel::private_copies(16)]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int reg_private_copies[64];

  // Checking of different argument values.
  //expected-warning@+2{{attribute 'numbanks' is already applied with different arguments}}
  [[intel::numbanks(8)]] //expected-note{{previous attribute is here}}
  [[intel::numbanks(16)]] unsigned int max_numb[64];

  // Checking of incompatible attributes.
  //expected-error@+2{{attributes are not compatible}}
  [[intel::fpga_register]]
  [[intel::numbanks(8)]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int reg_numbanks[64];

  //expected-error@+2{{attributes are not compatible}}
  [[intel::numbanks(16)]]
  [[intel::fpga_register]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int numbanks_reg[64];

  //expected-error@+2{{attributes are not compatible}}
  [[intel::fpga_register]]
  [[intel::merge("mrg1", "depth")]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int reg_merge[64];

  //expected-error@+3{{'max_replicates' and 'fpga_register' attributes are not compatible}}
  [[intel::fpga_register]]
  //expected-note@-1 {{conflicting attribute is here}}
  [[intel::max_replicates(2)]] unsigned int reg_maxrepl[64];

  //expected-error@+3{{'simple_dual_port' and 'fpga_register' attributes are not compatible}}
  [[intel::fpga_register]]
  //expected-note@-1 {{conflicting attribute is here}}
  [[intel::simple_dual_port]] unsigned int reg_dualport[64];

  //expected-error@+3{{'force_pow2_depth' and 'fpga_register' attributes are not compatible}}
  [[intel::fpga_register]]
  //expected-note@-1 {{conflicting attribute is here}}
  [[intel::force_pow2_depth(0)]] unsigned int reg_force_p2d[64];

  //expected-error@+3{{'fpga_register' and 'force_pow2_depth' attributes are not compatible}}
  [[intel::force_pow2_depth(1)]]
  //expected-note@-1 {{conflicting attribute is here}}
  [[intel::fpga_register]] unsigned int force_p2d_reg[64];

  // **memory
  //expected-error@+2{{attributes are not compatible}}
  [[intel::fpga_memory]]
  [[intel::fpga_register]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int mem_reg[64];

  // Check to see if there's a duplicate attribute with same Default values
  // already applied to the declaration.
  // No diagnostic is emitted because the arguments match.
  [[intel::fpga_memory]]
  [[intel::fpga_memory]] unsigned int mem_mem[64]; // OK

  //expected-warning@+1 {{unknown attribute 'memory' ignored}}
  [[intelfpga::memory]] unsigned int memory_var[64];

  // Check to see if there's a duplicate attribute with different values
  // already applied to the declaration.
  // Diagnostic is emitted because the arguments mismatch.
  //expected-warning@+2{{attribute 'fpga_memory' is already applied with different arguments}}
  [[intel::fpga_memory("MLAB")]] // expected-note {{previous attribute is here}}
  [[intel::fpga_memory("BLOCK_RAM")]] unsigned int mem_block_ram[32];

  // Check to see if there's a duplicate attribute with same values
  // already applied to the declaration.
  // No diagnostic is emitted because the arguments match.
  [[intel::fpga_memory("MLAB")]]
  [[intel::fpga_memory("MLAB")]] unsigned int mem_mlab[32]; // OK

  [[intel::fpga_memory("BLOCK_RAM")]]
  [[intel::fpga_memory("BLOCK_RAM")]] unsigned int mem_block[32]; // OK

  // Check to see if there's a duplicate attribute with different values
  // already applied to the declaration.
  // Diagnostic is emitted because the arguments mismatch.
  //expected-warning@+2{{attribute 'fpga_memory' is already applied with different arguments}}
  [[intel::fpga_memory]] // expected-note {{previous attribute is here}}
  [[intel::fpga_memory("MLAB")]] unsigned int mem_mlabs[64];

  // Check to see if there's a duplicate attribute with different values
  // already applied to the declaration.
  // Diagnostic is emitted because the arguments mismatch.
  //expected-warning@+2{{attribute 'fpga_memory' is already applied with different arguments}}
  [[intel::fpga_memory("BLOCK_RAM")]] // expected-note {{previous attribute is here}}
  [[intel::fpga_memory]] unsigned int mem_mlabs_block_ram[64];

  // **bankwidth
  //expected-warning@+1 {{unknown attribute 'bankwidth' ignored}}
  [[intelfpga::bankwidth(4)]] unsigned int bankwidth_var[32];

  // Checking of incompatible attributes.
  //expected-error@+2{{attributes are not compatible}}
  [[intel::bankwidth(16)]]
  [[intel::fpga_register]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int bankwidth_reg[64];

  // **max_replicates
  //expected-warning@+1 {{unknown attribute 'max_replicates' ignored}}
  [[intelfpga::max_replicates(2)]] unsigned int max_replicates_var[64];

  // Checking of different argument values.
  //expected-warning@+2{{attribute 'max_replicates' is already applied with different arguments}}
  [[intel::max_replicates(8)]] //expected-note{{previous attribute is here}}
  [[intel::max_replicates(16)]] unsigned int max_repl[64];

  //expected-error@+1{{'max_replicates' attribute requires a positive integral compile time constant expression}}
  [[intel::max_replicates(0)]] unsigned int maxrepl_zero[64];
  //expected-error@+1{{'max_replicates' attribute requires a positive integral compile time constant expression}}
  [[intel::max_replicates(-1)]] unsigned int maxrepl_negative[64];

  // Checking of incompatible attributes.
  //expected-error@+3{{'max_replicates' and 'fpga_register' attributes are not compatible}}
  [[intel::fpga_register]]
  //expected-note@-1 {{conflicting attribute is here}}
  [[intel::max_replicates(2)]] unsigned int maxrepl_reg[64];

  // **simple_dual_port
  //expected-error@+1{{'simple_dual_port' attribute takes no arguments}}
  [[intel::simple_dual_port(0)]] unsigned int sdp[64];

  //expected-warning@+1 {{unknown attribute 'simple_dual_port' ignored}}
  [[intelfpga::simple_dual_port]] unsigned int dual_port_var[64];

  //expected-note@+1 {{conflicting attribute is here}}
  [[intel::fpga_register]]
  //expected-error@+1{{'simple_dual_port' and 'fpga_register' attributes are not compatible}}
  [[intel::simple_dual_port]] unsigned int sdp_reg[64];

  //expected-warning@+2{{attribute 'simple_dual_port' is already applied}}
  [[intel::simple_dual_port]] //expected-note{{previous attribute is here}}
  [[intel::simple_dual_port]] unsigned int dual_port_var_dup[64];

  // Checking of different argument values.
  //expected-warning@+2{{attribute 'bankwidth' is already applied}}
  [[intel::bankwidth(8)]] // expected-note {{previous attribute is here}}
  [[intel::bankwidth(16)]] unsigned int bw_bw[64];

  //expected-error@+1{{must be a constant power of two greater than zero}}
  [[intel::bankwidth(3)]] unsigned int bw_invalid_value[64];

  //expected-error@+1{{requires a positive integral compile time constant expression}}
  [[intel::bankwidth(-4)]] unsigned int bw_negative[64];

  int i_bankwidth = 32; // expected-note {{declared here}}
  //expected-error@+1{{is not an integral constant expression}}
  [[intel::bankwidth(i_bankwidth)]]
  //expected-note@-1{{read of non-const variable 'i_bankwidth' is not allowed in a constant expression}}
  unsigned int bw_non_const[64];

  //expected-error@+1{{'bankwidth' attribute takes one argument}}
  [[intel::bankwidth(4, 8)]] unsigned int bw_two_args[64];

  //expected-error@+1{{requires a positive integral compile time constant expression}}
  [[intel::bankwidth(0)]] unsigned int bw_zero[64];

  // **private_copies
  //expected-warning@+1 {{unknown attribute 'private_copies' ignored}}
  [[intelfpga::private_copies(8)]] unsigned int private_copies_var[64];

  // Checking of incompatible attributes.
  //expected-error@+2{{attributes are not compatible}}
  [[intel::private_copies(16)]]
  [[intel::fpga_register]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int pc_reg[64];

  // Checking of different argument values.
  //expected-warning@+2{{attribute 'private_copies' is already applied with different arguments}}
  [[intel::private_copies(8)]] //expected-note{{previous attribute is here}}
  [[intel::private_copies(16)]] unsigned int pc_pc[64];

  //expected-error@+1{{'private_copies' attribute requires a non-negative integral compile time constant expression}}
  [[intel::private_copies(-4)]] unsigned int pc_negative[64];

  int i_private_copies = 32; // expected-note {{declared here}}
  //expected-error@+1{{expression is not an integral constant expression}}
  [[intel::private_copies(i_private_copies)]]
  //expected-note@-1{{read of non-const variable 'i_private_copies' is not allowed in a constant expression}}
  unsigned int pc_nonconst[64];

  //expected-error@+1{{'private_copies' attribute takes one argument}}
  [[intel::private_copies(4, 8)]] unsigned int pc_two_arg[64];

  // **numbanks
  //expected-error@+2{{attributes are not compatible}}
  [[intel::numbanks(16)]]
  [[intel::fpga_register]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int nb_reg[64];

  // Checking of different argument values.
  //expected-warning@+2{{attribute 'numbanks' is already applied}}
  [[intel::numbanks(8)]] // expected-note {{previous attribute is here}}
  [[intel::numbanks(16)]] unsigned int nb_nb[64];

  //expected-error@+1{{must be a constant power of two greater than zero}}
  [[intel::numbanks(15)]] unsigned int nb_invalid_arg[64];

  //expected-error@+1{{requires a positive integral compile time constant expression}}
  [[intel::numbanks(-4)]] unsigned int nb_negative[64];

  int i_numbanks = 32; // expected-note {{declared here}}
  //expected-error@+1{{is not an integral constant expression}}
  [[intel::numbanks(i_numbanks)]]
  //expected-note@-1{{read of non-const variable 'i_numbanks' is not allowed in a constant expression}}
  unsigned int nb_nonconst[64];

  //expected-error@+1{{'numbanks' attribute takes one argument}}
  [[intel::numbanks(4, 8)]] unsigned int nb_two_args[64];

  //expected-error@+1{{requires a positive integral compile time constant expression}}
  [[intel::numbanks(0)]] unsigned int nb_zero[64];

  //expected-warning@+1 {{unknown attribute 'numbanks' ignored}}
  [[intelfpga::numbanks(8)]] unsigned int numbanks_var[32];

  // **merge
  //expected-warning@+1 {{unknown attribute 'merge' ignored}}
  [[intelfpga::merge("mrg1", "depth")]] unsigned int merge_depth_var[64];

  //expected-error@+2{{attributes are not compatible}}
  [[intel::merge("mrg1", "depth")]]
  [[intel::fpga_register]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int mrg_reg[4];

  //expected-error@+1{{expected string literal as argument of 'merge' attribute}}
  [[intel::merge(3, 9.0f)]] unsigned int mrg_float[4];

  //expected-error@+1{{attribute requires exactly 2 arguments}}
  [[intel::merge("mrg2")]] unsigned int mrg_one_arg[4];

  //expected-error@+1{{attribute requires exactly 2 arguments}}
  [[intel::merge("mrg3", "depth", "oops")]] unsigned int mrg_three_arg[4];

  //expected-error@+1{{merge direction must be 'depth' or 'width'}}
  [[intel::merge("mrg4", "depths")]] unsigned int mrg_invalid_arg[4];

  // Checking of different argument values.
  //expected-warning@+2{{attribute 'merge' is already applied with different arguments}}
  [[intel::merge("mrg4", "depth")]] // expected-note {{previous attribute is here}}
  [[intel::merge("mrg5", "width")]] unsigned int mrg_mrg[4];

  //expected-warning@+2{{attribute 'merge' is already applied with different arguments}}
  [[intel::merge("mrg6", "depth")]] // expected-note {{previous attribute is here}}
  [[intel::merge("mrg6", "width")]] unsigned int mrg_mrg1[4];

  //expected-warning@+2{{attribute 'merge' is already applied with different arguments}}
  [[intel::merge("mrg7", "width")]] // expected-note {{previous attribute is here}}
  [[intel::merge("mrg8", "width")]] unsigned int mrg_mrg2[4];

  // Checking of duplicate argument values.
  // No diagnostic is emitted because the arguments match.
  [[intel::merge("mrg9", "depth")]]
  [[intel::merge("mrg9", "depth")]] unsigned int mrg_mrg3[4]; // OK

  // **bank_bits
  //expected-error@+2 1{{'fpga_register' and 'bank_bits' attributes are not compatible}}
  [[intel::bank_bits(2, 3)]]
  [[intel::fpga_register]]
  //expected-note@-2 1{{conflicting attribute is here}}
  unsigned int bb_reg[4];

  // Checking of different argument values.
  //expected-warning@+2{{attribute 'bank_bits' is already applied}}
  [[intel::bank_bits(42, 43)]]
  [[intel::bank_bits(1, 2)]] unsigned int bb_bb[4];

  //expected-error@+1{{the number of bank_bits must be equal to ceil(log2(numbanks))}}
  [[intel::numbanks(8), intel::bank_bits(3, 4)]] unsigned int bb_numbanks[4];

  //expected-error@+1{{bank_bits must be consecutive}}
  [[intel::bank_bits(3, 3, 4), intel::bankwidth(4)]] unsigned int bb_noncons[4];

  //expected-error@+1{{bank_bits must be consecutive}}
  [[intel::bank_bits(1, 3, 4), intel::bankwidth(4)]] unsigned int bb_noncons1[4];

  //expected-error@+1{{attribute takes at least 1 argument}}
  [[intel::bank_bits]] unsigned int bb_no_arg[4];

  //expected-error@+1{{'bank_bits' attribute requires a non-negative integral compile time constant expression}}
  [[intel::bank_bits(-1)]] unsigned int bb_negative_arg[4];

  //expected-warning@+1 {{unknown attribute 'bank_bits' ignored}}
  [[intelfpga::bank_bits(2, 3, 4, 5)]] unsigned int bankbits_var[64];

  // **force_pow2_depth
  //expected-warning@+1 {{unknown attribute 'force_pow2_depth' ignored}}
  [[intelfpga::force_pow2_depth(0)]] unsigned int arr_force_p2d_0_var[64];

  //expected-error@+1{{'force_pow2_depth' attribute requires integer constant value 0 or 1}}
  [[intel::force_pow2_depth(-1)]] unsigned int force_p2d_below_min[64];
  //expected-error@+1{{'force_pow2_depth' attribute requires integer constant value 0 or 1}}
  [[intel::force_pow2_depth(2)]] unsigned int force_p2d_above_max[64];

  //expected-error@+1{{'force_pow2_depth' attribute takes one argument}}
  [[intel::force_pow2_depth]] unsigned int force_p2d_no_args[64];
  //expected-error@+1{{'force_pow2_depth' attribute takes one argument}}
  [[intel::force_pow2_depth(0, 1)]] unsigned int force_p2d_2_args[64];

  // Checking of different argument values.
  //expected-note@+2{{previous attribute is here}}
  //expected-warning@+1{{attribute 'force_pow2_depth' is already applied with different arguments}}
  [[intel::force_pow2_depth(1), intel::force_pow2_depth(0)]] unsigned int force_p2d_dup[64];
}

void check_gnu_style() {
  // GNU style
  //expected-warning@+1{{unknown attribute 'numbanks' ignored}}
  int __attribute__((numbanks(4))) numbanks;

  //expected-warning@+1{{unknown attribute 'memory' ignored}}
  unsigned int __attribute__((memory("MLAB"))) memory;

  //expected-warning@+1{{unknown attribute 'bankwidth' ignored}}
  int __attribute__((bankwidth(8))) bankwidth;

  //expected-warning@+1{{unknown attribute 'register' ignored}}
  int __attribute__((register)) reg;

  //expected-warning@+1{{unknown attribute '__singlepump__' ignored}}
  unsigned int __attribute__((__singlepump__)) singlepump;

  //expected-warning@+1{{unknown attribute '__doublepump__' ignored}}
  unsigned int __attribute__((__doublepump__)) doublepump;

  //expected-warning@+1{{unknown attribute '__private_copies__' ignored}}
  int __attribute__((__private_copies__(4))) private_copies;

  //expected-warning@+1{{unknown attribute '__merge__' ignored}}
  int __attribute__((__merge__("mrg1","depth"))) merge;

  //expected-warning@+1{{unknown attribute 'max_replicates' ignored}}
  int __attribute__((max_replicates(2))) max_repl;

  //expected-warning@+1{{unknown attribute 'simple_dual_port' ignored}}
  int __attribute__((simple_dual_port)) dualport;

  //expected-warning@+1{{unknown attribute 'bank_bits' ignored}}
  int __attribute__((bank_bits(4))) bankbits;

  //expected-warning@+1{{unknown attribute 'force_pow2_depth' ignored}}
  int __attribute__((force_pow2_depth(0))) force_p2d;
}

//expected-error@+1{{attribute only applies to local non-const variables and non-static data members}}
[[intel::private_copies(8)]]
__attribute__((opencl_constant)) unsigned int const_var[64] = {1, 2, 3};

void attr_on_const_error()
{
  //expected-error@+1{{attribute only applies to local non-const variables and non-static data members}}
  [[intel::private_copies(8)]] const int const_var[64] = {0, 1};
}

//expected-error@+1{{attribute only applies to local non-const variables and non-static data members}}
void attr_on_func_arg([[intel::private_copies(8)]] int pc) {}

//expected-error@+1{{attribute only applies to constant variables, local variables, static variables, agent memory arguments, and non-static data members}}
[[intel::force_pow2_depth(0)]]
__attribute__((opencl_global)) unsigned int ocl_glob_force_p2d[64] = {1, 2, 3};

//expected-no-error@+1
void force_p2d_attr_on_func_arg([[intel::force_pow2_depth(0)]] int pc) {}

template <int A, int B, int C, int D, int E>
void check_template_parameters() {
  // OK	
  [[intel::force_pow2_depth(E)]] const int const_force_p2d_templ[64] = {0, 1};

  //expected-error@+1{{'numbanks' attribute takes one argument}}
  [[intel::numbanks(A, B)]] int numbanks_negative;

  //expected-error@+1{{'max_replicates' attribute requires a positive integral compile time constant expression}}
  [[intel::max_replicates(D)]]
  [[intel::max_replicates(C)]]
  unsigned int max_replicates_duplicate;

  // Test that checks template instantiations for different arg values.
  [[intel::max_replicates(4)]] // expected-note {{previous attribute is here}}
  // expected-warning@+1 {{attribute 'max_replicates' is already applied with different arguments}}
  [[intel::max_replicates(C)]] unsigned int max_repl_duplicate[64];

  // Test that checks template instantiations for different arg values.
  [[intel::private_copies(4)]] // expected-note {{previous attribute is here}}
  // expected-warning@+1 {{attribute 'private_copies' is already applied with different arguments}}
  [[intel::private_copies(C)]] unsigned int var_private_copies;

  //expected-error@+3{{'max_replicates' and 'fpga_register' attributes are not compatible}}
  [[intel::fpga_register]]
  //expected-note@-1 {{conflicting attribute is here}}
  [[intel::max_replicates(C)]] unsigned int maxrepl_reg;

  // Test that checks template instantiations for different arg values.
  [[intel::numbanks(4)]] // expected-note {{previous attribute is here}}
  // expected-warning@+1 {{attribute 'numbanks' is already applied with different arguments}}
  [[intel::numbanks(C)]] unsigned int num_banks[64];

  //expected-error@+3{{'numbanks' and 'fpga_register' attributes are not compatible}}
  [[intel::fpga_register]]
  //expected-note@-1 {{conflicting attribute is here}}
  [[intel::numbanks(C)]] unsigned int max_numbanks_reg;

  // Test that checks template instantiations for different arg values.
  [[intel::bankwidth(4)]] // expected-note {{previous attribute is here}}
  // expected-warning@+1 {{attribute 'bankwidth' is already applied with different arguments}}
  [[intel::bankwidth(C)]] unsigned int bank_width[64];

  //expected-error@+3{{'bankwidth' and 'fpga_register' attributes are not compatible}}
  [[intel::fpga_register]]
  //expected-note@-1 {{conflicting attribute is here}}
  [[intel::bankwidth(C)]] unsigned int max_bankwidth_reg;

  //expected-error@+1{{'force_pow2_depth' attribute requires integer constant value 0 or 1}}
  [[intel::force_pow2_depth(A)]] unsigned int force_p2d_below_min[64];

  //expected-error@+1{{'force_pow2_depth' attribute takes one argument}}
  [[intel::force_pow2_depth(E, E)]] unsigned int force_p2d_2_args[64];

  //expected-error@+3{{'force_pow2_depth' and 'fpga_register' attributes are not compatible}}
  [[intel::fpga_register]]
  //expected-note@-1{{conflicting attribute is here}}
  [[intel::force_pow2_depth(E)]] unsigned int reg_force_p2d[64];

  // Test that checks template instantiations for different arg values.
  //expected-note@+2{{previous attribute is here}}
  //expected-warning@+1{{attribute 'force_pow2_depth' is already applied with different arguments}}
  [[intel::force_pow2_depth(E), intel::force_pow2_depth(0)]] unsigned int force_p2d_dup[64];
}

int main() {
  deviceQueue.submit([&](sycl::handler &h) {
    h.single_task<class kernel_function>([]() {
      diagnostics();
      check_gnu_style();
      //expected-note@+1{{in instantiation of function template specialization}}
      check_template_parameters<2, 4, 8, -1, 1>();
    });
  });

  return 0;
}

// Merging of different arg values.
//expected-warning@+2{{attribute 'max_replicates' is already applied with different arguments}}
[[intel::max_replicates(12)]] extern const int var_max_replicates_1;
[[intel::max_replicates(14)]] const int var_max_replicates_1 = 0;
//expected-note@-2{{previous attribute is here}}

// Merging of incompatible attributes.
//expected-error@+3{{'fpga_register' and 'max_replicates' attributes are not compatible}}
//expected-note@+1{{conflicting attribute is here}}
[[intel::max_replicates(12)]] extern const int var_max_replicates_2;
[[intel::fpga_register]] const int var_max_replicates_2 =0;

// Merging of different arg values.
//expected-warning@+2{{attribute 'force_pow2_depth' is already applied with different arguments}}
[[intel::force_pow2_depth(1)]] extern const int var_force_pow2_depth_1;
[[intel::force_pow2_depth(0)]] const int var_force_pow2_depth_1 = 0;
//expected-note@-2{{previous attribute is here}}

// Merging of incompatible attributes.
//expected-error@+3{{'fpga_register' and 'force_pow2_depth' attributes are not compatible}}
//expected-note@+1{{conflicting attribute is here}}
[[intel::force_pow2_depth(1)]] extern const int var_force_pow2_depth_2;
[[intel::fpga_register]] const int var_force_pow2_depth_2 =0;

// Merging of different arg values.
//expected-warning@+2{{attribute 'numbanks' is already applied with different arguments}}
[[intel::numbanks(8)]] extern const int var_numbanks_1;
[[intel::numbanks(16)]] const int var_numbanks_1 = 0;
//expected-note@-2{{previous attribute is here}}

// Merging of incompatible attributes.
//expected-error@+3{{'fpga_register' and 'numbanks' attributes are not compatible}}
//expected-note@+1{{conflicting attribute is here}}
[[intel::numbanks(16)]] extern const int var_numbanks_2;
[[intel::fpga_register]] const int var_numbanks_2 =0;

// Merging of different arg values.
//expected-warning@+2{{attribute 'bankwidth' is already applied with different arguments}}
[[intel::bankwidth(8)]] extern const int var_bankwidth_1;
[[intel::bankwidth(16)]] const int var_bankwidth_1 = 0;
//expected-note@-2{{previous attribute is here}}

// Merging of incompatible attributes.
//expected-error@+3{{'fpga_register' and 'bankwidth' attributes are not compatible}}
//expected-note@+1{{conflicting attribute is here}}
[[intel::bankwidth(8)]] extern const int var_bankwidth_2;
[[intel::fpga_register]] const int var_bankwidth_2 =0;
