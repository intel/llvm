// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -sycl-std=2020 -Wno-return-type -fcxx-exceptions -fsyntax-only -ast-dump -verify -pedantic %s | FileCheck %s

#include "sycl.hpp"

sycl::queue deviceQueue;

//CHECK: FunctionDecl{{.*}}check_ast
void check_ast()
{
  //CHECK: VarDecl{{.*}}doublepump
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGADoublePumpAttr
  [[intel::doublepump]] unsigned int doublepump[64];

  //CHECK: VarDecl{{.*}}memory
  //CHECK: IntelFPGAMemoryAttr
  [[intel::fpga_memory]] unsigned int memory[64];

  //CHECK: VarDecl{{.*}}memory_mlab
  //CHECK: IntelFPGAMemoryAttr{{.*}}MLAB
  [[intel::fpga_memory("MLAB")]] unsigned int memory_mlab[64];

  //CHECK: VarDecl{{.*}}mem_blockram
  //CHECK: IntelFPGAMemoryAttr{{.*}}BlockRAM
  [[intel::fpga_memory("BLOCK_RAM")]] unsigned int mem_blockram[32];

  //CHECK: VarDecl{{.*}}reg
  //CHECK: IntelFPGARegisterAttr
  [[intel::fpga_register]] unsigned int reg[64];

  //CHECK: VarDecl{{.*}}singlepump
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGASinglePumpAttr
  [[intel::singlepump]] unsigned int singlepump[64];

  //CHECK: VarDecl{{.*}}bankwidth
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGABankWidthAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}4
  //CHECK-NEXT: IntegerLiteral{{.*}}4{{$}}
  [[intel::bankwidth(4)]] unsigned int bankwidth[32];

  //CHECK: VarDecl{{.*}}numbanks
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGANumBanksAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}8
  //CHECK-NEXT: IntegerLiteral{{.*}}8{{$}}
  [[intel::numbanks(8)]] unsigned int numbanks[32];

  //CHECK: VarDecl{{.*}}private_copies
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGAPrivateCopiesAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}8
  //CHECK-NEXT: IntegerLiteral{{.*}}8{{$}}
  [[intel::private_copies(8)]] unsigned int private_copies[64];

  //CHECK: VarDecl{{.*}}merge_depth
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGAMergeAttr{{.*}}"mrg1" "depth"{{$}}
  [[intel::merge("mrg1", "depth")]] unsigned int merge_depth[64];

  //CHECK: VarDecl{{.*}}merge_width
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGAMergeAttr{{.*}}"mrg2" "width"{{$}}
  [[intel::merge("mrg2", "width")]] unsigned int merge_width[64];

  //CHECK: VarDecl{{.*}}bankbits
  //CHECK: IntelFPGANumBanksAttr{{.*}}Implicit{{$}}
  //CHECK-NEXT: IntegerLiteral{{.*}}16{{$}}
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGABankBitsAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}2
  //CHECK-NEXT: IntegerLiteral{{.*}}2{{$}}
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}3
  //CHECK-NEXT: IntegerLiteral{{.*}}3{{$}}
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}4
  //CHECK-NEXT: IntegerLiteral{{.*}}4{{$}}
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}5
  //CHECK-NEXT: IntegerLiteral{{.*}}5{{$}}
  [[intel::bank_bits(2, 3, 4, 5)]] unsigned int bankbits[64];

  //CHECK: VarDecl{{.*}}bank_bits_width
  //CHECK-NEXT: IntelFPGANumBanksAttr{{.*}}Implicit{{$}}
  //CHECK-NEXT: IntegerLiteral{{.*}}4{{$}}
  //CHECK-NEXT: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK-NEXT: IntelFPGABankBitsAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}2
  //CHECK-NEXT: IntegerLiteral{{.*}}2{{$}}
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}3
  //CHECK-NEXT: IntegerLiteral{{.*}}3{{$}}
  //CHECK-NEXT: IntelFPGABankWidthAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}16
  //CHECK-NEXT: IntegerLiteral{{.*}}16{{$}}
  [[intel::bank_bits(2, 3), intel::bankwidth(16)]] unsigned int bank_bits_width[64];

  //CHECK: VarDecl{{.*}}doublepump_mlab
  //CHECK: IntelFPGADoublePumpAttr
  //CHECK: IntelFPGAMemoryAttr{{.*}}MLAB{{$}}
  [[intel::doublepump]]
  [[intel::fpga_memory("MLAB")]] unsigned int doublepump_mlab[64];

  // Add implicit memory attribute.
  //CHECK: VarDecl{{.*}}max_replicates
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGAMaxReplicatesAttr
  //CHECK: ConstantExpr
  //CHECK-NEXT: value:{{.*}}2
  //CHECK: IntegerLiteral{{.*}}2{{$}}
  [[intel::max_replicates(2)]] unsigned int max_replicates[64];

  //CHECK: VarDecl{{.*}}dual_port
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGASimpleDualPortAttr
  [[intel::simple_dual_port]] unsigned int dual_port[64];

  //CHECK: VarDecl{{.*}}arr_force_p2d_0
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGAForcePow2DepthAttr
  //CHECK: ConstantExpr
  //CHECK-NEXT: value:{{.*}}0
  //CHECK: IntegerLiteral{{.*}}0{{$}}
  [[intel::force_pow2_depth(0)]] unsigned int arr_force_p2d_0[64];

  //CHECK: VarDecl{{.*}}arr_force_p2d_1
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGAForcePow2DepthAttr
  //CHECK: ConstantExpr
  //CHECK-NEXT: value:{{.*}}1
  //CHECK: IntegerLiteral{{.*}}1{{$}}
  [[intel::force_pow2_depth(1)]] unsigned int arr_force_p2d_1[64];
  [[intel::fpga_register]] int var_reg;
  [[intel::numbanks(4), intel::bankwidth(16), intel::singlepump]] int var_singlepump;
  [[intel::numbanks(4), intel::bankwidth(16), intel::doublepump]] int var_doublepump;
  [[intel::numbanks(4), intel::bankwidth(16)]] int var_numbanks_bankwidth;
  [[intel::bank_bits(2, 3), intel::bankwidth(16)]] int var_bank_bits_width;
  [[intel::max_replicates(2)]] int var_max_repl;
  [[intel::simple_dual_port]] int var_dual_port;
  [[intel::force_pow2_depth(1)]] int var_force_p2d;
  [[intel::force_pow2_depth(1)]] const int const_force_p2d[64] = {0, 1};

  // Check duplicate argument values with implicit memory attribute.
  //CHECK: VarDecl{{.*}}var_max_replicates
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGAMaxReplicatesAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}12
  //CHECK-NEXT: IntegerLiteral{{.*}}12{{$}}
  [[intel::max_replicates(12)]]
  [[intel::max_replicates(12)]] int var_max_replicates; // OK

  // Check duplicate argument values.
  //CHECK: VarDecl{{.*}}var_private_copies
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGAPrivateCopiesAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}12
  //CHECK-NEXT: IntegerLiteral{{.*}}12{{$}}
  [[intel::private_copies(12)]]
  [[intel::private_copies(12)]] int var_private_copies; // OK

  // Checking of duplicate argument values.
  //CHECK: VarDecl{{.*}}var_forcep2d
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGAForcePow2DepthAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}1
  //CHECK-NEXT: IntegerLiteral{{.*}}1{{$}}
  [[intel::force_pow2_depth(1)]]
  [[intel::force_pow2_depth(1)]] int var_forcep2d; // OK
}

//CHECK: FunctionDecl{{.*}}diagnostics
void diagnostics()
{
  // **doublepump
  //CHECK: VarDecl{{.*}}doublepump
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGADoublePumpAttr
  //expected-warning@+2 {{attribute 'intelfpga::doublepump' is deprecated}}
  //expected-note@+1 {{did you mean to use 'intel::doublepump' instead?}}
  [[intelfpga::doublepump]] unsigned int doublepump[64];

  //expected-error@+2{{attributes are not compatible}}
  [[intel::doublepump]]
  [[intel::singlepump]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int dpump_spump[64];

  //expected-warning@+2{{attribute 'doublepump' is already applied}}
  [[intel::doublepump]]
  [[intel::doublepump]] unsigned int dpump[64];

  //expected-error@+2{{attributes are not compatible}}
  [[intel::doublepump]]
  [[intel::fpga_register]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int dpump_reg[64];

  // **singlepump
  //CHECK: VarDecl{{.*}}singlepump
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGASinglePumpAttr
  //expected-warning@+2 {{attribute 'intelfpga::singlepump' is deprecated}}
  //expected-note@+1 {{did you mean to use 'intel::singlepump' instead?}}
  [[intelfpga::singlepump]] unsigned int singlepump[64];

  //expected-error@+1{{attributes are not compatible}}
  [[intel::singlepump, intel::doublepump]]
  //expected-note@-1 {{conflicting attribute is here}}
  unsigned int spump_dpump[64];

  //expected-warning@+2{{attribute 'singlepump' is already applied}}
  [[intel::singlepump]]
  [[intel::singlepump]] unsigned int spump[64];

  //expected-error@+2{{attributes are not compatible}}
  [[intel::singlepump]]
  [[intel::fpga_register]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int spump_reg[64];

  // **fpga_register
  //CHECK: VarDecl{{.*}}reg
  //CHECK: IntelFPGARegisterAttr
  //expected-warning@+2 {{attribute 'intelfpga::register' is deprecated}}
  //expected-note@+1 {{did you mean to use 'intel::fpga_register' instead?}}
  [[intelfpga::register]] unsigned int reg[64];

  //expected-warning@+1{{attribute 'fpga_register' is already applied}}
  [[intel::fpga_register]] [[intel::fpga_register]] unsigned int reg_reg[64];

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

  //expected-error@+2{{attributes are not compatible}}
  [[intel::fpga_register]]
  [[intel::numbanks(8)]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int reg_numbanks[64];

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
  //CHECK: VarDecl{{.*}}memory
  //CHECK: IntelFPGAMemoryAttr
  //expected-warning@+2 {{attribute 'intelfpga::memory' is deprecated}}
  //expected-note@+1 {{did you mean to use 'intel::fpga_memory' instead?}}
  [[intelfpga::memory]] unsigned int memory[64];

  //expected-error@+2{{attributes are not compatible}}
  [[intel::fpga_memory]]
  [[intel::fpga_register]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int mem_reg[64];

  //expected-warning@+1{{attribute 'fpga_memory' is already applied}}
  [[intel::fpga_memory]] [[intel::fpga_memory]] unsigned int mem_mem[64];

  // bankwidth
  //CHECK: VarDecl{{.*}}bankwidth
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGABankWidthAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}4
  //CHECK-NEXT: IntegerLiteral{{.*}}4{{$}}
  //expected-warning@+2 {{attribute 'intelfpga::bankwidth' is deprecated}}
  //expected-note@+1 {{did you mean to use 'intel::bankwidth' instead?}}
  [[intelfpga::bankwidth(4)]] unsigned int bankwidth[32];

  //expected-error@+2{{attributes are not compatible}}
  [[intel::bankwidth(16)]]
  [[intel::fpga_register]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int bankwidth_reg[64];

  // **max_replicates
  // Add implicit memory attribute.
  //CHECK: VarDecl{{.*}}max_replicates
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGAMaxReplicatesAttr
  //CHECK: ConstantExpr
  //CHECK-NEXT: value:{{.*}}2
  //CHECK: IntegerLiteral{{.*}}2{{$}}
  //expected-warning@+2 {{attribute 'intelfpga::max_replicates' is deprecated}}
  //expected-note@+1 {{did you mean to use 'intel::max_replicates' instead?}}
  [[intelfpga::max_replicates(2)]] unsigned int max_replicates[64];

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

  //CHECK: VarDecl{{.*}}dual_port
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGASimpleDualPortAttr
  //expected-warning@+2 {{attribute 'intelfpga::simple_dual_port' is deprecated}}
  //expected-note@+1 {{did you mean to use 'intel::simple_dual_port' instead?}}
  [[intelfpga::simple_dual_port]] unsigned int dual_port[64];

  //expected-note@+1 {{conflicting attribute is here}}
  [[intel::fpga_register]]
  //expected-error@+1{{'simple_dual_port' and 'fpga_register' attributes are not compatible}}
  [[intel::simple_dual_port]] unsigned int sdp_reg[64];

  //CHECK: VarDecl{{.*}}bw_bw
  //CHECK: IntelFPGABankWidthAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}8
  //CHECK-NEXT: IntegerLiteral{{.*}}8{{$}}
  //CHECK: IntelFPGABankWidthAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}16
  //CHECK-NEXT: IntegerLiteral{{.*}}16{{$}}
  //expected-warning@+2{{attribute 'bankwidth' is already applied}}
  [[intel::bankwidth(8)]]
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

  // private_copies_
  //CHECK: VarDecl{{.*}}private_copies
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGAPrivateCopiesAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}8
  //CHECK-NEXT: IntegerLiteral{{.*}}8{{$}}
  //expected-warning@+2 {{attribute 'intelfpga::private_copies' is deprecated}}
  //expected-note@+1 {{did you mean to use 'intel::private_copies' instead?}}
  [[intelfpga::private_copies(8)]] unsigned int private_copies[64];

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

  // numbanks
  //CHECK: VarDecl{{.*}}numbanks
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGANumBanksAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}8
  //CHECK-NEXT: IntegerLiteral{{.*}}8{{$}}
  //expected-warning@+2 {{attribute 'intelfpga::numbanks' is deprecated}}
  //expected-note@+1 {{did you mean to use 'intel::numbanks' instead?}}
  [[intelfpga::numbanks(8)]] unsigned int numbanks[32];

  //expected-error@+2{{attributes are not compatible}}
  [[intel::numbanks(16)]]
  [[intel::fpga_register]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int nb_reg[64];

  //CHECK: VarDecl{{.*}}nb_nb
  //CHECK: IntelFPGANumBanksAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}8
  //CHECK-NEXT: IntegerLiteral{{.*}}8{{$}}
  //CHECK: IntelFPGANumBanksAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}16
  //CHECK-NEXT: IntegerLiteral{{.*}}16{{$}}
  //expected-warning@+2{{attribute 'numbanks' is already applied}}
  [[intel::numbanks(8)]]
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

  // merge
  //CHECK: VarDecl{{.*}}merge_depth
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGAMergeAttr{{.*}}"mrg1" "depth"{{$}}
  //expected-warning@+2 {{attribute 'intelfpga::merge' is deprecated}}
  //expected-note@+1 {{did you mean to use 'intel::merge' instead?}}
  [[intelfpga::merge("mrg1", "depth")]] unsigned int merge_depth[64];

  //expected-error@+2{{attributes are not compatible}}
  [[intel::merge("mrg1", "depth")]]
  [[intel::fpga_register]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int mrg_reg[4];

  //expected-error@+1{{attribute requires a string}}
  [[intel::merge(3, 9.0f)]] unsigned int mrg_float[4];

  //expected-error@+1{{attribute requires exactly 2 arguments}}
  [[intel::merge("mrg2")]] unsigned int mrg_one_arg[4];

  //expected-error@+1{{attribute requires exactly 2 arguments}}
  [[intel::merge("mrg3", "depth", "oops")]] unsigned int mrg_three_arg[4];

  //expected-error@+1{{merge direction must be 'depth' or 'width'}}
  [[intel::merge("mrg4", "depths")]] unsigned int mrg_invalid_arg[4];

  //Last one is applied and others ignored.
  //CHECK: VarDecl{{.*}}mrg_mrg
  //CHECK: IntelFPGAMergeAttr{{.*}}"mrg4" "depth"{{$}}
  //CHECK: IntelFPGAMergeAttr{{.*}}"mrg5" "width"{{$}}
  //expected-warning@+2{{attribute 'merge' is already applied}}
  [[intel::merge("mrg4", "depth")]]
  [[intel::merge("mrg5", "width")]] unsigned int mrg_mrg[4];

  // bank_bits
  //CHECK: VarDecl{{.*}}bankbits
  //CHECK: IntelFPGANumBanksAttr{{.*}}Implicit{{$}}
  //CHECK-NEXT: IntegerLiteral{{.*}}16{{$}}
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGABankBitsAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}2
  //CHECK-NEXT: IntegerLiteral{{.*}}2{{$}}
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}3
  //CHECK-NEXT: IntegerLiteral{{.*}}3{{$}}
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}4
  //CHECK-NEXT: IntegerLiteral{{.*}}4{{$}}
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}5
  //CHECK-NEXT: IntegerLiteral{{.*}}5{{$}}
  //expected-warning@+2 {{attribute 'intelfpga::bank_bits' is deprecated}}
  //expected-note@+1 {{did you mean to use 'intel::bank_bits' instead?}}
  [[intelfpga::bank_bits(2, 3, 4, 5)]] unsigned int bankbits[64];

  //expected-error@+2 1{{'fpga_register' and 'bank_bits' attributes are not compatible}}
  [[intel::bank_bits(2, 3)]]
  [[intel::fpga_register]]
  //expected-note@-2 1{{conflicting attribute is here}}
  unsigned int bb_reg[4];

  //CHECK: VarDecl{{.*}}bb_bb
  //CHECK: IntelFPGABankBitsAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}42
  //CHECK-NEXT: IntegerLiteral{{.*}}42{{$}}
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}43
  //CHECK-NEXT: IntegerLiteral{{.*}}43{{$}}
  //CHECK: IntelFPGABankBitsAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}1
  //CHECK-NEXT: IntegerLiteral{{.*}}1{{$}}
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}2
  //CHECK-NEXT: IntegerLiteral{{.*}}2{{$}}
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

  // force_pow2_depth
  //CHECK: VarDecl{{.*}}arr_force_p2d_0
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGAForcePow2DepthAttr
  //CHECK: ConstantExpr
  //CHECK-NEXT: value:{{.*}}0
  //CHECK: IntegerLiteral{{.*}}0{{$}}
  //expected-warning@+2 {{attribute 'intelfpga::force_pow2_depth' is deprecated}}
  //expected-note@+1 {{did you mean to use 'intel::force_pow2_depth' instead?}}
  [[intelfpga::force_pow2_depth(0)]] unsigned int arr_force_p2d_0[64];

  //expected-error@+1{{'force_pow2_depth' attribute requires integer constant between 0 and 1 inclusive}}
  [[intel::force_pow2_depth(-1)]] unsigned int force_p2d_below_min[64];
  //expected-error@+1{{'force_pow2_depth' attribute requires integer constant between 0 and 1 inclusive}}
  [[intel::force_pow2_depth(2)]] unsigned int force_p2d_above_max[64];

  //expected-error@+1{{'force_pow2_depth' attribute takes one argument}}
  [[intel::force_pow2_depth]] unsigned int force_p2d_no_args[64];
  //expected-error@+1{{'force_pow2_depth' attribute takes one argument}}
  [[intel::force_pow2_depth(0, 1)]] unsigned int force_p2d_2_args[64];

  // Checking of different argument values.
  //CHECK: VarDecl{{.*}}force_p2d_dup
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGAForcePow2DepthAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}1
  //CHECK-NEXT: IntegerLiteral{{.*}}1{{$}}
  //expected-note@+2{{previous attribute is here}}
  //expected-warning@+1{{attribute 'force_pow2_depth' is already applied with different arguments}}
  [[intel::force_pow2_depth(1), intel::force_pow2_depth(0)]] unsigned int force_p2d_dup[64];
}

//CHECK: FunctionDecl{{.*}}check_gnu_style
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

struct foo {
  //CHECK: FieldDecl{{.*}}doublepump
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGADoublePumpAttr
  [[intel::doublepump]] unsigned int doublepump[64];

  //CHECK: FieldDecl{{.*}}memory
  //CHECK: IntelFPGAMemoryAttr
  [[intel::fpga_memory]] unsigned int memory[64];

  //CHECK: FieldDecl{{.*}}memory_mlab
  //CHECK: IntelFPGAMemoryAttr{{.*}}MLAB{{$}}
  [[intel::fpga_memory("MLAB")]] unsigned int memory_mlab[64];

  //CHECK: FieldDecl{{.*}}mem_blockram
  //CHECK: IntelFPGAMemoryAttr{{.*}}BlockRAM{{$}}
  [[intel::fpga_memory("BLOCK_RAM")]] unsigned int mem_blockram[64];

  //CHECK: FieldDecl{{.*}}mem_blockram_doublepump
  //CHECK: IntelFPGAMemoryAttr{{.*}}BlockRAM{{$}}
  //CHECK: IntelFPGADoublePumpAttr
  [[intel::fpga_memory("BLOCK_RAM")]]
  [[intel::doublepump]] unsigned int mem_blockram_doublepump[64];

  //CHECK: FieldDecl{{.*}}reg
  //CHECK: IntelFPGARegisterAttr
  [[intel::fpga_register]] unsigned int reg[64];

  //CHECK: FieldDecl{{.*}}singlepump
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGASinglePumpAttr
  [[intel::singlepump]] unsigned int singlepump[64];

  //CHECK: FieldDecl{{.*}}bankwidth
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGABankWidthAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}4
  //CHECK-NEXT: IntegerLiteral{{.*}}4{{$}}
  [[intel::bankwidth(4)]] unsigned int bankwidth[64];

  //CHECK: FieldDecl{{.*}}numbanks
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGANumBanksAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}8
  //CHECK-NEXT: IntegerLiteral{{.*}}8{{$}}
  [[intel::numbanks(8)]] unsigned int numbanks[64];

  //CHECK: FieldDecl{{.*}}private_copies
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGAPrivateCopiesAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}4
  //CHECK-NEXT: IntegerLiteral{{.*}}4{{$}}
  [[intel::private_copies(4)]] unsigned int private_copies[64];

  //CHECK: FieldDecl{{.*}}merge_depth
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGAMergeAttr{{.*}}"mrg1" "depth"{{$}}
  [[intel::merge("mrg1", "depth")]] unsigned int merge_depth[64];

  //CHECK: FieldDecl{{.*}}merge_width
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGAMergeAttr{{.*}}"mrg2" "width"{{$}}
  [[intel::merge("mrg2", "width")]] unsigned int merge_width[64];

  //CHECK: FieldDecl{{.*}}bankbits
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGABankBitsAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}2
  //CHECK-NEXT: IntegerLiteral{{.*}}2{{$}}
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}3
  //CHECK-NEXT: IntegerLiteral{{.*}}3{{$}}
  [[intel::bank_bits(2, 3)]] unsigned int bankbits[64];

  //CHECK: FieldDecl{{.*}}force_p2d_field
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGAForcePow2DepthAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}1
  //CHECK-NEXT: IntegerLiteral{{.*}}1{{$}}
  [[intel::force_pow2_depth(1)]] unsigned int force_p2d_field[64];
};

//CHECK: FunctionDecl{{.*}}used check_template_parameters
template <int A, int B, int C, int D, int E>
void check_template_parameters() {
  //CHECK: VarDecl{{.*}}numbanks
  //CHECK-NEXT: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK-NEXT: IntelFPGANumBanksAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}8
  //CHECK-NEXT: SubstNonTypeTemplateParmExpr
  //CHECK-NEXT: NonTypeTemplateParmDecl
  //CHECK-NEXT: IntegerLiteral{{.*}}8{{$}}
  [[intel::numbanks(C)]] unsigned int numbanks;

  //CHECK: VarDecl{{.*}}private_copies
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGAPrivateCopiesAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}8
  //CHECK-NEXT: SubstNonTypeTemplateParmExpr
  //CHECK-NEXT: NonTypeTemplateParmDecl
  //CHECK-NEXT: IntegerLiteral{{.*}}8{{$}}
  [[intel::private_copies(C)]] unsigned int private_copies;

  //CHECK: VarDecl{{.*}}bank_bits_width
  //CHECK: IntelFPGANumBanksAttr{{.*}}Implicit{{$}}
  //CHECK-NEXT: IntegerLiteral{{.*}}4{{$}}
  //CHECK-NEXT: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK-NEXT: IntelFPGABankBitsAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}2
  //CHECK-NEXT: SubstNonTypeTemplateParmExpr
  //CHECK-NEXT: NonTypeTemplateParmDecl
  //CHECK-NEXT: IntegerLiteral{{.*}}2{{$}}
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}3
  //CHECK-NEXT: IntegerLiteral{{.*}}3{{$}}
  //CHECK: IntelFPGABankWidthAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}8
  //CHECK-NEXT: SubstNonTypeTemplateParmExpr
  //CHECK-NEXT: NonTypeTemplateParmDecl
  //CHECK-NEXT: IntegerLiteral{{.*}}8{{$}}
  [[intel::bank_bits(A, 3), intel::bankwidth(C)]] unsigned int bank_bits_width;

  // Add implicit memory attribute.
  //CHECK: VarDecl{{.*}}max_replicates
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGAMaxReplicatesAttr
  //CHECK: ConstantExpr
  //CHECK-NEXT: value:{{.*}}2
  //CHECK-NEXT: SubstNonTypeTemplateParmExpr
  //CHECK: IntegerLiteral{{.*}}2{{$}}
  [[intel::max_replicates(A)]] unsigned int max_replicates;

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

  //expected-error@+1{{'force_pow2_depth' attribute requires integer constant between 0 and 1 inclusive}}
  [[intel::force_pow2_depth(A)]] unsigned int force_p2d_below_min[64];

  //expected-error@+1{{'force_pow2_depth' attribute takes one argument}}
  [[intel::force_pow2_depth(E, E)]] unsigned int force_p2d_2_args[64];

  //expected-error@+3{{'force_pow2_depth' and 'fpga_register' attributes are not compatible}}
  [[intel::fpga_register]]
  //expected-note@-1{{conflicting attribute is here}}
  [[intel::force_pow2_depth(E)]] unsigned int reg_force_p2d[64];

  // Test that checks template instantiations for different arg values.
  //CHECK: VarDecl{{.*}}force_p2d_dup
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGAForcePow2DepthAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}1
  //CHECK-NEXT: SubstNonTypeTemplateParmExpr
  //CHECK-NEXT: NonTypeTemplateParmDecl
  //CHECK-NEXT: IntegerLiteral{{.*}}1{{$}}
  //expected-note@+2{{previous attribute is here}}
  //expected-warning@+1{{attribute 'force_pow2_depth' is already applied with different arguments}}
  [[intel::force_pow2_depth(E), intel::force_pow2_depth(0)]] unsigned int force_p2d_dup[64];
}

template <int A>
struct templ_st {
  //CHECK: FieldDecl{{.*}}templ_force_p2d_field
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGAForcePow2DepthAttr
  //CHECK: ConstantExpr
  //CHECK-NEXT: value:{{.*}}0
  //CHECK-NEXT: SubstNonTypeTemplateParmExpr
  //CHECK-NEXT: NonTypeTemplateParmDecl
  //CHECK-NEXT: IntegerLiteral{{.*}}0{{$}}
  [[intel::force_pow2_depth(A)]] unsigned int templ_force_p2d_field[64];
};

int main() {
  deviceQueue.submit([&](sycl::handler &h) {
    h.single_task<class kernel_function>([]() {
      check_ast();
      diagnostics();
      check_gnu_style();
      //expected-note@+1{{in instantiation of function template specialization}}
      check_template_parameters<2, 4, 8, -1, 1>();
      struct templ_st<0> ts {};
    });
  });

  return 0;
}
