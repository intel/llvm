// RUN: %clang_cc1 -fsycl -fsycl-is-device -Wno-return-type -fcxx-exceptions -fsyntax-only -ast-dump -Wno-sycl-2017-compat -verify -pedantic %s | FileCheck %s

//CHECK: FunctionDecl{{.*}}check_ast
void check_ast()
{
  //CHECK: VarDecl{{.*}}doublepump
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGADoublePumpAttr
  [[INTEL::doublepump]] unsigned int doublepump[64];

  //CHECK: VarDecl{{.*}}memory
  //CHECK: IntelFPGAMemoryAttr
  [[INTEL::fpga_memory]] unsigned int memory[64];

  //CHECK: VarDecl{{.*}}memory_mlab
  //CHECK: IntelFPGAMemoryAttr{{.*}}MLAB
  [[INTEL::fpga_memory("MLAB")]] unsigned int memory_mlab[64];

  //CHECK: VarDecl{{.*}}mem_blockram
  //CHECK: IntelFPGAMemoryAttr{{.*}}BlockRAM
  [[INTEL::fpga_memory("BLOCK_RAM")]] unsigned int mem_blockram[32];

  //CHECK: VarDecl{{.*}}reg
  //CHECK: IntelFPGARegisterAttr
  [[INTEL::fpga_register]] unsigned int reg[64];

  //CHECK: VarDecl{{.*}}singlepump
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGASinglePumpAttr
  [[INTEL::singlepump]] unsigned int singlepump[64];

  //CHECK: VarDecl{{.*}}bankwidth
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGABankWidthAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}4
  //CHECK-NEXT: IntegerLiteral{{.*}}4{{$}}
  [[INTEL::bankwidth(4)]] unsigned int bankwidth[32];

  //CHECK: VarDecl{{.*}}numbanks
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGANumBanksAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}8
  //CHECK-NEXT: IntegerLiteral{{.*}}8{{$}}
  [[INTEL::numbanks(8)]] unsigned int numbanks[32];

  //CHECK: VarDecl{{.*}}private_copies
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGAPrivateCopiesAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}8
  //CHECK-NEXT: IntegerLiteral{{.*}}8{{$}}
  [[INTEL::private_copies(8)]] unsigned int private_copies[64];

  //CHECK: VarDecl{{.*}}merge_depth
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGAMergeAttr{{.*}}"mrg1" "depth"{{$}}
  [[INTEL::merge("mrg1","depth")]] unsigned int merge_depth[64];

  //CHECK: VarDecl{{.*}}merge_width
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGAMergeAttr{{.*}}"mrg2" "width"{{$}}
  [[INTEL::merge("mrg2","width")]] unsigned int merge_width[64];

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
  [[INTEL::bank_bits(2,3,4,5)]] unsigned int bankbits[64];

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
  [[INTEL::bank_bits(2,3), INTEL::bankwidth(16)]]  unsigned int bank_bits_width[64];

  //CHECK: VarDecl{{.*}}doublepump_mlab
  //CHECK: IntelFPGADoublePumpAttr
  //CHECK: IntelFPGAMemoryAttr{{.*}}MLAB{{$}}
  [[INTEL::doublepump]]
  [[INTEL::fpga_memory("MLAB")]] unsigned int doublepump_mlab[64];

  //CHECK: VarDecl{{.*}}max_replicates
  //CHECK: IntelFPGAMaxReplicatesAttr
  //CHECK: ConstantExpr
  //CHECK-NEXT: value:{{.*}}2
  //CHECK: IntegerLiteral{{.*}}2{{$}}
  [[INTEL::max_replicates(2)]] unsigned int max_replicates[64];

  //CHECK: VarDecl{{.*}}dual_port
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGASimpleDualPortAttr
  [[INTEL::simple_dual_port]] unsigned int dual_port[64];

  //CHECK: VarDecl{{.*}}arr_force_p2d_0
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGAForcePow2DepthAttr
  //CHECK: ConstantExpr
  //CHECK-NEXT: value:{{.*}}0
  //CHECK: IntegerLiteral{{.*}}0{{$}}
  [[INTEL::force_pow2_depth(0)]] unsigned int arr_force_p2d_0[64];

  //CHECK: VarDecl{{.*}}arr_force_p2d_1
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGAForcePow2DepthAttr
  //CHECK: ConstantExpr
  //CHECK-NEXT: value:{{.*}}1
  //CHECK: IntegerLiteral{{.*}}1{{$}}
  [[INTEL::force_pow2_depth(1)]] unsigned int arr_force_p2d_1[64];

  [[INTEL::fpga_register]] int var_reg;
  [[INTEL::numbanks(4), INTEL::bankwidth(16), INTEL::singlepump]] int var_singlepump;
  [[INTEL::numbanks(4), INTEL::bankwidth(16), INTEL::doublepump]] int var_doublepump;
  [[INTEL::numbanks(4), INTEL::bankwidth(16)]] int var_numbanks_bankwidth;
  [[INTEL::bank_bits(2, 3), INTEL::bankwidth(16)]] int var_bank_bits_width;
  [[INTEL::max_replicates(2)]] int var_max_repl;
  [[INTEL::simple_dual_port]] int var_dual_port;
  [[INTEL::force_pow2_depth(1)]] int var_force_p2d;
  [[INTEL::force_pow2_depth(1)]] const int const_force_p2d[64] = {0, 1};
}

//CHECK: FunctionDecl{{.*}}diagnostics
void diagnostics()
{
  // **doublepump
  //expected-error@+2{{attributes are not compatible}}
  [[INTEL::doublepump]]
  [[INTEL::singlepump]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int dpump_spump[64];

  // expected-warning@+2 {{attribute 'doublepump' is deprecated}}
  // expected-note@+1 {{did you mean to use 'INTEL::doublepump' instead?}}
  [[intelfpga::doublepump]] unsigned int x[3];

  // expected-warning@+2 {{attribute 'singlepump' is deprecated}}
  // expected-note@+1 {{did you mean to use 'INTEL::singlepump' instead?}}
  [[intelfpga::singlepump]] unsigned int x1[3];

  // expected-warning@+2 {{attribute 'register' is deprecated}}
  // expected-note@+1 {{did you mean to use 'INTEL::fpga_register' instead?}}
  [[intelfpga::register]] unsigned int y1[3];

  // expected-warning@+2 {{attribute 'memory' is deprecated}}
  // expected-note@+1 {{did you mean to use 'INTEL::fpga_memory' instead?}}
  [[intelfpga::memory]] unsigned int y2[3];

  //expected-warning@+2{{attribute 'doublepump' is already applied}}
  [[INTEL::doublepump]]
  [[INTEL::doublepump]] unsigned int dpump[64];

  //expected-error@+2{{attributes are not compatible}}
  [[INTEL::doublepump]]
  [[INTEL::fpga_register]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int dpump_reg[64];

  // **singlepump
  //expected-error@+1{{attributes are not compatible}}
  [[INTEL::singlepump, INTEL::doublepump]]
  //expected-note@-1 {{conflicting attribute is here}}
  unsigned int spump_dpump[64];

  //expected-warning@+2{{attribute 'singlepump' is already applied}}
  [[INTEL::singlepump]]
  [[INTEL::singlepump]] unsigned int spump[64];

  //expected-error@+2{{attributes are not compatible}}
  [[INTEL::singlepump]]
  [[INTEL::fpga_register]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int spump_reg[64];

  // **fpga_register
  //expected-warning@+1{{attribute 'fpga_register' is already applied}}
  [[INTEL::fpga_register]] [[INTEL::fpga_register]] unsigned int reg_reg[64];

  //expected-error@+2{{attributes are not compatible}}
  [[INTEL::fpga_register]]
  [[INTEL::singlepump]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int reg_spump[64];

  //expected-error@+2{{attributes are not compatible}}
  [[INTEL::fpga_register]]
  [[INTEL::doublepump]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int reg_dpump[64];

  //expected-error@+2{{attributes are not compatible}}
  [[INTEL::fpga_register]]
  [[INTEL::fpga_memory]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int reg_memory[64];

  //expected-error@+2{{'bank_bits' and 'fpga_register' attributes are not compatible}}
  [[INTEL::fpga_register]]
  [[INTEL::bank_bits(4, 5)]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int reg_bankbits[64];

  // expected-warning@+2 {{attribute 'bank_bits' is deprecated}}
  // expected-note@+1 {{did you mean to use 'INTEL::bank_bits' instead?}}
  [[intelfpga::bank_bits(4, 5)]] unsigned int p[3];

  // expected-warning@+2 {{attribute 'bankwidth' is deprecated}}
  // expected-note@+1 {{did you mean to use 'INTEL::bankwidth' instead?}}
  [[intelfpga::bankwidth(2)]] unsigned int p1[3];

  // expected-warning@+2 {{attribute 'private_copies' is deprecated}}
  // expected-note@+1 {{did you mean to use 'INTEL::private_copies' instead?}}
  [[intelfpga::private_copies(3)]] unsigned int p2[3];

  // expected-warning@+2 {{attribute 'numbanks' is deprecated}}
  // expected-note@+1 {{did you mean to use 'INTEL::numbanks' instead?}}
  [[intelfpga::numbanks(8)]] unsigned int p3[3];

  //expected-error@+2{{'bankwidth' and 'fpga_register' attributes are not compatible}}
  [[INTEL::fpga_register]]
  [[INTEL::bankwidth(16)]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int reg_bankwidth[64];

  //expected-error@+2{{attributes are not compatible}}
  [[INTEL::fpga_register]]
  [[INTEL::private_copies(16)]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int reg_private_copies[64];

  //expected-error@+2{{attributes are not compatible}}
  [[INTEL::fpga_register]]
  [[INTEL::numbanks(8)]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int reg_numbanks[64];

  //expected-error@+2{{attributes are not compatible}}
  [[INTEL::fpga_register]]
  [[INTEL::merge("mrg1", "depth")]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int reg_merge[64];

  //expected-error@+3{{'max_replicates' and 'fpga_register' attributes are not compatible}}
  [[INTEL::fpga_register]]
  //expected-note@-1 {{conflicting attribute is here}}
  [[INTEL::max_replicates(2)]] unsigned int reg_maxrepl[64];

  //expected-error@+3{{'simple_dual_port' and 'fpga_register' attributes are not compatible}}
  [[INTEL::fpga_register]]
  //expected-note@-1 {{conflicting attribute is here}}
  [[INTEL::simple_dual_port]] unsigned int reg_dualport[64];

  //expected-error@+3{{'force_pow2_depth' and 'fpga_register' attributes are not compatible}}
  [[INTEL::fpga_register]]
  //expected-note@-1 {{conflicting attribute is here}}
  [[INTEL::force_pow2_depth(0)]] unsigned int reg_force_p2d[64];

  //expected-error@+3{{'fpga_register' and 'force_pow2_depth' attributes are not compatible}}
  [[INTEL::force_pow2_depth(1)]]
  //expected-note@-1 {{conflicting attribute is here}}
  [[INTEL::fpga_register]] unsigned int force_p2d_reg[64];

  // expected-warning@+2 {{attribute 'merge' is deprecated}}
  // expected-note@+1 {{did you mean to use 'INTEL::merge' instead?}}
  [[intelfpga::merge("mrg1", "depth")]] unsigned int r[3];

  // expected-warning@+2 {{attribute 'max_replicates' is deprecated}}
  // expected-note@+1 {{did you mean to use 'INTEL::max_replicates' instead?}}
  [[intelfpga::max_replicates(8)]] unsigned int r1[3];

  // expected-warning@+2 {{attribute 'force_pow2_depth' is deprecated}}
  // expected-note@+1 {{did you mean to use 'INTEL::force_pow2_depth' instead?}}
  [[intelfpga::force_pow2_depth(1)]] unsigned int r2[3];

  // expected-warning@+2 {{attribute 'simple_dual_port' is deprecated}}
  // expected-note@+1 {{did you mean to use 'INTEL::simple_dual_port' instead?}}
  [[intelfpga::simple_dual_port]] unsigned int r3[3];

  // **memory
  //expected-error@+2{{attributes are not compatible}}
  [[INTEL::fpga_memory]]
  [[INTEL::fpga_register]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int mem_reg[64];

  //expected-warning@+1{{attribute 'fpga_memory' is already applied}}
  [[INTEL::fpga_memory]] [[INTEL::fpga_memory]] unsigned int mem_mem[64];

  // bankwidth
  //expected-error@+2{{attributes are not compatible}}
  [[INTEL::bankwidth(16)]]
  [[INTEL::fpga_register]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int bankwidth_reg[64];

  // **max_replicates
  //expected-error@+1{{'max_replicates' attribute requires integer constant between 1 and 1048576 inclusive}}
  [[INTEL::max_replicates(0)]] unsigned int maxrepl_zero[64];
  //expected-error@+1{{'max_replicates' attribute requires integer constant between 1 and 1048576 inclusive}}
  [[INTEL::max_replicates(-1)]] unsigned int maxrepl_negative[64];

  //expected-error@+3{{'max_replicates' and 'fpga_register' attributes are not compatible}}
  [[INTEL::fpga_register]]
  //expected-note@-1 {{conflicting attribute is here}}
  [[INTEL::max_replicates(2)]] unsigned int maxrepl_reg[64];

  // **simple_dual_port
  //expected-error@+1{{'simple_dual_port' attribute takes no arguments}}
  [[INTEL::simple_dual_port(0)]] unsigned int sdp[64];

  //expected-note@+1 {{conflicting attribute is here}}
  [[INTEL::fpga_register]]
  //expected-error@+1{{'simple_dual_port' and 'fpga_register' attributes are not compatible}}
  [[INTEL::simple_dual_port]] unsigned int sdp_reg[64];

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
  [[INTEL::bankwidth(8)]]
  [[INTEL::bankwidth(16)]] unsigned int bw_bw[64];

  //expected-error@+1{{must be a constant power of two greater than zero}}
  [[INTEL::bankwidth(3)]] unsigned int bw_invalid_value[64];

  //expected-error@+1{{requires integer constant between 1 and 1048576}}
  [[INTEL::bankwidth(-4)]] unsigned int bw_negative[64];

  int i_bankwidth = 32; // expected-note {{declared here}}
  //expected-error@+1{{is not an integral constant expression}}
  [[INTEL::bankwidth(i_bankwidth)]]
  //expected-note@-1{{read of non-const variable 'i_bankwidth' is not allowed in a constant expression}}
  unsigned int bw_non_const[64];

  //expected-error@+1{{'bankwidth' attribute takes one argument}}
  [[INTEL::bankwidth(4,8)]] unsigned int bw_two_args[64];

  //expected-error@+1{{requires integer constant between 1 and 1048576}}
  [[INTEL::bankwidth(0)]] unsigned int bw_zero[64];


  // private_copies_
  //expected-error@+2{{attributes are not compatible}}
  [[INTEL::private_copies(16)]]
  [[INTEL::fpga_register]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int pc_reg[64];

  //CHECK: VarDecl{{.*}}pc_pc
  //CHECK: IntelFPGAPrivateCopiesAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}8
  //CHECK-NEXT: IntegerLiteral{{.*}}8{{$}}
  //CHECK: IntelFPGAPrivateCopiesAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}16
  //CHECK-NEXT: IntegerLiteral{{.*}}16{{$}}
  //expected-warning@+2{{is already applied}}
  [[INTEL::private_copies(8)]]
  [[INTEL::private_copies(16)]] unsigned int pc_pc[64];

  //expected-error@+1{{'private_copies' attribute requires integer constant between 0 and 1048576 inclusive}}
  [[INTEL::private_copies(-4)]] unsigned int pc_negative[64];

  int i_private_copies = 32; // expected-note {{declared here}}
  //expected-error@+1{{expression is not an integral constant expression}}
  [[INTEL::private_copies(i_private_copies)]]
  //expected-note@-1{{read of non-const variable 'i_private_copies' is not allowed in a constant expression}}
  unsigned int pc_nonconst[64];

  //expected-error@+1{{'private_copies' attribute takes one argument}}
  [[INTEL::private_copies(4,8)]] unsigned int pc_two_arg[64];

  // numbanks
  //expected-error@+2{{attributes are not compatible}}
  [[INTEL::numbanks(16)]]
  [[INTEL::fpga_register]]
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
  [[INTEL::numbanks(8)]]
  [[INTEL::numbanks(16)]]  unsigned int nb_nb[64];

  //expected-error@+1{{must be a constant power of two greater than zero}}
  [[INTEL::numbanks(15)]] unsigned int nb_invalid_arg[64];

  //expected-error@+1{{requires integer constant between 1 and 1048576}}
  [[INTEL::numbanks(-4)]] unsigned int nb_negative[64];

  int i_numbanks = 32; // expected-note {{declared here}}
  //expected-error@+1{{is not an integral constant expression}}
  [[INTEL::numbanks(i_numbanks)]]
  //expected-note@-1{{read of non-const variable 'i_numbanks' is not allowed in a constant expression}}
  unsigned int nb_nonconst[64];

  //expected-error@+1{{'numbanks' attribute takes one argument}}
  [[INTEL::numbanks(4,8)]] unsigned int nb_two_args[64];

  //expected-error@+1{{requires integer constant between 1 and 1048576}}
  [[INTEL::numbanks(0)]] unsigned int nb_zero[64];

  // merge
  //expected-error@+2{{attributes are not compatible}}
  [[INTEL::merge("mrg1", "depth")]]
  [[INTEL::fpga_register]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int mrg_reg[4];

  //expected-error@+1{{attribute requires a string}}
  [[INTEL::merge(3,9.0f)]] unsigned int mrg_float[4];

  //expected-error@+1{{attribute requires exactly 2 arguments}}
  [[INTEL::merge("mrg2")]] unsigned int mrg_one_arg[4];

  //expected-error@+1{{attribute requires exactly 2 arguments}}
  [[INTEL::merge("mrg3", "depth", "oops")]] unsigned int mrg_three_arg[4];

  //expected-error@+1{{merge direction must be 'depth' or 'width'}}
  [[INTEL::merge("mrg4", "depths")]] unsigned int mrg_invalid_arg[4];

  //Last one is applied and others ignored.
  //CHECK: VarDecl{{.*}}mrg_mrg
  //CHECK: IntelFPGAMergeAttr{{.*}}"mrg4" "depth"{{$}}
  //CHECK: IntelFPGAMergeAttr{{.*}}"mrg5" "width"{{$}}
  //expected-warning@+2{{attribute 'merge' is already applied}}
  [[INTEL::merge("mrg4", "depth")]]
  [[INTEL::merge("mrg5", "width")]]  unsigned int mrg_mrg[4];

  // bank_bits
  //expected-error@+2 1{{'fpga_register' and 'bank_bits' attributes are not compatible}}
  [[INTEL::bank_bits(2, 3)]]
  [[INTEL::fpga_register]]
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
  [[INTEL::bank_bits(42, 43)]]
  [[INTEL::bank_bits(1, 2)]]
  unsigned int bb_bb[4];

  //expected-error@+1{{the number of bank_bits must be equal to ceil(log2(numbanks))}}
  [[INTEL::numbanks(8), INTEL::bank_bits(3, 4)]] unsigned int bb_numbanks[4];

  //expected-error@+1{{bank_bits must be consecutive}}
  [[INTEL::bank_bits(3, 3, 4), INTEL::bankwidth(4)]]  unsigned int bb_noncons[4];

  //expected-error@+1{{bank_bits must be consecutive}}
  [[INTEL::bank_bits(1, 3, 4), INTEL::bankwidth(4)]] unsigned int bb_noncons1[4];

  //expected-error@+1{{attribute takes at least 1 argument}}
  [[INTEL::bank_bits]] unsigned int bb_no_arg[4];

  //expected-error@+1{{requires integer constant between 0 and 1048576}}
  [[INTEL::bank_bits(-1)]] unsigned int bb_negative_arg[4];

  // force_pow2_depth
  //expected-error@+1{{'force_pow2_depth' attribute requires integer constant between 0 and 1 inclusive}}
  [[INTEL::force_pow2_depth(-1)]] unsigned int force_p2d_below_min[64];
  //expected-error@+1{{'force_pow2_depth' attribute requires integer constant between 0 and 1 inclusive}}
  [[INTEL::force_pow2_depth(2)]] unsigned int force_p2d_above_max[64];

  //expected-error@+1{{'force_pow2_depth' attribute takes one argument}}
  [[INTEL::force_pow2_depth]] unsigned int force_p2d_no_args[64];
  //expected-error@+1{{'force_pow2_depth' attribute takes one argument}}
  [[INTEL::force_pow2_depth(0, 1)]] unsigned int force_p2d_2_args[64];

  //CHECK: VarDecl{{.*}}force_p2d_dup
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGAForcePow2DepthAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}1
  //CHECK-NEXT: IntegerLiteral{{.*}}1{{$}}
  //CHECK: IntelFPGAForcePow2DepthAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}0
  //CHECK-NEXT: IntegerLiteral{{.*}}0{{$}}
  //expected-warning@+1{{attribute 'force_pow2_depth' is already applied}}
  [[INTEL::force_pow2_depth(1), INTEL::force_pow2_depth(0)]] unsigned int force_p2d_dup[64];
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
[[INTEL::private_copies(8)]]
__attribute__((opencl_constant)) unsigned int const_var[64] = {1, 2, 3 };

void attr_on_const_error()
{
  //expected-error@+1{{attribute only applies to local non-const variables and non-static data members}}
  [[INTEL::private_copies(8)]] const int const_var[64] = {0, 1 };
}

//expected-error@+1{{attribute only applies to local non-const variables and non-static data members}}
void attr_on_func_arg([[INTEL::private_copies(8)]] int pc) {}

//expected-error@+1{{attribute only applies to constant variables, local variables, static variables, slave memory arguments, and non-static data members}}
[[INTEL::force_pow2_depth(0)]]
__attribute__((opencl_global)) unsigned int ocl_glob_force_p2d[64] = {1, 2, 3};

//expected-no-error@+1
void force_p2d_attr_on_func_arg([[INTEL::force_pow2_depth(0)]] int pc) {}

struct foo {
  //CHECK: FieldDecl{{.*}}doublepump
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGADoublePumpAttr
  [[INTEL::doublepump]] unsigned int doublepump[64];

  //CHECK: FieldDecl{{.*}}memory
  //CHECK: IntelFPGAMemoryAttr
  [[INTEL::fpga_memory]] unsigned int memory[64];

  //CHECK: FieldDecl{{.*}}memory_mlab
  //CHECK: IntelFPGAMemoryAttr{{.*}}MLAB{{$}}
  [[INTEL::fpga_memory("MLAB")]] unsigned int memory_mlab[64];

  //CHECK: FieldDecl{{.*}}mem_blockram
  //CHECK: IntelFPGAMemoryAttr{{.*}}BlockRAM{{$}}
  [[INTEL::fpga_memory("BLOCK_RAM")]] unsigned int mem_blockram[64];

  //CHECK: FieldDecl{{.*}}mem_blockram_doublepump
  //CHECK: IntelFPGAMemoryAttr{{.*}}BlockRAM{{$}}
  //CHECK: IntelFPGADoublePumpAttr
  [[INTEL::fpga_memory("BLOCK_RAM")]]
  [[INTEL::doublepump]] unsigned int mem_blockram_doublepump[64];

  //CHECK: FieldDecl{{.*}}reg
  //CHECK: IntelFPGARegisterAttr
  [[INTEL::fpga_register]] unsigned int reg[64];

  //CHECK: FieldDecl{{.*}}singlepump
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGASinglePumpAttr
  [[INTEL::singlepump]] unsigned int singlepump[64];

  //CHECK: FieldDecl{{.*}}bankwidth
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGABankWidthAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}4
  //CHECK-NEXT: IntegerLiteral{{.*}}4{{$}}
  [[INTEL::bankwidth(4)]] unsigned int bankwidth[64];

  //CHECK: FieldDecl{{.*}}numbanks
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGANumBanksAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}8
  //CHECK-NEXT: IntegerLiteral{{.*}}8{{$}}
  [[INTEL::numbanks(8)]] unsigned int numbanks[64];

  //CHECK: FieldDecl{{.*}}private_copies
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGAPrivateCopiesAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}4
  //CHECK-NEXT: IntegerLiteral{{.*}}4{{$}}
  [[INTEL::private_copies(4)]] unsigned int private_copies[64];

  //CHECK: FieldDecl{{.*}}merge_depth
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGAMergeAttr{{.*}}"mrg1" "depth"{{$}}
  [[INTEL::merge("mrg1", "depth")]] unsigned int merge_depth[64];

  //CHECK: FieldDecl{{.*}}merge_width
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGAMergeAttr{{.*}}"mrg2" "width"{{$}}
  [[INTEL::merge("mrg2", "width")]] unsigned int merge_width[64];

  //CHECK: FieldDecl{{.*}}bankbits
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGABankBitsAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}2
  //CHECK-NEXT: IntegerLiteral{{.*}}2{{$}}
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}3
  //CHECK-NEXT: IntegerLiteral{{.*}}3{{$}}
  [[INTEL::bank_bits(2,3)]] unsigned int bankbits[64];

  //CHECK: FieldDecl{{.*}}force_p2d_field
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGAForcePow2DepthAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}1
  //CHECK-NEXT: IntegerLiteral{{.*}}1{{$}}
  [[INTEL::force_pow2_depth(1)]] unsigned int force_p2d_field[64];
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
  [[INTEL::numbanks(C)]] unsigned int numbanks;

  //CHECK: VarDecl{{.*}}private_copies
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGAPrivateCopiesAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}8
  //CHECK-NEXT: SubstNonTypeTemplateParmExpr
  //CHECK-NEXT: NonTypeTemplateParmDecl
  //CHECK-NEXT: IntegerLiteral{{.*}}8{{$}}
  [[INTEL::private_copies(C)]] unsigned int private_copies;

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
  [[INTEL::bank_bits(A, 3), INTEL::bankwidth(C)]] unsigned int bank_bits_width;

  //CHECK: VarDecl{{.*}}max_replicates
  //CHECK: IntelFPGAMaxReplicatesAttr
  //CHECK: ConstantExpr
  //CHECK-NEXT: value:{{.*}}2
  //CHECK-NEXT: SubstNonTypeTemplateParmExpr
  //CHECK: IntegerLiteral{{.*}}2{{$}}
  [[INTEL::max_replicates(A)]]  unsigned int max_replicates;

  [[INTEL::force_pow2_depth(E)]] const int const_force_p2d_templ[64] = {0, 1};

  //expected-error@+1{{'numbanks' attribute takes one argument}}
  [[INTEL::numbanks(A, B)]] int numbanks_negative;

  //expected-error@+1{{'max_replicates' attribute requires integer constant between 1 and 1048576}}
  [[INTEL::max_replicates(D)]]
  [[INTEL::max_replicates(C)]]
  //expected-warning@-1{{attribute 'max_replicates' is already applied}}
  unsigned int max_replicates_duplicate;

  //expected-error@+3{{'max_replicates' and 'fpga_register' attributes are not compatible}}
  [[INTEL::fpga_register]]
  //expected-note@-1 {{conflicting attribute is here}}
  [[INTEL::max_replicates(C)]] unsigned int maxrepl_reg;

  //expected-error@+1{{'force_pow2_depth' attribute requires integer constant between 0 and 1 inclusive}}
  [[INTEL::force_pow2_depth(A)]] unsigned int force_p2d_below_min[64];

  //expected-error@+1{{'force_pow2_depth' attribute takes one argument}}
  [[INTEL::force_pow2_depth(E, E)]] unsigned int force_p2d_2_args[64];

  //expected-error@+3{{'force_pow2_depth' and 'fpga_register' attributes are not compatible}}
  [[INTEL::fpga_register]]
  //expected-note@-1{{conflicting attribute is here}}
  [[INTEL::force_pow2_depth(E)]] unsigned int reg_force_p2d[64];

  //CHECK: VarDecl{{.*}}force_p2d_dup
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGAForcePow2DepthAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}1
  //CHECK-NEXT: SubstNonTypeTemplateParmExpr
  //CHECK-NEXT: NonTypeTemplateParmDecl
  //CHECK-NEXT: IntegerLiteral{{.*}}1{{$}}
  //CHECK: IntelFPGAForcePow2DepthAttr
  //CHECK: ConstantExpr
  //CHECK-NEXT: value:{{.*}}0
  //CHECK-NEXT: IntegerLiteral{{.*}}0{{$}}
  //expected-warning@+1{{attribute 'force_pow2_depth' is already applied}}
  [[INTEL::force_pow2_depth(E), INTEL::force_pow2_depth(0)]] unsigned int force_p2d_dup[64];
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
  [[INTEL::force_pow2_depth(A)]] unsigned int templ_force_p2d_field[64];
};

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(const Func &kernelFunc) {
  kernelFunc();
}

int main() {
  kernel_single_task<class kernel_function>([]() {
    check_ast();
    diagnostics();
    check_gnu_style();
    //expected-note@+1{{in instantiation of function template specialization}}
    check_template_parameters<2, 4, 8, -1, 1>();
    struct templ_st<0> ts {};
  });
  return 0;
}
