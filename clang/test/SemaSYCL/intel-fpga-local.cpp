// RUN: %clang_cc1 -Wno-return-type -fsycl-is-device -fcxx-exceptions -fsyntax-only -ast-dump -verify -pedantic %s | FileCheck %s

//CHECK: FunctionDecl{{.*}}check_ast
void check_ast()
{
  //CHECK: VarDecl{{.*}}doublepump
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGADoublePumpAttr
  [[intelfpga::doublepump]] unsigned int doublepump[64];

  //CHECK: VarDecl{{.*}}memory
  //CHECK: IntelFPGAMemoryAttr
  [[intelfpga::memory]] unsigned int memory[64];

  //CHECK: VarDecl{{.*}}memory_mlab
  //CHECK: IntelFPGAMemoryAttr{{.*}}MLAB
  [[intelfpga::memory("MLAB")]] unsigned int memory_mlab[64];

  //CHECK: VarDecl{{.*}}mem_blockram
  //CHECK: IntelFPGAMemoryAttr{{.*}}BlockRAM
  [[intelfpga::memory("BLOCK_RAM")]] unsigned int mem_blockram[32];

  //CHECK: VarDecl{{.*}}reg
  //CHECK: IntelFPGARegisterAttr
  [[intelfpga::register]] unsigned int reg[64];

  //CHECK: VarDecl{{.*}}singlepump
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGASinglePumpAttr
  [[intelfpga::singlepump]] unsigned int singlepump[64];

  //CHECK: VarDecl{{.*}}bankwidth
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGABankWidthAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: IntegerLiteral{{.*}}4{{$}}
  [[intelfpga::bankwidth(4)]] unsigned int bankwidth[32];

  //CHECK: VarDecl{{.*}}numbanks
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGANumBanksAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: IntegerLiteral{{.*}}8{{$}}
  [[intelfpga::numbanks(8)]] unsigned int numbanks[32];

  //CHECK: VarDecl{{.*}}private_copies
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGAPrivateCopiesAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: IntegerLiteral{{.*}}8{{$}}
  [[intelfpga::private_copies(8)]] unsigned int private_copies[64];

  //CHECK: VarDecl{{.*}}merge_depth
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGAMergeAttr{{.*}}"mrg1" "depth"{{$}}
  [[intelfpga::merge("mrg1","depth")]]
  unsigned int merge_depth[64];

  //CHECK: VarDecl{{.*}}merge_width
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGAMergeAttr{{.*}}"mrg2" "width"{{$}}
  [[intelfpga::merge("mrg2","width")]]
  unsigned int merge_width[64];

  //CHECK: VarDecl{{.*}}bankbits
  //CHECK: IntelFPGANumBanksAttr{{.*}}Implicit{{$}}
  //CHECK-NEXT: IntegerLiteral{{.*}}16{{$}}
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGABankBitsAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: IntegerLiteral{{.*}}2{{$}}
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: IntegerLiteral{{.*}}3{{$}}
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: IntegerLiteral{{.*}}4{{$}}
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: IntegerLiteral{{.*}}5{{$}}
  [[intelfpga::bank_bits(2,3,4,5)]]
  unsigned int bankbits[64];

  //CHECK: VarDecl{{.*}}bank_bits_width
  //CHECK-NEXT: IntelFPGANumBanksAttr{{.*}}Implicit{{$}}
  //CHECK-NEXT: IntegerLiteral{{.*}}4{{$}}
  //CHECK-NEXT: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK-NEXT: IntelFPGABankBitsAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: IntegerLiteral{{.*}}2{{$}}
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: IntegerLiteral{{.*}}3{{$}}
  //CHECK-NEXT: IntelFPGABankWidthAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: IntegerLiteral{{.*}}16{{$}}
  [[intelfpga::bank_bits(2,3), intelfpga::bankwidth(16)]]
  unsigned int bank_bits_width[64];

  //CHECK: VarDecl{{.*}}doublepump_mlab
  //CHECK: IntelFPGADoublePumpAttr
  //CHECK: IntelFPGAMemoryAttr{{.*}}MLAB{{$}}
  [[intelfpga::doublepump]]
  [[intelfpga::memory("MLAB")]]
  unsigned int doublepump_mlab[64];

  //CHECK: VarDecl{{.*}}mlab_doublepump
  //CHECK: IntelFPGAMemoryAttr{{.*}}MLAB{{$}}
  //CHECK: IntelFPGADoublePumpAttr
  [[intelfpga::memory("MLAB")]]
  [[intelfpga::doublepump]]
  unsigned int mlab_doublepump[64];

  //CHECK: VarDecl{{.*}}max_replicates
  //CHECK: IntelFPGAMaxReplicatesAttr
  //CHECK: ConstantExpr
  //CHECK: IntegerLiteral{{.*}}2{{$}}
  [[intelfpga::max_replicates(2)]]
  unsigned int max_replicates[64];

  //CHECK: VarDecl{{.*}}dual_port
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGASimpleDualPortAttr
  [[intelfpga::simple_dual_port]]
  unsigned int dual_port[64];

  [[intelfpga::register]] int var_reg;
  [[intelfpga::numbanks(4), intelfpga::bankwidth(16), intelfpga::singlepump]] int var_singlepump;
  [[intelfpga::numbanks(4), intelfpga::bankwidth(16), intelfpga::doublepump]] int var_doublepump;
  [[intelfpga::numbanks(4), intelfpga::bankwidth(16)]] int var_numbanks_bankwidth;
  [[intelfpga::bank_bits(2,3), intelfpga::bankwidth(16)]] int var_bank_bits_width;
  [[intelfpga::max_replicates(2)]] int var_max_repl;
  [[intelfpga::simple_dual_port]] int var_dual_port;
}

//CHECK: FunctionDecl{{.*}}diagnostics
void diagnostics()
{
  // **doublepump
  //expected-error@+2{{attributes are not compatible}}
  [[intelfpga::doublepump]]
  [[intelfpga::singlepump]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int dpump_spump[64];

  //expected-warning@+2{{attribute 'doublepump' is already applied}}
  [[intelfpga::doublepump]]
  [[intelfpga::doublepump]]
  unsigned int dpump[64];

  //expected-error@+2{{attributes are not compatible}}
  [[intelfpga::doublepump]]
  [[intelfpga::register]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int dpump_reg[64];

  // **singlepump
  //expected-error@+1{{attributes are not compatible}}
  [[intelfpga::singlepump, intelfpga::doublepump]]
  //expected-note@-1 {{conflicting attribute is here}}
  unsigned int spump_dpump[64];

  //expected-warning@+2{{attribute 'singlepump' is already applied}}
  [[intelfpga::singlepump]]
  [[intelfpga::singlepump]]
  unsigned int spump[64];

  //expected-error@+2{{attributes are not compatible}}
  [[intelfpga::singlepump]]
  [[intelfpga::register]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int spump_reg[64];

  // **register
  //expected-warning@+1{{attribute 'register' is already applied}}
  [[intelfpga::register]] [[intelfpga::register]]
  unsigned int reg_reg[64];

  //expected-error@+2{{attributes are not compatible}}
  [[intelfpga::register]]
  [[intelfpga::singlepump]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int reg_spump[64];

  //expected-error@+2{{attributes are not compatible}}
  [[intelfpga::register]]
  [[intelfpga::doublepump]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int reg_dpump[64];

  //expected-error@+2{{attributes are not compatible}}
  [[intelfpga::register]]
  [[intelfpga::memory]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int reg_memory[64];

  //expected-error@+2{{'bank_bits' and 'register' attributes are not compatible}}
  [[intelfpga::register]]
  [[intelfpga::bank_bits(4,5)]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int reg_bankbits[64];

  //expected-error@+2{{'bankwidth' and 'register' attributes are not compatible}}
  [[intelfpga::register]]
  [[intelfpga::bankwidth(16)]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int reg_bankwidth[64];

  //expected-error@+2{{attributes are not compatible}}
  [[intelfpga::register]]
  [[intelfpga::private_copies(16)]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int reg_private_copies[64];


  //expected-error@+2{{attributes are not compatible}}
  [[intelfpga::register]]
  [[intelfpga::numbanks(8)]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int reg_numbanks[64];

  //expected-error@+2{{attributes are not compatible}}
  [[intelfpga::register]]
  [[intelfpga::merge("mrg1","depth")]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int reg_merge[64];

  //expected-error@+3{{'max_replicates' and 'register' attributes are not compatible}}
  [[intelfpga::register]]
  //expected-note@-1 {{conflicting attribute is here}}
  [[intelfpga::max_replicates(2)]] unsigned int reg_maxrepl[64];

  //expected-error@+3{{'simple_dual_port' and 'register' attributes are not compatible}}
  [[intelfpga::register]]
  //expected-note@-1 {{conflicting attribute is here}}
  [[intelfpga::simple_dual_port]] unsigned int reg_dualport[64];

  // **memory
  //expected-error@+2{{attributes are not compatible}}
  [[intelfpga::memory]]
  [[intelfpga::register]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int mem_reg[64];

  //expected-warning@+1{{attribute 'memory' is already applied}}
  [[intelfpga::memory]] [[intelfpga::memory]]
  unsigned int mem_mem[64];

  // bankwidth
  //expected-error@+2{{attributes are not compatible}}
  [[intelfpga::bankwidth(16)]]
  [[intelfpga::register]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int bankwidth_reg[64];

  // **max_replicates
  //expected-error@+1{{'max_replicates' attribute requires integer constant between 1 and 1048576 inclusive}}
  [[intelfpga::max_replicates(0)]] unsigned int maxrepl_zero[64];
  //expected-error@+1{{'max_replicates' attribute requires integer constant between 1 and 1048576 inclusive}}
  [[intelfpga::max_replicates(-1)]] unsigned int maxrepl_negative[64];

  //expected-error@+3{{'max_replicates' and 'register' attributes are not compatible}}
  [[intelfpga::register]]
  //expected-note@-1 {{conflicting attribute is here}}
  [[intelfpga::max_replicates(2)]]
  unsigned int maxrepl_reg[64];

  // **simple_dual_port
  //expected-error@+1{{'simple_dual_port' attribute takes no arguments}}
  [[intelfpga::simple_dual_port(0)]] unsigned int sdp[64];

  //expected-note@+1 {{conflicting attribute is here}}
  [[intelfpga::register]]
  //expected-error@+1{{'simple_dual_port' and 'register' attributes are not compatible}}
  [[intelfpga::simple_dual_port]]
  unsigned int sdp_reg[64];

  //CHECK: VarDecl{{.*}}bw_bw
  //CHECK: IntelFPGABankWidthAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: IntegerLiteral{{.*}}8{{$}}
  //CHECK: IntelFPGABankWidthAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: IntegerLiteral{{.*}}16{{$}}
  //expected-warning@+2{{attribute 'bankwidth' is already applied}}
  [[intelfpga::bankwidth(8)]]
  [[intelfpga::bankwidth(16)]]
  unsigned int bw_bw[64];

  //expected-error@+1{{must be a constant power of two greater than zero}}
  [[intelfpga::bankwidth(3)]]
  unsigned int bw_invalid_value[64];

  //expected-error@+1{{requires integer constant between 1 and 1048576}}
  [[intelfpga::bankwidth(-4)]]
  unsigned int bw_negative[64];

  int i_bankwidth = 32; // expected-note {{declared here}}
  //expected-error@+1{{is not an integral constant expression}}
  [[intelfpga::bankwidth(i_bankwidth)]]
  //expected-note@-1{{read of non-const variable 'i_bankwidth' is not allowed in a constant expression}}
  unsigned int bw_non_const[64];

  //expected-error@+1{{'bankwidth' attribute takes one argument}}
  [[intelfpga::bankwidth(4,8)]]
  unsigned int bw_two_args[64];

  //expected-error@+1{{requires integer constant between 1 and 1048576}}
  [[intelfpga::bankwidth(0)]]
  unsigned int bw_zero[64];


  // private_copies_
  //expected-error@+2{{attributes are not compatible}}
  [[intelfpga::private_copies(16)]]
  [[intelfpga::register]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int pc_reg[64];

  //CHECK: VarDecl{{.*}}pc_pc
  //CHECK: IntelFPGAPrivateCopiesAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: IntegerLiteral{{.*}}8{{$}}
  //CHECK: IntelFPGAPrivateCopiesAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: IntegerLiteral{{.*}}16{{$}}
  //expected-warning@+2{{is already applied}}
  [[intelfpga::private_copies(8)]]
  [[intelfpga::private_copies(16)]]
  unsigned int pc_pc[64];

  //expected-error@+1{{'private_copies' attribute requires integer constant between 0 and 1048576 inclusive}}
  [[intelfpga::private_copies(-4)]]
  unsigned int pc_negative[64];

  int i_private_copies = 32; // expected-note {{declared here}}
  //expected-error@+1{{expression is not an integral constant expression}}
  [[intelfpga::private_copies(i_private_copies)]]
  //expected-note@-1{{read of non-const variable 'i_private_copies' is not allowed in a constant expression}}
  unsigned int pc_nonconst[64];

  //expected-error@+1{{'private_copies' attribute takes one argument}}
  [[intelfpga::private_copies(4,8)]]
  unsigned int pc_two_arg[64];

  // numbanks
  //expected-error@+2{{attributes are not compatible}}
  [[intelfpga::numbanks(16)]]
  [[intelfpga::register]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int nb_reg[64];

  //CHECK: VarDecl{{.*}}nb_nb
  //CHECK: IntelFPGANumBanksAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: IntegerLiteral{{.*}}8{{$}}
  //CHECK: IntelFPGANumBanksAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: IntegerLiteral{{.*}}16{{$}}
  //expected-warning@+2{{attribute 'numbanks' is already applied}}
  [[intelfpga::numbanks(8)]]
  [[intelfpga::numbanks(16)]]
  unsigned int nb_nb[64];

  //expected-error@+1{{must be a constant power of two greater than zero}}
  [[intelfpga::numbanks(15)]]
  unsigned int nb_invalid_arg[64];

  //expected-error@+1{{requires integer constant between 1 and 1048576}}
  [[intelfpga::numbanks(-4)]]
  unsigned int nb_negative[64];

  int i_numbanks = 32; // expected-note {{declared here}}
  //expected-error@+1{{is not an integral constant expression}}
  [[intelfpga::numbanks(i_numbanks)]]
  //expected-note@-1{{read of non-const variable 'i_numbanks' is not allowed in a constant expression}}
  unsigned int nb_nonconst[64];

  //expected-error@+1{{'numbanks' attribute takes one argument}}
  [[intelfpga::numbanks(4,8)]]
  unsigned int nb_two_args[64];

  //expected-error@+1{{requires integer constant between 1 and 1048576}}
  [[intelfpga::numbanks(0)]]
  unsigned int nb_zero[64];

  // merge
  //expected-error@+2{{attributes are not compatible}}
  [[intelfpga::merge("mrg1","depth")]]
  [[intelfpga::register]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int mrg_reg[4];

  //expected-error@+1{{attribute requires a string}}
  [[intelfpga::merge(3,9.0f)]]
  unsigned int mrg_float[4];

  //expected-error@+1{{attribute requires exactly 2 arguments}}
  [[intelfpga::merge("mrg2")]]
  unsigned int mrg_one_arg[4];

  //expected-error@+1{{attribute requires exactly 2 arguments}}
  [[intelfpga::merge("mrg3","depth","oops")]]
  unsigned int mrg_three_arg[4];

  //expected-error@+1{{merge direction must be 'depth' or 'width'}}
  [[intelfpga::merge("mrg4","depths")]]
  unsigned int mrg_invalid_arg[4];

  //Last one is applied and others ignored.
  //CHECK: VarDecl{{.*}}mrg_mrg
  //CHECK: IntelFPGAMergeAttr{{.*}}"mrg4" "depth"{{$}}
  //CHECK: IntelFPGAMergeAttr{{.*}}"mrg5" "width"{{$}}
  //expected-warning@+2{{attribute 'merge' is already applied}}
  [[intelfpga::merge("mrg4","depth")]]
  [[intelfpga::merge("mrg5","width")]]
  unsigned int mrg_mrg[4];

  // bank_bits
  //expected-error@+2 1{{'register' and 'bank_bits' attributes are not compatible}}
  [[intelfpga::bank_bits(2,3)]]
  [[intelfpga::register]]
  //expected-note@-2 1{{conflicting attribute is here}}
  unsigned int bb_reg[4];

  //CHECK: VarDecl{{.*}}bb_bb
  //CHECK: IntelFPGABankBitsAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: IntegerLiteral{{.*}}42{{$}}
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: IntegerLiteral{{.*}}43{{$}}
  //CHECK: IntelFPGABankBitsAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: IntegerLiteral{{.*}}1{{$}}
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: IntegerLiteral{{.*}}2{{$}}
  //expected-warning@+2{{attribute 'bank_bits' is already applied}}
  [[intelfpga::bank_bits(42,43)]]
  [[intelfpga::bank_bits(1,2)]]
  unsigned int bb_bb[4];

  //expected-error@+1{{the number of bank_bits must be equal to ceil(log2(numbanks))}}
  [[intelfpga::numbanks(8), intelfpga::bank_bits(3,4)]]
  unsigned int bb_numbanks[4];

  //expected-error@+1{{bank_bits must be consecutive}}
  [[intelfpga::bank_bits(3,3,4), intelfpga::bankwidth(4)]]
  unsigned int bb_noncons[4];

  //expected-error@+1{{bank_bits must be consecutive}}
  [[intelfpga::bank_bits(1,3,4), intelfpga::bankwidth(4)]]
  unsigned int bb_noncons_2[4];

  //expected-error@+1{{attribute takes at least 1 argument}}
  [[intelfpga::bank_bits]]
  unsigned int bb_no_arg[4];

  //expected-error@+1{{requires integer constant between 0 and 1048576}}
  [[intelfpga::bank_bits(-1)]]
  unsigned int bb_negative_arg[4];
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
}

//expected-error@+1{{attribute only applies to local non-const variables and non-static data members}}
[[intelfpga::private_copies(8)]]
__attribute__((opencl_constant)) unsigned int const_var[64] = { 1, 2, 3 };

void attr_on_const_error()
{
  //expected-error@+1{{attribute only applies to local non-const variables and non-static data members}}
  [[intelfpga::private_copies(8)]] const int const_var[64] = { 0, 1 };
}

//expected-error@+1{{attribute only applies to local non-const variables and non-static data members}}
void attr_on_func_arg([[intelfpga::private_copies(8)]] int pc) {}

struct foo {
  //CHECK: FieldDecl{{.*}}doublepump
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGADoublePumpAttr
  [[intelfpga::doublepump]] unsigned int doublepump[64];

  //CHECK: FieldDecl{{.*}}memory
  //CHECK: IntelFPGAMemoryAttr
  [[intelfpga::memory]] unsigned int memory[64];

  //CHECK: FieldDecl{{.*}}memory_mlab
  //CHECK: IntelFPGAMemoryAttr{{.*}}MLAB{{$}}
  [[intelfpga::memory("MLAB")]] unsigned int memory_mlab[64];

  //CHECK: FieldDecl{{.*}}mem_blockram
  //CHECK: IntelFPGAMemoryAttr{{.*}}BlockRAM{{$}}
  [[intelfpga::memory("BLOCK_RAM")]] unsigned int mem_blockram[64];

  //CHECK: FieldDecl{{.*}}mem_blockram_doublepump
  //CHECK: IntelFPGAMemoryAttr{{.*}}BlockRAM{{$}}
  //CHECK: IntelFPGADoublePumpAttr
  [[intelfpga::memory("BLOCK_RAM")]]
  [[intelfpga::doublepump]] unsigned int mem_blockram_doublepump[64];

  //CHECK: FieldDecl{{.*}}reg
  //CHECK: IntelFPGARegisterAttr
  [[intelfpga::register]] unsigned int reg[64];

  //CHECK: FieldDecl{{.*}}singlepump
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGASinglePumpAttr
  [[intelfpga::singlepump]] unsigned int singlepump[64];

  //CHECK: FieldDecl{{.*}}bankwidth
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGABankWidthAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: IntegerLiteral{{.*}}4{{$}}
  [[intelfpga::bankwidth(4)]] unsigned int bankwidth[64];

  //CHECK: FieldDecl{{.*}}numbanks
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGANumBanksAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: IntegerLiteral{{.*}}8{{$}}
  [[intelfpga::numbanks(8)]] unsigned int numbanks[64];

  //CHECK: FieldDecl{{.*}}private_copies
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGAPrivateCopiesAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: IntegerLiteral{{.*}}4{{$}}
  [[intelfpga::private_copies(4)]] unsigned int private_copies[64];

  //CHECK: FieldDecl{{.*}}merge_depth
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGAMergeAttr{{.*}}"mrg1" "depth"{{$}}
  [[intelfpga::merge("mrg1", "depth")]] unsigned int merge_depth[64];

  //CHECK: FieldDecl{{.*}}merge_width
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGAMergeAttr{{.*}}"mrg2" "width"{{$}}
  [[intelfpga::merge("mrg2", "width")]] unsigned int merge_width[64];

  //CHECK: FieldDecl{{.*}}bankbits
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGABankBitsAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: IntegerLiteral{{.*}}2{{$}}
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: IntegerLiteral{{.*}}3{{$}}
  [[intelfpga::bank_bits(2,3)]] unsigned int bankbits[64];
};

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  kernelFunc();
}

int main() {
  kernel_single_task<class kernel_function>([]() {
    check_ast();
    diagnostics();
    check_gnu_style();
  });
  return 0;
}
