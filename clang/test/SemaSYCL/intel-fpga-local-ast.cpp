// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -sycl-std=2020 -ast-dump %s | FileCheck %s

// Tests AST for Intel FPGA memory attributes.

#include "sycl.hpp"

sycl::queue deviceQueue;

//CHECK: FunctionDecl{{.*}}check_ast
void check_ast()
{
  //CHECK: VarDecl{{.*}}doublepump
  //CHECK: SYCLIntelMemoryAttr{{.*}}Implicit
  //CHECK: SYCLIntelDoublePumpAttr
  [[intel::doublepump]] unsigned int doublepump[64];

  //CHECK: VarDecl{{.*}}memory
  //CHECK: SYCLIntelMemoryAttr
  [[intel::fpga_memory]] unsigned int memory[64];

  //CHECK: VarDecl{{.*}}memory_mlab
  //CHECK: SYCLIntelMemoryAttr{{.*}}MLAB
  [[intel::fpga_memory("MLAB")]] unsigned int memory_mlab[64];

  //CHECK: VarDecl{{.*}}mem_blockram
  //CHECK: SYCLIntelMemoryAttr{{.*}}BlockRAM
  [[intel::fpga_memory("BLOCK_RAM")]] unsigned int mem_blockram[32];

  //CHECK: VarDecl{{.*}}reg
  //CHECK: SYCLIntelRegisterAttr
  [[intel::fpga_register]] unsigned int reg[64];

  //CHECK: VarDecl{{.*}}singlepump
  //CHECK: SYCLIntelMemoryAttr{{.*}}Implicit
  //CHECK: SYCLIntelSinglePumpAttr
  [[intel::singlepump]] unsigned int singlepump[64];

  //CHECK: VarDecl{{.*}}bankwidth
  //CHECK: SYCLIntelMemoryAttr{{.*}}Implicit
  //CHECK: SYCLIntelBankWidthAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}4
  //CHECK-NEXT: IntegerLiteral{{.*}}4{{$}}
  [[intel::bankwidth(4)]] unsigned int bankwidth[32];

  //CHECK: VarDecl{{.*}}numbanks
  //CHECK: SYCLIntelMemoryAttr{{.*}}Implicit
  //CHECK: SYCLIntelNumBanksAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}8
  //CHECK-NEXT: IntegerLiteral{{.*}}8{{$}}
  [[intel::numbanks(8)]] unsigned int numbanks[32];

  //CHECK: VarDecl{{.*}}private_copies
  //CHECK: SYCLIntelMemoryAttr{{.*}}Implicit
  //CHECK: SYCLIntelPrivateCopiesAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}8
  //CHECK-NEXT: IntegerLiteral{{.*}}8{{$}}
  [[intel::private_copies(8)]] unsigned int private_copies[64];

  //CHECK: VarDecl{{.*}}merge_depth
  //CHECK: SYCLIntelMemoryAttr{{.*}}Implicit
  //CHECK: SYCLIntelMergeAttr{{.*}}"mrg1" "depth"{{$}}
  [[intel::merge("mrg1", "depth")]] unsigned int merge_depth[64];

  //CHECK: VarDecl{{.*}}merge_width
  //CHECK: SYCLIntelMemoryAttr{{.*}}Implicit
  //CHECK: SYCLIntelMergeAttr{{.*}}"mrg2" "width"{{$}}
  [[intel::merge("mrg2", "width")]] unsigned int merge_width[64];

  //CHECK: VarDecl{{.*}}bankbits
  //CHECK: SYCLIntelNumBanksAttr{{.*}}Implicit{{$}}
  //CHECK-NEXT: IntegerLiteral{{.*}}16{{$}}
  //CHECK: SYCLIntelMemoryAttr{{.*}}Implicit
  //CHECK: SYCLIntelBankBitsAttr
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
  //CHECK-NEXT: SYCLIntelNumBanksAttr{{.*}}Implicit
  //CHECK-NEXT: IntegerLiteral{{.*}}4{{$}}
  //CHECK-NEXT: SYCLIntelMemoryAttr{{.*}}Implicit
  //CHECK-NEXT: SYCLIntelBankBitsAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}2
  //CHECK-NEXT: IntegerLiteral{{.*}}2{{$}}
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}3
  //CHECK-NEXT: IntegerLiteral{{.*}}3{{$}}
  //CHECK-NEXT: SYCLIntelBankWidthAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}16
  //CHECK-NEXT: IntegerLiteral{{.*}}16{{$}}
  [[intel::bank_bits(2, 3), intel::bankwidth(16)]] unsigned int bank_bits_width[64];

  // Check user-specified numbanks attribute overrides implicit numbanks
  // attribute computed from the bank_bits attribute.
  //CHECK: VarDecl{{.*}}bank_bits_num 'unsigned int[64]'
  //CHECK-NOT:  SYCLIntelNumBanksAttr{{.*}}Implicit
  //CHECK-NEXT: SYCLIntelMemoryAttr{{.*}}Implicit
  //CHECK-NEXT: SYCLIntelBankBitsAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}2
  //CHECK-NEXT: IntegerLiteral{{.*}}2{{$}}
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}3
  //CHECK-NEXT: IntegerLiteral{{.*}}3{{$}}
  //CHECK-NEXT: SYCLIntelNumBanksAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}4
  //CHECK-NEXT: IntegerLiteral{{.*}}4{{$}}
  [[intel::bank_bits(2, 3), intel::numbanks(4)]] unsigned int bank_bits_num[64];

  //CHECK: VarDecl{{.*}}doublepump_mlab
  //CHECK: SYCLIntelDoublePumpAttr
  //CHECK: SYCLIntelMemoryAttr{{.*}}MLAB{{$}}
  [[intel::doublepump]]
  [[intel::fpga_memory("MLAB")]] unsigned int doublepump_mlab[64];

  // Add implicit memory attribute.
  //CHECK: VarDecl{{.*}}max_replicates
  //CHECK: SYCLIntelMemoryAttr{{.*}}Implicit
  //CHECK: SYCLIntelMaxReplicatesAttr
  //CHECK: ConstantExpr
  //CHECK-NEXT: value:{{.*}}2
  //CHECK: IntegerLiteral{{.*}}2{{$}}
  [[intel::max_replicates(2)]] unsigned int max_replicates[64];

  //CHECK: VarDecl{{.*}}dual_port
  //CHECK: SYCLIntelMemoryAttr{{.*}}Implicit
  //CHECK: SYCLIntelSimpleDualPortAttr
  [[intel::simple_dual_port]] unsigned int dual_port[64];

  //CHECK: VarDecl{{.*}}arr_force_p2d_0
  //CHECK: SYCLIntelMemoryAttr{{.*}}Implicit
  //CHECK: SYCLIntelForcePow2DepthAttr
  //CHECK: ConstantExpr
  //CHECK-NEXT: value:{{.*}}0
  //CHECK: IntegerLiteral{{.*}}0{{$}}
  [[intel::force_pow2_depth(0)]] unsigned int arr_force_p2d_0[64];

  //CHECK: VarDecl{{.*}}arr_force_p2d_1
  //CHECK: SYCLIntelMemoryAttr{{.*}}Implicit
  //CHECK: SYCLIntelForcePow2DepthAttr
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
  //CHECK: SYCLIntelMemoryAttr{{.*}}Implicit
  //CHECK: SYCLIntelMaxReplicatesAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}12
  //CHECK-NEXT: IntegerLiteral{{.*}}12{{$}}
  //CHECK-NOT:  SYCLIntelMaxReplicatesAttr
  [[intel::max_replicates(12)]]
  [[intel::max_replicates(12)]] int var_max_replicates; // OK

  // Check duplicate argument values.
  //CHECK: VarDecl{{.*}}var_private_copies
  //CHECK: SYCLIntelMemoryAttr{{.*}}Implicit
  //CHECK: SYCLIntelPrivateCopiesAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}12
  //CHECK-NEXT: IntegerLiteral{{.*}}12{{$}}
  //CHECK-NOT:  SYCLIntelPrivateCopiesAttr
  [[intel::private_copies(12)]]
  [[intel::private_copies(12)]] int var_private_copies; // OK

  // Checking of duplicate argument values.
  //CHECK: VarDecl{{.*}}var_forcep2d
  //CHECK: SYCLIntelMemoryAttr{{.*}}Implicit
  //CHECK: SYCLIntelForcePow2DepthAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}1
  //CHECK-NEXT: IntegerLiteral{{.*}}1{{$}}
  //CHECK-NOT:  SYCLIntelForcePow2DepthAttr
  [[intel::force_pow2_depth(1)]]
  [[intel::force_pow2_depth(1)]] int var_forcep2d; // OK

  // Test for Intel FPGA bankwidth memory attribute duplication.
  // No diagnostic is emitted because the arguments match.
  // Duplicate attribute is silently ignored.
  //CHECK: VarDecl{{.*}}var_bankwidth 'int'
  //CHECK: SYCLIntelMemoryAttr{{.*}}Implicit Default
  //CHECK: SYCLIntelBankWidthAttr
  //CHECK-NEXT: ConstantExpr{{.*}}'int'
  //CHECK-NEXT: value: Int 16
  //CHECK-NEXT: IntegerLiteral{{.*}}'int' 16
  //CHECK-NOT:  SYCLIntelBankWidthAttr
  [[intel::bankwidth(16)]]
  [[intel::bankwidth(16)]] int var_bankwidth; // OK

  // Test for Intel FPGA numbanks memory attribute duplication.
  // No diagnostic is emitted because the arguments match.
  // Duplicate attribute is silently ignored.
  //CHECK: VarDecl{{.*}}var_numbanks 'int'
  //CHECK: SYCLIntelMemoryAttr{{.*}}Implicit Default
  //CHECK: SYCLIntelNumBanksAttr
  //CHECK-NEXT: ConstantExpr{{.*}}'int'
  //CHECK-NEXT: value: Int 8
  //CHECK-NEXT: IntegerLiteral{{.*}}'int' 8
  //CHECK-NOT:  SYCLIntelNumBanksAttr
  [[intel::numbanks(8)]]
  [[intel::numbanks(8)]] int var_numbanks; // OK

  // Checking of different argument values.
  //CHECK: VarDecl{{.*}}bw_bw 'unsigned int[64]'
  //CHECK: SYCLIntelMemoryAttr{{.*}}Implicit Default
  //CHECK: SYCLIntelBankWidthAttr
  //CHECK-NEXT: ConstantExpr{{.*}}'int'
  //CHECK-NEXT: value: Int 8
  //CHECK-NEXT: IntegerLiteral{{.*}}'int' 8
  //CHECK-NOT:  SYCLIntelBankWidthAttr
  [[intel::bankwidth(8)]]
  [[intel::bankwidth(16)]] unsigned int bw_bw[64];

  // Checking of different argument values.
  //CHECK: VarDecl{{.*}}nb_nb 'unsigned int[64]'
  //CHECK: SYCLIntelMemoryAttr{{.*}}Implicit Default
  //CHECK: SYCLIntelNumBanksAttr
  //CHECK-NEXT: ConstantExpr{{.*}}'int'
  //CHECK-NEXT: value: Int 8
  //CHECK-NEXT: IntegerLiteral{{.*}}'int' 8
  //CHECK-NOT:  SYCLIntelNumBanksAttr
  [[intel::numbanks(8)]]
  [[intel::numbanks(16)]] unsigned int nb_nb[64];

  // Checking of different argument values.
  //CHECK: VarDecl{{.*}}mrg_mrg
  //CHECK: SYCLIntelMemoryAttr{{.*}}Implicit Default
  //CHECK: SYCLIntelMergeAttr{{.*}}"mrg4" "depth"
  //CHECK-NOT: SYCLIntelMergeAttr
  [[intel::merge("mrg4", "depth")]]
  [[intel::merge("mrg5", "width")]] unsigned int mrg_mrg[4];

  // Checking of duplicate argument values.
  //CHECK: VarDecl{{.*}}mrg_mrg6
  //CHECK: SYCLIntelMemoryAttr{{.*}}Implicit Default
  //CHECK: SYCLIntelMergeAttr{{.*}}"mrg6" "depth"
  //CHECK-NOT: SYCLIntelMergeAttr
  [[intel::merge("mrg6", "depth")]]
  [[intel::merge("mrg6", "depth")]] unsigned int mrg_mrg6[4];

  // Check to see if there's a duplicate attribute with different values
  // already applied to the declaration. Drop the duplicate attribute.
  // CHECK: VarDecl{{.*}}mem_block_ram
  // CHECK: SYCLIntelMemoryAttr{{.*}}MLAB
  // CHECK-NOT: SYCLIntelMemoryAttr
  [[intel::fpga_memory("MLAB")]]
  [[intel::fpga_memory("BLOCK_RAM")]] unsigned int mem_block_ram[32];

  // Check to see if there's a duplicate attribute with same Default values
  // already applied to the declaration. Drop the duplicate attribute.
  // CHECK: VarDecl{{.*}}mem_memory
  // CHECK: SYCLIntelMemoryAttr{{.*}}Default
  // CHECK-NOT: SYCLIntelMemoryAttr
  [[intel::fpga_memory]]
  [[intel::fpga_memory]] unsigned int mem_memory[64];

  // Check to see if there's a duplicate attribute with same values
  // already applied to the declaration. Drop the duplicate attribute.
  // CHECK: VarDecl{{.*}}mem_memory_block
  // CHECK: SYCLIntelMemoryAttr{{.*}}BlockRAM
  // CHECK-NOT: SYCLIntelMemoryAttr
  [[intel::fpga_memory("BLOCK_RAM")]]
  [[intel::fpga_memory("BLOCK_RAM")]] unsigned int mem_memory_block[64];

  // Check to see if there's a duplicate attribute with different values
  // already applied to the declaration. Drop the duplicate attribute.
  // CHECK: VarDecl{{.*}}mem_mlabs
  // CHECK: SYCLIntelMemoryAttr{{.*}}Default
  // CHECK-NOT: SYCLIntelMemoryAttr
  [[intel::fpga_memory]]
  [[intel::fpga_memory("MLAB")]] unsigned int mem_mlabs[64];

  // Check to see if there's a duplicate attribute with different values
  // already applied to the declaration. Drop the duplicate attribute.
  // CHECK: VarDecl{{.*}}mem_mlabs_block_ram
  // CHECK: SYCLIntelMemoryAttr{{.*}}BlockRAM
  // CHECK-NOT: SYCLIntelMemoryAttr
  [[intel::fpga_memory("BLOCK_RAM")]]
  [[intel::fpga_memory]] unsigned int mem_mlabs_block_ram[64];

  // FIXME: Duplicate attribute should be ignored.
  // Both are applied at the moment.
  //CHECK: VarDecl{{.*}}bb_bb
  //CHECK: SYCLIntelBankBitsAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}42
  //CHECK-NEXT: IntegerLiteral{{.*}}42{{$}}
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}43
  //CHECK-NEXT: IntegerLiteral{{.*}}43{{$}}
  [[intel::bank_bits(42, 43)]]
  [[intel::bank_bits(1, 2)]] unsigned int bb_bb[4];

  // Checking of different argument values.
  //CHECK: VarDecl{{.*}}force_p2d_dup
  //CHECK: SYCLIntelMemoryAttr{{.*}}Implicit
  //CHECK: SYCLIntelForcePow2DepthAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}1
  //CHECK-NEXT: IntegerLiteral{{.*}}1{{$}}
  //CHECK-NOT: SYCLIntelForcePow2DepthAttr
  [[intel::force_pow2_depth(1), intel::force_pow2_depth(0)]] unsigned int force_p2d_dup[64];

  // Checking of duplicate attributes.
  //CHECK: VarDecl{{.*}}dual_port1
  //CHECK: SYCLIntelMemoryAttr{{.*}}Implicit
  //CHECK: SYCLIntelSimpleDualPortAttr
  //CHECK-NOT: SYCLIntelSimpleDualPortAttr
  [[intel::simple_dual_port]]
  [[intel::simple_dual_port]] unsigned int dual_port1[64];

  // Checking of duplicate attributes.
  //CHECK: VarDecl{{.*}}doublepump1
  //CHECK: SYCLIntelMemoryAttr{{.*}}Implicit
  //CHECK: SYCLIntelDoublePumpAttr
  //CHECK-NOT: SYCLIntelDoublePumpAttr
  [[intel::doublepump]]
  [[intel::doublepump]] unsigned int doublepump1[64];

  // Checking of duplicate attributes.
  //CHECK: VarDecl{{.*}}singlepump1
  //CHECK: SYCLIntelMemoryAttr{{.*}}Implicit
  //CHECK: SYCLIntelSinglePumpAttr
  //CHECK-NOT: SYCLIntelSinglePumpAttr
  [[intel::singlepump]]
  [[intel::singlepump]] unsigned int singlepump1[64];

  // Checking of duplicate attributes.
  //CHECK: VarDecl{{.*}}reg1
  //CHECK: SYCLIntelRegisterAttr
  //CHECK-NOT: SYCLIntelRegisterAttr
  [[intel::fpga_register]]
  [[intel::fpga_register]] unsigned int reg1[64];
}

struct foo {
  //CHECK: FieldDecl{{.*}}doublepump
  //CHECK: SYCLIntelMemoryAttr{{.*}}Implicit
  //CHECK: SYCLIntelDoublePumpAttr
  [[intel::doublepump]] unsigned int doublepump[64];

  //CHECK: FieldDecl{{.*}}memory
  //CHECK: SYCLIntelMemoryAttr
  [[intel::fpga_memory]] unsigned int memory[64];

  //CHECK: FieldDecl{{.*}}memory_mlab
  //CHECK: SYCLIntelMemoryAttr{{.*}}MLAB{{$}}
  [[intel::fpga_memory("MLAB")]] unsigned int memory_mlab[64];

  //CHECK: FieldDecl{{.*}}mem_blockram
  //CHECK: SYCLIntelMemoryAttr{{.*}}BlockRAM{{$}}
  [[intel::fpga_memory("BLOCK_RAM")]] unsigned int mem_blockram[64];

  //CHECK: FieldDecl{{.*}}mem_blockram_doublepump
  //CHECK: SYCLIntelMemoryAttr{{.*}}BlockRAM{{$}}
  //CHECK: SYCLIntelDoublePumpAttr
  [[intel::fpga_memory("BLOCK_RAM")]]
  [[intel::doublepump]] unsigned int mem_blockram_doublepump[64];

  //CHECK: FieldDecl{{.*}}reg
  //CHECK: SYCLIntelRegisterAttr
  [[intel::fpga_register]] unsigned int reg[64];

  //CHECK: FieldDecl{{.*}}singlepump
  //CHECK: SYCLIntelMemoryAttr{{.*}}Implicit
  //CHECK: SYCLIntelSinglePumpAttr
  [[intel::singlepump]] unsigned int singlepump[64];

  //CHECK: FieldDecl{{.*}}bankwidth
  //CHECK: SYCLIntelMemoryAttr{{.*}}Implicit
  //CHECK: SYCLIntelBankWidthAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}4
  //CHECK-NEXT: IntegerLiteral{{.*}}4{{$}}
  [[intel::bankwidth(4)]] unsigned int bankwidth[64];

  //CHECK: FieldDecl{{.*}}numbanks
  //CHECK: SYCLIntelMemoryAttr{{.*}}Implicit
  //CHECK: SYCLIntelNumBanksAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}8
  //CHECK-NEXT: IntegerLiteral{{.*}}8{{$}}
  [[intel::numbanks(8)]] unsigned int numbanks[64];

  //CHECK: FieldDecl{{.*}}private_copies
  //CHECK: SYCLIntelMemoryAttr{{.*}}Implicit
  //CHECK: SYCLIntelPrivateCopiesAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}4
  //CHECK-NEXT: IntegerLiteral{{.*}}4{{$}}
  [[intel::private_copies(4)]] unsigned int private_copies[64];

  //CHECK: FieldDecl{{.*}}merge_depth
  //CHECK: SYCLIntelMemoryAttr{{.*}}Implicit
  //CHECK: SYCLIntelMergeAttr{{.*}}"mrg1" "depth"{{$}}
  [[intel::merge("mrg1", "depth")]] unsigned int merge_depth[64];

  //CHECK: FieldDecl{{.*}}merge_width
  //CHECK: SYCLIntelMemoryAttr{{.*}}Implicit
  //CHECK: SYCLIntelMergeAttr{{.*}}"mrg2" "width"{{$}}
  [[intel::merge("mrg2", "width")]] unsigned int merge_width[64];

  //CHECK: FieldDecl{{.*}}bankbits
  //CHECK: SYCLIntelMemoryAttr{{.*}}Implicit
  //CHECK: SYCLIntelBankBitsAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}2
  //CHECK-NEXT: IntegerLiteral{{.*}}2{{$}}
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}3
  //CHECK-NEXT: IntegerLiteral{{.*}}3{{$}}
  [[intel::bank_bits(2, 3)]] unsigned int bankbits[64];

  //CHECK: FieldDecl{{.*}}force_p2d_field
  //CHECK: SYCLIntelMemoryAttr{{.*}}Implicit
  //CHECK: SYCLIntelForcePow2DepthAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}1
  //CHECK-NEXT: IntegerLiteral{{.*}}1{{$}}
  [[intel::force_pow2_depth(1)]] unsigned int force_p2d_field[64];
};

//CHECK: FunctionDecl{{.*}}used check_template_parameters
template <int A, int B, int C, int D>
void check_template_parameters() {
  //CHECK: VarDecl{{.*}}numbanks
  //CHECK-NEXT: SYCLIntelMemoryAttr{{.*}}Implicit
  //CHECK-NEXT: SYCLIntelNumBanksAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}8
  //CHECK-NEXT: SubstNonTypeTemplateParmExpr
  //CHECK-NEXT: NonTypeTemplateParmDecl
  //CHECK-NEXT: IntegerLiteral{{.*}}8{{$}}
  [[intel::numbanks(C)]] unsigned int numbanks;

  //CHECK: VarDecl{{.*}}private_copies
  //CHECK: SYCLIntelMemoryAttr{{.*}}Implicit
  //CHECK: SYCLIntelPrivateCopiesAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}4
  //CHECK-NEXT: SubstNonTypeTemplateParmExpr
  //CHECK-NEXT: NonTypeTemplateParmDecl
  //CHECK-NEXT: IntegerLiteral{{.*}}4{{$}}
  [[intel::private_copies(B)]] unsigned int private_copies;

  //CHECK: VarDecl{{.*}}bank_bits_width
  //CHECK: SYCLIntelNumBanksAttr{{.*}}Implicit{{$}}
  //CHECK-NEXT: IntegerLiteral{{.*}}4{{$}}
  //CHECK-NEXT: SYCLIntelMemoryAttr{{.*}}Implicit
  //CHECK-NEXT: SYCLIntelBankBitsAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}2
  //CHECK-NEXT: SubstNonTypeTemplateParmExpr
  //CHECK-NEXT: NonTypeTemplateParmDecl
  //CHECK-NEXT: IntegerLiteral{{.*}}2{{$}}
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}3
  //CHECK-NEXT: IntegerLiteral{{.*}}3{{$}}
  //CHECK: SYCLIntelBankWidthAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}8
  //CHECK-NEXT: SubstNonTypeTemplateParmExpr
  //CHECK-NEXT: NonTypeTemplateParmDecl
  //CHECK-NEXT: IntegerLiteral{{.*}}8{{$}}
  [[intel::bank_bits(A, 3), intel::bankwidth(C)]] unsigned int bank_bits_width;

  // Add implicit memory attribute.
  //CHECK: VarDecl{{.*}}max_replicates
  //CHECK: SYCLIntelMemoryAttr{{.*}}Implicit
  //CHECK: SYCLIntelMaxReplicatesAttr
  //CHECK: ConstantExpr
  //CHECK-NEXT: value:{{.*}}2
  //CHECK-NEXT: SubstNonTypeTemplateParmExpr
  //CHECK: IntegerLiteral{{.*}}2{{$}}
  [[intel::max_replicates(A)]] unsigned int max_replicates;


  //CHECK: VarDecl{{.*}}force_p2d_dup
  //CHECK: SYCLIntelMemoryAttr{{.*}}Implicit
  //CHECK: SYCLIntelForcePow2DepthAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: value:{{.*}}1
  //CHECK-NEXT: SubstNonTypeTemplateParmExpr
  //CHECK-NEXT: NonTypeTemplateParmDecl
  //CHECK-NEXT: IntegerLiteral{{.*}}1{{$}}
  [[intel::force_pow2_depth(D)]] unsigned int force_p2d_dup[64];
}

template <int A>
struct templ_st {
  //CHECK: FieldDecl{{.*}}templ_force_p2d_field
  //CHECK: SYCLIntelMemoryAttr{{.*}}Implicit
  //CHECK: SYCLIntelForcePow2DepthAttr
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
      check_template_parameters<2, 4, 8, 1>();
      struct templ_st<0> ts {};
    });
  });

  return 0;
}

// Test that checks global constant variable (which allows the redeclaration) since
// SYCLIntelConstVar is one of the subjects listed for [[intel::max_replicates()]] attribute.

// Check redeclaration of duplicate argument values with implicit memory
// attribute. No diagnostic is generated.
//CHECK: VarDecl{{.*}}var_max_replicates 'const int' extern
//CHECK: SYCLIntelMemoryAttr{{.*}}Implicit Default
//CHECK: SYCLIntelMaxReplicatesAttr
//CHECK-NEXT: ConstantExpr{{.*}}'int'
//CHECK-NEXT: value: Int 12
//CHECK-NEXT: IntegerLiteral{{.*}}'int' 12
//CHECK: VarDecl{{.*}}var_max_replicates 'const int' cinit
//CHECK: IntegerLiteral{{.*}}'int' 0
//CHECK: SYCLIntelMemoryAttr{{.*}}Implicit Default
//CHECK: SYCLIntelMaxReplicatesAttr
//CHECK-NEXT: ConstantExpr{{.*}}'int'
//CHECK-NEXT: value: Int 12
//CHECK-NEXT: IntegerLiteral{{.*}}'int' 12
[[intel::max_replicates(12)]] extern const int var_max_replicates;
[[intel::max_replicates(12)]] const int var_max_replicates = 0; // OK

// Test that checks global constant variable (which allows the redeclaration) since
// IntelFPGAConstVar is one of the subjects listed for [[intel::force_pow2_depth()]] attribute.

// Check redeclaration of duplicate argument values with implicit memory.
// No diagnostic is generated.
//CHECK: VarDecl{{.*}}var_force_pow2_depth 'const int' extern
//CHECK: SYCLIntelMemoryAttr{{.*}}Implicit Default
//CHECK: SYCLIntelForcePow2DepthAttr
//CHECK-NEXT: ConstantExpr{{.*}}'int'
//CHECK-NEXT: value: Int 1
//CHECK-NEXT: IntegerLiteral{{.*}}'int' 1
//CHECK: VarDecl{{.*}}var_force_pow2_depth 'const int' cinit
//CHECK: IntegerLiteral{{.*}}'int' 0
//CHECK: SYCLIntelMemoryAttr{{.*}}Implicit Default
//CHECK: SYCLIntelForcePow2DepthAttr
//CHECK-NEXT: ConstantExpr{{.*}} 'int'
//CHECK-NEXT: value: Int 1
//CHECK-NEXT: IntegerLiteral{{.*}}'int' 1
[[intel::force_pow2_depth(1)]] extern const int var_force_pow2_depth;
[[intel::force_pow2_depth(1)]] const int var_force_pow2_depth = 0; // OK

// Test that checks global constant variable (which allows the redeclaration) since
// IntelFPGAConstVar is one of the subjects listed for [[intel::numbanks()]] attribute.

// Check redeclaration of duplicate argument values with implicit memory.
// No diagnostic is generated.
//CHECK: VarDecl{{.*}}var_numbanks 'const int' extern
//CHECK: SYCLIntelMemoryAttr{{.*}}Implicit Default
//CHECK: SYCLIntelNumBanksAttr
//CHECK-NEXT: ConstantExpr{{.*}}'int'
//CHECK-NEXT: value: Int 16
//CHECK-NEXT: IntegerLiteral{{.*}}'int' 16
//CHECK: VarDecl{{.*}}var_numbanks 'const int' cinit
//CHECK: IntegerLiteral{{.*}}'int' 0
//CHECK: SYCLIntelNumBanksAttr
//CHECK-NEXT: ConstantExpr{{.*}}'int'
//CHECK-NEXT: value: Int 16
//CHECK-NEXT: IntegerLiteral{{.*}}'int' 16
[[intel::numbanks(16)]] extern const int var_numbanks;
[[intel::numbanks(16)]] const int var_numbanks = 0; // OK

// Test that checks global constant variable (which allows the redeclaration) since
// IntelFPGAConstVar is one of the subjects listed for [[intel::bankwidth()]] attribute.

// Check redeclaration of duplicate argument values with implicit memory.
// No diagnostic is generated.
//CHECK: VarDecl{{.*}}var_bankwidth 'const int' extern
//CHECK: SYCLIntelMemoryAttr{{.*}}Implicit Default
//CHECK: SYCLIntelBankWidthAttr
//CHECK-NEXT: ConstantExpr{{.*}}'int'
//CHECK-NEXT: value: Int 8
//CHECK-NEXT: IntegerLiteral{{.*}}'int' 8
//CHECK: VarDecl{{.*}}var_bankwidth 'const int' cinit
//CHECK: IntegerLiteral{{.*}}'int' 0
//CHECK: SYCLIntelBankWidthAttr
//CHECK-NEXT: ConstantExpr{{.*}}'int'
//CHECK-NEXT: value: Int 8
//CHECK-NEXT: IntegerLiteral{{.*}}'int' 8
[[intel::bankwidth(8)]] extern const int var_bankwidth;
[[intel::bankwidth(8)]] const int var_bankwidth = 0; // OK
