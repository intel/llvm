///////////////////////////////////////////////////////////////////////////////////////////
// Test that clang-offload-wrapper in "-batch" mode can accept multiple batch/table files
///////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////
// Generate input and table for target 0
///////////////////////////////////////////////////////////////////////////////////////////

// Generate image files
// RUN: echo 't0-A' > %t.0A.bin
// RUN: echo 't0-B' > %t.0B.bin
// RUN: echo 't0-C' > %t.0C.bin

// Generate property files
// RUN: echo '[testProp-GroupA]'                                             >  %t.0-A.prop
// RUN: echo 'testProp-name2=1|0'                                            >> %t.0-A.prop

// RUN: echo '[testProp-GroupB]'                                             >  %t.0-B.prop
// RUN: echo 'testProp-name3=1|0'                                            >> %t.0-B.prop

// RUN: echo '[testProp-GroupC]'                                             >  %t.0-C.prop
// RUN: echo 'testProp-name5=1|0'                                            >> %t.0-C.prop

// Generate sym files
// RUN: echo '_entry_targ_0_A'                                               >  %t.0-A.sym
// RUN: echo '_entry_targ_0_B'                                               >  %t.0-B.sym
// RUN: echo '_entry_targ_0_C'                                               >  %t.0-C.sym

// Generate table file
// RUN: echo '[Code|Properties|Symbols]'                                     >  %t.0.table
// RUN: echo '%t.0A.bin|%t.0-A.prop|%t.0-A.sym'                              >> %t.0.table
// RUN: echo '%t.0B.bin|%t.0-B.prop|%t.0-B.sym'                              >> %t.0.table
// RUN: echo '%t.0C.bin|%t.0-C.prop|%t.0-C.sym'                              >> %t.0.table

///////////////////////////////////////////////////////////////////////////////////////////
// Generate input and table for target 1
///////////////////////////////////////////////////////////////////////////////////////////

// Generate image files
// RUN: echo 't1-A' > %t.1A.bin
// RUN: echo 't1-B' > %t.1B.bin
// RUN: echo 't1-C' > %t.1C.bin

// Generate property files
// RUN: echo '[testProp-GroupD]'                                             >  %t.1-A.prop
// RUN: echo 'testProp-name7=1|0'                                            >> %t.1-A.prop

// RUN: echo '[testProp-GroupE]'                                             >  %t.1-B.prop
// RUN: echo 'testProp-name11=1|0'                                           >> %t.1-B.prop

// RUN: echo '[testProp-GroupF]'                                             >  %t.1-C.prop
// RUN: echo 'testProp-name13=1|0'                                           >> %t.1-C.prop

// Generate sym files
// RUN: echo '_entry_targ_1_AA'                                              >  %t.1-A.sym
// RUN: echo '_entry_targ_1_BB'                                              >  %t.1-B.sym
// RUN: echo '_entry_targ_1_CC'                                              >  %t.1-C.sym

// Generate table file
// RUN: echo '[Code|Properties|Symbols]'                                     >  %t.1.table
// RUN: echo '%t.1A.bin|%t.1-A.prop|%t.1-A.sym'                              >> %t.1.table
// RUN: echo '%t.1B.bin|%t.1-B.prop|%t.1-B.sym'                              >> %t.1.table
// RUN: echo '%t.1C.bin|%t.1-C.prop|%t.1-C.sym'                              >> %t.1.table

///////////////////////////////////////////////////////////////////////////////////////////
// Generate input and table for target 2
///////////////////////////////////////////////////////////////////////////////////////////

// Generate image files
// RUN: echo 't2-A' > %t.2A.bin
// RUN: echo 't2-B' > %t.2B.bin
// RUN: echo 't2-C' > %t.2C.bin

// Generate property files
// RUN: echo '[testProp-GroupG]'                                             >  %t.2-A.prop
// RUN: echo 'testProp-name17=1|0'                                           >> %t.2-A.prop

// RUN: echo '[testProp-GroupH]'                                             >  %t.2-B.prop
// RUN: echo 'testProp-name19=1|0'                                           >> %t.2-B.prop

// RUN: echo '[testProp-GroupI]'                                             >  %t.2-C.prop
// RUN: echo 'testProp-name23=1|0'                                           >> %t.2-C.prop

// Generate sym files
// RUN: echo '_entry_targ_2_AAA'                                             >  %t.2-A.sym
// RUN: echo '_entry_targ_2_BBB'                                             >  %t.2-B.sym
// RUN: echo '_entry_targ_2_CCC'                                             >  %t.2-C.sym

// Generate table file
// RUN: echo '[Code|Properties|Symbols]'                                     >  %t.2.table
// RUN: echo '%t.2A.bin|%t.2-A.prop|%t.2-A.sym'                              >> %t.2.table
// RUN: echo '%t.2B.bin|%t.2-B.prop|%t.2-B.sym'                              >> %t.2.table
// RUN: echo '%t.2C.bin|%t.2-C.prop|%t.2-C.sym'                              >> %t.2.table

///////////////////////////////////////////////////////////////////////////////////////////
// Generate wrapped BC file with multiple targets using multiple batch (i.e. table) files
// and generate executable
///////////////////////////////////////////////////////////////////////////////////////////
// RUN: clang-offload-wrapper "-o=%t.wrapped.bc"           \
// RUN:   -kind=sycl -target=target-zero -batch %t.0.table \
// RUN:   -kind=sycl -target=target-one  -batch %t.1.table \
// RUN:   -kind=sycl -target=target-two  -batch %t.2.table
// RUN: %clang %s %t.wrapped.bc -o %t.fat.bin

///////////////////////////////////////////////////////////////////////////////////////////
// Check that resulting executable has all target images and entry points and properties
// are correct
///////////////////////////////////////////////////////////////////////////////////////////

// Extract images
//
// RUN: clang-offload-extract --stem=%t.extracted %t.fat.bin | FileCheck %s --check-prefix CHECK-EXTRACT
// CHECK-EXTRACT: Section {{.*}}: Image 1'-> File
// CHECK-EXTRACT: Section {{.*}}: Image 2'-> File
// CHECK-EXTRACT: Section {{.*}}: Image 3'-> File
// CHECK-EXTRACT: Section {{.*}}: Image 4'-> File
// CHECK-EXTRACT: Section {{.*}}: Image 5'-> File
// CHECK-EXTRACT: Section {{.*}}: Image 6'-> File
// CHECK-EXTRACT: Section {{.*}}: Image 7'-> File
// CHECK-EXTRACT: Section {{.*}}: Image 8'-> File
// CHECK-EXTRACT: Section {{.*}}: Image 9'-> File

//
// Check that extracted contents match the original images.
//
// RUN: diff %t.extracted.0 %t.0A.bin
// RUN: diff %t.extracted.1 %t.0B.bin
// RUN: diff %t.extracted.2 %t.0C.bin
// RUN: diff %t.extracted.3 %t.1A.bin
// RUN: diff %t.extracted.4 %t.1B.bin
// RUN: diff %t.extracted.5 %t.1C.bin
// RUN: diff %t.extracted.6 %t.2A.bin
// RUN: diff %t.extracted.7 %t.2B.bin
// RUN: diff %t.extracted.8 %t.2C.bin

// Check disassembly
//
// RUN: llvm-dis %t.wrapped.bc
// RUN: FileCheck %s --check-prefix CHECK-DISASM < %t.wrapped.ll

// CHECK-DISASM:      @.sycl_offloading.target.{{.*}} = internal unnamed_addr constant
// CHECK-DISASM-SAME: target-zero
// CHECK-DISASM:      @prop{{.*}} = internal unnamed_addr constant
// CHECK-DISASM-SAME: testProp-name2
// CHECK-DISASM:      @SYCL_PropSetName{{.*}} = internal unnamed_addr constant
// CHECK-DISASM-SAME: testProp-GroupA
// CHECK-DISASM:      @.sycl_offloading.{{.*}}.data = internal unnamed_addr constant
// CHECK-DISASM-SAME: __CLANG_OFFLOAD_BUNDLE__sycl-target-zero
// CHECK-DISASM:      __sycl_offload_entry_name{{.*}} = internal unnamed_addr constant
// CHECK-DISASM-SAME: _entry_targ_0_A

// CHECK-DISASM:      @.sycl_offloading.target.{{.*}} = internal unnamed_addr constant
// CHECK-DISASM-SAME: target-zero
// CHECK-DISASM:      @prop{{.*}} = internal unnamed_addr constant
// CHECK-DISASM-SAME: testProp-name3
// CHECK-DISASM:      @SYCL_PropSetName{{.*}} = internal unnamed_addr constant
// CHECK-DISASM-SAME: testProp-GroupB
// CHECK-DISASM:      @.sycl_offloading.{{.*}}.data = internal unnamed_addr constant
// CHECK-DISASM-SAME: __CLANG_OFFLOAD_BUNDLE__sycl-target-zero
// CHECK-DISASM:      __sycl_offload_entry_name{{.*}} = internal unnamed_addr constant
// CHECK-DISASM-SAME: _entry_targ_0_B

// CHECK-DISASM:      @.sycl_offloading.target.{{.*}} = internal unnamed_addr constant
// CHECK-DISASM-SAME: target-zero
// CHECK-DISASM:      @prop{{.*}} = internal unnamed_addr constant
// CHECK-DISASM-SAME: testProp-name5
// CHECK-DISASM:      @SYCL_PropSetName{{.*}} = internal unnamed_addr constant
// CHECK-DISASM-SAME: testProp-GroupC
// CHECK-DISASM:      @.sycl_offloading.{{.*}}.data = internal unnamed_addr constant
// CHECK-DISASM-SAME: __CLANG_OFFLOAD_BUNDLE__sycl-target-zero
// CHECK-DISASM:      __sycl_offload_entry_name{{.*}} = internal unnamed_addr constant
// CHECK-DISASM-SAME: _entry_targ_0_C

// CHECK-DISASM:      @.sycl_offloading.target.{{.*}} = internal unnamed_addr constant
// CHECK-DISASM-SAME: target-one
// CHECK-DISASM:      @prop{{.*}} = internal unnamed_addr constant
// CHECK-DISASM-SAME: testProp-name7
// CHECK-DISASM:      @SYCL_PropSetName{{.*}} = internal unnamed_addr constant
// CHECK-DISASM-SAME: testProp-GroupD
// CHECK-DISASM:      @.sycl_offloading.{{.*}}.data = internal unnamed_addr constant
// CHECK-DISASM-SAME: __CLANG_OFFLOAD_BUNDLE__sycl-target-one
// CHECK-DISASM:      __sycl_offload_entry_name{{.*}} = internal unnamed_addr constant
// CHECK-DISASM-SAME: _entry_targ_1_AA

// CHECK-DISASM:      @.sycl_offloading.target.{{.*}} = internal unnamed_addr constant
// CHECK-DISASM-SAME: target-one
// CHECK-DISASM:      @prop{{.*}} = internal unnamed_addr constant
// CHECK-DISASM-SAME: testProp-name11
// CHECK-DISASM:      @SYCL_PropSetName{{.*}} = internal unnamed_addr constant
// CHECK-DISASM-SAME: testProp-GroupE
// CHECK-DISASM:      @.sycl_offloading.{{.*}}.data = internal unnamed_addr constant
// CHECK-DISASM-SAME: __CLANG_OFFLOAD_BUNDLE__sycl-target-one
// CHECK-DISASM:      __sycl_offload_entry_name{{.*}} = internal unnamed_addr constant
// CHECK-DISASM-SAME: _entry_targ_1_BB

// CHECK-DISASM:      @.sycl_offloading.target.{{.*}} = internal unnamed_addr constant
// CHECK-DISASM-SAME: target-one
// CHECK-DISASM:      @prop{{.*}} = internal unnamed_addr constant
// CHECK-DISASM-SAME: testProp-name13
// CHECK-DISASM:      @SYCL_PropSetName{{.*}} = internal unnamed_addr constant
// CHECK-DISASM-SAME: testProp-GroupF
// CHECK-DISASM:      @.sycl_offloading.{{.*}}.data = internal unnamed_addr constant
// CHECK-DISASM-SAME: __CLANG_OFFLOAD_BUNDLE__sycl-target-one
// CHECK-DISASM:      __sycl_offload_entry_name{{.*}} = internal unnamed_addr constant
// CHECK-DISASM-SAME: _entry_targ_1_CC

// CHECK-DISASM:      @.sycl_offloading.target.{{.*}} = internal unnamed_addr constant
// CHECK-DISASM-SAME: target-two
// CHECK-DISASM:      @prop{{.*}} = internal unnamed_addr constant
// CHECK-DISASM-SAME: testProp-name17
// CHECK-DISASM:      @SYCL_PropSetName{{.*}} = internal unnamed_addr constant
// CHECK-DISASM-SAME: testProp-GroupG
// CHECK-DISASM:      @.sycl_offloading.{{.*}}.data = internal unnamed_addr constant
// CHECK-DISASM-SAME: __CLANG_OFFLOAD_BUNDLE__sycl-target-two
// CHECK-DISASM:      __sycl_offload_entry_name{{.*}} = internal unnamed_addr constant
// CHECK-DISASM-SAME: _entry_targ_2_AAA

// CHECK-DISASM:      @.sycl_offloading.target.{{.*}} = internal unnamed_addr constant
// CHECK-DISASM-SAME: target-two
// CHECK-DISASM:      @prop{{.*}} = internal unnamed_addr constant
// CHECK-DISASM-SAME: testProp-name19
// CHECK-DISASM:      @SYCL_PropSetName{{.*}} = internal unnamed_addr constant
// CHECK-DISASM-SAME: testProp-GroupH
// CHECK-DISASM:      @.sycl_offloading.{{.*}}.data = internal unnamed_addr constant
// CHECK-DISASM-SAME: __CLANG_OFFLOAD_BUNDLE__sycl-target-two
// CHECK-DISASM:      __sycl_offload_entry_name{{.*}} = internal unnamed_addr constant
// CHECK-DISASM-SAME: _entry_targ_2_BBB

// CHECK-DISASM:      @.sycl_offloading.target.{{.*}} = internal unnamed_addr constant
// CHECK-DISASM-SAME: target-two
// CHECK-DISASM:      @prop{{.*}} = internal unnamed_addr constant
// CHECK-DISASM-SAME: testProp-name23
// CHECK-DISASM:      @SYCL_PropSetName{{.*}} = internal unnamed_addr constant
// CHECK-DISASM-SAME: testProp-GroupI
// CHECK-DISASM:      @.sycl_offloading.{{.*}}.data = internal unnamed_addr constant
// CHECK-DISASM-SAME: __CLANG_OFFLOAD_BUNDLE__sycl-target-two
// CHECK-DISASM:      __sycl_offload_entry_name{{.*}} = internal unnamed_addr constant
// CHECK-DISASM-SAME: _entry_targ_2_CCC

///////////////////////////////////////////////////////////////////////////////////////////
// Generate the same wrapped BC file with a single "-batch" option.  This shows that
// "-batch" is a mode setting for all input, not an option to introduce a batch/table file.
// Note that "-batch" even affects prior input (e.g. %t.0.table).
///////////////////////////////////////////////////////////////////////////////////////////
// RUN: clang-offload-wrapper "-o=%t.wrapped2.bc"          \
// RUN:   -kind=sycl -target=target-zero        %t.0.table \
// RUN:   -kind=sycl -target=target-one         %t.1.table \
// RUN:   -kind=sycl -target=target-two  -batch %t.2.table

// Check that single "-batch" produces the same result as multi "-batch".
// RUN: cmp %t.wrapped.bc %t.wrapped2.bc

///////////////////////////////////////////////////////////////////////////////////////////
// Some code so that we can build an offload executable from this file.
///////////////////////////////////////////////////////////////////////////////////////////

void __sycl_register_lib(void* desc) {}
void __sycl_unregister_lib(void* desc) {}

int main(void) {
  return 0;
}
