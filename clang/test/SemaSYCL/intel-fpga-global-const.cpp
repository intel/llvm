// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -ast-dump -verify -pedantic %s | FileCheck %s

// Test that checks global constant variable (which allows the redeclaration) since
// SYCLIntelConstVar is one of the subjects listed for [[intel::max_replicates()]] attribute.

// Check duplicate argument values with implicit memory attribute.
//CHECK: VarDecl{{.*}}var_max_replicates
//CHECK: SYCLIntelMemoryAttr{{.*}}Implicit
//CHECK: SYCLIntelMaxReplicatesAttr
//CHECK-NEXT: ConstantExpr
//CHECK-NEXT: value:{{.*}}12
//CHECK-NEXT: IntegerLiteral{{.*}}12{{$}}
//CHECK: SYCLIntelMaxReplicatesAttr
//CHECK-NEXT: ConstantExpr
//CHECK-NEXT: value:{{.*}}12
//CHECK-NEXT: IntegerLiteral{{.*}}12{{$}}
[[intel::max_replicates(12)]] extern const int var_max_replicates;
[[intel::max_replicates(12)]] const int var_max_replicates = 0; // OK

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

// Test that checks global constant variable (which allows the redeclaration) since
// IntelFPGAConstVar is one of the subjects listed for [[intel::force_pow2_depth()]] attribute.

// Checking of duplicate argument values.
//CHECK: VarDecl{{.*}}force_pow2_depth
//CHECK: SYCLIntelMemoryAttr{{.*}}Implicit
//CHECK: SYCLIntelForcePow2DepthAttr
//CHECK-NEXT: ConstantExpr
//CHECK-NEXT: value:{{.*}}1
//CHECK-NEXT: IntegerLiteral{{.*}}1{{$}}
//CHECK: SYCLIntelForcePow2DepthAttr
//CHECK-NEXT: ConstantExpr
//CHECK-NEXT: value:{{.*}}1
//CHECK-NEXT: IntegerLiteral{{.*}}1{{$}}
[[intel::force_pow2_depth(1)]] extern const int var_force_pow2_depth;
[[intel::force_pow2_depth(1)]] const int var_force_pow2_depth = 0; // OK

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

// Test that checks global constant variable (which allows the redeclaration) since
// IntelFPGAConstVar is one of the subjects listed for [[intel::numbanks()]] attribute.

// Checking of duplicate argument values.
//CHECK: VarDecl{{.*}}numbanks
//CHECK: SYCLIntelMemoryAttr{{.*}}Implicit
//CHECK: SYCLIntelNumBanksAttr
//CHECK-NEXT: ConstantExpr{{.*}}'int'
//CHECK-NEXT: value: Int 16
//CHECK-NEXT: IntegerLiteral{{.*}}'int' 16
//CHECK: SYCLIntelNumBanksAttr
//CHECK-NEXT: ConstantExpr{{.*}}'int'
//CHECK-NEXT: value: Int 16
//CHECK-NEXT: IntegerLiteral{{.*}}'int' 16
[[intel::numbanks(16)]] extern const int var_numbanks;
[[intel::numbanks(16)]] const int var_numbanks = 0; // OK

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

// Test that checks global constant variable (which allows the redeclaration) since
// IntelFPGAConstVar is one of the subjects listed for [[intel::bankwidth()]] attribute.

// Checking of duplicate argument values.
//CHECK: VarDecl{{.*}}bankwidth
//CHECK: SYCLIntelMemoryAttr{{.*}}Implicit
//CHECK: SYCLIntelBankWidthAttr
//CHECK-NEXT: ConstantExpr{{.*}}'int'
//CHECK-NEXT: value: Int 8
//CHECK-NEXT: IntegerLiteral{{.*}}'int' 8
//CHECK: SYCLIntelBankWidthAttr
//CHECK-NEXT: ConstantExpr{{.*}}'int'
//CHECK-NEXT: value: Int 8
//CHECK-NEXT: IntegerLiteral{{.*}}'int' 8
[[intel::bankwidth(8)]] extern const int var_bankwidth;
[[intel::bankwidth(8)]] const int var_bankwidth = 0; // OK

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
