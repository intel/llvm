// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -ast-dump -verify -pedantic %s | FileCheck %s

// Test that checks global constant variable (which allows the redeclaration) since
// IntelFPGAConstVar is one of the subjects listed for [[intel::max_replicates()]] attribute.

// Check duplicate argument values with implicit memory attribute.
//CHECK: VarDecl{{.*}}var_max_replicates
//CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
//CHECK: IntelFPGAMaxReplicatesAttr
//CHECK-NEXT: ConstantExpr
//CHECK-NEXT: value:{{.*}}12
//CHECK-NEXT: IntegerLiteral{{.*}}12{{$}}
//CHECK: IntelFPGAMaxReplicatesAttr
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
//CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
//CHECK: IntelFPGAForcePow2DepthAttr
//CHECK-NEXT: ConstantExpr
//CHECK-NEXT: value:{{.*}}1
//CHECK-NEXT: IntegerLiteral{{.*}}1{{$}}
//CHECK: IntelFPGAForcePow2DepthAttr
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
