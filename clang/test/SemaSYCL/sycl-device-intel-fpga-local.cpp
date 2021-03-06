// RUN: %clang_cc1 -fsycl -fsycl-is-device -fsyntax-only -ast-dump -verify -pedantic %s | FileCheck %s

// Test that checks global constant variable (which does allow the redeclaration) since 
// IntelFPGAConstVar is one of the subjects listed for [[intel::max_replicates()]] attribute.

// Merging of different arg values
//CHECK: VarDecl{{.*}}var_max_replicates
//CHECK: IntelFPGAMaxReplicatesAttr
//CHECK-NEXT: ConstantExpr
//CHECK-NEXT: value:{{.*}}12
//CHECK-NEXT: IntegerLiteral{{.*}}12{{$}}
//expected-warning@+2{{attribute 'max_replicates' is already applied with different arguments}}
[[intel::max_replicates(12)]] extern const int var_max_replicates;
[[intel::max_replicates(14)]] const int var_max_replicates = 0;
//expected-note@-2{{previous attribute is here}}

// Merging of incompatible args 
//expected-error@+3{{'max_replicates' and 'fpga_register' attributes are not compatible}}
//expected-note@+1 {{conflicting attribute is here}}
[[intel::max_replicates(12)]] extern const int var_max_replicates_2;
[[intel::fpga_register]] const int var_max_replicates_2 =0;
