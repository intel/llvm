// RUN: %clang_cc1 -fsyntax-only -verify=precision-loss,precision-gain,size-change -Wimplicit-float-conversion -Wdouble-promotion -Wimplicit-float-size-conversion \
// RUN:   -triple x86_64-apple-darwin %s

// RUN: %clang_cc1 -fsyntax-only -verify=size-only,precision-gain,size-change -Wdouble-promotion -Wimplicit-float-size-conversion \
// RUN:   -triple x86_64-apple-darwin %s

// RUN: %clang_cc1 -fsyntax-only -verify=precision-increase -Wdouble-promotion \
// RUN:   -triple x86_64-apple-darwin %s

// RUN: %clang_cc1 -fsyntax-only -verify=precision-loss,size-change -Wimplicit-float-conversion \
// RUN:   -triple x86_64-apple-darwin %s

// RUN: %clang_cc1 -fsyntax-only -verify \
// RUN:   -triple x86_64-apple-darwin %s

// This test checks that floating point conversion warnings are emitted correctly when used in conjunction.

// expected-no-diagnostics
// size-only-warning@+2 {{implicit conversion between floating point types of different sizes}}
// precision-loss-warning@+1 {{implicit conversion loses floating-point precision: 'double' to 'float'}}
float PrecisionLoss = 1.1;
// precision-increase-warning@+2 {{implicit conversion increases floating-point precision: 'float' to 'double'}}
// precision-gain-warning@+1 {{implicit conversion increases floating-point precision: 'float' to 'double'}}
double PrecisionIncrease = 2.1f;
// size-change-warning@+1 {{implicit conversion between floating point types of different sizes}}
float SizeChange = 3.0;
