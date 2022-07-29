// RUN: %clang_cc1 -fsyntax-only -verify -Wimplicit-float-conversion -Wdouble-promotion -Wimplicit-float-size-conversion \
// RUN:   -nostdsysteminc -nobuiltininc -isystem %S/Inputs \
// RUN:   -triple x86_64-apple-darwin -DCHECK_ALL_THREE %s -Wno-unreachable-code

// RUN: %clang_cc1 -fsyntax-only -verify -Wdouble-promotion -Wimplicit-float-size-conversion \
// RUN:   -nostdsysteminc -nobuiltininc -isystem %S/Inputs \
// RUN:   -triple x86_64-apple-darwin -DCHECK_PROMOTION_SIZE %s -Wno-unreachable-code

// RUN: %clang_cc1 -fsyntax-only -verify -Wdouble-promotion \
// RUN:   -nostdsysteminc -nobuiltininc -isystem %S/Inputs \
// RUN:   -triple x86_64-apple-darwin -DCHECK_PROMOTION %s -Wno-unreachable-code

// RUN: %clang_cc1 -fsyntax-only -verify -Wimplicit-float-conversion \
// RUN:   -nostdsysteminc -nobuiltininc -isystem %S/Inputs \
// RUN:   -triple x86_64-apple-darwin -DCHECK_LOSS %s -Wno-unreachable-code

// RUN: %clang_cc1 -fsyntax-only -verify \
// RUN:   -nostdsysteminc -nobuiltininc -isystem %S/Inputs \
// RUN:   -triple x86_64-apple-darwin -DCHECK_NO_WARNINGS %s -Wno-unreachable-code

// This test checks that conversion warnings are emitted correctly when used in conjunction.

#include <conversion.h>

#if defined(CHECK_ALL_THREE)
float PrecisionLoss = 1.1;       // expected-warning {{implicit conversion loses floating-point precision: 'double' to 'float'}}
double PrecisionIncrease = 2.1f; // expected-warning {{implicit conversion increases floating-point precision: 'float' to 'double'}}
float SizeChange = 3.0;          // expected-warning {{implicit conversion between floating point types of different sizes}}

#elif defined(CHECK_PROMOTION_SIZE)
float PrecisionLoss = 1.1;       // expected-warning {{implicit conversion between floating point types of different sizes}}
double PrecisionIncrease = 2.1f; // expected-warning {{implicit conversion increases floating-point precision: 'float' to 'double'}}
float SizeChange = 3.0;          // expected-warning {{implicit conversion between floating point types of different sizes}}

#elif defined(CHECK_PROMOTION)
float PrecisionLoss = 1.1;
double PrecisionIncrease = 2.1f; // expected-warning {{implicit conversion increases floating-point precision: 'float' to 'double'}}
float SizeChange = 3.0;

#elif defined(CHECK_LOSS)
float PrecisionLoss = 1.1; // expected-warning {{implicit conversion loses floating-point precision: 'double' to 'float'}}
double PrecisionIncrease = 2.1f;
float SizeChange = 3.0; // expected-warning {{implicit conversion between floating point types of different sizes}}

#elif defined(CHECK_NO_WARNINGS)
// expected-no-diagnostics
float PrecisionLoss = 1.1;
double PrecisionIncrease = 2.1f;
float SizeChange = 3.0;
#endif
