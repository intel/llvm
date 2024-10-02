//==---- free_function_implicit_sycl_extern.cpp ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %clang_cc1 -fsycl-is-device -triple -spir64-unknown-unknown -verify %s

// expected-no-diagnostics

// This test confirms that functions or methods with add_ir_attributes_function
// attribute values in dependent contexts can be handled without assertions when
// checking for the presence of free function properties.

template <typename T> constexpr int value() { return 5; }

// In this struct the function the add_ir_attributes_function values for "S()"
// are as follows. Note that the "value" is represented as a CallExpr.
// `-SYCLAddIRAttributesFunctionAttr 0x562ec6c13390 < col:5, col : 67 >
// | -ConstantExpr 0x562ec6c13440 < col:49 > 'const char[5]' lvalue
// | |-value: LValue <todo>
// | `-StringLiteral 0x562ec6c13160 < col:49 > 'const char[5]' lvalue "name"
// `-CallExpr 0x562ec6c13220 < col:57, col : 66 > '<dependent type>'
//   `-UnresolvedLookupExpr 0x562ec6c131a8 < col:57, col : 64 > '<dependent type>' lvalue(ADL) = 'value' 0x562ec6bea700
//     `-TemplateArgument type 'T':'type-parameter-0-0'
//       `-TemplateTypeParmType 0x562ec6bea8b0 'T' dependent depth 0 index 0
//         `-TemplateTypeParm 0x562ec6bea860 'T'

template <typename T> struct S {
#if defined(__SYCL_DEVICE_ONLY__)
  [[__sycl_detail__::add_ir_attributes_function("name", value<T>())]]
#endif
  S() {
  }
};

// For the free function "f" the add_ir_attributes_function values are:
// | -SYCLAddIRAttributesFunctionAttr 0x56361c3c3ea8 < line:37 : 32, line : 39 : 15 >
// | |-ConstantExpr 0x56361c3c3f00 < line:38 : 5 > 'const char[5]' lvalue
// | | |-value: LValue <todo>
// | | `-StringLiteral 0x56361c398cf0 < col:5 > 'const char[5]' lvalue "name"
// | `-ConstantExpr 0x56361c3c3f60 < line:39 : 5, col : 14 > 'int'
// |   |-value: Int 5
// |   `-CallExpr 0x56361c3c3e88 < col:5, col : 14 > 'int'
// |     `-ImplicitCastExpr 0x56361c3c3e70 < col:5, col : 12 > 'int (*)()' < FunctionToPointerDecay >
// |       `-DeclRefExpr 0x56361c3c3dc0 < col:5, col : 12 > 'int ()' lvalue Function 0x56361c3c3cc8 'value' 'int ()' (FunctionTemplate 0x56361c398a90 'value')

template <typename T>
__attribute__((sycl_device)) [[__sycl_detail__::add_ir_attributes_function(
    "name",
    value<T>())]] [[__sycl_detail__::
                        add_ir_attributes_function("sycl-single-task-kernel",
                                                   0)]] void
f(T i) {}

template void f(int i);
