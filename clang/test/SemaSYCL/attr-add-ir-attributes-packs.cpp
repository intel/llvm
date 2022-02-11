// RUN: %clang_cc1 -fsycl-is-device -std=gnu++11 -fsyntax-only -verify %s
// RUN: not %clang_cc1 -fsycl-is-device -std=gnu++11 -ast-dump %s | FileCheck %s

constexpr const char AttrName1[] = "Attr1";
constexpr const char AttrName2[] = "Attr2";
constexpr const char AttrName3[] = "Attr3";
constexpr const char AttrVal1[] = "Val1";
constexpr const char AttrVal2[] = "Val2";
constexpr const char AttrVal3[] = "Val3";

template <int... Is> [[__sycl_detail__::add_ir_attributes_function("Attr1", "Attr2", "Attr3", Is...)]] void FunctionTemplate1() {}                     // expected-error {{attribute 'add_ir_attributes_function' must have an attribute value for each attribute name}}
template <int... Is> [[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, "Attr1", "Attr2", "Attr3", Is...)]] void FunctionTemplate2() {} // expected-error {{attribute 'add_ir_attributes_function' must have an attribute value for each attribute name}}
template <const char *...Names> [[__sycl_detail__::add_ir_attributes_function(Names..., 1, 2, 3)]] void FunctionTemplate3() {}                         // expected-error {{attribute 'add_ir_attributes_function' must have an attribute value for each attribute name}}
template <const char *...Names> [[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, Names..., 1, 2, 3)]] void FunctionTemplate4() {}     // expected-error {{attribute 'add_ir_attributes_function' must have an attribute value for each attribute name}}
template <const char *...Strs> [[__sycl_detail__::add_ir_attributes_function(Strs...)]] void FunctionTemplate5() {}                                    // expected-error {{attribute 'add_ir_attributes_function' must have an attribute value for each attribute name}}
template <const char *...Strs> [[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, Strs...)]] void FunctionTemplate6() {}                // expected-error {{attribute 'add_ir_attributes_function' must have an attribute value for each attribute name}}

void InstantiateFunctionTemplates() {
  // CHECK:      FunctionTemplateDecl {{.*}} FunctionTemplate1
  // CHECK-NEXT:   NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 0 index 0 ... Is
  // CHECK-NEXT:   FunctionDecl {{.*}} FunctionTemplate1 'void ()'
  // CHECK-NEXT:     CompoundStmt
  // CHECK-NEXT:     SYCLAddIRAttributesFunctionAttr
  // CHECK-NEXT:       ConstantExpr {{.*}} 'const char[6]' lvalue
  // CHECK-NEXT:         value: LValue
  // CHECK-NEXT:         StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
  // CHECK-NEXT:       ConstantExpr {{.*}} 'const char[6]' lvalue
  // CHECK-NEXT:         value: LValue
  // CHECK-NEXT:         StringLiteral {{.*}} 'const char[6]' lvalue "Attr2"
  // CHECK-NEXT:       ConstantExpr {{.*}} 'const char[6]' lvalue
  // CHECK-NEXT:         value: LValue
  // CHECK-NEXT:         StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
  // CHECK-NEXT:       PackExpansionExpr {{.*}} '<dependent type>'
  // CHECK-NEXT:         DeclRefExpr {{.*}} 'int' NonTypeTemplateParm {{.*}} 'Is' 'int'
  // CHECK-NEXT:   FunctionDecl {{.*}} used FunctionTemplate1 'void ()'
  // CHECK-NEXT:     TemplateArgument pack
  // CHECK-NEXT:       TemplateArgument integral 1
  // CHECK-NEXT:       TemplateArgument integral 2
  // CHECK-NEXT:       TemplateArgument integral 3
  // CHECK-NEXT:     CompoundStmt
  // CHECK-NEXT:     SYCLAddIRAttributesFunctionAttr
  // CHECK-NEXT:       ConstantExpr {{.*}} 'const char[6]' lvalue
  // CHECK-NEXT:         value: LValue
  // CHECK-NEXT:         StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
  // CHECK-NEXT:       ConstantExpr {{.*}} 'const char[6]' lvalue
  // CHECK-NEXT:         value: LValue
  // CHECK-NEXT:         StringLiteral {{.*}} 'const char[6]' lvalue "Attr2"
  // CHECK-NEXT:       ConstantExpr {{.*}} 'const char[6]' lvalue
  // CHECK-NEXT:         value: LValue
  // CHECK-NEXT:         StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
  // CHECK-NEXT:       ConstantExpr {{.*}} 'int'
  // CHECK-NEXT:         value: Int 1
  // CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'int'
  // CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 0 index 0 ... Is
  // CHECK-NEXT:           IntegerLiteral {{.*}} 'int' 1
  // CHECK-NEXT:       ConstantExpr {{.*}} 'int'
  // CHECK-NEXT:         value: Int 2
  // CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'int'
  // CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 0 index 0 ... Is
  // CHECK-NEXT:           IntegerLiteral {{.*}} 'int' 2
  // CHECK-NEXT:       ConstantExpr {{.*}} 'int'
  // CHECK-NEXT:         value: Int 3
  // CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'int'
  // CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 0 index 0 ... Is
  // CHECK-NEXT:           IntegerLiteral {{.*}} 'int' 3
  // CHECK-NEXT:   FunctionDecl {{.*}} used FunctionTemplate1 'void ()'
  // CHECK-NEXT:     TemplateArgument pack
  // CHECK-NEXT:       TemplateArgument integral 1
  // CHECK-NEXT:       TemplateArgument integral 2
  // CHECK-NEXT:     CompoundStmt
  FunctionTemplate1<1, 2, 3>();
  FunctionTemplate1<1, 2>(); // expected-note {{in instantiation of function template specialization 'FunctionTemplate1<1, 2>' requested here}}

  // CHECK:      FunctionTemplateDecl {{.*}} FunctionTemplate2
  // CHECK-NEXT:   NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 0 index 0 ... Is
  // CHECK-NEXT:   FunctionDecl {{.*}} FunctionTemplate2 'void ()'
  // CHECK-NEXT:     CompoundStmt
  // CHECK-NEXT:     SYCLAddIRAttributesFunctionAttr
  // CHECK-NEXT:       InitListExpr {{.*}} 'void'
  // CHECK-NEXT:         StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
  // CHECK-NEXT:         StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
  // CHECK-NEXT:       ConstantExpr {{.*}} 'const char[6]' lvalue
  // CHECK-NEXT:         value: LValue
  // CHECK-NEXT:         StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
  // CHECK-NEXT:       ConstantExpr {{.*}} 'const char[6]' lvalue
  // CHECK-NEXT:         value: LValue
  // CHECK-NEXT:         StringLiteral {{.*}} 'const char[6]' lvalue "Attr2"
  // CHECK-NEXT:       ConstantExpr {{.*}} 'const char[6]' lvalue
  // CHECK-NEXT:         value: LValue
  // CHECK-NEXT:         StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
  // CHECK-NEXT:       PackExpansionExpr {{.*}} '<dependent type>'
  // CHECK-NEXT:         DeclRefExpr {{.*}} 'int' NonTypeTemplateParm {{.*}} 'Is' 'int'
  // CHECK-NEXT:   FunctionDecl {{.*}} used FunctionTemplate2 'void ()'
  // CHECK-NEXT:     TemplateArgument pack
  // CHECK-NEXT:       TemplateArgument integral 1
  // CHECK-NEXT:       TemplateArgument integral 2
  // CHECK-NEXT:       TemplateArgument integral 3
  // CHECK-NEXT:     CompoundStmt
  // CHECK-NEXT:     SYCLAddIRAttributesFunctionAttr
  // CHECK-NEXT:       InitListExpr {{.*}} 'void'
  // CHECK-NEXT:         StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
  // CHECK-NEXT:         StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
  // CHECK-NEXT:       ConstantExpr {{.*}} 'const char[6]' lvalue
  // CHECK-NEXT:         value: LValue
  // CHECK-NEXT:         StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
  // CHECK-NEXT:       ConstantExpr {{.*}} 'const char[6]' lvalue
  // CHECK-NEXT:         value: LValue
  // CHECK-NEXT:         StringLiteral {{.*}} 'const char[6]' lvalue "Attr2"
  // CHECK-NEXT:       ConstantExpr {{.*}} 'const char[6]' lvalue
  // CHECK-NEXT:         value: LValue
  // CHECK-NEXT:         StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
  // CHECK-NEXT:       ConstantExpr {{.*}} 'int'
  // CHECK-NEXT:         value: Int 1
  // CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'int'
  // CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 0 index 0 ... Is
  // CHECK-NEXT:           IntegerLiteral {{.*}} 'int' 1
  // CHECK-NEXT:       ConstantExpr {{.*}} 'int'
  // CHECK-NEXT:         value: Int 2
  // CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'int'
  // CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 0 index 0 ... Is
  // CHECK-NEXT:           IntegerLiteral {{.*}} 'int' 2
  // CHECK-NEXT:       ConstantExpr {{.*}} 'int'
  // CHECK-NEXT:         value: Int 3
  // CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'int'
  // CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 0 index 0 ... Is
  // CHECK-NEXT:           IntegerLiteral {{.*}} 'int' 3
  // CHECK-NEXT:   FunctionDecl {{.*}} used FunctionTemplate2 'void ()'
  // CHECK-NEXT:     TemplateArgument pack
  // CHECK-NEXT:       TemplateArgument integral 1
  // CHECK-NEXT:       TemplateArgument integral 2
  // CHECK-NEXT:     CompoundStmt
  FunctionTemplate2<1, 2, 3>();
  FunctionTemplate2<1, 2>(); // expected-note {{in instantiation of function template specialization 'FunctionTemplate2<1, 2>' requested here}}

  // CHECK:      FunctionTemplateDecl {{.*}} FunctionTemplate3
  // CHECK-NEXT:   NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Names
  // CHECK-NEXT:   FunctionDecl {{.*}} FunctionTemplate3 'void ()'
  // CHECK-NEXT:     CompoundStmt
  // CHECK-NEXT:     SYCLAddIRAttributesFunctionAttr
  // CHECK-NEXT:       PackExpansionExpr {{.*}} '<dependent type>'
  // CHECK-NEXT:         DeclRefExpr {{.*}} 'const char *' NonTypeTemplateParm {{.*}} 'Names' 'const char *'
  // CHECK-NEXT:       ConstantExpr {{.*}} 'int'
  // CHECK-NEXT:         value: Int 1
  // CHECK-NEXT:         IntegerLiteral {{.*}} 'int' 1
  // CHECK-NEXT:       ConstantExpr {{.*}} 'int'
  // CHECK-NEXT:         value: Int 2
  // CHECK-NEXT:         IntegerLiteral {{.*}} 'int' 2
  // CHECK-NEXT:       ConstantExpr {{.*}} 'int'
  // CHECK-NEXT:         value: Int 3
  // CHECK-NEXT:         IntegerLiteral {{.*}} 'int' 3
  // CHECK-NEXT:   FunctionDecl {{.*}} used FunctionTemplate3 'void ()'
  // CHECK-NEXT:     TemplateArgument pack
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrName1' 'const char[6]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrName2' 'const char[6]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrName3' 'const char[6]'
  // CHECK-NEXT:     CompoundStmt
  // CHECK-NEXT:     SYCLAddIRAttributesFunctionAttr
  // CHECK-NEXT:       ConstantExpr
  // CHECK-NEXT:         value: LValue
  // CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Names
  // CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:             DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName1' 'const char[6]'
  // CHECK-NEXT:       ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:         value: LValue
  // CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Names
  // CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:             DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName2' 'const char[6]'
  // CHECK-NEXT:       ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:         value: LValue
  // CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Names
  // CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:             DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName3' 'const char[6]'
  // CHECK-NEXT:       ConstantExpr {{.*}} 'int'
  // CHECK-NEXT:         value: Int 1
  // CHECK-NEXT:         IntegerLiteral {{.*}} 'int' 1
  // CHECK-NEXT:       ConstantExpr {{.*}} 'int'
  // CHECK-NEXT:         value: Int 2
  // CHECK-NEXT:         IntegerLiteral {{.*}} 'int' 2
  // CHECK-NEXT:       ConstantExpr {{.*}} 'int'
  // CHECK-NEXT:         value: Int 3
  // CHECK-NEXT:         IntegerLiteral {{.*}} 'int' 3
  // CHECK-NEXT:   FunctionDecl {{.*}} used FunctionTemplate3 'void ()'
  // CHECK-NEXT:     TemplateArgument pack
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrName1' 'const char[6]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrName2' 'const char[6]'
  // CHECK-NEXT:     CompoundStmt
  FunctionTemplate3<AttrName1, AttrName2, AttrName3>();
  FunctionTemplate3<AttrName1, AttrName2>(); // expected-note {{in instantiation of function template specialization 'FunctionTemplate3<AttrName1, AttrName2>' requested here}}

  // CHECK:      FunctionTemplateDecl {{.*}} FunctionTemplate4
  // CHECK-NEXT:   NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Names
  // CHECK-NEXT:   FunctionDecl {{.*}} FunctionTemplate4 'void ()'
  // CHECK-NEXT:     CompoundStmt
  // CHECK-NEXT:     SYCLAddIRAttributesFunctionAttr
  // CHECK-NEXT:       InitListExpr {{.*}} 'void'
  // CHECK-NEXT:         StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
  // CHECK-NEXT:         StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
  // CHECK-NEXT:       PackExpansionExpr {{.*}} '<dependent type>'
  // CHECK-NEXT:         DeclRefExpr {{.*}} 'const char *' NonTypeTemplateParm {{.*}} 'Names' 'const char *'
  // CHECK-NEXT:       ConstantExpr {{.*}} 'int'
  // CHECK-NEXT:         value: Int 1
  // CHECK-NEXT:         IntegerLiteral {{.*}} 'int' 1
  // CHECK-NEXT:       ConstantExpr {{.*}} 'int'
  // CHECK-NEXT:         value: Int 2
  // CHECK-NEXT:         IntegerLiteral {{.*}} 'int' 2
  // CHECK-NEXT:       ConstantExpr {{.*}} 'int'
  // CHECK-NEXT:         value: Int 3
  // CHECK-NEXT:         IntegerLiteral {{.*}} 'int' 3
  // CHECK-NEXT:   FunctionDecl {{.*}} used FunctionTemplate4 'void ()'
  // CHECK-NEXT:     TemplateArgument pack
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrName1' 'const char[6]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrName2' 'const char[6]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrName3' 'const char[6]'
  // CHECK-NEXT:     CompoundStmt
  // CHECK-NEXT:     SYCLAddIRAttributesFunctionAttr
  // CHECK-NEXT:       InitListExpr {{.*}} 'void'
  // CHECK-NEXT:         StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
  // CHECK-NEXT:         StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
  // CHECK-NEXT:       ConstantExpr
  // CHECK-NEXT:         value: LValue
  // CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Names
  // CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:             DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName1' 'const char[6]'
  // CHECK-NEXT:       ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:         value: LValue
  // CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Names
  // CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:             DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName2' 'const char[6]'
  // CHECK-NEXT:       ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:         value: LValue
  // CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Names
  // CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:             DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName3' 'const char[6]'
  // CHECK-NEXT:       ConstantExpr {{.*}} 'int'
  // CHECK-NEXT:         value: Int 1
  // CHECK-NEXT:         IntegerLiteral {{.*}} 'int' 1
  // CHECK-NEXT:       ConstantExpr {{.*}} 'int'
  // CHECK-NEXT:         value: Int 2
  // CHECK-NEXT:         IntegerLiteral {{.*}} 'int' 2
  // CHECK-NEXT:       ConstantExpr {{.*}} 'int'
  // CHECK-NEXT:         value: Int 3
  // CHECK-NEXT:         IntegerLiteral {{.*}} 'int' 3
  // CHECK-NEXT:   FunctionDecl {{.*}} used FunctionTemplate4 'void ()'
  // CHECK-NEXT:     TemplateArgument pack
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrName1' 'const char[6]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrName2' 'const char[6]'
  // CHECK-NEXT:     CompoundStmt
  FunctionTemplate4<AttrName1, AttrName2, AttrName3>();
  FunctionTemplate4<AttrName1, AttrName2>(); // expected-note {{in instantiation of function template specialization 'FunctionTemplate4<AttrName1, AttrName2>' requested here}}

  // CHECK:      FunctionTemplateDecl {{.*}} FunctionTemplate5
  // CHECK-NEXT:   NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:   FunctionDecl {{.*}} FunctionTemplate5 'void ()'
  // CHECK-NEXT:     CompoundStmt
  // CHECK-NEXT:     SYCLAddIRAttributesFunctionAttr
  // CHECK-NEXT:       PackExpansionExpr {{.*}} '<dependent type>'
  // CHECK-NEXT:         DeclRefExpr {{.*}} 'const char *' NonTypeTemplateParm {{.*}} 'Strs' 'const char *'
  // CHECK-NEXT:   FunctionDecl {{.*}} used FunctionTemplate5 'void ()'
  // CHECK-NEXT:     TemplateArgument pack
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrName1' 'const char[6]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrVal1' 'const char[5]'
  // CHECK-NEXT:     CompoundStmt
  // CHECK-NEXT:     SYCLAddIRAttributesFunctionAttr
  // CHECK-NEXT:       ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:         value: LValue
  // CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:             DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName1' 'const char[6]'
  // CHECK-NEXT:       ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:         value: LValue
  // CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:             DeclRefExpr {{.*}} 'const char[5]' lvalue Var {{.*}} 'AttrVal1' 'const char[5]'
  // CHECK-NEXT:   FunctionDecl {{.*}} used FunctionTemplate5 'void ()'
  // CHECK-NEXT:     TemplateArgument pack
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrName1' 'const char[6]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrName2' 'const char[6]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrVal1' 'const char[5]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrVal2' 'const char[5]'
  // CHECK-NEXT:     CompoundStmt
  // CHECK-NEXT:     SYCLAddIRAttributesFunctionAttr
  // CHECK-NEXT:       ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:         value: LValue
  // CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:             DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName1' 'const char[6]'
  // CHECK-NEXT:       ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:         value: LValue
  // CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:             DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName2' 'const char[6]'
  // CHECK-NEXT:       ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:         value: LValue
  // CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:             DeclRefExpr {{.*}} 'const char[5]' lvalue Var {{.*}} 'AttrVal1' 'const char[5]'
  // CHECK-NEXT:       ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:         value: LValue
  // CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:             DeclRefExpr {{.*}} 'const char[5]' lvalue Var {{.*}} 'AttrVal2' 'const char[5]'
  // CHECK-NEXT:   FunctionDecl {{.*}} used FunctionTemplate5 'void ()'
  // CHECK-NEXT:     TemplateArgument pack
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrName1' 'const char[6]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrName2' 'const char[6]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrName3' 'const char[6]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrVal1' 'const char[5]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrVal2' 'const char[5]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrVal3' 'const char[5]'
  // CHECK-NEXT:     CompoundStmt
  // CHECK-NEXT:     SYCLAddIRAttributesFunctionAttr
  // CHECK-NEXT:       ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:         value: LValue
  // CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:             DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName1' 'const char[6]'
  // CHECK-NEXT:       ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:         value: LValue
  // CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:             DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName2' 'const char[6]'
  // CHECK-NEXT:       ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:         value: LValue
  // CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:             DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName3' 'const char[6]'
  // CHECK-NEXT:       ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:         value: LValue
  // CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:             DeclRefExpr {{.*}} 'const char[5]' lvalue Var {{.*}} 'AttrVal1' 'const char[5]'
  // CHECK-NEXT:       ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:         value: LValue
  // CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:             DeclRefExpr {{.*}} 'const char[5]' lvalue Var {{.*}} 'AttrVal2' 'const char[5]'
  // CHECK-NEXT:       ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:         value: LValue
  // CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:             DeclRefExpr {{.*}} 'const char[5]' lvalue Var {{.*}} 'AttrVal3' 'const char[5]'
  // CHECK-NEXT:   FunctionDecl {{.*}} used FunctionTemplate5 'void ()'
  // CHECK-NEXT:     TemplateArgument pack
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:        Var {{.*}} 'AttrName1' 'const char[6]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrName2' 'const char[6]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrName3' 'const char[6]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrVal1' 'const char[5]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrVal2' 'const char[5]'
  // CHECK-NEXT:     CompoundStmt
  FunctionTemplate5<AttrName1, AttrVal1>();
  FunctionTemplate5<AttrName1, AttrName2, AttrVal1, AttrVal2>();
  FunctionTemplate5<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2, AttrVal3>();
  FunctionTemplate5<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2>(); // expected-note {{in instantiation of function template specialization 'FunctionTemplate5<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2>' requested here}}

  // CHECK:      FunctionTemplateDecl {{.*}} FunctionTemplate6
  // CHECK-NEXT:   NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:   FunctionDecl {{.*}} FunctionTemplate6 'void ()'
  // CHECK-NEXT:     CompoundStmt
  // CHECK-NEXT:     SYCLAddIRAttributesFunctionAttr
  // CHECK-NEXT:       InitListExpr {{.*}} 'void'
  // CHECK-NEXT:         StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
  // CHECK-NEXT:         StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
  // CHECK-NEXT:       PackExpansionExpr {{.*}} '<dependent type>'
  // CHECK-NEXT:         DeclRefExpr {{.*}} 'const char *' NonTypeTemplateParm {{.*}} 'Strs' 'const char *'
  // CHECK-NEXT:   FunctionDecl {{.*}} used FunctionTemplate6 'void ()'
  // CHECK-NEXT:     TemplateArgument pack
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrName1' 'const char[6]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrVal1' 'const char[5]'
  // CHECK-NEXT:     CompoundStmt
  // CHECK-NEXT:     SYCLAddIRAttributesFunctionAttr
  // CHECK-NEXT:       InitListExpr {{.*}} 'void'
  // CHECK-NEXT:         StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
  // CHECK-NEXT:         StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
  // CHECK-NEXT:       ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:         value: LValue
  // CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:             DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName1' 'const char[6]'
  // CHECK-NEXT:       ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:         value: LValue
  // CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:             DeclRefExpr {{.*}} 'const char[5]' lvalue Var {{.*}} 'AttrVal1' 'const char[5]'
  // CHECK-NEXT:   FunctionDecl {{.*}} used FunctionTemplate6 'void ()'
  // CHECK-NEXT:     TemplateArgument pack
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrName1' 'const char[6]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrName2' 'const char[6]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrVal1' 'const char[5]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrVal2' 'const char[5]'
  // CHECK-NEXT:     CompoundStmt
  // CHECK-NEXT:     SYCLAddIRAttributesFunctionAttr
  // CHECK-NEXT:       InitListExpr {{.*}} 'void'
  // CHECK-NEXT:         StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
  // CHECK-NEXT:         StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
  // CHECK-NEXT:       ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:         value: LValue
  // CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:             DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName1' 'const char[6]'
  // CHECK-NEXT:       ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:         value: LValue
  // CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:             DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName2' 'const char[6]'
  // CHECK-NEXT:       ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:         value: LValue
  // CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:             DeclRefExpr {{.*}} 'const char[5]' lvalue Var {{.*}} 'AttrVal1' 'const char[5]'
  // CHECK-NEXT:       ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:         value: LValue
  // CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:             DeclRefExpr {{.*}} 'const char[5]' lvalue Var {{.*}} 'AttrVal2' 'const char[5]'
  // CHECK-NEXT:   FunctionDecl {{.*}} used FunctionTemplate6 'void ()'
  // CHECK-NEXT:     TemplateArgument pack
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrName1' 'const char[6]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrName2' 'const char[6]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrName3' 'const char[6]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrVal1' 'const char[5]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrVal2' 'const char[5]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrVal3' 'const char[5]'
  // CHECK-NEXT:     CompoundStmt
  // CHECK-NEXT:     SYCLAddIRAttributesFunctionAttr
  // CHECK-NEXT:       InitListExpr {{.*}} 'void'
  // CHECK-NEXT:         StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
  // CHECK-NEXT:         StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
  // CHECK-NEXT:       ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:         value: LValue
  // CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:             DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName1' 'const char[6]'
  // CHECK-NEXT:       ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:         value: LValue
  // CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:             DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName2' 'const char[6]'
  // CHECK-NEXT:       ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:         value: LValue
  // CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:             DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName3' 'const char[6]'
  // CHECK-NEXT:       ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:         value: LValue
  // CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:             DeclRefExpr {{.*}} 'const char[5]' lvalue Var {{.*}} 'AttrVal1' 'const char[5]'
  // CHECK-NEXT:       ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:         value: LValue
  // CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:             DeclRefExpr {{.*}} 'const char[5]' lvalue Var {{.*}} 'AttrVal2' 'const char[5]'
  // CHECK-NEXT:       ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:         value: LValue
  // CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:             DeclRefExpr {{.*}} 'const char[5]' lvalue Var {{.*}} 'AttrVal3' 'const char[5]'
  // CHECK-NEXT:   FunctionDecl {{.*}} used FunctionTemplate6 'void ()'
  // CHECK-NEXT:     TemplateArgument pack
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:        Var {{.*}} 'AttrName1' 'const char[6]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrName2' 'const char[6]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrName3' 'const char[6]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrVal1' 'const char[5]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrVal2' 'const char[5]'
  // CHECK-NEXT:     CompoundStmt
  FunctionTemplate6<AttrName1, AttrVal1>();
  FunctionTemplate6<AttrName1, AttrName2, AttrVal1, AttrVal2>();
  FunctionTemplate6<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2, AttrVal3>();
  FunctionTemplate6<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2>(); // expected-note {{in instantiation of function template specialization 'FunctionTemplate6<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2>' requested here}}
}

template <int... Is> struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", "Attr2", "Attr3", Is...)]] GlobalVarStructTemplate1{};                     // expected-error {{attribute 'add_ir_attributes_global_variable' must have an attribute value for each attribute name}}
template <int... Is> struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, "Attr1", "Attr2", "Attr3", Is...)]] GlobalVarStructTemplate2{}; // expected-error {{attribute 'add_ir_attributes_global_variable' must have an attribute value for each attribute name}}
template <const char *...Names> struct [[__sycl_detail__::add_ir_attributes_global_variable(Names..., 1, 2, 3)]] GlobalVarStructTemplate3{};                         // expected-error {{attribute 'add_ir_attributes_global_variable' must have an attribute value for each attribute name}}
template <const char *...Names> struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, Names..., 1, 2, 3)]] GlobalVarStructTemplate4{};     // expected-error {{attribute 'add_ir_attributes_global_variable' must have an attribute value for each attribute name}}
template <const char *...Strs> struct [[__sycl_detail__::add_ir_attributes_global_variable(Strs...)]] GlobalVarStructTemplate5{};                                    // expected-error {{attribute 'add_ir_attributes_global_variable' must have an attribute value for each attribute name}}
template <const char *...Strs> struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, Strs...)]] GlobalVarStructTemplate6{};                // expected-error {{attribute 'add_ir_attributes_global_variable' must have an attribute value for each attribute name}}

// CHECK:      ClassTemplateDecl {{.*}} GlobalVarStructTemplate1
// CHECK-NEXT:   NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 0 index 0 ... Is
// CHECK-NEXT:   CXXRecordDecl {{.*}} struct GlobalVarStructTemplate1 definition
// CHECK-NEXT:     DefinitionData
// CHECK-NEXT:       DefaultConstructor
// CHECK-NEXT:       CopyConstructor
// CHECK-NEXT:       MoveConstructor
// CHECK-NEXT:       CopyAssignment
// CHECK-NEXT:       MoveAssignment
// CHECK-NEXT:       Destructor
// CHECK-NEXT:     SYCLAddIRAttributesGlobalVariableAttr
// CHECK-NEXT:       ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:         value: LValue
// CHECK-NEXT:         StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
// CHECK-NEXT:       ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:         value: LValue
// CHECK-NEXT:         StringLiteral {{.*}} 'const char[6]' lvalue "Attr2"
// CHECK-NEXT:       ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:         value: LValue
// CHECK-NEXT:         StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
// CHECK-NEXT:       PackExpansionExpr {{.*}} '<dependent type>'
// CHECK-NEXT:         DeclRefExpr {{.*}} 'int' NonTypeTemplateParm {{.*}} 'Is' 'int'
// CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct GlobalVarStructTemplate1
// CHECK-NEXT:   ClassTemplateSpecializationDecl {{.*}} struct GlobalVarStructTemplate1 definition
// CHECK-NEXT:     DefinitionData
// CHECK-NEXT:       DefaultConstructor
// CHECK-NEXT:       CopyConstructor
// CHECK-NEXT:       MoveConstructor
// CHECK-NEXT:       CopyAssignment
// CHECK-NEXT:       MoveAssignment
// CHECK-NEXT:       Destructor
// CHECK-NEXT:     TemplateArgument pack
// CHECK-NEXT:       TemplateArgument integral 1
// CHECK-NEXT:       TemplateArgument integral 2
// CHECK-NEXT:       TemplateArgument integral 3
// CHECK-NEXT:     SYCLAddIRAttributesGlobalVariableAttr
// CHECK-NEXT:       ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:         value: LValue
// CHECK-NEXT:         StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
// CHECK-NEXT:       ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:         value: LValue
// CHECK-NEXT:         StringLiteral {{.*}} 'const char[6]' lvalue "Attr2"
// CHECK-NEXT:       ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:         value: LValue
// CHECK-NEXT:         StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
// CHECK-NEXT:       ConstantExpr {{.*}} 'int'
// CHECK-NEXT:         value: Int 1
// CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'int'
// CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 0 index 0 ... Is
// CHECK-NEXT:           IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT:       ConstantExpr {{.*}} 'int'
// CHECK-NEXT:         value: Int 2
// CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'int'
// CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 0 index 0 ... Is
// CHECK-NEXT:           IntegerLiteral {{.*}} 'int' 2
// CHECK-NEXT:       ConstantExpr {{.*}} 'int'
// CHECK-NEXT:         value: Int 3
// CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'int'
// CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 0 index 0 ... Is
// CHECK-NEXT:           IntegerLiteral {{.*}} 'int' 3
// CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct GlobalVarStructTemplate1
// CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit used constexpr GlobalVarStructTemplate1 'void () noexcept'
// CHECK-NEXT:       CompoundStmt {{.*}}
// CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr GlobalVarStructTemplate1 'void (const GlobalVarStructTemplate1<1, 2, 3> &)'
// CHECK-NEXT:       ParmVarDecl {{.*}} 'const GlobalVarStructTemplate1<1, 2, 3> &'
// CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr GlobalVarStructTemplate1 'void (GlobalVarStructTemplate1<1, 2, 3> &&)'
// CHECK-NEXT:       ParmVarDecl {{.*}} 'GlobalVarStructTemplate1<1, 2, 3> &&'
// CHECK-NEXT:   ClassTemplateSpecializationDecl {{.*}} struct GlobalVarStructTemplate1 definition
// CHECK-NEXT:     DefinitionData
// CHECK-NEXT:       DefaultConstructor
// CHECK-NEXT:       CopyConstructor
// CHECK-NEXT:       MoveConstructor
// CHECK-NEXT:       CopyAssignment
// CHECK-NEXT:       MoveAssignment
// CHECK-NEXT:       Destructor
// CHECK-NEXT:     TemplateArgument pack
// CHECK-NEXT:       TemplateArgument integral 1
// CHECK-NEXT:       TemplateArgument integral 2
// CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct GlobalVarStructTemplate1
// CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit used constexpr GlobalVarStructTemplate1 'void () noexcept'
// CHECK-NEXT:       CompoundStmt {{.*}}
// CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr GlobalVarStructTemplate1 'void (const GlobalVarStructTemplate1<1, 2> &)'
// CHECK-NEXT:       ParmVarDecl {{.*}} 'const GlobalVarStructTemplate1<1, 2> &'
// CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr GlobalVarStructTemplate1 'void (GlobalVarStructTemplate1<1, 2> &&)'
// CHECK-NEXT:       ParmVarDecl {{.*}} 'GlobalVarStructTemplate1<1, 2> &&'
GlobalVarStructTemplate1<1, 2, 3> InstantiatedGV1;
GlobalVarStructTemplate1<1, 2> InstantiatedGV2; // expected-note {{in instantiation of template class 'GlobalVarStructTemplate1<1, 2>' requested here}}

// CHECK:      ClassTemplateDecl {{.*}} GlobalVarStructTemplate2
// CHECK-NEXT:   NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 0 index 0 ... Is
// CHECK-NEXT:   CXXRecordDecl {{.*}} struct GlobalVarStructTemplate2 definition
// CHECK-NEXT:     DefinitionData
// CHECK-NEXT:       DefaultConstructor
// CHECK-NEXT:       CopyConstructor
// CHECK-NEXT:       MoveConstructor
// CHECK-NEXT:       CopyAssignment
// CHECK-NEXT:       MoveAssignment
// CHECK-NEXT:       Destructor
// CHECK-NEXT:     SYCLAddIRAttributesGlobalVariableAttr
// CHECK-NEXT:       InitListExpr {{.*}} 'void'
// CHECK-NEXT:         StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
// CHECK-NEXT:         StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
// CHECK-NEXT:       ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:         value: LValue
// CHECK-NEXT:         StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
// CHECK-NEXT:       ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:         value: LValue
// CHECK-NEXT:         StringLiteral {{.*}} 'const char[6]' lvalue "Attr2"
// CHECK-NEXT:       ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:         value: LValue
// CHECK-NEXT:         StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
// CHECK-NEXT:       PackExpansionExpr {{.*}} '<dependent type>'
// CHECK-NEXT:         DeclRefExpr {{.*}} 'int' NonTypeTemplateParm {{.*}} 'Is' 'int'
// CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct GlobalVarStructTemplate2
// CHECK-NEXT:   ClassTemplateSpecializationDecl {{.*}} struct GlobalVarStructTemplate2 definition
// CHECK-NEXT:     DefinitionData
// CHECK-NEXT:       DefaultConstructor
// CHECK-NEXT:       CopyConstructor
// CHECK-NEXT:       MoveConstructor
// CHECK-NEXT:       CopyAssignment
// CHECK-NEXT:       MoveAssignment
// CHECK-NEXT:       Destructor
// CHECK-NEXT:     TemplateArgument pack
// CHECK-NEXT:       TemplateArgument integral 1
// CHECK-NEXT:       TemplateArgument integral 2
// CHECK-NEXT:       TemplateArgument integral 3
// CHECK-NEXT:     SYCLAddIRAttributesGlobalVariableAttr
// CHECK-NEXT:       InitListExpr {{.*}} 'void'
// CHECK-NEXT:         StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
// CHECK-NEXT:         StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
// CHECK-NEXT:       ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:         value: LValue
// CHECK-NEXT:         StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
// CHECK-NEXT:       ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:         value: LValue
// CHECK-NEXT:         StringLiteral {{.*}} 'const char[6]' lvalue "Attr2"
// CHECK-NEXT:       ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:         value: LValue
// CHECK-NEXT:         StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
// CHECK-NEXT:       ConstantExpr {{.*}} 'int'
// CHECK-NEXT:         value: Int 1
// CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'int'
// CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 0 index 0 ... Is
// CHECK-NEXT:           IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT:       ConstantExpr {{.*}} 'int'
// CHECK-NEXT:         value: Int 2
// CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'int'
// CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 0 index 0 ... Is
// CHECK-NEXT:           IntegerLiteral {{.*}} 'int' 2
// CHECK-NEXT:       ConstantExpr {{.*}} 'int'
// CHECK-NEXT:         value: Int 3
// CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'int'
// CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 0 index 0 ... Is
// CHECK-NEXT:           IntegerLiteral {{.*}} 'int' 3
// CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct GlobalVarStructTemplate2
// CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit used constexpr GlobalVarStructTemplate2 'void () noexcept'
// CHECK-NEXT:       CompoundStmt {{.*}}
// CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr GlobalVarStructTemplate2 'void (const GlobalVarStructTemplate2<1, 2, 3> &)'
// CHECK-NEXT:       ParmVarDecl {{.*}} 'const GlobalVarStructTemplate2<1, 2, 3> &'
// CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr GlobalVarStructTemplate2 'void (GlobalVarStructTemplate2<1, 2, 3> &&)'
// CHECK-NEXT:       ParmVarDecl {{.*}} 'GlobalVarStructTemplate2<1, 2, 3> &&'
// CHECK-NEXT:   ClassTemplateSpecializationDecl {{.*}} struct GlobalVarStructTemplate2 definition
// CHECK-NEXT:     DefinitionData
// CHECK-NEXT:       DefaultConstructor
// CHECK-NEXT:       CopyConstructor
// CHECK-NEXT:       MoveConstructor
// CHECK-NEXT:       CopyAssignment
// CHECK-NEXT:       MoveAssignment
// CHECK-NEXT:       Destructor
// CHECK-NEXT:     TemplateArgument pack
// CHECK-NEXT:       TemplateArgument integral 1
// CHECK-NEXT:       TemplateArgument integral 2
// CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct GlobalVarStructTemplate2
// CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit used constexpr GlobalVarStructTemplate2 'void () noexcept'
// CHECK-NEXT:       CompoundStmt {{.*}}
// CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr GlobalVarStructTemplate2 'void (const GlobalVarStructTemplate2<1, 2> &)'
// CHECK-NEXT:       ParmVarDecl {{.*}} 'const GlobalVarStructTemplate2<1, 2> &'
// CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr GlobalVarStructTemplate2 'void (GlobalVarStructTemplate2<1, 2> &&)'
// CHECK-NEXT:       ParmVarDecl {{.*}} 'GlobalVarStructTemplate2<1, 2> &&'
GlobalVarStructTemplate2<1, 2, 3> InstantiatedGV3;
GlobalVarStructTemplate2<1, 2> InstantiatedGV4; // expected-note {{in instantiation of template class 'GlobalVarStructTemplate2<1, 2>' requested here}}

// CHECK:      ClassTemplateDecl {{.*}} GlobalVarStructTemplate3
// CHECK-NEXT:   NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Names
// CHECK-NEXT:   CXXRecordDecl {{.*}} struct GlobalVarStructTemplate3 definition
// CHECK-NEXT:     DefinitionData
// CHECK-NEXT:       DefaultConstructor
// CHECK-NEXT:       CopyConstructor
// CHECK-NEXT:       MoveConstructor
// CHECK-NEXT:       CopyAssignment
// CHECK-NEXT:       MoveAssignment
// CHECK-NEXT:       Destructor
// CHECK-NEXT:     SYCLAddIRAttributesGlobalVariableAttr
// CHECK-NEXT:       PackExpansionExpr {{.*}} '<dependent type>'
// CHECK-NEXT:         DeclRefExpr {{.*}} 'const char *' NonTypeTemplateParm {{.*}} 'Names' 'const char *'
// CHECK-NEXT:       ConstantExpr {{.*}} 'int'
// CHECK-NEXT:         value: Int 1
// CHECK-NEXT:         IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT:       ConstantExpr {{.*}} 'int'
// CHECK-NEXT:         value: Int 2
// CHECK-NEXT:         IntegerLiteral {{.*}} 'int' 2
// CHECK-NEXT:       ConstantExpr {{.*}} 'int'
// CHECK-NEXT:         value: Int 3
// CHECK-NEXT:         IntegerLiteral {{.*}} 'int' 3
// CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct GlobalVarStructTemplate3
// CHECK-NEXT:   ClassTemplateSpecializationDecl {{.*}} struct GlobalVarStructTemplate3 definition
// CHECK-NEXT:     DefinitionData
// CHECK-NEXT:       DefaultConstructor
// CHECK-NEXT:       CopyConstructor
// CHECK-NEXT:       MoveConstructor
// CHECK-NEXT:       CopyAssignment
// CHECK-NEXT:       MoveAssignment
// CHECK-NEXT:       Destructor
// CHECK-NEXT:     TemplateArgument pack
// CHECK-NEXT:       TemplateArgument decl
// CHECK-NEXT:         Var {{.*}} 'AttrName1' 'const char[6]'
// CHECK-NEXT:       TemplateArgument decl
// CHECK-NEXT:         Var {{.*}} 'AttrName2' 'const char[6]'
// CHECK-NEXT:       TemplateArgument decl
// CHECK-NEXT:         Var {{.*}} 'AttrName3' 'const char[6]'
// CHECK-NEXT:     SYCLAddIRAttributesGlobalVariableAttr
// CHECK-NEXT:       ConstantExpr {{.*}} 'const char *'
// CHECK-NEXT:         value: LValue
// CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
// CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Names
// CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
// CHECK-NEXT:             DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName1' 'const char[6]'
// CHECK-NEXT:       ConstantExpr {{.*}} 'const char *'
// CHECK-NEXT:         value: LValue
// CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
// CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Names
// CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
// CHECK-NEXT:             DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName2' 'const char[6]'
// CHECK-NEXT:       ConstantExpr {{.*}} 'const char *'
// CHECK-NEXT:         value: LValue
// CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
// CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Names
// CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
// CHECK-NEXT:             DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName3' 'const char[6]'
// CHECK-NEXT:       ConstantExpr {{.*}} 'int'
// CHECK-NEXT:         value: Int 1
// CHECK-NEXT:         IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT:       ConstantExpr {{.*}} 'int'
// CHECK-NEXT:         value: Int 2
// CHECK-NEXT:         IntegerLiteral {{.*}} 'int' 2
// CHECK-NEXT:       ConstantExpr {{.*}} 'int'
// CHECK-NEXT:         value: Int 3
// CHECK-NEXT:         IntegerLiteral {{.*}} 'int' 3
// CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct GlobalVarStructTemplate3
// CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit used constexpr GlobalVarStructTemplate3 'void () noexcept' inline default trivial
// CHECK-NEXT:       CompoundStmt
// CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr GlobalVarStructTemplate3 'void (const GlobalVarStructTemplate3<AttrName1, AttrName2, AttrName3> &)'
// CHECK-NEXT:       ParmVarDecl {{.*}} 'const GlobalVarStructTemplate3<AttrName1, AttrName2, AttrName3> &'
// CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr GlobalVarStructTemplate3 'void (GlobalVarStructTemplate3<AttrName1, AttrName2, AttrName3> &&)'
// CHECK-NEXT:       ParmVarDecl {{.*}} 'GlobalVarStructTemplate3<AttrName1, AttrName2, AttrName3> &&'
// CHECK-NEXT:   ClassTemplateSpecializationDecl {{.*}} struct GlobalVarStructTemplate3 definition
// CHECK-NEXT:     DefinitionData
// CHECK-NEXT:       DefaultConstructor
// CHECK-NEXT:       CopyConstructor
// CHECK-NEXT:       MoveConstructor
// CHECK-NEXT:       CopyAssignment
// CHECK-NEXT:       MoveAssignment
// CHECK-NEXT:       Destructor
// CHECK-NEXT:     TemplateArgument pack
// CHECK-NEXT:       TemplateArgument decl
// CHECK-NEXT:         Var {{.*}} 'AttrName1' 'const char[6]'
// CHECK-NEXT:       TemplateArgument decl
// CHECK-NEXT:         Var {{.*}} 'AttrName2' 'const char[6]'
// CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct GlobalVarStructTemplate3
// CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit used constexpr GlobalVarStructTemplate3 'void () noexcept' inline default trivial
// CHECK-NEXT:       CompoundStmt
// CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr GlobalVarStructTemplate3 'void (const GlobalVarStructTemplate3<AttrName1, AttrName2> &)'
// CHECK-NEXT:       ParmVarDecl {{.*}} 'const GlobalVarStructTemplate3<AttrName1, AttrName2> &'
// CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr GlobalVarStructTemplate3 'void (GlobalVarStructTemplate3<AttrName1, AttrName2> &&)'
// CHECK-NEXT:       ParmVarDecl {{.*}} 'GlobalVarStructTemplate3<AttrName1, AttrName2> &&'
GlobalVarStructTemplate3<AttrName1, AttrName2, AttrName3> InstantiatedGV5;
GlobalVarStructTemplate3<AttrName1, AttrName2> InstantiatedGV6; // expected-note {{in instantiation of template class 'GlobalVarStructTemplate3<AttrName1, AttrName2>' requested here}}

// CHECK:      ClassTemplateDecl {{.*}} GlobalVarStructTemplate4
// CHECK-NEXT:   NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Names
// CHECK-NEXT:   CXXRecordDecl {{.*}} struct GlobalVarStructTemplate4 definition
// CHECK-NEXT:     DefinitionData
// CHECK-NEXT:       DefaultConstructor
// CHECK-NEXT:       CopyConstructor
// CHECK-NEXT:       MoveConstructor
// CHECK-NEXT:       CopyAssignment
// CHECK-NEXT:       MoveAssignment
// CHECK-NEXT:       Destructor
// CHECK-NEXT:     SYCLAddIRAttributesGlobalVariableAttr
// CHECK-NEXT:       InitListExpr {{.*}} 'void'
// CHECK-NEXT:         StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
// CHECK-NEXT:         StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
// CHECK-NEXT:       PackExpansionExpr {{.*}} '<dependent type>'
// CHECK-NEXT:         DeclRefExpr {{.*}} 'const char *' NonTypeTemplateParm {{.*}} 'Names' 'const char *'
// CHECK-NEXT:       ConstantExpr {{.*}} 'int'
// CHECK-NEXT:         value: Int 1
// CHECK-NEXT:         IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT:       ConstantExpr {{.*}} 'int'
// CHECK-NEXT:         value: Int 2
// CHECK-NEXT:         IntegerLiteral {{.*}} 'int' 2
// CHECK-NEXT:       ConstantExpr {{.*}} 'int'
// CHECK-NEXT:         value: Int 3
// CHECK-NEXT:         IntegerLiteral {{.*}} 'int' 3
// CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct GlobalVarStructTemplate4
// CHECK-NEXT:   ClassTemplateSpecializationDecl {{.*}} struct GlobalVarStructTemplate4 definition
// CHECK-NEXT:     DefinitionData
// CHECK-NEXT:       DefaultConstructor
// CHECK-NEXT:       CopyConstructor
// CHECK-NEXT:       MoveConstructor
// CHECK-NEXT:       CopyAssignment
// CHECK-NEXT:       MoveAssignment
// CHECK-NEXT:       Destructor
// CHECK-NEXT:     TemplateArgument pack
// CHECK-NEXT:       TemplateArgument decl
// CHECK-NEXT:         Var {{.*}} 'AttrName1' 'const char[6]'
// CHECK-NEXT:       TemplateArgument decl
// CHECK-NEXT:         Var {{.*}} 'AttrName2' 'const char[6]'
// CHECK-NEXT:       TemplateArgument decl
// CHECK-NEXT:         Var {{.*}} 'AttrName3' 'const char[6]'
// CHECK-NEXT:     SYCLAddIRAttributesGlobalVariableAttr
// CHECK-NEXT:       InitListExpr {{.*}} 'void'
// CHECK-NEXT:         StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
// CHECK-NEXT:         StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
// CHECK-NEXT:       ConstantExpr {{.*}} 'const char *'
// CHECK-NEXT:         value: LValue
// CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
// CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Names
// CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
// CHECK-NEXT:             DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName1' 'const char[6]'
// CHECK-NEXT:       ConstantExpr {{.*}} 'const char *'
// CHECK-NEXT:         value: LValue
// CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
// CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Names
// CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
// CHECK-NEXT:             DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName2' 'const char[6]'
// CHECK-NEXT:       ConstantExpr {{.*}} 'const char *'
// CHECK-NEXT:         value: LValue
// CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
// CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Names
// CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
// CHECK-NEXT:             DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName3' 'const char[6]'
// CHECK-NEXT:       ConstantExpr {{.*}} 'int'
// CHECK-NEXT:         value: Int 1
// CHECK-NEXT:         IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT:       ConstantExpr {{.*}} 'int'
// CHECK-NEXT:         value: Int 2
// CHECK-NEXT:         IntegerLiteral {{.*}} 'int' 2
// CHECK-NEXT:       ConstantExpr {{.*}} 'int'
// CHECK-NEXT:         value: Int 3
// CHECK-NEXT:         IntegerLiteral {{.*}} 'int' 3
// CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct GlobalVarStructTemplate4
// CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit used constexpr GlobalVarStructTemplate4 'void () noexcept' inline default trivial
// CHECK-NEXT:       CompoundStmt
// CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr GlobalVarStructTemplate4 'void (const GlobalVarStructTemplate4<AttrName1, AttrName2, AttrName3> &)'
// CHECK-NEXT:       ParmVarDecl {{.*}} 'const GlobalVarStructTemplate4<AttrName1, AttrName2, AttrName3> &'
// CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr GlobalVarStructTemplate4 'void (GlobalVarStructTemplate4<AttrName1, AttrName2, AttrName3> &&)'
// CHECK-NEXT:       ParmVarDecl {{.*}} 'GlobalVarStructTemplate4<AttrName1, AttrName2, AttrName3> &&'
// CHECK-NEXT:   ClassTemplateSpecializationDecl {{.*}} struct GlobalVarStructTemplate4 definition
// CHECK-NEXT:     DefinitionData
// CHECK-NEXT:       DefaultConstructor
// CHECK-NEXT:       CopyConstructor
// CHECK-NEXT:       MoveConstructor
// CHECK-NEXT:       CopyAssignment
// CHECK-NEXT:       MoveAssignment
// CHECK-NEXT:       Destructor
// CHECK-NEXT:     TemplateArgument pack
// CHECK-NEXT:       TemplateArgument decl
// CHECK-NEXT:         Var {{.*}} 'AttrName1' 'const char[6]'
// CHECK-NEXT:       TemplateArgument decl
// CHECK-NEXT:         Var {{.*}} 'AttrName2' 'const char[6]'
// CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct GlobalVarStructTemplate4
// CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit used constexpr GlobalVarStructTemplate4 'void () noexcept' inline default trivial
// CHECK-NEXT:       CompoundStmt
// CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr GlobalVarStructTemplate4 'void (const GlobalVarStructTemplate4<AttrName1, AttrName2> &)'
// CHECK-NEXT:       ParmVarDecl {{.*}} 'const GlobalVarStructTemplate4<AttrName1, AttrName2> &'
// CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr GlobalVarStructTemplate4 'void (GlobalVarStructTemplate4<AttrName1, AttrName2> &&)'
// CHECK-NEXT:       ParmVarDecl {{.*}} 'GlobalVarStructTemplate4<AttrName1, AttrName2> &&'
GlobalVarStructTemplate4<AttrName1, AttrName2, AttrName3> InstantiatedGV7;
GlobalVarStructTemplate4<AttrName1, AttrName2> InstantiatedGV8; // expected-note {{in instantiation of template class 'GlobalVarStructTemplate4<AttrName1, AttrName2>' requested here}}

// CHECK:      ClassTemplateDecl {{.*}} GlobalVarStructTemplate5
// CHECK-NEXT:   NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
// CHECK-NEXT:   CXXRecordDecl {{.*}} struct GlobalVarStructTemplate5 definition
// CHECK-NEXT:     DefinitionData
// CHECK-NEXT:       DefaultConstructor
// CHECK-NEXT:       CopyConstructor
// CHECK-NEXT:       MoveConstructor
// CHECK-NEXT:       CopyAssignment
// CHECK-NEXT:       MoveAssignment
// CHECK-NEXT:       Destructor
// CHECK-NEXT:     SYCLAddIRAttributesGlobalVariableAttr
// CHECK-NEXT:       PackExpansionExpr {{.*}} '<dependent type>'
// CHECK-NEXT:         DeclRefExpr {{.*}} 'const char *' NonTypeTemplateParm {{.*}} 'Strs' 'const char *'
// CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct GlobalVarStructTemplate5
// CHECK-NEXT:   ClassTemplateSpecializationDecl {{.*}} struct GlobalVarStructTemplate5 definition
// CHECK-NEXT:     DefinitionData
// CHECK-NEXT:       DefaultConstructor
// CHECK-NEXT:       CopyConstructor
// CHECK-NEXT:       MoveConstructor
// CHECK-NEXT:       CopyAssignment
// CHECK-NEXT:       MoveAssignment
// CHECK-NEXT:       Destructor
// CHECK-NEXT:     TemplateArgument pack
// CHECK-NEXT:       TemplateArgument decl
// CHECK-NEXT:         Var {{.*}} 'AttrName1' 'const char[6]'
// CHECK-NEXT:       TemplateArgument decl
// CHECK-NEXT:         Var {{.*}} 'AttrVal1' 'const char[5]'
// CHECK-NEXT:     SYCLAddIRAttributesGlobalVariableAttr
// CHECK-NEXT:       ConstantExpr {{.*}} 'const char *'
// CHECK-NEXT:         value: LValue
// CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
// CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
// CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
// CHECK-NEXT:             DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName1' 'const char[6]'
// CHECK-NEXT:       ConstantExpr {{.*}} 'const char *'
// CHECK-NEXT:         value: LValue
// CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
// CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
// CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
// CHECK-NEXT:             DeclRefExpr {{.*}} 'const char[5]' lvalue Var {{.*}} 'AttrVal1' 'const char[5]'
// CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct GlobalVarStructTemplate5
// CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit used constexpr GlobalVarStructTemplate5 'void () noexcept'
// CHECK-NEXT:       CompoundStmt
// CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr GlobalVarStructTemplate5 'void (const GlobalVarStructTemplate5<AttrName1, AttrVal1> &)'
// CHECK-NEXT:       ParmVarDecl {{.*}} 'const GlobalVarStructTemplate5<AttrName1, AttrVal1> &'
// CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr GlobalVarStructTemplate5 'void (GlobalVarStructTemplate5<AttrName1, AttrVal1> &&)'
// CHECK-NEXT:       ParmVarDecl {{.*}} 'GlobalVarStructTemplate5<AttrName1, AttrVal1> &&'
// CHECK-NEXT:   ClassTemplateSpecializationDecl {{.*}} struct GlobalVarStructTemplate5 definition
// CHECK-NEXT:     DefinitionData
// CHECK-NEXT:       DefaultConstructor
// CHECK-NEXT:       CopyConstructor
// CHECK-NEXT:       MoveConstructor
// CHECK-NEXT:       CopyAssignment
// CHECK-NEXT:       MoveAssignment
// CHECK-NEXT:       Destructor
// CHECK-NEXT:     TemplateArgument pack
// CHECK-NEXT:       TemplateArgument decl
// CHECK-NEXT:         Var {{.*}} 'AttrName1' 'const char[6]'
// CHECK-NEXT:       TemplateArgument decl
// CHECK-NEXT:         Var {{.*}} 'AttrName2' 'const char[6]'
// CHECK-NEXT:       TemplateArgument decl
// CHECK-NEXT:         Var {{.*}} 'AttrVal1' 'const char[5]'
// CHECK-NEXT:       TemplateArgument decl
// CHECK-NEXT:         Var {{.*}} 'AttrVal2' 'const char[5]'
// CHECK-NEXT:     SYCLAddIRAttributesGlobalVariableAttr
// CHECK-NEXT:       ConstantExpr {{.*}} 'const char *'
// CHECK-NEXT:         value: LValue
// CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
// CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
// CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
// CHECK-NEXT:             DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName1' 'const char[6]'
// CHECK-NEXT:       ConstantExpr {{.*}} 'const char *'
// CHECK-NEXT:         value: LValue
// CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
// CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
// CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
// CHECK-NEXT:             DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName2' 'const char[6]'
// CHECK-NEXT:       ConstantExpr {{.*}} 'const char *'
// CHECK-NEXT:         value: LValue
// CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
// CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
// CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
// CHECK-NEXT:             DeclRefExpr {{.*}} 'const char[5]' lvalue Var {{.*}} 'AttrVal1' 'const char[5]'
// CHECK-NEXT:       ConstantExpr {{.*}} 'const char *'
// CHECK-NEXT:         value: LValue
// CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
// CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
// CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
// CHECK-NEXT:             DeclRefExpr {{.*}} 'const char[5]' lvalue Var {{.*}} 'AttrVal2' 'const char[5]'
// CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct GlobalVarStructTemplate5
// CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit used constexpr GlobalVarStructTemplate5 'void () noexcept'
// CHECK-NEXT:       CompoundStmt
// CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr GlobalVarStructTemplate5 'void (const GlobalVarStructTemplate5<AttrName1, AttrName2, AttrVal1, AttrVal2> &)'
// CHECK-NEXT:       ParmVarDecl {{.*}} 'const GlobalVarStructTemplate5<AttrName1, AttrName2, AttrVal1, AttrVal2> &'
// CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr GlobalVarStructTemplate5 'void (GlobalVarStructTemplate5<AttrName1, AttrName2, AttrVal1, AttrVal2> &&)'
// CHECK-NEXT:       ParmVarDecl {{.*}} 'GlobalVarStructTemplate5<AttrName1, AttrName2, AttrVal1, AttrVal2> &&'
// CHECK-NEXT:   ClassTemplateSpecializationDecl {{.*}} struct GlobalVarStructTemplate5 definition
// CHECK-NEXT:     DefinitionData
// CHECK-NEXT:       DefaultConstructor
// CHECK-NEXT:       CopyConstructor
// CHECK-NEXT:       MoveConstructor
// CHECK-NEXT:       CopyAssignment
// CHECK-NEXT:       MoveAssignment
// CHECK-NEXT:       Destructor
// CHECK-NEXT:     TemplateArgument pack
// CHECK-NEXT:       TemplateArgument decl
// CHECK-NEXT:         Var {{.*}} 'AttrName1' 'const char[6]'
// CHECK-NEXT:       TemplateArgument decl
// CHECK-NEXT:         Var {{.*}} 'AttrName2' 'const char[6]'
// CHECK-NEXT:       TemplateArgument decl
// CHECK-NEXT:         Var {{.*}} 'AttrName3' 'const char[6]'
// CHECK-NEXT:       TemplateArgument decl
// CHECK-NEXT:         Var {{.*}} 'AttrVal1' 'const char[5]'
// CHECK-NEXT:       TemplateArgument decl
// CHECK-NEXT:         Var {{.*}} 'AttrVal2' 'const char[5]'
// CHECK-NEXT:       TemplateArgument decl
// CHECK-NEXT:         Var {{.*}} 'AttrVal3' 'const char[5]'
// CHECK-NEXT:     SYCLAddIRAttributesGlobalVariableAttr
// CHECK-NEXT:       ConstantExpr {{.*}} 'const char *'
// CHECK-NEXT:         value: LValue
// CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
// CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
// CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
// CHECK-NEXT:             DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName1' 'const char[6]'
// CHECK-NEXT:       ConstantExpr {{.*}} 'const char *'
// CHECK-NEXT:         value: LValue
// CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
// CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
// CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
// CHECK-NEXT:             DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName2' 'const char[6]'
// CHECK-NEXT:       ConstantExpr {{.*}} 'const char *'
// CHECK-NEXT:         value: LValue
// CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
// CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
// CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
// CHECK-NEXT:             DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName3' 'const char[6]'
// CHECK-NEXT:       ConstantExpr {{.*}} 'const char *'
// CHECK-NEXT:         value: LValue
// CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
// CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
// CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
// CHECK-NEXT:             DeclRefExpr {{.*}} 'const char[5]' lvalue Var {{.*}} 'AttrVal1' 'const char[5]'
// CHECK-NEXT:       ConstantExpr {{.*}} 'const char *'
// CHECK-NEXT:         value: LValue
// CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
// CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
// CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
// CHECK-NEXT:             DeclRefExpr {{.*}} 'const char[5]' lvalue Var {{.*}} 'AttrVal2' 'const char[5]'
// CHECK-NEXT:       ConstantExpr {{.*}} 'const char *'
// CHECK-NEXT:         value: LValue
// CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
// CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
// CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
// CHECK-NEXT:             DeclRefExpr {{.*}} 'const char[5]' lvalue Var {{.*}} 'AttrVal3' 'const char[5]'
// CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct GlobalVarStructTemplate5
// CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit used constexpr GlobalVarStructTemplate5 'void () noexcept'
// CHECK-NEXT:       CompoundStmt
// CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr GlobalVarStructTemplate5 'void (const GlobalVarStructTemplate5<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2, AttrVal3> &)'
// CHECK-NEXT:       ParmVarDecl {{.*}} 'const GlobalVarStructTemplate5<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2, AttrVal3> &'
// CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr GlobalVarStructTemplate5 'void (GlobalVarStructTemplate5<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2, AttrVal3> &&)'
// CHECK-NEXT:       ParmVarDecl {{.*}} 'GlobalVarStructTemplate5<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2, AttrVal3> &&'
// CHECK-NEXT:   ClassTemplateSpecializationDecl {{.*}} struct GlobalVarStructTemplate5 definition
// CHECK-NEXT:     DefinitionData
// CHECK-NEXT:       DefaultConstructor
// CHECK-NEXT:       CopyConstructor
// CHECK-NEXT:       MoveConstructor
// CHECK-NEXT:       CopyAssignment
// CHECK-NEXT:       MoveAssignment
// CHECK-NEXT:       Destructor
// CHECK-NEXT:     TemplateArgument pack
// CHECK-NEXT:       TemplateArgument decl
// CHECK-NEXT:         Var {{.*}} 'AttrName1' 'const char[6]'
// CHECK-NEXT:       TemplateArgument decl
// CHECK-NEXT:         Var {{.*}} 'AttrName2' 'const char[6]'
// CHECK-NEXT:       TemplateArgument decl
// CHECK-NEXT:         Var {{.*}} 'AttrName3' 'const char[6]'
// CHECK-NEXT:       TemplateArgument decl
// CHECK-NEXT:         Var {{.*}} 'AttrVal1' 'const char[5]'
// CHECK-NEXT:       TemplateArgument decl
// CHECK-NEXT:         Var {{.*}} 'AttrVal2' 'const char[5]'
// CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct GlobalVarStructTemplate5
// CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit used constexpr GlobalVarStructTemplate5 'void () noexcept'
// CHECK-NEXT:       CompoundStmt
// CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr GlobalVarStructTemplate5 'void (const GlobalVarStructTemplate5<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2> &)'
// CHECK-NEXT:       ParmVarDecl {{.*}} 'const GlobalVarStructTemplate5<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2> &'
// CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr GlobalVarStructTemplate5 'void (GlobalVarStructTemplate5<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2> &&)'
// CHECK-NEXT:       ParmVarDecl {{.*}} 'GlobalVarStructTemplate5<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2> &&'
GlobalVarStructTemplate5<AttrName1, AttrVal1> InstantiatedGV9;
GlobalVarStructTemplate5<AttrName1, AttrName2, AttrVal1, AttrVal2> InstantiatedGV10;
GlobalVarStructTemplate5<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2, AttrVal3> InstantiatedGV11;
GlobalVarStructTemplate5<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2> InstantiatedGV12; // expected-note {{in instantiation of template class 'GlobalVarStructTemplate5<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2>' requested here}}

// CHECK:      ClassTemplateDecl {{.*}} GlobalVarStructTemplate6
// CHECK-NEXT:   NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
// CHECK-NEXT:   CXXRecordDecl {{.*}} struct GlobalVarStructTemplate6 definition
// CHECK-NEXT:     DefinitionData
// CHECK-NEXT:       DefaultConstructor
// CHECK-NEXT:       CopyConstructor
// CHECK-NEXT:       MoveConstructor
// CHECK-NEXT:       CopyAssignment
// CHECK-NEXT:       MoveAssignment
// CHECK-NEXT:       Destructor
// CHECK-NEXT:     SYCLAddIRAttributesGlobalVariableAttr
// CHECK-NEXT:       InitListExpr {{.*}} 'void'
// CHECK-NEXT:         StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
// CHECK-NEXT:         StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
// CHECK-NEXT:       PackExpansionExpr {{.*}} '<dependent type>'
// CHECK-NEXT:         DeclRefExpr {{.*}} 'const char *' NonTypeTemplateParm {{.*}} 'Strs' 'const char *'
// CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct GlobalVarStructTemplate6
// CHECK-NEXT:   ClassTemplateSpecializationDecl {{.*}} struct GlobalVarStructTemplate6 definition
// CHECK-NEXT:     DefinitionData
// CHECK-NEXT:       DefaultConstructor
// CHECK-NEXT:       CopyConstructor
// CHECK-NEXT:       MoveConstructor
// CHECK-NEXT:       CopyAssignment
// CHECK-NEXT:       MoveAssignment
// CHECK-NEXT:       Destructor
// CHECK-NEXT:     TemplateArgument pack
// CHECK-NEXT:       TemplateArgument decl
// CHECK-NEXT:         Var {{.*}} 'AttrName1' 'const char[6]'
// CHECK-NEXT:       TemplateArgument decl
// CHECK-NEXT:         Var {{.*}} 'AttrVal1' 'const char[5]'
// CHECK-NEXT:     SYCLAddIRAttributesGlobalVariableAttr
// CHECK-NEXT:       InitListExpr {{.*}} 'void'
// CHECK-NEXT:         StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
// CHECK-NEXT:         StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
// CHECK-NEXT:       ConstantExpr {{.*}} 'const char *'
// CHECK-NEXT:         value: LValue
// CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
// CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
// CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
// CHECK-NEXT:             DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName1' 'const char[6]'
// CHECK-NEXT:       ConstantExpr {{.*}} 'const char *'
// CHECK-NEXT:         value: LValue
// CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
// CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
// CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
// CHECK-NEXT:             DeclRefExpr {{.*}} 'const char[5]' lvalue Var {{.*}} 'AttrVal1' 'const char[5]'
// CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct GlobalVarStructTemplate6
// CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit used constexpr GlobalVarStructTemplate6 'void () noexcept'
// CHECK-NEXT:       CompoundStmt
// CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr GlobalVarStructTemplate6 'void (const GlobalVarStructTemplate6<AttrName1, AttrVal1> &)'
// CHECK-NEXT:       ParmVarDecl {{.*}} 'const GlobalVarStructTemplate6<AttrName1, AttrVal1> &'
// CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr GlobalVarStructTemplate6 'void (GlobalVarStructTemplate6<AttrName1, AttrVal1> &&)'
// CHECK-NEXT:       ParmVarDecl {{.*}} 'GlobalVarStructTemplate6<AttrName1, AttrVal1> &&'
// CHECK-NEXT:   ClassTemplateSpecializationDecl {{.*}} struct GlobalVarStructTemplate6 definition
// CHECK-NEXT:     DefinitionData
// CHECK-NEXT:       DefaultConstructor
// CHECK-NEXT:       CopyConstructor
// CHECK-NEXT:       MoveConstructor
// CHECK-NEXT:       CopyAssignment
// CHECK-NEXT:       MoveAssignment
// CHECK-NEXT:       Destructor
// CHECK-NEXT:     TemplateArgument pack
// CHECK-NEXT:       TemplateArgument decl
// CHECK-NEXT:         Var {{.*}} 'AttrName1' 'const char[6]'
// CHECK-NEXT:       TemplateArgument decl
// CHECK-NEXT:         Var {{.*}} 'AttrName2' 'const char[6]'
// CHECK-NEXT:       TemplateArgument decl
// CHECK-NEXT:         Var {{.*}} 'AttrVal1' 'const char[5]'
// CHECK-NEXT:       TemplateArgument decl
// CHECK-NEXT:         Var {{.*}} 'AttrVal2' 'const char[5]'
// CHECK-NEXT:     SYCLAddIRAttributesGlobalVariableAttr
// CHECK-NEXT:       InitListExpr {{.*}} 'void'
// CHECK-NEXT:         StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
// CHECK-NEXT:         StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
// CHECK-NEXT:       ConstantExpr {{.*}} 'const char *'
// CHECK-NEXT:         value: LValue
// CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
// CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
// CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
// CHECK-NEXT:             DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName1' 'const char[6]'
// CHECK-NEXT:       ConstantExpr {{.*}} 'const char *'
// CHECK-NEXT:         value: LValue
// CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
// CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
// CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
// CHECK-NEXT:             DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName2' 'const char[6]'
// CHECK-NEXT:       ConstantExpr {{.*}} 'const char *'
// CHECK-NEXT:         value: LValue
// CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
// CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
// CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
// CHECK-NEXT:             DeclRefExpr {{.*}} 'const char[5]' lvalue Var {{.*}} 'AttrVal1' 'const char[5]'
// CHECK-NEXT:       ConstantExpr {{.*}} 'const char *'
// CHECK-NEXT:         value: LValue
// CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
// CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
// CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
// CHECK-NEXT:             DeclRefExpr {{.*}} 'const char[5]' lvalue Var {{.*}} 'AttrVal2' 'const char[5]'
// CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct GlobalVarStructTemplate6
// CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit used constexpr GlobalVarStructTemplate6 'void () noexcept'
// CHECK-NEXT:       CompoundStmt
// CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr GlobalVarStructTemplate6 'void (const GlobalVarStructTemplate6<AttrName1, AttrName2, AttrVal1, AttrVal2> &)'
// CHECK-NEXT:       ParmVarDecl {{.*}} 'const GlobalVarStructTemplate6<AttrName1, AttrName2, AttrVal1, AttrVal2> &'
// CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr GlobalVarStructTemplate6 'void (GlobalVarStructTemplate6<AttrName1, AttrName2, AttrVal1, AttrVal2> &&)'
// CHECK-NEXT:       ParmVarDecl {{.*}} 'GlobalVarStructTemplate6<AttrName1, AttrName2, AttrVal1, AttrVal2> &&'
// CHECK-NEXT:   ClassTemplateSpecializationDecl {{.*}} struct GlobalVarStructTemplate6 definition
// CHECK-NEXT:     DefinitionData
// CHECK-NEXT:       DefaultConstructor
// CHECK-NEXT:       CopyConstructor
// CHECK-NEXT:       MoveConstructor
// CHECK-NEXT:       CopyAssignment
// CHECK-NEXT:       MoveAssignment
// CHECK-NEXT:       Destructor
// CHECK-NEXT:     TemplateArgument pack
// CHECK-NEXT:       TemplateArgument decl
// CHECK-NEXT:         Var {{.*}} 'AttrName1' 'const char[6]'
// CHECK-NEXT:       TemplateArgument decl
// CHECK-NEXT:         Var {{.*}} 'AttrName2' 'const char[6]'
// CHECK-NEXT:       TemplateArgument decl
// CHECK-NEXT:         Var {{.*}} 'AttrName3' 'const char[6]'
// CHECK-NEXT:       TemplateArgument decl
// CHECK-NEXT:         Var {{.*}} 'AttrVal1' 'const char[5]'
// CHECK-NEXT:       TemplateArgument decl
// CHECK-NEXT:         Var {{.*}} 'AttrVal2' 'const char[5]'
// CHECK-NEXT:       TemplateArgument decl
// CHECK-NEXT:         Var {{.*}} 'AttrVal3' 'const char[5]'
// CHECK-NEXT:     SYCLAddIRAttributesGlobalVariableAttr
// CHECK-NEXT:       InitListExpr {{.*}} 'void'
// CHECK-NEXT:         StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
// CHECK-NEXT:         StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
// CHECK-NEXT:       ConstantExpr {{.*}} 'const char *'
// CHECK-NEXT:         value: LValue
// CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
// CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
// CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
// CHECK-NEXT:             DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName1' 'const char[6]'
// CHECK-NEXT:       ConstantExpr {{.*}} 'const char *'
// CHECK-NEXT:         value: LValue
// CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
// CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
// CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
// CHECK-NEXT:             DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName2' 'const char[6]'
// CHECK-NEXT:       ConstantExpr {{.*}} 'const char *'
// CHECK-NEXT:         value: LValue
// CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
// CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
// CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
// CHECK-NEXT:             DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName3' 'const char[6]'
// CHECK-NEXT:       ConstantExpr {{.*}} 'const char *'
// CHECK-NEXT:         value: LValue
// CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
// CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
// CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
// CHECK-NEXT:             DeclRefExpr {{.*}} 'const char[5]' lvalue Var {{.*}} 'AttrVal1' 'const char[5]'
// CHECK-NEXT:       ConstantExpr {{.*}} 'const char *'
// CHECK-NEXT:         value: LValue
// CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
// CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
// CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
// CHECK-NEXT:             DeclRefExpr {{.*}} 'const char[5]' lvalue Var {{.*}} 'AttrVal2' 'const char[5]'
// CHECK-NEXT:       ConstantExpr {{.*}} 'const char *'
// CHECK-NEXT:         value: LValue
// CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
// CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
// CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
// CHECK-NEXT:             DeclRefExpr {{.*}} 'const char[5]' lvalue Var {{.*}} 'AttrVal3' 'const char[5]'
// CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct GlobalVarStructTemplate6
// CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit used constexpr GlobalVarStructTemplate6 'void () noexcept'
// CHECK-NEXT:       CompoundStmt
// CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr GlobalVarStructTemplate6 'void (const GlobalVarStructTemplate6<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2, AttrVal3> &)'
// CHECK-NEXT:       ParmVarDecl {{.*}} 'const GlobalVarStructTemplate6<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2, AttrVal3> &'
// CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr GlobalVarStructTemplate6 'void (GlobalVarStructTemplate6<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2, AttrVal3> &&)'
// CHECK-NEXT:       ParmVarDecl {{.*}} 'GlobalVarStructTemplate6<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2, AttrVal3> &&'
// CHECK-NEXT:   ClassTemplateSpecializationDecl {{.*}} struct GlobalVarStructTemplate6 definition
// CHECK-NEXT:     DefinitionData
// CHECK-NEXT:       DefaultConstructor
// CHECK-NEXT:       CopyConstructor
// CHECK-NEXT:       MoveConstructor
// CHECK-NEXT:       CopyAssignment
// CHECK-NEXT:       MoveAssignment
// CHECK-NEXT:       Destructor
// CHECK-NEXT:     TemplateArgument pack
// CHECK-NEXT:       TemplateArgument decl
// CHECK-NEXT:         Var {{.*}} 'AttrName1' 'const char[6]'
// CHECK-NEXT:       TemplateArgument decl
// CHECK-NEXT:         Var {{.*}} 'AttrName2' 'const char[6]'
// CHECK-NEXT:       TemplateArgument decl
// CHECK-NEXT:         Var {{.*}} 'AttrName3' 'const char[6]'
// CHECK-NEXT:       TemplateArgument decl
// CHECK-NEXT:         Var {{.*}} 'AttrVal1' 'const char[5]'
// CHECK-NEXT:       TemplateArgument decl
// CHECK-NEXT:         Var {{.*}} 'AttrVal2' 'const char[5]'
// CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct GlobalVarStructTemplate6
// CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit used constexpr GlobalVarStructTemplate6 'void () noexcept'
// CHECK-NEXT:       CompoundStmt
// CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr GlobalVarStructTemplate6 'void (const GlobalVarStructTemplate6<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2> &)'
// CHECK-NEXT:       ParmVarDecl {{.*}} 'const GlobalVarStructTemplate6<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2> &'
// CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr GlobalVarStructTemplate6 'void (GlobalVarStructTemplate6<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2> &&)'
// CHECK-NEXT:       ParmVarDecl {{.*}} 'GlobalVarStructTemplate6<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2> &&'
GlobalVarStructTemplate6<AttrName1, AttrVal1> InstantiatedGV13;
GlobalVarStructTemplate6<AttrName1, AttrName2, AttrVal1, AttrVal2> InstantiatedGV14;
GlobalVarStructTemplate6<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2, AttrVal3> InstantiatedGV15;
GlobalVarStructTemplate6<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2> InstantiatedGV16; // expected-note {{in instantiation of template class 'GlobalVarStructTemplate6<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2>' requested here}}

template <int... Is> struct SpecialClassStructTemplate1 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter("Attr1", "Attr2", "Attr3", Is...)]] int x) {} // expected-error {{attribute 'add_ir_attributes_kernel_parameter' must have an attribute value for each attribute name}}
};
template <int... Is> struct SpecialClassStructTemplate2 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, "Attr1", "Attr2", "Attr3", Is...)]] int x) {} // expected-error {{attribute 'add_ir_attributes_kernel_parameter' must have an attribute value for each attribute name}}
};
template <const char *...Names> struct SpecialClassStructTemplate3 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(Names..., 1, 2, 3)]] int x) {} // expected-error {{attribute 'add_ir_attributes_kernel_parameter' must have an attribute value for each attribute name}}
};
template <const char *...Names> struct SpecialClassStructTemplate4 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, Names..., 1, 2, 3)]] int x) {} // expected-error {{attribute 'add_ir_attributes_kernel_parameter' must have an attribute value for each attribute name}}
};
template <const char *...Strs> struct SpecialClassStructTemplate5 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(Strs...)]] int x) {} // expected-error {{attribute 'add_ir_attributes_kernel_parameter' must have an attribute value for each attribute name}}
};
template <const char *...Strs> struct SpecialClassStructTemplate6 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, Strs...)]] int x) {} // expected-error {{attribute 'add_ir_attributes_kernel_parameter' must have an attribute value for each attribute name}}
};

void InstantiateSpecialClassStructTemplates() {
  // CHECK:      ClassTemplateDecl {{.*}} SpecialClassStructTemplate1
  // CHECK-NEXT:   NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 0 index 0 ... Is
  // CHECK-NEXT:   CXXRecordDecl {{.*}} struct SpecialClassStructTemplate1 definition
  // CHECK-NEXT:     DefinitionData
  // CHECK-NEXT:       DefaultConstructor
  // CHECK-NEXT:       CopyConstructor
  // CHECK-NEXT:       MoveConstructor
  // CHECK-NEXT:       CopyAssignment
  // CHECK-NEXT:       MoveAssignment
  // CHECK-NEXT:       Destructor
  // CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct SpecialClassStructTemplate1
  // CHECK-NEXT:     CXXMethodDecl {{.*}} __init 'void (int)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} x 'int'
  // CHECK-NEXT:         SYCLAddIRAttributesKernelParameterAttr
  // CHECK-NEXT:           ConstantExpr {{.*}} 'const char[6]' lvalue
  // CHECK-NEXT:             value: LValue
  // CHECK-NEXT:             StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
  // CHECK-NEXT:           ConstantExpr {{.*}} 'const char[6]' lvalue
  // CHECK-NEXT:             value: LValue
  // CHECK-NEXT:             StringLiteral {{.*}} 'const char[6]' lvalue "Attr2"
  // CHECK-NEXT:           ConstantExpr {{.*}} 'const char[6]' lvalue
  // CHECK-NEXT:             value: LValue
  // CHECK-NEXT:             StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
  // CHECK-NEXT:           PackExpansionExpr {{.*}} '<dependent type>'
  // CHECK-NEXT:             DeclRefExpr {{.*}} 'int' NonTypeTemplateParm {{.*}} 'Is' 'int'
  // CHECK-NEXT:       CompoundStmt
  // CHECK-NEXT:   ClassTemplateSpecializationDecl {{.*}} struct SpecialClassStructTemplate1 definition
  // CHECK-NEXT:     DefinitionData
  // CHECK-NEXT:       DefaultConstructor
  // CHECK-NEXT:       CopyConstructor
  // CHECK-NEXT:       MoveConstructor
  // CHECK-NEXT:       CopyAssignment
  // CHECK-NEXT:       MoveAssignment
  // CHECK-NEXT:       Destructor
  // CHECK-NEXT:     TemplateArgument pack
  // CHECK-NEXT:       TemplateArgument integral 1
  // CHECK-NEXT:       TemplateArgument integral 2
  // CHECK-NEXT:       TemplateArgument integral 3
  // CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct SpecialClassStructTemplate1
  // CHECK-NEXT:     CXXMethodDecl {{.*}} __init 'void (int)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} x 'int'
  // CHECK-NEXT:         SYCLAddIRAttributesKernelParameterAttr
  // CHECK-NEXT:           ConstantExpr {{.*}} 'const char[6]' lvalue
  // CHECK-NEXT:             value: LValue
  // CHECK-NEXT:             StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
  // CHECK-NEXT:           ConstantExpr {{.*}} 'const char[6]' lvalue
  // CHECK-NEXT:             value: LValue
  // CHECK-NEXT:             StringLiteral {{.*}} 'const char[6]' lvalue "Attr2"
  // CHECK-NEXT:           ConstantExpr {{.*}} 'const char[6]' lvalue
  // CHECK-NEXT:             value: LValue
  // CHECK-NEXT:             StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
  // CHECK-NEXT:           ConstantExpr {{.*}} 'int'
  // CHECK-NEXT:             value: Int 1
  // CHECK-NEXT:             SubstNonTypeTemplateParmExpr {{.*}} 'int'
  // CHECK-NEXT:               NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 0 index 0 ... Is
  // CHECK-NEXT:               IntegerLiteral {{.*}} 'int' 1
  // CHECK-NEXT:           ConstantExpr {{.*}} 'int'
  // CHECK-NEXT:             value: Int 2
  // CHECK-NEXT:             SubstNonTypeTemplateParmExpr {{.*}} 'int'
  // CHECK-NEXT:               NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 0 index 0 ... Is
  // CHECK-NEXT:               IntegerLiteral {{.*}} 'int' 2
  // CHECK-NEXT:           ConstantExpr {{.*}} 'int'
  // CHECK-NEXT:             value: Int 3
  // CHECK-NEXT:             SubstNonTypeTemplateParmExpr {{.*}} 'int'
  // CHECK-NEXT:               NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 0 index 0 ... Is
  // CHECK-NEXT:               IntegerLiteral {{.*}} 'int' 3
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit used constexpr SpecialClassStructTemplate1 'void () noexcept'
  // CHECK-NEXT:       CompoundStmt
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr SpecialClassStructTemplate1 'void (const SpecialClassStructTemplate1<1, 2, 3> &)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'const SpecialClassStructTemplate1<1, 2, 3> &'
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr SpecialClassStructTemplate1 'void (SpecialClassStructTemplate1<1, 2, 3> &&)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'SpecialClassStructTemplate1<1, 2, 3> &&'
  // CHECK-NEXT:   ClassTemplateSpecializationDecl {{.*}} struct SpecialClassStructTemplate1 definition
  // CHECK-NEXT:     DefinitionData
  // CHECK-NEXT:       DefaultConstructor
  // CHECK-NEXT:       CopyConstructor
  // CHECK-NEXT:       MoveConstructor
  // CHECK-NEXT:       CopyAssignment
  // CHECK-NEXT:       MoveAssignment
  // CHECK-NEXT:       Destructor
  // CHECK-NEXT:     TemplateArgument pack
  // CHECK-NEXT:       TemplateArgument integral 1
  // CHECK-NEXT:       TemplateArgument integral 2
  // CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct SpecialClassStructTemplate1
  // CHECK-NEXT:     CXXMethodDecl {{.*}} __init 'void (int)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} x 'int'
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit used constexpr SpecialClassStructTemplate1 'void () noexcept'
  // CHECK-NEXT:       CompoundStmt
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr SpecialClassStructTemplate1 'void (const SpecialClassStructTemplate1<1, 2> &)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'const SpecialClassStructTemplate1<1, 2> &'
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr SpecialClassStructTemplate1 'void (SpecialClassStructTemplate1<1, 2> &&)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'SpecialClassStructTemplate1<1, 2> &&'
  SpecialClassStructTemplate1<1, 2, 3> InstantiatedSCS1;
  SpecialClassStructTemplate1<1, 2> InstantiatedSCS2; // expected-note {{in instantiation of template class 'SpecialClassStructTemplate1<1, 2>' requested here}}

  // CHECK:      ClassTemplateDecl {{.*}} SpecialClassStructTemplate2
  // CHECK-NEXT:   NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 0 index 0 ... Is
  // CHECK-NEXT:   CXXRecordDecl {{.*}} struct SpecialClassStructTemplate2 definition
  // CHECK-NEXT:     DefinitionData
  // CHECK-NEXT:       DefaultConstructor
  // CHECK-NEXT:       CopyConstructor
  // CHECK-NEXT:       MoveConstructor
  // CHECK-NEXT:       CopyAssignment
  // CHECK-NEXT:       MoveAssignment
  // CHECK-NEXT:       Destructor
  // CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct SpecialClassStructTemplate2
  // CHECK-NEXT:     CXXMethodDecl {{.*}} __init 'void (int)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} x 'int'
  // CHECK-NEXT:         SYCLAddIRAttributesKernelParameterAttr
  // CHECK-NEXT:           InitListExpr {{.*}} 'void'
  // CHECK-NEXT:             StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
  // CHECK-NEXT:             StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
  // CHECK-NEXT:           ConstantExpr {{.*}} 'const char[6]' lvalue
  // CHECK-NEXT:             value: LValue
  // CHECK-NEXT:             StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
  // CHECK-NEXT:           ConstantExpr {{.*}} 'const char[6]' lvalue
  // CHECK-NEXT:             value: LValue
  // CHECK-NEXT:             StringLiteral {{.*}} 'const char[6]' lvalue "Attr2"
  // CHECK-NEXT:           ConstantExpr {{.*}} 'const char[6]' lvalue
  // CHECK-NEXT:             value: LValue
  // CHECK-NEXT:             StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
  // CHECK-NEXT:           PackExpansionExpr {{.*}} '<dependent type>'
  // CHECK-NEXT:             DeclRefExpr {{.*}} 'int' NonTypeTemplateParm {{.*}} 'Is' 'int'
  // CHECK-NEXT:       CompoundStmt
  // CHECK-NEXT:   ClassTemplateSpecializationDecl {{.*}} struct SpecialClassStructTemplate2 definition
  // CHECK-NEXT:     DefinitionData
  // CHECK-NEXT:       DefaultConstructor
  // CHECK-NEXT:       CopyConstructor
  // CHECK-NEXT:       MoveConstructor
  // CHECK-NEXT:       CopyAssignment
  // CHECK-NEXT:       MoveAssignment
  // CHECK-NEXT:       Destructor
  // CHECK-NEXT:     TemplateArgument pack
  // CHECK-NEXT:       TemplateArgument integral 1
  // CHECK-NEXT:       TemplateArgument integral 2
  // CHECK-NEXT:       TemplateArgument integral 3
  // CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct SpecialClassStructTemplate2
  // CHECK-NEXT:     CXXMethodDecl {{.*}} __init 'void (int)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} x 'int'
  // CHECK-NEXT:         SYCLAddIRAttributesKernelParameterAttr
  // CHECK-NEXT:           InitListExpr {{.*}} 'void'
  // CHECK-NEXT:             StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
  // CHECK-NEXT:             StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
  // CHECK-NEXT:           ConstantExpr {{.*}} 'const char[6]' lvalue
  // CHECK-NEXT:             value: LValue
  // CHECK-NEXT:             StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
  // CHECK-NEXT:           ConstantExpr {{.*}} 'const char[6]' lvalue
  // CHECK-NEXT:             value: LValue
  // CHECK-NEXT:             StringLiteral {{.*}} 'const char[6]' lvalue "Attr2"
  // CHECK-NEXT:           ConstantExpr {{.*}} 'const char[6]' lvalue
  // CHECK-NEXT:             value: LValue
  // CHECK-NEXT:             StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
  // CHECK-NEXT:           ConstantExpr {{.*}} 'int'
  // CHECK-NEXT:             value: Int 1
  // CHECK-NEXT:             SubstNonTypeTemplateParmExpr {{.*}} 'int'
  // CHECK-NEXT:               NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 0 index 0 ... Is
  // CHECK-NEXT:               IntegerLiteral {{.*}} 'int' 1
  // CHECK-NEXT:           ConstantExpr {{.*}} 'int'
  // CHECK-NEXT:             value: Int 2
  // CHECK-NEXT:             SubstNonTypeTemplateParmExpr {{.*}} 'int'
  // CHECK-NEXT:               NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 0 index 0 ... Is
  // CHECK-NEXT:               IntegerLiteral {{.*}} 'int' 2
  // CHECK-NEXT:           ConstantExpr {{.*}} 'int'
  // CHECK-NEXT:             value: Int 3
  // CHECK-NEXT:             SubstNonTypeTemplateParmExpr {{.*}} 'int'
  // CHECK-NEXT:               NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 0 index 0 ... Is
  // CHECK-NEXT:               IntegerLiteral {{.*}} 'int' 3
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit used constexpr SpecialClassStructTemplate2 'void () noexcept'
  // CHECK-NEXT:       CompoundStmt
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr SpecialClassStructTemplate2 'void (const SpecialClassStructTemplate2<1, 2, 3> &)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'const SpecialClassStructTemplate2<1, 2, 3> &'
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr SpecialClassStructTemplate2 'void (SpecialClassStructTemplate2<1, 2, 3> &&)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'SpecialClassStructTemplate2<1, 2, 3> &&'
  // CHECK-NEXT:   ClassTemplateSpecializationDecl {{.*}} struct SpecialClassStructTemplate2 definition
  // CHECK-NEXT:     DefinitionData
  // CHECK-NEXT:       DefaultConstructor
  // CHECK-NEXT:       CopyConstructor
  // CHECK-NEXT:       MoveConstructor
  // CHECK-NEXT:       CopyAssignment
  // CHECK-NEXT:       MoveAssignment
  // CHECK-NEXT:       Destructor
  // CHECK-NEXT:     TemplateArgument pack
  // CHECK-NEXT:       TemplateArgument integral 1
  // CHECK-NEXT:       TemplateArgument integral 2
  // CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct SpecialClassStructTemplate2
  // CHECK-NEXT:     CXXMethodDecl {{.*}} __init 'void (int)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} x 'int'
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit used constexpr SpecialClassStructTemplate2 'void () noexcept'
  // CHECK-NEXT:       CompoundStmt
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr SpecialClassStructTemplate2 'void (const SpecialClassStructTemplate2<1, 2> &)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'const SpecialClassStructTemplate2<1, 2> &'
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr SpecialClassStructTemplate2 'void (SpecialClassStructTemplate2<1, 2> &&)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'SpecialClassStructTemplate2<1, 2> &&'
  SpecialClassStructTemplate2<1, 2, 3> InstantiatedSCS3;
  SpecialClassStructTemplate2<1, 2> InstantiatedSCS4; // expected-note {{in instantiation of template class 'SpecialClassStructTemplate2<1, 2>' requested here}}

  // CHECK:      ClassTemplateDecl {{.*}} SpecialClassStructTemplate3
  // CHECK-NEXT:   NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Names
  // CHECK-NEXT:   CXXRecordDecl {{.*}} struct SpecialClassStructTemplate3 definition
  // CHECK-NEXT:     DefinitionData
  // CHECK-NEXT:       DefaultConstructor
  // CHECK-NEXT:       CopyConstructor
  // CHECK-NEXT:       MoveConstructor
  // CHECK-NEXT:       CopyAssignment
  // CHECK-NEXT:       MoveAssignment
  // CHECK-NEXT:       Destructor
  // CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct SpecialClassStructTemplate3
  // CHECK-NEXT:     CXXMethodDecl {{.*}} __init 'void (int)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} x 'int'
  // CHECK-NEXT:         SYCLAddIRAttributesKernelParameterAttr
  // CHECK-NEXT:           PackExpansionExpr {{.*}} '<dependent type>'
  // CHECK-NEXT:             DeclRefExpr {{.*}} 'const char *' NonTypeTemplateParm {{.*}} 'Names' 'const char *'
  // CHECK-NEXT:           ConstantExpr {{.*}} 'int'
  // CHECK-NEXT:             value: Int 1
  // CHECK-NEXT:             IntegerLiteral {{.*}} 'int' 1
  // CHECK-NEXT:           ConstantExpr {{.*}} 'int'
  // CHECK-NEXT:             value: Int 2
  // CHECK-NEXT:             IntegerLiteral {{.*}} 'int' 2
  // CHECK-NEXT:           ConstantExpr {{.*}} 'int'
  // CHECK-NEXT:             value: Int 3
  // CHECK-NEXT:             IntegerLiteral {{.*}} 'int' 3
  // CHECK-NEXT:       CompoundStmt
  // CHECK-NEXT:   ClassTemplateSpecializationDecl {{.*}} struct SpecialClassStructTemplate3 definition
  // CHECK-NEXT:     DefinitionData
  // CHECK-NEXT:       DefaultConstructor
  // CHECK-NEXT:       CopyConstructor
  // CHECK-NEXT:       MoveConstructor
  // CHECK-NEXT:       CopyAssignment
  // CHECK-NEXT:       MoveAssignment
  // CHECK-NEXT:       Destructor
  // CHECK-NEXT:     TemplateArgument pack
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrName1' 'const char[6]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrName2' 'const char[6]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrName3' 'const char[6]'
  // CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct SpecialClassStructTemplate3
  // CHECK-NEXT:     CXXMethodDecl {{.*}} __init 'void (int)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} x 'int'
  // CHECK-NEXT:         SYCLAddIRAttributesKernelParameterAttr
  // CHECK-NEXT:           ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:             value: LValue
  // CHECK-NEXT:             SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:               NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Names
  // CHECK-NEXT:               ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:                 DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName1' 'const char[6]'
  // CHECK-NEXT:           ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:             value: LValue
  // CHECK-NEXT:             SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:               NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Names
  // CHECK-NEXT:               ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:                 DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName2' 'const char[6]'
  // CHECK-NEXT:           ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:             value: LValue
  // CHECK-NEXT:             SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:               NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Names
  // CHECK-NEXT:               ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:                 DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName3' 'const char[6]'
  // CHECK-NEXT:           ConstantExpr {{.*}} 'int'
  // CHECK-NEXT:             value: Int 1
  // CHECK-NEXT:             IntegerLiteral {{.*}} 'int' 1
  // CHECK-NEXT:           ConstantExpr {{.*}} 'int'
  // CHECK-NEXT:             value: Int 2
  // CHECK-NEXT:             IntegerLiteral {{.*}} 'int' 2
  // CHECK-NEXT:           ConstantExpr {{.*}} 'int'
  // CHECK-NEXT:             value: Int 3
  // CHECK-NEXT:             IntegerLiteral {{.*}} 'int' 3
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit used constexpr SpecialClassStructTemplate3 'void () noexcept'
  // CHECK-NEXT:       CompoundStmt
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr SpecialClassStructTemplate3 'void (const SpecialClassStructTemplate3<AttrName1, AttrName2, AttrName3> &)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'const SpecialClassStructTemplate3<AttrName1, AttrName2, AttrName3> &'
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr SpecialClassStructTemplate3 'void (SpecialClassStructTemplate3<AttrName1, AttrName2, AttrName3> &&)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'SpecialClassStructTemplate3<AttrName1, AttrName2, AttrName3> &&'
  // CHECK-NEXT:   ClassTemplateSpecializationDecl {{.*}} struct SpecialClassStructTemplate3 definition
  // CHECK-NEXT:     DefinitionData
  // CHECK-NEXT:       DefaultConstructor
  // CHECK-NEXT:       CopyConstructor
  // CHECK-NEXT:       MoveConstructor
  // CHECK-NEXT:       CopyAssignment
  // CHECK-NEXT:       MoveAssignment
  // CHECK-NEXT:       Destructor
  // CHECK-NEXT:     TemplateArgument pack
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrName1' 'const char[6]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrName2' 'const char[6]'
  // CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct SpecialClassStructTemplate3
  // CHECK-NEXT:     CXXMethodDecl {{.*}} __init 'void (int)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} x 'int'
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit used constexpr SpecialClassStructTemplate3 'void () noexcept'
  // CHECK-NEXT:       CompoundStmt
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr SpecialClassStructTemplate3 'void (const SpecialClassStructTemplate3<AttrName1, AttrName2> &)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'const SpecialClassStructTemplate3<AttrName1, AttrName2> &'
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr SpecialClassStructTemplate3 'void (SpecialClassStructTemplate3<AttrName1, AttrName2> &&)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'SpecialClassStructTemplate3<AttrName1, AttrName2> &&'
  SpecialClassStructTemplate3<AttrName1, AttrName2, AttrName3> InstantiatedSCS5;
  SpecialClassStructTemplate3<AttrName1, AttrName2> InstantiatedSCS6; // expected-note {{in instantiation of template class 'SpecialClassStructTemplate3<AttrName1, AttrName2>' requested here}}

  // CHECK:      ClassTemplateDecl {{.*}} SpecialClassStructTemplate4
  // CHECK-NEXT:   NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Names
  // CHECK-NEXT:   CXXRecordDecl {{.*}} struct SpecialClassStructTemplate4 definition
  // CHECK-NEXT:     DefinitionData
  // CHECK-NEXT:       DefaultConstructor
  // CHECK-NEXT:       CopyConstructor
  // CHECK-NEXT:       MoveConstructor
  // CHECK-NEXT:       CopyAssignment
  // CHECK-NEXT:       MoveAssignment
  // CHECK-NEXT:       Destructor
  // CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct SpecialClassStructTemplate4
  // CHECK-NEXT:     CXXMethodDecl {{.*}} __init 'void (int)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} x 'int'
  // CHECK-NEXT:         SYCLAddIRAttributesKernelParameterAttr
  // CHECK-NEXT:           InitListExpr {{.*}} 'void'
  // CHECK-NEXT:             StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
  // CHECK-NEXT:             StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
  // CHECK-NEXT:           PackExpansionExpr {{.*}} '<dependent type>'
  // CHECK-NEXT:             DeclRefExpr {{.*}} 'const char *' NonTypeTemplateParm {{.*}} 'Names' 'const char *'
  // CHECK-NEXT:           ConstantExpr {{.*}} 'int'
  // CHECK-NEXT:             value: Int 1
  // CHECK-NEXT:             IntegerLiteral {{.*}} 'int' 1
  // CHECK-NEXT:           ConstantExpr {{.*}} 'int'
  // CHECK-NEXT:             value: Int 2
  // CHECK-NEXT:             IntegerLiteral {{.*}} 'int' 2
  // CHECK-NEXT:           ConstantExpr {{.*}} 'int'
  // CHECK-NEXT:             value: Int 3
  // CHECK-NEXT:             IntegerLiteral {{.*}} 'int' 3
  // CHECK-NEXT:       CompoundStmt
  // CHECK-NEXT:   ClassTemplateSpecializationDecl {{.*}} struct SpecialClassStructTemplate4 definition
  // CHECK-NEXT:     DefinitionData
  // CHECK-NEXT:       DefaultConstructor
  // CHECK-NEXT:       CopyConstructor
  // CHECK-NEXT:       MoveConstructor
  // CHECK-NEXT:       CopyAssignment
  // CHECK-NEXT:       MoveAssignment
  // CHECK-NEXT:       Destructor
  // CHECK-NEXT:     TemplateArgument pack
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrName1' 'const char[6]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrName2' 'const char[6]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrName3' 'const char[6]'
  // CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct SpecialClassStructTemplate4
  // CHECK-NEXT:     CXXMethodDecl {{.*}} __init 'void (int)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} x 'int'
  // CHECK-NEXT:         SYCLAddIRAttributesKernelParameterAttr
  // CHECK-NEXT:           InitListExpr {{.*}} 'void'
  // CHECK-NEXT:             StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
  // CHECK-NEXT:             StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
  // CHECK-NEXT:           ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:             value: LValue
  // CHECK-NEXT:             SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:               NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Names
  // CHECK-NEXT:               ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:                 DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName1' 'const char[6]'
  // CHECK-NEXT:           ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:             value: LValue
  // CHECK-NEXT:             SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:               NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Names
  // CHECK-NEXT:               ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:                 DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName2' 'const char[6]'
  // CHECK-NEXT:           ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:             value: LValue
  // CHECK-NEXT:             SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:               NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Names
  // CHECK-NEXT:               ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:                 DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName3' 'const char[6]'
  // CHECK-NEXT:           ConstantExpr {{.*}} 'int'
  // CHECK-NEXT:             value: Int 1
  // CHECK-NEXT:             IntegerLiteral {{.*}} 'int' 1
  // CHECK-NEXT:           ConstantExpr {{.*}} 'int'
  // CHECK-NEXT:             value: Int 2
  // CHECK-NEXT:             IntegerLiteral {{.*}} 'int' 2
  // CHECK-NEXT:           ConstantExpr {{.*}} 'int'
  // CHECK-NEXT:             value: Int 3
  // CHECK-NEXT:             IntegerLiteral {{.*}} 'int' 3
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit used constexpr SpecialClassStructTemplate4 'void () noexcept'
  // CHECK-NEXT:       CompoundStmt
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr SpecialClassStructTemplate4 'void (const SpecialClassStructTemplate4<AttrName1, AttrName2, AttrName3> &)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'const SpecialClassStructTemplate4<AttrName1, AttrName2, AttrName3> &'
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr SpecialClassStructTemplate4 'void (SpecialClassStructTemplate4<AttrName1, AttrName2, AttrName3> &&)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'SpecialClassStructTemplate4<AttrName1, AttrName2, AttrName3> &&'
  // CHECK-NEXT:   ClassTemplateSpecializationDecl {{.*}} struct SpecialClassStructTemplate4 definition
  // CHECK-NEXT:     DefinitionData
  // CHECK-NEXT:       DefaultConstructor
  // CHECK-NEXT:       CopyConstructor
  // CHECK-NEXT:       MoveConstructor
  // CHECK-NEXT:       CopyAssignment
  // CHECK-NEXT:       MoveAssignment
  // CHECK-NEXT:       Destructor
  // CHECK-NEXT:     TemplateArgument pack
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrName1' 'const char[6]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrName2' 'const char[6]'
  // CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct SpecialClassStructTemplate4
  // CHECK-NEXT:     CXXMethodDecl {{.*}} __init 'void (int)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} x 'int'
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit used constexpr SpecialClassStructTemplate4 'void () noexcept'
  // CHECK-NEXT:       CompoundStmt
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr SpecialClassStructTemplate4 'void (const SpecialClassStructTemplate4<AttrName1, AttrName2> &)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'const SpecialClassStructTemplate4<AttrName1, AttrName2> &'
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr SpecialClassStructTemplate4 'void (SpecialClassStructTemplate4<AttrName1, AttrName2> &&)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'SpecialClassStructTemplate4<AttrName1, AttrName2> &&'
  SpecialClassStructTemplate4<AttrName1, AttrName2, AttrName3> InstantiatedSCS7;
  SpecialClassStructTemplate4<AttrName1, AttrName2> InstantiatedSCS8; // expected-note {{in instantiation of template class 'SpecialClassStructTemplate4<AttrName1, AttrName2>' requested here}}

  // CHECK:      ClassTemplateDecl {{.*}} SpecialClassStructTemplate5
  // CHECK-NEXT:   NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:   CXXRecordDecl {{.*}} struct SpecialClassStructTemplate5 definition
  // CHECK-NEXT:     DefinitionData
  // CHECK-NEXT:       DefaultConstructor
  // CHECK-NEXT:       CopyConstructor
  // CHECK-NEXT:       MoveConstructor
  // CHECK-NEXT:       CopyAssignment
  // CHECK-NEXT:       MoveAssignment
  // CHECK-NEXT:       Destructor
  // CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct SpecialClassStructTemplate5
  // CHECK-NEXT:     CXXMethodDecl {{.*}} __init 'void (int)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} x 'int'
  // CHECK-NEXT:         SYCLAddIRAttributesKernelParameterAttr
  // CHECK-NEXT:           PackExpansionExpr {{.*}} '<dependent type>'
  // CHECK-NEXT:             DeclRefExpr {{.*}} 'const char *' NonTypeTemplateParm {{.*}} 'Strs' 'const char *'
  // CHECK-NEXT:       CompoundStmt
  // CHECK-NEXT:   ClassTemplateSpecializationDecl {{.*}} struct SpecialClassStructTemplate5 definition
  // CHECK-NEXT:     DefinitionData
  // CHECK-NEXT:       DefaultConstructor
  // CHECK-NEXT:       CopyConstructor
  // CHECK-NEXT:       MoveConstructor
  // CHECK-NEXT:       CopyAssignment
  // CHECK-NEXT:       MoveAssignment
  // CHECK-NEXT:       Destructor
  // CHECK-NEXT:     TemplateArgument pack
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrName1' 'const char[6]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrVal1' 'const char[5]'
  // CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct SpecialClassStructTemplate5
  // CHECK-NEXT:     CXXMethodDecl {{.*}} __init 'void (int)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} x 'int'
  // CHECK-NEXT:         SYCLAddIRAttributesKernelParameterAttr
  // CHECK-NEXT:           ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:             value: LValue
  // CHECK-NEXT:             SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:               NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:               ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:                 DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName1' 'const char[6]'
  // CHECK-NEXT:           ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:             value: LValue
  // CHECK-NEXT:             SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:               NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:               ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:                 DeclRefExpr {{.*}} 'const char[5]' lvalue Var {{.*}} 'AttrVal1' 'const char[5]'
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit used constexpr SpecialClassStructTemplate5 'void () noexcept'
  // CHECK-NEXT:       CompoundStmt
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr SpecialClassStructTemplate5 'void (const SpecialClassStructTemplate5<AttrName1, AttrVal1> &)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'const SpecialClassStructTemplate5<AttrName1, AttrVal1> &'
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr SpecialClassStructTemplate5 'void (SpecialClassStructTemplate5<AttrName1, AttrVal1> &&)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'SpecialClassStructTemplate5<AttrName1, AttrVal1> &&'
  // CHECK-NEXT:   ClassTemplateSpecializationDecl {{.*}} struct SpecialClassStructTemplate5 definition
  // CHECK-NEXT:     DefinitionData
  // CHECK-NEXT:       DefaultConstructor
  // CHECK-NEXT:       CopyConstructor
  // CHECK-NEXT:       MoveConstructor
  // CHECK-NEXT:       CopyAssignment
  // CHECK-NEXT:       MoveAssignment
  // CHECK-NEXT:       Destructor
  // CHECK-NEXT:     TemplateArgument pack
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrName1' 'const char[6]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrName2' 'const char[6]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrVal1' 'const char[5]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrVal2' 'const char[5]'
  // CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct SpecialClassStructTemplate5
  // CHECK-NEXT:     CXXMethodDecl {{.*}} __init 'void (int)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} x 'int'
  // CHECK-NEXT:         SYCLAddIRAttributesKernelParameterAttr
  // CHECK-NEXT:           ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:             value: LValue
  // CHECK-NEXT:             SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:               NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:               ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:                 DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName1' 'const char[6]'
  // CHECK-NEXT:           ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:             value: LValue
  // CHECK-NEXT:             SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:               NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:               ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:                 DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName2' 'const char[6]'
  // CHECK-NEXT:           ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:             value: LValue
  // CHECK-NEXT:             SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:               NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:               ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:                 DeclRefExpr {{.*}} 'const char[5]' lvalue Var {{.*}} 'AttrVal1' 'const char[5]'
  // CHECK-NEXT:           ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:             value: LValue
  // CHECK-NEXT:             SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:               NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:               ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:                 DeclRefExpr {{.*}} 'const char[5]' lvalue Var {{.*}} 'AttrVal2' 'const char[5]'
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit used constexpr SpecialClassStructTemplate5 'void () noexcept'
  // CHECK-NEXT:       CompoundStmt
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr SpecialClassStructTemplate5 'void (const SpecialClassStructTemplate5<AttrName1, AttrName2, AttrVal1, AttrVal2> &)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'const SpecialClassStructTemplate5<AttrName1, AttrName2, AttrVal1, AttrVal2> &'
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr SpecialClassStructTemplate5 'void (SpecialClassStructTemplate5<AttrName1, AttrName2, AttrVal1, AttrVal2> &&)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'SpecialClassStructTemplate5<AttrName1, AttrName2, AttrVal1, AttrVal2> &&'
  // CHECK-NEXT:   ClassTemplateSpecializationDecl {{.*}} struct SpecialClassStructTemplate5 definition
  // CHECK-NEXT:     DefinitionData
  // CHECK-NEXT:       DefaultConstructor
  // CHECK-NEXT:       CopyConstructor
  // CHECK-NEXT:       MoveConstructor
  // CHECK-NEXT:       CopyAssignment
  // CHECK-NEXT:       MoveAssignment
  // CHECK-NEXT:       Destructor
  // CHECK-NEXT:     TemplateArgument pack
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrName1' 'const char[6]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrName2' 'const char[6]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrName3' 'const char[6]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrVal1' 'const char[5]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrVal2' 'const char[5]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrVal3' 'const char[5]'
  // CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct SpecialClassStructTemplate5
  // CHECK-NEXT:     CXXMethodDecl {{.*}} __init 'void (int)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} x 'int'
  // CHECK-NEXT:         SYCLAddIRAttributesKernelParameterAttr
  // CHECK-NEXT:           ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:             value: LValue
  // CHECK-NEXT:             SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:               NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:               ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:                 DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName1' 'const char[6]'
  // CHECK-NEXT:           ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:             value: LValue
  // CHECK-NEXT:             SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:               NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:               ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:                 DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName2' 'const char[6]'
  // CHECK-NEXT:           ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:             value: LValue
  // CHECK-NEXT:             SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:               NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:               ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:                 DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName3' 'const char[6]'
  // CHECK-NEXT:           ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:             value: LValue
  // CHECK-NEXT:             SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:               NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:               ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:                 DeclRefExpr {{.*}} 'const char[5]' lvalue Var {{.*}} 'AttrVal1' 'const char[5]'
  // CHECK-NEXT:           ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:             value: LValue
  // CHECK-NEXT:             SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:               NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:               ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:                 DeclRefExpr {{.*}} 'const char[5]' lvalue Var {{.*}} 'AttrVal2' 'const char[5]'
  // CHECK-NEXT:           ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:             value: LValue
  // CHECK-NEXT:             SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:               NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:               ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:                 DeclRefExpr {{.*}} 'const char[5]' lvalue Var {{.*}} 'AttrVal3' 'const char[5]'
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit used constexpr SpecialClassStructTemplate5 'void () noexcept'
  // CHECK-NEXT:       CompoundStmt
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr SpecialClassStructTemplate5 'void (const SpecialClassStructTemplate5<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2, AttrVal3> &)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'const SpecialClassStructTemplate5<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2, AttrVal3> &'
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr SpecialClassStructTemplate5 'void (SpecialClassStructTemplate5<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2, AttrVal3> &&)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'SpecialClassStructTemplate5<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2, AttrVal3> &&'
  // CHECK-NEXT:   ClassTemplateSpecializationDecl {{.*}} struct SpecialClassStructTemplate5 definition
  // CHECK-NEXT:     DefinitionData
  // CHECK-NEXT:       DefaultConstructor
  // CHECK-NEXT:       CopyConstructor
  // CHECK-NEXT:       MoveConstructor
  // CHECK-NEXT:       CopyAssignment
  // CHECK-NEXT:       MoveAssignment
  // CHECK-NEXT:       Destructor
  // CHECK-NEXT:     TemplateArgument pack
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrName1' 'const char[6]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrName2' 'const char[6]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrName3' 'const char[6]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrVal1' 'const char[5]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrVal2' 'const char[5]'
  // CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct SpecialClassStructTemplate5
  // CHECK-NEXT:     CXXMethodDecl {{.*}} __init 'void (int)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} x 'int'
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit used constexpr SpecialClassStructTemplate5 'void () noexcept'
  // CHECK-NEXT:       CompoundStmt
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr SpecialClassStructTemplate5 'void (const SpecialClassStructTemplate5<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2> &)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'const SpecialClassStructTemplate5<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2> &'
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr SpecialClassStructTemplate5 'void (SpecialClassStructTemplate5<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2> &&)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'SpecialClassStructTemplate5<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2> &&'
  SpecialClassStructTemplate5<AttrName1, AttrVal1> InstantiatedSCS9;
  SpecialClassStructTemplate5<AttrName1, AttrName2, AttrVal1, AttrVal2> InstantiatedSCS10;
  SpecialClassStructTemplate5<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2, AttrVal3> InstantiatedSCS11;
  SpecialClassStructTemplate5<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2> InstantiatedSCS12; // expected-note {{in instantiation of template class 'SpecialClassStructTemplate5<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2>' requested here}}

  // CHECK:      ClassTemplateDecl {{.*}} SpecialClassStructTemplate6
  // CHECK-NEXT:   NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:   CXXRecordDecl {{.*}} struct SpecialClassStructTemplate6 definition
  // CHECK-NEXT:     DefinitionData
  // CHECK-NEXT:       DefaultConstructor
  // CHECK-NEXT:       CopyConstructor
  // CHECK-NEXT:       MoveConstructor
  // CHECK-NEXT:       CopyAssignment
  // CHECK-NEXT:       MoveAssignment
  // CHECK-NEXT:       Destructor
  // CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct SpecialClassStructTemplate6
  // CHECK-NEXT:     CXXMethodDecl {{.*}} __init 'void (int)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} x 'int'
  // CHECK-NEXT:         SYCLAddIRAttributesKernelParameterAttr
  // CHECK-NEXT:           InitListExpr {{.*}} 'void'
  // CHECK-NEXT:             StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
  // CHECK-NEXT:             StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
  // CHECK-NEXT:           PackExpansionExpr {{.*}} '<dependent type>'
  // CHECK-NEXT:             DeclRefExpr {{.*}} 'const char *' NonTypeTemplateParm {{.*}} 'Strs' 'const char *'
  // CHECK-NEXT:       CompoundStmt
  // CHECK-NEXT:   ClassTemplateSpecializationDecl {{.*}} struct SpecialClassStructTemplate6 definition
  // CHECK-NEXT:     DefinitionData
  // CHECK-NEXT:       DefaultConstructor
  // CHECK-NEXT:       CopyConstructor
  // CHECK-NEXT:       MoveConstructor
  // CHECK-NEXT:       CopyAssignment
  // CHECK-NEXT:       MoveAssignment
  // CHECK-NEXT:       Destructor
  // CHECK-NEXT:     TemplateArgument pack
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrName1' 'const char[6]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrVal1' 'const char[5]'
  // CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct SpecialClassStructTemplate6
  // CHECK-NEXT:     CXXMethodDecl {{.*}} __init 'void (int)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} x 'int'
  // CHECK-NEXT:         SYCLAddIRAttributesKernelParameterAttr
  // CHECK-NEXT:           InitListExpr {{.*}} 'void'
  // CHECK-NEXT:             StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
  // CHECK-NEXT:             StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
  // CHECK-NEXT:           ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:             value: LValue
  // CHECK-NEXT:             SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:               NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:               ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:                 DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName1' 'const char[6]'
  // CHECK-NEXT:           ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:             value: LValue
  // CHECK-NEXT:             SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:               NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:               ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:                 DeclRefExpr {{.*}} 'const char[5]' lvalue Var {{.*}} 'AttrVal1' 'const char[5]'
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit used constexpr SpecialClassStructTemplate6 'void () noexcept'
  // CHECK-NEXT:       CompoundStmt
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr SpecialClassStructTemplate6 'void (const SpecialClassStructTemplate6<AttrName1, AttrVal1> &)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'const SpecialClassStructTemplate6<AttrName1, AttrVal1> &'
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr SpecialClassStructTemplate6 'void (SpecialClassStructTemplate6<AttrName1, AttrVal1> &&)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'SpecialClassStructTemplate6<AttrName1, AttrVal1> &&'
  // CHECK-NEXT:   ClassTemplateSpecializationDecl {{.*}} struct SpecialClassStructTemplate6 definition
  // CHECK-NEXT:     DefinitionData
  // CHECK-NEXT:       DefaultConstructor
  // CHECK-NEXT:       CopyConstructor
  // CHECK-NEXT:       MoveConstructor
  // CHECK-NEXT:       CopyAssignment
  // CHECK-NEXT:       MoveAssignment
  // CHECK-NEXT:       Destructor
  // CHECK-NEXT:     TemplateArgument pack
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrName1' 'const char[6]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrName2' 'const char[6]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrVal1' 'const char[5]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrVal2' 'const char[5]'
  // CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct SpecialClassStructTemplate6
  // CHECK-NEXT:     CXXMethodDecl {{.*}} __init 'void (int)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} x 'int'
  // CHECK-NEXT:         SYCLAddIRAttributesKernelParameterAttr
  // CHECK-NEXT:           InitListExpr {{.*}} 'void'
  // CHECK-NEXT:             StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
  // CHECK-NEXT:             StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
  // CHECK-NEXT:           ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:             value: LValue
  // CHECK-NEXT:             SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:               NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:               ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:                 DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName1' 'const char[6]'
  // CHECK-NEXT:           ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:             value: LValue
  // CHECK-NEXT:             SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:               NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:               ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:                 DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName2' 'const char[6]'
  // CHECK-NEXT:           ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:             value: LValue
  // CHECK-NEXT:             SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:               NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:               ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:                 DeclRefExpr {{.*}} 'const char[5]' lvalue Var {{.*}} 'AttrVal1' 'const char[5]'
  // CHECK-NEXT:           ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:             value: LValue
  // CHECK-NEXT:             SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:               NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:               ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:                 DeclRefExpr {{.*}} 'const char[5]' lvalue Var {{.*}} 'AttrVal2' 'const char[5]'
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit used constexpr SpecialClassStructTemplate6 'void () noexcept'
  // CHECK-NEXT:       CompoundStmt
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr SpecialClassStructTemplate6 'void (const SpecialClassStructTemplate6<AttrName1, AttrName2, AttrVal1, AttrVal2> &)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'const SpecialClassStructTemplate6<AttrName1, AttrName2, AttrVal1, AttrVal2> &'
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr SpecialClassStructTemplate6 'void (SpecialClassStructTemplate6<AttrName1, AttrName2, AttrVal1, AttrVal2> &&)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'SpecialClassStructTemplate6<AttrName1, AttrName2, AttrVal1, AttrVal2> &&'
  // CHECK-NEXT:   ClassTemplateSpecializationDecl {{.*}} struct SpecialClassStructTemplate6 definition
  // CHECK-NEXT:     DefinitionData
  // CHECK-NEXT:       DefaultConstructor
  // CHECK-NEXT:       CopyConstructor
  // CHECK-NEXT:       MoveConstructor
  // CHECK-NEXT:       CopyAssignment
  // CHECK-NEXT:       MoveAssignment
  // CHECK-NEXT:       Destructor
  // CHECK-NEXT:     TemplateArgument pack
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrName1' 'const char[6]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrName2' 'const char[6]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrName3' 'const char[6]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrVal1' 'const char[5]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrVal2' 'const char[5]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrVal3' 'const char[5]'
  // CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct SpecialClassStructTemplate6
  // CHECK-NEXT:     CXXMethodDecl {{.*}} __init 'void (int)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} x 'int'
  // CHECK-NEXT:         SYCLAddIRAttributesKernelParameterAttr
  // CHECK-NEXT:           InitListExpr {{.*}} 'void'
  // CHECK-NEXT:             StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
  // CHECK-NEXT:             StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
  // CHECK-NEXT:           ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:             value: LValue
  // CHECK-NEXT:             SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:               NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:               ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:                 DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName1' 'const char[6]'
  // CHECK-NEXT:           ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:             value: LValue
  // CHECK-NEXT:             SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:               NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:               ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:                 DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName2' 'const char[6]'
  // CHECK-NEXT:           ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:             value: LValue
  // CHECK-NEXT:             SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:               NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:               ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:                 DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName3' 'const char[6]'
  // CHECK-NEXT:           ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:             value: LValue
  // CHECK-NEXT:             SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:               NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:               ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:                 DeclRefExpr {{.*}} 'const char[5]' lvalue Var {{.*}} 'AttrVal1' 'const char[5]'
  // CHECK-NEXT:           ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:             value: LValue
  // CHECK-NEXT:             SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:               NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:               ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:                 DeclRefExpr {{.*}} 'const char[5]' lvalue Var {{.*}} 'AttrVal2' 'const char[5]'
  // CHECK-NEXT:           ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:             value: LValue
  // CHECK-NEXT:             SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:               NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:               ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:                 DeclRefExpr {{.*}} 'const char[5]' lvalue Var {{.*}} 'AttrVal3' 'const char[5]'
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit used constexpr SpecialClassStructTemplate6 'void () noexcept'
  // CHECK-NEXT:       CompoundStmt
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr SpecialClassStructTemplate6 'void (const SpecialClassStructTemplate6<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2, AttrVal3> &)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'const SpecialClassStructTemplate6<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2, AttrVal3> &'
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr SpecialClassStructTemplate6 'void (SpecialClassStructTemplate6<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2, AttrVal3> &&)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'SpecialClassStructTemplate6<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2, AttrVal3> &&'
  // CHECK-NEXT:   ClassTemplateSpecializationDecl {{.*}} struct SpecialClassStructTemplate6 definition
  // CHECK-NEXT:     DefinitionData
  // CHECK-NEXT:       DefaultConstructor
  // CHECK-NEXT:       CopyConstructor
  // CHECK-NEXT:       MoveConstructor
  // CHECK-NEXT:       CopyAssignment
  // CHECK-NEXT:       MoveAssignment
  // CHECK-NEXT:       Destructor
  // CHECK-NEXT:     TemplateArgument pack
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrName1' 'const char[6]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrName2' 'const char[6]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrName3' 'const char[6]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrVal1' 'const char[5]'
  // CHECK-NEXT:       TemplateArgument decl
  // CHECK-NEXT:         Var {{.*}} 'AttrVal2' 'const char[5]'
  // CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct SpecialClassStructTemplate6
  // CHECK-NEXT:     CXXMethodDecl {{.*}} __init 'void (int)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} x 'int'
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit used constexpr SpecialClassStructTemplate6 'void () noexcept'
  // CHECK-NEXT:       CompoundStmt
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr SpecialClassStructTemplate6 'void (const SpecialClassStructTemplate6<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2> &)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'const SpecialClassStructTemplate6<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2> &'
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr SpecialClassStructTemplate6 'void (SpecialClassStructTemplate6<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2> &&)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'SpecialClassStructTemplate6<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2> &&'
  SpecialClassStructTemplate6<AttrName1, AttrVal1> InstantiatedSCS13;
  SpecialClassStructTemplate6<AttrName1, AttrName2, AttrVal1, AttrVal2> InstantiatedSCS14;
  SpecialClassStructTemplate6<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2, AttrVal3> InstantiatedSCS15;
  SpecialClassStructTemplate6<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2> InstantiatedSCS16; // expected-note {{in instantiation of template class 'SpecialClassStructTemplate6<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2>' requested here}}
}
