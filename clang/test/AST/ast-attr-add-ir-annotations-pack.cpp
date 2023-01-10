// RUN: not %clang_cc1 -fsycl-is-device -ast-dump %s | FileCheck %s

// Tests the AST produced from instantiating templates using the
// __sycl_detail__::add_ir_annotations_member attribute with pack expansion
// arguments.

constexpr const char AttrName1[] = "Attr1";
constexpr const char AttrName2[] = "Attr2";
constexpr const char AttrName3[] = "Attr3";
constexpr const char AttrVal1[] = "Val1";
constexpr const char AttrVal2[] = "Val2";
constexpr const char AttrVal3[] = "Val3";

template <int... Is> struct ClassWithAnnotFieldTemplate1 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member("Attr1", "Attr2", "Attr3", Is...)]];
};
template <int... Is> struct ClassWithAnnotFieldTemplate2 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member({"Attr1", "Attr3"}, "Attr1", "Attr2", "Attr3", Is...)]];
};
template <const char *...Names> struct ClassWithAnnotFieldTemplate3 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(Names..., 1, 2, 3)]];
};
template <const char *...Names> struct ClassWithAnnotFieldTemplate4 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member({"Attr1", "Attr3"}, Names..., 1, 2, 3)]];
};
template <const char *...Strs> struct ClassWithAnnotFieldTemplate5 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(Strs...)]];
};
template <const char *...Strs> struct ClassWithAnnotFieldTemplate6 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member({"Attr1", "Attr3"}, Strs...)]];
};

void InstantiateClassWithAnnotFieldTemplates() {
  // CHECK:      ClassTemplateDecl {{.*}} ClassWithAnnotFieldTemplate1
  // CHECK-NEXT:   NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 0 index 0 ... Is
  // CHECK-NEXT:   CXXRecordDecl {{.*}} struct ClassWithAnnotFieldTemplate1 definition
  // CHECK-NEXT:     DefinitionData
  // CHECK-NEXT:       DefaultConstructor
  // CHECK-NEXT:       CopyConstructor
  // CHECK-NEXT:       MoveConstructor
  // CHECK-NEXT:       CopyAssignment
  // CHECK-NEXT:       MoveAssignment
  // CHECK-NEXT:       Destructor
  // CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct ClassWithAnnotFieldTemplate1
  // CHECK-NEXT:     FieldDecl {{.*}} ptr 'int *'
  // CHECK-NEXT:       SYCLAddIRAnnotationsMemberAttr
  // CHECK-NEXT:         ConstantExpr {{.*}} 'const char[6]' lvalue
  // CHECK-NEXT:           value: LValue
  // CHECK-NEXT:           StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
  // CHECK-NEXT:         ConstantExpr {{.*}} 'const char[6]' lvalue
  // CHECK-NEXT:           value: LValue
  // CHECK-NEXT:           StringLiteral {{.*}} 'const char[6]' lvalue "Attr2"
  // CHECK-NEXT:         ConstantExpr {{.*}} 'const char[6]' lvalue
  // CHECK-NEXT:           value: LValue
  // CHECK-NEXT:           StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
  // CHECK-NEXT:         PackExpansionExpr {{.*}} '<dependent type>'
  // CHECK-NEXT:           DeclRefExpr {{.*}} 'int' NonTypeTemplateParm {{.*}} 'Is' 'int'
  // CHECK-NEXT:   ClassTemplateSpecializationDecl {{.*}} struct ClassWithAnnotFieldTemplate1 definition
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
  // CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct ClassWithAnnotFieldTemplate1
  // CHECK-NEXT:     FieldDecl {{.*}} referenced ptr 'int *'
  // CHECK-NEXT:       SYCLAddIRAnnotationsMemberAttr
  // CHECK-NEXT:         ConstantExpr {{.*}} 'const char[6]' lvalue
  // CHECK-NEXT:           value: LValue
  // CHECK-NEXT:           StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
  // CHECK-NEXT:         ConstantExpr {{.*}} 'const char[6]' lvalue
  // CHECK-NEXT:           value: LValue
  // CHECK-NEXT:           StringLiteral {{.*}} 'const char[6]' lvalue "Attr2"
  // CHECK-NEXT:         ConstantExpr {{.*}} 'const char[6]' lvalue
  // CHECK-NEXT:           value: LValue
  // CHECK-NEXT:           StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
  // CHECK-NEXT:         ConstantExpr {{.*}} 'int'
  // CHECK-NEXT:           value: Int 1
  // CHECK-NEXT:           SubstNonTypeTemplateParmExpr {{.*}} 'int'
  // CHECK-NEXT:             NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 0 index 0 ... Is
  // CHECK-NEXT:             IntegerLiteral {{.*}} 'int' 1
  // CHECK-NEXT:         ConstantExpr {{.*}} 'int'
  // CHECK-NEXT:           value: Int 2
  // CHECK-NEXT:           SubstNonTypeTemplateParmExpr {{.*}} 'int'
  // CHECK-NEXT:             NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 0 index 0 ... Is
  // CHECK-NEXT:             IntegerLiteral {{.*}} 'int' 2
  // CHECK-NEXT:         ConstantExpr {{.*}} 'int'
  // CHECK-NEXT:           value: Int 3
  // CHECK-NEXT:           SubstNonTypeTemplateParmExpr {{.*}} 'int'
  // CHECK-NEXT:             NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 0 index 0 ... Is
  // CHECK-NEXT:             IntegerLiteral {{.*}} 'int' 3
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit used ClassWithAnnotFieldTemplate1 'void () noexcept'
  // CHECK-NEXT:       CompoundStmt
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr ClassWithAnnotFieldTemplate1 'void (const ClassWithAnnotFieldTemplate1<1, 2, 3> &)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'const ClassWithAnnotFieldTemplate1<1, 2, 3> &'
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr ClassWithAnnotFieldTemplate1 'void (ClassWithAnnotFieldTemplate1<1, 2, 3> &&)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'ClassWithAnnotFieldTemplate1<1, 2, 3> &&'
  // CHECK-NEXT:   ClassTemplateSpecializationDecl {{.*}} struct ClassWithAnnotFieldTemplate1 definition
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
  // CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct ClassWithAnnotFieldTemplate1
  // CHECK-NEXT:     FieldDecl {{.*}} referenced ptr 'int *'
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit used ClassWithAnnotFieldTemplate1 'void () noexcept'
  // CHECK-NEXT:       CompoundStmt
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr ClassWithAnnotFieldTemplate1 'void (const ClassWithAnnotFieldTemplate1<1, 2> &)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'const ClassWithAnnotFieldTemplate1<1, 2> &'
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr ClassWithAnnotFieldTemplate1 'void (ClassWithAnnotFieldTemplate1<1, 2> &&)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'ClassWithAnnotFieldTemplate1<1, 2> &&'
  ClassWithAnnotFieldTemplate1<1, 2, 3> InstantiatedCWAFS1;
  ClassWithAnnotFieldTemplate1<1, 2> InstantiatedCWAFS2;

  // CHECK:      ClassTemplateDecl {{.*}} ClassWithAnnotFieldTemplate2
  // CHECK-NEXT:   NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 0 index 0 ... Is
  // CHECK-NEXT:   CXXRecordDecl {{.*}} struct ClassWithAnnotFieldTemplate2 definition
  // CHECK-NEXT:     DefinitionData
  // CHECK-NEXT:       DefaultConstructor
  // CHECK-NEXT:       CopyConstructor
  // CHECK-NEXT:       MoveConstructor
  // CHECK-NEXT:       CopyAssignment
  // CHECK-NEXT:       MoveAssignment
  // CHECK-NEXT:       Destructor
  // CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct ClassWithAnnotFieldTemplate2
  // CHECK-NEXT:     FieldDecl {{.*}} ptr 'int *'
  // CHECK-NEXT:       SYCLAddIRAnnotationsMemberAttr
  // CHECK-NEXT:         InitListExpr {{.*}} 'void'
  // CHECK-NEXT:           StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
  // CHECK-NEXT:           StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
  // CHECK-NEXT:         ConstantExpr {{.*}} 'const char[6]' lvalue
  // CHECK-NEXT:           value: LValue
  // CHECK-NEXT:           StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
  // CHECK-NEXT:         ConstantExpr {{.*}} 'const char[6]' lvalue
  // CHECK-NEXT:           value: LValue
  // CHECK-NEXT:           StringLiteral {{.*}} 'const char[6]' lvalue "Attr2"
  // CHECK-NEXT:         ConstantExpr {{.*}} 'const char[6]' lvalue
  // CHECK-NEXT:           value: LValue
  // CHECK-NEXT:           StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
  // CHECK-NEXT:         PackExpansionExpr {{.*}} '<dependent type>'
  // CHECK-NEXT:           DeclRefExpr {{.*}} 'int' NonTypeTemplateParm {{.*}} 'Is' 'int'
  // CHECK-NEXT:   ClassTemplateSpecializationDecl {{.*}} struct ClassWithAnnotFieldTemplate2 definition
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
  // CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct ClassWithAnnotFieldTemplate2
  // CHECK-NEXT:     FieldDecl {{.*}} referenced ptr 'int *'
  // CHECK-NEXT:       SYCLAddIRAnnotationsMemberAttr
  // CHECK-NEXT:         InitListExpr {{.*}} 'void'
  // CHECK-NEXT:           StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
  // CHECK-NEXT:           StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
  // CHECK-NEXT:         ConstantExpr {{.*}} 'const char[6]' lvalue
  // CHECK-NEXT:           value: LValue
  // CHECK-NEXT:           StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
  // CHECK-NEXT:         ConstantExpr {{.*}} 'const char[6]' lvalue
  // CHECK-NEXT:           value: LValue
  // CHECK-NEXT:           StringLiteral {{.*}} 'const char[6]' lvalue "Attr2"
  // CHECK-NEXT:         ConstantExpr {{.*}} 'const char[6]' lvalue
  // CHECK-NEXT:           value: LValue
  // CHECK-NEXT:           StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
  // CHECK-NEXT:         ConstantExpr {{.*}} 'int'
  // CHECK-NEXT:           value: Int 1
  // CHECK-NEXT:           SubstNonTypeTemplateParmExpr {{.*}} 'int'
  // CHECK-NEXT:             NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 0 index 0 ... Is
  // CHECK-NEXT:             IntegerLiteral {{.*}} 'int' 1
  // CHECK-NEXT:         ConstantExpr {{.*}} 'int'
  // CHECK-NEXT:           value: Int 2
  // CHECK-NEXT:           SubstNonTypeTemplateParmExpr {{.*}} 'int'
  // CHECK-NEXT:             NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 0 index 0 ... Is
  // CHECK-NEXT:             IntegerLiteral {{.*}} 'int' 2
  // CHECK-NEXT:         ConstantExpr {{.*}} 'int'
  // CHECK-NEXT:           value: Int 3
  // CHECK-NEXT:           SubstNonTypeTemplateParmExpr {{.*}} 'int'
  // CHECK-NEXT:             NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 0 index 0 ... Is
  // CHECK-NEXT:             IntegerLiteral {{.*}} 'int' 3
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit used ClassWithAnnotFieldTemplate2 'void () noexcept'
  // CHECK-NEXT:       CompoundStmt
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr ClassWithAnnotFieldTemplate2 'void (const ClassWithAnnotFieldTemplate2<1, 2, 3> &)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'const ClassWithAnnotFieldTemplate2<1, 2, 3> &'
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr ClassWithAnnotFieldTemplate2 'void (ClassWithAnnotFieldTemplate2<1, 2, 3> &&)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'ClassWithAnnotFieldTemplate2<1, 2, 3> &&'
  // CHECK-NEXT:   ClassTemplateSpecializationDecl {{.*}} struct ClassWithAnnotFieldTemplate2 definition
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
  // CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct ClassWithAnnotFieldTemplate2
  // CHECK-NEXT:     FieldDecl {{.*}} referenced ptr 'int *'
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit used ClassWithAnnotFieldTemplate2 'void () noexcept'
  // CHECK-NEXT:       CompoundStmt
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr ClassWithAnnotFieldTemplate2 'void (const ClassWithAnnotFieldTemplate2<1, 2> &)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'const ClassWithAnnotFieldTemplate2<1, 2> &'
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr ClassWithAnnotFieldTemplate2 'void (ClassWithAnnotFieldTemplate2<1, 2> &&)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'ClassWithAnnotFieldTemplate2<1, 2> &&'
  ClassWithAnnotFieldTemplate2<1, 2, 3> InstantiatedCWAFS3;
  ClassWithAnnotFieldTemplate2<1, 2> InstantiatedCWAFS4;

  // CHECK:      ClassTemplateDecl {{.*}} ClassWithAnnotFieldTemplate3
  // CHECK-NEXT:   NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Names
  // CHECK-NEXT:   CXXRecordDecl {{.*}} struct ClassWithAnnotFieldTemplate3 definition
  // CHECK-NEXT:     DefinitionData
  // CHECK-NEXT:       DefaultConstructor
  // CHECK-NEXT:       CopyConstructor
  // CHECK-NEXT:       MoveConstructor
  // CHECK-NEXT:       CopyAssignment
  // CHECK-NEXT:       MoveAssignment
  // CHECK-NEXT:       Destructor
  // CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct ClassWithAnnotFieldTemplate3
  // CHECK-NEXT:     FieldDecl {{.*}} ptr 'int *'
  // CHECK-NEXT:       SYCLAddIRAnnotationsMemberAttr
  // CHECK-NEXT:         PackExpansionExpr {{.*}} '<dependent type>'
  // CHECK-NEXT:           DeclRefExpr {{.*}} 'const char *' NonTypeTemplateParm {{.*}} 'Names' 'const char *'
  // CHECK-NEXT:         ConstantExpr {{.*}} 'int'
  // CHECK-NEXT:           value: Int 1
  // CHECK-NEXT:           IntegerLiteral {{.*}} 'int' 1
  // CHECK-NEXT:         ConstantExpr {{.*}} 'int'
  // CHECK-NEXT:           value: Int 2
  // CHECK-NEXT:           IntegerLiteral {{.*}} 'int' 2
  // CHECK-NEXT:         ConstantExpr {{.*}} 'int'
  // CHECK-NEXT:           value: Int 3
  // CHECK-NEXT:           IntegerLiteral {{.*}} 'int' 3
  // CHECK-NEXT:   ClassTemplateSpecializationDecl {{.*}} struct ClassWithAnnotFieldTemplate3 definition
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
  // CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct ClassWithAnnotFieldTemplate3
  // CHECK-NEXT:     FieldDecl {{.*}} referenced ptr 'int *'
  // CHECK-NEXT:       SYCLAddIRAnnotationsMemberAttr
  // CHECK-NEXT:         ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:           value: LValue
  // CHECK-NEXT:           SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:             NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Names
  // CHECK-NEXT:             ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:               DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName1' 'const char[6]'
  // CHECK-NEXT:         ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:           value: LValue
  // CHECK-NEXT:           SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:             NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Names
  // CHECK-NEXT:             ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:               DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName2' 'const char[6]'
  // CHECK-NEXT:         ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:           value: LValue
  // CHECK-NEXT:           SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:             NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Names
  // CHECK-NEXT:             ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:               DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName3' 'const char[6]'
  // CHECK-NEXT:         ConstantExpr {{.*}} 'int'
  // CHECK-NEXT:           value: Int 1
  // CHECK-NEXT:           IntegerLiteral {{.*}} 'int' 1
  // CHECK-NEXT:         ConstantExpr {{.*}} 'int'
  // CHECK-NEXT:           value: Int 2
  // CHECK-NEXT:           IntegerLiteral {{.*}} 'int' 2
  // CHECK-NEXT:         ConstantExpr {{.*}} 'int'
  // CHECK-NEXT:           value: Int 3
  // CHECK-NEXT:           IntegerLiteral {{.*}} 'int' 3
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit used ClassWithAnnotFieldTemplate3 'void () noexcept'
  // CHECK-NEXT:       CompoundStmt
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr ClassWithAnnotFieldTemplate3 'void (const ClassWithAnnotFieldTemplate3<AttrName1, AttrName2, AttrName3> &)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'const ClassWithAnnotFieldTemplate3<AttrName1, AttrName2, AttrName3> &'
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr ClassWithAnnotFieldTemplate3 'void (ClassWithAnnotFieldTemplate3<AttrName1, AttrName2, AttrName3> &&)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'ClassWithAnnotFieldTemplate3<AttrName1, AttrName2, AttrName3> &&'
  // CHECK-NEXT:   ClassTemplateSpecializationDecl {{.*}} struct ClassWithAnnotFieldTemplate3 definition
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
  // CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct ClassWithAnnotFieldTemplate3
  // CHECK-NEXT:     FieldDecl {{.*}} referenced ptr 'int *'
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit used ClassWithAnnotFieldTemplate3 'void () noexcept'
  // CHECK-NEXT:       CompoundStmt
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr ClassWithAnnotFieldTemplate3 'void (const ClassWithAnnotFieldTemplate3<AttrName1, AttrName2> &)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'const ClassWithAnnotFieldTemplate3<AttrName1, AttrName2> &'
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr ClassWithAnnotFieldTemplate3 'void (ClassWithAnnotFieldTemplate3<AttrName1, AttrName2> &&)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'ClassWithAnnotFieldTemplate3<AttrName1, AttrName2> &&'
  ClassWithAnnotFieldTemplate3<AttrName1, AttrName2, AttrName3> InstantiatedCWAFS5;
  ClassWithAnnotFieldTemplate3<AttrName1, AttrName2> InstantiatedCWAFS6;

  // CHECK:      ClassTemplateDecl {{.*}} ClassWithAnnotFieldTemplate4
  // CHECK-NEXT:   NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Names
  // CHECK-NEXT:   CXXRecordDecl {{.*}} struct ClassWithAnnotFieldTemplate4 definition
  // CHECK-NEXT:     DefinitionData
  // CHECK-NEXT:       DefaultConstructor
  // CHECK-NEXT:       CopyConstructor
  // CHECK-NEXT:       MoveConstructor
  // CHECK-NEXT:       CopyAssignment
  // CHECK-NEXT:       MoveAssignment
  // CHECK-NEXT:       Destructor
  // CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct ClassWithAnnotFieldTemplate4
  // CHECK-NEXT:     FieldDecl {{.*}} ptr 'int *'
  // CHECK-NEXT:       SYCLAddIRAnnotationsMemberAttr
  // CHECK-NEXT:         InitListExpr {{.*}} 'void'
  // CHECK-NEXT:           StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
  // CHECK-NEXT:           StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
  // CHECK-NEXT:         PackExpansionExpr {{.*}} '<dependent type>'
  // CHECK-NEXT:           DeclRefExpr {{.*}} 'const char *' NonTypeTemplateParm {{.*}} 'Names' 'const char *'
  // CHECK-NEXT:         ConstantExpr {{.*}} 'int'
  // CHECK-NEXT:           value: Int 1
  // CHECK-NEXT:           IntegerLiteral {{.*}} 'int' 1
  // CHECK-NEXT:         ConstantExpr {{.*}} 'int'
  // CHECK-NEXT:           value: Int 2
  // CHECK-NEXT:           IntegerLiteral {{.*}} 'int' 2
  // CHECK-NEXT:         ConstantExpr {{.*}} 'int'
  // CHECK-NEXT:           value: Int 3
  // CHECK-NEXT:           IntegerLiteral {{.*}} 'int' 3
  // CHECK-NEXT:   ClassTemplateSpecializationDecl {{.*}} struct ClassWithAnnotFieldTemplate4 definition
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
  // CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct ClassWithAnnotFieldTemplate4
  // CHECK-NEXT:     FieldDecl {{.*}} referenced ptr 'int *'
  // CHECK-NEXT:       SYCLAddIRAnnotationsMemberAttr
  // CHECK-NEXT:         InitListExpr {{.*}} 'void'
  // CHECK-NEXT:           StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
  // CHECK-NEXT:           StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
  // CHECK-NEXT:         ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:           value: LValue
  // CHECK-NEXT:           SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:             NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Names
  // CHECK-NEXT:             ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:               DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName1' 'const char[6]'
  // CHECK-NEXT:         ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:           value: LValue
  // CHECK-NEXT:           SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:             NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Names
  // CHECK-NEXT:             ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:               DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName2' 'const char[6]'
  // CHECK-NEXT:         ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:           value: LValue
  // CHECK-NEXT:           SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:             NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Names
  // CHECK-NEXT:             ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:               DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName3' 'const char[6]'
  // CHECK-NEXT:         ConstantExpr {{.*}} 'int'
  // CHECK-NEXT:           value: Int 1
  // CHECK-NEXT:           IntegerLiteral {{.*}} 'int' 1
  // CHECK-NEXT:         ConstantExpr {{.*}} 'int'
  // CHECK-NEXT:           value: Int 2
  // CHECK-NEXT:           IntegerLiteral {{.*}} 'int' 2
  // CHECK-NEXT:         ConstantExpr {{.*}} 'int'
  // CHECK-NEXT:           value: Int 3
  // CHECK-NEXT:           IntegerLiteral {{.*}} 'int' 3
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit used ClassWithAnnotFieldTemplate4 'void () noexcept'
  // CHECK-NEXT:       CompoundStmt
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr ClassWithAnnotFieldTemplate4 'void (const ClassWithAnnotFieldTemplate4<AttrName1, AttrName2, AttrName3> &)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'const ClassWithAnnotFieldTemplate4<AttrName1, AttrName2, AttrName3> &'
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr ClassWithAnnotFieldTemplate4 'void (ClassWithAnnotFieldTemplate4<AttrName1, AttrName2, AttrName3> &&)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'ClassWithAnnotFieldTemplate4<AttrName1, AttrName2, AttrName3> &&'
  // CHECK-NEXT:   ClassTemplateSpecializationDecl {{.*}} struct ClassWithAnnotFieldTemplate4 definition
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
  // CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct ClassWithAnnotFieldTemplate4
  // CHECK-NEXT:     FieldDecl {{.*}} referenced ptr 'int *'
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit used ClassWithAnnotFieldTemplate4 'void () noexcept' inline default trivial
  // CHECK-NEXT:       CompoundStmt
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr ClassWithAnnotFieldTemplate4 'void (const ClassWithAnnotFieldTemplate4<AttrName1, AttrName2> &)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'const ClassWithAnnotFieldTemplate4<AttrName1, AttrName2> &'
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr ClassWithAnnotFieldTemplate4 'void (ClassWithAnnotFieldTemplate4<AttrName1, AttrName2> &&)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'ClassWithAnnotFieldTemplate4<AttrName1, AttrName2> &&'
  ClassWithAnnotFieldTemplate4<AttrName1, AttrName2, AttrName3> InstantiatedCWAFS7;
  ClassWithAnnotFieldTemplate4<AttrName1, AttrName2> InstantiatedCWAFS8;

  // CHECK:      ClassTemplateDecl {{.*}} ClassWithAnnotFieldTemplate5
  // CHECK-NEXT:   NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:   CXXRecordDecl {{.*}} struct ClassWithAnnotFieldTemplate5 definition
  // CHECK-NEXT:     DefinitionData
  // CHECK-NEXT:       DefaultConstructor
  // CHECK-NEXT:       CopyConstructor
  // CHECK-NEXT:       MoveConstructor
  // CHECK-NEXT:       CopyAssignment
  // CHECK-NEXT:       MoveAssignment
  // CHECK-NEXT:       Destructor
  // CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct ClassWithAnnotFieldTemplate5
  // CHECK-NEXT:     FieldDecl {{.*}} ptr 'int *'
  // CHECK-NEXT:       SYCLAddIRAnnotationsMemberAttr
  // CHECK-NEXT:         PackExpansionExpr {{.*}} '<dependent type>'
  // CHECK-NEXT:           DeclRefExpr {{.*}} 'const char *' NonTypeTemplateParm {{.*}} 'Strs' 'const char *'
  // CHECK-NEXT:   ClassTemplateSpecializationDecl {{.*}} struct ClassWithAnnotFieldTemplate5 definition
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
  // CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct ClassWithAnnotFieldTemplate5
  // CHECK-NEXT:     FieldDecl {{.*}} referenced ptr 'int *'
  // CHECK-NEXT:       SYCLAddIRAnnotationsMemberAttr
  // CHECK-NEXT:         ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:           value: LValue
  // CHECK-NEXT:           SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:             NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:             ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:               DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName1' 'const char[6]'
  // CHECK-NEXT:         ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:           value: LValue
  // CHECK-NEXT:           SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:             NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:             ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:               DeclRefExpr {{.*}} 'const char[5]' lvalue Var {{.*}} 'AttrVal1' 'const char[5]'
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit used ClassWithAnnotFieldTemplate5 'void () noexcept'
  // CHECK-NEXT:       CompoundStmt
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr ClassWithAnnotFieldTemplate5 'void (const ClassWithAnnotFieldTemplate5<AttrName1, AttrVal1> &)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'const ClassWithAnnotFieldTemplate5<AttrName1, AttrVal1> &'
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr ClassWithAnnotFieldTemplate5 'void (ClassWithAnnotFieldTemplate5<AttrName1, AttrVal1> &&)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'ClassWithAnnotFieldTemplate5<AttrName1, AttrVal1> &&'
  // CHECK-NEXT:   ClassTemplateSpecializationDecl {{.*}} struct ClassWithAnnotFieldTemplate5 definition
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
  // CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct ClassWithAnnotFieldTemplate5
  // CHECK-NEXT:     FieldDecl {{.*}} referenced ptr 'int *'
  // CHECK-NEXT:       SYCLAddIRAnnotationsMemberAttr
  // CHECK-NEXT:         ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:           value: LValue
  // CHECK-NEXT:           SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:             NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:             ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:               DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName1' 'const char[6]'
  // CHECK-NEXT:         ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:           value: LValue
  // CHECK-NEXT:           SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:             NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:             ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:               DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName2' 'const char[6]'
  // CHECK-NEXT:         ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:           value: LValue
  // CHECK-NEXT:           SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:             NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:             ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:               DeclRefExpr {{.*}} 'const char[5]' lvalue Var {{.*}} 'AttrVal1' 'const char[5]'
  // CHECK-NEXT:         ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:           value: LValue
  // CHECK-NEXT:           SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:             NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:             ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:               DeclRefExpr {{.*}} 'const char[5]' lvalue Var {{.*}} 'AttrVal2' 'const char[5]'
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit used ClassWithAnnotFieldTemplate5 'void () noexcept'
  // CHECK-NEXT:       CompoundStmt
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr ClassWithAnnotFieldTemplate5 'void (const ClassWithAnnotFieldTemplate5<AttrName1, AttrName2, AttrVal1, AttrVal2> &)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'const ClassWithAnnotFieldTemplate5<AttrName1, AttrName2, AttrVal1, AttrVal2> &'
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr ClassWithAnnotFieldTemplate5 'void (ClassWithAnnotFieldTemplate5<AttrName1, AttrName2, AttrVal1, AttrVal2> &&)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'ClassWithAnnotFieldTemplate5<AttrName1, AttrName2, AttrVal1, AttrVal2> &&'
  // CHECK-NEXT:   ClassTemplateSpecializationDecl {{.*}} struct ClassWithAnnotFieldTemplate5 definition
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
  // CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct ClassWithAnnotFieldTemplate5
  // CHECK-NEXT:     FieldDecl {{.*}} referenced ptr 'int *'
  // CHECK-NEXT:       SYCLAddIRAnnotationsMemberAttr
  // CHECK-NEXT:         ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:           value: LValue
  // CHECK-NEXT:           SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:             NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:             ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:               DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName1' 'const char[6]'
  // CHECK-NEXT:         ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:           value: LValue
  // CHECK-NEXT:           SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:             NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:             ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:               DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName2' 'const char[6]'
  // CHECK-NEXT:         ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:           value: LValue
  // CHECK-NEXT:           SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:             NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:             ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:               DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName3' 'const char[6]'
  // CHECK-NEXT:         ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:           value: LValue
  // CHECK-NEXT:           SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:             NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:             ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:               DeclRefExpr {{.*}} 'const char[5]' lvalue Var {{.*}} 'AttrVal1' 'const char[5]'
  // CHECK-NEXT:         ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:           value: LValue
  // CHECK-NEXT:           SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:             NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:             ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:               DeclRefExpr {{.*}} 'const char[5]' lvalue Var {{.*}} 'AttrVal2' 'const char[5]'
  // CHECK-NEXT:         ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:           value: LValue
  // CHECK-NEXT:           SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:             NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:             ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:               DeclRefExpr {{.*}} 'const char[5]' lvalue Var {{.*}} 'AttrVal3' 'const char[5]'
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit used ClassWithAnnotFieldTemplate5 'void () noexcept'
  // CHECK-NEXT:       CompoundStmt
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr ClassWithAnnotFieldTemplate5 'void (const ClassWithAnnotFieldTemplate5<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2, AttrVal3> &)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'const ClassWithAnnotFieldTemplate5<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2, AttrVal3> &'
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr ClassWithAnnotFieldTemplate5 'void (ClassWithAnnotFieldTemplate5<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2, AttrVal3> &&)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'ClassWithAnnotFieldTemplate5<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2, AttrVal3> &&'
  // CHECK-NEXT:   ClassTemplateSpecializationDecl {{.*}} struct ClassWithAnnotFieldTemplate5 definition
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
  // CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct ClassWithAnnotFieldTemplate5
  // CHECK-NEXT:     FieldDecl {{.*}} referenced ptr 'int *'
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit used ClassWithAnnotFieldTemplate5 'void () noexcept'
  // CHECK-NEXT:       CompoundStmt
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr ClassWithAnnotFieldTemplate5 'void (const ClassWithAnnotFieldTemplate5<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2> &)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'const ClassWithAnnotFieldTemplate5<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2> &'
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr ClassWithAnnotFieldTemplate5 'void (ClassWithAnnotFieldTemplate5<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2> &&)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'ClassWithAnnotFieldTemplate5<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2> &&'
  ClassWithAnnotFieldTemplate5<AttrName1, AttrVal1> InstantiatedCWAFS9;
  ClassWithAnnotFieldTemplate5<AttrName1, AttrName2, AttrVal1, AttrVal2> InstantiatedCWAFS10;
  ClassWithAnnotFieldTemplate5<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2, AttrVal3> InstantiatedCWAFS11;
  ClassWithAnnotFieldTemplate5<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2> InstantiatedCWAFS12;

  // CHECK:      ClassTemplateDecl {{.*}} ClassWithAnnotFieldTemplate6
  // CHECK-NEXT:   NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:   CXXRecordDecl {{.*}} struct ClassWithAnnotFieldTemplate6 definition
  // CHECK-NEXT:     DefinitionData
  // CHECK-NEXT:       DefaultConstructor
  // CHECK-NEXT:       CopyConstructor
  // CHECK-NEXT:       MoveConstructor
  // CHECK-NEXT:       CopyAssignment
  // CHECK-NEXT:       MoveAssignment
  // CHECK-NEXT:       Destructor
  // CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct ClassWithAnnotFieldTemplate6
  // CHECK-NEXT:     FieldDecl {{.*}} ptr 'int *'
  // CHECK-NEXT:       SYCLAddIRAnnotationsMemberAttr
  // CHECK-NEXT:         InitListExpr {{.*}} 'void'
  // CHECK-NEXT:           StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
  // CHECK-NEXT:           StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
  // CHECK-NEXT:         PackExpansionExpr {{.*}} '<dependent type>'
  // CHECK-NEXT:           DeclRefExpr {{.*}} 'const char *' NonTypeTemplateParm {{.*}} 'Strs' 'const char *'
  // CHECK-NEXT:   ClassTemplateSpecializationDecl {{.*}} struct ClassWithAnnotFieldTemplate6 definition
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
  // CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct ClassWithAnnotFieldTemplate6
  // CHECK-NEXT:     FieldDecl {{.*}} referenced ptr 'int *'
  // CHECK-NEXT:       SYCLAddIRAnnotationsMemberAttr
  // CHECK-NEXT:         InitListExpr {{.*}} 'void'
  // CHECK-NEXT:           StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
  // CHECK-NEXT:           StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
  // CHECK-NEXT:         ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:           value: LValue
  // CHECK-NEXT:           SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:             NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:             ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:               DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName1' 'const char[6]'
  // CHECK-NEXT:         ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:           value: LValue
  // CHECK-NEXT:           SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:             NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:             ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:               DeclRefExpr {{.*}} 'const char[5]' lvalue Var {{.*}} 'AttrVal1' 'const char[5]'
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit used ClassWithAnnotFieldTemplate6 'void () noexcept'
  // CHECK-NEXT:       CompoundStmt
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr ClassWithAnnotFieldTemplate6 'void (const ClassWithAnnotFieldTemplate6<AttrName1, AttrVal1> &)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'const ClassWithAnnotFieldTemplate6<AttrName1, AttrVal1> &'
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr ClassWithAnnotFieldTemplate6 'void (ClassWithAnnotFieldTemplate6<AttrName1, AttrVal1> &&)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'ClassWithAnnotFieldTemplate6<AttrName1, AttrVal1> &&'
  // CHECK-NEXT:   ClassTemplateSpecializationDecl {{.*}} struct ClassWithAnnotFieldTemplate6 definition
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
  // CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct ClassWithAnnotFieldTemplate6
  // CHECK-NEXT:     FieldDecl {{.*}} referenced ptr 'int *'
  // CHECK-NEXT:       SYCLAddIRAnnotationsMemberAttr
  // CHECK-NEXT:         InitListExpr {{.*}} 'void'
  // CHECK-NEXT:           StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
  // CHECK-NEXT:           StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
  // CHECK-NEXT:         ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:           value: LValue
  // CHECK-NEXT:           SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:             NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:             ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:               DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName1' 'const char[6]'
  // CHECK-NEXT:         ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:           value: LValue
  // CHECK-NEXT:           SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:             NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:             ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:               DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName2' 'const char[6]'
  // CHECK-NEXT:         ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:           value: LValue
  // CHECK-NEXT:           SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:             NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:             ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:               DeclRefExpr {{.*}} 'const char[5]' lvalue Var {{.*}} 'AttrVal1' 'const char[5]'
  // CHECK-NEXT:         ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:           value: LValue
  // CHECK-NEXT:           SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:             NonTypeTemplateParmDecl {{.*}} col:26 referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:             ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:               DeclRefExpr {{.*}} 'const char[5]' lvalue Var {{.*}} 'AttrVal2' 'const char[5]'
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit used ClassWithAnnotFieldTemplate6 'void () noexcept'
  // CHECK-NEXT:       CompoundStmt
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr ClassWithAnnotFieldTemplate6 'void (const ClassWithAnnotFieldTemplate6<AttrName1, AttrName2, AttrVal1, AttrVal2> &)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'const ClassWithAnnotFieldTemplate6<AttrName1, AttrName2, AttrVal1, AttrVal2> &'
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr ClassWithAnnotFieldTemplate6 'void (ClassWithAnnotFieldTemplate6<AttrName1, AttrName2, AttrVal1, AttrVal2> &&)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'ClassWithAnnotFieldTemplate6<AttrName1, AttrName2, AttrVal1, AttrVal2> &&'
  // CHECK-NEXT:   ClassTemplateSpecializationDecl {{.*}} struct ClassWithAnnotFieldTemplate6 definition
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
  // CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct ClassWithAnnotFieldTemplate6
  // CHECK-NEXT:     FieldDecl {{.*}} referenced ptr 'int *'
  // CHECK-NEXT:       SYCLAddIRAnnotationsMemberAttr
  // CHECK-NEXT:         InitListExpr {{.*}} 'void'
  // CHECK-NEXT:           StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
  // CHECK-NEXT:           StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
  // CHECK-NEXT:         ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:           value: LValue
  // CHECK-NEXT:           SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:             NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:             ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:               DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName1' 'const char[6]'
  // CHECK-NEXT:         ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:           value: LValue
  // CHECK-NEXT:           SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:             NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:             ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:               DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName2' 'const char[6]'
  // CHECK-NEXT:         ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:           value: LValue
  // CHECK-NEXT:           SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:             NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:             ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:               DeclRefExpr {{.*}} 'const char[6]' lvalue Var {{.*}} 'AttrName3' 'const char[6]'
  // CHECK-NEXT:         ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:           value: LValue
  // CHECK-NEXT:           SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:             NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:             ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:               DeclRefExpr {{.*}} 'const char[5]' lvalue Var {{.*}} 'AttrVal1' 'const char[5]'
  // CHECK-NEXT:         ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:           value: LValue
  // CHECK-NEXT:           SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:             NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:             ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:               DeclRefExpr {{.*}} 'const char[5]' lvalue Var {{.*}} 'AttrVal2' 'const char[5]'
  // CHECK-NEXT:         ConstantExpr {{.*}} 'const char *'
  // CHECK-NEXT:           value: LValue
  // CHECK-NEXT:           SubstNonTypeTemplateParmExpr {{.*}} 'const char *'
  // CHECK-NEXT:             NonTypeTemplateParmDecl {{.*}} referenced 'const char *' depth 0 index 0 ... Strs
  // CHECK-NEXT:             ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT:               DeclRefExpr {{.*}} 'const char[5]' lvalue Var {{.*}} 'AttrVal3' 'const char[5]'
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit used ClassWithAnnotFieldTemplate6 'void () noexcept'
  // CHECK-NEXT:       CompoundStmt
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr ClassWithAnnotFieldTemplate6 'void (const ClassWithAnnotFieldTemplate6<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2, AttrVal3> &)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'const ClassWithAnnotFieldTemplate6<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2, AttrVal3> &'
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr ClassWithAnnotFieldTemplate6 'void (ClassWithAnnotFieldTemplate6<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2, AttrVal3> &&)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'ClassWithAnnotFieldTemplate6<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2, AttrVal3> &&'
  // CHECK-NEXT:   ClassTemplateSpecializationDecl {{.*}} struct ClassWithAnnotFieldTemplate6 definition
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
  // CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct ClassWithAnnotFieldTemplate6
  // CHECK-NEXT:     FieldDecl {{.*}} referenced ptr 'int *'
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit used ClassWithAnnotFieldTemplate6 'void () noexcept'
  // CHECK-NEXT:       CompoundStmt
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr ClassWithAnnotFieldTemplate6 'void (const ClassWithAnnotFieldTemplate6<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2> &)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'const ClassWithAnnotFieldTemplate6<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2> &'
  // CHECK-NEXT:     CXXConstructorDecl {{.*}} implicit constexpr ClassWithAnnotFieldTemplate6 'void (ClassWithAnnotFieldTemplate6<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2> &&)'
  // CHECK-NEXT:       ParmVarDecl {{.*}} 'ClassWithAnnotFieldTemplate6<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2> &&'
  ClassWithAnnotFieldTemplate6<AttrName1, AttrVal1> InstantiatedCWAFS13;
  ClassWithAnnotFieldTemplate6<AttrName1, AttrName2, AttrVal1, AttrVal2> InstantiatedCWAFS14;
  ClassWithAnnotFieldTemplate6<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2, AttrVal3> InstantiatedCWAFS15;
  ClassWithAnnotFieldTemplate6<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2> InstantiatedCWAFS16;

  (void)*InstantiatedCWAFS1.ptr;
  (void)*InstantiatedCWAFS2.ptr;
  (void)*InstantiatedCWAFS3.ptr;
  (void)*InstantiatedCWAFS4.ptr;
  (void)*InstantiatedCWAFS5.ptr;
  (void)*InstantiatedCWAFS6.ptr;
  (void)*InstantiatedCWAFS7.ptr;
  (void)*InstantiatedCWAFS8.ptr;
  (void)*InstantiatedCWAFS9.ptr;
  (void)*InstantiatedCWAFS10.ptr;
  (void)*InstantiatedCWAFS11.ptr;
  (void)*InstantiatedCWAFS12.ptr;
  (void)*InstantiatedCWAFS13.ptr;
  (void)*InstantiatedCWAFS14.ptr;
  (void)*InstantiatedCWAFS15.ptr;
  (void)*InstantiatedCWAFS16.ptr;
}
