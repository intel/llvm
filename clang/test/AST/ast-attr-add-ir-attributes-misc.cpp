// RUN: %clang_cc1 -fsycl-is-device -std=gnu++11 -ast-dump %s | FileCheck %s

// Tests the AST produced from various uses of add_ir_attributes_* attributes.

// CHECK:      FunctionDecl [[OverloadedFunction1ID1:0x[0-9a-f]+]] {{.*}} OverloadedFunction1 'void (float)'
// CHECK-NEXT:   ParmVarDecl {{.*}} 'float'
// CHECK-NEXT:   SYCLAddIRAttributesFunctionAttr
// CHECK-NEXT:     ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:       value: LValue
// CHECK-NEXT:       StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
// CHECK-NEXT:     ConstantExpr {{.*}} 'int'
// CHECK-NEXT:       value: Int 1
// CHECK-NEXT:       IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT: FunctionDecl {{.*}} OverloadedFunction1 'void (int)'
// CHECK-NEXT:   ParmVarDecl {{.*}} 'int'
// CHECK-NEXT:   CompoundStmt
// CHECK-NEXT:   SYCLAddIRAttributesFunctionAttr
// CHECK-NEXT:     ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:       value: LValue
// CHECK-NEXT:       StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
// CHECK-NEXT:     ConstantExpr {{.*}} 'bool'
// CHECK-NEXT:       value: Int 0
// CHECK-NEXT:       CXXBoolLiteralExpr {{.*}} 'bool' false
// CHECK-NEXT: FunctionDecl {{.*}} prev [[OverloadedFunction1ID1]] {{.*}} OverloadedFunction1 'void (float)'
// CHECK-NEXT:   ParmVarDecl {{.*}} 'float'
// CHECK-NEXT:   CompoundStmt
// CHECK-NEXT:   SYCLAddIRAttributesFunctionAttr {{.*}} Inherited
// CHECK-NEXT:     ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:       value: LValue
// CHECK-NEXT:       StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
// CHECK-NEXT:     ConstantExpr {{.*}} 'int'
// CHECK-NEXT:       value: Int 1
// CHECK-NEXT:       IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT: FunctionDecl {{.*}} OverloadedFunction1 'void (bool)'
// CHECK-NEXT:   ParmVarDecl {{.*}} 'bool'
// CHECK-NEXT:   CompoundStmt
[[__sycl_detail__::add_ir_attributes_function("Attr1", 1)]] void OverloadedFunction1(float);
[[__sycl_detail__::add_ir_attributes_function("Attr3", false)]] void OverloadedFunction1(int) {}
void OverloadedFunction1(float) {}
void OverloadedFunction1(bool) {}

// CHECK:      CXXRecordDecl {{.*}} class ClassWithSpecials1 definition
// CHECK-NEXT:   DefinitionData
// CHECK-NEXT:     DefaultConstructor
// CHECK-NEXT:     CopyConstructor
// CHECK-NEXT:     MoveConstructor
// CHECK-NEXT:     CopyAssignment
// CHECK-NEXT:     MoveAssignment
// CHECK-NEXT:     Destructor
// CHECK-NEXT:   CXXRecordDecl {{.*}} implicit referenced class ClassWithSpecials1
// CHECK-NEXT:   CXXConstructorDecl {{.*}} ClassWithSpecials1 'void ()'
// CHECK-NEXT:     SYCLAddIRAttributesFunctionAttr
// CHECK-NEXT:       ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:         value: LValue
// CHECK-NEXT:         StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
// CHECK-NEXT:       ConstantExpr {{.*}} 'int'
// CHECK-NEXT:         value: Int 1
// CHECK-NEXT:         IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT:   CXXDestructorDecl {{.*}} ~ClassWithSpecials1 'void ()'
// CHECK-NEXT:     SYCLAddIRAttributesFunctionAttr
// CHECK-NEXT:       ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:         value: LValue
// CHECK-NEXT:         StringLiteral {{.*}} 'const char[6]' lvalue "Attr2"
// CHECK-NEXT:       ConstantExpr {{.*}} 'int'
// CHECK-NEXT:         value: Int 2
// CHECK-NEXT:         IntegerLiteral {{.*}} 'int' 2
// CHECK-NEXT:   CXXMethodDecl {{.*}} operator+ 'ClassWithSpecials1 (const ClassWithSpecials1 &) const'
// CHECK-NEXT:     ParmVarDecl {{.*}} 'const ClassWithSpecials1 &'
// CHECK-NEXT:     SYCLAddIRAttributesFunctionAttr
// CHECK-NEXT:       ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:         value: LValue <todo>
// CHECK-NEXT:         StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
// CHECK-NEXT:       ConstantExpr {{.*}} 'int'
// CHECK-NEXT:         value: Int 3
// CHECK-NEXT:         IntegerLiteral {{.*}} 'int' 3
class ClassWithSpecials1 {
  [[__sycl_detail__::add_ir_attributes_function("Attr1", 1)]] ClassWithSpecials1();
  [[__sycl_detail__::add_ir_attributes_function("Attr2", 2)]] ~ClassWithSpecials1();
  [[__sycl_detail__::add_ir_attributes_function("Attr3", 3)]] ClassWithSpecials1 operator+(const ClassWithSpecials1 &) const;
};

// CHECK:      CXXRecordDecl {{.*}} referenced class BaseClass1 definition
// CHECK-NEXT:   DefinitionData
// CHECK-NEXT:     DefaultConstructor
// CHECK-NEXT:     CopyConstructor
// CHECK-NEXT:     MoveConstructor
// CHECK-NEXT:     CopyAssignment
// CHECK-NEXT:     MoveAssignment
// CHECK-NEXT:     Destructor
// CHECK-NEXT:   SYCLAddIRAttributesGlobalVariableAttr
// CHECK-NEXT:     ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:       value: LValue
// CHECK-NEXT:       StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
// CHECK-NEXT:     ConstantExpr {{.*}} 'int'
// CHECK-NEXT:       value: Int 1
// CHECK-NEXT:       IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT:   CXXRecordDecl {{.*}} implicit class BaseClass1
// CHECK-NEXT: CXXRecordDecl {{.*}} class SubClass1 definition
// CHECK-NEXT:   DefinitionData
// CHECK-NEXT:     DefaultConstructor
// CHECK-NEXT:     CopyConstructor
// CHECK-NEXT:     MoveConstructor
// CHECK-NEXT:     CopyAssignment
// CHECK-NEXT:     MoveAssignment
// CHECK-NEXT:     Destructor
// CHECK-NEXT:   private 'BaseClass1'
// CHECK-NEXT:   SYCLAddIRAttributesGlobalVariableAttr
// CHECK-NEXT:     ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:       value: LValue <todo>
// CHECK-NEXT:       StringLiteral {{.*}} 'const char[6]' lvalue "Attr2"
// CHECK-NEXT:     ConstantExpr {{.*}} 'bool'
// CHECK-NEXT:       value: Int 1
// CHECK-NEXT:       CXXBoolLiteralExpr {{.*}} 'bool' true
// CHECK-NEXT:   CXXRecordDecl {{.*}} implicit class SubClass1
// CHECK-NEXT: CXXRecordDecl {{.*}} class SubClass2 definition
// CHECK-NEXT:   DefinitionData
// CHECK-NEXT:     DefaultConstructor
// CHECK-NEXT:     CopyConstructor
// CHECK-NEXT:     MoveConstructor
// CHECK-NEXT:     CopyAssignment
// CHECK-NEXT:     MoveAssignment
// CHECK-NEXT:     Destructor
// CHECK-NEXT:   private 'BaseClass1'
// CHECK-NEXT:   CXXRecordDecl {{.*}} implicit class SubClass2
class [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", 1)]] BaseClass1{};
class [[__sycl_detail__::add_ir_attributes_global_variable("Attr2", true)]] SubClass1 : BaseClass1{};
class SubClass2 : BaseClass1 {};

// CHECK:      ClassTemplateDecl {{.*}} TemplateClass1
// CHECK-NEXT:   TemplateTypeParmDecl {{.*}} typename depth 0 index 0 T
// CHECK-NEXT:   NonTypeTemplateParmDecl {{.*}} 'int' depth 0 index 1 I
// CHECK-NEXT:   CXXRecordDecl {{.*}} class TemplateClass1 definition
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
// CHECK-NEXT:       ConstantExpr {{.*}} 'int'
// CHECK-NEXT:         value: Int 1
// CHECK-NEXT:         IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT:     CXXRecordDecl {{.*}} implicit class TemplateClass1
// CHECK-NEXT:     CXXMethodDecl {{.*}} TemplateClassMethod 'void (int)'
// CHECK-NEXT:       ParmVarDecl {{.*}} 'int'
// CHECK-NEXT:         SYCLAddIRAttributesKernelParameterAttr
// CHECK-NEXT:           ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:             value: LValue
// CHECK-NEXT:             StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
// CHECK-NEXT:           ConstantExpr {{.*}} 'int'
// CHECK-NEXT:             value: Int 3
// CHECK-NEXT:             IntegerLiteral {{.*}} 'int' 3
// CHECK-NEXT:       SYCLAddIRAttributesFunctionAttr
// CHECK-NEXT:         ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:           value: LValue
// CHECK-NEXT:           StringLiteral {{.*}} 'const char[6]' lvalue "Attr2"
// CHECK-NEXT:         ConstantExpr {{.*}} 'int'
// CHECK-NEXT:           value: Int 2
// CHECK-NEXT:           IntegerLiteral {{.*}} 'int' 2
// CHECK-NEXT:   ClassTemplateSpecialization {{.*}} 'TemplateClass1'
// CHECK-NEXT:   ClassTemplateSpecialization {{.*}} 'TemplateClass1'
// CHECK-NEXT: ClassTemplateSpecializationDecl {{.*}} class TemplateClass1 definition
// CHECK-NEXT:     DefinitionData
// CHECK-NEXT:       DefaultConstructor
// CHECK-NEXT:       CopyConstructor
// CHECK-NEXT:       MoveConstructor
// CHECK-NEXT:       CopyAssignment
// CHECK-NEXT:       MoveAssignment
// CHECK-NEXT:       Destructor
// CHECK-NEXT:   TemplateArgument type 'float'
// CHECK-NEXT:     BuiltinType {{.*}} 'float'
// CHECK-NEXT:   TemplateArgument integral '3'
// CHECK-NEXT:   SYCLAddIRAttributesGlobalVariableAttr
// CHECK-NEXT:     ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:       value: LValue
// CHECK-NEXT:       StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
// CHECK-NEXT:     ConstantExpr {{.*}} 'int'
// CHECK-NEXT:       value: Int 1
// CHECK-NEXT:       IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT:   CXXRecordDecl {{.*}} implicit class TemplateClass1
// CHECK-NEXT:   CXXMethodDecl {{.*}} TemplateClassMethod 'void (int)'
// CHECK-NEXT:     ParmVarDecl {{.*}} 'int'
// CHECK-NEXT:       SYCLAddIRAttributesKernelParameterAttr
// CHECK-NEXT:         ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:           value: LValue
// CHECK-NEXT:           StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
// CHECK-NEXT:         ConstantExpr {{.*}} 'int'
// CHECK-NEXT:           value: Int 3
// CHECK-NEXT:           IntegerLiteral {{.*}} 'int' 3
// CHECK-NEXT:     SYCLAddIRAttributesFunctionAttr
// CHECK-NEXT:       ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:         value: LValue
// CHECK-NEXT:         StringLiteral {{.*}} 'const char[6]' lvalue "Attr2"
// CHECK-NEXT:       ConstantExpr {{.*}} 'int'
// CHECK-NEXT:         value: Int 2
// CHECK-NEXT:         IntegerLiteral {{.*}} 'int' 2
// CHECK-NEXT: ClassTemplatePartialSpecializationDecl {{.*}} struct TemplateClass1 definition
// CHECK-NEXT:     DefinitionData
// CHECK-NEXT:       DefaultConstructor
// CHECK-NEXT:       CopyConstructor
// CHECK-NEXT:       MoveConstructor
// CHECK-NEXT:       CopyAssignment
// CHECK-NEXT:       MoveAssignment
// CHECK-NEXT:       Destructor
// CHECK-NEXT:   TemplateArgument type 'int'
// CHECK-NEXT:     BuiltinType {{.*}} 'int'
// CHECK-NEXT:   TemplateArgument expr
// CHECK-NEXT:     DeclRefExpr {{.*}} 'int' NonTypeTemplateParm {{.*}} 'I' 'int'
// CHECK-NEXT:   NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 0 index 0 I
// CHECK-NEXT:   CXXRecordDecl {{.*}} implicit struct TemplateClass1
// CHECK-NEXT:   CXXMethodDecl {{.*}} TemplateClassMethod 'void (int)'
// CHECK-NEXT:     ParmVarDecl {{.*}} 'int'
// CHECK-NEXT: ClassTemplateSpecializationDecl {{.*}} struct TemplateClass1 definition
// CHECK-NEXT:     DefinitionData
// CHECK-NEXT:       DefaultConstructor
// CHECK-NEXT:       CopyConstructor
// CHECK-NEXT:       MoveConstructor
// CHECK-NEXT:       CopyAssignment
// CHECK-NEXT:       MoveAssignment
// CHECK-NEXT:       Destructor
// CHECK-NEXT:   TemplateArgument type 'bool'
// CHECK-NEXT:     BuiltinType {{.*}} 'bool'
// CHECK-NEXT:   TemplateArgument integral '2'
// CHECK-NEXT:   CXXRecordDecl {{.*}} implicit struct TemplateClass1
// CHECK-NEXT:   CXXMethodDecl {{.*}} TemplateClassMethod 'void (int)'
// CHECK-NEXT:     ParmVarDecl {{.*}} 'int'
template <typename T, int I>
class [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", 1)]] TemplateClass1 {
  [[__sycl_detail__::add_ir_attributes_function("Attr2", 2)]] void TemplateClassMethod([[__sycl_detail__::add_ir_attributes_kernel_parameter("Attr3", 3)]] int);
};
template class TemplateClass1<float, 3>;
template <int I> struct TemplateClass1<int, I> {
  void TemplateClassMethod(int);
};
template <> struct TemplateClass1<bool, 2> {
  void TemplateClassMethod(int);
};
