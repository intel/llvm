// RUN: %clang_cc1 -fsycl-is-device -std=gnu++11 -ast-dump %s | FileCheck %s

// Tests the AST produced from add_ir_attributes_* for valid and invalid
// redeclarations.

// CHECK:      FunctionDecl [[FunctionRedecl1ID1:0x[0-9a-f]+]] {{.*}} FunctionRedecl1 'void ()'
// CHECK-NEXT:   SYCLAddIRAttributesFunctionAttr
// CHECK-NEXT:     ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:       value: LValue
// CHECK-NEXT:       StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
// CHECK-NEXT:     ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:       value: LValue
// CHECK-NEXT:       StringLiteral {{.*}} 'const char[6]' lvalue "Attr2"
// CHECK-NEXT:     ConstantExpr {{.*}} 'int'
// CHECK-NEXT:       value: Int 1
// CHECK-NEXT:       IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT:     ConstantExpr {{.*}} 'bool'
// CHECK-NEXT:       value: Int 1
// CHECK-NEXT:       CXXBoolLiteralExpr {{.*}} 'bool' true
// CHECK-NEXT: FunctionDecl {{.*}} prev [[FunctionRedecl1ID1]] {{.*}} FunctionRedecl1 'void ()'
// CHECK-NEXT:   SYCLAddIRAttributesFunctionAttr
// CHECK-NEXT:     ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:       value: LValue
// CHECK-NEXT:       StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
// CHECK-NEXT:     ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:       value: LValue
// CHECK-NEXT:       StringLiteral {{.*}} 'const char[6]' lvalue "Attr2"
// CHECK-NEXT:     ConstantExpr {{.*}} 'int'
// CHECK-NEXT:       value: Int 1
// CHECK-NEXT:       IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT:     ConstantExpr {{.*}} 'bool'
// CHECK-NEXT:       value: Int 1
// CHECK-NEXT:       CXXBoolLiteralExpr {{.*}} 'bool' true
[[__sycl_detail__::add_ir_attributes_function("Attr1", "Attr2", 1, true)]] void FunctionRedecl1();
void FunctionRedecl1();

// CHECK:      CXXRecordDecl [[GlobalVarStructRedecl1ID1:0x[0-9a-f]+]] {{.*}} struct GlobalVarStructRedecl1
// CHECK-NEXT:   SYCLAddIRAttributesGlobalVariableAttr
// CHECK-NEXT:     ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:       value: LValue
// CHECK-NEXT:       StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
// CHECK-NEXT:     ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:       value: LValue
// CHECK-NEXT:       StringLiteral {{.*}} 'const char[6]' lvalue "Attr2"
// CHECK-NEXT:     ConstantExpr {{.*}} 'int'
// CHECK-NEXT:       value: Int 1
// CHECK-NEXT:       IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT:     ConstantExpr {{.*}} 'bool'
// CHECK-NEXT:       value: Int 1
// CHECK-NEXT:       CXXBoolLiteralExpr {{.*}} 'bool' true
// CHECK-NEXT: CXXRecordDecl {{.*}} prev [[GlobalVarStructRedecl1ID1]] {{.*}} struct GlobalVarStructRedecl1
// CHECK-NEXT:   SYCLAddIRAttributesGlobalVariableAttr
// CHECK-NEXT:     ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:       value: LValue
// CHECK-NEXT:       StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
// CHECK-NEXT:     ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:       value: LValue
// CHECK-NEXT:       StringLiteral {{.*}} 'const char[6]' lvalue "Attr2"
// CHECK-NEXT:     ConstantExpr {{.*}} 'int'
// CHECK-NEXT:       value: Int 1
// CHECK-NEXT:       IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT:     ConstantExpr {{.*}} 'bool'
// CHECK-NEXT:       value: Int 1
// CHECK-NEXT:       CXXBoolLiteralExpr {{.*}} 'bool' true
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", "Attr2", 1, true)]] GlobalVarStructRedecl1;
struct GlobalVarStructRedecl1;

// CHECK:      CXXRecordDecl {{.*}} referenced struct GlobalVarStructBase definition
// CHECK-NEXT:   DefinitionData
// CHECK-NEXT:     DefaultConstructor
// CHECK-NEXT:     CopyConstructor
// CHECK-NEXT:     MoveConstructor
// CHECK-NEXT:     CopyAssignment
// CHECK-NEXT:     MoveAssignment
// CHECK-NEXT:     Destructor
// CHECK-NEXT:   CXXRecordDecl {{.*}} implicit struct GlobalVarStructBase
// CHECK-NEXT: CXXRecordDecl {{.*}} referenced struct GlobalVarStructInherit1 definition
// CHECK-NEXT:   DefinitionData
// CHECK-NEXT:     DefaultConstructor
// CHECK-NEXT:     CopyConstructor
// CHECK-NEXT:     MoveConstructor
// CHECK-NEXT:     CopyAssignment
// CHECK-NEXT:     MoveAssignment
// CHECK-NEXT:     Destructor
// CHECK-NEXT:   public 'GlobalVarStructBase'
// CHECK-NEXT:   SYCLAddIRAttributesGlobalVariableAttr
// CHECK-NEXT:     ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:       value: LValue
// CHECK-NEXT:       StringLiteral {{.*}} lvalue "Attr1"
// CHECK-NEXT:     ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:       value: LValue
// CHECK-NEXT:       StringLiteral {{.*}} 'const char[6]' lvalue "Attr2"
// CHECK-NEXT:     ConstantExpr {{.*}} <col:79> 'int'
// CHECK-NEXT:       value: Int 1
// CHECK-NEXT:       IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT:     ConstantExpr {{.*}} 'bool'
// CHECK-NEXT:       value: Int 1
// CHECK-NEXT:       CXXBoolLiteralExpr {{.*}} 'bool' true
// CHECK-NEXT:   CXXRecordDecl {{.*}} implicit struct GlobalVarStructInherit1
// CHECK-NEXT: CXXRecordDecl {{.*}} struct GlobalVarStructInherit2 definition
// CHECK-NEXT:   DefinitionData
// CHECK-NEXT:     DefaultConstructor
// CHECK-NEXT:     CopyConstructor
// CHECK-NEXT:     MoveConstructor
// CHECK-NEXT:     CopyAssignment
// CHECK-NEXT:     MoveAssignment
// CHECK-NEXT:     Destructor
// CHECK-NEXT:   public 'GlobalVarStructInherit1'
// CHECK-NEXT:   CXXRecordDecl {{.*}} implicit struct GlobalVarStructInherit2
// CHECK-NEXT: CXXRecordDecl {{.*}} struct GlobalVarStructInherit3 definition
// CHECK-NEXT:   DefinitionData
// CHECK-NEXT:     DefaultConstructor
// CHECK-NEXT:     CopyConstructor
// CHECK-NEXT:     MoveConstructor
// CHECK-NEXT:     CopyAssignment
// CHECK-NEXT:     MoveAssignment
// CHECK-NEXT:     Destructor
// CHECK-NEXT:   public 'GlobalVarStructInherit1'
// CHECK-NEXT:   SYCLAddIRAttributesGlobalVariableAttr
// CHECK-NEXT:     ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:       value: LValue
// CHECK-NEXT:       StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
// CHECK-NEXT:     ConstantExpr {{.*}} 'bool'
// CHECK-NEXT:       value: Int 0
// CHECK-NEXT:       CXXBoolLiteralExpr {{.*}} 'bool' false
// CHECK-NEXT:   CXXRecordDecl {{.*}} implicit struct GlobalVarStructInherit3
struct GlobalVarStructBase {};
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", "Attr2", 1, true)]] GlobalVarStructInherit1 : GlobalVarStructBase{};
struct GlobalVarStructInherit2 : GlobalVarStructInherit1 {};
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr3", false)]] GlobalVarStructInherit3 : GlobalVarStructInherit1{};

// CHECK:      CXXRecordDecl {{.*}} referenced struct SpecialClassStructBase definition
// CHECK-NEXT:   DefinitionData
// CHECK-NEXT:     DefaultConstructor
// CHECK-NEXT:     CopyConstructor
// CHECK-NEXT:     MoveConstructor
// CHECK-NEXT:     CopyAssignment
// CHECK-NEXT:     MoveAssignment
// CHECK-NEXT:     Destructor
// CHECK-NEXT:   SYCLSpecialClassAttr
// CHECK-NEXT:   CXXRecordDecl {{.*}} implicit struct SpecialClassStructBase
// CHECK-NEXT:   CXXMethodDecl [[InitBase:0x[0-9a-f]+]] {{.*}} __init 'void (int)' virtual
// CHECK-NEXT:     ParmVarDecl {{.*}} x 'int'
// CHECK-NEXT:     CompoundStmt
// CHECK-NEXT:   CXXMethodDecl {{.*}} implicit operator= 'SpecialClassStructBase &(const SpecialClassStructBase &)'
// CHECK-NEXT:     ParmVarDecl {{.*}} 'const SpecialClassStructBase &'
// CHECK-NEXT:   CXXMethodDecl {{.*}} implicit operator= 'SpecialClassStructBase &(SpecialClassStructBase &&)'
// CHECK-NEXT:     ParmVarDecl {{.*}} 'SpecialClassStructBase &&'
// CHECK-NEXT:   CXXDestructorDecl {{.*}} implicit ~SpecialClassStructBase 'void ()'
// CHECK-NEXT: CXXRecordDecl {{.*}} referenced struct SpecialClassStructInherit1 definition
// CHECK-NEXT:   DefinitionData
// CHECK-NEXT:     DefaultConstructor
// CHECK-NEXT:     CopyConstructor
// CHECK-NEXT:     MoveConstructor
// CHECK-NEXT:     CopyAssignment
// CHECK-NEXT:     MoveAssignment
// CHECK-NEXT:     Destructor
// CHECK-NEXT:   public 'SpecialClassStructBase'
// CHECK-NEXT:   SYCLSpecialClassAttr
// CHECK-NEXT:   CXXRecordDecl {{.*}} implicit struct SpecialClassStructInherit1
// CHECK-NEXT:   CXXMethodDecl [[InitInherit1:0x[0-9a-f]+]] {{.*}} __init 'void (int)'
// CHECK-NEXT:     Overrides: [ [[InitBase]] SpecialClassStructBase::__init 'void (int)' ]
// CHECK-NEXT:     ParmVarDecl {{.*}} x 'int'
// CHECK-NEXT:       SYCLAddIRAttributesKernelParameterAttr
// CHECK-NEXT:         ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:           value: LValue
// CHECK-NEXT:           StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
// CHECK-NEXT:         ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:           value: LValue
// CHECK-NEXT:           StringLiteral {{.*}} 'const char[6]' lvalue "Attr2"
// CHECK-NEXT:         ConstantExpr {{.*}} 'int'
// CHECK-NEXT:           value: Int 1
// CHECK-NEXT:           IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT:         ConstantExpr {{.*}} 'bool'
// CHECK-NEXT:           value: Int 1
// CHECK-NEXT:           CXXBoolLiteralExpr {{.*}} 'bool' true
// CHECK-NEXT:     CompoundStmt
// CHECK-NEXT:     OverrideAttr
// CHECK-NEXT:   CXXMethodDecl {{.*}} implicit operator= 'SpecialClassStructInherit1 &(const SpecialClassStructInherit1 &)'
// CHECK-NEXT:     ParmVarDecl {{.*}} 'const SpecialClassStructInherit1 &'
// CHECK-NEXT:   CXXMethodDecl {{.*}} implicit operator= 'SpecialClassStructInherit1 &(SpecialClassStructInherit1 &&)'
// CHECK-NEXT:     ParmVarDecl {{.*}} 'SpecialClassStructInherit1 &&'
// CHECK-NEXT:   CXXDestructorDecl {{.*}} implicit ~SpecialClassStructInherit1 'void ()'
// CHECK-NEXT: CXXRecordDecl {{.*}} struct SpecialClassStructInherit2 definition
// CHECK-NEXT:   DefinitionData
// CHECK-NEXT:     DefaultConstructor
// CHECK-NEXT:     CopyConstructor
// CHECK-NEXT:     MoveConstructor
// CHECK-NEXT:     CopyAssignment
// CHECK-NEXT:     MoveAssignment
// CHECK-NEXT:     Destructor
// CHECK-NEXT:   public 'SpecialClassStructInherit1'
// CHECK-NEXT:   SYCLSpecialClassAttr
// CHECK-NEXT:   CXXRecordDecl {{.*}} implicit struct SpecialClassStructInherit2
// CHECK-NEXT:   CXXMethodDecl {{.*}} __init 'void (int)'
// CHECK-NEXT:     Overrides: [ [[InitInherit1]] SpecialClassStructInherit1::__init 'void (int)' ]
// CHECK-NEXT:     ParmVarDecl {{.*}} x 'int'
// CHECK-NEXT:       SYCLAddIRAttributesKernelParameterAttr
// CHECK-NEXT:         ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:           value: LValue
// CHECK-NEXT:           StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
// CHECK-NEXT:         ConstantExpr {{.*}} 'bool'
// CHECK-NEXT:           value: Int 0
// CHECK-NEXT:           CXXBoolLiteralExpr {{.*}} 'bool' false
// CHECK-NEXT:     CompoundStmt
// CHECK-NEXT:     OverrideAttr
// CHECK-NEXT:   CXXMethodDecl {{.*}} implicit operator= 'SpecialClassStructInherit2 &(const SpecialClassStructInherit2 &)'
// CHECK-NEXT:     ParmVarDecl {{.*}} 'const SpecialClassStructInherit2 &'
// CHECK-NEXT:   CXXMethodDecl {{.*}} implicit operator= 'SpecialClassStructInherit2 &(SpecialClassStructInherit2 &&)'
// CHECK-NEXT:     ParmVarDecl {{.*}} 'SpecialClassStructInherit2 &&'
// CHECK-NEXT:   CXXDestructorDecl {{.*}} implicit ~SpecialClassStructInherit2 'void ()'
struct __attribute__((sycl_special_class)) SpecialClassStructBase {
  virtual void __init(int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructInherit1 : SpecialClassStructBase {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter("Attr1", "Attr2", 1, true)]] int x) override {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructInherit2 : SpecialClassStructInherit1 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter("Attr3", false)]] int x) override {}
};
