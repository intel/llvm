// RUN: %clang_cc1 -fsycl-is-device -std=gnu++11 -ast-dump %s | FileCheck %s

// Tests the AST produced from add_ir_attributes_* for valid redeclarations.

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

// CHECK:      FunctionDecl [[FunctionRedecl2ID1:0x[0-9a-f]+]] {{.*}} FunctionRedecl2 'void ()'
// CHECK-NEXT: FunctionDecl [[FunctionRedecl2ID2:0x[0-9a-f]+]] prev [[FunctionRedecl2ID1]] {{.*}} FunctionRedecl2 'void ()'
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
// CHECK-NEXT: FunctionDecl [[FunctionRedecl2ID3:0x[0-9a-f]+]] prev [[FunctionRedecl2ID2]] {{.*}} FunctionRedecl2 'void ()'
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
// CHECK-NEXT: FunctionDecl [[FunctionRedecl2ID4:0x[0-9a-f]+]] prev [[FunctionRedecl2ID3]] {{.*}} FunctionRedecl2 'void ()'
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
// CHECK-NEXT: FunctionDecl {{.*}} prev [[FunctionRedecl2ID4]] {{.*}} FunctionRedecl2 'void ()'
// CHECK-NEXT:   CompoundStmt
// CHECK-NEXT:   SYCLAddIRAttributesFunctionAttr
// CHECK-NEXT:     ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:       value: LValue
// CHECK-NEXT:       StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
// CHECK-NEXT:     ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:       value: LValue
// CHECK-NEXT:       StringLiteral {{.*}} 'const char[6]' lvalue "Attr2"
// CHECK-NEXT:     ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:       value: LValue
// CHECK-NEXT:       StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
// CHECK-NEXT:     ConstantExpr {{.*}} 'int'
// CHECK-NEXT:       value: Int 1
// CHECK-NEXT:       IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT:     ConstantExpr {{.*}} 'bool'
// CHECK-NEXT:       value: Int 1
// CHECK-NEXT:       CXXBoolLiteralExpr {{.*}} 'bool' true
// CHECK-NEXT:     ConstantExpr {{.*}} 'bool'
// CHECK-NEXT:       value: Int 0
// CHECK-NEXT:       CXXBoolLiteralExpr {{.*}} 'bool' false
void FunctionRedecl2();
[[__sycl_detail__::add_ir_attributes_function("Attr1", "Attr2", 1, true)]] void FunctionRedecl2();
[[__sycl_detail__::add_ir_attributes_function("Attr1", "Attr2", 1, true)]] void FunctionRedecl2();
[[__sycl_detail__::add_ir_attributes_function("Attr2", "Attr1", true, 1)]] void FunctionRedecl2();
[[__sycl_detail__::add_ir_attributes_function("Attr3", false)]] void FunctionRedecl2(){};

// CHECK:      FunctionDecl [[FunctionRedecl3ID1:0x[0-9a-f]+]] {{.*}} FunctionRedecl3 'void ()'
// CHECK-NEXT:   SYCLAddIRAttributesFunctionAttr
// CHECK-NEXT:     InitListExpr {{.*}} 'void'
// CHECK-NEXT:       StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
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
// CHECK-NEXT: FunctionDecl {{.*}} prev [[FunctionRedecl3ID1]] {{.*}} FunctionRedecl3 'void ()'
// CHECK-NEXT:   SYCLAddIRAttributesFunctionAttr {{.*}} Inherited
// CHECK-NEXT:     InitListExpr {{.*}} 'void'
// CHECK-NEXT:       StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
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
[[__sycl_detail__::add_ir_attributes_function({"Attr3"}, "Attr1", "Attr2", 1, true)]] void FunctionRedecl3();
void FunctionRedecl3();

// CHECK:      FunctionDecl [[FunctionRedecl4ID1:0x[0-9a-f]+]] {{.*}} FunctionRedecl4 'void ()'
// CHECK-NEXT: FunctionDecl [[FunctionRedecl4ID2:0x[0-9a-f]+]] prev [[FunctionRedecl4ID1]] {{.*}} FunctionRedecl4 'void ()'
// CHECK-NEXT:   SYCLAddIRAttributesFunctionAttr
// CHECK-NEXT:     InitListExpr {{.*}} 'void'
// CHECK-NEXT:       StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
// CHECK-NEXT:       StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
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
// CHECK-NEXT: FunctionDecl [[FunctionRedecl4ID3:0x[0-9a-f]+]] prev [[FunctionRedecl4ID2]] {{.*}} FunctionRedecl4 'void ()'
// CHECK-NEXT:   SYCLAddIRAttributesFunctionAttr {{.*}} Inherited
// CHECK-NEXT:     InitListExpr {{.*}} 'void'
// CHECK-NEXT:       StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
// CHECK-NEXT:       StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
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
// CHECK-NEXT: FunctionDecl [[FunctionRedecl4ID4:0x[0-9a-f]+]] prev [[FunctionRedecl4ID3]] {{.*}} FunctionRedecl4 'void ()'
// CHECK-NEXT:   SYCLAddIRAttributesFunctionAttr {{.*}} Inherited
// CHECK-NEXT:     InitListExpr {{.*}} 'void'
// CHECK-NEXT:       StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
// CHECK-NEXT:       StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
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
// CHECK-NEXT: FunctionDecl [[FunctionRedecl4ID5:0x[0-9a-f]+]] prev [[FunctionRedecl4ID4]] {{.*}} FunctionRedecl4 'void ()'
// CHECK-NEXT:   SYCLAddIRAttributesFunctionAttr {{.*}} Inherited
// CHECK-NEXT:     InitListExpr {{.*}} 'void'
// CHECK-NEXT:       StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
// CHECK-NEXT:       StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
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
// CHECK-NEXT: FunctionDecl {{.*}} prev [[FunctionRedecl4ID5]] {{.*}} FunctionRedecl4 'void ()'
// CHECK-NEXT:   CompoundStmt
// CHECK-NEXT:   SYCLAddIRAttributesFunctionAttr {{.*}} Inherited
// CHECK-NEXT:     InitListExpr {{.*}} 'void'
// CHECK-NEXT:       StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
// CHECK-NEXT:       StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
// CHECK-NEXT:     ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:       value: LValue
// CHECK-NEXT:       StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
// CHECK-NEXT:     ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:       value: LValue
// CHECK-NEXT:       StringLiteral {{.*}} 'const char[6]' lvalue "Attr2"
// CHECK-NEXT:     ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:       value: LValue
// CHECK-NEXT:       StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
// CHECK-NEXT:     ConstantExpr {{.*}} 'int'
// CHECK-NEXT:       value: Int 1
// CHECK-NEXT:       IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT:     ConstantExpr {{.*}} 'bool'
// CHECK-NEXT:       value: Int 1
// CHECK-NEXT:       CXXBoolLiteralExpr {{.*}} 'bool' true
// CHECK-NEXT:     ConstantExpr {{.*}} 'bool'
// CHECK-NEXT:       value: Int 0
// CHECK-NEXT:       CXXBoolLiteralExpr {{.*}} 'bool' false
void FunctionRedecl4();
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, "Attr1", "Attr2", 1, true)]] void FunctionRedecl4();
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, "Attr1", "Attr2", 1, true)]] void FunctionRedecl4();
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, "Attr2", "Attr1", true, 1)]] void FunctionRedecl4();
[[__sycl_detail__::add_ir_attributes_function({"Attr3", "Attr1"}, "Attr1", "Attr2", 1, true)]] void FunctionRedecl4();
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, "Attr3", false)]] void FunctionRedecl4(){};

// CHECK:      FunctionDecl {{.*}} FunctionDecl2 'void ()'
// CHECK-NEXT:   CompoundStmt
// CHECK-NEXT:   SYCLAddIRAttributesFunctionAttr
// CHECK-NEXT:     ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:       value: LValue
// CHECK-NEXT:       StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
// CHECK-NEXT:     ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:       value: LValue
// CHECK-NEXT:       StringLiteral {{.*}} 'const char[6]' lvalue "Attr2"
// CHECK-NEXT:     ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:       value: LValue
// CHECK-NEXT:       StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
// CHECK-NEXT:     ConstantExpr {{.*}} 'bool'
// CHECK-NEXT:       value: Int 0
// CHECK-NEXT:       CXXBoolLiteralExpr {{.*}} 'bool' false
// CHECK-NEXT:     ConstantExpr {{.*}} 'bool'
// CHECK-NEXT:       value: Int 1
// CHECK-NEXT:       CXXBoolLiteralExpr {{.*}} 'bool' true
// CHECK-NEXT:     ConstantExpr {{.*}} 'int'
// CHECK-NEXT:       value: Int 1
// CHECK-NEXT:       IntegerLiteral {{.*}} 'int' 1
[[__sycl_detail__::add_ir_attributes_function("Attr1", "Attr2", 1, true)]]
[[__sycl_detail__::add_ir_attributes_function("Attr1", "Attr2", 1, true)]]
[[__sycl_detail__::add_ir_attributes_function("Attr2", "Attr1", true, 1)]]
[[__sycl_detail__::add_ir_attributes_function("Attr3", false)]]
void FunctionDecl2(){};

// CHECK:      FunctionDecl {{.*}} FunctionDecl3 'void ()'
// CHECK-NEXT:   CompoundStmt {{.*}}
// CHECK-NEXT:   SYCLAddIRAttributesFunctionAttr {{.*}}
// CHECK-NEXT:     InitListExpr {{.*}} 'void'
// CHECK-NEXT:       StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
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
[[__sycl_detail__::add_ir_attributes_function({"Attr3"}, "Attr1", "Attr2", 1, true)]]
void FunctionDecl3(){};

// CHECK:      FunctionDecl {{.*}} FunctionDecl4 'void ()'
// CHECK-NEXT:   CompoundStmt {{.*}}
// CHECK-NEXT:   SYCLAddIRAttributesFunctionAttr
// CHECK-NEXT:     InitListExpr {{.*}} 'void'
// CHECK-NEXT:       StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
// CHECK-NEXT:       StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
// CHECK-NEXT:     ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:       value: LValue
// CHECK-NEXT:       StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
// CHECK-NEXT:     ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:       value: LValue
// CHECK-NEXT:       StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
// CHECK-NEXT:     ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:       value: LValue
// CHECK-NEXT:       StringLiteral {{.*}} 'const char[6]' lvalue "Attr2"
// CHECK-NEXT:     ConstantExpr {{.*}} 'bool'
// CHECK-NEXT:       value: Int 0
// CHECK-NEXT:       CXXBoolLiteralExpr {{.*}} 'bool' false
// CHECK-NEXT:     ConstantExpr {{.*}} 'int'
// CHECK-NEXT:       value: Int 1
// CHECK-NEXT:       IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT:     ConstantExpr {{.*}} 'bool'
// CHECK-NEXT:       value: Int 1
// CHECK-NEXT:       CXXBoolLiteralExpr {{.*}} 'bool' true
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, "Attr1", "Attr2", 1, true)]]
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, "Attr1", "Attr2", 1, true)]]
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, "Attr2", "Attr1", true, 1)]]
[[__sycl_detail__::add_ir_attributes_function({"Attr3", "Attr1"}, "Attr1", "Attr2", 1, true)]]
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, "Attr3", false)]]
void FunctionDecl4(){};

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

// CHECK:      CXXRecordDecl [[GlobalVarStructRedecl2ID1:0x[0-9a-f]+]] {{.*}} struct GlobalVarStructRedecl2
// CHECK-NEXT: CXXRecordDecl [[GlobalVarStructRedecl2ID2:0x[0-9a-f]+]] prev [[GlobalVarStructRedecl2ID1]] {{.*}} struct GlobalVarStructRedecl2
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
// CHECK-NEXT: CXXRecordDecl [[GlobalVarStructRedecl2ID3:0x[0-9a-f]+]] prev [[GlobalVarStructRedecl2ID2]] {{.*}} struct GlobalVarStructRedecl2
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
// CHECK-NEXT: CXXRecordDecl [[GlobalVarStructRedecl2ID4:0x[0-9a-f]+]] prev [[GlobalVarStructRedecl2ID3]] {{.*}} struct GlobalVarStructRedecl2
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
// CHECK-NEXT: CXXRecordDecl {{.*}} prev [[GlobalVarStructRedecl2ID4]] {{.*}} struct GlobalVarStructRedecl2 definition
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
// CHECK-NEXT:     ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:       value: LValue
// CHECK-NEXT:       StringLiteral {{.*}} 'const char[6]' lvalue "Attr2"
// CHECK-NEXT:     ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:       value: LValue
// CHECK-NEXT:       StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
// CHECK-NEXT:     ConstantExpr {{.*}} 'int'
// CHECK-NEXT:       value: Int 1
// CHECK-NEXT:       IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT:     ConstantExpr {{.*}} 'bool'
// CHECK-NEXT:       value: Int 1
// CHECK-NEXT:       CXXBoolLiteralExpr {{.*}} 'bool' true
// CHECK-NEXT:     ConstantExpr {{.*}} 'bool'
// CHECK-NEXT:       value: Int 0
// CHECK-NEXT:       CXXBoolLiteralExpr {{.*}} 'bool' false
// CHECK-NEXT:   CXXRecordDecl {{.*}} implicit struct GlobalVarStructRedecl2
struct GlobalVarStructRedecl2;
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", "Attr2", 1, true)]] GlobalVarStructRedecl2;
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", "Attr2", 1, true)]] GlobalVarStructRedecl2;
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr2", "Attr1", true, 1)]] GlobalVarStructRedecl2;
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr3", false)]] GlobalVarStructRedecl2{};

// CHECK:      CXXRecordDecl [[GlobalVarStructRedecl3ID1:0x[0-9a-f]+]] {{.*}} struct GlobalVarStructRedecl3
// CHECK-NEXT: CXXRecordDecl [[GlobalVarStructRedecl3ID2:0x[0-9a-f]+]] prev [[GlobalVarStructRedecl3ID1]] {{.*}} struct GlobalVarStructRedecl3
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
// CHECK-NEXT: CXXRecordDecl [[GlobalVarStructRedecl3ID3:0x[0-9a-f]+]] prev [[GlobalVarStructRedecl3ID2]] {{.*}} struct GlobalVarStructRedecl3
// CHECK-NEXT:   SYCLAddIRAttributesGlobalVariableAttr {{.*}} Inherited
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
// CHECK-NEXT: CXXRecordDecl [[GlobalVarStructRedecl3ID4:0x[0-9a-f]+]] prev [[GlobalVarStructRedecl3ID3]] {{.*}} struct GlobalVarStructRedecl3
// CHECK-NEXT:   SYCLAddIRAttributesGlobalVariableAttr {{.*}} Inherited
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
// CHECK-NEXT: CXXRecordDecl {{.*}} prev [[GlobalVarStructRedecl3ID4]] {{.*}} struct GlobalVarStructRedecl3 definition
// CHECK-NEXT:   DefinitionData
// CHECK-NEXT:     DefaultConstructor
// CHECK-NEXT:     CopyConstructor
// CHECK-NEXT:     MoveConstructor
// CHECK-NEXT:     CopyAssignment
// CHECK-NEXT:     MoveAssignment
// CHECK-NEXT:     Destructor
// CHECK-NEXT:   SYCLAddIRAttributesGlobalVariableAttr {{.*}} Inherited
// CHECK-NEXT:     ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:       value: LValue
// CHECK-NEXT:       StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
// CHECK-NEXT:     ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:       value: LValue
// CHECK-NEXT:       StringLiteral {{.*}} 'const char[6]' lvalue "Attr2"
// CHECK-NEXT:     ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:       value: LValue
// CHECK-NEXT:       StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
// CHECK-NEXT:     ConstantExpr {{.*}} 'int'
// CHECK-NEXT:       value: Int 1
// CHECK-NEXT:       IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT:     ConstantExpr {{.*}} 'bool'
// CHECK-NEXT:       value: Int 1
// CHECK-NEXT:       CXXBoolLiteralExpr {{.*}} 'bool' true
// CHECK-NEXT:     ConstantExpr {{.*}} 'bool'
// CHECK-NEXT:       value: Int 0
// CHECK-NEXT:       CXXBoolLiteralExpr {{.*}} 'bool' false
// CHECK-NEXT:   CXXRecordDecl {{.*}} implicit struct GlobalVarStructRedecl3
struct GlobalVarStructRedecl3;
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", "Attr2", 1, true)]] GlobalVarStructRedecl3;
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", "Attr2", 1, true)]] GlobalVarStructRedecl3;
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr2", "Attr1", true, 1)]] GlobalVarStructRedecl3;
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr3", false)]] GlobalVarStructRedecl3{};

// CHECK:      CXXRecordDecl [[GlobalVarStructRedecl4ID1:0x[0-9a-f]+]] {{.*}} struct GlobalVarStructRedecl4
// CHECK-NEXT:   SYCLAddIRAttributesGlobalVariableAttr
// CHECK-NEXT:     InitListExpr {{.*}} 'void'
// CHECK-NEXT:       StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
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
// CHECK-NEXT: CXXRecordDecl {{.*}} prev [[GlobalVarStructRedecl4ID1]] {{.*}} struct GlobalVarStructRedecl4 definition
// CHECK-NEXT:   DefinitionData
// CHECK-NEXT:     DefaultConstructor
// CHECK-NEXT:     CopyConstructor
// CHECK-NEXT:     MoveConstructor
// CHECK-NEXT:     CopyAssignment
// CHECK-NEXT:     MoveAssignment
// CHECK-NEXT:     Destructor
// CHECK-NEXT:   SYCLAddIRAttributesGlobalVariableAttr {{.*}} Inherited
// CHECK-NEXT:     InitListExpr {{.*}} 'void'
// CHECK-NEXT:       StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
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
// CHECK-NEXT:   CXXRecordDecl {{.*}} implicit struct GlobalVarStructRedecl4
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr3"}, "Attr1", "Attr2", 1, true)]] GlobalVarStructRedecl4;
struct GlobalVarStructRedecl4{};

// CHECK:      CXXRecordDecl {{.*}} struct GlobalVarStructDecl1 definition
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
// CHECK-NEXT:       StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
// CHECK-NEXT:     ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:       value: LValue
// CHECK-NEXT:       StringLiteral {{.*}} 'const char[6]' lvalue "Attr2"
// CHECK-NEXT:     ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:       value: LValue
// CHECK-NEXT:       StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
// CHECK-NEXT:     ConstantExpr {{.*}} 'bool'
// CHECK-NEXT:       value: Int 0
// CHECK-NEXT:       CXXBoolLiteralExpr {{.*}} 'bool' false
// CHECK-NEXT:     ConstantExpr {{.*}} 'bool'
// CHECK-NEXT:       value: Int 1
// CHECK-NEXT:       CXXBoolLiteralExpr {{.*}} 'bool' true
// CHECK-NEXT:     ConstantExpr {{.*}} 'int'
// CHECK-NEXT:       value: Int 1
// CHECK-NEXT:       IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT:   CXXRecordDecl {{.*}} implicit struct GlobalVarStructDecl1
struct
[[__sycl_detail__::add_ir_attributes_global_variable("Attr1", "Attr2", 1, true)]]
[[__sycl_detail__::add_ir_attributes_global_variable("Attr1", "Attr2", 1, true)]]
[[__sycl_detail__::add_ir_attributes_global_variable("Attr2", "Attr1", true, 1)]]
[[__sycl_detail__::add_ir_attributes_global_variable("Attr3", false)]]
GlobalVarStructDecl1{};

// CHECK:      CXXRecordDecl {{.*}} struct GlobalVarStructDecl2 definition
// CHECK-NEXT:   DefinitionData
// CHECK-NEXT:     DefaultConstructor
// CHECK-NEXT:     CopyConstructor
// CHECK-NEXT:     MoveConstructor
// CHECK-NEXT:     CopyAssignment
// CHECK-NEXT:     MoveAssignment
// CHECK-NEXT:     Destructor
// CHECK-NEXT:   SYCLAddIRAttributesGlobalVariableAttr
// CHECK-NEXT:     InitListExpr {{.*}} 'void'
// CHECK-NEXT:       StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
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
// CHECK-NEXT:   CXXRecordDecl {{.*}} implicit struct GlobalVarStructDecl2
struct
[[__sycl_detail__::add_ir_attributes_global_variable({"Attr3"}, "Attr1", "Attr2", 1, true)]]
GlobalVarStructDecl2{};

// CHECK:      CXXRecordDecl {{.*}} struct GlobalVarStructDecl3 definition
// CHECK-NEXT:   DefinitionData
// CHECK-NEXT:     DefaultConstructor
// CHECK-NEXT:     CopyConstructor
// CHECK-NEXT:     MoveConstructor
// CHECK-NEXT:     CopyAssignment
// CHECK-NEXT:     MoveAssignment
// CHECK-NEXT:     Destructor
// CHECK-NEXT:   SYCLAddIRAttributesGlobalVariableAttr
// CHECK-NEXT:     InitListExpr {{.*}} 'void'
// CHECK-NEXT:       StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
// CHECK-NEXT:       StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
// CHECK-NEXT:     ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:       value: LValue
// CHECK-NEXT:       StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
// CHECK-NEXT:     ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:       value: LValue
// CHECK-NEXT:       StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
// CHECK-NEXT:     ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:       value: LValue
// CHECK-NEXT:       StringLiteral {{.*}} 'const char[6]' lvalue "Attr2"
// CHECK-NEXT:     ConstantExpr {{.*}} 'bool'
// CHECK-NEXT:       value: Int 0
// CHECK-NEXT:       CXXBoolLiteralExpr {{.*}} 'bool' false
// CHECK-NEXT:     ConstantExpr {{.*}} 'int'
// CHECK-NEXT:       value: Int 1
// CHECK-NEXT:       IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT:     ConstantExpr {{.*}} 'bool'
// CHECK-NEXT:       value: Int 1
// CHECK-NEXT:       CXXBoolLiteralExpr {{.*}} 'bool' true
// CHECK-NEXT:   CXXRecordDecl {{.*}} implicit struct GlobalVarStructDecl3
struct
[[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, "Attr1", "Attr2", 1, true)]]
[[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, "Attr1", "Attr2", 1, true)]]
[[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, "Attr2", "Attr1", true, 1)]]
[[__sycl_detail__::add_ir_attributes_global_variable({"Attr3", "Attr1"}, "Attr1", "Attr2", 1, true)]]
[[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, "Attr3", false)]]
GlobalVarStructDecl3{};

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

// CHECK:      CXXRecordDecl {{.*}} struct SpecialClassStruct1 definition
// CHECK-NEXT:   DefinitionData
// CHECK-NEXT:     DefaultConstructor
// CHECK-NEXT:     CopyConstructor
// CHECK-NEXT:     MoveConstructor
// CHECK-NEXT:     CopyAssignment
// CHECK-NEXT:     MoveAssignment
// CHECK-NEXT:     Destructor
// CHECK-NEXT:   SYCLSpecialClassAttr
// CHECK-NEXT:   CXXRecordDecl {{.*}} implicit struct SpecialClassStruct1
// CHECK-NEXT:   CXXMethodDecl {{.*}} __init 'void (int)'
// CHECK-NEXT:     ParmVarDecl {{.*}} x 'int'
// CHECK-NEXT:       SYCLAddIRAttributesKernelParameterAttr
// CHECK-NEXT:         ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:           value: LValue 
// CHECK-NEXT:           StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
// CHECK-NEXT:         ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:           value: LValue 
// CHECK-NEXT:           StringLiteral {{.*}} 'const char[6]' lvalue "Attr2"
// CHECK-NEXT:         ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:           value: LValue 
// CHECK-NEXT:           StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
// CHECK-NEXT:         ConstantExpr {{.*}} 'bool'
// CHECK-NEXT:           value: Int 0
// CHECK-NEXT:           CXXBoolLiteralExpr {{.*}} 'bool' false
// CHECK-NEXT:         ConstantExpr {{.*}} 'bool'
// CHECK-NEXT:           value: Int 1
// CHECK-NEXT:           CXXBoolLiteralExpr {{.*}} 'bool' true
// CHECK-NEXT:         ConstantExpr {{.*}} 'int'
// CHECK-NEXT:           value: Int 1
// CHECK-NEXT:           IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT:     CompoundStmt
// CHECK-NEXT:   CXXMethodDecl {{.*}} implicit operator= 'SpecialClassStruct1 &(const SpecialClassStruct1 &)'
// CHECK-NEXT:     ParmVarDecl {{.*}} 'const SpecialClassStruct1 &'
// CHECK-NEXT:   CXXMethodDecl {{.*}} implicit operator= 'SpecialClassStruct1 &(SpecialClassStruct1 &&)'
// CHECK-NEXT:     ParmVarDecl {{.*}} 'SpecialClassStruct1 &&'
// CHECK-NEXT:   CXXDestructorDecl {{.*}} implicit ~SpecialClassStruct1 'void ()'
struct __attribute__((sycl_special_class)) SpecialClassStruct1 {
  virtual void __init(
    [[__sycl_detail__::add_ir_attributes_kernel_parameter("Attr1", "Attr2", 1, true)]]
    [[__sycl_detail__::add_ir_attributes_kernel_parameter("Attr2", "Attr1", true, 1)]]
    [[__sycl_detail__::add_ir_attributes_kernel_parameter("Attr3", false)]]
    int x) {}
};

// CHECK:      CXXRecordDecl {{.*}} struct SpecialClassStruct2 definition
// CHECK-NEXT:   DefinitionData
// CHECK-NEXT:     DefaultConstructor
// CHECK-NEXT:     CopyConstructor
// CHECK-NEXT:     MoveConstructor
// CHECK-NEXT:     CopyAssignment
// CHECK-NEXT:     MoveAssignment
// CHECK-NEXT:     Destructor
// CHECK-NEXT:   SYCLSpecialClassAttr {{.*}}
// CHECK-NEXT:   CXXRecordDecl {{.*}} implicit struct SpecialClassStruct2
// CHECK-NEXT:   CXXMethodDecl {{.*}} __init 'void (int)'
// CHECK-NEXT:     ParmVarDecl {{.*}} x 'int'
// CHECK-NEXT:       SYCLAddIRAttributesKernelParameterAttr {{.*}}
// CHECK-NEXT:         InitListExpr {{.*}} 'void'
// CHECK-NEXT:           StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
// CHECK-NEXT:           StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
// CHECK-NEXT:         ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:           value: LValue
// CHECK-NEXT:           StringLiteral {{.*}} 'const char[6]' lvalue "Attr3"
// CHECK-NEXT:         ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:           value: LValue
// CHECK-NEXT:           StringLiteral {{.*}} 'const char[6]' lvalue "Attr2"
// CHECK-NEXT:         ConstantExpr {{.*}} 'const char[6]' lvalue
// CHECK-NEXT:           value: LValue
// CHECK-NEXT:           StringLiteral {{.*}} 'const char[6]' lvalue "Attr1"
// CHECK-NEXT:         ConstantExpr {{.*}} 'bool'
// CHECK-NEXT:           value: Int 0
// CHECK-NEXT:           CXXBoolLiteralExpr {{.*}} 'bool' false
// CHECK-NEXT:         ConstantExpr {{.*}} 'bool'
// CHECK-NEXT:           value: Int 1
// CHECK-NEXT:           CXXBoolLiteralExpr {{.*}} 'bool' true
// CHECK-NEXT:         ConstantExpr {{.*}} 'int'
// CHECK-NEXT:           value: Int 1
// CHECK-NEXT:           IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT:     CompoundStmt {{.*}}
// CHECK-NEXT:   CXXMethodDecl {{.*}} implicit operator= 'SpecialClassStruct2 &(const SpecialClassStruct2 &)'
// CHECK-NEXT:     ParmVarDecl {{.*}} 'const SpecialClassStruct2 &'
// CHECK-NEXT:   CXXMethodDecl {{.*}} implicit operator= 'SpecialClassStruct2 &(SpecialClassStruct2 &&)'
// CHECK-NEXT:     ParmVarDecl {{.*}} 'SpecialClassStruct2 &&'
// CHECK-NEXT:   CXXDestructorDecl {{.*}} implicit ~SpecialClassStruct2 'void ()'
struct __attribute__((sycl_special_class)) SpecialClassStruct2 {
  virtual void __init(
    [[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, "Attr1", "Attr2", 1, true)]]
    [[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, "Attr2", "Attr1", true, 1)]]
    [[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, "Attr3", false)]]
    int x) {}
};
