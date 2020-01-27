// RUN: %clang_cc1 -triple x86_64-gnu-linux -O3 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,LIN,NoNewStructPathTBAA
// RUN: %clang_cc1 -triple x86_64-gnu-linux -O3 -disable-llvm-passes -new-struct-path-tbaa -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,LIN,NewStructPathTBAA

// RUN: %clang_cc1 -triple x86_64-windows-pc -O3 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,WIN,NoNewStructPathTBAA
// RUN: %clang_cc1 -triple x86_64-windows-pc -O3 -disable-llvm-passes -new-struct-path-tbaa -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,WIN,NewStructPathTBAA


// Ensure that the layout for these structs is the same as the normal bitfield
// layouts.
struct BitFieldsByte {
  _ExtInt(7) A : 3;
  _ExtInt(7) B : 3;
  _ExtInt(7) C : 2;
};
// CHECK: %struct.BitFieldsByte = type { i8 }

struct BitFieldsShort {
  _ExtInt(15) A : 3;
  _ExtInt(15) B : 3;
  _ExtInt(15) C : 2;
};
// LIN: %struct.BitFieldsShort = type { i8, i8 }
// WIN: %struct.BitFieldsShort = type { i16 }

struct BitFieldsInt {
  _ExtInt(31) A : 3;
  _ExtInt(31) B : 3;
  _ExtInt(31) C : 2;
};
// LIN: %struct.BitFieldsInt = type { i8, [3 x i8] }
// WIN: %struct.BitFieldsInt = type { i32 }

struct BitFieldsLong {
  _ExtInt(63) A : 3;
  _ExtInt(63) B : 3;
  _ExtInt(63) C : 2;
};
// LIN: %struct.BitFieldsLong = type { i8, [7 x i8] }
// WIN: %struct.BitFieldsLong = type { i64 }

struct HasExtIntFirst {
  _ExtInt(35) A;
  int B;
};
// CHECK: %struct.HasExtIntFirst = type { i35, i32 }

struct HasExtIntLast {
  int A;
  _ExtInt(35) B;
};
// CHECK: %struct.HasExtIntLast = type { i32, i35 }

struct HasExtIntMiddle {
  int A;
  _ExtInt(35) B;
  int C;
};
// CHECK: %struct.HasExtIntMiddle = type { i32, i35, i32 }

// Force emitting of the above structs.
void StructEmit() {
  BitFieldsByte A;
  BitFieldsShort B;
  BitFieldsInt C;
  BitFieldsLong D;

  HasExtIntFirst E;
  HasExtIntLast F;
  HasExtIntMiddle G;
}

void BitfieldAssignment() {
  // LIN: define void @_Z18BitfieldAssignmentv
  // WIN: define dso_local void  @"?BitfieldAssignment@@YAXXZ"
  BitFieldsByte B;
  B.A = 3;
  B.B = 2;
  B.C = 1;
  // First one is used for the lifetime start, skip that.
  // CHECK: bitcast %struct.BitFieldsByte*
  // CHECK: %[[BFType:.+]] = bitcast %struct.BitFieldsByte*
  // CHECK: %[[LOADA:.+]] = load i8, i8* %[[BFType]]
  // CHECK: %[[CLEARA:.+]] = and i8 %[[LOADA]], -8
  // CHECK: %[[SETA:.+]] = or i8 %[[CLEARA]], 3
  // CHECK: %[[BFType:.+]] = bitcast %struct.BitFieldsByte*
  // CHECK: %[[LOADB:.+]] = load i8, i8* %[[BFType]]
  // CHECK: %[[CLEARB:.+]] = and i8 %[[LOADB]], -57
  // CHECK: %[[SETB:.+]] = or i8 %[[CLEARB]], 16
  // CHECK: %[[BFType:.+]] = bitcast %struct.BitFieldsByte*
  // CHECK: %[[LOADC:.+]] = load i8, i8* %[[BFType]]
  // CHECK: %[[CLEARC:.+]] = and i8 %[[LOADC]], 63
  // CHECK: %[[SETC:.+]] = or i8 %[[CLEARC]], 64
}

enum AsEnumUnderlyingType : _ExtInt(9) {
  A,B,C
};

void UnderlyingTypeUseage(AsEnumUnderlyingType Param) {
  // LIN: define void @_Z20UnderlyingTypeUseage20AsEnumUnderlyingType(i9 %
  // WIN: define dso_local void @"?UnderlyingTypeUseage@@YAXW4AsEnumUnderlyingType@@@Z"(i9 %
  AsEnumUnderlyingType Var;
  // CHECK: alloca i9, align 2
  // CHECK: store i9 %{{.*}}, align 2
}

unsigned _ExtInt(33) ManglingTestRetParam(unsigned _ExtInt(33) Param) {
// LIN: define i33 @_Z20ManglingTestRetParamU8_UExtIntLj33E(i33 %
// WIN: define dso_local i33 @"?ManglingTestRetParam@@YAU?$_UExtInt@$0CB@@__clang@@U12@@Z"(i33
  return 0;
}

_ExtInt(33) ManglingTestRetParam(_ExtInt(33) Param) {
// LIN: define i33 @_Z20ManglingTestRetParamU7_ExtIntLj33E(i33 %
// WIN: define dso_local i33 @"?ManglingTestRetParam@@YAU?$_ExtInt@$0CB@@__clang@@U12@@Z"(i33
  return 0;
}

template<typename T>
void ManglingTestTemplateParam(T&);
template<_ExtInt(99) T>
void ManglingTestNTTP();

void ManglingInstantiator() {
  // LIN: define void @_Z20ManglingInstantiatorv()
  // WIN: define dso_local void @"?ManglingInstantiator@@YAXXZ"()
  _ExtInt(93) A;
  ManglingTestTemplateParam(A);
// LIN: call void @_Z25ManglingTestTemplateParamIU7_ExtIntLj93EEvRT_(i93*
// WIN: call void @"??$ManglingTestTemplateParam@U?$_ExtInt@$0FN@@__clang@@@@YAXAEAU?$_ExtInt@$0FN@@__clang@@@Z"(i93*
  constexpr _ExtInt(93) B = 993;
  ManglingTestNTTP<38>();
// LIN: call void @_Z16ManglingTestNTTPILU7_ExtIntLj99E38EEvv()
// WIN: call void @"??$ManglingTestNTTP@$0CG@@@YAXXZ"()
  ManglingTestNTTP<B>();
// LIN: call void @_Z16ManglingTestNTTPILU7_ExtIntLj99E993EEvv()
// WIN: call void @"??$ManglingTestNTTP@$0DOB@@@YAXXZ"()
}

// TODO: UBSan, and in particular -fsanitize=signed-integer-overflow (this will break; UBSan's static data representation assumes that integer types have power-of-two sizes)

void ShiftExtIntByConstant(_ExtInt(28) Ext) {
// LIN: define void @_Z21ShiftExtIntByConstantU7_ExtIntLj28E
// WIN: define dso_local void @"?ShiftExtIntByConstant@@YAXU?$_ExtInt@$0BM@@__clang@@@Z"
  Ext << 7;
  // CHECK: shl i28 %{{.+}}, 7
  Ext >> 7;
  // CHECK: ashr i28 %{{.+}}, 7
  Ext << -7;
  // CHECK: shl i28 %{{.+}}, -7
  Ext >> -7;
  // CHECK: ashr i28 %{{.+}}, -7

  // UB in C/C++, Defined in OpenCL.
  Ext << 29;
  // CHECK: shl i28 %{{.+}}, 29 
  Ext >> 29;
  // CHECK: ashr i28 %{{.+}}, 29
}

void ConstantShiftByExtInt(_ExtInt(28) Ext, _ExtInt(65) LargeExt) {
  // LIN: define void @_Z21ConstantShiftByExtIntU7_ExtIntLj28EU7_ExtIntLj65E
  // WIN: define dso_local void @"?ConstantShiftByExtInt@@YAXU?$_ExtInt@$0BM@@__clang@@U?$_ExtInt@$0EB@@2@@Z"
  10 << Ext;
  // CHECK: %[[PROMO:.+]] = zext i28 %{{.+}} to i32
  // CHECK: shl i32 10, %[[PROMO]]
  10 >> Ext;
  // CHECK: %[[PROMO:.+]] = zext i28 %{{.+}} to i32
  // CHECK: ashr i32 10, %[[PROMO]]
  10 << LargeExt;
  // CHECK: %[[PROMO:.+]] = trunc i65 %{{.+}} to i32
  // CHECK: shl i32 10, %[[PROMO]]
  10 >> LargeExt;
  // CHECK: %[[PROMO:.+]] = trunc i65 %{{.+}} to i32
  // CHECK: ashr i32 10, %[[PROMO]]
}

void Shift(_ExtInt(28) Ext, _ExtInt(65) LargeExt, int i) {
  // LIN: define void @_Z5ShiftU7_ExtIntLj28EU7_ExtIntLj65Ei
  // WIN: define dso_local void @"?Shift@@YAXU?$_ExtInt@$0BM@@__clang@@U?$_ExtInt@$0EB@@2@H@Z"
  i << Ext;
  // CHECK: %[[PROMO:.+]] = zext i28 %{{.+}} to i32
  // CHECK: shl i32 {{.+}}, %[[PROMO]]
  i >> Ext;
  // CHECK: %[[PROMO:.+]] = zext i28 %{{.+}} to i32
  // CHECK: ashr i32 {{.+}}, %[[PROMO]]

  i << LargeExt;
  // CHECK: %[[PROMO:.+]] = trunc i65 %{{.+}} to i32
  // CHECK: shl i32 {{.+}}, %[[PROMO]]
  i >> LargeExt;
  // CHECK: %[[PROMO:.+]] = trunc i65 %{{.+}} to i32
  // CHECK: ashr i32 {{.+}}, %[[PROMO]]

  Ext << i;
  // CHECK: %[[PROMO:.+]] = trunc i32 %{{.+}} to i28
  // CHECK: shl i28 {{.+}}, %[[PROMO]]
  Ext >> i;
  // CHECK: %[[PROMO:.+]] = trunc i32 %{{.+}} to i28
  // CHECK: ashr i28 {{.+}}, %[[PROMO]]

  LargeExt << i;
  // CHECK: %[[PROMO:.+]] = zext i32 %{{.+}} to i65
  // CHECK: shl i65 {{.+}}, %[[PROMO]]
  LargeExt >> i;
  // CHECK: %[[PROMO:.+]] = zext i32 %{{.+}} to i65
  // CHECK: ashr i65 {{.+}}, %[[PROMO]]

  Ext << LargeExt;
  // CHECK: %[[PROMO:.+]] = trunc i65 %{{.+}} to i28
  // CHECK: shl i28 {{.+}}, %[[PROMO]]
  Ext >> LargeExt;
  // CHECK: %[[PROMO:.+]] = trunc i65 %{{.+}} to i28
  // CHECK: ashr i28 {{.+}}, %[[PROMO]]

  LargeExt << Ext;
  // CHECK: %[[PROMO:.+]] = zext i28 %{{.+}} to i65
  // CHECK: shl i65 {{.+}}, %[[PROMO]]
  LargeExt >> Ext;
  // CHECK: %[[PROMO:.+]] = zext i28 %{{.+}} to i65
  // CHECK: ashr i65 {{.+}}, %[[PROMO]]
}

// Ensure that these types don't alias the normal int types.
void TBAATest(_ExtInt(sizeof(int) * 8) ExtInt,
              unsigned _ExtInt(sizeof(int) * 8) ExtUInt,
              _ExtInt(6) Other) {
  // CHECK-DAG: store i32 %{{.+}}, i32* %{{.+}}, align 4, !tbaa ![[EXTINT_TBAA:.+]]
  // CHECK-DAG: store i32 %{{.+}}, i32* %{{.+}}, align 4, !tbaa ![[EXTINT_TBAA]]
  // CHECK-DAG: store i6 %{{.+}}, i6* %{{.+}}, align 1, !tbaa ![[EXTINT6_TBAA:.+]]
  ExtInt = 5;
  ExtUInt = 5;
  Other = 5;
}


// NoNewStructPathTBAA-DAG: ![[CHAR_TBAA_ROOT:.+]] = !{!"omnipotent char", ![[TBAA_ROOT:.+]], i64 0}
// NoNewStructPathTBAA-DAG: ![[TBAA_ROOT]] = !{!"Simple C++ TBAA"}
// NoNewStructPathTBAA-DAG: ![[EXTINT_TBAA]] = !{![[EXTINT_TBAA_ROOT:.+]], ![[EXTINT_TBAA_ROOT]], i64 0}
// NoNewStructPathTBAA-DAG: ![[EXTINT_TBAA_ROOT]] = !{!"_ExtInt(32)", ![[CHAR_TBAA_ROOT]], i64 0}
// NoNewStructPathTBAA-DAG: ![[EXTINT6_TBAA]] = !{![[EXTINT6_TBAA_ROOT:.+]], ![[EXTINT6_TBAA_ROOT]], i64 0}
// NoNewStructPathTBAA-DAG: ![[EXTINT6_TBAA_ROOT]] = !{!"_ExtInt(6)", ![[CHAR_TBAA_ROOT]], i64 0}

// NewStructPathTBAA-DAG: ![[CHAR_TBAA_ROOT:.+]] = !{![[TBAA_ROOT:.+]], i64 1, !"omnipotent char"}
// NewStructPathTBAA-DAG: ![[TBAA_ROOT]] = !{!"Simple C++ TBAA"}
// NewStructPathTBAA-DAG: ![[EXTINT_TBAA]] = !{![[EXTINT_TBAA_ROOT:.+]], ![[EXTINT_TBAA_ROOT]], i64 0, i64 4}
// NewStructPathTBAA-DAG: ![[EXTINT_TBAA_ROOT]] = !{![[CHAR_TBAA_ROOT]], i64 4, !"_ExtInt(32)"}
// NewStructPathTBAA-DAG: ![[EXTINT6_TBAA]] = !{![[EXTINT6_TBAA_ROOT:.+]], ![[EXTINT6_TBAA_ROOT]], i64 0, i64 1}
// NewStructPathTBAA-DAG: ![[EXTINT6_TBAA_ROOT]] = !{![[CHAR_TBAA_ROOT]], i64 1, !"_ExtInt(6)"}
