; REQUIRES: spirv-dis
; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-dis --raw-id %t.spv | FileCheck --check-prefix CHECK-SPIRV %s
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r -o %t.rev.bc %t.spv
; RUN: llvm-dis -o - %t.rev.bc | FileCheck --check-prefix CHECK-LLVM %s

target triple = "spir64-unknown-unknown"

; CHECK-SPIRV-DAG:                   [[ushort:%[a-z0-9_.]+]] = OpTypeInt 16 0
; CHECK-SPIRV-DAG:                     [[uint:%[a-z0-9_.]+]] = OpTypeInt 32 0
; CHECK-SPIRV-DAG:                    [[ulong:%[a-z0-9_.]+]] = OpTypeInt 64 0
; CHECK-SPIRV-DAG:                 [[ushort_0:%[a-z0-9_.]+]] = OpConstant [[ushort]] 0
; CHECK-SPIRV-DAG:                   [[uint_0:%[a-z0-9_.]+]] = OpConstant [[uint]] 0
; CHECK-SPIRV-DAG:                  [[ulong_0:%[a-z0-9_.]+]] = OpConstant [[ulong]] 0
; CHECK-SPIRV-DAG:                     [[bool:%[a-z0-9_.]+]] = OpTypeBool
; CHECK-SPIRV-DAG:                    [[var_4:%[a-z0-9_.]+]] = OpTypeFunction [[bool]] [[ushort]] [[ushort]]
; CHECK-SPIRV-DAG:                [[_struct_9:%[a-z0-9_.]+]] = OpTypeStruct [[ushort]] [[ushort]]
; CHECK-SPIRV-DAG:  [[_ptr_Function__struct_9:%[a-z0-9_.]+]] = OpTypePointer Function [[_struct_9]]
; CHECK-SPIRV-DAG:               [[_struct_18:%[a-z0-9_.]+]] = OpTypeStruct [[ushort]] [[bool]]
; CHECK-SPIRV-DAG:                   [[var_25:%[a-z0-9_.]+]] = OpTypeFunction [[bool]] [[uint]] [[uint]]
; CHECK-SPIRV-DAG:               [[_struct_30:%[a-z0-9_.]+]] = OpTypeStruct [[uint]] [[uint]]
; CHECK-SPIRV-DAG: [[_ptr_Function__struct_30:%[a-z0-9_.]+]] = OpTypePointer Function [[_struct_30]]
; CHECK-SPIRV-DAG:               [[_struct_39:%[a-z0-9_.]+]] = OpTypeStruct [[uint]] [[bool]]
; CHECK-SPIRV-DAG:                   [[var_46:%[a-z0-9_.]+]] = OpTypeFunction [[bool]] [[ulong]] [[ulong]]
; CHECK-SPIRV-DAG:               [[_struct_51:%[a-z0-9_.]+]] = OpTypeStruct [[ulong]] [[ulong]]
; CHECK-SPIRV-DAG: [[_ptr_Function__struct_51:%[a-z0-9_.]+]] = OpTypePointer Function [[_struct_51]]
; CHECK-SPIRV-DAG:               [[_struct_60:%[a-z0-9_.]+]] = OpTypeStruct [[ulong]] [[bool]]
; CHECK-SPIRV-DAG:                   [[v4bool:%[a-z0-9_.]+]] = OpTypeVector [[bool]] 4
; CHECK-SPIRV-DAG:                   [[v4uint:%[a-z0-9_.]+]] = OpTypeVector [[uint]] 4
; CHECK-SPIRV-DAG:                   [[var_68:%[a-z0-9_.]+]] = OpTypeFunction [[v4bool]] [[v4uint]] [[v4uint]]
; CHECK-SPIRV-DAG:               [[_struct_73:%[a-z0-9_.]+]] = OpTypeStruct [[v4uint]] [[v4uint]]
; CHECK-SPIRV-DAG: [[_ptr_Function__struct_73:%[a-z0-9_.]+]] = OpTypePointer Function [[_struct_73]]
; CHECK-SPIRV-DAG:               [[_struct_82:%[a-z0-9_.]+]] = OpTypeStruct [[v4uint]] [[v4bool]]
; CHECK-SPIRV-DAG:                   [[var_19:%[a-z0-9_.]+]] = OpUndef [[_struct_18]]
; CHECK-SPIRV-DAG:                   [[var_40:%[a-z0-9_.]+]] = OpUndef [[_struct_39]]
; CHECK-SPIRV-DAG:                   [[var_61:%[a-z0-9_.]+]] = OpUndef [[_struct_60]]
; CHECK-SPIRV-DAG:                   [[var_80:%[a-z0-9_.]+]] = OpConstantNull [[v4uint]]
; CHECK-SPIRV-DAG:                   [[var_83:%[a-z0-9_.]+]] = OpUndef [[_struct_82]]

; CHECK-LLVM-DAG: [[structtype:%[a-z0-9._]+]] = type { i16, i16 }
; CHECK-LLVM-DAG: [[structtype_0:%[a-z0-9._]+]] = type { i16, i1 }
; CHECK-LLVM-DAG: [[structtype_1:%[a-z0-9._]+]] = type { i32, i32 }
; CHECK-LLVM-DAG: [[structtype_2:%[a-z0-9._]+]] = type { i32, i1 }
; CHECK-LLVM-DAG: [[structtype_3:%[a-z0-9._]+]] = type { i64, i64 }
; CHECK-LLVM-DAG: [[structtype_4:%[a-z0-9._]+]] = type { i64, i1 }
; CHECK-LLVM-DAG: [[structtype_5:%[a-z0-9._]+]] = type { <4 x i32>, <4 x i32> }
; CHECK-LLVM-DAG: [[structtype_6:%[a-z0-9._]+]] = type { <4 x i32>, <4 x i1> }

define spir_func i1 @test_usub_with_overflow_i16(i16 %a, i16 %b) {
entry:
  %res = call {i16, i1} @llvm.usub.with.overflow.i16(i16 %a, i16 %b)
  %0 = extractvalue {i16, i1} %res, 0
  %1 = extractvalue {i16, i1} %res, 1
  ret i1 %1
}

; CHECK-SPIRV:               [[a:%[a-z0-9_.]+]] = OpFunctionParameter [[ushort]]
; CHECK-SPIRV:               [[b:%[a-z0-9_.]+]] = OpFunctionParameter [[ushort]]
; CHECK-SPIRV:           [[entry:%[a-z0-9_.]+]] = OpLabel
; CHECK-SPIRV:          [[var_11:%[a-z0-9_.]+]] = OpVariable [[_ptr_Function__struct_9]] Function
; CHECK-SPIRV:          [[var_12:%[a-z0-9_.]+]] = OpISubBorrow [[_struct_9]] [[a]] [[b]]
; CHECK-SPIRV:                                    OpStore [[var_11]] [[var_12]]
; CHECK-SPIRV:          [[var_13:%[a-z0-9_.]+]] = OpLoad [[_struct_9]] [[var_11]] Aligned 2
; CHECK-SPIRV:          [[var_14:%[a-z0-9_.]+]] = OpCompositeExtract [[ushort]] [[var_13]] 0
; CHECK-SPIRV:          [[var_15:%[a-z0-9_.]+]] = OpCompositeExtract [[ushort]] [[var_13]] 1
; CHECK-SPIRV:          [[var_17:%[a-z0-9_.]+]] = OpINotEqual [[bool]] [[var_15]] [[ushort_0]]
; CHECK-SPIRV:          [[var_20:%[a-z0-9_.]+]] = OpCompositeInsert [[_struct_18]] [[var_14]] [[var_19]] 0
; CHECK-SPIRV:          [[var_21:%[a-z0-9_.]+]] = OpCompositeInsert [[_struct_18]] [[var_17]] [[var_20]] 1
; CHECK-SPIRV:          [[var_22:%[a-z0-9_.]+]] = OpCompositeExtract [[ushort]] [[var_21]] 0
; CHECK-SPIRV:          [[var_23:%[a-z0-9_.]+]] = OpCompositeExtract [[bool]] [[var_21]] 1
; CHECK-SPIRV:                                    OpReturnValue [[var_23]]

; CHECK-LLVM:   %0 = alloca [[structtype]], align 8
; CHECK-LLVM:   call spir_func void @_Z18__spirv_ISubBorrowss(ptr sret([[structtype]]) %0, i16 %a, i16 %b)
; CHECK-LLVM:   %1 = load [[structtype]], ptr %0, align 2
; CHECK-LLVM:   %2 = extractvalue [[structtype]] %1, 0
; CHECK-LLVM:   %3 = extractvalue [[structtype]] %1, 1
; CHECK-LLVM:   %4 = icmp ne i16 %3, 0
; CHECK-LLVM:   %5 = insertvalue [[structtype_0]] undef, i16 %2, 0
; CHECK-LLVM:   %6 = insertvalue [[structtype_0]] %5, i1 %4, 1
; CHECK-LLVM:   %7 = extractvalue [[structtype_0]] %6, 0
; CHECK-LLVM:   %8 = extractvalue [[structtype_0]] %6, 1
; CHECK-LLVM:   ret i1 %8
define spir_func i1 @test_usub_with_overflow_i32(i32 %a, i32 %b) {
entry:
  %res = call {i32, i1} @llvm.usub.with.overflow.i32(i32 %a, i32 %b)
  %0 = extractvalue {i32, i1} %res, 0
  %1 = extractvalue {i32, i1} %res, 1
  ret i1 %1
}

; CHECK-SPIRV:             [[a_0:%[a-z0-9_.]+]] = OpFunctionParameter [[uint]]
; CHECK-SPIRV:             [[b_0:%[a-z0-9_.]+]] = OpFunctionParameter [[uint]]
; CHECK-SPIRV:         [[entry_0:%[a-z0-9_.]+]] = OpLabel
; CHECK-SPIRV:          [[var_32:%[a-z0-9_.]+]] = OpVariable [[_ptr_Function__struct_30]] Function
; CHECK-SPIRV:          [[var_33:%[a-z0-9_.]+]] = OpISubBorrow [[_struct_30]] [[a_0]] [[b_0]]
; CHECK-SPIRV:                                    OpStore [[var_32]] [[var_33]]
; CHECK-SPIRV:          [[var_34:%[a-z0-9_.]+]] = OpLoad [[_struct_30]] [[var_32]] Aligned 4
; CHECK-SPIRV:          [[var_35:%[a-z0-9_.]+]] = OpCompositeExtract [[uint]] [[var_34]] 0
; CHECK-SPIRV:          [[var_36:%[a-z0-9_.]+]] = OpCompositeExtract [[uint]] [[var_34]] 1
; CHECK-SPIRV:          [[var_38:%[a-z0-9_.]+]] = OpINotEqual [[bool]] [[var_36]] [[uint_0]]
; CHECK-SPIRV:          [[var_41:%[a-z0-9_.]+]] = OpCompositeInsert [[_struct_39]] [[var_35]] [[var_40]] 0
; CHECK-SPIRV:          [[var_42:%[a-z0-9_.]+]] = OpCompositeInsert [[_struct_39]] [[var_38]] [[var_41]] 1
; CHECK-SPIRV:          [[var_43:%[a-z0-9_.]+]] = OpCompositeExtract [[uint]] [[var_42]] 0
; CHECK-SPIRV:          [[var_44:%[a-z0-9_.]+]] = OpCompositeExtract [[bool]] [[var_42]] 1
; CHECK-SPIRV:                                    OpReturnValue [[var_44]]


; CHECK-LLVM:   %0 = alloca [[structtype_1]], align 8
; CHECK-LLVM:   call spir_func void @_Z18__spirv_ISubBorrowii(ptr sret([[structtype_1]]) %0, i32 %a, i32 %b)
; CHECK-LLVM:   %1 = load [[structtype_1]], ptr %0, align 4
; CHECK-LLVM:   %2 = extractvalue [[structtype_1]] %1, 0
; CHECK-LLVM:   %3 = extractvalue [[structtype_1]] %1, 1
; CHECK-LLVM:   %4 = icmp ne i32 %3, 0
; CHECK-LLVM:   %5 = insertvalue [[structtype_2]] undef, i32 %2, 0
; CHECK-LLVM:   %6 = insertvalue [[structtype_2]] %5, i1 %4, 1
; CHECK-LLVM:   %7 = extractvalue [[structtype_2]] %6, 0
; CHECK-LLVM:   %8 = extractvalue [[structtype_2]] %6, 1
; CHECK-LLVM:   ret i1 %8
define spir_func i1 @test_usub_with_overflow_i64(i64 %a, i64 %b) {
entry:
  %res = call {i64, i1} @llvm.usub.with.overflow.i64(i64 %a, i64 %b)
  %0 = extractvalue {i64, i1} %res, 0
  %1 = extractvalue {i64, i1} %res, 1
  ret i1 %1
}

; CHECK-SPIRV:             [[a_1:%[a-z0-9_.]+]] = OpFunctionParameter [[ulong]]
; CHECK-SPIRV:             [[b_1:%[a-z0-9_.]+]] = OpFunctionParameter [[ulong]]
; CHECK-SPIRV:         [[entry_1:%[a-z0-9_.]+]] = OpLabel
; CHECK-SPIRV:          [[var_53:%[a-z0-9_.]+]] = OpVariable [[_ptr_Function__struct_51]] Function
; CHECK-SPIRV:          [[var_54:%[a-z0-9_.]+]] = OpISubBorrow [[_struct_51]] [[a_1]] [[b_1]]
; CHECK-SPIRV:                                    OpStore [[var_53]] [[var_54]]
; CHECK-SPIRV:          [[var_55:%[a-z0-9_.]+]] = OpLoad [[_struct_51]] [[var_53]] Aligned 4
; CHECK-SPIRV:          [[var_56:%[a-z0-9_.]+]] = OpCompositeExtract [[ulong]] [[var_55]] 0
; CHECK-SPIRV:          [[var_57:%[a-z0-9_.]+]] = OpCompositeExtract [[ulong]] [[var_55]] 1
; CHECK-SPIRV:          [[var_59:%[a-z0-9_.]+]] = OpINotEqual [[bool]] [[var_57]] [[ulong_0]]
; CHECK-SPIRV:          [[var_62:%[a-z0-9_.]+]] = OpCompositeInsert [[_struct_60]] [[var_56]] [[var_61]] 0
; CHECK-SPIRV:          [[var_63:%[a-z0-9_.]+]] = OpCompositeInsert [[_struct_60]] [[var_59]] [[var_62]] 1
; CHECK-SPIRV:          [[var_64:%[a-z0-9_.]+]] = OpCompositeExtract [[ulong]] [[var_63]] 0
; CHECK-SPIRV:          [[var_65:%[a-z0-9_.]+]] = OpCompositeExtract [[bool]] [[var_63]] 1
; CHECK-SPIRV:                                    OpReturnValue [[var_65]]

; CHECK-LLVM:   %0 = alloca [[structtype_3]], align 8
; CHECK-LLVM:   call spir_func void @_Z18__spirv_ISubBorrowll(ptr sret([[structtype_3]]) %0, i64 %a, i64 %b)
; CHECK-LLVM:   %1 = load [[structtype_3]], ptr %0, align 4
; CHECK-LLVM:   %2 = extractvalue [[structtype_3]] %1, 0
; CHECK-LLVM:   %3 = extractvalue [[structtype_3]] %1, 1
; CHECK-LLVM:   %4 = icmp ne i64 %3, 0
; CHECK-LLVM:   %5 = insertvalue [[structtype_4]] undef, i64 %2, 0
; CHECK-LLVM:   %6 = insertvalue [[structtype_4]] %5, i1 %4, 1
; CHECK-LLVM:   %7 = extractvalue [[structtype_4]] %6, 0
; CHECK-LLVM:   %8 = extractvalue [[structtype_4]] %6, 1
; CHECK-LLVM:   ret i1 %8
define spir_func <4 x i1> @test_usub_with_overflow_v4i32(<4 x i32> %a, <4 x i32> %b) {
entry:
  %res = call {<4 x i32>, <4 x i1>} @llvm.usub.with.overflow.v4i32(<4 x i32> %a, <4 x i32> %b) 
  %0 = extractvalue {<4 x i32>, <4 x i1>} %res, 0
  %1 = extractvalue {<4 x i32>, <4 x i1>} %res, 1
  ret <4 x i1> %1
}

; CHECK-SPIRV:             [[a_2:%[a-z0-9_.]+]] = OpFunctionParameter [[v4uint]]
; CHECK-SPIRV:             [[b_2:%[a-z0-9_.]+]] = OpFunctionParameter [[v4uint]]
; CHECK-SPIRV:         [[entry_2:%[a-z0-9_.]+]] = OpLabel
; CHECK-SPIRV:          [[var_75:%[a-z0-9_.]+]] = OpVariable [[_ptr_Function__struct_73]] Function
; CHECK-SPIRV:          [[var_76:%[a-z0-9_.]+]] = OpISubBorrow [[_struct_73]] [[a_2]] [[b_2]]
; CHECK-SPIRV:                                    OpStore [[var_75]] [[var_76]]
; CHECK-SPIRV:          [[var_77:%[a-z0-9_.]+]] = OpLoad [[_struct_73]] [[var_75]] Aligned 16
; CHECK-SPIRV:          [[var_78:%[a-z0-9_.]+]] = OpCompositeExtract [[v4uint]] [[var_77]] 0
; CHECK-SPIRV:          [[var_79:%[a-z0-9_.]+]] = OpCompositeExtract [[v4uint]] [[var_77]] 1
; CHECK-SPIRV:          [[var_81:%[a-z0-9_.]+]] = OpINotEqual [[v4bool]] [[var_79]] [[var_80]]
; CHECK-SPIRV:          [[var_84:%[a-z0-9_.]+]] = OpCompositeInsert [[_struct_82]] [[var_78]] [[var_83]] 0
; CHECK-SPIRV:          [[var_85:%[a-z0-9_.]+]] = OpCompositeInsert [[_struct_82]] [[var_81]] [[var_84]] 1
; CHECK-SPIRV:          [[var_86:%[a-z0-9_.]+]] = OpCompositeExtract [[v4uint]] [[var_85]] 0
; CHECK-SPIRV:          [[var_87:%[a-z0-9_.]+]] = OpCompositeExtract [[v4bool]] [[var_85]] 1
; CHECK-SPIRV:                                    OpReturnValue [[var_87]]

; CHECK-LLVM:   %0 = alloca [[structtype_5]], align 16
; CHECK-LLVM:   call spir_func void @_Z18__spirv_ISubBorrowDv4_iS_(ptr sret([[structtype_5]]) %0, <4 x i32> %a, <4 x i32> %b)
; CHECK-LLVM:   %1 = load [[structtype_5]], ptr %0, align 16
; CHECK-LLVM:   %2 = extractvalue [[structtype_5]] %1, 0
; CHECK-LLVM:   %3 = extractvalue [[structtype_5]] %1, 1
; CHECK-LLVM:   %4 = icmp ne <4 x i32> %3, zeroinitializer
; CHECK-LLVM:   %5 = insertvalue [[structtype_6]] undef, <4 x i32> %2, 0
; CHECK-LLVM:   %6 = insertvalue [[structtype_6]] %5, <4 x i1> %4, 1
; CHECK-LLVM:   %7 = extractvalue [[structtype_6]] %6, 0
; CHECK-LLVM:   %8 = extractvalue [[structtype_6]] %6, 1
; CHECK-LLVM:   ret <4 x i1> %8
declare {i16, i1} @llvm.usub.with.overflow.i16(i16 %a, i16 %b)
declare {i32, i1} @llvm.usub.with.overflow.i32(i32 %a, i32 %b)
declare {i64, i1} @llvm.usub.with.overflow.i64(i64 %a, i64 %b)
declare {<4 x i32>, <4 x i1>} @llvm.usub.with.overflow.v4i32(<4 x i32> %a, <4 x i32> %b)
declare void @_Z18__spirv_ISubBorrowii(ptr sret({i32, i32}), i32, i32)
