; Test checks if spirv.CooperativeMatrixKHR type is extracted from
; joint_matrix struct when it's used in AccessChain function call

; RUN: opt -passes=sycl-joint-matrix-transform < %s -S | FileCheck %s

; ModuleID = 'test.bc'
source_filename = "test.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir64-unknown-unknown"

%"struct.sycl::joint_matrix" = type { target("spirv.CooperativeMatrixKHR", i8, 3, 16, 64, 0) }
%"struct.sycl::_V1::long" = type { i64 }

define weak_odr dso_local spir_kernel void @test(i64 %ind) {
; CHECK-LABEL: define weak_odr dso_local spir_kernel void @test(
; CHECK-SAME: i64 [[IND:%.*]]) {

; non-matrix alloca not touched
; CHECK:         [[NOT_MATR:%.*]] = alloca [2 x [4 x %"struct.sycl::_V1::long"]]
; both matrix-related allocas updated to use target extension types
; CHECK-NEXT:    [[MATR:%.*]] = alloca target("spirv.CooperativeMatrixKHR", i8, 3, 16, 64, 0)
; CHECK-NEXT:    [[MATR_ARR:%.*]] = alloca [2 x [4 x target("spirv.CooperativeMatrixKHR", i8, 3, 16, 64, 0)]]

; CHECK-NEXT:    [[ASCAST:%.*]] = addrspacecast ptr [[MATR]] to ptr addrspace(4)
; no gep inserted, since not needed
; CHECK-NEXT:    call spir_func ptr addrspace(4) @_Z19__spirv_AccessChain{{.*}}(ptr addrspace(4) noundef [[ASCAST]], i64 noundef 0)

; CHECK:         [[GEP:%.*]] = getelementptr inbounds [2 x [4 x %"struct.sycl::joint_matrix"]], ptr [[MATR_ARR]], i64 0, i64 [[IND]], i64 [[IND]]
; CHECK-NEXT:    [[ASCAST_1:%.*]] = addrspacecast ptr [[GEP]] to ptr addrspace(4)
; CHECK-NEXT:    [[ASCAST_2:%.*]] = addrspacecast ptr [[GEP]] to ptr addrspace(4)
; gep is inserted for each of the accesschain calls to extract target extension type
; CHECK-NEXT:    [[TMP2:%.*]] = getelementptr inbounds %"struct.sycl::joint_matrix", ptr addrspace(4) [[ASCAST_1]], i64 0, i32 0
; CHECK-NEXT:    call spir_func ptr addrspace(4) @_Z19__spirv_AccessChain{{.*}}(ptr addrspace(4) noundef [[TMP2]], i64 noundef 0)
; CHECK:         [[TMP5:%.*]] = getelementptr inbounds %"struct.sycl::joint_matrix", ptr addrspace(4) [[ASCAST_2]], i64 0, i32 0
; CHECK-NEXT:    call spir_func ptr addrspace(4) @_Z19__spirv_AccessChain{{.*}}(ptr addrspace(4) noundef [[TMP5]], i64 noundef 0)

; negative test - not touching non-matrix code
; CHECK:         [[GEP_1:%.*]] = getelementptr inbounds [2 x [4 x %"struct.sycl::_V1::long"]], ptr [[NOT_MATR]], i64 0, i64 [[IND]], i64 [[IND]]
; CHECK-NEXT:    [[ASCAST_3:%.*]] = addrspacecast ptr [[GEP_1]] to ptr addrspace(4)
; CHECK-NEXT:    call spir_func ptr addrspace(4) @_Z19__spirv_AccessChain{{.*}}(ptr addrspace(4) noundef [[ASCAST_3]], i64 noundef 0)

entry:
  ; allocas
  %matr = alloca %"struct.sycl::joint_matrix", align 8
  %matr.arr = alloca [2 x [4 x %"struct.sycl::joint_matrix"]], align 8
  %not.matr = alloca [2 x [4 x %"struct.sycl::_V1::long"]], align 8

  ; simple case
  %ascast = addrspacecast ptr %matr to ptr addrspace(4)
  %0 = call spir_func ptr addrspace(4) @_Z19__spirv_AccessChainIiiLm16ELm16ELN5__spv9MatrixUseE2ELNS0_5Scope4FlagE3EEPT_PPNS0_28__spirv_CooperativeMatrixKHRIT0(ptr addrspace(4) noundef %ascast, i64 noundef 0)
  %1 = load i8, ptr addrspace(4) %0

  ; gep with non-zero inidices and multiple access chains per 1 alloca
  %gep = getelementptr inbounds [2 x [4 x %"struct.sycl::joint_matrix"]], ptr %matr.arr, i64 0, i64 %ind, i64 %ind
  %ascast.1 = addrspacecast ptr %gep to ptr addrspace(4)
  %ascast.2 = addrspacecast ptr %gep to ptr addrspace(4)
  %2 = call spir_func ptr addrspace(4) @_Z19__spirv_AccessChainIiiLm16ELm16ELN5__spv9MatrixUseE2ELNS0_5Scope4FlagE3EEPT_PPNS0_28__spirv_CooperativeMatrixKHRIT0(ptr addrspace(4) noundef %ascast.1, i64 noundef 0)
  %3 = load i8, ptr addrspace(4) %2
  %4 = call spir_func ptr addrspace(4) @_Z19__spirv_AccessChainIiiLm16ELm16ELN5__spv9MatrixUseE2ELNS0_5Scope4FlagE3EEPT_PPNS0_28__spirv_CooperativeMatrixKHRIT0(ptr addrspace(4) noundef %ascast.2, i64 noundef 0)
  %5 = load i8, ptr addrspace(4) %4

  ; negative test - not touching non-matrix code
  %gep.1 = getelementptr inbounds [2 x [4 x %"struct.sycl::_V1::long"]], ptr %not.matr, i64 0, i64 %ind, i64 %ind
  %ascast.3 = addrspacecast ptr %gep.1 to ptr addrspace(4)
  %6 = call spir_func ptr addrspace(4) @_Z19__spirv_AccessChainIiiLm16ELm16ELN5__spv9MatrixUseE2ELNS0_5Scope4FlagE3EEPT_PPNS0_28__spirv_CooperativeMatrixKHRIT0(ptr addrspace(4) noundef %ascast.3, i64 noundef 0)
  %7 = load i8, ptr addrspace(4) %6

  ret void
}

declare dso_local spir_func ptr addrspace(4) @_Z19__spirv_AccessChainIiiLm16ELm16ELN5__spv9MatrixUseE2ELNS0_5Scope4FlagE3EEPT_PPNS0_28__spirv_CooperativeMatrixKHRIT0(ptr addrspace(4) noundef, i64 noundef)
