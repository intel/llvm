; The test checks, that users of the call to joint_matrix_layout_to_spv matrix
; are replaced with the layout constant.

; RUN: opt -passes=sycl-joint-matrix-transform < %s -S | FileCheck %s

; ModuleID = 'test.bc'
source_filename = "test.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir64-unknown-unknown"

$joint_matrix_layout_to_spv = comdat any

; CHECK: define weak_odr dso_local spir_kernel void @test
; CHECK-NEXT: entry:
; CHECK-NEXT: %{{.*}} = call spir_func noundef target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2) @_Z32__spirv_CooperativeMatrixLoadKHR{{.*}}(ptr addrspace(1){{.*}}, i32 noundef 0, i64 noundef{{.*}}
; CHECK-NEXT: %{{.*}} = call spir_func noundef target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2) @_Z32__spirv_CooperativeMatrixLoadKHR{{.*}}(ptr addrspace(1){{.*}}, i32 noundef 1, i64 noundef{{.*}}
; CHECK-NEXT: %{{.*}} = call spir_func noundef target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2) @_Z32__spirv_CooperativeMatrixLoadKHR{{.*}}(ptr addrspace(1){{.*}}, i32 noundef 1, i64 noundef{{.*}}
; CHECK-NEXT: %{{.*}} = call spir_func noundef target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2) @_Z32__spirv_CooperativeMatrixLoadKHR{{.*}}(ptr addrspace(1){{.*}}, i32 noundef 2, i64 noundef{{.*}}
; CHECK-NEXT: ret void

; CHECK-NOT: joint_matrix_layout_to_spv

define weak_odr dso_local spir_kernel void @test(ptr addrspace(1) %matrix.1, ptr addrspace(1) %matrix.2, i64 noundef %stride) {
entry:
  %layout.1 = alloca i32, align 4
  %layout.2 = alloca i32, align 4
  %layout.ascast.1 = addrspacecast ptr %layout.1 to ptr addrspace(4)
  %layout.ascast.2 = addrspacecast ptr %layout.2 to ptr addrspace(4)
  store i32 0, ptr addrspace(4) %layout.ascast.1, align 4
  store i32 1, ptr addrspace(4) %layout.ascast.2, align 4

  %layout.val.1 = load i32, ptr addrspace(4) %layout.ascast.1, align 4
  %layout.spv.1 = call spir_func noundef i32 @joint_matrix_layout_to_spv(i32 noundef %layout.val.1)
  %mload.1 = call spir_func noundef target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2) @_Z32__spirv_CooperativeMatrixLoadKHRIU3AS1ffLm16ELm16ELN5__spv9MatrixUseE2ELNS1_12MatrixLayoutE3ELNS1_5Scope4FlagE3EEPNS1_28__spirv_CooperativeMatrixKHRIT0_XT5_EXT1_EXT2_EXT3_EEEPT_S3_mi(ptr addrspace(1) noundef %matrix.1, i32 noundef %layout.spv.1, i64 noundef %stride, i32 noundef 0)

  %layout.val.2 = load i32, ptr addrspace(4) %layout.ascast.2, align 4
  %layout.spv.2 = call spir_func noundef i32 @joint_matrix_layout_to_spv(i32 noundef %layout.val.2)
  %mload.2 = call spir_func noundef target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2) @_Z32__spirv_CooperativeMatrixLoadKHRIU3AS1ffLm16ELm16ELN5__spv9MatrixUseE2ELNS1_12MatrixLayoutE3ELNS1_5Scope4FlagE3EEPNS1_28__spirv_CooperativeMatrixKHRIT0_XT5_EXT1_EXT2_EXT3_EEEPT_S3_mi(ptr addrspace(1) noundef %matrix.2, i32 noundef %layout.spv.2, i64 noundef %stride, i32 noundef 0)

  %layout.spv.3 = call spir_func noundef i32 @joint_matrix_layout_to_spv(i32 noundef %layout.val.2)
  %mload.3 = call spir_func noundef target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2) @_Z32__spirv_CooperativeMatrixLoadKHRIU3AS1ffLm16ELm16ELN5__spv9MatrixUseE2ELNS1_12MatrixLayoutE3ELNS1_5Scope4FlagE3EEPNS1_28__spirv_CooperativeMatrixKHRIT0_XT5_EXT1_EXT2_EXT3_EEEPT_S3_mi(ptr addrspace(1) noundef %matrix.2, i32 noundef %layout.spv.3, i64 noundef %stride, i32 noundef 0)

  store i32 2, ptr addrspace(4) %layout.ascast.2, align 4
  %layout.val.4 = load i32, ptr addrspace(4) %layout.ascast.2, align 4
  %layout.spv.4 = call spir_func noundef i32 @joint_matrix_layout_to_spv(i32 noundef %layout.val.4)
  %mload.4 = call spir_func noundef target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2) @_Z32__spirv_CooperativeMatrixLoadKHRIU3AS1ffLm16ELm16ELN5__spv9MatrixUseE2ELNS1_12MatrixLayoutE3ELNS1_5Scope4FlagE3EEPNS1_28__spirv_CooperativeMatrixKHRIT0_XT5_EXT1_EXT2_EXT3_EEEPT_S3_mi(ptr addrspace(1) noundef %matrix.2, i32 noundef %layout.spv.4, i64 noundef %stride, i32 noundef 0)
  ret void
}

declare dso_local spir_func noundef target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2) @_Z32__spirv_CooperativeMatrixLoadKHRIU3AS1ffLm16ELm16ELN5__spv9MatrixUseE2ELNS1_12MatrixLayoutE3ELNS1_5Scope4FlagE3EEPNS1_28__spirv_CooperativeMatrixKHRIT0_XT5_EXT1_EXT2_EXT3_EEEPT_S3_mi(ptr addrspace(1) noundef, i32 noundef, i64 noundef, i32 noundef)

define linkonce_odr dso_local spir_func noundef i32 @joint_matrix_layout_to_spv(i32 noundef %Layout) comdat {
entry:
  %retval = alloca i32, align 4
  %Layout.addr = alloca i32, align 4
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  %Layout.addr.ascast = addrspacecast ptr %Layout.addr to ptr addrspace(4)
  store i32 %Layout, ptr addrspace(4) %Layout.addr.ascast, align 4
  %0 = load i32, ptr addrspace(4) %Layout.addr.ascast, align 4
  switch i32 %0, label %sw.epilog [
    i32 0, label %sw.bb
    i32 1, label %sw.bb1
    i32 2, label %sw.bb2
    i32 3, label %sw.bb3
  ]

sw.bb:                                            ; preds = %entry
  store i32 0, ptr addrspace(4) %retval.ascast, align 4
  br label %return

sw.bb1:                                           ; preds = %entry
  store i32 1, ptr addrspace(4) %retval.ascast, align 4
  br label %return

sw.bb2:                                           ; preds = %entry
  store i32 2, ptr addrspace(4) %retval.ascast, align 4
  br label %return

sw.bb3:                                           ; preds = %entry
  store i32 3, ptr addrspace(4) %retval.ascast, align 4
  br label %return

sw.epilog:                                        ; preds = %entry
  call void @llvm.trap()
  unreachable

return:                                           ; preds = %sw.bb3, %sw.bb2, %sw.bb1, %sw.bb
  %1 = load i32, ptr addrspace(4) %retval.ascast, align 4
  ret i32 %1
}
