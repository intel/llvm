; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.txt
; RUN: FileCheck < %t.txt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV-NOT: llvm.memmove

; CHECK-SPIRV-DAG: TypeInt [[#TYPEINT:]] 32
; CHECK-SPIRV-DAG: TypeInt [[#I8:]] 8
; CHECK-SPIRV-DAG: Constant [[#TYPEINT]] [[#C128:]] 128
; CHECK-SPIRV-DAG: Constant [[#TYPEINT]] [[#C68:]] 68
; CHECK-SPIRV-DAG: Constant [[#TYPEINT]] [[#C72:]] 72
; CHECK-SPIRV-DAG: Constant [[#TYPEINT]] [[#C32:]] 32
; CHECK-SPIRV-DAG: TypeStruct [[#TYPESTRUCTWRONG:]] [[#]] [[#]] [[#]] [[#]]
; CHECK-SPIRV-DAG: TypeStruct [[#TYPESTRUCT:]] [[#]] [[#]] [[#]]
; CHECK-SPIRV-DAG: TypePointer [[#STRUCTGLOBAL_PTR:]] 5 [[#TYPESTRUCT]]
; CHECK-SPIRV-DAG: TypePointer [[#I8GLOBAL_PTR:]] 5 [[#I8]]
; CHECK-SPIRV-DAG: TypePointer [[#I8PRIVATE_PTR:]] 7 [[#I8]]
; CHECK-SPIRV-DAG: TypePointer [[#STRUCTGENERIC_PTR:]] 8 [[#TYPESTRUCT]]
; CHECK-SPIRV-DAG: TypePointer [[#I8GENERIC_PTR:]] 8 [[#I8]]

; CHECK-SPIRV-LABEL: [[#]] Function [[#]]
; CHECK-SPIRV: FunctionParameter [[#STRUCTGLOBAL_PTR]] [[#ARG_IN:]]
; CHECK-SPIRV: FunctionParameter [[#STRUCTGLOBAL_PTR]] [[#ARG_OUT:]]
;
; CHECK-SPIRV: Bitcast [[#I8GLOBAL_PTR]] [[#I8_ARG_IN:]] [[#ARG_IN]]
; CHECK-SPIRV: Bitcast [[#I8GLOBAL_PTR]] [[#I8_ARG_OUT:]] [[#ARG_OUT]]
; CHECK-SPIRV: Variable [[#]] [[#MEM:]]
; CHECK-SPIRV: Bitcast [[#]] [[#TMP:]] [[#MEM]]
; CHECK-SPIRV: LifetimeStart [[#TMP]]
; CHECK-SPIRV: Bitcast [[#]] [[#TMP:]] [[#MEM]]
; CHECK-SPIRV: CopyMemorySized [[#TMP]] [[#I8_ARG_IN]] [[#C128]] 2 64
; CHECK-SPIRV: Bitcast [[#]] [[#TMP:]] [[#MEM]]
; CHECK-SPIRV: CopyMemorySized [[#I8_ARG_OUT]] [[#TMP]] [[#C128]] 2 64
; CHECK-SPIRV: Bitcast [[#]] [[#TMP:]] [[#MEM]]
; CHECK-SPIRV: LifetimeStop [[#TMP]]

; CHECK-SPIRV-LABEL: [[#]] Function [[#]]
; CHECK-SPIRV: FunctionParameter [[#STRUCTGLOBAL_PTR]] [[#ARG_IN:]]
; CHECK-SPIRV: FunctionParameter [[#STRUCTGENERIC_PTR]] [[#ARG_OUT:]]
;
; CHECK-SPIRV: Bitcast [[#I8GLOBAL_PTR]] [[#I8_ARG_IN:]] [[#ARG_IN]]
; CHECK-SPIRV: Bitcast [[#I8GENERIC_PTR]] [[#I8_ARG_OUT_GENERIC:]] [[#ARG_OUT]]
; CHECK-SPIRV: GenericCastToPtr [[#I8GLOBAL_PTR]] [[#I8_ARG_OUT:]] [[#I8_ARG_OUT_GENERIC]]
; CHECK-SPIRV: Variable [[#]] [[#MEM:]]
; CHECK-SPIRV: Bitcast [[#]] [[#TMP:]] [[#MEM]]
; CHECK-SPIRV: LifetimeStart [[#TMP]]
; CHECK-SPIRV: Bitcast [[#]] [[#TMP:]] [[#MEM]]
; CHECK-SPIRV: CopyMemorySized [[#TMP]] [[#I8_ARG_IN]] [[#C68]] 2 64
; CHECK-SPIRV: Bitcast [[#]] [[#TMP:]] [[#MEM]]
; CHECK-SPIRV: CopyMemorySized [[#I8_ARG_OUT]] [[#TMP]] [[#C68]] 2 64
; CHECK-SPIRV: Bitcast [[#]] [[#TMP:]] [[#MEM]]
; CHECK-SPIRV: LifetimeStop [[#TMP]]

; CHECK-SPIRV-LABEL: [[#]] Function [[#]]
; CHECK-SPIRV: FunctionParameter [[#I8GLOBAL_PTR]] [[#ARG_IN:]]
; CHECK-SPIRV: FunctionParameter [[#I8GLOBAL_PTR]] [[#ARG_OUT:]]
;
; CHECK-SPIRV: Variable [[#]] [[#MEM:]]
; CHECK-SPIRV: Bitcast [[#]] [[#TMP:]] [[#MEM]]
; CHECK-SPIRV: LifetimeStart [[#TMP]]
; CHECK-SPIRV: Bitcast [[#]] [[#TMP:]] [[#MEM]]
; CHECK-SPIRV: CopyMemorySized [[#TMP]] [[#ARG_IN]] [[#C72]] 0
; CHECK-SPIRV: Bitcast [[#]] [[#TMP:]] [[#MEM]]
; CHECK-SPIRV: CopyMemorySized [[#ARG_OUT]] [[#TMP]] [[#C72]] 0
; CHECK-SPIRV: Bitcast [[#]] [[#TMP:]] [[#MEM]]
; CHECK-SPIRV: LifetimeStop [[#TMP]]

; xCHECK-SPIRV-LABEL: [[#]] Function [[#]]
;
; xCHECK-SPIRV: Label
; xCHECK-SPIRV: Label
; xCHECK-SPIRV: Label
; xCHECK-SPIRV: Label
; xCHECK-SPIRV: Variable [[#]] [[#MEM:]]
; xCHECK-SPIRV: Bitcast [[#]] [[#TMP:]] [[#MEM]]
; xCHECK-SPIRV: LifetimeStart [[#TMP]]
; xCHECK-SPIRV: Bitcast [[#]] [[#TMP:]] [[#MEM]]
; xCHECK-SPIRV: CopyMemorySized [[#TMP]] [[#]] [[#C32]] 2 8
; xCHECK-SPIRV: Bitcast [[#]] [[#TMP:]] [[#MEM]]
; xCHECK-SPIRV: CopyMemorySized [[#]] [[#TMP]] [[#C32]] 2 8
; xCHECK-SPIRV: Bitcast [[#]] [[#TMP:]] [[#MEM]]
; xCHECK-SPIRV: LifetimeStop [[#TMP]]

; CHECK-LLVM-NOT: llvm.memmove

; CHECK-LLVM-LABEL: @test_full_move
; CHECK-LLVM: %[[i8_in:.*]] = bitcast %struct.SomeStruct addrspace(1)* %in to i8 addrspace(1)*
; CHECK-LLVM: %[[i8_out:.*]] = bitcast %struct.SomeStruct addrspace(1)* %out to i8 addrspace(1)*
; CHECK-LLVM: %[[local:.*]] = alloca [128 x i8]
; CHECK-LLVM: %[[i8_local:.*]] = bitcast [128 x i8]* %[[local]] to i8*
; CHECK-LLVM: call void @llvm.lifetime.start.p0i8({{.*}}, i8* %[[i8_local]])
; CHECK-LLVM: %[[i8_local:.*]] = bitcast [128 x i8]* %[[local]] to i8*
; CHECK-LLVM: call void @llvm.memcpy.p0i8.p1i8.i32(i8* align 64 %[[i8_local]],
; CHECK-LLVM-SAME: i8 addrspace(1)* align 64 %[[i8_in]], i32 128, i1 false)
; CHECK-LLVM: %[[i8_local:.*]] = bitcast [128 x i8]* %[[local]] to i8*
; CHECK-LLVM: call void @llvm.memcpy.p1i8.p0i8.i32(i8 addrspace(1)* align 64 %[[i8_out]],
; CHECK-LLVM-SAME: i8* align 64 %[[i8_local]], i32 128, i1 false)
; CHECK-LLVM: %[[i8_local:.*]] = bitcast [128 x i8]* %[[local]] to i8*
; CHECK-LLVM: call void @llvm.lifetime.end.p0i8({{.*}}, i8* %[[i8_local]])

; CHECK-LLVM-LABEL: @test_partial_move
; CHECK-LLVM: %[[i8_in:.*]] = bitcast %struct.SomeStruct addrspace(1)* %in to i8 addrspace(1)*
; CHECK-LLVM: %[[i8_out_generic:.*]] = bitcast %struct.SomeStruct addrspace(4)* %out to i8 addrspace(4)*
; CHECK-LLVM: %[[i8_out:.*]] = addrspacecast i8 addrspace(4)* %[[i8_out_generic]] to i8 addrspace(1)*
; CHECK-LLVM: %[[local:.*]] = alloca [68 x i8]
; CHECK-LLVM: %[[i8_local:.*]] = bitcast [68 x i8]* %[[local]] to i8*
; CHECK-LLVM: call void @llvm.lifetime.start.p0i8({{.*}}, i8* %[[i8_local]])
; CHECK-LLVM: %[[i8_local:.*]] = bitcast [68 x i8]* %[[local]] to i8*
; CHECK-LLVM: call void @llvm.memcpy.p0i8.p1i8.i32(i8* align 64 %[[i8_local]],
; CHECK-LLVM-SAME: i8 addrspace(1)* align 64 %[[i8_in]], i32 68, i1 false)
; CHECK-LLVM: %[[i8_local:.*]] = bitcast [68 x i8]* %[[local]] to i8*
; CHECK-LLVM: call void @llvm.memcpy.p1i8.p0i8.i32(i8 addrspace(1)* align 64 %[[i8_out]],
; CHECK-LLVM-SAME: i8* align 64 %[[i8_local]], i32 68, i1 false)
; CHECK-LLVM: %[[i8_local:.*]] = bitcast [68 x i8]* %[[local]] to i8*
; CHECK-LLVM: call void @llvm.lifetime.end.p0i8({{.*}}, i8* %[[i8_local]])

; CHECK-LLVM-LABEL: @test_array
; CHECK-LLVM: %[[#ALLOCA:]] = alloca [72 x i8]
; CHECK-LLVM: %[[#TMP0:]] = bitcast [72 x i8]* %[[#ALLOCA]] to i8*
; CHECK-LLVM: call void @llvm.lifetime.start.p0i8({{.*}}, i8* %[[#TMP0]])
; CHECK-LLVM: %[[#TMP1:]] = bitcast [72 x i8]* %[[#ALLOCA]] to i8*
; CHECK-LLVM: call void @llvm.memcpy.p0i8.p1i8.i32(i8* %[[#TMP1]], i8 addrspace(1)* %in, i32 72, i1 false)
; CHECK-LLVM: %[[#TMP2:]] = bitcast [72 x i8]* %[[#ALLOCA]] to i8*
; CHECK-LLVM: call void @llvm.memcpy.p1i8.p0i8.i32(i8 addrspace(1)* %out, i8* %[[#TMP2]], i32 72, i1 false)
; CHECK-LLVM: %[[#TMP3:]] = bitcast [72 x i8]* %[[#ALLOCA]] to i8*
; CHECK-LLVM: call void @llvm.lifetime.end.p0i8({{.*}}, i8* %[[#TMP3]])

; CHECK-LLVM-LABEL: @test_phi
; CHECK-LLVM: %[[#ALLOCA:]] = alloca [32 x i8]
; CHECK-LLVM: %[[#TMP0:]] = bitcast [32 x i8]* %[[#ALLOCA]] to i8*
; CHECK-LLVM: call void @llvm.lifetime.start.p0i8({{.*}}, i8* %[[#TMP0]])
; CHECK-LLVM: %[[#TMP1:]] = bitcast [32 x i8]* %[[#ALLOCA]] to i8*
; CHECK-LLVM: call void @llvm.memcpy.p0i8.p4i8.i64(i8* align 8 %[[#TMP1]], i8 addrspace(4)* align 8 %phi, i64 32, i1 false)
; CHECK-LLVM: %[[#TMP2:]] = bitcast [32 x i8]* %[[#ALLOCA]] to i8*
; CHECK-LLVM: call void @llvm.memcpy.p4i8.p0i8.i64(i8 addrspace(4)* align 8 %[[#]], i8* align 8 %[[#TMP2]], i64 32, i1 false)
; CHECK-LLVM: %[[#TMP3:]] = bitcast [32 x i8]* %[[#ALLOCA]] to i8*
; CHECK-LLVM: call void @llvm.lifetime.end.p0i8({{.*}}, i8* %[[#TMP3]])


target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir-unknown-unknown"

%struct.SomeStruct = type { <16 x float>, i32, [60 x i8] }
%class.kfunc = type <{ i32, i32, i32, [4 x i8] }>

@InvocIndex = external local_unnamed_addr addrspace(1) constant i64, align 8
@"func_object1" = internal addrspace(3) global %class.kfunc zeroinitializer, align 8

; Function Attrs: nounwind
define spir_kernel void @test_full_move(%struct.SomeStruct addrspace(1)* nocapture readonly %in, %struct.SomeStruct addrspace(1)* nocapture %out) #0 !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !4 !kernel_arg_type_qual !5 {
  %1 = bitcast %struct.SomeStruct addrspace(1)* %in to i8 addrspace(1)*
  %2 = bitcast %struct.SomeStruct addrspace(1)* %out to i8 addrspace(1)*
  call void @llvm.memmove.p1i8.p1i8.i32(i8 addrspace(1)* align 64 %2, i8 addrspace(1)* align 64 %1, i32 128, i1 false)
  ret void
}

define spir_kernel void @test_partial_move(%struct.SomeStruct addrspace(1)* nocapture readonly %in, %struct.SomeStruct addrspace(4)* nocapture %out) {
  %1 = bitcast %struct.SomeStruct addrspace(1)* %in to i8 addrspace(1)*
  %2 = bitcast %struct.SomeStruct addrspace(4)* %out to i8 addrspace(4)*
  %3 = addrspacecast i8 addrspace(4)* %2 to i8 addrspace(1)*
  call void @llvm.memmove.p1i8.p1i8.i32(i8 addrspace(1)* align 64 %3, i8 addrspace(1)* align 64 %1, i32 68, i1 false)
  ret void
}

; Function Attrs: nounwind
define spir_kernel void @test_array(i8 addrspace(1)* %in, i8 addrspace(1)* %out) {
  call void @llvm.memmove.p1i8.p1i8.i32(i8 addrspace(1)* %out, i8 addrspace(1)* %in, i32 72, i1 false)
  ret void
}

; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @test_phi() local_unnamed_addr {
entry:
  %0 = alloca i32, align 8
  %1 = addrspacecast i32* %0 to i32 addrspace(4)*
  %2 = load i64, i64 addrspace(1)* @InvocIndex, align 8
  %cmp = icmp eq i64 %2, 0
  br i1 %cmp, label %leader, label %entry.merge_crit_edge

entry.merge_crit_edge:                            ; preds = %entry
  %3 = bitcast i32 addrspace(4)* %1 to i8 addrspace(4)*
  br label %merge

leader:                                           ; preds = %entry
  %4 = bitcast i32 addrspace(4)* %1 to i8 addrspace(4)*
  br label %merge

merge:                                            ; preds = %entry.merge_crit_edge, %leader
  %phi = phi i8 addrspace(4)* [ %3, %entry.merge_crit_edge ], [ %4, %leader ]
  %5 = addrspacecast i8 addrspace(3)* bitcast (%class.kfunc addrspace(3)* @"func_object1" to i8 addrspace(3)*) to i8 addrspace(4)*
  call void @llvm.memmove.p4i8.p4i8.i64(i8 addrspace(4)* align 8 dereferenceable(32) %5, i8 addrspace(4)* align 8 dereferenceable(32) %phi, i64 32, i1 false)
  ret void
}

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.memmove.p4i8.p4i8.i64(i8 addrspace(4)* nocapture writeonly, i8 addrspace(4)* nocapture readonly, i64, i1 immarg)

; Function Attrs: nounwind
declare void @llvm.memmove.p1i8.p1i8.i32(i8 addrspace(1)* nocapture, i8 addrspace(1)* nocapture readonly, i32, i1) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no_infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!7}
!opencl.ocl.version = !{!7}
!opencl.used.extensions = !{!8}
!opencl.used.optional.core.features = !{!8}
!opencl.compiler.options = !{!8}

!1 = !{i32 1, i32 1}
!2 = !{!"none", !"none"}
!3 = !{!"struct SomeStruct*", !"struct SomeStruct*"}
!4 = !{!"struct SomeStruct*", !"struct SomeStruct*"}
!5 = !{!"const", !""}
!7 = !{i32 1, i32 2}
!8 = !{}
