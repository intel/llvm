; RUN: sycl-post-link -spec-const=native < %s -o %t.files.table
; RUN: FileCheck %s -input-file=%t.files_0.prop
; RUN: %if asserts %{ sycl-post-link -debug-only=SpecConst -spec-const=native < %s 2>&1 | FileCheck %s --check-prefix=CHECK-LOG %}
;
; This test checks that composite specialization constants with implicit padding
; at the end of the composite type will have an additional padding descriptor at
; the end of the descriptor list. 

; ModuleID = 'test.bc'
source_filename = "llvm-link"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%"class.cl::sycl::specialization_id" = type { %struct.TestStruct }
%struct.TestStruct = type { i32, i8 }

$_ZTSZZ4mainENKUlRN2cl4sycl7handlerEE_clES2_E10KernelName = comdat any

@__usid_str = private unnamed_addr constant [33 x i8] c"fb86570d411366d1____ZL9SpecConst\00", align 1
@_ZL9SpecConst = internal addrspace(1) constant %"class.cl::sycl::specialization_id" zeroinitializer, align 4

; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @_ZTSZZ4mainENKUlRN2cl4sycl7handlerEE_clES2_E10KernelName() local_unnamed_addr #0 comdat !kernel_arg_buffer_location !5 !sycl_kernel_omit_args !6 {
entry:
  %tmp.i = alloca %struct.TestStruct, align 4
  %tmp.ascast.i = addrspacecast %struct.TestStruct* %tmp.i to %struct.TestStruct addrspace(4)*
  %0 = bitcast %struct.TestStruct* %tmp.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %0) #3
  call spir_func void @_Z40__sycl_getComposite2020SpecConstantValueI10TestStructET_PKcPKvS5_(%struct.TestStruct addrspace(4)* sret(%struct.TestStruct) align 4 %tmp.ascast.i, i8 addrspace(4)* addrspacecast (i8* getelementptr inbounds ([33 x i8], [33 x i8]* @__usid_str, i64 0, i64 0) to i8 addrspace(4)*), i8 addrspace(4)* addrspacecast (i8 addrspace(1)* bitcast (%"class.cl::sycl::specialization_id" addrspace(1)* @_ZL9SpecConst to i8 addrspace(1)*) to i8 addrspace(4)*), i8 addrspace(4)* null) #4
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %0) #3
  ret void
}

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: convergent
declare dso_local spir_func void @_Z40__sycl_getComposite2020SpecConstantValueI10TestStructET_PKcPKvS5_(%struct.TestStruct addrspace(4)* sret(%struct.TestStruct) align 4, i8 addrspace(4)*, i8 addrspace(4)*, i8 addrspace(4)*) local_unnamed_addr #2

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

attributes #0 = { convergent norecurse "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="test.cpp" "uniform-work-group-size"="true" }
attributes #1 = { argmemonly nofree nosync nounwind willreturn }
attributes #2 = { convergent "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #3 = { nounwind }
attributes #4 = { convergent }

!opencl.spir.version = !{!0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0}
!spirv.Source = !{!1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1}
!llvm.ident = !{!2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2}
!llvm.module.flags = !{!3, !4}

!0 = !{i32 1, i32 2}
!1 = !{i32 4, i32 100000}
!2 = !{!"clang version 14.0.0"}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{i32 7, !"frame-pointer", i32 2}
!5 = !{i32 -1}
!6 = !{i1 true}

; We expect the following in the descriptor list for SpecConst:
; First 8 bytes are the size of the list.
; Each 12 bytes after the size comprise a descriptor consisting of 3 32-bit
; unsigned integers. For SpecConst these are:
;            ID             |     Composite offset    |          Size
; 0x00 0x00 0x00 0x00 (0)   | 0x00 0x00 0x00 0x00 (0) | 0x04 0x00 0x00 0x00 (4)
; 0x01 0x00 0x00 0x00 (1)   | 0x04 0x00 0x00 0x00 (4) | 0x01 0x00 0x00 0x00 (1)
; 0xff 0xff 0xff 0xff (max) | 0x05 0x00 0x00 0x00 (5) | 0x03 0x00 0x00 0x00 (3)
; Most important for this test is the last descriptor which represents 3-bytes
; implicit padding at the end of the composite type of the spec constant.
;
; CHECK: [SYCL/specialization constants]
; CHECK-NEXT: fb86570d411366d1____ZL9SpecConst=2
; CHECK-LOG: sycl.specialization-constants
; CHECK-LOG:[[UNIQUE_PREFIX:[0-9a-zA-Z]+]]={0, 0, 4}
; CHECK-LOG:[[UNIQUE_PREFIX]]={1, 4, 1}
; CHECK-LOG:[[UNIQUE_PREFIX]]={4294967295, 5, 3}
; CHECK-LOG: sycl.specialization-constants-default-values
; CHECK-LOG:{0, 8, 0}
