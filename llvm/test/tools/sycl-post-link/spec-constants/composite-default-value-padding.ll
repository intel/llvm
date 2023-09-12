; RUN: sycl-post-link -spec-const=emulation < %s -o %t.files.table
; RUN: FileCheck %s -input-file=%t.files_0.prop
;
; This test checks that composite specialization constants with padding gets the
; correct padding in their default values to prevent values being inserted at
; incorrect offsets.

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64_x86_64-unknown-unknown"

%"class.cl::sycl::specialization_id.7" = type { i8 }
%"class.cl::sycl::range" = type { %"class.cl::sycl::detail::array" }
%"class.cl::sycl::detail::array" = type { [1 x i64] }
%struct.TestStruct = type <{ i32, i8, [3 x i8] }>

$_ZTSZZ4mainENKUlRN2cl4sycl7handlerEE_clES2_EUlNS0_14kernel_handlerEE_ = comdat any

@__usid_str = private unnamed_addr constant [37 x i8] c"9d329ad59055e972____ZL12StructSpecId\00", align 1
@_ZL12StructSpecId = internal addrspace(1) constant { { i32, i8 } } { { i32, i8 } { i32 20, i8 99 } }, align 4
@__usid_str.1 = private unnamed_addr constant [35 x i8] c"9d329ad59055e972____ZL10BoolSpecId\00", align 1
@_ZL10BoolSpecId = internal addrspace(1) constant %"class.cl::sycl::specialization_id.7" { i8 1 }, align 1

; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @_ZTSZZ4mainENKUlRN2cl4sycl7handlerEE_clES2_EUlNS0_14kernel_handlerEE_(%struct.TestStruct addrspace(1)* %_arg_, %"class.cl::sycl::range"* byval(%"class.cl::sycl::range") align 8 %_arg_1, %"class.cl::sycl::range"* byval(%"class.cl::sycl::range") align 8 %_arg_2, %"class.cl::sycl::range"* byval(%"class.cl::sycl::range") align 8 %_arg_3, i8 addrspace(1)* %_arg_4, %"class.cl::sycl::range"* byval(%"class.cl::sycl::range") align 8 %_arg_6, %"class.cl::sycl::range"* byval(%"class.cl::sycl::range") align 8 %_arg_7, %"class.cl::sycl::range"* byval(%"class.cl::sycl::range") align 8 %_arg_8, i8 addrspace(1)* %_arg__specialization_constants_buffer) local_unnamed_addr #0 comdat !kernel_arg_buffer_location !5 {
entry:
  %ref.tmp.i = alloca %struct.TestStruct, align 4
  %0 = getelementptr inbounds %"class.cl::sycl::range", %"class.cl::sycl::range"* %_arg_3, i64 0, i32 0, i32 0, i64 0
  %1 = addrspacecast i64* %0 to i64 addrspace(4)*
  %2 = load i64, i64 addrspace(4)* %1, align 8
  %add.ptr.i = getelementptr inbounds %struct.TestStruct, %struct.TestStruct addrspace(1)* %_arg_, i64 %2
  %3 = getelementptr inbounds %"class.cl::sycl::range", %"class.cl::sycl::range"* %_arg_8, i64 0, i32 0, i32 0, i64 0
  %4 = addrspacecast i64* %3 to i64 addrspace(4)*
  %5 = load i64, i64 addrspace(4)* %4, align 8
  %add.ptr.i33 = getelementptr inbounds i8, i8 addrspace(1)* %_arg_4, i64 %5
  %6 = addrspacecast i8 addrspace(1)* %_arg__specialization_constants_buffer to i8 addrspace(4)*
  %ref.tmp.ascast.i = addrspacecast %struct.TestStruct* %ref.tmp.i to %struct.TestStruct addrspace(4)*
  %7 = bitcast %struct.TestStruct* %ref.tmp.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %7) #5
  call spir_func void @_Z40__sycl_getComposite2020SpecConstantValueI10TestStructET_PKcPKvS5_(%struct.TestStruct addrspace(4)* sret(%struct.TestStruct) align 4 %ref.tmp.ascast.i, i8 addrspace(4)* addrspacecast (i8* getelementptr inbounds ([37 x i8], [37 x i8]* @__usid_str, i64 0, i64 0) to i8 addrspace(4)*), i8 addrspace(4)* addrspacecast (i8 addrspace(1)* bitcast ({ { i32, i8 } } addrspace(1)* @_ZL12StructSpecId to i8 addrspace(1)*) to i8 addrspace(4)*), i8 addrspace(4)* %6) #4
  %8 = bitcast %struct.TestStruct addrspace(1)* %add.ptr.i to i8 addrspace(1)*
  %9 = addrspacecast i8 addrspace(1)* %8 to i8 addrspace(4)*
  %10 = addrspacecast i8* %7 to i8 addrspace(4)*
  call void @llvm.memcpy.p4i8.p4i8.i64(i8 addrspace(4)* noundef align 4 dereferenceable(5) %9, i8 addrspace(4)* noundef align 4 dereferenceable(5) %10, i64 5, i1 false), !tbaa.struct !6
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %7) #5
  %call.i.i.i = call spir_func zeroext i1 @_Z37__sycl_getScalar2020SpecConstantValueIbET_PKcPKvS4_(i8 addrspace(4)* addrspacecast (i8* getelementptr inbounds ([35 x i8], [35 x i8]* @__usid_str.1, i64 0, i64 0) to i8 addrspace(4)*), i8 addrspace(4)* addrspacecast (i8 addrspace(1)* getelementptr inbounds (%"class.cl::sycl::specialization_id.7", %"class.cl::sycl::specialization_id.7" addrspace(1)* @_ZL10BoolSpecId, i64 0, i32 0) to i8 addrspace(4)*), i8 addrspace(4)* %6) #4
  %arrayidx.ascast.i.i = addrspacecast i8 addrspace(1)* %add.ptr.i33 to i8 addrspace(4)*
  %frombool.i = zext i1 %call.i.i.i to i8
  store i8 %frombool.i, i8 addrspace(4)* %arrayidx.ascast.i.i, align 1, !tbaa !12
  ret void
}

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: convergent
declare dso_local spir_func void @_Z40__sycl_getComposite2020SpecConstantValueI10TestStructET_PKcPKvS5_(%struct.TestStruct addrspace(4)* sret(%struct.TestStruct) align 4, i8 addrspace(4)*, i8 addrspace(4)*, i8 addrspace(4)*) local_unnamed_addr #2

; Function Attrs: argmemonly nofree nounwind willreturn
declare void @llvm.memcpy.p4i8.p4i8.i64(i8 addrspace(4)* noalias nocapture writeonly, i8 addrspace(4)* noalias nocapture readonly, i64, i1 immarg) #3

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: convergent
declare dso_local spir_func zeroext i1 @_Z37__sycl_getScalar2020SpecConstantValueIbET_PKcPKvS4_(i8 addrspace(4)*, i8 addrspace(4)*, i8 addrspace(4)*) local_unnamed_addr #2

attributes #0 = { convergent norecurse "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="test.cpp" "uniform-work-group-size"="true" }
attributes #1 = { argmemonly nofree nosync nounwind willreturn }
attributes #2 = { convergent "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #3 = { argmemonly nofree nounwind willreturn }
attributes #4 = { convergent }
attributes #5 = { nounwind }

!opencl.spir.version = !{!0}
!spirv.Source = !{!1}
!llvm.ident = !{!2}
!llvm.module.flags = !{!3, !4}

!0 = !{i32 1, i32 2}
!1 = !{i32 4, i32 100000}
!2 = !{!"clang version 14.0.0 (https://github.com/intel/llvm)"}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{i32 7, !"frame-pointer", i32 2}
!5 = !{i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1}
!6 = !{i64 0, i64 4, !7, i64 4, i64 1, !11}
!7 = !{!8, !8, i64 0}
!8 = !{!"int", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C++ TBAA"}
!11 = !{!9, !9, i64 0}
!12 = !{!13, !13, i64 0}
!13 = !{!"bool", !9, i64 0}

; Make sure the specialization constants occur in the order with the padded
; struct first followed by the boolean specialization constant.
; Most important information from the corresponding encoded data is the size of
; the specialization constants, i.e. 8 and 1 bytes respectively.
; CHECK: [SYCL/specialization constants]
; CHECK-NEXT: 9d329ad59055e972____ZL12StructSpecId=2|gBAAAAAAAAAAAAAAAAAAAgAAAAA
; CHECK-NEXT: 9d329ad59055e972____ZL10BoolSpecId=2|gBAAAAAAAAQAAAAAAAAAAEAAAAA

; Ensure that the default values are correct.
; IBAAAAAAAAAFAAAAjBAAAEA is decoded to "0x48 0x0 0x0 0x0 0x0 0x0 0x0 0x0 0x14
; 0x0 0x0 0x0 0x63 0x0 0x0 0x0 0x1" which consists of:
;  1. 8 bytes denoting the bit-size of the byte array, here 72 bits or 9 bytes.
;  2. 4 bytes with the default value of the 32-bit integer member of
;     %struct.TestStruct. Its value being 20.
;  3. 1 byte with the default value of the char member of %struct.TestStruct.
;     Its value being 'c'.
;  4. 3 bytes of padding for %struct.TestStruct.
;  5. 1 byte with the default value of the boolean specialization constant. Its
;     value being true.
; CHECK: [SYCL/specialization constants default values]
; CHECK-NEXT: all=2|IBAAAAAAAAAFAAAAjBAAAEA
