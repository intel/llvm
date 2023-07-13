; RUN: sycl-post-link --ir-output-only --spec-const=rt %s -S -o - | FileCheck %s

; This test is intended to check that CUDASpecConstantToSymbolPass does not
; modify non-nvptx triples.

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%"class.sycl::_V1::id" = type { %"class.sycl::_V1::detail::array" }
%"class.sycl::_V1::detail::array" = type { [1 x i64] }
%struct.spec_const_struct = type <{ i32, [4 x i8], i64, i8, [7 x i8] }>

$_ZTS17spec_const_kernel = comdat any

@__usid_str2 = private unnamed_addr constant [40 x i8] c"4a70cebf3f21eeb5____ZL15value_id_struct\00", align 1
@_ZL15value_id_struct = internal addrspace(1) constant { { i32, i64, i8 } } { { i32, i64, i8 } { i32 1, i64 2, i8 3 } }, align 8

; CHECK-NOT: sycl_specialization_constants_kernel__ZTS17spec_const_kernel

; CHECK: void @_ZTS17spec_const_kernel
; CHECK: %_arg__specialization_constants_buffer
; Function Attrs: convergent noinline norecurse
define weak_odr dso_local void @_ZTS17spec_const_kernel(i64 addrspace(1)* noundef align 8 %_arg_acc, %"class.sycl::_V1::id"* noundef byval(%"class.sycl::_V1::id") align 8 %_arg_acc3, i8 addrspace(1)* noundef align 1 %_arg__specialization_constants_buffer) local_unnamed_addr comdat {
entry:
  %0 = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %_arg_acc3, i64 0, i32 0, i32 0, i64 0
  %1 = load i64, i64* %0, align 8
  %add.ptr.i = getelementptr inbounds i64, i64 addrspace(1)* %_arg_acc, i64 %1
; CHECK: addrspacecast i8 addrspace(1)* %_arg__specialization_constants_buffer to i8*
  %2 = addrspacecast i8 addrspace(1)* %_arg__specialization_constants_buffer to i8*
  %gep = getelementptr i8, i8* %2, i32 0
  %bc = bitcast i8* %gep to i64*
  %load = load i64, i64* %bc, align 8
  %gep1 = getelementptr i8, i8* %2, i32 8
  %bc2 = bitcast i8* %gep1 to i8*
  %load3 = load i8, i8* %bc2, align 1
  %gep4 = getelementptr i8, i8* %2, i32 9
  %bc5 = bitcast i8* %gep4 to %struct.spec_const_struct*
  %load6 = load %struct.spec_const_struct, %struct.spec_const_struct* %bc5, align 1
  %3 = extractvalue %struct.spec_const_struct %load6, 3
  %conv.i = sext i8 %load3 to i64
  %add.i = add nsw i64 %load, %conv.i
  %conv4.i = sext i8 %3 to i64
  %arrayidx.ascast.i.i = addrspacecast i64 addrspace(1)* %add.ptr.i to i64*
  %4 = load i64, i64* %arrayidx.ascast.i.i, align 8
  %add5.i = add i64 %add.i, %4
  %add7.i = add i64 %add5.i, %conv4.i
  store i64 %add7.i, i64* %arrayidx.ascast.i.i, align 8
  ret void
}

!nvvm.annotations = !{!0}
!opencl.spir.version = !{!1}
!spirv.Source = !{!2}
!llvm.module.flags = !{!3}
!sycl.specialization-constants = !{!4}
!sycl.specialization-constants-default-values = !{!5}
!sycl.specialization-constants-kernel = !{!6}

; CHECK: !{void (i64 addrspace(1)*, %"class.sycl::_V1::id"*, i8 addrspace(1)*)* @_ZTS17spec_const_kernel, !"kernel", i32 1}
!0 = !{void (i64 addrspace(1)*, %"class.sycl::_V1::id"*, i8 addrspace(1)*)* @_ZTS17spec_const_kernel, !"kernel", i32 1}
!1 = !{i32 1, i32 2}
!2 = !{i32 4, i32 100000}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{!"4a70cebf3f21eeb5____ZL15value_id_struct", i32 2, i32 0, i32 24}
!5 = !{{ i32, i64, i8 } { i32 1, i64 2, i8 3 }}
!6 = !{!"_ZTS17spec_const_kernel", !4}
