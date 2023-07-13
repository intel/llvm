; RUN: sycl-post-link --ir-output-only --spec-const=rt %s -S -o - | FileCheck %s

; This test is intended to check that CUDASpecConstantToSymbolPass correctly
; handles situations where _arg__specialization_constants_buffer is present,
; however SpecConstants pass has not identified any uses of spec constants, and
; hence both the implicit argument and the allocation should be present.

target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

%"class.sycl::_V1::id" = type { %"class.sycl::_V1::detail::array" }
%"class.sycl::_V1::detail::array" = type { [1 x i64] }
%struct.spec_const_struct = type <{ i32, [4 x i8], i64, i8, [7 x i8] }>

$_ZTS17spec_const_kernel = comdat any

@__usid_str2 = private unnamed_addr constant [40 x i8] c"4a70cebf3f21eeb5____ZL15value_id_struct\00", align 1
@_ZL15value_id_struct = internal addrspace(1) constant { { i32, i64, i8 } } { { i32, i64, i8 } { i32 1, i64 2, i8 3 } }, align 8

; CHECK: sycl_specialization_constants_kernel__ZTS17spec_const_kernel

; CHECK: void @_ZTS17spec_const_kernel
; CHECK: %_arg__specialization_constants_buffer
; Function Attrs: convergent noinline norecurse
define weak_odr dso_local void @_ZTS17spec_const_kernel(i64 addrspace(1)* noundef align 8 %_arg_acc, %"class.sycl::_V1::id"* noundef byval(%"class.sycl::_V1::id") align 8 %_arg_acc3, i8 addrspace(1)* noundef align 1 %_arg__specialization_constants_buffer) local_unnamed_addr comdat {
entry:
  ret void
}

!nvvm.annotations = !{!0}
!opencl.spir.version = !{!1}
!spirv.Source = !{!2}
!llvm.module.flags = !{!3}
!sycl.specialization-constants = !{!4}
!sycl.specialization-constants-default-values = !{!5}

; CHECK: !{void (i64 addrspace(1)*, %"class.sycl::_V1::id"*, i8 addrspace(1)*)* @_ZTS17spec_const_kernel, !"kernel", i32 1}
!0 = !{void (i64 addrspace(1)*, %"class.sycl::_V1::id"*, i8 addrspace(1)*)* @_ZTS17spec_const_kernel, !"kernel", i32 1}
!1 = !{i32 1, i32 2}
!2 = !{i32 4, i32 100000}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{!"4a70cebf3f21eeb5____ZL15value_id_struct", i32 2, i32 0, i32 24}
!5 = !{{ i32, i64, i8 } { i32 1, i64 2, i8 3 }}
!6 = !{!"_ZTS17spec_const_kernel", !4}
