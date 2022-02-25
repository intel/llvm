; RUN: opt -LowerESIMD -S < %s | FileCheck %s

; This test checks that LowerESIMD pass correctly interpretes the
; 'kernel_arg_accessor_ptr' metadata. Particularly, that it generates additional
; vector of per-argument metadata (accessible from "genx.kernels" top-level
; metadata node):
; - for those arguments having non-zero in the corresponding
;   'kernel_arg_accessor_ptr' position:
;   * "argument kind" metadata element is set to '2' - 'surface'
;   * "argument descriptor" metadata element  is set to 'buffer_t'
; - for those pointer arguments having '0' in the corresponding
;   'kernel_arg_accessor_ptr' position, the kind/descriptor is set to
;   '0'/'svmptr_t'

define weak_odr dso_local spir_kernel void @ESIMDKernel(i32 %_arg_, float addrspace(1)* %_arg_1, float addrspace(1)* %_arg_3, i32 %_arg_5, float addrspace(1)* %_arg_7) !kernel_arg_accessor_ptr !0 !sycl_explicit_simd !1 !intel_reqd_sub_group_size !2 {
; CHECK: {{.*}} spir_kernel void @ESIMDKernel({{.*}}) #[[GENX_MAIN:[0-9]+]]
  ret void
}

; kernel_arg_accessor_ptr:
; arg0=<scalar>
; arg1=<ptr from accessor>
; arg2=<ptr from accessor>
; arg3=<scalar>
; arg4=<ptr>
; buffer_t and argument kind 2 (surface) metadata must be added for args 1 and 2
!0 = !{i32 0, i32 1, i32 1, i32 0, i32 0}
!1 = !{}
!2 = !{i32 1}

; CHECK: attributes #[[GENX_MAIN]] = { "CMGenxMain" "oclrt"="1" }
; CHECK: !genx.kernels = !{![[GENX_KERNELS:[0-9]+]]}
; CHECK: ![[GENX_KERNELS]] = !{void (i32, float addrspace(1)*, float addrspace(1)*, i32, float addrspace(1)*)* @ESIMDKernel, !"ESIMDKernel", ![[ARG_KINDS:[0-9]+]], i32 0, i32 0, ![[ARG_IO_KINDS:[0-9]+]], ![[ARG_DESCS:[0-9]+]], i32 0, i32 0}
; CHECK: ![[ARG_KINDS]] = !{i32 0, i32 2, i32 2, i32 0, i32 0}
; CHECK: ![[ARG_IO_KINDS]] = !{i32 0, i32 0, i32 0, i32 0, i32 0}
; CHECK: ![[ARG_DESCS]] = !{!"", !"buffer_t", !"buffer_t", !"", !"svmptr_t"}

