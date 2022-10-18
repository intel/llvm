; This test checks that the post-link tool properly generates "assert used"
; property in split mode - it should include only kernels that call assertions
; in their call graph.

; RUN: sycl-post-link -split=auto -symbols -S %s -o %t.table
; RUN: FileCheck %s -input-file=%t_0.prop -check-prefix=PRESENCE-CHECK
; RUN: FileCheck %s -input-file=%t_0.prop -check-prefix=ABSENCE-CHECK

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-linux"

@_ZL2GV = internal addrspace(1) constant [1 x i32] [i32 42], align 4
@.str = private unnamed_addr addrspace(1) constant [2 x i8] c"0\00", align 1
@.str.1 = private unnamed_addr addrspace(1) constant [11 x i8] c"assert.cpp\00", align 1
@__PRETTY_FUNCTION__._Z3foov = private unnamed_addr addrspace(1) constant [11 x i8] c"void foo()\00", align 1
@__spirv_BuiltInGlobalInvocationId = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32
@__spirv_BuiltInLocalInvocationId = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32
@_ZL10assert_fmt = internal addrspace(2) constant [85 x i8] c"%s:%d: %s: global id: [%lu,%lu,%lu], local id: [%lu,%lu,%lu] Assertion `%s` failed.\0A\00", align 1

; PRESENCE-CHECK: [SYCL/assert used]

; PRESENCE-CHECK-DAG: _ZTSZ4mainE11TU0_kernel0
define dso_local spir_kernel void @_ZTSZ4mainE11TU0_kernel0() #0 {
entry:
  call spir_func void @_Z3foov()
  ret void
}

define dso_local spir_func void @_Z3foov() {
entry:
  %a = alloca i32, align 4
  %ptr = bitcast i32* %a to i32 (i32)*
  %call = call spir_func i32 %ptr(i32 1)
  %add = add nsw i32 2, %call
  store i32 %add, i32* %a, align 4
  tail call spir_func void @__assert_fail(i8 addrspace(4)* getelementptr inbounds ([2 x i8], [2 x i8] addrspace(4)* addrspacecast ([2 x i8] addrspace(1)* @.str to [2 x i8] addrspace(4)*), i64 0, i64 0), i8 addrspace(4)* getelementptr inbounds ([11 x i8], [11 x i8] addrspace(4)* addrspacecast ([11 x i8] addrspace(1)* @.str.1 to [11 x i8] addrspace(4)*), i64 0, i64 0), i32 8, i8 addrspace(4)* getelementptr inbounds ([11 x i8], [11 x i8] addrspace(4)* addrspacecast ([11 x i8] addrspace(1)* @__PRETTY_FUNCTION__._Z3foov to [11 x i8] addrspace(4)*), i64 0, i64 0))
  ret void
}

; PRESENCE-CHECK-DAG: _ZTSZ4mainE10TU1_kernel
define dso_local spir_kernel void @_ZTSZ4mainE10TU1_kernel() #1 {
entry:
  call spir_func void @_Z4foo2v()
  ret void
}

; ABSENCE-CHECK-NOT: _ZTSZ4mainE11TU0_kernel1
define dso_local spir_kernel void @_ZTSZ4mainE11TU0_kernel1() #0 {
entry:
  call spir_func void @_Z4foo1v()
  ret void
}

; Function Attrs: nounwind
define dso_local spir_func void @_Z4foo1v() {
entry:
  %a = alloca i32, align 4
  store i32 2, i32* %a, align 4
  ret void
}

; Function Attrs: nounwind
define dso_local spir_func void @_Z4foo2v() {
entry:
  %a = alloca i32, align 4
  %0 = load i32, i32 addrspace(4)* getelementptr inbounds ([1 x i32], [1 x i32] addrspace(4)* addrspacecast ([1 x i32] addrspace(1)* @_ZL2GV to [1 x i32] addrspace(4)*), i64 0, i64 0), align 4
  %add = add nsw i32 4, %0
  store i32 %add, i32* %a, align 4
  tail call spir_func void @__assert_fail(i8 addrspace(4)* getelementptr inbounds ([2 x i8], [2 x i8] addrspace(4)* addrspacecast ([2 x i8] addrspace(1)* @.str to [2 x i8] addrspace(4)*), i64 0, i64 0), i8 addrspace(4)* getelementptr inbounds ([11 x i8], [11 x i8] addrspace(4)* addrspacecast ([11 x i8] addrspace(1)* @.str.1 to [11 x i8] addrspace(4)*), i64 0, i64 0), i32 8, i8 addrspace(4)* getelementptr inbounds ([11 x i8], [11 x i8] addrspace(4)* addrspacecast ([11 x i8] addrspace(1)* @__PRETTY_FUNCTION__._Z3foov to [11 x i8] addrspace(4)*), i64 0, i64 0))
  ret void
}

; Function Attrs: convergent norecurse mustprogress
define weak dso_local spir_func void @__assert_fail(i8 addrspace(4)* %expr, i8 addrspace(4)* %file, i32 %line, i8 addrspace(4)* %func) local_unnamed_addr {
entry:
  %call = tail call spir_func i64 @_Z28__spirv_GlobalInvocationId_xv()
  %call1 = tail call spir_func i64 @_Z28__spirv_GlobalInvocationId_yv()
  %call2 = tail call spir_func i64 @_Z28__spirv_GlobalInvocationId_zv()
  %call3 = tail call spir_func i64 @_Z27__spirv_LocalInvocationId_xv()
  %call4 = tail call spir_func i64 @_Z27__spirv_LocalInvocationId_yv()
  %call5 = tail call spir_func i64 @_Z27__spirv_LocalInvocationId_zv()
  tail call spir_func void @__devicelib_assert_fail(i8 addrspace(4)* %expr, i8 addrspace(4)* %file, i32 %line, i8 addrspace(4)* %func, i64 %call, i64 %call1, i64 %call2, i64 %call3, i64 %call4, i64 %call5)
  ret void
}

; Function Attrs: inlinehint norecurse mustprogress
declare dso_local spir_func i64 @_Z28__spirv_GlobalInvocationId_xv() local_unnamed_addr

; Function Attrs: inlinehint norecurse mustprogress
declare dso_local spir_func i64 @_Z28__spirv_GlobalInvocationId_yv() local_unnamed_addr

; Function Attrs: inlinehint norecurse mustprogress
declare dso_local spir_func i64 @_Z28__spirv_GlobalInvocationId_zv() local_unnamed_addr

; Function Attrs: inlinehint norecurse mustprogress
declare dso_local spir_func i64 @_Z27__spirv_LocalInvocationId_xv() local_unnamed_addr

; Function Attrs: inlinehint norecurse mustprogress
declare dso_local spir_func i64 @_Z27__spirv_LocalInvocationId_yv() local_unnamed_addr

; Function Attrs: inlinehint norecurse mustprogress
declare dso_local spir_func i64 @_Z27__spirv_LocalInvocationId_zv() local_unnamed_addr

; Function Attrs: convergent norecurse mustprogress
define weak dso_local spir_func void @__devicelib_assert_fail(i8 addrspace(4)* %expr, i8 addrspace(4)* %file, i32 %line, i8 addrspace(4)* %func, i64 %gid0, i64 %gid1, i64 %gid2, i64 %lid0, i64 %lid1, i64 %lid2) local_unnamed_addr {
entry:
  %call = tail call spir_func i32 (i8 addrspace(2)*, ...) @_Z18__spirv_ocl_printfPU3AS2Kcz(i8 addrspace(2)* getelementptr inbounds ([85 x i8], [85 x i8] addrspace(2)* @_ZL10assert_fmt, i64 0, i64 0), i8 addrspace(4)* %file, i32 %line, i8 addrspace(4)* %func, i64 %gid0, i64 %gid1, i64 %gid2, i64 %lid0, i64 %lid1, i64 %lid2, i8 addrspace(4)* %expr)
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func i32 @_Z18__spirv_ocl_printfPU3AS2Kcz(i8 addrspace(2)*, ...) local_unnamed_addr

attributes #0 = { "sycl-module-id"="TU1.cpp" }
attributes #1 = { "sycl-module-id"="TU2.cpp" }

!opencl.spir.version = !{!0, !0}
!spirv.Source = !{!1, !1}

!0 = !{i32 1, i32 2}
!1 = !{i32 4, i32 100000}
