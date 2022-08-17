; This test checks "integrated" invoke_simd support by the sycl-post-link tool:
; - data flow analysis is able to determine actual target of
;   _Z33__regcall3____builtin_invoke_simd*
; - functions shared by SYCL and ESIMD callgraphs get cloned and renamed, thus
;   making sure no functions are shared by the callgraphs (currently required by
;   IGC)

; RUN: sycl-post-link -lower-esimd -symbols -split=auto -S %s -o %t.table
; RUN: FileCheck %s -input-file=%t.table
; RUN: FileCheck %s -input-file=%t_0.ll --check-prefixes CHECK-IR
; RUN: FileCheck %s -input-file=%t_0.sym --check-prefixes CHECK-SYM

;------------- Verify generated table file.
; CHECK: [Code|Properties|Symbols]
; CHECK: {{.*}}_0.ll|{{.*}}_0.prop|{{.*}}_0.sym
; CHECK-EMPTY:

;------------- Verify generated symbol file.
; CHECK-SYM: SYCL_kernel
; CHECK-SYM: _SIMD_CALLEE
; CHECK-SYM: ESIMD_kernel
; CHECK-SYM-EMPTY:

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; Function Attrs: convergent

declare dso_local spir_func <4 x float> @__intrin(i64)

define dso_local spir_func <4 x float> @block_read(i64 %addr) noinline {
;------------- This function is participates in both SYCL and ESIMD callgraphs -
;------------- check that it is duplicated:
; CHECK-IR-DAG: define dso_local spir_func <4 x float> @block_read.esimd
  %res = call spir_func <4 x float> @__intrin(i64 %addr)
  ret <4 x float> %res
}

; Function Attrs: convergent
declare dso_local spir_func float @_Z33__regcall3____builtin_invoke_simdXX(<4 x float> (float addrspace(4)*, <4 x float>, i32)*, float addrspace(4)*, float, i32) local_unnamed_addr

;------------- This is also an entry point, because of the "sycl-module-id" attribute #0.
; Function Attrs: convergent mustprogress noinline norecurse optnone
define dso_local x86_regcallcc <4 x float> @_SIMD_CALLEE(float addrspace(4)* %A, <4 x float> %non_uni_val, i32 %uni_val) #0 !sycl_explicit_simd !0 !intel_reqd_sub_group_size !0 {
; Verify that correct attributes are attached to the function:
; CHECK-IR-DAG: {{.*}} @_SIMD_CALLEE(float addrspace(4)* %A, <4 x float> %non_uni_val, i32 %uni_val) #[[ATTR1:[0-9]+]]
entry:
  %AA = ptrtoint float addrspace(4)* %A to i64
  %ii = zext i32 %uni_val to i64
  %addr = add nuw nsw i64 %ii, %AA
  %data = call spir_func <4 x float> @block_read(i64 %addr)
  %add = fadd <4 x float> %non_uni_val, %data
  ret <4 x float> %add
}

define internal spir_func float @foo(float addrspace(1)* %ptr) align 2 {
entry:
;------------- Typical data flow of the @_SIMD_CALLEE function address in worst
;------------- case (-O0), when invoke_simd uses function name:
;------------- float res = invoke_simd(sg, SIMD_CALLEE, uniform{ A }, x, uniform{ y });
  %f.addr.i = alloca <4 x float> (float addrspace(4)*, <4 x float>, i32)*, align 8
  %f.addr.ascast.i = addrspacecast <4 x float> (float addrspace(4)*, <4 x float>, i32)** %f.addr.i to <4 x float> (float addrspace(4)*, <4 x float>, i32)* addrspace(4)*
  store <4 x float> (float addrspace(4)*, <4 x float>, i32)* @_SIMD_CALLEE, <4 x float> (float addrspace(4)*, <4 x float>, i32)* addrspace(4)* %f.addr.ascast.i, align 8
  %FUNC_PTR = load <4 x float> (float addrspace(4)*, <4 x float>, i32)*, <4 x float> (float addrspace(4)*, <4 x float>, i32)* addrspace(4)* %f.addr.ascast.i, align 8

;------------- Data flow for the parameters of SIMD_CALLEE
  %param_A = addrspacecast float addrspace(1)* %ptr to float addrspace(4)*
  %param_non_uni_val = load float, float addrspace(4)* %param_A, align 4

;------------- The invoke_simd calls.
  %res1 = call spir_func float @_Z33__regcall3____builtin_invoke_simdXX(<4 x float> (float addrspace(4)*, <4 x float>, i32)* %FUNC_PTR, float addrspace(4)* %param_A, float %param_non_uni_val, i32 10)
; Verify that %FUNC_PTR is replaced with @_SIMD_CALLEE:
;  CHECK-IR-DAG: %{{.*}} = call spir_func float @_Z33__regcall3____builtin_invoke_simdXX(<4 x float> (float addrspace(4)*, <4 x float>, i32)* @_SIMD_CALLEE, float addrspace(4)* %param_A, float %param_non_uni_val, i32 10)
  ret float %res1
}

;------------- SYCL kernel, an entry point
define dso_local spir_kernel void @SYCL_kernel(float addrspace(1)* %ptr) #2 {
entry:
  %res1 = call spir_func float @foo(float addrspace(1)* %ptr)
  %ptri = ptrtoint float addrspace(1)* %ptr to i64
  %vec = call spir_func <4 x float> @block_read(i64 %ptri)
  %res2 = extractelement <4 x float> %vec, i32 0
  %res = fadd float %res1, %res2
  store float %res, float addrspace(1)* %ptr
  ret void
}

;------------- ESIMD kernel, an entry point
define dso_local spir_kernel void @ESIMD_kernel(float addrspace(1)* %ptr) #3 !sycl_explicit_simd !0 {
entry:
  %ptr_as4 = addrspacecast float addrspace(1)* %ptr to float addrspace(4)*
  %res = call x86_regcallcc <4 x float> @_SIMD_CALLEE(float addrspace(4)* %ptr_as4, <4 x float> <float 1.0, float 1.0, float 1.0, float 1.0>, i32 0)
  %ptr_x16 = bitcast float addrspace(4)* %ptr_as4 to <4 x float> addrspace(4)*
  store <4 x float> %res, <4 x float> addrspace(4)* %ptr_x16
  ret void
}


; Check that VCStackCall attribute is added to the invoke_simd target functions:
attributes #0 = { convergent mustprogress norecurse "sycl-module-id"="invoke_simd.cpp" }
; CHECK-IR-DAG: attributes #[[ATTR1]] = { {{.*}} "VCStackCall" {{.*}}"sycl-module-id"="invoke_simd.cpp" }

attributes #2 = { "sycl-module-id"="a.cpp" }
attributes #3 = { "sycl-module-id"="b.cpp" }


!0 = !{}
