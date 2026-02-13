;; This test is derived from generic_as_variadic_no_opt.ll.
;; This test ensures that when SYCLMutatePrintfAddrspace analyzes a load
;; of the format pointer for a printf, it will not crash if a
;; corresponding store is not found.

; RUN: not opt < %s -passes=SYCLMutatePrintfAddrspace -S 2>&1 | FileCheck %s

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

$_ZN2cl4sycl3ext6oneapi12experimental6printfIcJfEEEiPKT_DpT0_ = comdat any

; CHECK: error: experimental::printf requires format string to reside in constant address space. The compiler wasn't able to automatically convert your format string into constant address space when processing builtin _Z18__spirv_ocl_printf{{.*}} called in function {{.*}}.

; Function Attrs: convergent mustprogress noinline norecurse optnone
define linkonce_odr dso_local spir_func i32 @_ZN2cl4sycl3ext6oneapi12experimental6printfIcJfEEEiPKT_DpT0_(ptr addrspace(4) %__format, float %args) #2 comdat {
entry:
  %retval = alloca i32, align 4
  %__format.addr = alloca ptr addrspace(4), align 8
  %args.addr = alloca float, align 4
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  %__format.addr.ascast = addrspacecast ptr %__format.addr to ptr addrspace(4)
  %args.addr.ascast = addrspacecast ptr %args.addr to ptr addrspace(4)
;  Remove store to ensure SYCLMutatePrintfAddrspace will not crash.
;  store ptr addrspace(4) %__format, ptr addrspace(4) %__format.addr.ascast, align 8
  store float %args, ptr addrspace(4) %args.addr.ascast, align 4  
  %0 = load ptr addrspace(4), ptr addrspace(4) %__format.addr.ascast, align 8
  %1 = load float, ptr addrspace(4) %args.addr.ascast, align 4
  %call = call spir_func i32 (ptr addrspace(4), ...) @_Z18__spirv_ocl_printfPKcz(ptr addrspace(4) %0, float %1) #8
  ret i32 %call
}

; Function Attrs: convergent
declare dso_local spir_func i32 @_Z18__spirv_ocl_printfPKcz(ptr addrspace(4), ...) #7

attributes #2 = { convergent mustprogress noinline norecurse optnone "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #7 = { convergent "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #8 = { convergent }

!llvm.module.flags = !{!0, !1}
!opencl.spir.version = !{!2}
!spirv.Source = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{i32 1, i32 2}
!3 = !{i32 4, i32 100000}
