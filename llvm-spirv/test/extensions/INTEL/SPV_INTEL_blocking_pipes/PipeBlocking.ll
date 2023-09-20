; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_blocking_pipes,+SPV_INTEL_arbitrary_precision_integers -spirv-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; FIXME: add more negative test cases
; RUN: llvm-spirv %t.spt -to-binary -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.bc
; RUN: llvm-dis < %t.bc | FileCheck %s --check-prefix=CHECK-LLVM

; RUN: llvm-spirv -r %t.spv -o %t.bc --spirv-target-env=SPV-IR
; RUN: llvm-dis < %t.bc | FileCheck %s --check-prefix=CHECK-SPV-IR

; ModuleID = 'test/CodeGenOpenCL/pipe_builtin.cl'
source_filename = "test/CodeGenOpenCL/pipe_builtin.cl"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

%opencl.pipe_ro_t = type opaque
%opencl.pipe_wo_t = type opaque

; CHECK-SPIRV: 2 Capability BlockingPipesINTEL
; CHECK-SPIRV: 8 Extension "SPV_INTEL_blocking_pipes"
; CHECK-SPIRV: 3 TypePipe [[PipeRTy:[0-9]+]] 0
; CHECK-SPIRV: 3 TypePipe [[PipeWTy:[0-9]+]] 1
; CHECK-SPIRV: 6 Load [[PipeRTy]] [[PipeR:[0-9]+]] {{[0-9]+}} {{[0-9]+}}
; CHECK-SPIRV: 5 ReadPipeBlockingINTEL [[PipeR]] {{[0-9]+}} {{[0-9]+}} {{[0-9]+}}
; CHECK-SPIRV: 6 Load [[PipeRTy]] [[PipeR:[0-9]+]] {{[0-9]+}} {{[0-9]+}}
; CHECK-SPIRV: 5 ReadPipeBlockingINTEL [[PipeR]] {{[0-9]+}} {{[0-9]+}} {{[0-9]+}}
; CHECK-SPIRV: 6 Load [[PipeWTy]] [[PipeW:[0-9]+]] {{[0-9]+}} {{[0-9]+}}
; CHECK-SPIRV: 5 WritePipeBlockingINTEL [[PipeW]] {{[0-9]+}} {{[0-9]+}} {{[0-9]+}}
; CHECK-SPIRV: 6 Load [[PipeWTy]] [[PipeW:[0-9]+]] {{[0-9]+}} {{[0-9]+}}
; CHECK-SPIRV: 5 WritePipeBlockingINTEL [[PipeW]] {{[0-9]+}} {{[0-9]+}} {{[0-9]+}}

; CHECK-LLVM: call spir_func void @__read_pipe_2_bl(ptr addrspace(1) %0, ptr addrspace(4) %{{[0-9]+}}, i32 4, i32 4)
; CHECK-LLVM: call spir_func void @__read_pipe_2_bl(ptr addrspace(1) %0, ptr addrspace(4) %{{[0-9]+}}, i32 4, i32 4)
; CHECK-LLVM: call spir_func void @__write_pipe_2_bl(ptr addrspace(1) %0, ptr addrspace(4) %{{[0-9]+}}, i32 4, i32 4)
; CHECK-LLVM: call spir_func void @__write_pipe_2_bl(ptr addrspace(1) %0, ptr addrspace(4) %{{[0-9]+}}, i32 4, i32 4)

; CHECK-LLVM: call spir_func void @__write_pipe_2_bl(ptr addrspace(1){{.*}}, ptr addrspace(4) %{{[0-9]+}}, i32 2, i32 2)

; CHECK-SPV-IR: call spir_func void @_Z30__spirv_WritePipeBlockingINTEL{{.*}}(target("spirv.Pipe", 1){{.*}}, ptr addrspace(4){{.*}}, i32 2, i32 2)

; Function Attrs: convergent noinline nounwind optnone
define spir_func void @foo(target("spirv.Pipe", 0) %p, ptr addrspace(1) %ptr) #0 {
entry:
  %p.addr = alloca target("spirv.Pipe", 0), align 8
  %ptr.addr = alloca ptr addrspace(1), align 8
  store target("spirv.Pipe", 0) %p, target("spirv.Pipe", 0)* %p.addr, align 8
  store ptr addrspace(1) %ptr, ptr %ptr.addr, align 8
  %0 = load target("spirv.Pipe", 0), target("spirv.Pipe", 0)* %p.addr, align 8
  %1 = load ptr addrspace(1), ptr %ptr.addr, align 8
  %2 = addrspacecast ptr addrspace(1) %1 to ptr addrspace(4)
  call spir_func void @_Z29__spirv_ReadPipeBlockingINTELIiEv8ocl_pipePiii(target("spirv.Pipe", 0) %0, ptr addrspace(4) %2, i32 4, i32 4)
  ret void
}

declare dso_local spir_func void @_Z29__spirv_ReadPipeBlockingINTELIiEv8ocl_pipePiii(target("spirv.Pipe", 0), ptr addrspace(4), i32, i32)

; Function Attrs: convergent noinline nounwind optnone
define spir_func void @bar(target("spirv.Pipe", 0) %p, ptr addrspace(1) %ptr) #0 {
entry:
  %p.addr = alloca target("spirv.Pipe", 0), align 8
  %ptr.addr = alloca ptr addrspace(1), align 8
  store target("spirv.Pipe", 0) %p, target("spirv.Pipe", 0)* %p.addr, align 8
  store ptr addrspace(1) %ptr, ptr %ptr.addr, align 8
  %0 = load target("spirv.Pipe", 0), target("spirv.Pipe", 0)* %p.addr, align 8
  %1 = load ptr addrspace(1), ptr %ptr.addr, align 8
  %2 = addrspacecast ptr addrspace(1) %1 to ptr addrspace(4)
  call spir_func void @_Z29__spirv_ReadPipeBlockingINTELIiEv8ocl_pipePvii(target("spirv.Pipe", 0) %0, ptr addrspace(4) %2, i32 4, i32 4)
  ret void
}

declare dso_local spir_func void @_Z29__spirv_ReadPipeBlockingINTELIiEv8ocl_pipePvii(target("spirv.Pipe", 0), ptr addrspace(4), i32, i32)

; Function Attrs: convergent noinline nounwind optnone
define spir_func void @boo(target("spirv.Pipe", 1) %p, ptr addrspace(1) %ptr) #0 {
entry:
  %p.addr = alloca target("spirv.Pipe", 1), align 8
  %ptr.addr = alloca ptr addrspace(1), align 8
  store target("spirv.Pipe", 1) %p, target("spirv.Pipe", 1)* %p.addr, align 8
  store ptr addrspace(1) %ptr, ptr %ptr.addr, align 8
  %0 = load target("spirv.Pipe", 1), target("spirv.Pipe", 1)* %p.addr, align 8
  %1 = load ptr addrspace(1), ptr %ptr.addr, align 8
  %2 = addrspacecast ptr addrspace(1) %1 to ptr addrspace(4)
  call spir_func void @_Z30__spirv_WritePipeBlockingINTELIKiEv8ocl_pipePiii(target("spirv.Pipe", 1) %0, ptr addrspace(4) %2, i32 4, i32 4)
  ret void
}

declare dso_local spir_func void @_Z30__spirv_WritePipeBlockingINTELIKiEv8ocl_pipePiii(target("spirv.Pipe", 1), ptr addrspace(4), i32, i32)

; Function Attrs: convergent noinline nounwind optnone
define spir_func void @baz(target("spirv.Pipe", 1) %p, ptr addrspace(1) %ptr) #0 {
entry:
  %p.addr = alloca target("spirv.Pipe", 1), align 8
  %ptr.addr = alloca ptr addrspace(1), align 8
  store target("spirv.Pipe", 1) %p, target("spirv.Pipe", 1)* %p.addr, align 8
  store ptr addrspace(1) %ptr, ptr %ptr.addr, align 8
  %0 = load target("spirv.Pipe", 1), target("spirv.Pipe", 1)* %p.addr, align 8
  %1 = load ptr addrspace(1), ptr %ptr.addr, align 8
  %2 = addrspacecast ptr addrspace(1) %1 to ptr addrspace(4)
  call spir_func void @_Z30__spirv_WritePipeBlockingINTELIKiEv8ocl_pipePvii(target("spirv.Pipe", 1) %0, ptr addrspace(4) %2, i32 4, i32 4)
  ret void
}

declare dso_local spir_func void @_Z30__spirv_WritePipeBlockingINTELIKiEv8ocl_pipePvii(target("spirv.Pipe", 1), ptr addrspace(4), i32, i32)

; CHECK-LLVM: declare spir_func void @__read_pipe_2_bl(ptr addrspace(1), ptr addrspace(4), i32, i32)
; CHECK-LLVM: declare spir_func void @__write_pipe_2_bl(ptr addrspace(1), ptr addrspace(4), i32, i32)

; Function Attrs: convergent mustprogress norecurse nounwind
define linkonce_odr dso_local spir_func void @WritePipeBLockingi9Pointer(ptr addrspace(4) align 2 dereferenceable(2) %_Data) {
entry:
  %_Data.addr = alloca ptr addrspace(4), align 8
  %_WPipe = alloca target("spirv.Pipe", 1), align 8
  %_Data.addr.ascast = addrspacecast ptr %_Data.addr to ptr addrspace(4)
  %_WPipe.ascast = addrspacecast target("spirv.Pipe", 1)* %_WPipe to target("spirv.Pipe", 1) addrspace(4)*
  store ptr addrspace(4) %_Data, ptr addrspace(4) %_Data.addr.ascast, align 8
  %0 = bitcast target("spirv.Pipe", 1)* %_WPipe to ptr
  %1 = load target("spirv.Pipe", 1), target("spirv.Pipe", 1) addrspace(4)* %_WPipe.ascast, align 8
  %2 = load ptr addrspace(4), ptr addrspace(4) %_Data.addr.ascast, align 8
  call spir_func void @_Z30__spirv_WritePipeBlockingINTELIDU9_Ev8ocl_pipePKT_ii(target("spirv.Pipe", 1) %1, ptr addrspace(4) %2, i32 2, i32 2)
  ret void
}

declare dso_local spir_func void @_Z30__spirv_WritePipeBlockingINTELIDU9_Ev8ocl_pipePKT_ii(target("spirv.Pipe", 1), ptr addrspace(4), i32, i32)

attributes #0 = { convergent noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 2, i32 0}
!2 = !{!"clang version 9.0.0 (https://github.com/MrSidims/llvm.git c627b787284c5bcc917ea9742908baa1b856e176)"}
