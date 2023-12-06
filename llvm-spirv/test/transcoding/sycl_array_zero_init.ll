; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck %s

; ModuleID = 'sycl_array_zero_init.cpp'
source_filename = "sycl_array_zero_init.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-linux"

%array = type { [3 x i64] }

$main = comdat any

@constinit = private addrspace(4) global [3 x i64] zeroinitializer, align 8

define weak_odr dso_local spir_kernel void @main() #0 comdat !kernel_arg_addr_space !4 !kernel_arg_access_qual !4 !kernel_arg_type !4 !kernel_arg_base_type !4 !kernel_arg_type_qual !4 {
  %1 = alloca %array, align 8
  call void @llvm.lifetime.start.p0(i64 24, ptr %1) #2
  %2 = addrspacecast ptr %1  to ptr addrspace(4)
  call void @llvm.memcpy.p4.p4.i64(ptr addrspace(4) align 8 %2, ptr addrspace(4) align 8 @constinit, i64 24, i1 false), !tbaa.struct !7
; CHECK: call void @llvm.memcpy.p4.p4.i64(ptr addrspace(4) align 8 %{{[0-9]+}}, ptr addrspace(4) align 8 @constinit, i64 24, i1 false)
  call void @llvm.lifetime.end.p0(i64 24, ptr %1) #2
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p4.p4.i64(ptr addrspace(4) nocapture writeonly, ptr addrspace(4) nocapture readonly, i64, i1 immarg) #1

attributes #0 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { nounwind }

!llvm.module.flags = !{!0}
!opencl.spir.version = !{!1}
!spirv.Source = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{i32 4, i32 100000}
!3 = !{!"clang version 9.0.0"}
!4 = !{}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}
!7 = !{i64 0, i64 24, !8}
!8 = !{!5, !5, i64 0}
