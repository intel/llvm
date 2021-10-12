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
  %2 = bitcast %array* %1  to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* %2) #2
  %3 = addrspacecast %array* %1  to %array addrspace(4)*
  %4 = getelementptr inbounds %array, %array addrspace(4)* %3, i32 0, i32 0
  %5 = bitcast [3 x i64] addrspace(4)* %4 to i8 addrspace(4)*
; CHECK: %[[V_ARR:[0-9]+]] = bitcast [3 x i64] addrspace(4)* %{{[0-9]+}} to i8 addrspace(4)*
  %6 = bitcast [3 x i64] addrspace(4)* @constinit to i8 addrspace(4)*
  call void @llvm.memcpy.p4i8.p4i8.i64(i8 addrspace(4)* align 8 %5, i8 addrspace(4)* align 8 %6, i64 24, i1 false), !tbaa.struct !7
; CHECK: call void @llvm.memset.p4i8.i64(i8 addrspace(4)* align 8 %[[V_ARR]], i8 0, i64 24, i1 false)
  %7 = bitcast %array* %1  to i8*
  call void @llvm.lifetime.end.p0i8(i64 24, i8* %7) #2
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p4i8.p4i8.i64(i8 addrspace(4)* nocapture writeonly, i8 addrspace(4)* nocapture readonly, i64, i1 immarg) #1

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
