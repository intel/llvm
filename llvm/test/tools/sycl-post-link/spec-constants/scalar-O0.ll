; RUN: sycl-post-link --ir-output-only -spec-const=default %s -S -o - | \
; RUN:   FileCheck %s -check-prefixes=CHECK,CHECK-DEF
; RUN: sycl-post-link --ir-output-only -spec-const=rt %s -S -o - | \
; RUN:   FileCheck %s -check-prefixes=CHECK,CHECK-RT

; This test checks that the post link tool is able to correctly transform
; specialization constant intrinsics in an unoptimized device code compiled

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-linux-sycldevice"

%"UserSpecConstIDType" = type { i8 }

$FOO = comdat any

@__unique_stable_name.FOO = private unnamed_addr constant [18 x i8] c"_ZTS11MyBoolConst\00", align 1

declare dso_local spir_func zeroext i1 @_Z33__sycl_getScalarSpecConstantValueIbET_PKc(i8 addrspace(4)*) #7

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: norecurse
define spir_func zeroext i1 @FOO(%"UserSpecConstIDType" addrspace(4)* %0) comdat align 2 {
  %2 = alloca %"UserSpecConstIDType" addrspace(4)*, align 8
  %3 = alloca i8 addrspace(4)*, align 8
  store %"UserSpecConstIDType" addrspace(4)* %0, %"UserSpecConstIDType" addrspace(4)** %2, align 8, !tbaa !8
  %4 = load %"UserSpecConstIDType" addrspace(4)*, %"UserSpecConstIDType" addrspace(4)** %2, align 8
  %5 = bitcast i8 addrspace(4)** %3 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %5) #8
  store i8 addrspace(4)* addrspacecast (i8* getelementptr inbounds ([18 x i8], [18 x i8]* @__unique_stable_name.FOO, i64 0, i64 0) to i8 addrspace(4)*), i8 addrspace(4)** %3, align 8, !tbaa !8
; CHECK-NOT: store{{.*}}@__unique_stable_name.FOO
  %6 = load i8 addrspace(4)*, i8 addrspace(4)** %3, align 8, !tbaa !8
  %7 = call spir_func zeroext i1 @_Z33__sycl_getScalarSpecConstantValueIbET_PKc(i8 addrspace(4)* %6)
; with -spec-const=rt the __sycl_getSpecConstantValue is replaced with
;  SPIRV intrinsic
; CHECK-RT: %{{[0-9]+}} = call i1 @_Z20__spirv_SpecConstantib(i32 0, i1 false)
  %8 = bitcast i8 addrspace(4)** %3 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %8) #8
  ret i1 %7
; with -spec-const=default the __sycl_getSpecConstantValue is replaced with
;  default C++ value:
; CHECK-DEF: ret i1 false
}

attributes #1 = { argmemonly nounwind willreturn }
attributes #7 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #8 = { nounwind }

!llvm.module.flags = !{!0}
!opencl.spir.version = !{!1}
!spirv.Source = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{i32 4, i32 100000}
!3 = !{!"clang version 11.0.0 (https://github.com/kbobrovs/llvm.git c9a794cf3cc06924bc0777f5facb507a98fad0a0)"}
!8 = !{!9, !9, i64 0}
!9 = !{!"any pointer", !10, i64 0}
!10 = !{!"omnipotent char", !11, i64 0}
!11 = !{!"Simple C++ TBAA"}

; CHECK-RT: !sycl.specialization-constants = !{![[#MD:]]}
; CHECK-RT: ![[#MD]] = !{!"_ZTS11MyBoolConst", i32 0, i32 0, i32 1}
