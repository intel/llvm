; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_usm_storage_classes -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s --check-prefixes=CHECK-SPIRV,CHECK-SPIRV-EXT
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s --check-prefixes=CHECK-SPIRV,CHECK-SPIRV-NO-EXT
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM-NO-USM

; CHECK-SPIRV-EXT: Capability USMStorageClassesINTEL
; CHECK-SPIRV-NO-EXT-NO: Capability USMStorageClassesINTEL
; CHECK-SPIRV: Name [[DEVICE:[0-9]+]] "DEVICE"
; CHECK-SPIRV: Name [[HOST:[0-9]+]] "HOST"
; CHECK-SPIRV: Name [[GLOB_ARG1:[0-9]+]] "arg_glob.addr"
; CHECK-SPIRV: Name [[DEVICE_ARG1:[0-9]+]] "arg_dev.addr"
; CHECK-SPIRV: Name [[GLOB_ARG2:[0-9]+]] "arg_glob.addr"
; CHECK-SPIRV: Name [[HOST_ARG1:[0-9]+]] "arg_host.addr"
; CHECK-SPIRV: Name [[GLOB_ARG3:[0-9]+]] "arg_glob.addr"
; CHECK-SPIRV: Name [[DEVICE_ARG2:[0-9]+]] "arg_dev.addr"
; CHECK-SPIRV: Name [[GLOB_ARG4:[0-9]+]] "arg_glob.addr"
; CHECK-SPIRV: Name [[HOST_ARG2:[0-9]+]] "arg_host.addr"
; CHECK-SPIRV-EXT: TypePointer [[DEVICE_TY:[0-9]+]] 5936 {{[0-9]+}}
; CHECK-SPIRV-EXT: TypePointer [[HOST_TY:[0-9]+]] 5937 {{[0-9]+}}
; CHECK-SPIRV-NO-EXT: TypePointer [[GLOB_TY:[0-9]+]] 5 {{[0-9]+}}
; CHECK-SPIRV-EXT: Load [[DEVICE_TY]] {{[0-9]+}} [[DEVICE]] {{[0-9]+}} {{[0-9]+}}
; CHECK-SPIRV-EXT: Load [[HOST_TY]] {{[0-9]+}} [[HOST]] {{[0-9]+}} {{[0-9]+}}
; CHECK-SPIRV-NO-EXT: Load [[GLOB_TY]] {{[0-9]+}} [[DEVICE]] {{[0-9]+}} {{[0-9]+}}
; CHECK-SPIRV-NO-EXT: Load [[GLOB_TY]] {{[0-9]+}} [[HOST]] {{[0-9]+}} {{[0-9]+}}

; ModuleID = 'intel_usm_addrspaces.cpp'
source_filename = "intel_usm_addrspaces.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%"class._ZTSZ4mainE3$_0.anon" = type { i8 }

; Function Attrs: norecurse nounwind
define spir_kernel void @_ZTSZ4mainE11fake_kernel() #0 !kernel_arg_addr_space !4 !kernel_arg_access_qual !4 !kernel_arg_type !4 !kernel_arg_base_type !4 !kernel_arg_type_qual !4 {
entry:
  %0 = alloca %"class._ZTSZ4mainE3$_0.anon", align 1
  %1 = bitcast %"class._ZTSZ4mainE3$_0.anon"* %0 to i8*
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %1) #4
  %2 = addrspacecast %"class._ZTSZ4mainE3$_0.anon"* %0 to %"class._ZTSZ4mainE3$_0.anon" addrspace(4)*
  call spir_func void @"_ZZ4mainENK3$_0clEv"(%"class._ZTSZ4mainE3$_0.anon" addrspace(4)* %2)
  %3 = bitcast %"class._ZTSZ4mainE3$_0.anon"* %0 to i8*
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %3) #4
  ret void
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: inlinehint norecurse nounwind
define internal spir_func void @"_ZZ4mainENK3$_0clEv"(%"class._ZTSZ4mainE3$_0.anon" addrspace(4)* %this) #2 align 2 {
entry:
  %this.addr = alloca %"class._ZTSZ4mainE3$_0.anon" addrspace(4)*, align 8
  store %"class._ZTSZ4mainE3$_0.anon" addrspace(4)* %this, %"class._ZTSZ4mainE3$_0.anon" addrspace(4)** %this.addr, align 8, !tbaa !5
  call spir_func void @_Z6usagesv()
  ret void
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: norecurse nounwind
define spir_func void @_Z6usagesv() #3 {
entry:
; CHECK-LLVM: %DEVICE = alloca i32 addrspace(5)*, align 8
; CHECK-LLVM-NO-USM: %DEVICE = alloca i32 addrspace(1)*, align 8
  %DEVICE = alloca i32 addrspace(5)*, align 8

; CHECK-LLVM: %HOST = alloca i32 addrspace(6)*, align 8
; CHECK-LLVM-NO-USM: %HOST = alloca i32 addrspace(1)*, align 8
  %HOST = alloca i32 addrspace(6)*, align 8

; CHECK-LLVM: bitcast i32 addrspace(5)** %DEVICE to i8*
; CHECK-LLVM-NO-USM: bitcast i32 addrspace(1)** %DEVICE to i8*
  %0 = bitcast i32 addrspace(5)** %DEVICE to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %0) #4

; CHECK-LLVM: bitcast i32 addrspace(6)** %HOST to i8*
; CHECK-LLVM-NO-USM: bitcast i32 addrspace(1)** %HOST to i8*
  %1 = bitcast i32 addrspace(6)** %HOST to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %1) #4

; CHECK-LLVM: %[[DLOAD_E:[0-9]+]] = load i32 addrspace(5)*, i32 addrspace(5)** %DEVICE, align 8
; CHECK-LLVM-NO-USM: %[[DLOAD_NE:[0-9]+]] = load i32 addrspace(1)*, i32 addrspace(1)** %DEVICE, align 8
  %2 = load i32 addrspace(5)*, i32 addrspace(5)** %DEVICE, align 8, !tbaa !5

; CHECK-LLVM: addrspacecast i32 addrspace(5)* %[[DLOAD_E]] to i32 addrspace(4)*
; CHECK-LLVM-NO-USM: addrspacecast i32 addrspace(1)* %[[DLOAD_NE]] to i32 addrspace(4)*
  %3 = addrspacecast i32 addrspace(5)* %2 to i32 addrspace(4)*
  call spir_func void @_Z3fooPi(i32 addrspace(4)* %3)

; CHECK-LLVM: %[[HLOAD_E:[0-9]+]] = load i32 addrspace(6)*, i32 addrspace(6)** %HOST, align 8
; CHECK-LLVM-NO-USM: %[[HLOAD_NE:[0-9]+]] = load i32 addrspace(1)*, i32 addrspace(1)** %HOST, align 8
  %4 = load i32 addrspace(6)*, i32 addrspace(6)** %HOST, align 8, !tbaa !5

; CHECK-LLVM: addrspacecast i32 addrspace(6)* %[[HLOAD_E]] to i32 addrspace(4)*
; CHECK-LLVM-NO-USM: addrspacecast i32 addrspace(1)* %[[HLOAD_NE]] to i32 addrspace(4)*
  %5 = addrspacecast i32 addrspace(6)* %4 to i32 addrspace(4)*
  call spir_func void @_Z3fooPi(i32 addrspace(4)* %5)

; CHECK-LLVM: bitcast i32 addrspace(6)** %HOST to i8*
; CHECK-LLVM-NO-USM: bitcast i32 addrspace(1)** %HOST to i8*
  %6 = bitcast i32 addrspace(6)** %HOST to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %6) #4

; CHECK-LLVM: bitcast i32 addrspace(5)** %DEVICE to i8*
; CHECK-LLVM-NO-USM: bitcast i32 addrspace(1)** %DEVICE to i8*
  %7 = bitcast i32 addrspace(5)** %DEVICE to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %7) #4

  ret void
}

; Function Attrs: norecurse nounwind
define spir_func void @_Z3fooPi(i32 addrspace(4)* %Data) #3 {
entry:
  %Data.addr = alloca i32 addrspace(4)*, align 8
  store i32 addrspace(4)* %Data, i32 addrspace(4)** %Data.addr, align 8, !tbaa !5
  ret void
}

; CHECK-SPIRV: Load {{[0-9]+}} [[CAST_FROM1:[0-9]+]] [[GLOB_ARG1]] {{[0-9]+}} {{[0-9]+}}
; CHECK-SPIRV-EXT: CrossWorkgroupCastToPtrINTEL {{[0-9]+}} [[CAST_TO1:[0-9]+]] [[CAST_FROM1]]
; CHECK-SPIRV-NO-EXT-NOT: GenericCastToPtr {{[0-9]+}} [[CAST_TO1:[0-9]+]] [[CAST_FROM1]]
; CHECL-SPIRV: Store [[DEVICE_ARG1]] [[CAST_TO1]] {{[0-9]+}} {{[0-9]+}}
; Function Attrs: norecurse nounwind
define spir_func void @_Z3booPii(i32 addrspace(1)* %arg_glob, i32 addrspace(5)* %arg_dev) #3 !kernel_arg_addr_space !9 {
entry:
  %arg_glob.addr = alloca i32 addrspace(1)*, align 4
  %arg_dev.addr = alloca i32 addrspace(5)*, align 4
  store i32 addrspace(1)* %arg_glob, i32 addrspace(1)** %arg_glob.addr, align 4
  store i32 addrspace(5)* %arg_dev, i32 addrspace(5)** %arg_dev.addr, align 4
  %0 = load i32 addrspace(1)*, i32 addrspace(1)** %arg_glob.addr, align 4
; CHECK-LLVM: addrspacecast i32 addrspace(1)* %{{[0-9]+}} to i32 addrspace(5)*
; CHECK-LLVM-NO-USM-NOT: addrspacecast i32 addrspace(1)* %{{[0-9]+}} to i32 addrspace(5)*
  %1 = addrspacecast i32 addrspace(1)* %0 to i32 addrspace(5)*
  store i32 addrspace(5)* %1, i32 addrspace(5)** %arg_dev.addr, align 4
  ret void
}

; CHECK-SPIRV: Load {{[0-9]+}} [[CAST_FROM2:[0-9]+]] [[GLOB_ARG2]] {{[0-9]+}} {{[0-9]+}}
; CHECK-SPIRV-EXT: CrossWorkgroupCastToPtrINTEL {{[0-9]+}} [[CAST_TO2:[0-9]+]] [[CAST_FROM2]]
; CHECK-SPIRV-NO-EXT-NOT: GenericCastToPtr {{[0-9]+}} [[CAST_TO2:[0-9]+]] [[CAST_FROM2]]
; CHECL-SPIRV: Store [[HOST_ARG1]] [[CAST_TO2]] {{[0-9]+}} {{[0-9]+}}
; Function Attrs: norecurse nounwind
define spir_func void @_Z3gooPii(i32 addrspace(1)* %arg_glob, i32 addrspace(6)* %arg_host) #3 !kernel_arg_addr_space !10 {
entry:
  %arg_glob.addr = alloca i32 addrspace(1)*, align 4
  %arg_host.addr = alloca i32 addrspace(6)*, align 4
  store i32 addrspace(1)* %arg_glob, i32 addrspace(1)** %arg_glob.addr, align 4
  store i32 addrspace(6)* %arg_host, i32 addrspace(6)** %arg_host.addr, align 4
  %0 = load i32 addrspace(1)*, i32 addrspace(1)** %arg_glob.addr, align 4
; CHECK-LLVM: addrspacecast i32 addrspace(1)* %{{[0-9]+}} to i32 addrspace(6)*
; CHECK-LLVM-NO-USM-NOT: addrspacecast i32 addrspace(1)* %{{[0-9]+}} to i32 addrspace(6)*
  %1 = addrspacecast i32 addrspace(1)* %0 to i32 addrspace(6)*
  store i32 addrspace(6)* %1, i32 addrspace(6)** %arg_host.addr, align 4
  ret void
}

; CHECK-SPIRV: Load {{[0-9]+}} [[CAST_FROM3:[0-9]+]] [[DEVICE_ARG2]] {{[0-9]+}} {{[0-9]+}}
; CHECK-SPIRV-EXT: PtrCastToCrossWorkgroupINTEL {{[0-9]+}} [[CAST_TO3:[0-9]+]] [[CAST_FROM3]]
; CHECK-SPIRV-NO-EXT-NOT: PtrCastToGeneric {{[0-9]+}} [[CAST_TO3:[0-9]+]] [[CAST_FROM3]]
; CHECL-SPIRV: Store [[GLOB_ARG3]] [[CAST_TO3]] {{[0-9]+}} {{[0-9]+}}
; Function Attrs: norecurse nounwind
define spir_func void @_Z3zooPii(i32 addrspace(1)* %arg_glob, i32 addrspace(5)* %arg_dev) #3 !kernel_arg_addr_space !9 {
entry:
  %arg_glob.addr = alloca i32 addrspace(1)*, align 4
  %arg_dev.addr = alloca i32 addrspace(5)*, align 4
  store i32 addrspace(1)* %arg_glob, i32 addrspace(1)** %arg_glob.addr, align 4
  store i32 addrspace(5)* %arg_dev, i32 addrspace(5)** %arg_dev.addr, align 4
  %0 = load i32 addrspace(5)*, i32 addrspace(5)** %arg_dev.addr, align 4
; CHECK-LLVM: addrspacecast i32 addrspace(5)* %{{[0-9]+}} to i32 addrspace(1)*
; CHECK-LLVM-NO-USM-NOT: addrspacecast i32 addrspace(5)* %{{[0-9]+}} to i32 addrspace(1)*
  %1 = addrspacecast i32 addrspace(5)* %0 to i32 addrspace(1)*
  store i32 addrspace(1)* %1, i32 addrspace(1)** %arg_glob.addr, align 4
  ret void
}

; CHECK-SPIRV: Load {{[0-9]+}} [[CAST_FROM4:[0-9]+]] [[HOST_ARG2]] {{[0-9]+}} {{[0-9]+}}
; CHECK-SPIRV-EXT: PtrCastToCrossWorkgroupINTEL {{[0-9]+}} [[CAST_TO4:[0-9]+]] [[CAST_FROM4]]
; CHECK-SPIRV-NO-EXT-NOT: PtrCastToGeneric {{[0-9]+}} [[CAST_TO4:[0-9]+]] [[CAST_FROM4]]
; CHECL-SPIRV: Store [[GLOB_ARG4]] [[CAST_TO4]] {{[0-9]+}} {{[0-9]+}}
; Function Attrs: norecurse nounwind
define spir_func void @_Z3mooPii(i32 addrspace(1)* %arg_glob, i32 addrspace(6)* %arg_host) #3 !kernel_arg_addr_space !10 {
entry:
  %arg_glob.addr = alloca i32 addrspace(1)*, align 4
  %arg_host.addr = alloca i32 addrspace(6)*, align 4
  store i32 addrspace(1)* %arg_glob, i32 addrspace(1)** %arg_glob.addr, align 4
  store i32 addrspace(6)* %arg_host, i32 addrspace(6)** %arg_host.addr, align 4
  %0 = load i32 addrspace(6)*, i32 addrspace(6)** %arg_host.addr, align 4
; CHECK-LLVM: addrspacecast i32 addrspace(6)* %{{[0-9]+}} to i32 addrspace(1)*
; CHECK-LLVM-NO-USM-NOT: addrspacecast i32 addrspace(6)* %{{[0-9]+}} to i32 addrspace(1)*
  %1 = addrspacecast i32 addrspace(6)* %0 to i32 addrspace(1)*
  store i32 addrspace(1)* %1, i32 addrspace(1)** %arg_glob.addr, align 4
  ret void
}

attributes #0 = { norecurse nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "sycl-module-id"="intel_usm_addrspaces.cpp" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind willreturn }
attributes #2 = { inlinehint norecurse nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { norecurse nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0}
!opencl.spir.version = !{!1}
!spirv.Source = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{i32 4, i32 100000}
!3 = !{!"clang version 11.0.0"}
!4 = !{}
!5 = !{!6, !6, i64 0}
!6 = !{!"any pointer", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C++ TBAA"}
!9 = !{i32 1, i32 5}
!10 = !{i32 1, i32 6}
