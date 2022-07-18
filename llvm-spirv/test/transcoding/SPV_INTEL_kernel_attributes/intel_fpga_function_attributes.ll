;; Can be compiled using https://github.com/intel/llvm SYCL compiler from:
;; class Foo {
;; public:
;;   [[intel::no_global_work_offset,
;;     intel::max_global_work_dim(1),
;;     intel::max_work_group_size(1,1,1),
;;     intel::num_simd_work_items(8),
;;     intel::stall_enable,
;;     intel::scheduler_target_fmax_mhz(1000),
;;     intel::loop_fuse_independent(3),
;;     intel::initiation_interval(10),
;;     intel::max_concurrency(12),
;;     intel::disable_loop_pipelining]] void operator()() {}
;; };
;;
;; template <typename name, typename Func>
;; __attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
;;   kernelFunc();
;; }
;;
;; void bar() {
;;   Foo boo;
;;   kernel<class kernel_name>(boo);
;;   kernel<class kernel_name2>([]() [[intel::no_global_work_offset(0)]]{});
;; }

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_kernel_attributes,+SPV_INTEL_fpga_cluster_attributes,+SPV_INTEL_loop_fuse,+SPV_INTEL_fpga_invocation_pipelining_attributes -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; RUN: llvm-spirv -spirv-text -r %t.spt -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: Capability KernelAttributesINTEL
; CHECK-SPIRV: Capability FPGAKernelAttributesINTEL
; CHECK-SPIRV: Capability FPGAClusterAttributesINTEL
; CHECK-SPIRV: Capability LoopFuseINTEL
; CHECK-SPIRV: Capability FPGAInvocationPipeliningAttributesINTEL
; CHECK-SPIRV: Extension "SPV_INTEL_fpga_cluster_attributes"
; CHECK-SPIRV: Extension "SPV_INTEL_fpga_invocation_pipelining_attributes"
; CHECK-SPIRV: Extension "SPV_INTEL_loop_fuse"
; CHECK-SPIRV: EntryPoint {{.*}} [[FUNCENTRY2:[0-9]+]] "_ZTSZ3barvE11kernel_name3"
; CHECK-SPIRV: ExecutionMode [[FUNCENTRY:[0-9]+]] 5893 1 1 1
; CHECK-SPIRV: ExecutionMode [[FUNCENTRY]] 5894 1
; CHECK-SPIRV: ExecutionMode [[FUNCENTRY]] 5895
; CHECK-SPIRV: ExecutionMode [[FUNCENTRY]] 5896 8
; CHECK-SPIRV: ExecutionMode [[FUNCENTRY]] 5903 1000
; CHECK-SPIRV-DAG: Decorate [[FUNCENTRY]] StallEnableINTEL
; CHECK-SPIRV-DAG: Decorate [[FUNCENTRY]] FuseLoopsInFunctionINTEL 3 1
; CHECK-SPIRV-DAG: Decorate [[FUNCENTRY]] InitiationIntervalINTEL 10
; CHECK-SPIRV-DAG: Decorate [[FUNCENTRY]] MaxConcurrencyINTEL 12
; CHECK-SPIRV-DAG: Decorate [[FUNCENTRY]] PipelineEnableINTEL 0
; CHECK-SPIRV: Decorate [[FUNCENTRY2]] PipelineEnableINTEL 1
; CHECK-SPIRV: Function {{.*}} [[FUNCENTRY]] {{.*}}
; CHECK-SPIRV: Function {{.*}} [[FUNCENTRY2]] {{.*}}

; CHECK-LLVM: define spir_kernel void {{.*}}kernel_name()
; CHECK-LLVM-SAME: !stall_enable ![[ONEMD:[0-9]+]] !loop_fuse ![[FUSE:[0-9]+]]
; CHECK-LLVM-SAME: !initiation_interval ![[II:[0-9]+]]
; CHECK-LLVM-SAME: !max_concurrency ![[MAXCON:[0-9]+]]
; CHECK-LLVM-SAME: !disable_loop_pipelining ![[ONEMD]]
; CHECK-LLVM-SAME: !max_work_group_size ![[MAXWG:[0-9]+]]
; CHECK-LLVM-SAME: !no_global_work_offset ![[OFFSET:[0-9]+]]
; CHECK-LLVM-SAME: !max_global_work_dim ![[ONEMD]] !num_simd_work_items ![[NUMSIMD:[0-9]+]]
; CHECK-LLVM-SAME: !scheduler_target_fmax_mhz ![[MAXMHZ:[0-9]+]]
; CHECK-LLVM-NOT: define spir_kernel void {{.*}}kernel_name2 {{.*}} !no_global_work_offset {{.*}}
; CHECK-LLVM: define spir_kernel void {{.*}}kernel_name3()
; CHECK-LLVM-SAME: !disable_loop_pipelining ![[ONEMD2:[0-9]+]]
; CHECK-LLVM: ![[OFFSET]] = !{}
; CHECK-LLVM: ![[ONEMD]] = !{i32 1}
; CHECK-LLVM: ![[FUSE]] = !{i32 3, i32 1}
; CHECK-LLVM: ![[II]] = !{i32 10}
; CHECK-LLVM: ![[MAXCON]] = !{i32 12}
; CHECK-LLVM: ![[MAXWG]] = !{i32 1, i32 1, i32 1}
; CHECK-LLVM: ![[NUMSIMD]] = !{i32 8}
; CHECK-LLVM: ![[MAXMHZ]] = !{i32 1000}
; CHECK-LLVM: ![[ONEMD2]] = !{i32 0}

; ModuleID = 'kernel-attrs.cpp'
source_filename = "kernel-attrs.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-linux"

%class._ZTS3Foo.Foo = type { i8 }
%"class._ZTSZ3barvE3$_0.anon" = type { i8 }

$_ZN3FooclEv = comdat any

; Function Attrs: nounwind
define spir_kernel void @_ZTSZ3barvE11kernel_name() #0 !kernel_arg_addr_space !4 !kernel_arg_access_qual !4 !kernel_arg_type !4 !kernel_arg_base_type !4 !kernel_arg_type_qual !4 !num_simd_work_items !5 !max_work_group_size !6 !max_global_work_dim !7 !no_global_work_offset !4 !stall_enable !7 !scheduler_target_fmax_mhz !12 !loop_fuse !13 !initiation_interval !14 !max_concurrency !15 !disable_loop_pipelining !7 {
entry:
  %Foo = alloca %class._ZTS3Foo.Foo, align 1
  %0 = bitcast %class._ZTS3Foo.Foo* %Foo to i8*
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %0) #4
  %1 = addrspacecast %class._ZTS3Foo.Foo* %Foo to %class._ZTS3Foo.Foo addrspace(4)*
  call spir_func void @_ZN3FooclEv(%class._ZTS3Foo.Foo addrspace(4)* %1)
  %2 = bitcast %class._ZTS3Foo.Foo* %Foo to i8*
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %2) #4
  ret void
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: nounwind
define linkonce_odr spir_func void @_ZN3FooclEv(%class._ZTS3Foo.Foo addrspace(4)* %this) #2 comdat align 2 {
entry:
  %this.addr = alloca %class._ZTS3Foo.Foo addrspace(4)*, align 8
  store %class._ZTS3Foo.Foo addrspace(4)* %this, %class._ZTS3Foo.Foo addrspace(4)** %this.addr, align 8, !tbaa !8
  %this1 = load %class._ZTS3Foo.Foo addrspace(4)*, %class._ZTS3Foo.Foo addrspace(4)** %this.addr, align 8
  ret void
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1
; Function Attrs: nounwind
define spir_kernel void @_ZTSZ3barvE12kernel_name2() #0 !kernel_arg_addr_space !4 !kernel_arg_access_qual !4 !kernel_arg_type !4 !kernel_arg_base_type !4 !kernel_arg_type_qual !4 {
entry:
  %0 = alloca %"class._ZTSZ3barvE3$_0.anon", align 1
  %1 = bitcast %"class._ZTSZ3barvE3$_0.anon"* %0 to i8*
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %1) #4
  %2 = addrspacecast %"class._ZTSZ3barvE3$_0.anon"* %0 to %"class._ZTSZ3barvE3$_0.anon" addrspace(4)*
  call spir_func void @"_ZZ3barvENK3$_0clEv"(%"class._ZTSZ3barvE3$_0.anon" addrspace(4)* %2)
  %3 = bitcast %"class._ZTSZ3barvE3$_0.anon"* %0 to i8*
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %3) #4
  ret void
}
; Function Attrs: inlinehint nounwind
define internal spir_func void @"_ZZ3barvENK3$_0clEv"(%"class._ZTSZ3barvE3$_0.anon" addrspace(4)* %this) #3 align 2 {
entry:
  %this.addr = alloca %"class._ZTSZ3barvE3$_0.anon" addrspace(4)*, align 8
  store %"class._ZTSZ3barvE3$_0.anon" addrspace(4)* %this, %"class._ZTSZ3barvE3$_0.anon" addrspace(4)** %this.addr, align 8, !tbaa !8
  %this1 = load %"class._ZTSZ3barvE3$_0.anon" addrspace(4)*, %"class._ZTSZ3barvE3$_0.anon" addrspace(4)** %this.addr, align 8
  ret void
}

; Function Attrs: nounwind
define spir_kernel void @_ZTSZ3barvE11kernel_name3() #0 !disable_loop_pipelining !16 {
entry:
  %Foo = alloca %class._ZTS3Foo.Foo, align 1
  %0 = bitcast %class._ZTS3Foo.Foo* %Foo to i8*
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %0) #4
  %1 = addrspacecast %class._ZTS3Foo.Foo* %Foo to %class._ZTS3Foo.Foo addrspace(4)*
  call spir_func void @_ZN3FooclEv(%class._ZTS3Foo.Foo addrspace(4)* %1)
  %2 = bitcast %class._ZTS3Foo.Foo* %Foo to i8*
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %2) #4
  ret void
}

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "sycl-module-id"="kernel-attrs.cpp" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind willreturn }
attributes #2 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { inlinehint nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
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
!5 = !{i32 8}
!6 = !{i32 1, i32 1, i32 1}
!7 = !{i32 1}
!8 = !{!9, !9, i64 0}
!9 = !{!"any pointer", !10, i64 0}
!10 = !{!"omnipotent char", !11, i64 0}
!11 = !{!"Simple C++ TBAA"}
!12 = !{i32 1000}
!13 = !{i32 3, i32 1}
!14 = !{i32 10}
!15 = !{i32 12}
!16 = !{i32 0}
