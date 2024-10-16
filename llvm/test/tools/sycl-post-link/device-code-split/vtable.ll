; This test ensures that sycl-post-link properly handles cases when one global
; object used in a kernel, is being initialized with another global object.
;
; To make the example more realistic, this IR comes from the following SYCL
; snippet:
;
; class Base {
; public:
;   virtual int display() { return 1; }
; };
; 
; class Derived1 : public Base {
; public:
;   int display() { return 2; }
; };
; 
; class Derived2 : public Base {
; public:
;   int display() { return 3; }
; };
; 
; int main() {
;   sycl::queue Q;
; 
;   auto *Storage =
;       sycl::malloc_device<char>(sizeof(Derived1) + sizeof(Derived2), Q);
;   auto *Ptrs = sycl::malloc_device<Base *>(2, Q);
; 
;   Q.single_task([=] {
;      Ptrs[0] = new (&Storage[0]) Derived1;
;      Ptrs[1] = new (&Storage[sizeof(Derived1)]) Derived2;
;    }).wait();
; }
;
; Compiled with clang++ -fsycl -fsycl-device-only -O2
;     -Xclang -fsycl-allow-virtual-functions -fno-sycl-instrument-device-code
;
; The aim of the test is to check that 'display' method referenced from global
; variables storing vtable, are also included into the final module, even though
; they are not directly used in a kernel otherwise.
;
; RUN: sycl-post-link -properties -split=auto -S < %s -o %t.table
; RUN: FileCheck %s -input-file=%t_0.ll
;
; RUN: sycl-module-split -split=auto -S < %s -o %t2
; RUN: FileCheck %s -input-file=%t2_0.ll
;
; CHECK-DAG: @_ZTV8Derived1 = {{.*}} @_ZN8Derived17displayEv
; CHECK-DAG: @_ZTV8Derived2 = {{.*}} @_ZN8Derived27displayEv
;
; CHECK-DAG: define {{.*}} i32 @_ZN8Derived17displayEv
; CHECK-DAG: define {{.*}} i32 @_ZN8Derived27displayEv

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%class.Base = type { i32 (...)** }
%class.Derived1 = type { %class.Base }
%class.Derived2 = type { %class.Base }

@_ZTV8Derived1 = linkonce_odr dso_local unnamed_addr constant { [3 x i8*] } { [3 x i8*] [i8* null, i8* bitcast ({ i8 addrspace(4)*, i8*, i8* }* @_ZTI8Derived1 to i8*), i8* bitcast (i32 (%class.Derived1 addrspace(4)*)* @_ZN8Derived17displayEv to i8*)] }, align 8
@_ZTVN10__cxxabiv120__si_class_type_infoE = external dso_local addrspace(1) global i8 addrspace(4)*
@_ZTS8Derived1 = linkonce_odr dso_local constant [10 x i8] c"8Derived1\00", align 1
@_ZTVN10__cxxabiv117__class_type_infoE = external dso_local addrspace(1) global i8 addrspace(4)*
@_ZTS4Base = linkonce_odr dso_local constant [6 x i8] c"4Base\00", align 1
@_ZTI4Base = linkonce_odr dso_local constant { i8 addrspace(4)*, i8* } { i8 addrspace(4)* bitcast (i8 addrspace(4)* addrspace(4)* getelementptr inbounds (i8 addrspace(4)*, i8 addrspace(4)* addrspace(4)* addrspacecast (i8 addrspace(4)* addrspace(1)* @_ZTVN10__cxxabiv117__class_type_infoE to i8 addrspace(4)* addrspace(4)*), i64 2) to i8 addrspace(4)*), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @_ZTS4Base, i32 0, i32 0) }, align 8
@_ZTI8Derived1 = linkonce_odr dso_local constant { i8 addrspace(4)*, i8*, i8* } { i8 addrspace(4)* bitcast (i8 addrspace(4)* addrspace(4)* getelementptr inbounds (i8 addrspace(4)*, i8 addrspace(4)* addrspace(4)* addrspacecast (i8 addrspace(4)* addrspace(1)* @_ZTVN10__cxxabiv120__si_class_type_infoE to i8 addrspace(4)* addrspace(4)*), i64 2) to i8 addrspace(4)*), i8* getelementptr inbounds ([10 x i8], [10 x i8]* @_ZTS8Derived1, i32 0, i32 0), i8* bitcast ({ i8 addrspace(4)*, i8* }* @_ZTI4Base to i8*) }, align 8
@_ZTV8Derived2 = linkonce_odr dso_local unnamed_addr constant { [3 x i8*] } { [3 x i8*] [i8* null, i8* bitcast ({ i8 addrspace(4)*, i8*, i8* }* @_ZTI8Derived2 to i8*), i8* bitcast (i32 (%class.Derived2 addrspace(4)*)* @_ZN8Derived27displayEv to i8*)] }, align 8
@_ZTS8Derived2 = linkonce_odr dso_local constant [10 x i8] c"8Derived2\00", align 1
@_ZTI8Derived2 = linkonce_odr dso_local constant { i8 addrspace(4)*, i8*, i8* } { i8 addrspace(4)* bitcast (i8 addrspace(4)* addrspace(4)* getelementptr inbounds (i8 addrspace(4)*, i8 addrspace(4)* addrspace(4)* addrspacecast (i8 addrspace(4)* addrspace(1)* @_ZTVN10__cxxabiv120__si_class_type_infoE to i8 addrspace(4)* addrspace(4)*), i64 2) to i8 addrspace(4)*), i8* getelementptr inbounds ([10 x i8], [10 x i8]* @_ZTS8Derived2, i32 0, i32 0), i8* bitcast ({ i8 addrspace(4)*, i8* }* @_ZTI4Base to i8*) }, align 8

; Function Attrs: mustprogress norecurse nounwind
define weak_odr dso_local spir_kernel void @_ZTSZ4mainEUlvE_(%class.Base addrspace(4)* addrspace(1)* noundef align 8 %_arg_Ptrs, i8 addrspace(1)* noundef align 1 %_arg_Storage) local_unnamed_addr #0 !srcloc !48 !kernel_arg_buffer_location !49 !sycl_fixed_targets !50 !sycl_kernel_omit_args !51 {
entry:
  %0 = bitcast i8 addrspace(1)* %_arg_Storage to %class.Derived1 addrspace(1)*
  %1 = addrspacecast i8 addrspace(1)* %_arg_Storage to %class.Derived1 addrspace(4)*
  %2 = getelementptr %class.Derived1, %class.Derived1 addrspace(1)* %0, i64 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [3 x i8*] }, { [3 x i8*] }* @_ZTV8Derived1, i64 0, i32 0, i64 2) to i32 (...)**), i32 (...)** addrspace(1)* %2, align 8, !tbaa !52
  %3 = getelementptr %class.Derived1, %class.Derived1 addrspace(4)* %1, i64 0, i32 0
  store %class.Base addrspace(4)* %3, %class.Base addrspace(4)* addrspace(1)* %_arg_Ptrs, align 8, !tbaa !55
  %arrayidx4.i5 = getelementptr inbounds i8, i8 addrspace(1)* %_arg_Storage, i64 8
  %arrayidx4.i = addrspacecast i8 addrspace(1)* %arrayidx4.i5 to i8 addrspace(4)*
  %4 = bitcast i8 addrspace(1)* %arrayidx4.i5 to i32 (...)** addrspace(1)*
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [3 x i8*] }, { [3 x i8*] }* @_ZTV8Derived2, i64 0, i32 0, i64 2) to i32 (...)**), i32 (...)** addrspace(1)* %4, align 8, !tbaa !52
  %arrayidx6.i6 = getelementptr inbounds %class.Base addrspace(4)*, %class.Base addrspace(4)* addrspace(1)* %_arg_Ptrs, i64 1
  %5 = bitcast %class.Base addrspace(4)* addrspace(1)* %arrayidx6.i6 to i8 addrspace(4)* addrspace(1)*
  store i8 addrspace(4)* %arrayidx4.i, i8 addrspace(4)* addrspace(1)* %5, align 8, !tbaa !55
  ret void
}

; Function Attrs: mustprogress norecurse nounwind
define linkonce_odr dso_local spir_func noundef i32 @_ZN8Derived17displayEv(%class.Derived1 addrspace(4)* noundef align 8 dereferenceable_or_null(8) %this) unnamed_addr #1 align 2 !srcloc !58 {
entry:
  ret i32 2
}

; Function Attrs: mustprogress norecurse nounwind
define linkonce_odr dso_local spir_func noundef i32 @_ZN8Derived27displayEv(%class.Derived2 addrspace(4)* noundef align 8 dereferenceable_or_null(8) %this) unnamed_addr #1 align 2 !srcloc !59 {
entry:
  ret i32 3
}

declare dso_local spir_func i32 @_Z18__spirv_ocl_printfPU3AS2Kcz(i8 addrspace(2)*, ...)

attributes #0 = { mustprogress norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="vf2.cpp" "sycl-optlevel"="2" "sycl-single-task" "uniform-work-group-size"="true" }
attributes #1 = { mustprogress norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-optlevel"="2" }

!llvm.module.flags = !{!0, !1}
!opencl.spir.version = !{!2}
!spirv.Source = !{!3}
!sycl_aspects = !{!4, !5, !6, !7, !8, !9, !10, !11, !12, !13, !14, !15, !16, !17, !18, !19, !20, !21, !22, !23, !24, !25, !26, !27, !28, !29, !30, !31, !32, !33, !34, !35, !36, !37, !38, !39, !40, !41, !42, !43, !44, !45, !46}
!llvm.ident = !{!47}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{i32 1, i32 2}
!3 = !{i32 4, i32 100000}
!4 = !{!"cpu", i32 1}
!5 = !{!"gpu", i32 2}
!6 = !{!"accelerator", i32 3}
!7 = !{!"custom", i32 4}
!8 = !{!"fp16", i32 5}
!9 = !{!"fp64", i32 6}
!10 = !{!"image", i32 9}
!11 = !{!"online_compiler", i32 10}
!12 = !{!"online_linker", i32 11}
!13 = !{!"queue_profiling", i32 12}
!14 = !{!"usm_device_allocations", i32 13}
!15 = !{!"usm_host_allocations", i32 14}
!16 = !{!"usm_shared_allocations", i32 15}
!17 = !{!"usm_system_allocations", i32 17}
!18 = !{!"ext_intel_pci_address", i32 18}
!19 = !{!"ext_intel_gpu_eu_count", i32 19}
!20 = !{!"ext_intel_gpu_eu_simd_width", i32 20}
!21 = !{!"ext_intel_gpu_slices", i32 21}
!22 = !{!"ext_intel_gpu_subslices_per_slice", i32 22}
!23 = !{!"ext_intel_gpu_eu_count_per_subslice", i32 23}
!24 = !{!"ext_intel_max_mem_bandwidth", i32 24}
!25 = !{!"ext_intel_mem_channel", i32 25}
!26 = !{!"usm_atomic_host_allocations", i32 26}
!27 = !{!"usm_atomic_shared_allocations", i32 27}
!28 = !{!"atomic64", i32 28}
!29 = !{!"ext_intel_device_info_uuid", i32 29}
!30 = !{!"ext_oneapi_srgb", i32 30}
!31 = !{!"ext_oneapi_native_assert", i32 31}
!32 = !{!"host_debuggable", i32 32}
!33 = !{!"ext_intel_gpu_hw_threads_per_eu", i32 33}
!34 = !{!"ext_oneapi_cuda_async_barrier", i32 34}
!35 = !{!"ext_oneapi_bfloat16_math_functions", i32 35}
!36 = !{!"ext_intel_free_memory", i32 36}
!37 = !{!"ext_intel_device_id", i32 37}
!38 = !{!"ext_intel_memory_clock_rate", i32 38}
!39 = !{!"ext_intel_memory_bus_width", i32 39}
!40 = !{!"emulated", i32 40}
!41 = !{!"ext_intel_legacy_image", i32 41}
!42 = !{!"int64_base_atomics", i32 7}
!43 = !{!"int64_extended_atomics", i32 8}
!44 = !{!"usm_system_allocator", i32 17}
!45 = !{!"usm_restricted_shared_allocations", i32 16}
!46 = !{!"host", i32 0}
!47 = !{!"clang version 17.0.0 "}
!48 = !{i32 546}
!49 = !{i32 -1, i32 -1}
!50 = !{}
!51 = !{i1 false, i1 false}
!52 = !{!53, !53, i64 0}
!53 = !{!"vtable pointer", !54, i64 0}
!54 = !{!"Simple C++ TBAA"}
!55 = !{!56, !56, i64 0}
!56 = !{!"any pointer", !57, i64 0}
!57 = !{!"omnipotent char", !54, i64 0}
!58 = !{i32 193}
!59 = !{i32 273}
