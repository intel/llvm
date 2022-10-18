; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc

; --- Source code ---
; Generated with "intel/llvm/clang++ -fsycl-device-only  -emit-llvm -S"
;
;#include <CL/sycl.hpp>
;
;class CustomClass {
;public:
;   CustomClass(int value);
;private:
;   int data;
;}; 
;
;CustomClass::CustomClass(int value) : data(value) {}
;
;int main() {
;   cl::sycl::queue queue;
;   queue.submit( [&]( cl::sycl::handler& handler ) {
;         handler.single_task< class SYCLCustom >([=]() {
;               CustomClass dummyClass(1);
;            } );
;      } );
;
;   return 0;
;}

; ModuleID = 'custom_class.ll'
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-linux"

%class._ZTS11CustomClass.CustomClass = type { i32 }
%"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon" = type { i8 }

$"_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE10SYCLCustom" = comdat any

@_ZN11CustomClassC1Ei = dso_local unnamed_addr alias void (%class._ZTS11CustomClass.CustomClass addrspace(4)*, i32), void (%class._ZTS11CustomClass.CustomClass addrspace(4)*, i32)* @_ZN11CustomClassC2Ei

; Function Attrs: norecurse
define weak_odr dso_local spir_kernel void @"_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE10SYCLCustom"() #0 comdat !kernel_arg_addr_space !4 !kernel_arg_access_qual !4 !kernel_arg_type !4 !kernel_arg_base_type !4 !kernel_arg_type_qual !4 {
entry:
  %0 = alloca %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon", align 1
  %1 = bitcast %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon"* %0 to i8*
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %1) #4
  %2 = addrspacecast %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon"* %0 to %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon" addrspace(4)*
  call spir_func void @"_ZZZ4mainENK3$_0clERN2cl4sycl7handlerEENKUlvE_clEv"(%"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon" addrspace(4)* %2)
  %3 = bitcast %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon"* %0 to i8*
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %3) #4
  ret void
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: inlinehint norecurse
define internal spir_func void @"_ZZZ4mainENK3$_0clERN2cl4sycl7handlerEENKUlvE_clEv"(%"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon" addrspace(4)* %this) #2 align 2 {
entry:
  %dummyClass = alloca %class._ZTS11CustomClass.CustomClass, align 4
  %0 = bitcast %class._ZTS11CustomClass.CustomClass* %dummyClass to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %0) #4
  %1 = addrspacecast %class._ZTS11CustomClass.CustomClass* %dummyClass to %class._ZTS11CustomClass.CustomClass addrspace(4)*
  call spir_func void @_ZN11CustomClassC1Ei(%class._ZTS11CustomClass.CustomClass addrspace(4)* %1, i32 1)
  %2 = bitcast %class._ZTS11CustomClass.CustomClass* %dummyClass to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %2) #4
  ret void
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: norecurse nounwind
define dso_local spir_func void @_ZN11CustomClassC2Ei(%class._ZTS11CustomClass.CustomClass addrspace(4)* %this, i32 %value) unnamed_addr #3 align 2 {
entry:
  %data = getelementptr inbounds %class._ZTS11CustomClass.CustomClass, %class._ZTS11CustomClass.CustomClass addrspace(4)* %this, i32 0, i32 0
  store i32 %value, i32 addrspace(4)* %data, align 4, !tbaa !5
  ret void
}

attributes #0 = { norecurse "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind willreturn }
attributes #2 = { inlinehint norecurse "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { norecurse nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
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
!5 = !{!6, !7, i64 0}
!6 = !{!"_ZTS11CustomClass", !7, i64 0}
!7 = !{!"int", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C++ TBAA"}
