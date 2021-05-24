; RUN: sycl-post-link -spec-const=rt -S %s --ir-output-only -o %t.ll
; RUN: FileCheck %s --input-file=%t.ll --implicit-check-not "call i8 bitcast"

; CHECK-LABEL: void @kernel_A
; CHECK: %[[CALL:.*]] = call i8 @_Z20__spirv_SpecConstantia(i32 [[#]], i8 1)
; CHECK: trunc i8 %[[CALL]] to i1
;
; CHECK-LABEL: void @kernel_B
; CHECK: call i8 @_Z20__spirv_SpecConstantia(i32 [[#]], i8

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown-sycldevice"

%struct.spec_id_type = type { i8 }
%struct.spec_id_type2 = type { %struct.user_type }
%struct.user_type = type { float, i32, i8 }

@name_A = private unnamed_addr addrspace(1) constant [7 x i8] c"name_A\00", align 1
@name_B = private unnamed_addr addrspace(1) constant [7 x i8] c"name_B\00", align 1

@spec_const1 = linkonce_odr addrspace(1) constant %struct.spec_id_type { i8 1 }, align 1
@spec_const2 = linkonce_odr addrspace(1) constant %struct.spec_id_type2 { %struct.user_type { float 2.000000e+01, i32 20, i8 20 } }, align 4

declare spir_func zeroext i1 @_Z37__sycl_getScalar2020SpecConstantValueIbET_PKcPKvS4_(i8 addrspace(4)*, i8 addrspace(4)*, i8 addrspace(4)*) #1

declare spir_func void @_Z40__sycl_getComposite2020SpecConstantValueIN14get_spec_const13testing_types8no_cnstrEET_PKcPKvS7_(%struct.user_type addrspace(4)* sret(%struct.user_type) align 4, i8 addrspace(4)*, i8 addrspace(4)*, i8 addrspace(4)*) #1

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.memcpy.p4i8.p4i8.i64(i8 addrspace(4)* noalias nocapture writeonly, i8 addrspace(4)* noalias nocapture readonly, i64, i1 immarg) #2

; Function Attrs: convergent norecurse
define weak_odr spir_kernel void @kernel_A(i8 addrspace(1)* %_arg) #0 {
entry:
  %call = tail call spir_func zeroext i1 @_Z37__sycl_getScalar2020SpecConstantValueIbET_PKcPKvS4_(i8 addrspace(4)* getelementptr inbounds ([7 x i8], [7 x i8] addrspace(4)* addrspacecast ([7 x i8] addrspace(1)* @name_A to [7 x i8] addrspace(4)*), i64 0, i64 0), i8 addrspace(4)* addrspacecast (i8 addrspace(1)* getelementptr inbounds (%struct.spec_id_type, %struct.spec_id_type addrspace(1)* @spec_const1, i64 0, i32 0) to i8 addrspace(4)*), i8 addrspace(4)* null) #4
  %frombool = zext i1 %call to i8
  store i8 %frombool, i8 addrspace(1)* %_arg, align 1
  ret void
}

; Function Attrs: convergent norecurse
define weak_odr spir_kernel void @kernel_B(%struct.user_type addrspace(1)* %_arg) #2 {
entry:
  %ref.tmp.i = alloca %struct.user_type, align 4
  %ref.tmp.acast.i = addrspacecast %struct.user_type addrspace(0)* %ref.tmp.i to %struct.user_type addrspace(4)*
  call spir_func void @_Z40__sycl_getComposite2020SpecConstantValueIN14get_spec_const13testing_types8no_cnstrEET_PKcPKvS7_(%struct.user_type addrspace(4)* sret(%struct.user_type) align 4 %ref.tmp.acast.i, i8 addrspace(4)* getelementptr inbounds ([7 x i8], [7 x i8] addrspace(4)* addrspacecast ([7 x i8] addrspace(1)* @name_B to [7 x i8] addrspace(4)*), i64 0, i64 0), i8 addrspace(4)* addrspacecast (i8 addrspace(1)* bitcast (%struct.spec_id_type2 addrspace(1)* @spec_const2 to i8 addrspace(1)*) to i8 addrspace(4)*), i8 addrspace(4)* null) #4
  %0 = bitcast %struct.user_type* %ref.tmp.i to i8*
  %1 = bitcast %struct.user_type addrspace(1)* %_arg to i8 addrspace(1)*
  %2 = addrspacecast i8 addrspace(1)* %1 to i8 addrspace(4)*
  %3 = addrspacecast i8* %0 to i8 addrspace(4)*
  call void @llvm.memcpy.p4i8.p4i8.i64(i8 addrspace(4)* noundef align 4 dereferenceable(12) %2, i8 addrspace(4)* noundef align 4 dereferenceable(12) %3, i64 12, i1 false)
  ret void
}

attributes #0 = { convergent norecurse }
attributes #1 = { convergent }
attributes #2 = { argmemonly nofree nosync nounwind willreturn }
