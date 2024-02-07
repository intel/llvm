; RUN: sycl-post-link -spec-const=native -S < %s --ir-output-only -o %t.ll
; RUN: FileCheck %s --input-file=%t.ll --implicit-check-not "call i8 bitcast" --check-prefixes=CHECK,CHECK-RT
; RUN: sycl-post-link -spec-const=emulation -S < %s --ir-output-only -o %t.ll
; RUN: FileCheck %s --input-file=%t.ll --check-prefixes=CHECK,CHECK-DEF
; RUN: sycl-post-link -debug-only=SpecConst -spec-const=native -S < %s 2>&1 | FileCheck %s --check-prefixes=CHECK-LOG,CHECK-LOG-NATIVE
; RUN: sycl-post-link -debug-only=SpecConst -spec-const=emulation -S < %s 2>&1 | FileCheck %s --check-prefixes=CHECK-LOG,CHECK-LOG-EMULATION

; CHECK-LABEL: void @kernel_A
; CHECK-RT: %[[CALL:.*]] = call i8 @_Z20__spirv_SpecConstantia(i32 [[#]], i8 1)
; CHECK-RT: trunc i8 %[[CALL]] to i1
;
; CHECK-DEF: %[[GEP:gep.*]] = getelementptr i8, ptr addrspace(4) null, i32 0
; CHECK-DEF: %[[LOAD:load.*]] = load i8, ptr addrspace(4) %[[GEP]], align 1
; CHECK-DEF: %[[TOBOOL:tobool.*]] = trunc i8 %[[LOAD]] to i1
;
; CHECK-LABEL: void @kernel_B
; CHECK-RT: call i8 @_Z20__spirv_SpecConstantia(i32 [[#]], i8
;
; CHECK-DEF: %[[GEP:gep.*]] = getelementptr i8, ptr addrspace(4) null, i32 4
; CHECK-DEF: %[[BC:bc.*]] = bitcast ptr addrspace(4) %[[GEP]] to ptr addrspace(4)
; CHECK-DEF: %[[LOAD:load.*]] = load %struct.user_type, ptr addrspace(4) %[[BC]], align 4

; CHECK-LOG: sycl.specialization-constants
; CHECK-LOG:[[UNIQUE_PREFIX:[0-9a-zA-Z]+]]={0, 0, 1}
; CHECK-LOG-NATIVE:[[UNIQUE_PREFIX2:[0-9a-zA-Z]+]]={1, 0, 4}
; CHECK-LOG-NATIVE:[[UNIQUE_PREFIX2]]={2, 4, 4}
; CHECK-LOG-NATIVE:[[UNIQUE_PREFIX2]]={3, 8, 1}
; CHECK-LOG-NATIVE:[[UNIQUE_PREFIX2]]={4294967295, 9, 3}
; CHECK-LOG-EMULATION:[[UNIQUE_PREFIX]]={4294967295, 1, 3}
; CHECK-LOG-EMULATION:[[UNIQUE_PREFIX2:[0-9a-zA-Z]+]]={1, 0, 12}
; CHECK-LOG: sycl.specialization-constants-default-values
; CHECK-LOG:{0, 1, 1}
; CHECK-LOG-NATIVE:{1, 4, 2.000000e+01}
; CHECK-LOG-NATIVE:{5, 4, 20}
; CHECK-LOG-NATIVE:{9, 1, 20}
; CHECK-LOG-EMULATION:{1, 3, 0}
; CHECK-LOG-EMULATION:{4, 4, 2.000000e+01}
; CHECK-LOG-EMULATION:{8, 4, 20}
; CHECK-LOG-EMULATION:{12, 1, 20}

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%struct.spec_id_type = type { i8 }
%struct.spec_id_type2 = type { %struct.user_type }
%struct.user_type = type { float, i32, i8 }

@name_A = private unnamed_addr addrspace(1) constant [7 x i8] c"name_A\00", align 1
@name_B = private unnamed_addr addrspace(1) constant [7 x i8] c"name_B\00", align 1

@spec_const1 = linkonce_odr addrspace(1) constant %struct.spec_id_type { i8 1 }, align 1
@spec_const2 = linkonce_odr addrspace(1) constant %struct.spec_id_type2 { %struct.user_type { float 2.000000e+01, i32 20, i8 20 } }, align 4

declare spir_func zeroext i1 @_Z37__sycl_getScalar2020SpecConstantValueIbET_PKcPKvS4_(ptr addrspace(4), ptr addrspace(4), ptr addrspace(4)) #1

declare spir_func void @_Z40__sycl_getComposite2020SpecConstantValueIN14get_spec_const13testing_types8no_cnstrEET_PKcPKvS7_(ptr addrspace(4) sret(%struct.user_type) align 4, ptr addrspace(4), ptr addrspace(4), ptr addrspace(4)) #1

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.memcpy.p4.p4.i64(ptr addrspace(4) noalias nocapture writeonly, ptr addrspace(4) noalias nocapture readonly, i64, i1 immarg) #2

; Function Attrs: convergent norecurse
define weak_odr spir_kernel void @kernel_A(ptr addrspace(1) %_arg) #0 {
entry:
  %call = tail call spir_func zeroext i1 @_Z37__sycl_getScalar2020SpecConstantValueIbET_PKcPKvS4_(ptr addrspace(4) getelementptr inbounds ([7 x i8], ptr addrspace(4) addrspacecast (ptr addrspace(1) @name_A to ptr addrspace(4)), i64 0, i64 0), ptr addrspace(4) addrspacecast (ptr addrspace(1) getelementptr inbounds (%struct.spec_id_type, ptr addrspace(1) @spec_const1, i64 0, i32 0) to ptr addrspace(4)), ptr addrspace(4) null) #4
  %frombool = zext i1 %call to i8
  store i8 %frombool, ptr addrspace(1) %_arg, align 1
  ret void
}

; Function Attrs: convergent norecurse
define weak_odr spir_kernel void @kernel_B(ptr addrspace(1) %_arg) #2 {
entry:
  %ref.tmp.i = alloca %struct.user_type, align 4
  %ref.tmp.acast.i = addrspacecast ptr addrspace(0) %ref.tmp.i to ptr addrspace(4)
  call spir_func void @_Z40__sycl_getComposite2020SpecConstantValueIN14get_spec_const13testing_types8no_cnstrEET_PKcPKvS7_(ptr addrspace(4) sret(%struct.user_type) align 4 %ref.tmp.acast.i, ptr addrspace(4) getelementptr inbounds ([7 x i8], ptr addrspace(4) addrspacecast (ptr addrspace(1) @name_B to ptr addrspace(4)), i64 0, i64 0), ptr addrspace(4) addrspacecast (ptr addrspace(1) @spec_const2 to ptr addrspace(4)), ptr addrspace(4) null) #4
  %0 = addrspacecast ptr addrspace(1) %_arg to ptr addrspace(4)
  %1 = addrspacecast ptr %ref.tmp.i to ptr addrspace(4)
  call void @llvm.memcpy.p4.p4.i64(ptr addrspace(4) noundef align 4 dereferenceable(12) %0, ptr addrspace(4) noundef align 4 dereferenceable(12) %1, i64 12, i1 false)
  ret void
}

attributes #0 = { convergent norecurse }
attributes #1 = { convergent }
attributes #2 = { argmemonly nofree nosync nounwind willreturn }
