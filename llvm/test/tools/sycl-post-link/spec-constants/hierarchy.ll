; For C++ types that come from nested class hierarchy, the LLVM type corresponding
; to that type seems to match the nested structure. However, it also seems that 
; when defining a constant for that type, the LLVM value defining the constant has a type 
; that is different, and is esssentially a flattened out version of the C++ type.
; For example, this test is IR generated from getting the value of a spec constant
; of a struct `scary` that has a deep nested hierarchy, but the specialization_id holding
; the default value of `scary` is a flat struct with all the fields of `scary` flattened out.
; (compare %struct.scary and @_ZL16scary_spec_const)
; This test makes that the spec constant pass can handle such cases.
; (note: IR generated from sycl/test-e2e/SpecConstants/2020/hierarchy.cpp)
; RUN: sycl-post-link -properties --spec-const=native -S %s -o %t.table
; RUN: FileCheck %s -input-file=%t_0.ll

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir64-unknown-unknown"

%struct.anon = type { i32, i32 }
%struct.anon.0 = type { i32 }
%struct.scary = type { %struct.layer4.base, [15 x i8] }
%struct.layer4.base = type { %struct.layer3.base }
%struct.layer3.base = type <{ %struct.layer2, [4 x i8], %struct.foo.base }>
%struct.layer2 = type { %struct.layer1 }
%struct.layer1 = type { %struct.base }
%struct.base = type { float, i8, i32, %struct.anon }
%struct.foo.base = type <{ i32, [4 x i8], [5 x i64], [5 x %struct.anon.0], [5 x i8] }>

@__usid_str = private unnamed_addr constant [44 x i8] c"uid52dfb70f8b72bae7____ZL16scary_spec_const\00", align 1
@_ZL16scary_spec_const = internal addrspace(1) constant { { float, i8, i32, %struct.anon, [4 x i8], i32, [5 x i64], [5 x %struct.anon.0], [5 x i8], [15 x i8] } } { { float, i8, i32, %struct.anon, [4 x i8], i32, [5 x i64], [5 x %struct.anon.0], [5 x i8], [15 x i8] } { float 0.000000e+00, i8 98, i32 0, %struct.anon zeroinitializer, [4 x i8] undef, i32 3, [5 x i64] [i64 5, i64 0, i64 0, i64 0, i64 0], [5 x %struct.anon.0] [%struct.anon.0 { i32 1 }, %struct.anon.0 { i32 2 }, %struct.anon.0 zeroinitializer, %struct.anon.0 zeroinitializer, %struct.anon.0 zeroinitializer], [5 x i8] c"abc\00\00", [15 x i8] undef } }, align 16

define weak_odr dso_local spir_kernel void @_ZTSZ4mainEUlN4sycl3_V114kernel_handlerEE_(ptr addrspace(1) noundef align 16 %_arg_p) {
entry:
  %ref.tmp.i = alloca %struct.scary, align 16
  %ref.tmp.ascast.i = addrspacecast ptr %ref.tmp.i to ptr addrspace(4)
  call spir_func void @_Z40__sycl_getComposite2020SpecConstantValueI5scaryET_PKcPKvS5_(ptr addrspace(4) dead_on_unwind writable sret(%struct.scary) align 16 %ref.tmp.ascast.i, ptr addrspace(4) noundef addrspacecast (ptr @__usid_str to ptr addrspace(4)), ptr addrspace(4) noundef addrspacecast (ptr addrspace(1) @_ZL16scary_spec_const to ptr addrspace(4)), ptr addrspace(4) noundef null)
  call void @llvm.memcpy.p1.p0.i64(ptr addrspace(1) align 16 %_arg_p, ptr align 16 %ref.tmp.i, i64 97, i1 false)
  ret void
}

declare dso_local spir_func void @_Z40__sycl_getComposite2020SpecConstantValueI5scaryET_PKcPKvS5_(ptr addrspace(4) dead_on_unwind writable sret(%struct.scary) align 16, ptr addrspace(4) noundef, ptr addrspace(4) noundef, ptr addrspace(4) noundef)


; CHECK:  %[[#SCV0:]] = {{.*}}@{{.*}}SpecConstant{{.*}}(i32 0, float 0.000000e+00)
; CHECK:  %[[#SCV1:]] = {{.*}}@{{.*}}SpecConstant{{.*}}(i32 1, i8 98)
; CHECK:  %[[#SCV2:]] = {{.*}}@{{.*}}SpecConstant{{.*}}(i32 2, i32 0)
; CHECK:  %[[#SCV3:]] = {{.*}}@{{.*}}SpecConstant{{.*}}(i32 3, i32 0)
; CHECK:  %[[#SCV4:]] = {{.*}}@{{.*}}SpecConstant{{.*}}(i32 4, i32 0)
; CHECK:  %[[#SCV5:]] = {{.*}}@{{.*}}SpecConstantComposite{{.*}}(i32 %[[#SCV3]], i32 %[[#SCV4]])
; CHECK:  %[[#SCV6:]] = {{.*}}@{{.*}}SpecConstantComposite{{.*}}(float %[[#SCV0]], i8 %[[#SCV1]], i32 %[[#SCV2]], %struct.anon %[[#SCV5]])
; CHECK:  %[[#SCV7:]] = {{.*}}@{{.*}}SpecConstantComposite{{.*}}(%struct.base %[[#SCV6]])
; CHECK:  %[[#SCV8:]] = {{.*}}@{{.*}}SpecConstantComposite{{.*}}(%struct.layer1 %[[#SCV7]])
; CHECK:  %[[#SCV9:]] = {{.*}}@{{.*}}SpecConstant{{.*}}(i32 6, i32 3)
; CHECK: %[[#SCV10:]] = {{.*}}@{{.*}}SpecConstant{{.*}}(i32 8, i64 5)
; CHECK: %[[#SCV11:]] = {{.*}}@{{.*}}SpecConstant{{.*}}(i32 9, i64 0)
; CHECK: %[[#SCV12:]] = {{.*}}@{{.*}}SpecConstant{{.*}}(i32 10, i64 0)
; CHECK: %[[#SCV13:]] = {{.*}}@{{.*}}SpecConstant{{.*}}(i32 11, i64 0)
; CHECK: %[[#SCV14:]] = {{.*}}@{{.*}}SpecConstant{{.*}}(i32 12, i64 0)
; CHECK: %[[#SCV15:]] = {{.*}}@{{.*}}SpecConstantComposite{{.*}}(i64 %[[#SCV10]], i64 %[[#SCV11]], i64 %[[#SCV12]], i64 %[[#SCV13]], i64 %[[#SCV14]])
; CHECK: %[[#SCV16:]] = {{.*}}@{{.*}}SpecConstant{{.*}}(i32 13, i32 1)
; CHECK: %[[#SCV17:]] = {{.*}}@{{.*}}SpecConstantComposite{{.*}}(i32 %[[#SCV16]])
; CHECK: %[[#SCV18:]] = {{.*}}@{{.*}}SpecConstant{{.*}}(i32 14, i32 2)
; CHECK: %[[#SCV19:]] = {{.*}}@{{.*}}SpecConstantComposite{{.*}}(i32 %[[#SCV18]])
; CHECK: %[[#SCV20:]] = {{.*}}@{{.*}}SpecConstant{{.*}}(i32 15, i32 0)
; CHECK: %[[#SCV21:]] = {{.*}}@{{.*}}SpecConstantComposite{{.*}}(i32 %[[#SCV20]])
; CHECK: %[[#SCV22:]] = {{.*}}@{{.*}}SpecConstant{{.*}}(i32 16, i32 0)
; CHECK: %[[#SCV23:]] = {{.*}}@{{.*}}SpecConstantComposite{{.*}}(i32 %[[#SCV22]])
; CHECK: %[[#SCV24:]] = {{.*}}@{{.*}}SpecConstant{{.*}}(i32 17, i32 0)
; CHECK: %[[#SCV25:]] = {{.*}}@{{.*}}SpecConstantComposite{{.*}}(i32 %[[#SCV24]])
; CHECK: %[[#SCV26:]] = {{.*}}@{{.*}}SpecConstantComposite{{.*}}(%struct.anon.0 %[[#SCV17]], %struct.anon.0 %[[#SCV19]], %struct.anon.0 %[[#SCV21]], %struct.anon.0 %[[#SCV23]], %struct.anon.0 %[[#SCV25]])
; CHECK: %[[#SCV27:]] = {{.*}}@{{.*}}SpecConstant{{.*}}(i32 18, i8 97)
; CHECK: %[[#SCV28:]] = {{.*}}@{{.*}}SpecConstant{{.*}}(i32 19, i8 98)
; CHECK: %[[#SCV29:]] = {{.*}}@{{.*}}SpecConstant{{.*}}(i32 20, i8 99)
; CHECK: %[[#SCV30:]] = {{.*}}@{{.*}}SpecConstant{{.*}}(i32 21, i8 0)
; CHECK: %[[#SCV31:]] = {{.*}}@{{.*}}SpecConstant{{.*}}(i32 22, i8 0)
; CHECK: %[[#SCV32:]] = {{.*}}@{{.*}}SpecConstantComposite{{.*}}(i8 %[[#SCV27]], i8 %[[#SCV28]], i8 %[[#SCV29]], i8 %[[#SCV30]], i8 %[[#SCV31]])
; CHECK: %[[#SCV33:]] = {{.*}}@{{.*}}SpecConstantComposite{{.*}}(i32 %[[#SCV9]], [4 x i8] undef, [5 x i64] %[[#SCV15]], [5 x %struct.anon.0] %[[#SCV26]], [5 x i8] %[[#SCV32]])
; CHECK: %[[#SCV34:]] = {{.*}}@{{.*}}SpecConstantComposite{{.*}}(%struct.layer2 %[[#SCV8]], [4 x i8] undef, %struct.foo.base %[[#SCV33]])
; CHECK: %[[#SCV35:]] = {{.*}}@{{.*}}SpecConstantComposite{{.*}}(%struct.layer3.base %[[#SCV34]])
; CHECK: %[[#SCV36:]] = {{.*}}@{{.*}}SpecConstantComposite{{.*}}(%struct.layer4.base %[[#SCV35]], [15 x i8] undef)

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p1.p0.i64(ptr addrspace(1) noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg)
