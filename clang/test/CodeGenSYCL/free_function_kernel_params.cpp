// RUN: %clang_cc1 -internal-isystem %S/Inputs -fsycl-is-device -triple spir64 \
// RUN: -emit-llvm %s -o - | FileCheck %s
// This test checks parameter IR generation for free functions with parameters
// of non-decomposed struct type, work group memory type and dynamic work group memory type.

#include "sycl.hpp"

struct NoPointers {
  int f;
};

struct Pointers {
  int * a;
  float * b;
};

struct Agg {
  NoPointers F1;
  int F2;
  int *F3;
  Pointers F4;
};

__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 0)]]
void ff_4(NoPointers S1, Pointers S2, Agg S3) {
}

constexpr int TestArrSize = 3;

template <int ArrSize>
struct KArgWithPtrArray {
  int *data[ArrSize];
  int start[ArrSize];
  int end[ArrSize];
  constexpr int getArrSize() { return ArrSize; }
};

template <int ArrSize>
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 0)]]
void ff_6(KArgWithPtrArray<ArrSize> KArg) {
  for (int j = 0; j < ArrSize; j++)
    for (int i = KArg.start[j]; i <= KArg.end[j]; i++)
      KArg.data[j][i] = KArg.start[j] + KArg.end[j];
}

template void ff_6(KArgWithPtrArray<TestArrSize> KArg);

// CHECK: %struct.NoPointers = type { i32 }
// CHECK: %struct.Pointers = type { ptr addrspace(4), ptr addrspace(4) }
// CHECK: %struct.Agg = type { %struct.NoPointers, i32, ptr addrspace(4), %struct.Pointers }
// CHECK: %struct.__generated_Pointers = type { ptr addrspace(1), ptr addrspace(1) }
// CHECK: %struct.__generated_Agg = type { %struct.NoPointers, i32, ptr addrspace(1), %struct.__generated_Pointers.0 }
// CHECK: %struct.__generated_Pointers.0 = type { ptr addrspace(1), ptr addrspace(1) }
// CHECK: %struct.__generated_KArgWithPtrArray = type { [3 x ptr addrspace(1)], [3 x i32], [3 x i32] }
// CHECK: %struct.KArgWithPtrArray = type { [3 x ptr addrspace(4)], [3 x i32], [3 x i32] }
// CHECK: define dso_local spir_kernel void @{{.*}}__sycl_kernel{{.*}}(ptr noundef byval(%struct.NoPointers) align 4 %__arg_S1, ptr noundef byval(%struct.__generated_Pointers) align 8 %__arg_S2, ptr noundef byval(%struct.__generated_Agg) align 8 %__arg_S3)
// CHECK: define dso_local spir_kernel void @{{.*}}__sycl_kernel_ff_6{{.*}}(ptr noundef byval(%struct.__generated_KArgWithPtrArray) align 8 %__arg_KArg)

__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 0)]]
void ff_7(sycl::work_group_memory<int> mem) {
}

// CHECK:  define dso_local spir_kernel void @{{.*}}__sycl_kernel_ff_7{{.*}}(ptr addrspace(3) noundef align 4 %__arg_Ptr)
// CHECK:  %__arg_Ptr.addr = alloca ptr addrspace(3), align 8
// CHECK-NEXT:  %mem = alloca %"class.sycl::_V1::work_group_memory", align 8
// CHECK:  %__arg_Ptr.addr.ascast = addrspacecast ptr %__arg_Ptr.addr to ptr addrspace(4)
// CHECK-NEXT:  %mem.ascast = addrspacecast ptr %mem to ptr addrspace(4)
// CHECK:  store ptr addrspace(3) %__arg_Ptr, ptr addrspace(4) %__arg_Ptr.addr.ascast, align 8
// CHECK-NEXT:  [[REGISTER:%[a-zA-Z0-9_]+]] = load ptr addrspace(3), ptr addrspace(4) %__arg_Ptr.addr.ascast, align 8
// CHECK-NEXT:  call spir_func void @{{.*}}work_group_memory{{.*}}__init{{.*}}(ptr addrspace(4) noundef align 8 dereferenceable_or_null(8) %mem.ascast, ptr addrspace(3) noundef [[REGISTER]])

__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 0)]]
void ff_7(sycl::dynamic_work_group_memory<int> DynMem) {
}

// CHECK:  define dso_local spir_kernel void @{{.*}}__sycl_kernel_ff_7{{.*}}(ptr addrspace(3) noundef align 4 %__arg_Ptr)
// CHECK:  %__arg_Ptr.addr = alloca ptr addrspace(3), align 8
// CHECK-NEXT: %DynMem = alloca %"class.sycl::_V1::dynamic_work_group_memory", align 8 
// CHECK:  %__arg_Ptr.addr.ascast = addrspacecast ptr %__arg_Ptr.addr to ptr addrspace(4)
// CHECK-NEXT:  %DynMem.ascast = addrspacecast ptr %DynMem to ptr addrspace(4)
// CHECK:  store ptr addrspace(3) %__arg_Ptr, ptr addrspace(4) %__arg_Ptr.addr.ascast, align 8
// CHECK-NEXT:  [[REGISTER:%[a-zA-Z0-9_]+]] = load ptr addrspace(3), ptr addrspace(4) %__arg_Ptr.addr.ascast, align 8
// CHECK-NEXT:  call spir_func void @{{.*}}dynamic_work_group_memory{{.*}}__init{{.*}}(ptr addrspace(4) noundef align 8 dereferenceable_or_null(8) %DynMem.ascast, ptr addrspace(3) noundef [[REGISTER]])

__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 0)]]
void ff_8(sycl::accessor<int, 1, sycl::access::mode::read> acc) {
} 

// CHECK: define dso_local spir_kernel void @_Z18__sycl_kernel_ff_8N4sycl3_V18accessorIiLi1ELNS0_6access4modeE1024ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE(ptr addrspace(1) noundef align 4 %__arg_Ptr, ptr noundef byval(%"struct.sycl::_V1::range") align 4 %__arg_AccessRange, ptr noundef byval(%"struct.sycl::_V1::range") align 4 %__arg_MemRange, ptr noundef byval(%"struct.sycl::_V1::id") align 4 %__arg_Offset) #5 !srcloc !9 !kernel_arg_buffer_location !23 !sycl_fixed_targets !6 {
// CHECK-NEXT:  %__arg_Ptr.addr = alloca ptr addrspace(1), align 8
// CHECK-NEXT:  %acc = alloca %"class.sycl::_V1::accessor", align 4
// CHECK-NEXT:  %agg.tmp = alloca %"struct.sycl::_V1::range", align 4
// CHECK-NEXT:  %agg.tmp1 = alloca %"struct.sycl::_V1::range", align 4
// CHECK-NEXT:  %agg.tmp2 = alloca %"struct.sycl::_V1::id", align 4
// CHECK-NEXT:  %agg.tmp3 = alloca %"class.sycl::_V1::accessor", align 4
// CHECK-NEXT:  %__arg_Ptr.addr.ascast = addrspacecast ptr %__arg_Ptr.addr to ptr addrspace(4)
// CHECK-NEXT:  %acc.ascast = addrspacecast ptr %acc to ptr addrspace(4)
// CHECK-NEXT:  %agg.tmp.ascast = addrspacecast ptr %agg.tmp to ptr addrspace(4)
// CHECK-NEXT:  %agg.tmp1.ascast = addrspacecast ptr %agg.tmp1 to ptr addrspace(4)
// CHECK-NEXT:  %agg.tmp2.ascast = addrspacecast ptr %agg.tmp2 to ptr addrspace(4)
// CHECK-NEXT:  %agg.tmp3.ascast = addrspacecast ptr %agg.tmp3 to ptr addrspace(4)
// CHECK-NEXT:  store ptr addrspace(1) %__arg_Ptr, ptr addrspace(4) %__arg_Ptr.addr.ascast, align 8
// CHECK-NEXT:  %__arg_AccessRange.ascast = addrspacecast ptr %__arg_AccessRange to ptr addrspace(4)
// CHECK-NEXT:  %__arg_MemRange.ascast = addrspacecast ptr %__arg_MemRange to ptr addrspace(4)
// CHECK-NEXT:  %__arg_Offset.ascast = addrspacecast ptr %__arg_Offset to ptr addrspace(4)


__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 0)]]
void ff_8(sycl::local_accessor<int, 1> lacc) {
} 

// CHECK : define dso_local spir_kernel void @_Z18__sycl_kernel_ff_8N4sycl3_V114local_accessorIiLi1EEE(ptr addrspace(3) noundef align 4 %__arg_Ptr, ptr noundef byval(%"struct.sycl::_V1::range") align 4 %__arg_AccessRange, ptr noundef byval(%"struct.sycl::_V1::range") align 4 %__arg_MemRange, ptr noundef byval(%"struct.sycl::_V1::id") align 4 %__arg_Offset) #5 !srcloc !9 !kernel_arg_buffer_location !22 !sycl_fixed_targets !6 {
  // CHECK: %__arg_Ptr.addr = alloca ptr addrspace(3), align 8
  // CHECK-NEXT: %lacc = alloca %"class.sycl::_V1::local_accessor", align 4
  // CHECK-NEXT: %agg.tmp = alloca %"struct.sycl::_V1::range", align 4
  // CHECK-NEXT: %agg.tmp1 = alloca %"struct.sycl::_V1::range", align 4
  // CHECK-NEXT: %agg.tmp2 = alloca %"struct.sycl::_V1::id", align 4
  // CHECK-NEXT: %agg.tmp3 = alloca %"class.sycl::_V1::local_accessor", align 4
  // CHECK-NEXT: %__arg_Ptr.addr.ascast = addrspacecast ptr %__arg_Ptr.addr to ptr addrspace(4)
  // CHECK-NEXT: %lacc.ascast = addrspacecast ptr %lacc to ptr addrspace(4)
  // CHECK-NEXT: %agg.tmp.ascast = addrspacecast ptr %agg.tmp to ptr addrspace(4)
  // CHECK-NEXT: %agg.tmp1.ascast = addrspacecast ptr %agg.tmp1 to ptr addrspace(4)
  // CHECK-NEXT: %agg.tmp2.ascast = addrspacecast ptr %agg.tmp2 to ptr addrspace(4)
  // CHECK-NEXT: %agg.tmp3.ascast = addrspacecast ptr %agg.tmp3 to ptr addrspace(4)
  // CHECK-NEXT: store ptr addrspace(3) %__arg_Ptr, ptr addrspace(4) %__arg_Ptr.addr.ascast, align 8
  // CHECK-NEXT: %__arg_AccessRange.ascast = addrspacecast ptr %__arg_AccessRange to ptr addrspace(4)
  // CHECK-NEXT: %__arg_MemRange.ascast = addrspacecast ptr %__arg_MemRange to ptr addrspace(4)
  // CHECK-NEXT: %__arg_Offset.ascast = addrspacecast ptr %__arg_Offset to ptr addrspace(4)
  // CHECK-NEXT: call spir_func void @_ZN4sycl3_V114local_accessorIiLi1EEC1Ev(ptr addrspace(4) noundef align 4 dereferenceable_or_null(24) %lacc.ascast) #8
  // CHECK: call spir_func void @_ZN4sycl3_V114local_accessorIiLi1EE6__initEPU3AS3iNS0_5rangeILi1EEES6_NS0_2idILi1EEE(ptr addrspace(4) noundef align 4 dereferenceable_or_null(24) %lacc.ascast, ptr addrspace(3) noundef %0, ptr noundef byval(%"struct.sycl::_V1::range") align 4 %agg.tmp.ascast.ascast, ptr noundef byval(%"struct.sycl::_V1::range") align 4 %agg.tmp1.ascast.ascast, ptr noundef byval(%"struct.sycl::_V1::id") align 4 %agg.tmp2.ascast.ascast) #8
  // CHECK: call spir_func void @_Z4ff_8N4sycl3_V114local_accessorIiLi1EEE(ptr noundef byval(%"class.sycl::_V1::local_accessor") align 4 %agg.tmp3.ascast.ascast) #8

// CHECK: declare spir_func void @_ZN4sycl3_V114local_accessorIiLi1EEC1Ev(ptr addrspace(4) noundef align 4 dereferenceable_or_null(24)) unnamed_addr #7

// CHECK: define linkonce_odr spir_func void @_ZN4sycl3_V114local_accessorIiLi1EE6__initEPU3AS3iNS0_5rangeILi1EEES6_NS0_2idILi1EEE(ptr addrspace(4) noundef align 4 dereferenceable_or_null(24) %this, ptr addrspace(3) noundef %Ptr, ptr noundef byval(%"struct.sycl::_V1::range") align 4 %AccessRange, ptr noundef byval(%"struct.sycl::_V1::range") align 4 %MemRange, ptr noundef byval(%"struct.sycl::_V1::id") align 4 %Offset) #6 comdat align 2 !srcloc !23 {

__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 0)]]
void ff_8(sycl::sampler S) {
} 

// CHECK: define dso_local spir_kernel void @_Z18__sycl_kernel_ff_8N4sycl3_V17samplerE(target("spirv.Sampler") %__arg_Sampler) #5 !srcloc !10 !kernel_arg_buffer_location !16 !sycl_fixed_targets !6 {
// CHECK-NEXT:   %__arg_Sampler.addr = alloca target("spirv.Sampler"), align 8
// CHECK-NEXT:  %S = alloca %"class.sycl::_V1::sampler", align 8
// CHECK-NEXT:   %agg.tmp = alloca %"class.sycl::_V1::sampler", align 8
// CHECK-NEXT:   %__arg_Sampler.addr.ascast = addrspacecast ptr %__arg_Sampler.addr to ptr addrspace(4)
// CHECK-NEXT:   %S.ascast = addrspacecast ptr %S to ptr addrspace(4)
// CHECK-NEXT:   %agg.tmp.ascast = addrspacecast ptr %agg.tmp to ptr addrspace(4)
// CHECK-NEXT:  store target("spirv.Sampler") %__arg_Sampler, ptr addrspace(4) %__arg_Sampler.addr.ascast, align 8
// CHECK-NEXT:  %0 = load target("spirv.Sampler"), ptr addrspace(4) %__arg_Sampler.addr.ascast, align 8
// CHECK-NEXT:  call spir_func void @_ZN4sycl3_V17sampler6__initE11ocl_sampler(ptr addrspace(4) noundef align 8 dereferenceable_or_null(8) %S.ascast, target("spirv.Sampler") %0) #8
// CHECK-NEXT:  call void @llvm.memcpy.p4.p4.i64(ptr addrspace(4) align 8 %agg.tmp.ascast, ptr addrspace(4) align 8 %S.ascast, i64 8, i1 false)
// CHECK-NEXT:  %agg.tmp.ascast.ascast = addrspacecast ptr addrspace(4) %agg.tmp.ascast to ptr
// CHECK-NEXT:   call spir_func void @_Z4ff_8N4sycl3_V17samplerE(ptr noundef byval(%"class.sycl::_V1::sampler") align 8 %agg.tmp.ascast.ascast) #8

// CHECK: define linkonce_odr spir_func void @_ZN4sycl3_V17sampler6__initE11ocl_sampler(ptr addrspace(4) noundef align 8 dereferenceable_or_null(8) %this, target("spirv.Sampler") %Sampler) #6 comdat align 2 !srcloc !24 {

__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 0)]]
void ff_8(sycl::stream str) {
} 

// CHECK: define dso_local spir_kernel void @_Z18__sycl_kernel_ff_8N4sycl3_V16streamE(ptr addrspace(1) noundef align 1 %__arg_Ptr, ptr noundef byval(%"struct.sycl::_V1::range") align 4 %__arg_AccessRange, ptr noundef byval(%"struct.sycl::_V1::range") align 4 %__arg_MemRange, ptr noundef byval(%"struct.sycl::_V1::id") align 4 %__arg_Offset, i32 noundef %__arg__FlushBufferSize) #5 !srcloc !11 !kernel_arg_buffer_location !25 !sycl_fixed_targets !6 {
// CHECK-NEXT:  %__arg_Ptr.addr = alloca ptr addrspace(1), align 8
// CHECK-NEXT:   %__arg__FlushBufferSize.addr = alloca i32, align 4
// CHECK-NEXT:   %str = alloca %"class.sycl::_V1::stream", align 4
// CHECK-NEXT:  %agg.tmp = alloca %"struct.sycl::_V1::range", align 4
// CHECK-NEXT:  %agg.tmp1 = alloca %"struct.sycl::_V1::range", align 4
// CHECK-NEXT:   %agg.tmp2 = alloca %"struct.sycl::_V1::id", align 4
// CHECK-NEXT:  %agg.tmp3 = alloca %"class.sycl::_V1::stream", align 4
// CHECK-NEXT:   %__arg_Ptr.addr.ascast = addrspacecast ptr %__arg_Ptr.addr to ptr addrspace(4)
// CHECK-NEXT:   %__arg__FlushBufferSize.addr.ascast = addrspacecast ptr %__arg__FlushBufferSize.addr to ptr addrspace(4)
// CHECK-NEXT:  %str.ascast = addrspacecast ptr %str to ptr addrspace(4)
// CHECK-NEXT:   %agg.tmp.ascast = addrspacecast ptr %agg.tmp to ptr addrspace(4)
// CHECK-NEXT:   %agg.tmp1.ascast = addrspacecast ptr %agg.tmp1 to ptr addrspace(4)
// CHECK-NEXT:   %agg.tmp2.ascast = addrspacecast ptr %agg.tmp2 to ptr addrspace(4)
// CHECK-NEXT:   %agg.tmp3.ascast = addrspacecast ptr %agg.tmp3 to ptr addrspace(4)
// CHECK-NEXT:  store ptr addrspace(1) %__arg_Ptr, ptr addrspace(4) %__arg_Ptr.addr.ascast, align 8
// CHECK-NEXT:  %__arg_AccessRange.ascast = addrspacecast ptr %__arg_AccessRange to ptr addrspace(4)
// CHECK-NEXT:  %__arg_MemRange.ascast = addrspacecast ptr %__arg_MemRange to ptr addrspace(4)
// CHECK-NEXT:  %__arg_Offset.ascast = addrspacecast ptr %__arg_Offset to ptr addrspace(4)
// CHECK-NEXT:  store i32 %__arg__FlushBufferSize, ptr addrspace(4) %__arg__FlushBufferSize.addr.ascast, align 4
// CHECK-NEXT:  call spir_func void @_ZN4sycl3_V16streamC1Ev(ptr addrspace(4) noundef align 4 dereferenceable_or_null(16) %str.ascast) #8
// CHECK-NEXT:  %0 = load ptr addrspace(1), ptr addrspace(4) %__arg_Ptr.addr.ascast, align 8
// CHECK:  call spir_func void @_ZN4sycl3_V16stream6__initEPU3AS1cNS0_5rangeILi1EEES5_NS0_2idILi1EEEi(ptr addrspace(4) noundef align 4 dereferenceable_or_null(16) %str.ascast, ptr addrspace(1) noundef %0, ptr noundef byval(%"struct.sycl::_V1::range") align 4 %agg.tmp.ascast.ascast, ptr noundef byval(%"struct.sycl::_V1::range") align 4 %agg.tmp1.ascast.ascast, ptr noundef byval(%"struct.sycl::_V1::id") align 4 %agg.tmp2.ascast.ascast, i32 noundef %1) #8
// CHECK:define linkonce_odr spir_func void @_ZN4sycl3_V16streamC1Ev(ptr addrspace(4) noundef align 4 dereferenceable_or_null(16) %this) unnamed_addr #6 comdat align 2 !srcloc !26 {
// CHECK: define linkonce_odr spir_func void @_ZN4sycl3_V16stream6__initEPU3AS1cNS0_5rangeILi1EEES5_NS0_2idILi1EEEi(ptr addrspace(4) noundef align 4 dereferenceable_or_null(16) %this, ptr addrspace(1) noundef %Ptr, ptr noundef byval(%"struct.sycl::_V1::range") align 4 %AccessRange, ptr noundef byval(%"struct.sycl::_V1::range") align 4 %MemRange, ptr noundef byval(%"struct.sycl::_V1::id") align 4 %Offset, i32 noundef %_FlushBufferSize) #6 comdat align 2 !srcloc !27 {

__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 0)]]
void ff_8(sycl::ext::oneapi::experimental::annotated_arg<int> arg) {
} 

// CHECK: define dso_local spir_kernel void @_Z18__sycl_kernel_ff_8N4sycl3_V13ext6oneapi12experimental13annotated_argIiJEEE(i32 noundef %__arg__obj) #5 !srcloc !12 !kernel_arg_buffer_location !16 !sycl_fixed_targets !6 {
// CHECK-NEXT:  %__arg__obj.addr = alloca i32, align 4
// CHECK-NEXT:  %arg = alloca %"class.sycl::_V1::ext::oneapi::experimental::annotated_arg", align 4
// CHECK-NEXT:  %agg.tmp = alloca %"class.sycl::_V1::ext::oneapi::experimental::annotated_arg", align 4
// CHECK-NEXT:  %__arg__obj.addr.ascast = addrspacecast ptr %__arg__obj.addr to ptr addrspace(4)
// CHECK-NEXT:  %arg.ascast = addrspacecast ptr %arg to ptr addrspace(4)
// CHECK-NEXT:  %agg.tmp.ascast = addrspacecast ptr %agg.tmp to ptr addrspace(4)
// CHECK-NEXT:  store i32 %__arg__obj, ptr addrspace(4) %__arg__obj.addr.ascast, align 4
// CHECK-NEXT:  %0 = load i32, ptr addrspace(4) %__arg__obj.addr.ascast, align 4
// CHECK-NEXT:  call spir_func void @_ZN4sycl3_V13ext6oneapi12experimental13annotated_argIiJEE6__initEi(ptr addrspace(4) noundef align 4 dereferenceable_or_null(4) %arg.ascast, i32 noundef %0) #8
// CHECK: define linkonce_odr spir_func void @_ZN4sycl3_V13ext6oneapi12experimental13annotated_argIiJEE6__initEi(ptr addrspace(4) noundef align 4 dereferenceable_or_null(4) %this, i32 noundef %_obj) #6 comdat align 2 !srcloc !33 {

__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 0)]]
void ff_8(sycl::ext::oneapi::experimental::annotated_ptr<int> ptr) {
}

// CHECK: define dso_local spir_kernel void @_Z18__sycl_kernel_ff_8N4sycl3_V13ext6oneapi12experimental13annotated_ptrIiJEEE(ptr addrspace(4) noundef align 4 %__arg__obj) #5 !srcloc !13 !kernel_arg_buffer_location !16 !sycl_fixed_targets !6 {
// CHECK-NEXT:   %__arg__obj.addr = alloca ptr addrspace(4), align 8
// CHECK-NEXT:   %ptr = alloca %"class.sycl::_V1::ext::oneapi::experimental::annotated_ptr", align 8
// CHECK-NEXT:  %agg.tmp = alloca %"class.sycl::_V1::ext::oneapi::experimental::annotated_ptr", align 8
// CHECK-NEXT:  %__arg__obj.addr.ascast = addrspacecast ptr %__arg__obj.addr to ptr addrspace(4)
// CHECK-NEXT:  %ptr.ascast = addrspacecast ptr %ptr to ptr addrspace(4)
// CHECK-NEXT:  %agg.tmp.ascast = addrspacecast ptr %agg.tmp to ptr addrspace(4)
// CHECK-NEXT:  store ptr addrspace(4) %__arg__obj, ptr addrspace(4) %__arg__obj.addr.ascast, align 8
// CHECK-NEXT:  %0 = load ptr addrspace(4), ptr addrspace(4) %__arg__obj.addr.ascast, align 8
// CHECK-NEXT:  call spir_func void @_ZN4sycl3_V13ext6oneapi12experimental13annotated_ptrIiJEE6__initEPi(ptr addrspace(4) noundef align 8 dereferenceable_or_null(8) %ptr.ascast, ptr addrspace(4) noundef %0) #8
// CHECK-NEXT:  call void @llvm.memcpy.p4.p4.i64(ptr addrspace(4) align 8 %agg.tmp.ascast, ptr addrspace(4) align 8 %ptr.ascast, i64 8, i1 false)
// CHECK-NEXT:  %agg.tmp.ascast.ascast = addrspacecast ptr addrspace(4) %agg.tmp.ascast to ptr
// CHECK-NEXT:  call spir_func void @_Z4ff_8N4sycl3_V13ext6oneapi12experimental13annotated_ptrIiJEEE(ptr noundef byval(%"class.sycl::_V1::ext::oneapi::experimental::annotated_ptr") align 8 %agg.tmp.ascast.ascast) #8
// CHECK: define linkonce_odr spir_func void @_ZN4sycl3_V13ext6oneapi12experimental13annotated_ptrIiJEE6__initEPi(ptr addrspace(4) noundef align 8 dereferenceable_or_null(8) %this, ptr addrspace(4) noundef %_obj) #6 comdat align 2 !srcloc !34 {
