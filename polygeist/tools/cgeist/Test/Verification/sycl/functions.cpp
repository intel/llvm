// RUN: clang++ -fsycl -fsycl-device-only -O0 -w -emit-mlir %s -o - | FileCheck %s --check-prefix=CHECK-MLIR
// RUN: clang++ -fsycl -fsycl-device-only -O0 -w -emit-mlir %s -o - -Xcgeist -no-mangled-function-name | FileCheck %s --check-prefix=CHECK-MLIR-NO-MANGLED-FUNCTION-NAME
// COM: These two should obtain the same results.
// RUN: clang++ -fsycl -fsycl-device-only -O0 -w -S -emit-llvm -fsycl-targets=spir64-unknown-unknown-syclmlir %s -o - | FileCheck %s --check-prefix=CHECK-LLVM
// RUN: clang++ -fsycl -fsycl-device-only -O0 -w -S -emit-llvm -fsycl-targets=spir64-unknown-unknown-syclmlir %s -o - -Xcgeist -no-mangled-function-name | FileCheck %s --check-prefix=CHECK-LLVM

#include <sycl/sycl.hpp>

// CHECK-MLIR-NO-MANGLED-FUNCTION-NAME-NOT: {{^(sycl\.constructor|sycl\.call){,0}.*}} MangledFunctionName

// CHECK-MLIR-DAG: !sycl_array_1_ = !sycl.array<[1], (memref<1xi64, 4>)>
// CHECK-MLIR-DAG: !sycl_array_2_ = !sycl.array<[2], (memref<2xi64, 4>)>
// CHECK-MLIR-DAG: !sycl_id_1_ = !sycl.id<[1], (!sycl_array_1_)>
// CHECK-MLIR-DAG: !sycl_id_2_ = !sycl.id<[2], (!sycl_array_2_)>
// CHECK-MLIR-DAG: !sycl_range_1_ = !sycl.range<[1], (!sycl_array_1_)>
// CHECK-MLIR-DAG: !sycl_range_2_ = !sycl.range<[2], (!sycl_array_2_)>
// CHECK-MLIR-DAG: !sycl_accessor_impl_device_1_ = !sycl.accessor_impl_device<[1], (!sycl_id_1_, !sycl_range_1_, !sycl_range_1_)>
// CHECK-MLIR-DAG: !sycl_accessor_impl_device_2_ = !sycl.accessor_impl_device<[2], (!sycl_id_2_, !sycl_range_2_, !sycl_range_2_)>
// CHECK-MLIR-DAG: !sycl_accessor_1_i32_rw_gb = !sycl.accessor<[1, i32, read_write, global_buffer], (!sycl_accessor_impl_device_1_, !llvm.struct<(memref<?xi32, 1>)>)>
// CHECK-MLIR-DAG: !sycl_accessor_2_i32_rw_gb = !sycl.accessor<[2, i32, read_write, global_buffer], (!sycl_accessor_impl_device_2_, !llvm.struct<(memref<?xi32, 1>)>)>
// CHECK-MLIR-DAG: ![[ACC_STRUCT:.*]] = !sycl.accessor<[1, !llvm.struct<(i32)>, read_write, global_buffer], (!sycl_accessor_impl_device_1_, !llvm.struct<(ptr<struct<(i32)>, 1>)>)>
// CHECK-MLIR-DAG: ![[ACC_SUBSCRIPT:.*]] = !sycl.accessor_subscript<[1], (!sycl_id_2_, !sycl_accessor_2_i32_rw_gb)>
// CHECK-MLIR-DAG: ![[ITEM_BASE1:.*]] = !sycl.item_base<[1, true], (!sycl_range_1_, !sycl_id_1_, !sycl_id_1_)
// CHECK-MLIR-DAG: ![[ITEM_BASE2:.*]] = !sycl.item_base<[2, true], (!sycl_range_2_, !sycl_id_2_, !sycl_id_2_)>
// CHECK-MLIR-DAG: ![[ITEM1:.*]] = !sycl.item<[1, true], (![[ITEM_BASE1]])>
// CHECK-MLIR-DAG: ![[ITEM2:.*]] = !sycl.item<[2, true], (![[ITEM_BASE2]])>

// CHECK-LLVM-DAG: %"class.sycl::_V1::detail::array.1" = type { [1 x i64] }
// CHECK-LLVM-DAG: %"class.sycl::_V1::detail::array.2" = type { [2 x i64] }
// CHECK-LLVM-DAG: %"class.sycl::_V1::id.1" = type { %"class.sycl::_V1::detail::array.1" }
// CHECK-LLVM-DAG: %"class.sycl::_V1::id.2" = type { %"class.sycl::_V1::detail::array.2" }
// CHECK-LLVM-DAG: %"class.sycl::_V1::range.1" = type { %"class.sycl::_V1::detail::array.1" }
// CHECK-LLVM-DAG: %"class.sycl::_V1::range.2" = type { %"class.sycl::_V1::detail::array.2" }
// CHECK-LLVM-DAG: %"class.sycl::_V1::detail::AccessorImplDevice.1" = type { %"class.sycl::_V1::id.1", %"class.sycl::_V1::range.1", %"class.sycl::_V1::range.1" }
// CHECK-LLVM-DAG: %"class.sycl::_V1::detail::AccessorImplDevice.2" = type { %"class.sycl::_V1::id.2", %"class.sycl::_V1::range.2", %"class.sycl::_V1::range.2" }
// CHECK-LLVM-DAG: %"class.sycl::_V1::accessor.1" = type { %"class.sycl::_V1::detail::AccessorImplDevice.1", { i32 addrspace(1)* } }
// CHECK-LLVM-DAG: %"class.sycl::_V1::accessor.2" = type { %"class.sycl::_V1::detail::AccessorImplDevice.2", { i32 addrspace(1)* } }
// CHECK-LLVM-DAG: %"class.sycl::_V1::accessor.1.1" = type { %"class.sycl::_V1::detail::AccessorImplDevice.1", { { i32 } addrspace(1)* } }
// CHECK-LLVM-DAG: %"class.sycl::_V1::detail::accessor_common.AccessorSubscript.1" = type { %"class.sycl::_V1::id.2", %"class.sycl::_V1::accessor.2" }

template <typename T> SYCL_EXTERNAL void keep(T);

// CHECK-MLIR-LABEL: func.func @_Z29accessor_subscript_operator_0N4sycl3_V18accessorIiLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEENS0_2idILi2EEE(
// CHECK-MLIR:           %{{.*}}: memref<?x!sycl_accessor_2_i32_rw_gb> {llvm.align = 8 : i64, llvm.byval = !sycl_accessor_2_i32_rw_gb, llvm.noundef}, 
// CHECK_MLIR_SAME:      %{{.*}}: memref<?x!sycl_id_2_> {llvm.align = 8 : i64, llvm.byval = !sycl_id_2_, llvm.noundef})
// CHECK-MLIR: %{{.*}} = sycl.accessor.subscript %{{.*}}[%{{.*}}] {ArgumentTypes = [memref<?x!sycl_accessor_2_i32_rw_gb, 4>, memref<?x!sycl_id_2_>], FunctionName = @"operator[]", MangledFunctionName = @_ZNK4sycl3_V18accessorIiLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEixILi2EvEERiNS0_2idILi2EEE, TypeName = @accessor} : (!sycl_accessor_2_i32_rw_gb, !sycl_id_2_) -> memref<?xi32, 4>

// CHECK-LLVM-LABEL: define spir_func void @_Z29accessor_subscript_operator_0N4sycl3_V18accessorIiLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEENS0_2idILi2EEE(
// CHECK-LLVM:           %"class.sycl::_V1::accessor.2"* noundef byval(%"class.sycl::_V1::accessor.2") align 8 %0, %"class.sycl::_V1::id.2"* noundef byval(%"class.sycl::_V1::id.2") align 8 %1) #[[FUNCATTRS:[0-9]+]]
// CHECK-LLVM:  %{{.*}} = call spir_func i32 addrspace(4)* @_ZNK4sycl3_V18accessorIiLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEixILi2EvEERiNS0_2idILi2EEE(%"class.sycl::_V1::accessor.2" addrspace(4)* %{{.*}}, %"class.sycl::_V1::id.2"* %{{.*}})

SYCL_EXTERNAL void accessor_subscript_operator_0(sycl::accessor<sycl::cl_int, 2> acc, sycl::id<2> index) {
  keep(acc[index]);
}

// CHECK-MLIR-LABEL: func.func @_Z29accessor_subscript_operator_1N4sycl3_V18accessorIiLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEEm(
// CHECK_MLIR:           %{{.*}}: memref<?x!sycl_accessor_2_i32_rw_gb>, %{{.*}}: i64)
// CHECK-MLIR: %{{.*}} = sycl.accessor.subscript %{{.*}}[%{{.*}}] {ArgumentTypes = [memref<?x!sycl_accessor_2_i32_rw_gb, 4>, i64], FunctionName = @"operator[]", MangledFunctionName = @_ZNK4sycl3_V18accessorIiLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEixILi2EvEEDam, TypeName = @accessor} : (!sycl_accessor_2_i32_rw_gb, i64) -> ![[ACC_SUBSCRIPT]]

// CHECK-LLVM-LABEL: define spir_func void @_Z29accessor_subscript_operator_1N4sycl3_V18accessorIiLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEEm(
// CHECK-LLVM:           %"class.sycl::_V1::accessor.2"* noundef byval(%"class.sycl::_V1::accessor.2") align 8 %0, i64 noundef %1) #[[FUNCATTRS]]
// CHECK-LLVM: %{{.*}} = call spir_func %"class.sycl::_V1::detail::accessor_common.AccessorSubscript.1" @_ZNK4sycl3_V18accessorIiLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEixILi2EvEEDam(%"class.sycl::_V1::accessor.2" addrspace(4)* %{{.*}}, i64 %1)

SYCL_EXTERNAL void accessor_subscript_operator_1(sycl::accessor<sycl::cl_int, 2> acc, size_t index) {
  keep(acc[index]);
}

// CHECK-MLIR-LABEL: func.func @_Z29accessor_subscript_operator_2N4sycl3_V18accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEEm(
// CHECK-MLIR:           %{{.*}}: memref<?x!sycl_accessor_1_i32_rw_gb> {llvm.align = 8 : i64, llvm.byval = !sycl_accessor_1_i32_rw_gb, llvm.noundef}, 
// CHECK-MLIR-SAME:      %{{.*}}: i64 {llvm.noundef})
// CHECK-MLIR: %{{.*}} = sycl.accessor.subscript %{{.*}}[%{{.*}}] {ArgumentTypes = [memref<?x!sycl_accessor_1_i32_rw_gb, 4>, memref<?x!sycl_id_1_>], FunctionName = @"operator[]", MangledFunctionName = @_ZNK4sycl3_V18accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEixILi1EvEERiNS0_2idILi1EEE, TypeName = @accessor} : (!sycl_accessor_1_i32_rw_gb, !sycl_id_1_) -> memref<?xi32, 4>

// CHECK-LLVM-LABEL: define spir_func void @_Z29accessor_subscript_operator_2N4sycl3_V18accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEEm(
// CHECK-LLVM:           %"class.sycl::_V1::accessor.1"* noundef byval(%"class.sycl::_V1::accessor.1") align 8 %0, i64 noundef %1) #[[FUNCATTRS]]  
// CHECK-LLVM:  %{{.*}} = call spir_func i32 addrspace(4)* @_ZNK4sycl3_V18accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEixILi1EvEERiNS0_2idILi1EEE(%"class.sycl::_V1::accessor.1" addrspace(4)* %{{.*}}, %"class.sycl::_V1::id.1"* %{{.*}})

SYCL_EXTERNAL void accessor_subscript_operator_2(sycl::accessor<sycl::cl_int, 1> acc, size_t index) {
  keep(acc[index]);
}

// CHECK-MLIR-LABEL: func.func @_Z29accessor_subscript_operator_3N4sycl3_V18accessorI6StructLi1ELNS0_6access4modeE1026ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEEm(
// CHECK-MLIR:          %{{.*}}: memref<?x![[ACC_STRUCT]]> {llvm.align = 8 : i64, llvm.byval = ![[ACC_STRUCT]], llvm.noundef}, 
// CHECK-MLIR-SAME:     %{{.*}}: i64 {llvm.noundef})
// CHECK-MLIR: %{{.*}} = sycl.accessor.subscript %{{.*}}[%{{.*}}] {ArgumentTypes = [memref<?x![[ACC_STRUCT]], 4>, memref<?x!sycl_id_1_>], FunctionName = @"operator[]", MangledFunctionName = @_ZNK4sycl3_V18accessorI6StructLi1ELNS0_6access4modeE1026ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEixILi1EvEERS2_NS0_2idILi1EEE, TypeName = @accessor} : (![[ACC_STRUCT]], !sycl_id_1_) -> !llvm.ptr<struct<(i32)>, 4> 

// CHECK-LLVM-LABEL: define spir_func void @_Z29accessor_subscript_operator_3N4sycl3_V18accessorI6StructLi1ELNS0_6access4modeE1026ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEEm(
// CHECK-LLVM:           %"class.sycl::_V1::accessor.1.1"* noundef byval(%"class.sycl::_V1::accessor.1.1") align 8 %0, i64 noundef %1) #[[FUNCATTRS]] 
// CHECK-LLVM:  %{{.*}} = call spir_func { i32 } addrspace(4)* @_ZNK4sycl3_V18accessorI6StructLi1ELNS0_6access4modeE1026ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEixILi1EvEERS2_NS0_2idILi1EEE(%"class.sycl::_V1::accessor.1.1" addrspace(4)* %{{.*}}, %"class.sycl::_V1::id.1"* %{{.*}})
typedef struct {
  unsigned field;
} Struct;
SYCL_EXTERNAL void accessor_subscript_operator_3(sycl::accessor<Struct, 1> acc, size_t index) {
  keep(acc[index].field);
}

// CHECK-MLIR-LABEL: func.func @_Z11range_get_0N4sycl3_V15rangeILi2EEEi(
// CHECK-MLIR:           %{{.*}}: memref<?x!sycl_range_2_> {llvm.align = 8 : i64, llvm.byval = !sycl_range_2_, llvm.noundef}, 
// CHECK-MLIR-SAME:      %{{.*}}: i32 {llvm.noundef})
// CHECK-MLIR: %{{.*}} = "sycl.range.get"(%{{.*}}, %{{.*}}) {ArgumentTypes = [memref<?x!sycl_array_2_, 4>, i32], FunctionName = @get, MangledFunctionName = @_ZNK4sycl3_V16detail5arrayILi2EE3getEi, TypeName = @array} : (!sycl_range_2_, i32) -> i64

// CHECK-LLVM-LABEL: define spir_func void @_Z11range_get_0N4sycl3_V15rangeILi2EEEi(
// CHECK-LLVM:           %"class.sycl::_V1::range.2"* noundef byval(%"class.sycl::_V1::range.2") align 8 %0, i32 noundef %1) #[[FUNCATTRS]] 
// CHECK-LLVM: %{{.*}} = call spir_func i64 @_ZNK4sycl3_V16detail5arrayILi2EE3getEi(%"class.sycl::_V1::detail::array.2" addrspace(4)* %{{.*}}, i32 %1)

SYCL_EXTERNAL void range_get_0(sycl::range<2> r, int dimension) {
  keep(r.get(dimension));
}

// CHECK-MLIR-LABEL: func.func @_Z11range_get_1N4sycl3_V15rangeILi2EEEi(
// CHECK-MLIR:           %{{.*}}: memref<?x!sycl_range_2_> {llvm.align = 8 : i64, llvm.byval = !sycl_range_2_, llvm.noundef}
// CHECK-MLIR-SAME:      %{{.*}}: i32 {llvm.noundef})
// CHECK-MLIR: %{{.*}} = "sycl.range.get"(%{{.*}}, %{{.*}}) {ArgumentTypes = [memref<?x!sycl_array_2_, 4>, i32], FunctionName = @"operator[]", MangledFunctionName = @_ZN4sycl3_V16detail5arrayILi2EEixEi, TypeName = @array} : (!sycl_range_2_, i32) -> memref<?xi64, 4>

// CHECK-LLVM-LABEL: define spir_func void @_Z11range_get_1N4sycl3_V15rangeILi2EEEi(
// CHECK-LLVM:           %"class.sycl::_V1::range.2"* noundef byval(%"class.sycl::_V1::range.2") align 8 %0, i32 noundef %1) #[[FUNCATTRS]]   
// CHECK-LLVM: %{{.*}} = call spir_func i64 addrspace(4)* @_ZN4sycl3_V16detail5arrayILi2EEixEi(%"class.sycl::_V1::detail::array.2" addrspace(4)* %{{.*}}, i32 %1)

SYCL_EXTERNAL void range_get_1(sycl::range<2> r, int dimension) {
  keep(r[dimension]);
}

// CHECK-MLIR-LABEL: func.func @_Z11range_get_2N4sycl3_V15rangeILi2EEEi(
// CHECK-MLIR:           %{{.*}}: memref<?x!sycl_range_2_> {llvm.align = 8 : i64, llvm.byval = !sycl_range_2_, llvm.noundef}
// CHECK-MLIR-SAME:      %{{.*}}: i32 {llvm.noundef})
// CHECK-MLIR: %{{.*}} = "sycl.range.get"(%{{.*}}, %{{.*}}) {ArgumentTypes = [memref<?x!sycl_array_2_, 4>, i32], FunctionName = @"operator[]", MangledFunctionName = @_ZNK4sycl3_V16detail5arrayILi2EEixEi, TypeName = @array} : (!sycl_range_2_, i32) -> i64

// CHECK-LLVM-LABEL: define spir_func void @_Z11range_get_2N4sycl3_V15rangeILi2EEEi(
// CHECK-LLVM:           %"class.sycl::_V1::range.2"* noundef byval(%"class.sycl::_V1::range.2") align 8 %0, i32 noundef %1) #[[FUNCATTRS]]     
// CHECK-LLVM: %{{.*}} = call spir_func i64 @_ZNK4sycl3_V16detail5arrayILi2EEixEi(%"class.sycl::_V1::detail::array.2" addrspace(4)* %{{.*}}, i32 %1)

SYCL_EXTERNAL void range_get_2(const sycl::range<2> r, int dimension) {
  keep(r[dimension]);
}

// CHECK-MLIR-LABEL: func.func @_Z10range_sizeN4sycl3_V15rangeILi2EEE(
// CHECK-MLIR:           %{{.*}}: memref<?x!sycl_range_2_> {llvm.align = 8 : i64, llvm.byval = !sycl_range_2_, llvm.noundef})
// CHECK-MLIR: %{{.*}} = "sycl.range.size"(%{{.*}}) {ArgumentTypes = [memref<?x!sycl_range_2_, 4>], FunctionName = @size, MangledFunctionName = @_ZNK4sycl3_V15rangeILi2EE4sizeEv, TypeName = @range} : (!sycl_range_2_) -> i64

// CHECK-LLVM-LABEL: define spir_func void @_Z10range_sizeN4sycl3_V15rangeILi2EEE(
// CHECK-LLVM:           %"class.sycl::_V1::range.2"* noundef byval(%"class.sycl::_V1::range.2") align 8 %0) #[[FUNCATTRS]]       
// CHECK-LLVM: %{{.*}} = call spir_func i64 @_ZNK4sycl3_V15rangeILi2EE4sizeEv(%"class.sycl::_V1::range.2" addrspace(4)* %{{.*}})

SYCL_EXTERNAL void range_size(sycl::range<2> r) {
  keep(r.size());
}

// CHECK-MLIR-LABEL: func.func @_Z25nd_range_get_global_rangeN4sycl3_V18nd_rangeILi2EEE(
// CHECK-MLIR:           %{{.*}}: memref<?x!sycl_nd_range_2_> {llvm.align = 8 : i64, llvm.byval = !sycl_nd_range_2_, llvm.noundef})
// CHECK-MLIR: %{{.*}} = "sycl.nd_range.get_global_range"(%{{.*}}) {ArgumentTypes = [memref<?x!sycl_nd_range_2_, 4>], FunctionName = @get_global_range, MangledFunctionName = @_ZNK4sycl3_V18nd_rangeILi2EE16get_global_rangeEv, TypeName = @nd_range} : (!sycl_nd_range_2_) -> !sycl_range_2_

// CHECK-MLIR:           func.func @_ZNK4sycl3_V18nd_rangeILi2EE16get_global_rangeEv(%[[VAL_0:.*]]: memref<?x!sycl_nd_range_2_, 4> {llvm.align = 8 : i64, llvm.dereferenceable_or_null = 48 : i64, llvm.noundef}) -> !sycl_range_2_
// CHECK-MLIR-NEXT:             %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK-MLIR-NEXT:             %[[VAL_2:.*]] = memref.alloca() : memref<1x!sycl_range_2_>
// CHECK-MLIR-NEXT:             %[[VAL_3:.*]] = "polygeist.subindex"(%[[VAL_0]], %[[VAL_1]]) : (memref<?x!sycl_nd_range_2_, 4>, index) -> memref<?x!sycl_range_2_, 4>
// CHECK-MLIR-NEXT:             %[[VAL_4:.*]] = "polygeist.memref2pointer"(%[[VAL_2]]) : (memref<1x!sycl_range_2_>) -> !llvm.ptr<!sycl_range_2_>
// CHECK-MLIR-NEXT:             %[[VAL_5:.*]] = llvm.addrspacecast %[[VAL_4]] : !llvm.ptr<!sycl_range_2_> to !llvm.ptr<!sycl_range_2_, 4>
// CHECK-MLIR-NEXT:             %[[VAL_6:.*]] = "polygeist.pointer2memref"(%[[VAL_5]]) : (!llvm.ptr<!sycl_range_2_, 4>) -> memref<?x!sycl_range_2_, 4>
// CHECK-MLIR-NEXT:             sycl.constructor(%[[VAL_6]], %[[VAL_3]]) {MangledFunctionName = @_ZN4sycl3_V15rangeILi2EEC1ERKS2_, TypeName = @range} : (memref<?x!sycl_range_2_, 4>, memref<?x!sycl_range_2_, 4>) -> ()
// CHECK-MLIR-NEXT:             %[[VAL_7:.*]] = affine.load %[[VAL_2]][0] : memref<1x!sycl_range_2_>
// CHECK-MLIR-NEXT:             return %[[VAL_7]] : !sycl_range_2_
// CHECK-MLIR-NEXT:           }

// CHECK-LLVM-LABEL: define spir_func void @_Z25nd_range_get_global_rangeN4sycl3_V18nd_rangeILi2EEE(
// CHECK-LLVM:           %"class.sycl::_V1::nd_range.2"* noundef byval(%"class.sycl::_V1::nd_range.2") align 8 %0) #[[FUNCATTRS]]  
// CHECK-LLVM: %{{.*}} = call spir_func %"class.sycl::_V1::range.2" @_ZNK4sycl3_V18nd_rangeILi2EE16get_global_rangeEv(%"class.sycl::_V1::nd_range.2" addrspace(4)* %{{.*}})

// VAL_3 has incorrect type. Issue 7972 open in GitHub to address this.
// CHECK-LLVM: define linkonce_odr spir_func %"class.sycl::_V1::range.2" @_ZNK4sycl3_V18nd_rangeILi2EE16get_global_rangeEv(%"class.sycl::_V1::nd_range.2" addrspace(4)* noundef align 8 [[VAL_0:%.*]]) #[[FUNCATTRS]] {
// CHECK-LLVM-NEXT:   [[VAL_2:%.*]] = alloca %"class.sycl::_V1::range.2", align 8
// CHECK-LLVM-NEXT:   [[VAL_3:%.*]] = getelementptr %"class.sycl::_V1::nd_range.2", %"class.sycl::_V1::nd_range.2" addrspace(4)* [[VAL_0]], i32 0, i32 0
// CHECK-LLVM-NEXT:   [[VAL_4:%.*]] = addrspacecast %"class.sycl::_V1::range.2"* [[VAL_2]] to %"class.sycl::_V1::range.2" addrspace(4)*
// CHECK-LLVM-NEXT:   call spir_func void @_ZN4sycl3_V15rangeILi2EEC1ERKS2_(%"class.sycl::_V1::range.2" addrspace(4)* [[VAL_4]], %"class.sycl::_V1::range.2" addrspace(4)* [[VAL_3]])
// CHECK-LLVM-NEXT:   [[VAL_5:%.*]]  = load %"class.sycl::_V1::range.2", %"class.sycl::_V1::range.2"* [[VAL_2]], align 8
// CHECK-LLVM-NEXT:   ret %"class.sycl::_V1::range.2" [[VAL_5]] 
// CHECK-LLVM-NEXT: }

SYCL_EXTERNAL void nd_range_get_global_range(sycl::nd_range<2> nd_range) {
  keep(nd_range.get_global_range());
}

// CHECK-MLIR-LABEL: func.func @_Z24nd_range_get_local_rangeN4sycl3_V18nd_rangeILi2EEE(
// CHECK-MLIR:           %{{.*}}: memref<?x!sycl_nd_range_2_> {llvm.align = 8 : i64, llvm.byval = !sycl_nd_range_2_, llvm.noundef})
// CHECK-MLIR: %{{.*}} = "sycl.nd_range.get_local_range"(%{{.*}}) {ArgumentTypes = [memref<?x!sycl_nd_range_2_, 4>], FunctionName = @get_local_range, MangledFunctionName = @_ZNK4sycl3_V18nd_rangeILi2EE15get_local_rangeEv, TypeName = @nd_range} : (!sycl_nd_range_2_) -> !sycl_range_2_

// CHECK-LLVM-LABEL: define spir_func void @_Z24nd_range_get_local_rangeN4sycl3_V18nd_rangeILi2EEE(
// CHECK-LLVM:           %"class.sycl::_V1::nd_range.2"* noundef byval(%"class.sycl::_V1::nd_range.2") align 8 %0) #[[FUNCATTRS]]
// CHECK-LLVM: %{{.*}} = call spir_func %"class.sycl::_V1::range.2" @_ZNK4sycl3_V18nd_rangeILi2EE15get_local_rangeEv(%"class.sycl::_V1::nd_range.2" addrspace(4)* %{{.*}})

SYCL_EXTERNAL void nd_range_get_local_range(sycl::nd_range<2> nd_range) {
  keep(nd_range.get_local_range());
}

// CHECK-MLIR-LABEL: func.func @_Z24nd_range_get_group_rangeN4sycl3_V18nd_rangeILi2EEE(
// CHECK-MLIR:           %{{.*}}: memref<?x!sycl_nd_range_2_> {llvm.align = 8 : i64, llvm.byval = !sycl_nd_range_2_, llvm.noundef})
// CHECK-MLIR: %{{.*}} = "sycl.nd_range.get_group_range"(%{{.*}}) {ArgumentTypes = [memref<?x!sycl_nd_range_2_, 4>], FunctionName = @get_group_range, MangledFunctionName = @_ZNK4sycl3_V18nd_rangeILi2EE15get_group_rangeEv, TypeName = @nd_range} : (!sycl_nd_range_2_) -> !sycl_range_2_

// CHECK-LLVM-LABEL: define spir_func void @_Z24nd_range_get_group_rangeN4sycl3_V18nd_rangeILi2EEE(
// CHECK-LLVM:           %"class.sycl::_V1::nd_range.2"* noundef byval(%"class.sycl::_V1::nd_range.2") align 8 %0) #[[FUNCATTRS]]  
// CHECK-LLVM: %{{.*}} = call spir_func %"class.sycl::_V1::range.2" @_ZNK4sycl3_V18nd_rangeILi2EE15get_group_rangeEv(%"class.sycl::_V1::nd_range.2" addrspace(4)* %{{.*}})

SYCL_EXTERNAL void nd_range_get_group_range(sycl::nd_range<2> nd_range) {
  keep(nd_range.get_group_range());
}

// CHECK-MLIR-LABEL: func.func @_Z8id_get_0N4sycl3_V12idILi1EEEi(
// CHECK-MLIR:           %arg0: memref<?x!sycl_id_1_>  {llvm.align = 8 : i64, llvm.byval = !sycl_id_1_, llvm.noundef}, %arg1: i32  {llvm.noundef})
// CHECK-MLIR: %{{.*}} = "sycl.id.get"(%{{.*}}, %arg1) {ArgumentTypes = [memref<?x!sycl_array_1_, 4>, i32], FunctionName = @get, MangledFunctionName = @_ZNK4sycl3_V16detail5arrayILi1EE3getEi, TypeName = @array} : (!sycl_id_1_, i32) -> i64

// CHECK-LLVM-LABEL: define spir_func void @_Z8id_get_0N4sycl3_V12idILi1EEEi(
// CHECK-LLVM:           %"class.sycl::_V1::id.1"* noundef byval(%"class.sycl::_V1::id.1") align 8 %0, i32 noundef %1) #[[FUNCATTRS]]
// CHECK-LLVM: %{{.*}} = call spir_func i64 @_ZNK4sycl3_V16detail5arrayILi1EE3getEi(%"class.sycl::_V1::detail::array.1" addrspace(4)* %{{.*}}, i32 %1)

SYCL_EXTERNAL void id_get_0(sycl::id<1> id, int dimension) {
  keep(id.get(dimension));
}

// CHECK-MLIR-LABEL: func.func @_Z8id_get_1N4sycl3_V12idILi1EEEi(
// CHECK-MLIR:           %arg0: memref<?x!sycl_id_1_>  {llvm.align = 8 : i64, llvm.byval = !sycl_id_1_, llvm.noundef}, %arg1: i32  {llvm.noundef})  
// CHECK-MLIR: %{{.*}} = "sycl.id.get"(%{{.*}}, %arg1) {ArgumentTypes = [memref<?x!sycl_array_1_, 4>, i32], FunctionName = @"operator[]", MangledFunctionName = @_ZN4sycl3_V16detail5arrayILi1EEixEi, TypeName = @array} : (!sycl_id_1_, i32) -> memref<?xi64, 4>

// CHECK-LLVM-LABEL: define spir_func void @_Z8id_get_1N4sycl3_V12idILi1EEEi(
// CHECK-LLVM:           %"class.sycl::_V1::id.1"* noundef byval(%"class.sycl::_V1::id.1") align 8 %0, i32 noundef %1) #[[FUNCATTRS]]  
// CHECK-LLVM: %{{.*}} = call spir_func i64 addrspace(4)* @_ZN4sycl3_V16detail5arrayILi1EEixEi(%"class.sycl::_V1::detail::array.1" addrspace(4)* %{{.*}}, i32 %1)

SYCL_EXTERNAL void id_get_1(sycl::id<1> id, int dimension) {
  keep(id[dimension]);
}

// CHECK-MLIR-LABEL: func.func @_Z8id_get_2N4sycl3_V12idILi1EEEi(
// CHECK-MLIR:           %arg0: memref<?x!sycl_id_1_>  {llvm.align = 8 : i64, llvm.byval = !sycl_id_1_, llvm.noundef}, %arg1: i32  {llvm.noundef})    
// CHECK-MLIR: %{{.*}} = "sycl.id.get"(%{{.*}}, %arg1) {ArgumentTypes = [memref<?x!sycl_array_1_, 4>, i32], FunctionName = @"operator[]", MangledFunctionName = @_ZNK4sycl3_V16detail5arrayILi1EEixEi, TypeName = @array} : (!sycl_id_1_, i32) -> i64

// CHECK-LLVM-LABEL: define spir_func void @_Z8id_get_2N4sycl3_V12idILi1EEEi(
// CHECK-LLVM:           %"class.sycl::_V1::id.1"* noundef byval(%"class.sycl::_V1::id.1") align 8 %0, i32 noundef %1) #[[FUNCATTRS]]    
// CHECK-LLVM: %{{.*}} = call spir_func i64 @_ZNK4sycl3_V16detail5arrayILi1EEixEi(%"class.sycl::_V1::detail::array.1" addrspace(4)* %{{.*}}, i32 %1)

SYCL_EXTERNAL void id_get_2(const sycl::id<1> id, int dimension) {
  keep(id[dimension]);
}

// CHECK-MLIR-LABEL: func.func @_Z8id_get_3N4sycl3_V12idILi1EEE(
// CHECK-MLIR:           %arg0: memref<?x!sycl_id_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_id_1_, llvm.noundef}
// CHECK-MLIR: %{{.*}} = "sycl.id.get"(%{{.*}}) {ArgumentTypes = [memref<?x!sycl_id_1_, 4>], FunctionName = @"operator unsigned long", MangledFunctionName = @_ZNK4sycl3_V12idILi1EEcvmEv, TypeName = @id} : (!sycl_id_1_) -> i64

SYCL_EXTERNAL void id_get_3(sycl::id<1> id) {
  keep(static_cast<size_t>(id));
}

// CHECK-MLIR-LABEL: func.func @_Z13item_get_id_0N4sycl3_V14itemILi1ELb1EEE(
// CHECK-MLIR:           %arg0: memref<?x![[ITEM1]]> {llvm.align = 8 : i64, llvm.byval = ![[ITEM1]], llvm.noundef})
// CHECK-MLIR: %{{.*}} = "sycl.item.get_id"(%{{.*}}) {ArgumentTypes = [memref<?x![[ITEM1]], 4>], FunctionName = @get_id, MangledFunctionName = @_ZNK4sycl3_V14itemILi1ELb1EE6get_idEv, TypeName = @item} : (![[ITEM1]]) -> !sycl_id_1_

// CHECK-LLVM-LABEL: define spir_func void @_Z13item_get_id_0N4sycl3_V14itemILi1ELb1EEE(
// CHECK-LLVM:           %"class.sycl::_V1::item.1.true"* noundef byval(%"class.sycl::_V1::item.1.true") align 8 %0) #[[FUNCATTRS]]
// CHECK-LLVM: %{{.*}} = call spir_func %"class.sycl::_V1::id.1" @_ZNK4sycl3_V14itemILi1ELb1EE6get_idEv(%"class.sycl::_V1::item.1.true" addrspace(4)* %{{.*}})

SYCL_EXTERNAL void item_get_id_0(sycl::item<1> item) {
  keep(item.get_id());
}

// CHECK-MLIR-LABEL: func.func @_Z13item_get_id_1N4sycl3_V14itemILi1ELb1EEEi(
// CHECK-MLIR:           %arg0: memref<?x![[ITEM1]]> {llvm.align = 8 : i64, llvm.byval = ![[ITEM1]], llvm.noundef}, %arg1: i32 {llvm.noundef})
// CHECK-MLIR: %{{.*}} = "sycl.item.get_id"(%{{.*}}, %arg1) {ArgumentTypes = [memref<?x![[ITEM1]], 4>, i32], FunctionName = @get_id, MangledFunctionName = @_ZNK4sycl3_V14itemILi1ELb1EE6get_idEi, TypeName = @item} : (![[ITEM1]], i32) -> i64

SYCL_EXTERNAL void item_get_id_1(sycl::item<1> item, int dimension) {
  keep(item.get_id(dimension));
}

// CHECK-MLIR-LABEL: func.func @_Z13item_get_id_2N4sycl3_V14itemILi1ELb1EEEi(
// CHECK-MLIR:           %arg0: memref<?x![[ITEM1]]> {llvm.align = 8 : i64, llvm.byval = ![[ITEM1]], llvm.noundef}, %arg1: i32 {llvm.noundef})
// CHECK-MLIR: %{{.*}} = "sycl.item.get_id"(%{{.*}}, %arg1) {ArgumentTypes = [memref<?x![[ITEM1]], 4>, i32], FunctionName = @"operator[]", MangledFunctionName = @_ZNK4sycl3_V14itemILi1ELb1EEixEi, TypeName = @item} : (![[ITEM1]], i32) -> i64

SYCL_EXTERNAL void item_get_id_2(sycl::item<1> item, int dimension) {
  keep(item[dimension]);
}

// CHECK-MLIR-LABEL: func.func @_Z13item_get_id_3N4sycl3_V14itemILi1ELb1EEE(
// CHECK-MLIR:           %arg0: memref<?x![[ITEM1]]> {llvm.align = 8 : i64, llvm.byval = ![[ITEM1]], llvm.noundef})
// CHECK-MLIR: %{{.*}} = "sycl.item.get_id"(%{{.*}}) {ArgumentTypes = [memref<?x![[ITEM1]], 4>], FunctionName = @"operator unsigned long", MangledFunctionName = @_ZNK4sycl3_V14itemILi1ELb1EEcvmEv, TypeName = @item} : (![[ITEM1]]) -> i64

// CHECK-LLVM-LABEL: define spir_func void @_Z13item_get_id_3N4sycl3_V14itemILi1ELb1EEE(
// CHECK-LLVM:           %"class.sycl::_V1::item.1.true"* noundef byval(%"class.sycl::_V1::item.1.true") align 8 %0) #[[FUNCATTRS]]  
// CHECK-LLVM: %{{.*}} = call spir_func i64 @_ZNK4sycl3_V14itemILi1ELb1EEcvmEv(%"class.sycl::_V1::item.1.true" addrspace(4)* %{{.*}})

SYCL_EXTERNAL void item_get_id_3(sycl::item<1> item) {
  keep(static_cast<size_t>(item));
}

// CHECK-MLIR-LABEL: func.func @_Z16item_get_range_0N4sycl3_V14itemILi1ELb1EEE(
// CHECK-MLIR:           %arg0: memref<?x![[ITEM1]]> {llvm.align = 8 : i64, llvm.byval = ![[ITEM1]], llvm.noundef})
// CHECK-MLIR: %{{.*}} = "sycl.item.get_range"(%{{.*}}) {ArgumentTypes = [memref<?x![[ITEM1]], 4>], FunctionName = @get_range, MangledFunctionName = @_ZNK4sycl3_V14itemILi1ELb1EE9get_rangeEv, TypeName = @item} : (![[ITEM1]]) -> !sycl_range_1_

// CHECK-LLVM-LABEL: define spir_func void @_Z16item_get_range_0N4sycl3_V14itemILi1ELb1EEE(
// CHECK-LLVM:           %"class.sycl::_V1::item.1.true"* noundef byval(%"class.sycl::_V1::item.1.true") align 8 %0) #[[FUNCATTRS]]    
// CHECK-LLVM: %{{.*}} = call spir_func %"class.sycl::_V1::range.1" @_ZNK4sycl3_V14itemILi1ELb1EE9get_rangeEv(%"class.sycl::_V1::item.1.true" addrspace(4)* %{{.*}})

SYCL_EXTERNAL void item_get_range_0(sycl::item<1> item) {
  keep(item.get_range());
}

// CHECK-MLIR-LABEL: func.func @_Z16item_get_range_1N4sycl3_V14itemILi1ELb1EEEi(
// CHECK-MLIR:           %arg0: memref<?x![[ITEM1]]> {llvm.align = 8 : i64, llvm.byval = ![[ITEM1]], llvm.noundef}, %arg1: i32 {llvm.noundef})
// CHECK-MLIR: %{{.*}} = "sycl.item.get_range"(%{{.*}}, %arg1) {ArgumentTypes = [memref<?x![[ITEM1]], 4>, i32], FunctionName = @get_range, MangledFunctionName = @_ZNK4sycl3_V14itemILi1ELb1EE9get_rangeEi, TypeName = @item} : (![[ITEM1]], i32) -> i64

SYCL_EXTERNAL void item_get_range_1(sycl::item<1> item, int dimension) {
  keep(item.get_range(dimension));
}

// CHECK-MLIR-LABEL: func.func @_Z18item_get_linear_idN4sycl3_V14itemILi1ELb1EEE(
// CHECK-MLIR:           %arg0: memref<?x![[ITEM1]]> {llvm.align = 8 : i64, llvm.byval = ![[ITEM1]], llvm.noundef})
// CHECK-MLIR: %{{.*}} = "sycl.item.get_linear_id"(%{{.*}}) {ArgumentTypes = [memref<?x![[ITEM1]], 4>], FunctionName = @get_linear_id, MangledFunctionName = @_ZNK4sycl3_V14itemILi1ELb1EE13get_linear_idEv, TypeName = @item} : (![[ITEM1]]) -> i64

SYCL_EXTERNAL void item_get_linear_id(sycl::item<1> item) {
  keep(item.get_linear_id());
}

// CHECK-MLIR-LABEL: func.func @_Z23nd_item_get_global_id_0N4sycl3_V17nd_itemILi1EEE(
// CHECK-MLIR:           %arg0: memref<?x!sycl_nd_item_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_nd_item_1_, llvm.noundef})
// CHECK-MLIR: %{{.*}} = "sycl.nd_item.get_global_id"(%{{.*}}) {ArgumentTypes = [memref<?x!sycl_nd_item_1_, 4>], FunctionName = @get_global_id, MangledFunctionName = @_ZNK4sycl3_V17nd_itemILi1EE13get_global_idEv, TypeName = @nd_item} : (!sycl_nd_item_1_) -> !sycl_id_1_

// CHECK-LLVM-LABEL: define spir_func void @_Z23nd_item_get_global_id_0N4sycl3_V17nd_itemILi1EEE(
// CHECK-LLVM:           %"class.sycl::_V1::nd_item.1"* noundef byval(%"class.sycl::_V1::nd_item.1") align 8 %0) #[[FUNCATTRS]]
// CHECK-LLVM: %{{.*}} = call spir_func %"class.sycl::_V1::id.1" @_ZNK4sycl3_V17nd_itemILi1EE13get_global_idEv(%"class.sycl::_V1::nd_item.1" addrspace(4)* %{{.*}})

SYCL_EXTERNAL void nd_item_get_global_id_0(sycl::nd_item<1> nd_item) {
  keep(nd_item.get_global_id());
}

// CHECK-MLIR-LABEL: func.func @_Z23nd_item_get_global_id_1N4sycl3_V17nd_itemILi1EEEi(
// CHECK-MLIR:           %arg0: memref<?x!sycl_nd_item_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_nd_item_1_, llvm.noundef}, %arg1: i32 {llvm.noundef})
// CHECK-MLIR: %{{.*}} = "sycl.nd_item.get_global_id"(%{{.*}}, %arg1) {ArgumentTypes = [memref<?x!sycl_nd_item_1_, 4>, i32], FunctionName = @get_global_id, MangledFunctionName = @_ZNK4sycl3_V17nd_itemILi1EE13get_global_idEi, TypeName = @nd_item} : (!sycl_nd_item_1_, i32) -> i64

SYCL_EXTERNAL void nd_item_get_global_id_1(sycl::nd_item<1> nd_item, int dimension) {
  keep(nd_item.get_global_id(dimension));
}

// CHECK-MLIR-LABEL: func.func @_Z28nd_item_get_global_linear_idN4sycl3_V17nd_itemILi1EEE(
// CHECK-MLIR:           %arg0: memref<?x!sycl_nd_item_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_nd_item_1_, llvm.noundef})
// CHECK-MLIR: %{{.*}} = "sycl.nd_item.get_global_linear_id"(%{{.*}}) {ArgumentTypes = [memref<?x!sycl_nd_item_1_, 4>], FunctionName = @get_global_linear_id, MangledFunctionName = @_ZNK4sycl3_V17nd_itemILi1EE20get_global_linear_idEv, TypeName = @nd_item} : (!sycl_nd_item_1_) -> i64

SYCL_EXTERNAL void nd_item_get_global_linear_id(sycl::nd_item<1> nd_item) {
  keep(nd_item.get_global_linear_id());
}

// CHECK-MLIR-LABEL: func.func @_Z22nd_item_get_local_id_0N4sycl3_V17nd_itemILi1EEE(
// CHECK-MLIR:           %arg0: memref<?x!sycl_nd_item_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_nd_item_1_, llvm.noundef})
// CHECK-MLIR: %{{.*}} = "sycl.nd_item.get_local_id"(%{{.*}}) {ArgumentTypes = [memref<?x!sycl_nd_item_1_, 4>], FunctionName = @get_local_id, MangledFunctionName = @_ZNK4sycl3_V17nd_itemILi1EE12get_local_idEv, TypeName = @nd_item} : (!sycl_nd_item_1_) -> !sycl_id_1_

// CHECK-LLVM-LABEL: define spir_func void @_Z22nd_item_get_local_id_0N4sycl3_V17nd_itemILi1EEE(
// CHECK-LLVM:           %"class.sycl::_V1::nd_item.1"* noundef byval(%"class.sycl::_V1::nd_item.1") align 8 %0) #[[FUNCATTRS]]  
// CHECK-LLVM: %{{.*}} = call spir_func %"class.sycl::_V1::id.1" @_ZNK4sycl3_V17nd_itemILi1EE12get_local_idEv(%"class.sycl::_V1::nd_item.1" addrspace(4)* %{{.*}})

SYCL_EXTERNAL void nd_item_get_local_id_0(sycl::nd_item<1> nd_item) {
  keep(nd_item.get_local_id());
}

// CHECK-MLIR-LABEL: func.func @_Z22nd_item_get_local_id_1N4sycl3_V17nd_itemILi1EEEi(
// CHECK-MLIR:           %arg0: memref<?x!sycl_nd_item_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_nd_item_1_, llvm.noundef}, %arg1: i32 {llvm.noundef})
// CHECK-MLIR: %{{.*}} = "sycl.nd_item.get_local_id"(%{{.*}}, %arg1) {ArgumentTypes = [memref<?x!sycl_nd_item_1_, 4>, i32], FunctionName = @get_local_id, MangledFunctionName = @_ZNK4sycl3_V17nd_itemILi1EE12get_local_idEi, TypeName = @nd_item} : (!sycl_nd_item_1_, i32) -> i64

SYCL_EXTERNAL void nd_item_get_local_id_1(sycl::nd_item<1> nd_item, int dimension) {
  keep(nd_item.get_local_id(dimension));
}

// CHECK-MLIR-LABEL: func.func @_Z27nd_item_get_local_linear_idN4sycl3_V17nd_itemILi1EEE(
// CHECK-MLIR:           %arg0: memref<?x!sycl_nd_item_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_nd_item_1_, llvm.noundef})
// CHECK-MLIR: %{{.*}} = "sycl.nd_item.get_local_linear_id"(%{{.*}}) {ArgumentTypes = [memref<?x!sycl_nd_item_1_, 4>], FunctionName = @get_local_linear_id, MangledFunctionName = @_ZNK4sycl3_V17nd_itemILi1EE19get_local_linear_idEv, TypeName = @nd_item} : (!sycl_nd_item_1_) -> i64

// CHECK-LLVM-LABEL: define spir_func void @_Z27nd_item_get_local_linear_idN4sycl3_V17nd_itemILi1EEE(
// CHECK-LLVM:           %"class.sycl::_V1::nd_item.1"* noundef byval(%"class.sycl::_V1::nd_item.1") align 8 %0) #[[FUNCATTRS]]    
// CHECK-LLVM: %{{.*}} = call spir_func i64 @_ZNK4sycl3_V17nd_itemILi1EE19get_local_linear_idEv(%"class.sycl::_V1::nd_item.1" addrspace(4)* %{{.*}})

SYCL_EXTERNAL void nd_item_get_local_linear_id(sycl::nd_item<1> nd_item) {
  keep(nd_item.get_local_linear_id());
}

// CHECK-MLIR-LABEL: func.func @_Z19nd_item_get_group_0N4sycl3_V17nd_itemILi1EEE(
// CHECK-MLIR:           %arg0: memref<?x!sycl_nd_item_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_nd_item_1_, llvm.noundef})  
// CHECK-MLIR: %{{.*}} = "sycl.nd_item.get_group"(%{{.*}}) {ArgumentTypes = [memref<?x!sycl_nd_item_1_, 4>], FunctionName = @get_group, MangledFunctionName = @_ZNK4sycl3_V17nd_itemILi1EE9get_groupEv, TypeName = @nd_item} : (!sycl_nd_item_1_) -> !sycl_group_1_

// CHECK-LLVM-LABEL: define spir_func void @_Z19nd_item_get_group_0N4sycl3_V17nd_itemILi1EEE(
// CHECK-LLVM:           %"class.sycl::_V1::nd_item.1"* noundef byval(%"class.sycl::_V1::nd_item.1") align 8 %0) #[[FUNCATTRS]]      
// CHECK-LLVM: %{{.*}} = call spir_func %"class.sycl::_V1::group.1" @_ZNK4sycl3_V17nd_itemILi1EE9get_groupEv(%"class.sycl::_V1::nd_item.1" addrspace(4)* %{{.*}})

SYCL_EXTERNAL void nd_item_get_group_0(sycl::nd_item<1> nd_item) {
  keep(nd_item.get_group());
}

// CHECK-MLIR-LABEL: func.func @_Z19nd_item_get_group_1N4sycl3_V17nd_itemILi1EEEi(
// CHECK-MLIR:           %arg0: memref<?x!sycl_nd_item_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_nd_item_1_, llvm.noundef}, %arg1: i32 {llvm.noundef})
// CHECK-MLIR: %{{.*}} = "sycl.nd_item.get_group"(%{{.*}}, %arg1) {ArgumentTypes = [memref<?x!sycl_nd_item_1_, 4>, i32], FunctionName = @get_group, MangledFunctionName = @_ZNK4sycl3_V17nd_itemILi1EE9get_groupEi, TypeName = @nd_item} : (!sycl_nd_item_1_, i32) -> i64

SYCL_EXTERNAL void nd_item_get_group_1(sycl::nd_item<1> nd_item, int dimension) {
  keep(nd_item.get_group(dimension));
}

// CHECK-MLIR-LABEL: func.func @_Z27nd_item_get_group_linear_idN4sycl3_V17nd_itemILi1EEE(
// CHECK-MLIR:           %arg0: memref<?x!sycl_nd_item_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_nd_item_1_, llvm.noundef})
// CHECK-MLIR: %{{.*}} = "sycl.nd_item.get_group_linear_id"(%{{.*}}) {ArgumentTypes = [memref<?x!sycl_nd_item_1_, 4>], FunctionName = @get_group_linear_id, MangledFunctionName = @_ZNK4sycl3_V17nd_itemILi1EE19get_group_linear_idEv, TypeName = @nd_item} : (!sycl_nd_item_1_) -> i64

SYCL_EXTERNAL void nd_item_get_group_linear_id(sycl::nd_item<1> nd_item) {
  keep(nd_item.get_group_linear_id());
}

// CHECK-MLIR-LABEL: func.func @_Z25nd_item_get_group_range_0N4sycl3_V17nd_itemILi1EEE(
// CHECK-MLIR:           %arg0: memref<?x!sycl_nd_item_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_nd_item_1_, llvm.noundef})
// CHECK-MLIR: %{{.*}} = "sycl.nd_item.get_group_range"(%{{.*}}) {ArgumentTypes = [memref<?x!sycl_nd_item_1_, 4>], FunctionName = @get_group_range, MangledFunctionName = @_ZNK4sycl3_V17nd_itemILi1EE15get_group_rangeEv, TypeName = @nd_item} : (!sycl_nd_item_1_) -> !sycl_range_1_

// CHECK-LLVM-LABEL: define spir_func void @_Z25nd_item_get_group_range_0N4sycl3_V17nd_itemILi1EEE(
// CHECK-LLVM:           %"class.sycl::_V1::nd_item.1"* noundef byval(%"class.sycl::_V1::nd_item.1") align 8 %0) #[[FUNCATTRS]]        
// CHECK-LLVM: %{{.*}} = call spir_func %"class.sycl::_V1::range.1" @_ZNK4sycl3_V17nd_itemILi1EE15get_group_rangeEv(%"class.sycl::_V1::nd_item.1" addrspace(4)* %{{.*}})

SYCL_EXTERNAL void nd_item_get_group_range_0(sycl::nd_item<1> nd_item) {
  keep(nd_item.get_group_range());
}

// CHECK-MLIR-LABEL: func.func @_Z25nd_item_get_group_range_1N4sycl3_V17nd_itemILi1EEEi(
// CHECK-MLIR:           %arg0: memref<?x!sycl_nd_item_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_nd_item_1_, llvm.noundef}, %arg1: i32 {llvm.noundef})
// CHECK-MLIR: %{{.*}} = "sycl.nd_item.get_group_range"(%{{.*}}, %arg1) {ArgumentTypes = [memref<?x!sycl_nd_item_1_, 4>, i32], FunctionName = @get_group_range, MangledFunctionName = @_ZNK4sycl3_V17nd_itemILi1EE15get_group_rangeEi, TypeName = @nd_item} : (!sycl_nd_item_1_, i32) -> i64

SYCL_EXTERNAL void nd_item_get_group_range_1(sycl::nd_item<1> nd_item, int dimension) {
  keep(nd_item.get_group_range(dimension));
}

// CHECK-MLIR-LABEL: func.func @_Z26nd_item_get_global_range_0N4sycl3_V17nd_itemILi1EEE(
// CHECK-MLIR:           %arg0: memref<?x!sycl_nd_item_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_nd_item_1_, llvm.noundef})
// CHECK-MLIR: %{{.*}} = "sycl.nd_item.get_global_range"(%{{.*}}) {ArgumentTypes = [memref<?x!sycl_nd_item_1_, 4>], FunctionName = @get_global_range, MangledFunctionName = @_ZNK4sycl3_V17nd_itemILi1EE16get_global_rangeEv, TypeName = @nd_item} : (!sycl_nd_item_1_) -> !sycl_range_1_

// CHECK-LLVM-LABEL: define spir_func void @_Z26nd_item_get_global_range_0N4sycl3_V17nd_itemILi1EEE(
// CHECK-LLVM:           %"class.sycl::_V1::nd_item.1"* noundef byval(%"class.sycl::_V1::nd_item.1") align 8 %0) #[[FUNCATTRS]]
// CHECK-LLVM: %{{.*}} = call spir_func %"class.sycl::_V1::range.1" @_ZNK4sycl3_V17nd_itemILi1EE16get_global_rangeEv(%"class.sycl::_V1::nd_item.1" addrspace(4)* %{{.*}})

SYCL_EXTERNAL void nd_item_get_global_range_0(sycl::nd_item<1> nd_item) {
  keep(nd_item.get_global_range());
}

// CHECK-MLIR-LABEL: func.func @_Z26nd_item_get_global_range_1N4sycl3_V17nd_itemILi1EEEi(
// CHECK-MLIR:           %arg0: memref<?x!sycl_nd_item_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_nd_item_1_, llvm.noundef}, %arg1: i32 {llvm.noundef})
// CHECK-MLIR: %{{.*}} = "sycl.nd_item.get_global_range"(%{{.*}}, %arg1) {ArgumentTypes = [memref<?x!sycl_nd_item_1_, 4>, i32], FunctionName = @get_global_range, MangledFunctionName = @_ZNK4sycl3_V17nd_itemILi1EE16get_global_rangeEi, TypeName = @nd_item} : (!sycl_nd_item_1_, i32) -> i64

// CHECK-LLVM-LABEL: define spir_func void @_Z26nd_item_get_global_range_1N4sycl3_V17nd_itemILi1EEEi(
// CHECK-LLVM:           %"class.sycl::_V1::nd_item.1"* noundef byval(%"class.sycl::_V1::nd_item.1") align 8 %0, i32 noundef %1) #[[FUNCATTRS]]      
// CHECK-LLVM: %{{.*}} = call spir_func i64 @_ZNK4sycl3_V17nd_itemILi1EE16get_global_rangeEi(%"class.sycl::_V1::nd_item.1" addrspace(4)* %{{.*}}, i32 %1)

SYCL_EXTERNAL void nd_item_get_global_range_1(sycl::nd_item<1> nd_item, int dimension) {
  keep(nd_item.get_global_range(dimension));
}

// CHECK-MLIR-LABEL: func.func @_Z25nd_item_get_local_range_0N4sycl3_V17nd_itemILi1EEE(
// CHECK-MLIR:           %arg0: memref<?x!sycl_nd_item_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_nd_item_1_, llvm.noundef})
// CHECK-MLIR: %{{.*}} = "sycl.nd_item.get_local_range"(%{{.*}}) {ArgumentTypes = [memref<?x!sycl_nd_item_1_, 4>], FunctionName = @get_local_range, MangledFunctionName = @_ZNK4sycl3_V17nd_itemILi1EE15get_local_rangeEv, TypeName = @nd_item} : (!sycl_nd_item_1_) -> !sycl_range_1_

// CHECK-LLVM-LABEL: define spir_func void @_Z25nd_item_get_local_range_0N4sycl3_V17nd_itemILi1EEE(
// CHECK-LLVM:           %"class.sycl::_V1::nd_item.1"* noundef byval(%"class.sycl::_V1::nd_item.1") align 8 %0) #[[FUNCATTRS]]
// CHECK-LLVM: %{{.*}} = call spir_func %"class.sycl::_V1::range.1" @_ZNK4sycl3_V17nd_itemILi1EE15get_local_rangeEv(%"class.sycl::_V1::nd_item.1" addrspace(4)* %{{.*}})

SYCL_EXTERNAL void nd_item_get_local_range_0(sycl::nd_item<1> nd_item) {
  keep(nd_item.get_local_range());
}

// CHECK-MLIR-LABEL: func.func @_Z25nd_item_get_local_range_1N4sycl3_V17nd_itemILi1EEEi(
// CHECK-MLIR:           %arg0: memref<?x!sycl_nd_item_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_nd_item_1_, llvm.noundef}, %arg1: i32 {llvm.noundef})
// CHECK-MLIR: %{{.*}} = "sycl.nd_item.get_local_range"(%{{.*}}, %arg1) {ArgumentTypes = [memref<?x!sycl_nd_item_1_, 4>, i32], FunctionName = @get_local_range, MangledFunctionName = @_ZNK4sycl3_V17nd_itemILi1EE15get_local_rangeEi, TypeName = @nd_item} : (!sycl_nd_item_1_, i32) -> i64

// CHECK-LLVM-LABEL: define spir_func void @_Z25nd_item_get_local_range_1N4sycl3_V17nd_itemILi1EEEi(
// CHECK-LLVM:           %"class.sycl::_V1::nd_item.1"* noundef byval(%"class.sycl::_V1::nd_item.1") align 8 %0, i32 noundef %1) #[[FUNCATTRS]]        
// CHECK-LLVM: %{{.*}} = call spir_func i64 @_ZNK4sycl3_V17nd_itemILi1EE15get_local_rangeEi(%"class.sycl::_V1::nd_item.1" addrspace(4)* %{{.*}}, i32 %1)

SYCL_EXTERNAL void nd_item_get_local_range_1(sycl::nd_item<1> nd_item, int dimension) {
  keep(nd_item.get_local_range(dimension));
}

// CHECK-MLIR-LABEL: func.func @_Z20nd_item_get_nd_rangeN4sycl3_V17nd_itemILi1EEE(
// CHECK-MLIR:           %arg0: memref<?x!sycl_nd_item_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_nd_item_1_, llvm.noundef})
// CHECK-MLIR: %{{.*}} = "sycl.nd_item.get_nd_range"(%{{.*}}) {ArgumentTypes = [memref<?x!sycl_nd_item_1_, 4>], FunctionName = @get_nd_range, MangledFunctionName = @_ZNK4sycl3_V17nd_itemILi1EE12get_nd_rangeEv, TypeName = @nd_item} : (!sycl_nd_item_1_) -> !sycl_nd_range_1_

// CHECK-LLVM-LABEL: define spir_func void @_Z20nd_item_get_nd_rangeN4sycl3_V17nd_itemILi1EEE(
// CHECK-LLVM:           %"class.sycl::_V1::nd_item.1"* noundef byval(%"class.sycl::_V1::nd_item.1") align 8 %0) #[[FUNCATTRS]]          
// CHECK-LLVM: %{{.*}} = call spir_func %"class.sycl::_V1::nd_range.1" @_ZNK4sycl3_V17nd_itemILi1EE12get_nd_rangeEv(%"class.sycl::_V1::nd_item.1" addrspace(4)* %{{.*}})

SYCL_EXTERNAL void nd_item_get_nd_range(sycl::nd_item<1> nd_item) {
  keep(nd_item.get_nd_range());
}

// CHECK-MLIR-LABEL: func.func @_Z20group_get_group_id_0N4sycl3_V15groupILi1EEE(
// CHECK-MLIR:           %arg0: memref<?x!sycl_group_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_group_1_, llvm.noundef})
// CHECK-MLIR: %{{.*}} = "sycl.group.get_group_id"(%{{.*}}) {ArgumentTypes = [memref<?x!sycl_group_1_, 4>], FunctionName = @get_group_id, MangledFunctionName = @_ZNK4sycl3_V15groupILi1EE12get_group_idEv, TypeName = @group} : (!sycl_group_1_) -> !sycl_id_1_

// CHECK-LLVM-LABEL: define spir_func void @_Z20group_get_group_id_0N4sycl3_V15groupILi1EEE(

// CHECK-LLVM: %{{.*}} = call spir_func %"class.sycl::_V1::id.1" @_ZNK4sycl3_V15groupILi1EE12get_group_idEv(%"class.sycl::_V1::group.1" addrspace(4)* %{{.*}})

SYCL_EXTERNAL void group_get_group_id_0(sycl::group<1> group) {
  keep(group.get_group_id());
}

// CHECK-MLIR-LABEL: func.func @_Z20group_get_group_id_1N4sycl3_V15groupILi1EEEi(
// CHECK-MLIR:           %arg0: memref<?x!sycl_group_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_group_1_, llvm.noundef}, %arg1: i32 {llvm.noundef})
// CHECK-MLIR: %{{.*}} = "sycl.group.get_group_id"(%{{.*}}, %arg1) {ArgumentTypes = [memref<?x!sycl_group_1_, 4>, i32], FunctionName = @get_group_id, MangledFunctionName = @_ZNK4sycl3_V15groupILi1EE12get_group_idEi, TypeName = @group} : (!sycl_group_1_, i32) -> i64

// CHECK-LLVM-LABEL: define spir_func void @_Z20group_get_group_id_1N4sycl3_V15groupILi1EEEi(
// CHECK-LLVM:           %"class.sycl::_V1::group.1"* noundef byval(%"class.sycl::_V1::group.1") align 8 %0, i32 noundef %1) #[[FUNCATTRS]]
// CHECK-LLVM: %{{.*}} = call spir_func i64 @_ZNK4sycl3_V15groupILi1EE12get_group_idEi(%"class.sycl::_V1::group.1" addrspace(4)* %{{.*}}, i32 %1)

SYCL_EXTERNAL void group_get_group_id_1(sycl::group<1> group, int dimension) {
  keep(group.get_group_id(dimension));
}

// CHECK-MLIR-LABEL: func.func @_Z20group_get_group_id_2N4sycl3_V15groupILi1EEEi(
// CHECK-MLIR:           %arg0: memref<?x!sycl_group_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_group_1_, llvm.noundef}, %arg1: i32 {llvm.noundef})
// CHECK-MLIR: %{{.*}} = "sycl.group.get_group_id"(%{{.*}}, %arg1) {ArgumentTypes = [memref<?x!sycl_group_1_, 4>, i32], FunctionName = @"operator[]", MangledFunctionName = @_ZNK4sycl3_V15groupILi1EEixEi, TypeName = @group} : (!sycl_group_1_, i32) -> i64

// CHECK-LLVM-LABEL: define spir_func void @_Z20group_get_group_id_2N4sycl3_V15groupILi1EEEi(
// CHECK-LLVM:           %"class.sycl::_V1::group.1"* noundef byval(%"class.sycl::_V1::group.1") align 8 %0, i32 noundef %1) #[[FUNCATTRS]]
// CHECK-LLVM: %{{.*}} = call spir_func i64 @_ZNK4sycl3_V15groupILi1EEixEi(%"class.sycl::_V1::group.1" addrspace(4)* %{{.*}}, i32 %1)

SYCL_EXTERNAL void group_get_group_id_2(sycl::group<1> group, int dimension) {
  keep(group[dimension]);
}

// CHECK-MLIR-LABEL: func.func @_Z20group_get_local_id_0N4sycl3_V15groupILi1EEE(
// CHECK-MLIR:           %arg0: memref<?x!sycl_group_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_group_1_, llvm.noundef})
// CHECK-MLIR: %{{.*}} = "sycl.group.get_local_id"(%{{.*}}) {ArgumentTypes = [memref<?x!sycl_group_1_, 4>], FunctionName = @get_local_id, MangledFunctionName = @_ZNK4sycl3_V15groupILi1EE12get_local_idEv, TypeName = @group} : (!sycl_group_1_) -> !sycl_id_1_

// CHECK-LLVM-LABEL: define spir_func void @_Z20group_get_local_id_0N4sycl3_V15groupILi1EEE(
// CHECK-LLVM:           %"class.sycl::_V1::group.1"* noundef byval(%"class.sycl::_V1::group.1") align 8 %0) #[[FUNCATTRS]]   
// CHECK-LLVM: %{{.*}} = call spir_func %"class.sycl::_V1::id.1" @_ZNK4sycl3_V15groupILi1EE12get_local_idEv(%"class.sycl::_V1::group.1" addrspace(4)* %{{.*}})

SYCL_EXTERNAL void group_get_local_id_0(sycl::group<1> group) {
  keep(group.get_local_id());
}

// CHECK-MLIR-LABEL: func.func @_Z20group_get_local_id_1N4sycl3_V15groupILi1EEEi(
// CHECK-MLIR:           %arg0: memref<?x!sycl_group_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_group_1_, llvm.noundef}, %arg1: i32 {llvm.noundef})
// CHECK-MLIR: %{{.*}} = "sycl.group.get_local_id"(%{{.*}}, %arg1) {ArgumentTypes = [memref<?x!sycl_group_1_, 4>, i32], FunctionName = @get_local_id, MangledFunctionName = @_ZNK4sycl3_V15groupILi1EE12get_local_idEi, TypeName = @group} : (!sycl_group_1_, i32) -> i64

// CHECK-LLVM-LABEL: define spir_func void @_Z20group_get_local_id_1N4sycl3_V15groupILi1EEEi(
// CHECK-LLVM:           %"class.sycl::_V1::group.1"* noundef byval(%"class.sycl::_V1::group.1") align 8 %0, i32 noundef %1) #[[FUNCATTRS]]  
// CHECK-LLVM: %{{.*}} = call spir_func i64 @_ZNK4sycl3_V15groupILi1EE12get_local_idEi(%"class.sycl::_V1::group.1" addrspace(4)* %{{.*}}, i32 %1)

SYCL_EXTERNAL void group_get_local_id_1(sycl::group<1> group, int dimension) {
  keep(group.get_local_id(dimension));
}

// CHECK-MLIR-LABEL: func.func @_Z23group_get_local_range_0N4sycl3_V15groupILi1EEE(
// CHECK-MLIR:           %arg0: memref<?x!sycl_group_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_group_1_, llvm.noundef}
// CHECK-MLIR: %{{.*}} = "sycl.group.get_local_range"(%{{.*}}) {ArgumentTypes = [memref<?x!sycl_group_1_, 4>], FunctionName = @get_local_range, MangledFunctionName = @_ZNK4sycl3_V15groupILi1EE15get_local_rangeEv, TypeName = @group} : (!sycl_group_1_) -> !sycl_range_1_

// CHECK-LLVM-LABEL: define spir_func void @_Z23group_get_local_range_0N4sycl3_V15groupILi1EEE(
// CHECK-LLVM:           %"class.sycl::_V1::group.1"* noundef byval(%"class.sycl::_V1::group.1") align 8 %0) #[[FUNCATTRS]]     
// CHECK-LLVM: %{{.*}} = call spir_func %"class.sycl::_V1::range.1" @_ZNK4sycl3_V15groupILi1EE15get_local_rangeEv(%"class.sycl::_V1::group.1" addrspace(4)* %{{.*}})

SYCL_EXTERNAL void group_get_local_range_0(sycl::group<1> group) {
  keep(group.get_local_range());
}

// CHECK-MLIR-LABEL: func.func @_Z23group_get_local_range_1N4sycl3_V15groupILi1EEEi(
// CHECK-MLIR:           %arg0: memref<?x!sycl_group_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_group_1_, llvm.noundef}, %arg1: i32 {llvm.noundef})
// CHECK-MLIR: %{{.*}} = "sycl.group.get_local_range"(%{{.*}}, %arg1) {ArgumentTypes = [memref<?x!sycl_group_1_, 4>, i32], FunctionName = @get_local_range, MangledFunctionName = @_ZNK4sycl3_V15groupILi1EE15get_local_rangeEi, TypeName = @group} : (!sycl_group_1_, i32) -> i64

// CHECK-LLVM-LABEL: define spir_func void @_Z23group_get_local_range_1N4sycl3_V15groupILi1EEEi(
// CHECK-LLVM:           %"class.sycl::_V1::group.1"* noundef byval(%"class.sycl::_V1::group.1") align 8 %0, i32 noundef %1) #[[FUNCATTRS]]    
// CHECK-LLVM: %{{.*}} = call spir_func i64 @_ZNK4sycl3_V15groupILi1EE15get_local_rangeEi(%"class.sycl::_V1::group.1" addrspace(4)* %{{.*}}, i32 %1)

SYCL_EXTERNAL void group_get_local_range_1(sycl::group<1> group, int dimension) {
  keep(group.get_local_range(dimension));
}

// CHECK-MLIR-LABEL: func.func @_Z23group_get_group_range_0N4sycl3_V15groupILi1EEE(
// CHECK-MLIR:           %arg0: memref<?x!sycl_group_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_group_1_, llvm.noundef}  
// CHECK-MLIR: %{{.*}} = "sycl.group.get_group_range"(%{{.*}}) {ArgumentTypes = [memref<?x!sycl_group_1_, 4>], FunctionName = @get_group_range, MangledFunctionName = @_ZNK4sycl3_V15groupILi1EE15get_group_rangeEv, TypeName = @group} : (!sycl_group_1_) -> !sycl_range_1_

// CHECK-LLVM-LABEL: define spir_func void @_Z23group_get_group_range_0N4sycl3_V15groupILi1EEE(
// CHECK-LLVM:           %"class.sycl::_V1::group.1"* noundef byval(%"class.sycl::_V1::group.1") align 8 %0) #[[FUNCATTRS]]       
// CHECK-LLVM: %{{.*}} = call spir_func %"class.sycl::_V1::range.1" @_ZNK4sycl3_V15groupILi1EE15get_group_rangeEv(%"class.sycl::_V1::group.1" addrspace(4)* %{{.*}})

SYCL_EXTERNAL void group_get_group_range_0(sycl::group<1> group) {
  keep(group.get_group_range());
}

// CHECK-MLIR-LABEL: func.func @_Z23group_get_group_range_1N4sycl3_V15groupILi1EEEi(
// CHECK-MLIR:           %arg0: memref<?x!sycl_group_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_group_1_, llvm.noundef}    
// CHECK-MLIR: %{{.*}} = "sycl.group.get_group_range"(%{{.*}}, %arg1) {ArgumentTypes = [memref<?x!sycl_group_1_, 4>, i32], FunctionName = @get_group_range, MangledFunctionName = @_ZNK4sycl3_V15groupILi1EE15get_group_rangeEi, TypeName = @group} : (!sycl_group_1_, i32) -> i64

// CHECK-LLVM-LABEL: define spir_func void @_Z23group_get_group_range_1N4sycl3_V15groupILi1EEEi(
// CHECK-LLVM:           %"class.sycl::_V1::group.1"* noundef byval(%"class.sycl::_V1::group.1") align 8 %0, i32 noundef %1) #[[FUNCATTRS]]      
// CHECK-LLVM: %{{.*}} = call spir_func i64 @_ZNK4sycl3_V15groupILi1EE15get_group_rangeEi(%"class.sycl::_V1::group.1" addrspace(4)* %{{.*}}, i32 %1)

SYCL_EXTERNAL void group_get_group_range_1(sycl::group<1> group, int dimension) {
  keep(group.get_group_range(dimension));
}

// CHECK-MLIR-LABEL: func.func @_Z25group_get_max_local_rangeN4sycl3_V15groupILi1EEE(
// CHECK-MLIR:           %arg0: memref<?x!sycl_group_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_group_1_, llvm.noundef}      
// CHECK-MLIR: %{{.*}} = "sycl.group.get_max_local_range"(%{{.*}}) {ArgumentTypes = [memref<?x!sycl_group_1_, 4>], FunctionName = @get_max_local_range, MangledFunctionName = @_ZNK4sycl3_V15groupILi1EE19get_max_local_rangeEv, TypeName = @group} : (!sycl_group_1_) -> !sycl_range_1_

// CHECK-LLVM-LABEL: define spir_func void @_Z25group_get_max_local_rangeN4sycl3_V15groupILi1EEE(
// CHECK-LLVM:           %"class.sycl::_V1::group.1"* noundef byval(%"class.sycl::_V1::group.1") align 8 %0) #[[FUNCATTRS]]         
// CHECK-LLVM: %{{.*}} = call spir_func %"class.sycl::_V1::range.1" @_ZNK4sycl3_V15groupILi1EE19get_max_local_rangeEv(%"class.sycl::_V1::group.1" addrspace(4)* %{{.*}})

SYCL_EXTERNAL void group_get_max_local_range(sycl::group<1> group) {
  keep(group.get_max_local_range());
}

// CHECK-MLIR-LABEL: func.func @_Z25group_get_group_linear_idN4sycl3_V15groupILi1EEE(
// CHECK-MLIR:           %arg0: memref<?x!sycl_group_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_group_1_, llvm.noundef}        
// CHECK-MLIR: %{{.*}} = "sycl.group.get_group_linear_id"(%{{.*}}) {ArgumentTypes = [memref<?x!sycl_group_1_, 4>], FunctionName = @get_group_linear_id, MangledFunctionName = @_ZNK4sycl3_V15groupILi1EE19get_group_linear_idEv, TypeName = @group} : (!sycl_group_1_) -> i64

// CHECK-LLVM-LABEL: define spir_func void @_Z25group_get_group_linear_idN4sycl3_V15groupILi1EEE(
// CHECK-LLVM:           %"class.sycl::_V1::group.1"* noundef byval(%"class.sycl::_V1::group.1") align 8 %0) #[[FUNCATTRS]]           
// CHECK-LLVM: %{{.*}} = call spir_func i64 @_ZNK4sycl3_V15groupILi1EE19get_group_linear_idEv(%"class.sycl::_V1::group.1" addrspace(4)* %{{.*}})

SYCL_EXTERNAL void group_get_group_linear_id(sycl::group<1> group) {
  keep(group.get_group_linear_id());
}

// CHECK-MLIR-LABEL: func.func @_Z25group_get_local_linear_idN4sycl3_V15groupILi1EEE(
// CHECK-MLIR:           %arg0: memref<?x!sycl_group_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_group_1_, llvm.noundef}          
// CHECK-MLIR: %{{.*}} = "sycl.group.get_local_linear_id"(%{{.*}}) {ArgumentTypes = [memref<?x!sycl_group_1_, 4>], FunctionName = @get_local_linear_id, MangledFunctionName = @_ZNK4sycl3_V15groupILi1EE19get_local_linear_idEv, TypeName = @group} : (!sycl_group_1_) -> i64

// CHECK-LLVM-LABEL: define spir_func void @_Z25group_get_local_linear_idN4sycl3_V15groupILi1EEE(
// CHECK-LLVM:           %"class.sycl::_V1::group.1"* noundef byval(%"class.sycl::_V1::group.1") align 8 %0) #[[FUNCATTRS]]         
// CHECK-LLVM: %{{.*}} = call spir_func i64 @_ZNK4sycl3_V15groupILi1EE19get_local_linear_idEv(%"class.sycl::_V1::group.1" addrspace(4)* %{{.*}})

SYCL_EXTERNAL void group_get_local_linear_id(sycl::group<1> group) {
  keep(group.get_local_linear_id());
}

// CHECK-MLIR-LABEL: func.func @_Z28group_get_group_linear_rangeN4sycl3_V15groupILi1EEE(
// CHECK-MLIR:           %arg0: memref<?x!sycl_group_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_group_1_, llvm.noundef}            
// CHECK-MLIR: %{{.*}} = "sycl.group.get_group_linear_range"(%{{.*}}) {ArgumentTypes = [memref<?x!sycl_group_1_, 4>], FunctionName = @get_group_linear_range, MangledFunctionName = @_ZNK4sycl3_V15groupILi1EE22get_group_linear_rangeEv, TypeName = @group} : (!sycl_group_1_) -> i64

// CHECK-LLVM-LABEL: define spir_func void @_Z28group_get_group_linear_rangeN4sycl3_V15groupILi1EEE(
// CHECK-LLVM:           %"class.sycl::_V1::group.1"* noundef byval(%"class.sycl::_V1::group.1") align 8 %0) #[[FUNCATTRS]]           
// CHECK-LLVM: %{{.*}} = call spir_func i64 @_ZNK4sycl3_V15groupILi1EE22get_group_linear_rangeEv(%"class.sycl::_V1::group.1" addrspace(4)* %{{.*}})

SYCL_EXTERNAL void group_get_group_linear_range(sycl::group<1> group) {
  keep(group.get_group_linear_range());
}

// CHECK-MLIR-LABEL: func.func @_Z28group_get_local_linear_rangeN4sycl3_V15groupILi1EEE(
// CHECK-MLIR:           %arg0: memref<?x!sycl_group_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_group_1_, llvm.noundef})
// CHECK-MLIR: %{{.*}} = "sycl.group.get_local_linear_range"(%{{.*}}) {ArgumentTypes = [memref<?x!sycl_group_1_, 4>], FunctionName = @get_local_linear_range, MangledFunctionName = @_ZNK4sycl3_V15groupILi1EE22get_local_linear_rangeEv, TypeName = @group} : (!sycl_group_1_) -> i64

// CHECK-LLVM-LABEL: define spir_func void @_Z28group_get_local_linear_rangeN4sycl3_V15groupILi1EEE(
// CHECK-LLVM:           %"class.sycl::_V1::group.1"* noundef byval(%"class.sycl::_V1::group.1") align 8 %0) #[[FUNCATTRS]]             
// CHECK-LLVM: %{{.*}} = call spir_func i64 @_ZNK4sycl3_V15groupILi1EE22get_local_linear_rangeEv(%"class.sycl::_V1::group.1" addrspace(4)* %{{.*}})

SYCL_EXTERNAL void group_get_local_linear_range(sycl::group<1> group) {
  keep(group.get_local_linear_range());
}

// CHECK-MLIR-LABEL: func.func @_Z8method_2N4sycl3_V14itemILi2ELb1EEE(
// CHECK-MLIR:           %arg0: memref<?x![[ITEM2]]> {llvm.align = 8 : i64, llvm.byval = ![[ITEM2]], llvm.noundef})
// CHECK-MLIR-NEXT: %0 = "polygeist.memref2pointer"(%arg0) : (memref<?x![[ITEM2]]>) -> !llvm.ptr<![[ITEM2]]>
// CHECK-MLIR-NEXT: %1 = llvm.addrspacecast %0 : !llvm.ptr<![[ITEM2]]> to !llvm.ptr<![[ITEM2]], 4>
// CHECK-MLIR-NEXT: %2 = "polygeist.pointer2memref"(%1) : (!llvm.ptr<![[ITEM2]], 4>) -> memref<?x![[ITEM2]], 4>
// CHECK-MLIR-NEXT: %3 = sycl.call(%2, %2) {FunctionName = @"operator==", MangledFunctionName = @_ZNK4sycl3_V14itemILi2ELb1EEeqERKS2_, TypeName = @item} : (memref<?x![[ITEM2]], 4>, memref<?x![[ITEM2]], 4>) -> i1
// CHECK-MLIR-NEXT: return
// CHECK-MLIR-NEXT: }

// CHECK-LLVM-LABEL: define spir_func void @_Z8method_2N4sycl3_V14itemILi2ELb1EEE(
// CHECK-LLVM:           %"class.sycl::_V1::item.2.true"* noundef byval(%"class.sycl::_V1::item.2.true") align 8 %0) #[[FUNCATTRS]]
// CHECK-LLVM-NEXT:  %2 = addrspacecast %"class.sycl::_V1::item.2.true"* %0 to %"class.sycl::_V1::item.2.true" addrspace(4)*
// CHECK-LLVM-NEXT:  %3 = call spir_func i1 @_ZNK4sycl3_V14itemILi2ELb1EEeqERKS2_(%"class.sycl::_V1::item.2.true" addrspace(4)* %2, %"class.sycl::_V1::item.2.true" addrspace(4)* %2)
// CHECK-LLVM-NEXT:  ret void
// CHECK-LLVM-NEXT: }

SYCL_EXTERNAL void method_2(sycl::item<2, true> item) {
  auto id = item.operator==(item);
}

// CHECK-MLIR-LABEL: func.func @_Z4op_1N4sycl3_V12idILi2EEES2_(
// CHECK-MLIR:         %arg0: memref<?x!sycl_id_2_> {llvm.align = 8 : i64, llvm.byval = !sycl_id_2_, llvm.noundef}, %arg1: memref<?x!sycl_id_2_> {llvm.align = 8 : i64, llvm.byval = !sycl_id_2_, llvm.noundef})
// CHECK-MLIR-NEXT: %0 = "polygeist.memref2pointer"(%arg0) : (memref<?x!sycl_id_2_>) -> !llvm.ptr<!sycl_id_2_>
// CHECK-MLIR-NEXT: %1 = llvm.addrspacecast %0 : !llvm.ptr<!sycl_id_2_> to !llvm.ptr<!sycl_id_2_, 4>
// CHECK-MLIR-NEXT: %2 = "polygeist.pointer2memref"(%1) : (!llvm.ptr<!sycl_id_2_, 4>) -> memref<?x!sycl_id_2_, 4>
// CHECK-MLIR-NEXT: %3 = "polygeist.memref2pointer"(%arg1) : (memref<?x!sycl_id_2_>) -> !llvm.ptr<!sycl_id_2_>
// CHECK-MLIR-NEXT: %4 = llvm.addrspacecast %3 : !llvm.ptr<!sycl_id_2_> to !llvm.ptr<!sycl_id_2_, 4>
// CHECK-MLIR-NEXT: %5 = "polygeist.pointer2memref"(%4) : (!llvm.ptr<!sycl_id_2_, 4>) -> memref<?x!sycl_id_2_, 4>
// CHECK-MLIR-NEXT: %6 = sycl.call(%2, %5) {FunctionName = @"operator==", MangledFunctionName = @_ZNK4sycl3_V12idILi2EEeqERKS2_, TypeName = @id} : (memref<?x!sycl_id_2_, 4>, memref<?x!sycl_id_2_, 4>) -> i1
// CHECK-MLIR-NEXT: return
// CHECK-MLIR-NEXT: }

// CHECK-LLVM-LABEL: define spir_func void @_Z4op_1N4sycl3_V12idILi2EEES2_(
// CHECK-LLVM            %"class.sycl::_V1::id.2"* noundef byval(%"class.sycl::_V1::id.2") align 8 %0, %"class.sycl::_V1::id.2"* noundef byval(%"class.sycl::_V1::id.2") align 8 %1) #[[FUNCATTRS]]
// CHECK-LLVM-NEXT: %3 = addrspacecast %"class.sycl::_V1::id.2"* %0 to %"class.sycl::_V1::id.2" addrspace(4)*
// CHECK-LLVM-NEXT: %4 = addrspacecast %"class.sycl::_V1::id.2"* %1 to %"class.sycl::_V1::id.2" addrspace(4)*
// CHECK-LLVM-NEXT: %5 = call spir_func i1 @_ZNK4sycl3_V12idILi2EEeqERKS2_(%"class.sycl::_V1::id.2" addrspace(4)* %3, %"class.sycl::_V1::id.2" addrspace(4)* %4)
// CHECK-LLVM-NEXT: ret void
// CHECK-LLVM-NEXT: }

SYCL_EXTERNAL void op_1(sycl::id<2> a, sycl::id<2> b) {
  auto id = a == b;
}

// CHECK-MLIR-LABEL: func.func @_Z8static_1N4sycl3_V12idILi2EEES2_(
// CHECK-MLIR:         %arg0: memref<?x!sycl_id_2_> {llvm.align = 8 : i64, llvm.byval = !sycl_id_2_, llvm.noundef}, %arg1: memref<?x!sycl_id_2_> {llvm.align = 8 : i64, llvm.byval = !sycl_id_2_, llvm.noundef})
// CHECK-MLIR-NEXT: %c1_i32 = arith.constant 1 : i32
// CHECK-MLIR-NEXT: %c0_i32 = arith.constant 0 : i32
// CHECK-MLIR-NEXT: %0 = affine.load %arg0[0] : memref<?x!sycl_id_2_>
// CHECK-MLIR-NEXT: %1 = "sycl.id.get"(%0, %c0_i32) {ArgumentTypes = [memref<?x!sycl_array_2_, 4>, i32], FunctionName = @get, MangledFunctionName = @_ZNK4sycl3_V16detail5arrayILi2EE3getEi, TypeName = @array} : (!sycl_id_2_, i32) -> i64
// CHECK-MLIR-NEXT: %2 = "sycl.id.get"(%0, %c1_i32) {ArgumentTypes = [memref<?x!sycl_array_2_, 4>, i32], FunctionName = @get, MangledFunctionName = @_ZNK4sycl3_V16detail5arrayILi2EE3getEi, TypeName = @array} : (!sycl_id_2_, i32) -> i64
// CHECK-MLIR-NEXT: %3 = arith.addi %1, %2 : i64
// CHECK-MLIR-NEXT: %4 = sycl.call(%3) {FunctionName = @abs, MangledFunctionName = @_ZN4sycl3_V13absImEENSt9enable_ifIXsr6detail14is_ugenintegerIT_EE5valueES3_E4typeES3_} : (i64) -> i64
// CHECK-MLIR-NEXT: return
// CHECK-MLIR-NEXT: }

// CHECK-LLVM-LABEL: define spir_func void @_Z8static_1N4sycl3_V12idILi2EEES2_(
// CHECK-LLVM-SAME:      %"class.sycl::_V1::id.2"* noundef byval(%"class.sycl::_V1::id.2") align 8 %0, %"class.sycl::_V1::id.2"* noundef byval(%"class.sycl::_V1::id.2") align 8 %1) #[[FUNCATTRS]]  
// CHECK-LLVM-NEXT: %3 = alloca %"class.sycl::_V1::id.2", align 8
// CHECK-LLVM-NEXT: %4 = alloca %"class.sycl::_V1::id.2", align 8
// CHECK-LLVM-NEXT: %5 = load %"class.sycl::_V1::id.2", %"class.sycl::_V1::id.2"* %0, align 8
// CHECK-LLVM-NEXT: store %"class.sycl::_V1::id.2" %5, %"class.sycl::_V1::id.2"* %3, align 8
// CHECK-LLVM-NEXT: %6 = addrspacecast %"class.sycl::_V1::id.2"* %3 to %"class.sycl::_V1::id.2" addrspace(4)*
// CHECK-LLVM-NEXT: %7 = bitcast %"class.sycl::_V1::id.2" addrspace(4)* %6 to %"class.sycl::_V1::detail::array.2" addrspace(4)*
// CHECK-LLVM-NEXT: %8 = call spir_func i64 @_ZNK4sycl3_V16detail5arrayILi2EE3getEi(%"class.sycl::_V1::detail::array.2" addrspace(4)* %7, i32 0)
// CHECK-LLVM-NEXT: store %"class.sycl::_V1::id.2" %5, %"class.sycl::_V1::id.2"* %4, align 8
// CHECK-LLVM-NEXT: %9 = addrspacecast %"class.sycl::_V1::id.2"* %4 to %"class.sycl::_V1::id.2" addrspace(4)*
// CHECK-LLVM-NEXT: %10 = bitcast %"class.sycl::_V1::id.2" addrspace(4)* %9 to %"class.sycl::_V1::detail::array.2" addrspace(4)*
// CHECK-LLVM-NEXT: %11 = call spir_func i64 @_ZNK4sycl3_V16detail5arrayILi2EE3getEi(%"class.sycl::_V1::detail::array.2" addrspace(4)* %10, i32 1)
// CHECK-LLVM-NEXT: %12 = add i64 %8, %11
// CHECK-LLVM-NEXT: %13 = call spir_func i64 @_ZN4sycl3_V13absImEENSt9enable_ifIXsr6detail14is_ugenintegerIT_EE5valueES3_E4typeES3_(i64 %12)
// CHECK-LLVM-NEXT: ret void
// CHECK-LLVM-NEXT: }

SYCL_EXTERNAL void static_1(sycl::id<2> a, sycl::id<2> b) {
  auto abs = sycl::abs(a.get(0) + a.get(1));
}

// CHECK-LLVM: attributes #[[FUNCATTRS]] = { convergent mustprogress noinline norecurse nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="{{.*}}polygeist/tools/cgeist/Test/Verification/sycl/functions.cpp" }
