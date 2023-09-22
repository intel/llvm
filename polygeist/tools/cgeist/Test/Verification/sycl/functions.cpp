// RUN: clang++  -fsycl -fsycl-device-only -O0 -w -emit-mlir %s -o - | FileCheck %s --check-prefix=CHECK-MLIR
// RUN: clang++  -fsycl -fsycl-device-only -fsycl-targets=spir64-unknown-unknown-syclmlir -O0 -w -emit-llvm %s -o %t && rm %t

#include <sycl/sycl.hpp>

// CHECK-MLIR-DAG: !sycl_array_1_ = !sycl.array<[1], (memref<1xi64, 4>)>
// CHECK-MLIR-DAG: !sycl_array_2_ = !sycl.array<[2], (memref<2xi64, 4>)>
// CHECK-MLIR-DAG: !sycl_id_1_ = !sycl.id<[1], (!sycl_array_1_)>
// CHECK-MLIR-DAG: !sycl_id_2_ = !sycl.id<[2], (!sycl_array_2_)>
// CHECK-MLIR-DAG: !sycl_range_1_ = !sycl.range<[1], (!sycl_array_1_)>
// CHECK-MLIR-DAG: !sycl_range_2_ = !sycl.range<[2], (!sycl_array_2_)>
// CHECK-MLIR-DAG: !sycl_accessor_impl_device_1_ = !sycl.accessor_impl_device<[1], (!sycl_id_1_, !sycl_range_1_, !sycl_range_1_)>
// CHECK-MLIR-DAG: !sycl_accessor_impl_device_2_ = !sycl.accessor_impl_device<[2], (!sycl_id_2_, !sycl_range_2_, !sycl_range_2_)>
// CHECK-MLIR-DAG: !sycl_accessor_1_i32_rw_dev = !sycl.accessor<[1, i32, read_write, device], (!sycl_accessor_impl_device_1_, !llvm.struct<(memref<?xi32, 1>)>)>
// CHECK-MLIR-DAG: !sycl_accessor_2_i32_rw_dev = !sycl.accessor<[2, i32, read_write, device], (!sycl_accessor_impl_device_2_, !llvm.struct<(memref<?xi32, 1>)>)>
// CHECK-MLIR-DAG: ![[ACC_STRUCT:.*]] = !sycl.accessor<[1, !llvm.struct<(i32)>, read_write, device], (!sycl_accessor_impl_device_1_, !llvm.struct<(ptr<1>)>)>
// CHECK-MLIR-DAG: ![[ACC_SUBSCRIPT:.*]] = !sycl.accessor_subscript<[1], (!sycl_id_2_, !sycl_accessor_2_i32_rw_dev)>
// CHECK-MLIR-DAG: ![[ITEM_BASE1:.*]] = !sycl.item_base<[1, true], (!sycl_range_1_, !sycl_id_1_, !sycl_id_1_)
// CHECK-MLIR-DAG: ![[ITEM_BASE2:.*]] = !sycl.item_base<[2, true], (!sycl_range_2_, !sycl_id_2_, !sycl_id_2_)>
// CHECK-MLIR-DAG: ![[ITEM1:.*]] = !sycl.item<[1, true], (![[ITEM_BASE1]])>
// CHECK-MLIR-DAG: ![[ITEM2:.*]] = !sycl.item<[2, true], (![[ITEM_BASE2]])>

template <typename T> SYCL_EXTERNAL void keep(T);

// COM: Commenting out the checks below, this is the code that should be
// generated. Currently the DPC++ SYCL RT implementation of
// accessor::get_pointer is non-conforming. Once that problem is fixed we
// enable the checks below.

// COM-MLIR-LABEL: func.func @_Z20accessor_get_pointerN4sycl3_V18accessorIiLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE(
// COM-MLIR:           %{{.*}}: memref<?x!sycl_accessor_2_i32_rw_dev> {llvm.align = 8 : i64, llvm.byval = !sycl_accessor_2_i32_rw_dev, llvm.noundef})
// COM-MLIR: %{{.*}} = sycl.accessor.get_pointer(%{{.*}}) : (memref<?x!sycl_accessor_2_i32_rw_dev>) -> memref<?xi32, 1>

SYCL_EXTERNAL void accessor_get_pointer(sycl::accessor<sycl::cl_int, 2> acc) {
  keep(acc.get_pointer());
}

// CHECK-MLIR-LABEL: func.func @_Z18accessor_get_rangeN4sycl3_V18accessorIiLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE(
// CHECK-MLIR:           %{{.*}}: memref<?x!sycl_accessor_2_i32_rw_dev> {llvm.align = 8 : i64, llvm.byval = !sycl_accessor_2_i32_rw_dev, llvm.noundef})
// CHECK-MLIR: %{{.*}} = sycl.accessor.get_range(%arg0) : (memref<?x!sycl_accessor_2_i32_rw_dev>) -> !sycl_range_2_

SYCL_EXTERNAL void accessor_get_range(sycl::accessor<sycl::cl_int, 2> acc) {
  keep(acc.get_range());
}

// CHECK-MLIR-LABEL: func.func @_Z13accessor_sizeN4sycl3_V18accessorIiLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE(
// CHECK-MLIR:           %{{.*}}: memref<?x!sycl_accessor_2_i32_rw_dev> {llvm.align = 8 : i64, llvm.byval = !sycl_accessor_2_i32_rw_dev, llvm.noundef})
// CHECK-MLIR: %{{.*}} = sycl.accessor.size(%arg0) : (memref<?x!sycl_accessor_2_i32_rw_dev>) -> i64

SYCL_EXTERNAL void accessor_size(sycl::accessor<sycl::cl_int, 2> acc) {
  keep(acc.size());
}

// CHECK-MLIR-LABEL: func.func @_Z29accessor_subscript_operator_0N4sycl3_V18accessorIiLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEENS0_2idILi2EEE(
// CHECK-MLIR:           %{{.*}}: memref<?x!sycl_accessor_2_i32_rw_dev> {llvm.align = 8 : i64, llvm.byval = !sycl_accessor_2_i32_rw_dev, llvm.noundef}, 
// CHECK_MLIR_SAME:      %{{.*}}: memref<?x!sycl_id_2_> {llvm.align = 8 : i64, llvm.byval = !sycl_id_2_, llvm.noundef})
// CHECK-MLIR: %{{.*}} = sycl.accessor.subscript %{{.*}}[%{{.*}}] : (memref<?x!sycl_accessor_2_i32_rw_dev>, memref<?x!sycl_id_2_>) -> memref<?xi32, 4>

SYCL_EXTERNAL void accessor_subscript_operator_0(sycl::accessor<sycl::cl_int, 2> acc, sycl::id<2> index) {
  keep(acc[index]);
}

// CHECK-MLIR-LABEL: func.func @_Z29accessor_subscript_operator_1N4sycl3_V18accessorIiLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEEm(
// CHECK_MLIR:           %{{.*}}: memref<?x!sycl_accessor_2_i32_rw_dev>, %{{.*}}: i64)
// CHECK-MLIR: %{{.*}} = sycl.accessor.subscript %{{.*}}[%{{.*}}] : (memref<?x!sycl_accessor_2_i32_rw_dev>, i64) -> ![[ACC_SUBSCRIPT]]

SYCL_EXTERNAL void accessor_subscript_operator_1(sycl::accessor<sycl::cl_int, 2> acc, size_t index) {
  keep(acc[index]);
}

// CHECK-MLIR-LABEL: func.func @_Z29accessor_subscript_operator_2N4sycl3_V18accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEEm(
// CHECK-MLIR:           %{{.*}}: memref<?x!sycl_accessor_1_i32_rw_dev> {llvm.align = 8 : i64, llvm.byval = !sycl_accessor_1_i32_rw_dev, llvm.noundef}, 
// CHECK-MLIR-SAME:      %{{.*}}: i64 {llvm.noundef})
// CHECK-MLIR: %{{.*}} = sycl.accessor.subscript %{{.*}}[%{{.*}}] : (memref<?x!sycl_accessor_1_i32_rw_dev>, memref<?x!sycl_id_1_>) -> memref<?xi32, 4>

SYCL_EXTERNAL void accessor_subscript_operator_2(sycl::accessor<sycl::cl_int, 1> acc, size_t index) {
  keep(acc[index]);
}

// CHECK-MLIR-LABEL: func.func @_Z29accessor_subscript_operator_3N4sycl3_V18accessorI6StructLi1ELNS0_6access4modeE1026ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEEm(
// CHECK-MLIR:          %{{.*}}: memref<?x![[ACC_STRUCT]]> {llvm.align = 8 : i64, llvm.byval = ![[ACC_STRUCT]], llvm.noundef}, 
// CHECK-MLIR-SAME:     %{{.*}}: i64 {llvm.noundef})
// CHECK-MLIR: %{{.*}} = sycl.accessor.subscript %{{.*}}[%{{.*}}] : (memref<?x![[ACC_STRUCT]]>, memref<?x!sycl_id_1_>) -> !llvm.ptr<4> 

typedef struct {
  unsigned field;
} Struct;
SYCL_EXTERNAL void accessor_subscript_operator_3(sycl::accessor<Struct, 1> acc, size_t index) {
  keep(acc[index].field);
}

// CHECK-MLIR-LABEL: func.func @_Z11range_get_0N4sycl3_V15rangeILi2EEEi(
// CHECK-MLIR:           %{{.*}}: memref<?x!sycl_range_2_> {llvm.align = 8 : i64, llvm.byval = !sycl_range_2_, llvm.noundef}, 
// CHECK-MLIR-SAME:      %{{.*}}: i32 {llvm.noundef})
// CHECK-MLIR: %{{.*}} = sycl.range.get %{{.*}}[%{{.*}}] : (memref<?x!sycl_range_2_>, i32) -> i64

SYCL_EXTERNAL void range_get_0(sycl::range<2> r, int dimension) {
  keep(r.get(dimension));
}

// CHECK-MLIR-LABEL: func.func @_Z11range_get_1N4sycl3_V15rangeILi2EEEi(
// CHECK-MLIR:           %{{.*}}: memref<?x!sycl_range_2_> {llvm.align = 8 : i64, llvm.byval = !sycl_range_2_, llvm.noundef}
// CHECK-MLIR-SAME:      %{{.*}}: i32 {llvm.noundef})
// CHECK-MLIR: %{{.*}} = sycl.range.get %{{.*}}[%{{.*}}] : (memref<?x!sycl_range_2_>, i32) -> memref<?xi64, 4>


SYCL_EXTERNAL void range_get_1(sycl::range<2> r, int dimension) {
  keep(r[dimension]);
}

// CHECK-MLIR-LABEL: func.func @_Z11range_get_2N4sycl3_V15rangeILi2EEEi(
// CHECK-MLIR:           %{{.*}}: memref<?x!sycl_range_2_> {llvm.align = 8 : i64, llvm.byval = !sycl_range_2_, llvm.noundef}
// CHECK-MLIR-SAME:      %{{.*}}: i32 {llvm.noundef})
// CHECK-MLIR: %{{.*}} = sycl.range.get %{{.*}}[%{{.*}}] : (memref<?x!sycl_range_2_>, i32) -> i64

SYCL_EXTERNAL void range_get_2(const sycl::range<2> r, int dimension) {
  keep(r[dimension]);
}

// CHECK-MLIR-LABEL: func.func @_Z10range_sizeN4sycl3_V15rangeILi2EEE(
// CHECK-MLIR:           %{{.*}}: memref<?x!sycl_range_2_> {llvm.align = 8 : i64, llvm.byval = !sycl_range_2_, llvm.noundef})
// CHECK-MLIR: %{{.*}} = sycl.range.size(%{{.*}}) : (memref<?x!sycl_range_2_>) -> i64


SYCL_EXTERNAL void range_size(sycl::range<2> r) {
  keep(r.size());
}

// CHECK-MLIR-LABEL: func.func @_Z25nd_range_get_global_rangeN4sycl3_V18nd_rangeILi2EEE(
// CHECK-MLIR:           %{{.*}}: memref<?x!sycl_nd_range_2_> {llvm.align = 8 : i64, llvm.byval = !sycl_nd_range_2_, llvm.noundef})
// CHECK-MLIR: %{{.*}} = sycl.nd_range.get_global_range(%{{.*}}) : (memref<?x!sycl_nd_range_2_>) -> !sycl_range_2_

// CHECK-MLIR:           func.func @_ZNK4sycl3_V18nd_rangeILi2EE16get_global_rangeEv(%[[VAL_0:.*]]: memref<?x!sycl_nd_range_2_, 4> {llvm.align = 8 : i64, llvm.dereferenceable_or_null = 48 : i64, llvm.noundef}) -> !sycl_range_2_
// CHECK-MLIR-NEXT:             %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK-MLIR-NEXT:             %[[VAL_2:.*]] = memref.alloca() : memref<1x!sycl_range_2_>
// CHECK-MLIR-NEXT:             %cast = memref.cast %alloca : memref<1x!sycl_range_2_> to memref<?x!sycl_range_2_>
// CHECK-MLIR-NEXT:             %[[VAL_3:.*]] = "polygeist.subindex"(%[[VAL_0]], %[[VAL_1]]) : (memref<?x!sycl_nd_range_2_, 4>, index) -> memref<?x!sycl_range_2_, 4>
// CHECK-MLIR-NEXT:             %[[VAL_4:.*]] = memref.memory_space_cast %cast : memref<?x!sycl_range_2_> to memref<?x!sycl_range_2_, 4>
// CHECK-MLIR-NEXT:             sycl.constructor @range(%[[VAL_4]], %[[VAL_3]]) {MangledFunctionName = @{{.*}}} : (memref<?x!sycl_range_2_, 4>, memref<?x!sycl_range_2_, 4>)
// CHECK-MLIR-NEXT:             %[[VAL_7:.*]] = affine.load %[[VAL_2]][0] : memref<1x!sycl_range_2_>
// CHECK-MLIR-NEXT:             return %[[VAL_7]] : !sycl_range_2_
// CHECK-MLIR-NEXT:           }

SYCL_EXTERNAL void nd_range_get_global_range(sycl::nd_range<2> nd_range) {
  keep(nd_range.get_global_range());
}

// CHECK-MLIR-LABEL: func.func @_Z24nd_range_get_local_rangeN4sycl3_V18nd_rangeILi2EEE(
// CHECK-MLIR:           %{{.*}}: memref<?x!sycl_nd_range_2_> {llvm.align = 8 : i64, llvm.byval = !sycl_nd_range_2_, llvm.noundef})
// CHECK-MLIR: %{{.*}} = sycl.nd_range.get_local_range(%{{.*}}) : (memref<?x!sycl_nd_range_2_>) -> !sycl_range_2_

SYCL_EXTERNAL void nd_range_get_local_range(sycl::nd_range<2> nd_range) {
  keep(nd_range.get_local_range());
}

// CHECK-MLIR-LABEL: func.func @_Z24nd_range_get_group_rangeN4sycl3_V18nd_rangeILi2EEE(
// CHECK-MLIR:           %{{.*}}: memref<?x!sycl_nd_range_2_> {llvm.align = 8 : i64, llvm.byval = !sycl_nd_range_2_, llvm.noundef})
// CHECK-MLIR: %{{.*}} = sycl.nd_range.get_group_range(%{{.*}}) : (memref<?x!sycl_nd_range_2_>) -> !sycl_range_2_

SYCL_EXTERNAL void nd_range_get_group_range(sycl::nd_range<2> nd_range) {
  keep(nd_range.get_group_range());
}

// CHECK-MLIR-LABEL: func.func @_Z8id_get_0N4sycl3_V12idILi1EEEi(
// CHECK-MLIR:           %arg0: memref<?x!sycl_id_1_>  {llvm.align = 8 : i64, llvm.byval = !sycl_id_1_, llvm.noundef}, %arg1: i32  {llvm.noundef})
// CHECK-MLIR: %{{.*}} = sycl.id.get %{{.*}}[%arg1] : (memref<?x!sycl_id_1_>, i32) -> i64

SYCL_EXTERNAL void id_get_0(sycl::id<1> id, int dimension) {
  keep(id.get(dimension));
}

// CHECK-MLIR-LABEL: func.func @_Z8id_get_1N4sycl3_V12idILi1EEEi(
// CHECK-MLIR:           %arg0: memref<?x!sycl_id_1_>  {llvm.align = 8 : i64, llvm.byval = !sycl_id_1_, llvm.noundef}, %arg1: i32  {llvm.noundef})  
// CHECK-MLIR: %{{.*}} = sycl.id.get %{{.*}}[%arg1] : (memref<?x!sycl_id_1_>, i32) -> memref<?xi64, 4>

SYCL_EXTERNAL void id_get_1(sycl::id<1> id, int dimension) {
  keep(id[dimension]);
}

// CHECK-MLIR-LABEL: func.func @_Z8id_get_2N4sycl3_V12idILi1EEEi(
// CHECK-MLIR:           %arg0: memref<?x!sycl_id_1_>  {llvm.align = 8 : i64, llvm.byval = !sycl_id_1_, llvm.noundef}, %arg1: i32  {llvm.noundef})    
// CHECK-MLIR: %{{.*}} = sycl.id.get %{{.*}}[%arg1] : (memref<?x!sycl_id_1_>, i32) -> i64

SYCL_EXTERNAL void id_get_2(const sycl::id<1> id, int dimension) {
  keep(id[dimension]);
}

// CHECK-MLIR-LABEL: func.func @_Z8id_get_3N4sycl3_V12idILi1EEE(
// CHECK-MLIR:           %arg0: memref<?x!sycl_id_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_id_1_, llvm.noundef}
// CHECK-MLIR: %{{.*}} = sycl.id.get %{{.*}}[] : (memref<?x!sycl_id_1_>) -> i64

SYCL_EXTERNAL void id_get_3(sycl::id<1> id) {
  keep(static_cast<size_t>(id));
}

// CHECK-MLIR-LABEL: func.func @_Z13item_get_id_0N4sycl3_V14itemILi1ELb1EEE(
// CHECK-MLIR:           %arg0: memref<?x![[ITEM1]]> {llvm.align = 8 : i64, llvm.byval = ![[ITEM1]], llvm.noundef})
// CHECK-MLIR: %{{.*}} = sycl.item.get_id(%{{.*}}) : (memref<?x![[ITEM1]]>) -> !sycl_id_1_

SYCL_EXTERNAL void item_get_id_0(sycl::item<1> item) {
  keep(item.get_id());
}

// CHECK-MLIR-LABEL: func.func @_Z13item_get_id_1N4sycl3_V14itemILi1ELb1EEEi(
// CHECK-MLIR:           %arg0: memref<?x![[ITEM1]]> {llvm.align = 8 : i64, llvm.byval = ![[ITEM1]], llvm.noundef}, %arg1: i32 {llvm.noundef})
// CHECK-MLIR: %{{.*}} = sycl.item.get_id(%{{.*}}, %arg1) : (memref<?x![[ITEM1]]>, i32) -> i64

SYCL_EXTERNAL void item_get_id_1(sycl::item<1> item, int dimension) {
  keep(item.get_id(dimension));
}

// CHECK-MLIR-LABEL: func.func @_Z13item_get_id_2N4sycl3_V14itemILi1ELb1EEEi(
// CHECK-MLIR:           %arg0: memref<?x![[ITEM1]]> {llvm.align = 8 : i64, llvm.byval = ![[ITEM1]], llvm.noundef}, %arg1: i32 {llvm.noundef})
// CHECK-MLIR: %{{.*}} = sycl.item.get_id(%{{.*}}, %arg1) : (memref<?x![[ITEM1]]>, i32) -> i64

SYCL_EXTERNAL void item_get_id_2(sycl::item<1> item, int dimension) {
  keep(item[dimension]);
}

// CHECK-MLIR-LABEL: func.func @_Z13item_get_id_3N4sycl3_V14itemILi1ELb1EEE(
// CHECK-MLIR:           %arg0: memref<?x![[ITEM1]]> {llvm.align = 8 : i64, llvm.byval = ![[ITEM1]], llvm.noundef})
// CHECK-MLIR: %{{.*}} = sycl.item.get_id(%{{.*}}) : (memref<?x![[ITEM1]]>) -> i64

SYCL_EXTERNAL void item_get_id_3(sycl::item<1> item) {
  keep(static_cast<size_t>(item));
}

// CHECK-MLIR-LABEL: func.func @_Z16item_get_range_0N4sycl3_V14itemILi1ELb1EEE(
// CHECK-MLIR:           %arg0: memref<?x![[ITEM1]]> {llvm.align = 8 : i64, llvm.byval = ![[ITEM1]], llvm.noundef})
// CHECK-MLIR: %{{.*}} = sycl.item.get_range(%{{.*}}) : (memref<?x![[ITEM1]]>) -> !sycl_range_1_

SYCL_EXTERNAL void item_get_range_0(sycl::item<1> item) {
  keep(item.get_range());
}

// CHECK-MLIR-LABEL: func.func @_Z16item_get_range_1N4sycl3_V14itemILi1ELb1EEEi(
// CHECK-MLIR:           %arg0: memref<?x![[ITEM1]]> {llvm.align = 8 : i64, llvm.byval = ![[ITEM1]], llvm.noundef}, %arg1: i32 {llvm.noundef})
// CHECK-MLIR: %{{.*}} = sycl.item.get_range(%{{.*}}, %arg1) : (memref<?x![[ITEM1]]>, i32) -> i64

SYCL_EXTERNAL void item_get_range_1(sycl::item<1> item, int dimension) {
  keep(item.get_range(dimension));
}

// CHECK-MLIR-LABEL: func.func @_Z18item_get_linear_idN4sycl3_V14itemILi1ELb1EEE(
// CHECK-MLIR:           %arg0: memref<?x![[ITEM1]]> {llvm.align = 8 : i64, llvm.byval = ![[ITEM1]], llvm.noundef})
// CHECK-MLIR: %{{.*}} = sycl.item.get_linear_id(%{{.*}}) : (memref<?x![[ITEM1]]>) -> i64

SYCL_EXTERNAL void item_get_linear_id(sycl::item<1> item) {
  keep(item.get_linear_id());
}

// CHECK-MLIR-LABEL: func.func @_Z23nd_item_get_global_id_0N4sycl3_V17nd_itemILi1EEE(
// CHECK-MLIR:           %arg0: memref<?x!sycl_nd_item_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_nd_item_1_, llvm.noundef})
// CHECK-MLIR: %{{.*}} = sycl.nd_item.get_global_id(%{{.*}}) : (memref<?x!sycl_nd_item_1_>) -> !sycl_id_1_

SYCL_EXTERNAL void nd_item_get_global_id_0(sycl::nd_item<1> nd_item) {
  keep(nd_item.get_global_id());
}

// CHECK-MLIR-LABEL: func.func @_Z23nd_item_get_global_id_1N4sycl3_V17nd_itemILi1EEEi(
// CHECK-MLIR:           %arg0: memref<?x!sycl_nd_item_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_nd_item_1_, llvm.noundef}, %arg1: i32 {llvm.noundef})
// CHECK-MLIR: %{{.*}} = sycl.nd_item.get_global_id(%{{.*}}, %arg1) : (memref<?x!sycl_nd_item_1_>, i32) -> i64

SYCL_EXTERNAL void nd_item_get_global_id_1(sycl::nd_item<1> nd_item, int dimension) {
  keep(nd_item.get_global_id(dimension));
}

// CHECK-MLIR-LABEL: func.func @_Z28nd_item_get_global_linear_idN4sycl3_V17nd_itemILi1EEE(
// CHECK-MLIR:           %arg0: memref<?x!sycl_nd_item_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_nd_item_1_, llvm.noundef})
// CHECK-MLIR: %{{.*}} = sycl.nd_item.get_global_linear_id(%{{.*}}) : (memref<?x!sycl_nd_item_1_>) -> i64

SYCL_EXTERNAL void nd_item_get_global_linear_id(sycl::nd_item<1> nd_item) {
  keep(nd_item.get_global_linear_id());
}

// CHECK-MLIR-LABEL: func.func @_Z22nd_item_get_local_id_0N4sycl3_V17nd_itemILi1EEE(
// CHECK-MLIR:           %arg0: memref<?x!sycl_nd_item_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_nd_item_1_, llvm.noundef})
// CHECK-MLIR: %{{.*}} = sycl.nd_item.get_local_id(%{{.*}}) : (memref<?x!sycl_nd_item_1_>) -> !sycl_id_1_

SYCL_EXTERNAL void nd_item_get_local_id_0(sycl::nd_item<1> nd_item) {
  keep(nd_item.get_local_id());
}

// CHECK-MLIR-LABEL: func.func @_Z22nd_item_get_local_id_1N4sycl3_V17nd_itemILi1EEEi(
// CHECK-MLIR:           %arg0: memref<?x!sycl_nd_item_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_nd_item_1_, llvm.noundef}, %arg1: i32 {llvm.noundef})
// CHECK-MLIR: %{{.*}} = sycl.nd_item.get_local_id(%{{.*}}, %arg1) : (memref<?x!sycl_nd_item_1_>, i32) -> i64

SYCL_EXTERNAL void nd_item_get_local_id_1(sycl::nd_item<1> nd_item, int dimension) {
  keep(nd_item.get_local_id(dimension));
}

// CHECK-MLIR-LABEL: func.func @_Z27nd_item_get_local_linear_idN4sycl3_V17nd_itemILi1EEE(
// CHECK-MLIR:           %arg0: memref<?x!sycl_nd_item_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_nd_item_1_, llvm.noundef})
// CHECK-MLIR: %{{.*}} = sycl.nd_item.get_local_linear_id(%{{.*}}) : (memref<?x!sycl_nd_item_1_>) -> i64

SYCL_EXTERNAL void nd_item_get_local_linear_id(sycl::nd_item<1> nd_item) {
  keep(nd_item.get_local_linear_id());
}

// CHECK-MLIR-LABEL: func.func @_Z19nd_item_get_group_0N4sycl3_V17nd_itemILi1EEE(
// CHECK-MLIR:           %arg0: memref<?x!sycl_nd_item_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_nd_item_1_, llvm.noundef})  
// CHECK-MLIR: %{{.*}} = sycl.nd_item.get_group(%{{.*}}) : (memref<?x!sycl_nd_item_1_>) -> !sycl_group_1_

SYCL_EXTERNAL void nd_item_get_group_0(sycl::nd_item<1> nd_item) {
  keep(nd_item.get_group());
}

// CHECK-MLIR-LABEL: func.func @_Z19nd_item_get_group_1N4sycl3_V17nd_itemILi1EEEi(
// CHECK-MLIR:           %arg0: memref<?x!sycl_nd_item_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_nd_item_1_, llvm.noundef}, %arg1: i32 {llvm.noundef})
// CHECK-MLIR: %{{.*}} = sycl.nd_item.get_group(%{{.*}}, %arg1) : (memref<?x!sycl_nd_item_1_>, i32) -> i64

SYCL_EXTERNAL void nd_item_get_group_1(sycl::nd_item<1> nd_item, int dimension) {
  keep(nd_item.get_group(dimension));
}

// CHECK-MLIR-LABEL: func.func @_Z27nd_item_get_group_linear_idN4sycl3_V17nd_itemILi1EEE(
// CHECK-MLIR:           %arg0: memref<?x!sycl_nd_item_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_nd_item_1_, llvm.noundef})
// CHECK-MLIR: %{{.*}} = sycl.nd_item.get_group_linear_id(%{{.*}}) : (memref<?x!sycl_nd_item_1_>) -> i64

SYCL_EXTERNAL void nd_item_get_group_linear_id(sycl::nd_item<1> nd_item) {
  keep(nd_item.get_group_linear_id());
}

// CHECK-MLIR-LABEL: func.func @_Z25nd_item_get_group_range_0N4sycl3_V17nd_itemILi1EEE(
// CHECK-MLIR:           %arg0: memref<?x!sycl_nd_item_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_nd_item_1_, llvm.noundef})
// CHECK-MLIR: %{{.*}} = sycl.nd_item.get_group_range(%{{.*}}) : (memref<?x!sycl_nd_item_1_>) -> !sycl_range_1_

SYCL_EXTERNAL void nd_item_get_group_range_0(sycl::nd_item<1> nd_item) {
  keep(nd_item.get_group_range());
}

// CHECK-MLIR-LABEL: func.func @_Z25nd_item_get_group_range_1N4sycl3_V17nd_itemILi1EEEi(
// CHECK-MLIR:           %arg0: memref<?x!sycl_nd_item_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_nd_item_1_, llvm.noundef}, %arg1: i32 {llvm.noundef})
// CHECK-MLIR: %{{.*}} = sycl.nd_item.get_group_range(%{{.*}}, %arg1) : (memref<?x!sycl_nd_item_1_>, i32) -> i64

SYCL_EXTERNAL void nd_item_get_group_range_1(sycl::nd_item<1> nd_item, int dimension) {
  keep(nd_item.get_group_range(dimension));
}

// CHECK-MLIR-LABEL: func.func @_Z26nd_item_get_global_range_0N4sycl3_V17nd_itemILi1EEE(
// CHECK-MLIR:           %arg0: memref<?x!sycl_nd_item_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_nd_item_1_, llvm.noundef})
// CHECK-MLIR: %{{.*}} = sycl.nd_item.get_global_range(%{{.*}}) : (memref<?x!sycl_nd_item_1_>) -> !sycl_range_1_

SYCL_EXTERNAL void nd_item_get_global_range_0(sycl::nd_item<1> nd_item) {
  keep(nd_item.get_global_range());
}

// CHECK-MLIR-LABEL: func.func @_Z26nd_item_get_global_range_1N4sycl3_V17nd_itemILi1EEEi(
// CHECK-MLIR:           %arg0: memref<?x!sycl_nd_item_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_nd_item_1_, llvm.noundef}, %arg1: i32 {llvm.noundef})
// CHECK-MLIR: %{{.*}} = sycl.nd_item.get_global_range(%{{.*}}, %arg1) : (memref<?x!sycl_nd_item_1_>, i32) -> i64

SYCL_EXTERNAL void nd_item_get_global_range_1(sycl::nd_item<1> nd_item, int dimension) {
  keep(nd_item.get_global_range(dimension));
}

// CHECK-MLIR-LABEL: func.func @_Z25nd_item_get_local_range_0N4sycl3_V17nd_itemILi1EEE(
// CHECK-MLIR:           %arg0: memref<?x!sycl_nd_item_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_nd_item_1_, llvm.noundef})
// CHECK-MLIR: %{{.*}} = sycl.nd_item.get_local_range(%{{.*}}) : (memref<?x!sycl_nd_item_1_>) -> !sycl_range_1_

SYCL_EXTERNAL void nd_item_get_local_range_0(sycl::nd_item<1> nd_item) {
  keep(nd_item.get_local_range());
}

// CHECK-MLIR-LABEL: func.func @_Z25nd_item_get_local_range_1N4sycl3_V17nd_itemILi1EEEi(
// CHECK-MLIR:           %arg0: memref<?x!sycl_nd_item_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_nd_item_1_, llvm.noundef}, %arg1: i32 {llvm.noundef})
// CHECK-MLIR: %{{.*}} = sycl.nd_item.get_local_range(%{{.*}}, %arg1) : (memref<?x!sycl_nd_item_1_>, i32) -> i64

SYCL_EXTERNAL void nd_item_get_local_range_1(sycl::nd_item<1> nd_item, int dimension) {
  keep(nd_item.get_local_range(dimension));
}

// CHECK-MLIR-LABEL: func.func @_Z20nd_item_get_nd_rangeN4sycl3_V17nd_itemILi1EEE(
// CHECK-MLIR:           %arg0: memref<?x!sycl_nd_item_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_nd_item_1_, llvm.noundef})
// CHECK-MLIR: %{{.*}} = sycl.nd_item.get_nd_range(%{{.*}}) : (memref<?x!sycl_nd_item_1_>) -> !sycl_nd_range_1_

SYCL_EXTERNAL void nd_item_get_nd_range(sycl::nd_item<1> nd_item) {
  keep(nd_item.get_nd_range());
}

// CHECK-MLIR-LABEL: func.func @_Z20group_get_group_id_0N4sycl3_V15groupILi1EEE(
// CHECK-MLIR:           %arg0: memref<?x!sycl_group_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_group_1_, llvm.noundef})
// CHECK-MLIR: %{{.*}} = sycl.group.get_group_id(%{{.*}}) : (memref<?x!sycl_group_1_>) -> !sycl_id_1_

SYCL_EXTERNAL void group_get_group_id_0(sycl::group<1> group) {
  keep(group.get_group_id());
}

// CHECK-MLIR-LABEL: func.func @_Z20group_get_group_id_1N4sycl3_V15groupILi1EEEi(
// CHECK-MLIR:           %arg0: memref<?x!sycl_group_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_group_1_, llvm.noundef}, %arg1: i32 {llvm.noundef})
// CHECK-MLIR: %{{.*}} = sycl.group.get_group_id(%{{.*}}, %arg1) : (memref<?x!sycl_group_1_>, i32) -> i64

SYCL_EXTERNAL void group_get_group_id_1(sycl::group<1> group, int dimension) {
  keep(group.get_group_id(dimension));
}

// CHECK-MLIR-LABEL: func.func @_Z20group_get_group_id_2N4sycl3_V15groupILi1EEEi(
// CHECK-MLIR:           %arg0: memref<?x!sycl_group_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_group_1_, llvm.noundef}, %arg1: i32 {llvm.noundef})
// CHECK-MLIR: %{{.*}} = sycl.group.get_group_id(%{{.*}}, %arg1) : (memref<?x!sycl_group_1_>, i32) -> i64

SYCL_EXTERNAL void group_get_group_id_2(sycl::group<1> group, int dimension) {
  keep(group[dimension]);
}

// CHECK-MLIR-LABEL: func.func @_Z20group_get_local_id_0N4sycl3_V15groupILi1EEE(
// CHECK-MLIR:           %arg0: memref<?x!sycl_group_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_group_1_, llvm.noundef})
// CHECK-MLIR: %{{.*}} = sycl.group.get_local_id(%{{.*}}) : (memref<?x!sycl_group_1_>) -> !sycl_id_1_

SYCL_EXTERNAL void group_get_local_id_0(sycl::group<1> group) {
  keep(group.get_local_id());
}

// CHECK-MLIR-LABEL: func.func @_Z20group_get_local_id_1N4sycl3_V15groupILi1EEEi(
// CHECK-MLIR:           %arg0: memref<?x!sycl_group_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_group_1_, llvm.noundef}, %arg1: i32 {llvm.noundef})
// CHECK-MLIR: %{{.*}} = sycl.group.get_local_id(%{{.*}}, %arg1) : (memref<?x!sycl_group_1_>, i32) -> i64

SYCL_EXTERNAL void group_get_local_id_1(sycl::group<1> group, int dimension) {
  keep(group.get_local_id(dimension));
}

// CHECK-MLIR-LABEL: func.func @_Z23group_get_local_range_0N4sycl3_V15groupILi1EEE(
// CHECK-MLIR:           %arg0: memref<?x!sycl_group_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_group_1_, llvm.noundef}
// CHECK-MLIR: %{{.*}} = sycl.group.get_local_range(%{{.*}}) : (memref<?x!sycl_group_1_>) -> !sycl_range_1_

SYCL_EXTERNAL void group_get_local_range_0(sycl::group<1> group) {
  keep(group.get_local_range());
}

// CHECK-MLIR-LABEL: func.func @_Z23group_get_local_range_1N4sycl3_V15groupILi1EEEi(
// CHECK-MLIR:           %arg0: memref<?x!sycl_group_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_group_1_, llvm.noundef}, %arg1: i32 {llvm.noundef})
// CHECK-MLIR: %{{.*}} = sycl.group.get_local_range(%{{.*}}, %arg1) : (memref<?x!sycl_group_1_>, i32) -> i64

SYCL_EXTERNAL void group_get_local_range_1(sycl::group<1> group, int dimension) {
  keep(group.get_local_range(dimension));
}

// CHECK-MLIR-LABEL: func.func @_Z23group_get_group_range_0N4sycl3_V15groupILi1EEE(
// CHECK-MLIR:           %arg0: memref<?x!sycl_group_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_group_1_, llvm.noundef}  
// CHECK-MLIR: %{{.*}} = sycl.group.get_group_range(%{{.*}}) : (memref<?x!sycl_group_1_>) -> !sycl_range_1_

SYCL_EXTERNAL void group_get_group_range_0(sycl::group<1> group) {
  keep(group.get_group_range());
}

// CHECK-MLIR-LABEL: func.func @_Z23group_get_group_range_1N4sycl3_V15groupILi1EEEi(
// CHECK-MLIR:           %arg0: memref<?x!sycl_group_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_group_1_, llvm.noundef}    
// CHECK-MLIR: %{{.*}} = sycl.group.get_group_range(%{{.*}}, %arg1) : (memref<?x!sycl_group_1_>, i32) -> i64

SYCL_EXTERNAL void group_get_group_range_1(sycl::group<1> group, int dimension) {
  keep(group.get_group_range(dimension));
}

// CHECK-MLIR-LABEL: func.func @_Z25group_get_max_local_rangeN4sycl3_V15groupILi1EEE(
// CHECK-MLIR:           %arg0: memref<?x!sycl_group_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_group_1_, llvm.noundef}      
// CHECK-MLIR: %{{.*}} = sycl.group.get_max_local_range(%{{.*}}) : (memref<?x!sycl_group_1_>) -> !sycl_range_1_

SYCL_EXTERNAL void group_get_max_local_range(sycl::group<1> group) {
  keep(group.get_max_local_range());
}

// CHECK-MLIR-LABEL: func.func @_Z25group_get_group_linear_idN4sycl3_V15groupILi1EEE(
// CHECK-MLIR:           %arg0: memref<?x!sycl_group_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_group_1_, llvm.noundef}        
// CHECK-MLIR: %{{.*}} = sycl.group.get_group_linear_id(%{{.*}}) : (memref<?x!sycl_group_1_>) -> i64

SYCL_EXTERNAL void group_get_group_linear_id(sycl::group<1> group) {
  keep(group.get_group_linear_id());
}

// CHECK-MLIR-LABEL: func.func @_Z25group_get_local_linear_idN4sycl3_V15groupILi1EEE(
// CHECK-MLIR:           %arg0: memref<?x!sycl_group_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_group_1_, llvm.noundef}          
// CHECK-MLIR: %{{.*}} = sycl.group.get_local_linear_id(%{{.*}}) : (memref<?x!sycl_group_1_>) -> i64

SYCL_EXTERNAL void group_get_local_linear_id(sycl::group<1> group) {
  keep(group.get_local_linear_id());
}

// CHECK-MLIR-LABEL: func.func @_Z28group_get_group_linear_rangeN4sycl3_V15groupILi1EEE(
// CHECK-MLIR:           %arg0: memref<?x!sycl_group_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_group_1_, llvm.noundef}            
// CHECK-MLIR: %{{.*}} = sycl.group.get_group_linear_range(%{{.*}}) : (memref<?x!sycl_group_1_>) -> i64

SYCL_EXTERNAL void group_get_group_linear_range(sycl::group<1> group) {
  keep(group.get_group_linear_range());
}

// CHECK-MLIR-LABEL: func.func @_Z28group_get_local_linear_rangeN4sycl3_V15groupILi1EEE(
// CHECK-MLIR:           %arg0: memref<?x!sycl_group_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_group_1_, llvm.noundef})
// CHECK-MLIR: %{{.*}} = sycl.group.get_local_linear_range(%{{.*}}) : (memref<?x!sycl_group_1_>) -> i64

SYCL_EXTERNAL void group_get_local_linear_range(sycl::group<1> group) {
  keep(group.get_local_linear_range());
}

// CHECK-MLIR-LABEL: func.func @_Z8method_2N4sycl3_V14itemILi2ELb1EEE(
// CHECK-MLIR:           %arg0: memref<?x![[ITEM2]]> {llvm.align = 8 : i64, llvm.byval = ![[ITEM2]], llvm.noundef})
// CHECK-MLIR-NEXT: %memspacecast = memref.memory_space_cast %arg0 : memref<?x![[ITEM2]]> to memref<?x![[ITEM2]], 4>
// CHECK-MLIR-NEXT: %0 = sycl.call @"operator=="(%memspacecast, %memspacecast) {MangledFunctionName = @{{.*}}, TypeName = @item} : (memref<?x![[ITEM2]], 4>, memref<?x![[ITEM2]], 4>) -> i1
// CHECK-MLIR-NEXT: return
// CHECK-MLIR-NEXT: }

SYCL_EXTERNAL void method_2(sycl::item<2, true> item) {
  auto id = item.operator==(item);
}

// CHECK-MLIR-LABEL: func.func @_Z4op_1N4sycl3_V12idILi2EEES2_(
// CHECK-MLIR:         %arg0: memref<?x!sycl_id_2_> {llvm.align = 8 : i64, llvm.byval = !sycl_id_2_, llvm.noundef}, %arg1: memref<?x!sycl_id_2_> {llvm.align = 8 : i64, llvm.byval = !sycl_id_2_, llvm.noundef})
// CHECK-MLIR-NEXT: %memspacecast = memref.memory_space_cast %arg0 : memref<?x!sycl_id_2_> to memref<?x!sycl_id_2_, 4>
// CHECK-MLIR-NEXT: %memspacecast_0 = memref.memory_space_cast %arg1 : memref<?x!sycl_id_2_> to memref<?x!sycl_id_2_, 4>
// CHECK-MLIR-NEXT: %0 = sycl.call @"operator=="(%memspacecast, %memspacecast_0) {MangledFunctionName = @{{.*}}, TypeName = @id} : (memref<?x!sycl_id_2_, 4>, memref<?x!sycl_id_2_, 4>) -> i1
// CHECK-MLIR-NEXT: return
// CHECK-MLIR-NEXT: }

SYCL_EXTERNAL void op_1(sycl::id<2> a, sycl::id<2> b) {
  auto id = a == b;
}

// CHECK-MLIR-LABEL: func.func @_Z8static_1N4sycl3_V12idILi2EEES2_(
// CHECK-MLIR:         %arg0: memref<?x!sycl_id_2_> {llvm.align = 8 : i64, llvm.byval = !sycl_id_2_, llvm.noundef}, %arg1: memref<?x!sycl_id_2_> {llvm.align = 8 : i64, llvm.byval = !sycl_id_2_, llvm.noundef})
// CHECK-MLIR-NEXT: %c1_i32 = arith.constant 1 : i32
// CHECK-MLIR-NEXT: %c0_i32 = arith.constant 0 : i32
// CHECK-MLIR-NEXT: %0 = sycl.id.get %arg0[%c0_i32] : (memref<?x!sycl_id_2_>, i32) -> i64
// CHECK-MLIR-NEXT: %1 = sycl.id.get %arg0[%c1_i32] : (memref<?x!sycl_id_2_>, i32) -> i64
// CHECK-MLIR-NEXT: %2 = arith.addi %0, %1 : i64
// CHECK-MLIR-NEXT: %3 = sycl.call @abs(%2) {MangledFunctionName = @{{.*}}} : (i64) -> i64
// CHECK-MLIR-NEXT: return
// CHECK-MLIR-NEXT: }

SYCL_EXTERNAL void static_1(sycl::id<2> a, sycl::id<2> b) {
  auto abs = sycl::abs(a.get(0) + a.get(1));
}
