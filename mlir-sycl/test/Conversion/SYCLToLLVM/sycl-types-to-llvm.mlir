// RUN: sycl-mlir-opt -split-input-file -convert-sycl-to-llvm -verify-diagnostics %s | FileCheck %s

// CHECK: llvm.func @test_array.1(%arg0: !llvm.[[ARRAY_1:struct<"class.cl::sycl::detail::array.*", \(array<1 x i64>\)>]])
// CHECK: llvm.func @test_array.2(%arg0: !llvm.[[ARRAY_2:struct<"class.cl::sycl::detail::array.*", \(array<2 x i64>\)>]])
// CHECK: llvm.func @test_id(%arg0: !llvm.[[ID_1:struct<"class.cl::sycl::id.*", \(]][[ARRAY_1]][[SUFFIX:\)>]], %arg1: !llvm.[[ID_1]][[ARRAY_1]][[SUFFIX]])
// CHECK: llvm.func @test_range.1(%arg0: !llvm.[[RANGE_1:struct<"class.cl::sycl::range.*", \(]][[ARRAY_1]][[SUFFIX]])
// CHECK: llvm.func @test_range.2(%arg0: !llvm.[[RANGE_2:struct<"class.cl::sycl::range.*", \(]][[ARRAY_2]][[SUFFIX]])
// CHECK: llvm.func @test_accessorImplDevice(%arg0: !llvm.[[ACCESSORIMPLDEVICE_1:struct<"class.cl::sycl::detail::AccessorImplDevice.*", \(]][[ID_1]][[ARRAY_1]][[SUFFIX]], [[RANGE_1]][[ARRAY_1]][[SUFFIX]], [[RANGE_1]][[ARRAY_1]][[SUFFIX]][[SUFFIX]])
// CHECK: llvm.func @test_accessor.1(%arg0: !llvm.[[ACCESSOR_1:struct<"class.cl::sycl::accessor.*", \(]][[ACCESSORIMPLDEVICE_1]][[ID_1]][[ARRAY_1]][[SUFFIX]], [[RANGE_1]][[ARRAY_1]][[SUFFIX]], [[RANGE_1]][[ARRAY_1]][[SUFFIX]][[SUFFIX]], struct<(ptr<i32, 1>)>)>)
// CHECK: llvm.func @test_accessor.2(%arg0: !llvm.[[ACCESSOR_2:struct<"class.cl::sycl::accessor.*", \(]][[ACCESSORIMPLDEVICE_2:struct<"class.cl::sycl::detail::AccessorImplDevice.*", \(]][[ID_2:struct<"class.cl::sycl::id.*", \(]][[ARRAY_2]][[SUFFIX]], [[RANGE_2]][[ARRAY_2]][[SUFFIX]], [[RANGE_2]][[ARRAY_2]][[SUFFIX]][[SUFFIX]], struct<(ptr<i64, 1>)>)>)
// CHECK: llvm.func @test_item_base.true(%arg0: !llvm.[[ITEM_BASE_1_TRUE:struct<"class.cl::sycl::detail::ItemBase.1.true", \(]][[RANGE_1]][[ARRAY_1]][[SUFFIX]], [[ID_1]][[ARRAY_1]][[SUFFIX]], [[ID_1]][[ARRAY_1]][[SUFFIX]][[SUFFIX]])
// CHECK: llvm.func @test_item_base.false(%arg0: !llvm.[[ITEM_BASE_2_FALSE:struct<"class.cl::sycl::detail::ItemBase.2.false", \(]][[RANGE_2]][[ARRAY_2]][[SUFFIX]], [[ID_2]][[ARRAY_2]][[SUFFIX]][[SUFFIX]])
// CHECK: llvm.func @test_item(%arg0: !llvm.[[ITEM_1_TRUE:struct<"class.cl::sycl::item.1.true", \(]][[ITEM_BASE_1_TRUE]][[RANGE_1]][[ARRAY_1]][[SUFFIX]], [[ID_1]][[ARRAY_1]][[SUFFIX]], [[ID_1]][[ARRAY_1]][[SUFFIX]][[SUFFIX]][[SUFFIX]])

module {
  func.func @test_array.1(%arg0: !sycl.array<[1], (memref<1xi64>)>) {
    return
  }
  func.func @test_array.2(%arg0: !sycl.array<[2], (memref<2xi64>)>) {
    return
  }
  func.func @test_id(%arg0: !sycl.id<1>, %arg1: !sycl.id<1>) {
    return
  }
  func.func @test_range.1(%arg0: !sycl.range<1>) {
    return
  }
  func.func @test_range.2(%arg0: !sycl.range<2>) {
    return
  }
  func.func @test_accessorImplDevice(%arg0: !sycl.accessor_impl_device<[1], (!sycl.id<1>, !sycl.range<1>, !sycl.range<1>)>) {
    return
  }
  func.func @test_accessor.1(%arg0: !sycl.accessor<[1, i32, write, global_buffer], (!sycl.accessor_impl_device<[1], (!sycl.id<1>, !sycl.range<1>, !sycl.range<1>)>)>) {
    return
  }
  func.func @test_accessor.2(%arg0: !sycl.accessor<[2, i64, write, global_buffer], (!sycl.accessor_impl_device<[2], (!sycl.id<2>, !sycl.range<2>, !sycl.range<2>)>)>) {
    return
  }
  func.func @test_item_base.true(%arg0: !sycl.item_base<[1, true], (!sycl.range<1>, !sycl.id<1>, !sycl.id<1>)>) {
    return
  }
  func.func @test_item_base.false(%arg0: !sycl.item_base<[2, false], (!sycl.range<2>, !sycl.id<2>)>) {
    return
  }
  func.func @test_item(%arg0: !sycl.item<[1, true], (!sycl.item_base<[1, true], (!sycl.range<1>, !sycl.id<1>, !sycl.id<1>)>)>) {
    return
  }
}
