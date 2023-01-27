// RUN: clang++ -fsycl -fsycl-device-only -O0 -w -emit-mlir -o - %s | FileCheck %s 

#include <sycl/accessor.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;
static constexpr unsigned N = 8;

// CHECK-LABEL:  func.func @_ZN4sycl3_V18accessorINS0_3vecIfLi4EEELi1ELNS0_6access4modeE1024ELNS4_6targetE2017ELNS4_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEE6__initE14ocl_image1d_ro(%arg0: memref<?x!sycl_accessor_1_21sycl2Evec3C5Bf322C_45D2C_28vector3C4xf323E293E_r_i, 4> {llvm.align = 8 : i64, llvm.dereferenceable_or_null = 32 : i64, llvm.noundef}, %arg1: !llvm.ptr<struct<"opencl.image1d_ro_t", opaque>, 1>)
// CHECK-NEXT:    %0 = "polygeist.memref2pointer"(%arg0) : (memref<?x!sycl_accessor_1_21sycl2Evec3C5Bf322C_45D2C_28vector3C4xf323E293E_r_i, 4>) -> !llvm.ptr<struct<(ptr<struct<"opencl.image1d_ro_t", opaque>, 1>, array<24 x i8>)>, 4>
// CHECK-NEXT:    sycl.call(%0, %arg1) {FunctionName = @imageAccessorInit, MangledFunctionName = @_ZN4sycl3_V16detail14image_accessorINS0_3vecIfLi4EEELi1ELNS0_6access4modeE1024ELNS5_6targetE2017ELNS5_11placeholderE0EE17imageAccessorInitE14ocl_image1d_ro, TypeName = @image_accessor} : (!llvm.ptr<struct<(ptr<struct<"opencl.image1d_ro_t", opaque>, 1>, array<24 x i8>)>, 4>, !llvm.ptr<struct<"opencl.image1d_ro_t", opaque>, 1>) -> ()
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// CHECK-LABEL: func.func private @_ZZZ9testImagevENKUlRN4sycl3_V17handlerEE_clES2_ENKUlNS0_4itemILi1ELb1EEEE_clES5_(%arg0: memref<?x!llvm.struct<(!sycl_accessor_1_21sycl2Evec3C5Bf322C_45D2C_28vector3C4xf323E293E_r_i)>, 4> {llvm.align = 8 : i64, llvm.dereferenceable_or_null = 32 : i64, llvm.noundef}, %arg1: memref<?x!sycl_item_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_item_1_, llvm.noundef})
// CHECK-DAG:     %c0_i32 = arith.constant 0 : i32
// CHECK-DAG:     %alloca = memref.alloca() : memref<1xi32>
// CHECK-DAG:     %0 = llvm.mlir.undef : i32
// CHECK-NEXT:    affine.store %0, %alloca[0] : memref<1xi32>
// CHECK-NEXT:    %1 = "polygeist.memref2pointer"(%arg0) : (memref<?x!llvm.struct<(!sycl_accessor_1_21sycl2Evec3C5Bf322C_45D2C_28vector3C4xf323E293E_r_i)>, 4>) -> !llvm.ptr<struct<(ptr<struct<"opencl.image1d_ro_t", opaque>, 1>, array<24 x i8>)>, 4>
// CHECK-NEXT:    %2 = affine.load %arg1[0] : memref<?x!sycl_item_1_>
// CHECK-NEXT:    %3 = "sycl.item.get_id"(%2, %c0_i32) {ArgumentTypes = [memref<?x!sycl_item_1_, 4>, i32], FunctionName = @"operator[]", MangledFunctionName = @_ZNK4sycl3_V14itemILi1ELb1EEixEi, TypeName = @item} : (!sycl_item_1_, i32) -> i64
// CHECK-NEXT:    %4 = arith.trunci %3 : i64 to i32
// CHECK-NEXT:    %5 = "polygeist.memref2pointer"(%alloca) : (memref<1xi32>) -> !llvm.ptr<i32>
// CHECK-NEXT:    %6 = llvm.addrspacecast %5 : !llvm.ptr<i32> to !llvm.ptr<i32, 4>
// CHECK-NEXT:    %7 = "polygeist.pointer2memref"(%6) : (!llvm.ptr<i32, 4>) -> memref<?xi32, 4>
// CHECK-NEXT:    llvm.store %4, %6 : !llvm.ptr<i32, 4>
// CHECK-NEXT:    %8 = sycl.call(%1, %7) {FunctionName = @read, MangledFunctionName = @_ZNK4sycl3_V16detail14image_accessorINS0_3vecIfLi4EEELi1ELNS0_6access4modeE1024ELNS5_6targetE2017ELNS5_11placeholderE0EE4readIiLi1EvEES4_RKT_, TypeName = @image_accessor} : (!llvm.ptr<struct<(ptr<struct<"opencl.image1d_ro_t", opaque>, 1>, array<24 x i8>)>, 4>, memref<?xi32, 4>) -> !sycl_vec_f32_4_
// CHECK-NEXT:    return
// CHECK-NEXT:  }

void testImage() {
  const image_channel_order ChanOrder = image_channel_order::rgba;
  const image_channel_type ChanType = image_channel_type::fp32;
  const range<1> ImgSize_1D(N);
  std::vector<float4> data_from_1D(ImgSize_1D.size(), {1, 2, 3, 4});

  {
    image<1> image_from_1D(data_from_1D.data(), ChanOrder, ChanType, ImgSize_1D);
    queue Q;
    Q.submit([&](handler &CGH) {
      auto readAcc = image_from_1D.get_access<float4, access::mode::read>(CGH);
      CGH.parallel_for<class ReadImg>(
          ImgSize_1D, [=](item<1> Item) {
            float4 Data = readAcc.read(int(Item[0]));
          });
    });
  }
}
