// REQUIRES: native_cpu_ock && linux
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=native_cpu -Xclang -sycl-std=2020 -mllvm -sycl-opt -mllvm -inline-threshold=500 -mllvm -sycl-native-cpu-no-vecz -mllvm -sycl-native-dump-device-ir %s | FileCheck %s

// Tests that sub-group shuffles work even when abi is different to what is expected

#include <sycl/detail/core.hpp>
#include <sycl/group_algorithm.hpp>
#include <sycl/marray.hpp>

static constexpr size_t NumElems = 5;
static constexpr size_t NumWorkItems = 64;


// CHECK: define internal double @__mux_sub_group_shuffle_up_v2i32_abi_wrapper
// CHECK: %[[UPV2I32_BITCAST_OP0:[0-9]+]] = bitcast double %0 to <2 x i32>
// CHECK: %[[UPV2I32_BITCAST_OP1:[0-9]+]] = bitcast double %1 to <2 x i32>
// CHECK: %[[UPV2I32_CALL_SHUFFLE:[0-9]+]] = call <2 x i32> @__mux_sub_group_shuffle_up_v2i32(<2 x i32> %[[UPV2I32_BITCAST_OP0]], <2 x i32> %[[UPV2I32_BITCAST_OP1]]
// CHECK: %[[UPV2I32_BITCAST_RESULT:[0-9]+]] = bitcast <2 x i32> %[[UPV2I32_CALL_SHUFFLE]] to double
// CHECK: ret double %[[UPV2I32_BITCAST_RESULT]]

// CHECK: define internal double @__mux_sub_group_shuffle_down_v2i32_abi_wrapper
// CHECK: %[[DOWNV2I32_BITCAST_OP0:[0-9]+]] = bitcast double %0 to <2 x i32>
// CHECK: %[[DOWNV2I32_BITCAST_OP1:[0-9]+]] = bitcast double %1 to <2 x i32>
// CHECK: %[[DOWNV2I32_CALL_SHUFFLE:[0-9]+]] = call <2 x i32> @__mux_sub_group_shuffle_down_v2i32(<2 x i32> %[[DOWNV2I32_BITCAST_OP0]], <2 x i32> %[[DOWNV2I32_BITCAST_OP1]]
// CHECK: %[[DOWNV2I32_BITCAST_RESULT:[0-9]+]] = bitcast <2 x i32> %[[DOWNV2I32_CALL_SHUFFLE]] to double
// CHECK: ret double %[[DOWNV2I32_BITCAST_RESULT]]

// CHECK: define internal double @__mux_sub_group_shuffle_xor_v2i32_abi_wrapper(double noundef %0, i32 noundef %1)

// CHECK-DAG: define internal <8 x float> @__mux_sub_group_shuffle_up_v8f32_abi_wrapper(ptr noundef byval(<8 x float>) align 32 %0, ptr noundef byval(<8 x float>) align 32 %1
// CHECK:   %[[UPV8F32_BYVAL_LOAD_OP0:[0-9]+]] = load <8 x float>, ptr %0, align 32
// CHECK:   %[[UPV8F32_BYVAL_LOAD_OP1:[0-9]+]] = load <8 x float>, ptr %1, align 32
// CHECK:   %[[UPV8F32_CALL_SHUFFLE:[0-9]+]] = call <8 x float> @__mux_sub_group_shuffle_up_v8f32(<8 x float> %[[UPV8F32_BYVAL_LOAD_OP0]], <8 x float> %[[UPV8F32_BYVAL_LOAD_OP1]], i32 %2)
// CHECK:   ret <8 x float> %[[UPV8F32_CALL_SHUFFLE:[0-9]+]]

// CHECK-DAG: define internal <8 x float> @__mux_sub_group_shuffle_down_v8f32_abi_wrapper(ptr noundef byval(<8 x float>) align 32 %0, ptr noundef byval(<8 x float>) align 32 %1
// CHECK:   %[[DOWNV8F32_BYVAL_LOAD_OP0:[0-9]+]] = load <8 x float>, ptr %0, align 32
// CHECK:   %[[DOWNV8F32_BYVAL_LOAD_OP1:[0-9]+]] = load <8 x float>, ptr %1, align 32
// CHECK:   %[[DOWNV8F32_CALL_SHUFFLE:[0-9]+]] = call <8 x float> @__mux_sub_group_shuffle_down_v8f32(<8 x float> %[[DOWNV8F32_BYVAL_LOAD_OP0]], <8 x float> %[[DOWNV8F32_BYVAL_LOAD_OP1]], i32 %2)
// CHECK:   ret <8 x float> %[[DOWNV8F32_CALL_SHUFFLE:[0-9]+]]

// CHECK-DAG: define internal <8 x float> @__mux_sub_group_shuffle_xor_v8f32_abi_wrapper(ptr noundef byval(<8 x float>) align 32 %0

template <typename ShiftType>
void ShiftLeftRightTest()
{
  sycl::queue Q;

  ShiftType ShiftLeftRes[NumWorkItems];
  ShiftType ShiftRightRes[NumWorkItems];
  ShiftType PermuteXorRes[NumWorkItems];
  ShiftType SelectRes[NumWorkItems];
  unsigned SubGroupSize = 0;

  {
    sycl::buffer<ShiftType, 1> ShiftLeftResBuff{ShiftLeftRes, NumWorkItems};
    sycl::buffer<ShiftType, 1> ShiftRightResBuff{ShiftRightRes, NumWorkItems};
    sycl::buffer<ShiftType, 1> PermuteXorResBuff{PermuteXorRes, NumWorkItems};
    sycl::buffer<ShiftType, 1> SelectResBuff{SelectRes, NumWorkItems};
    sycl::buffer<unsigned, 1> SubGroupSizeBuff{&SubGroupSize, 1};

    Q.submit([&](sycl::handler &CGH) {
      sycl::accessor ShiftLeftResAcc{ShiftLeftResBuff, CGH, sycl::write_only};
      sycl::accessor ShiftRightResAcc{ShiftRightResBuff, CGH, sycl::write_only};
      sycl::accessor PermuteXorResAcc{PermuteXorResBuff, CGH, sycl::write_only};
      sycl::accessor SubGroupSizeAcc{SubGroupSizeBuff, CGH, sycl::write_only};

      CGH.parallel_for(
          sycl::nd_range<1>{sycl::range<1>{NumWorkItems},
                            sycl::range<1>{NumWorkItems}},
          [=](sycl::nd_item<1> It) {
            int GID = It.get_global_linear_id();
            int ValueOffset = GID * NumElems;
            ShiftType ItemVal{0};
            for (int I = 0; I < NumElems; ++I)
              ItemVal[I] = ValueOffset + I;

            sycl::sub_group SG = It.get_sub_group();
            if (GID == 0)
              SubGroupSizeAcc[0] = SG.get_local_linear_range();

            ShiftLeftResAcc[GID] = sycl::shift_group_left(SG, ItemVal);
            ShiftRightResAcc[GID] = sycl::shift_group_right(SG, ItemVal);
            PermuteXorResAcc[GID] = sycl::permute_group_by_xor(SG, ItemVal, 1);
          });
    });
  }
}

int main() {
  ShiftLeftRightTest<sycl::vec<int, 2>>();
  ShiftLeftRightTest<sycl::vec<float, 8>>();  
  return 0;
}
