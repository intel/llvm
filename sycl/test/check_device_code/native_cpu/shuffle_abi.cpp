// REQUIRES: native_cpu_ock && linux

// This doesn't test every possible case since it is quite slow to compile.
// long and double are not tested as it seems to generate loops in the code
// rather than vector versions.

// RUN: %clangxx -DTYPE=int -DVEC_WIDTH=2 -DOPER=TF_SHIFT_UP -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=native_cpu -Xclang -sycl-std=2020 -mllvm -sycl-opt -mllvm -inline-threshold=500 -mllvm -sycl-native-cpu-no-vecz -mllvm -sycl-native-dump-device-ir %s | FileCheck --check-prefix UP_V2_INT %s
// RUN: %clangxx -DTYPE=short -DVEC_WIDTH=4 -DOPER=TF_SHIFT_DOWN -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=native_cpu -Xclang -sycl-std=2020 -mllvm -sycl-opt -mllvm -inline-threshold=500 -mllvm -sycl-native-cpu-no-vecz -mllvm -sycl-native-dump-device-ir %s | FileCheck --check-prefix DOWN_V4_SHORT %s
// RUN: %clangxx -DTYPE=char -DVEC_WIDTH=4 -DOPER=TF_SHIFT_XOR -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=native_cpu -Xclang -sycl-std=2020 -mllvm -sycl-opt -mllvm -inline-threshold=500 -mllvm -sycl-native-cpu-no-vecz -mllvm -sycl-native-dump-device-ir %s | FileCheck --check-prefix XOR_V4_CHAR %s
// RUN: %clangxx -DTYPE=float -DVEC_WIDTH=8 -DOPER=TF_SHIFT_UP -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=native_cpu -Xclang -sycl-std=2020 -mllvm -sycl-opt -mllvm -inline-threshold=500 -mllvm -sycl-native-cpu-no-vecz -mllvm -sycl-native-dump-device-ir %s | FileCheck --check-prefix UP_V8_FLOAT %s
// RUN: %clangxx -DTYPE="unsigned int" -DVEC_WIDTH=8 -DOPER=TF_SELECT -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=native_cpu -Xclang -sycl-std=2020 -mllvm -sycl-opt -mllvm -inline-threshold=500 -mllvm -sycl-native-cpu-no-vecz -mllvm -sycl-native-dump-device-ir %s | FileCheck --check-prefix SELECT_V8_SELECT_I32 %s

// Tests that sub-group shuffles work even when abi is different to what is
// expected

#include <sycl/sycl.hpp>

static constexpr size_t NumElems = VEC_WIDTH;
static constexpr size_t NumWorkItems = 64;

// UP_V2_INT: double @_Z30__spirv_SubgroupShuffleUpINTELIDv2_iET_S1_S1_j(double noundef %[[ARG0:[0-9]+]], double noundef %[[ARG1:[0-9]+]]
// UP_V2_INT: %[[UPV2I32_BITCAST_OP0:[0-9]+]] = bitcast double %[[ARG0]] to <2 x i32>
// UP_V2_INT: %[[UPV2I32_BITCAST_OP1:[0-9]+]] = bitcast double %[[ARG1]] to <2 x i32>
// UP_V2_INT: %[[UPV2I32_CALL_SHUFFLE:[0-9]+]] = call <2 x i32> @__mux_sub_group_shuffle_up_v2i32(<2 x i32> %[[UPV2I32_BITCAST_OP0]], <2 x i32> %[[UPV2I32_BITCAST_OP1]]
// UP_V2_INT: %[[UPV2I32_BITCAST_RESULT:[0-9]+]] = bitcast <2 x i32> %[[UPV2I32_CALL_SHUFFLE]] to double
// UP_V2_INT: ret double %[[UPV2I32_BITCAST_RESULT]]

// DOWN_V4_SHORT: double @_Z32__spirv_SubgroupShuffleDownINTELIDv4_sET_S1_S1_j(double noundef %[[ARG0:[0-9]+]], double noundef %[[ARG1:[0-9]+]]
// DOWN_V4_SHORT: %[[DOWNV4I16_BITCAST_OP0:[0-9]+]] = bitcast double %[[ARG0]] to <4 x i16>
// DOWN_V4_SHORT: %[[DOWNV4I16_BITCAST_OP1:[0-9]+]] = bitcast double %[[ARG1]] to <4 x i16>
// DOWN_V4_SHORT: %[[DOWNV4I16_CALL_SHUFFLE:[0-9]+]] = call <4 x i16> @__mux_sub_group_shuffle_down_v4i16(<4 x i16> %[[DOWNV4I16_BITCAST_OP0]], <4 x i16> %[[DOWNV4I16_BITCAST_OP1]]
// DOWN_V4_SHORT: %[[DOWNV4I16_BITCAST_RESULT:[0-9]+]] = bitcast <4 x i16> %[[DOWNV4I16_CALL_SHUFFLE]] to double
// DOWN_V4_SHORT: ret double %[[DOWNV4I16_BITCAST_RESULT]]

// XOR_V4_CHAR: i32 @_Z31__spirv_SubgroupShuffleXorINTELIDv4_aET_S1_j(i32 noundef %[[ARG0:[0-9]+]], i32
// XOR_V4_CHAR: %[[XORV4I8_BITCAST_OP0:[0-9]+]] = bitcast i32 %[[ARG0]] to <4 x i8>
// XOR_V4_CHAR: %[[XORV4I8_CALL_SHUFFLE:[0-9]+]] = call <4 x i8> @__mux_sub_group_shuffle_xor_v4i8(<4 x i8> %[[XORV4I8_BITCAST_OP0]], i32
// XOR_V4_CHAR: %[[XORV4I8_BITCAST_RESULT:[0-9]+]] = bitcast <4 x i8> %[[XORV4I8_CALL_SHUFFLE]] to i32
// XOR_V4_CHAR: ret i32 %[[XORV4I8_BITCAST_RESULT]]

// UP_V8_FLOAT: <8 x float> @_Z30__spirv_SubgroupShuffleUpINTELIDv8_fET_S1_S1_j(ptr noundef byval(<8 x float>) align 32 %[[ARG0:[0-9]+]], ptr noundef byval(<8 x float>) align 32 %[[ARG1:[0-9]+]]
// UP_V8_FLOAT:   %[[UPV8F32_BYVAL_LOAD_OP0:[0-9]+]] = load <8 x float>, ptr %[[ARG0]], align 32
// UP_V8_FLOAT: %[[UPV8F32_BYVAL_LOAD_OP1:[0-9]+]] = load <8 x float>, ptr %[[ARG1]], align 32
// UP_V8_FLOAT:   %[[UPV8F32_CALL_SHUFFLE:[0-9]+]] = call <8 x float> @__mux_sub_group_shuffle_up_v8f32(<8 x float> %[[UPV8F32_BYVAL_LOAD_OP0]], <8 x float> %[[UPV8F32_BYVAL_LOAD_OP1]], i32
// UP_V8_FLOAT:   ret <8 x float> %[[UPV8F32_CALL_SHUFFLE:[0-9]+]]

// SELECT_V8_SELECT_I32: <8 x i32> @_Z28__spirv_SubgroupShuffleINTELIDv8_jET_S1_j(ptr noundef byval(<8 x i32>) align 32 %[[ARG0:[0-9]+]],
// SELECT_V8_SELECT_I32: %[[SELV8I32_BYVAL_LOAD_OP0:[0-9]+]] = load <8 x i32>, ptr %[[ARG0]], align 32
// SELECT_V8_SELECT_I32:   %[[SELV8I32_CALL_SHUFFLE:[0-9]+]] = call <8 x i32> @__mux_sub_group_shuffle_v8i32(<8 x i32> %[[SELV8I32_BYVAL_LOAD_OP0]], i32
// SELECT_V8_SELECT_I32:   ret <8 x i32> %[[SELV8I32_CALL_SHUFFLE:[0-9]+]]

enum TEST_FUNC_CHOICE { TF_SHIFT_DOWN, TF_SHIFT_UP, TF_SHIFT_XOR, TF_SELECT };

template <typename ShiftType, enum TEST_FUNC_CHOICE Choice>
void ShuffleOpTest() {
  sycl::queue Q;

  ShiftType ShiftRes[NumWorkItems];

  {
    sycl::buffer<ShiftType, 1> ShuffleResBuf{ShiftRes, NumWorkItems};

    Q.submit([&](sycl::handler &CGH) {
      sycl::accessor ShuffleRes{ShuffleResBuf, CGH, sycl::write_only};

      CGH.parallel_for(
          sycl::nd_range<1>{sycl::range<1>{NumWorkItems},
                            sycl::range<1>{NumWorkItems}},
          [=](sycl::nd_item<1> It) {
            int GID = It.get_global_linear_id();
            ShiftType ItemVal{0};
            for (int I = 0; I < NumElems; ++I)
              ItemVal[I] = I;

            sycl::sub_group SG = It.get_sub_group();
            if (Choice == TF_SHIFT_DOWN) {
              ShuffleRes[GID] = sycl::shift_group_left(SG, ItemVal);
            } else if (Choice == TF_SHIFT_UP) {
              ShuffleRes[GID] = sycl::shift_group_right(SG, ItemVal);
            } else if (Choice == TF_SHIFT_XOR) {
              ShuffleRes[GID] = sycl::permute_group_by_xor(SG, ItemVal, 1);
            } else if (Choice == TF_SELECT) {
              ShuffleRes[GID] = sycl::select_from_group(SG, ItemVal, 1);
            }
          });
    });
  }
}

int main() {
  ShuffleOpTest<sycl::vec<TYPE, VEC_WIDTH>, OPER>();
  return 0;
}
