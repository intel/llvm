// RUN: clang++ -fsycl -fsycl-device-only -fsycl-targets=spir64-unknown-unknown-syclmlir -O0 -w -emit-mlir -o - %s | FileCheck %s
// RUN: clang++ -fsycl -fsycl-device-only -fsycl-targets=spir64-unknown-unknown-syclmlir -O0 -w -emit-llvm -S -o %t %s && rm %t

#include <CL/__spirv/spirv_types.hpp>
#include <CL/__spirv/spirv_vars.hpp>
#include <sycl/sycl.hpp>

#define ND_TEST(name, prefix, type)                                            \
  template <int Dims> SYCL_EXTERNAL type<Dims> test##name() {                  \
    return __spirv::prefix##name<Dims, type<Dims>>();                          \
  }                                                                            \
                                                                               \
  template type<1> test##name<1>();                                            \
  template type<2> test##name<2>();                                            \
  template type<3> test##name<3>();

#define INIT_TEST(name, type) ND_TEST(name, init, type)
#define INIT_ID_TEST(name) ND_TEST(name, init, sycl::id)
#define INIT_RANGE_TEST(name) ND_TEST(name, init, sycl::range)

// CHECK-LABEL:     func.func @_Z14testGlobalSizeILi1EEN4sycl3_V15rangeIXT_EEEv() -> !sycl_range_1_
// CHECK:             %[[VAL_151:.*]] = sycl.num_work_items  : !sycl_range_1_
// CHECK:             return %[[VAL_151]] : !sycl_range_1_
// CHECK:           }

// CHECK-LABEL:     func.func @_Z14testGlobalSizeILi2EEN4sycl3_V15rangeIXT_EEEv() -> !sycl_range_2_
// CHECK:             %[[VAL_152:.*]] = sycl.num_work_items  : !sycl_range_2_
// CHECK:             return %[[VAL_152]] : !sycl_range_2_
// CHECK:           }

// CHECK-LABEL:     func.func @_Z14testGlobalSizeILi3EEN4sycl3_V15rangeIXT_EEEv() -> !sycl_range_3_
// CHECK:             %[[VAL_153:.*]] = sycl.num_work_items  : !sycl_range_3_
// CHECK:             return %[[VAL_153]] : !sycl_range_3_
// CHECK:           }
INIT_RANGE_TEST(GlobalSize);

// CHECK-LABEL:     func.func @_Z22testGlobalInvocationIdILi1EEN4sycl3_V12idIXT_EEEv() -> !sycl_id_1_
// CHECK:             %[[VAL_154:.*]] = sycl.global_id  : !sycl_id_1_
// CHECK:             return %[[VAL_154]] : !sycl_id_1_
// CHECK:           }

// CHECK-LABEL:     func.func @_Z22testGlobalInvocationIdILi2EEN4sycl3_V12idIXT_EEEv() -> !sycl_id_2_
// CHECK:             %[[VAL_155:.*]] = sycl.global_id  : !sycl_id_2_
// CHECK:             return %[[VAL_155]] : !sycl_id_2_
// CHECK:           }

// CHECK-LABEL:     func.func @_Z22testGlobalInvocationIdILi3EEN4sycl3_V12idIXT_EEEv() -> !sycl_id_3_
// CHECK:             %[[VAL_156:.*]] = sycl.global_id  : !sycl_id_3_
// CHECK:             return %[[VAL_156]] : !sycl_id_3_
// CHECK:           }
INIT_ID_TEST(GlobalInvocationId)

// CHECK-LABEL:     func.func @_Z17testWorkgroupSizeILi1EEN4sycl3_V15rangeIXT_EEEv() -> !sycl_range_1_
// CHECK:             %[[VAL_157:.*]] = sycl.work_group_size  : !sycl_range_1_
// CHECK:             return %[[VAL_157]] : !sycl_range_1_
// CHECK:           }

// CHECK-LABEL:     func.func @_Z17testWorkgroupSizeILi2EEN4sycl3_V15rangeIXT_EEEv() -> !sycl_range_2_
// CHECK:             %[[VAL_158:.*]] = sycl.work_group_size  : !sycl_range_2_
// CHECK:             return %[[VAL_158]] : !sycl_range_2_
// CHECK:           }

// CHECK-LABEL:     func.func @_Z17testWorkgroupSizeILi3EEN4sycl3_V15rangeIXT_EEEv() -> !sycl_range_3_
// CHECK:             %[[VAL_159:.*]] = sycl.work_group_size  : !sycl_range_3_
// CHECK:             return %[[VAL_159]] : !sycl_range_3_
// CHECK:           }
INIT_RANGE_TEST(WorkgroupSize)

// CHECK-LABEL:     func.func @_Z17testNumWorkgroupsILi1EEN4sycl3_V15rangeIXT_EEEv() -> !sycl_range_1_
// CHECK:             %[[VAL_160:.*]] = sycl.num_work_groups  : !sycl_range_1_
// CHECK:             return %[[VAL_160]] : !sycl_range_1_
// CHECK:           }

// CHECK-LABEL:     func.func @_Z17testNumWorkgroupsILi2EEN4sycl3_V15rangeIXT_EEEv() -> !sycl_range_2_
// CHECK:             %[[VAL_161:.*]] = sycl.num_work_groups  : !sycl_range_2_
// CHECK:             return %[[VAL_161]] : !sycl_range_2_
// CHECK:           }

// CHECK-LABEL:     func.func @_Z17testNumWorkgroupsILi3EEN4sycl3_V15rangeIXT_EEEv() -> !sycl_range_3_
// CHECK:             %[[VAL_162:.*]] = sycl.num_work_groups  : !sycl_range_3_
// CHECK:             return %[[VAL_162]] : !sycl_range_3_
// CHECK:           }
INIT_RANGE_TEST(NumWorkgroups)

// CHECK-LABEL:     func.func @_Z21testLocalInvocationIdILi1EEN4sycl3_V12idIXT_EEEv() -> !sycl_id_1_
// CHECK:             %[[VAL_163:.*]] = sycl.local_id  : !sycl_id_1_
// CHECK:             return %[[VAL_163]] : !sycl_id_1_
// CHECK:           }

// CHECK-LABEL:     func.func @_Z21testLocalInvocationIdILi2EEN4sycl3_V12idIXT_EEEv() -> !sycl_id_2_
// CHECK:             %[[VAL_164:.*]] = sycl.local_id  : !sycl_id_2_
// CHECK:             return %[[VAL_164]] : !sycl_id_2_
// CHECK:           }

// CHECK-LABEL:     func.func @_Z21testLocalInvocationIdILi3EEN4sycl3_V12idIXT_EEEv() -> !sycl_id_3_
// CHECK:             %[[VAL_165:.*]] = sycl.local_id  : !sycl_id_3_
// CHECK:             return %[[VAL_165]] : !sycl_id_3_
// CHECK:           }
INIT_ID_TEST(LocalInvocationId)

// CHECK-LABEL:     func.func @_Z15testWorkgroupIdILi1EEN4sycl3_V12idIXT_EEEv() -> !sycl_id_1_
// CHECK:             %[[VAL_166:.*]] = sycl.work_group_id  : !sycl_id_1_
// CHECK:             return %[[VAL_166]] : !sycl_id_1_
// CHECK:           }

// CHECK-LABEL:     func.func @_Z15testWorkgroupIdILi2EEN4sycl3_V12idIXT_EEEv() -> !sycl_id_2_
// CHECK:             %[[VAL_167:.*]] = sycl.work_group_id  : !sycl_id_2_
// CHECK:             return %[[VAL_167]] : !sycl_id_2_
// CHECK:           }

// CHECK-LABEL:     func.func @_Z15testWorkgroupIdILi3EEN4sycl3_V12idIXT_EEEv() -> !sycl_id_3_
// CHECK:             %[[VAL_168:.*]] = sycl.work_group_id  : !sycl_id_3_
// CHECK:             return %[[VAL_168]] : !sycl_id_3_
// CHECK:           }
INIT_ID_TEST(WorkgroupId)

// CHECK-LABEL:     func.func @_Z16testGlobalOffsetILi1EEN4sycl3_V12idIXT_EEEv() -> !sycl_id_1_
// CHECK:             %[[VAL_169:.*]] = sycl.global_offset  : !sycl_id_1_
// CHECK:             return %[[VAL_169]] : !sycl_id_1_
// CHECK:           }

// CHECK-LABEL:     func.func @_Z16testGlobalOffsetILi2EEN4sycl3_V12idIXT_EEEv() -> !sycl_id_2_
// CHECK:             %[[VAL_170:.*]] = sycl.global_offset  : !sycl_id_2_
// CHECK:             return %[[VAL_170]] : !sycl_id_2_
// CHECK:           }

// CHECK-LABEL:     func.func @_Z16testGlobalOffsetILi3EEN4sycl3_V12idIXT_EEEv() -> !sycl_id_3_
// CHECK:             %[[VAL_171:.*]] = sycl.global_offset  : !sycl_id_3_
// CHECK:             return %[[VAL_171]] : !sycl_id_3_
// CHECK:           }
INIT_ID_TEST(GlobalOffset)

// CHECK-LABEL:     func.func @_Z16testSubgroupSizev() -> (i32 {llvm.noundef})
// CHECK:             %[[VAL_172:.*]] = sycl.sub_group_size : i32
// CHECK:             return %[[VAL_172]] : i32
// CHECK:           }
SYCL_EXTERNAL uint32_t testSubgroupSize() { return __spirv_SubgroupSize(); }

// CHECK-LABEL:     func.func @_Z19testSubgroupMaxSizev() -> (i32 {llvm.noundef})
// CHECK:             %[[VAL_173:.*]] = sycl.sub_group_max_size : i32
// CHECK:             return %[[VAL_173]] : i32
// CHECK:           }
SYCL_EXTERNAL uint32_t testSubgroupMaxSize() {
  return __spirv_SubgroupMaxSize();
}

// CHECK-LABEL:     func.func @_Z16testNumSubgroupsv() -> (i32 {llvm.noundef})
// CHECK:             %[[VAL_174:.*]] = sycl.num_sub_groups : i32
// CHECK:             return %[[VAL_174]] : i32
// CHECK:           }
SYCL_EXTERNAL uint32_t testNumSubgroups() { return __spirv_NumSubgroups(); }

// CHECK-LABEL:     func.func @_Z14testSubgroupIdv() -> (i32 {llvm.noundef})
// CHECK:             %[[VAL_175:.*]] = sycl.sub_group_id : i32
// CHECK:             return %[[VAL_175]] : i32
// CHECK:           }
SYCL_EXTERNAL uint32_t testSubgroupId() { return __spirv_SubgroupId(); }

// CHECK-LABEL:     func.func @_Z29testSubgroupLocalInvocationIdv() -> (i32 {llvm.noundef})
// CHECK:             %[[VAL_176:.*]] = sycl.sub_group_local_id : i32
// CHECK:             return %[[VAL_176]] : i32
// CHECK:           }
SYCL_EXTERNAL uint32_t testSubgroupLocalInvocationId() {
  return __spirv_SubgroupLocalInvocationId();
}
