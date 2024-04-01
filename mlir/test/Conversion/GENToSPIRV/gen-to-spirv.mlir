// RUN: mlir-opt -convert-gen-to-spirv -split-input-file %s | FileCheck %s

// CHECK-DAG:     spirv.GlobalVariable @__spirv_BuiltInNumWorkgroups built_in("NumWorkgroups") : !spirv.ptr<vector<3xi32>, Input>
// CHECK-DAG:     spirv.GlobalVariable @__spirv_BuiltInWorkgroupSize built_in("WorkgroupSize") : !spirv.ptr<vector<3xi32>, Input>
// CHECK-DAG:     spirv.GlobalVariable @__spirv_BuiltInWorkgroupId built_in("WorkgroupId") : !spirv.ptr<vector<3xi32>, Input>
// CHECK-DAG:     spirv.GlobalVariable @__spirv_BuiltInLocalInvocationId built_in("LocalInvocationId") : !spirv.ptr<vector<3xi32>, Input>

// CHECK-LABEL:   func.func @gen_nd_range(
// CHECK-SAME:                            %[[VAL_0:.*]]: i32) {
func.func @gen_nd_range(%dim: i32) {
// CHECK:           %[[VAL_1:.*]] = spirv.mlir.addressof @__spirv_BuiltInLocalInvocationId : !spirv.ptr<vector<3xi32>, Input>
// CHECK:           %[[VAL_2:.*]] = spirv.Load "Input" %[[VAL_1]] : vector<3xi32>
// CHECK:           %[[VAL_3:.*]] = spirv.VectorExtractDynamic %[[VAL_2]]{{\[}}%[[VAL_0]]] : vector<3xi32>, i32
  %0 = gen.local_id %dim
// CHECK:           %[[VAL_4:.*]] = spirv.mlir.addressof @__spirv_BuiltInWorkgroupId : !spirv.ptr<vector<3xi32>, Input>
// CHECK:           %[[VAL_5:.*]] = spirv.Load "Input" %[[VAL_4]] : vector<3xi32>
// CHECK:           %[[VAL_6:.*]] = spirv.VectorExtractDynamic %[[VAL_5]]{{\[}}%[[VAL_0]]] : vector<3xi32>, i32
  %1 = gen.work_group_id %dim
// CHECK:           %[[VAL_7:.*]] = spirv.mlir.addressof @__spirv_BuiltInWorkgroupSize : !spirv.ptr<vector<3xi32>, Input>
// CHECK:           %[[VAL_8:.*]] = spirv.Load "Input" %[[VAL_7]] : vector<3xi32>
// CHECK:           %[[VAL_9:.*]] = spirv.VectorExtractDynamic %[[VAL_8]]{{\[}}%[[VAL_0]]] : vector<3xi32>, i32
  %2 = gen.work_group_size %dim
// CHECK:           %[[VAL_10:.*]] = spirv.mlir.addressof @__spirv_BuiltInNumWorkgroups : !spirv.ptr<vector<3xi32>, Input>
// CHECK:           %[[VAL_11:.*]] = spirv.Load "Input" %[[VAL_10]] : vector<3xi32>
// CHECK:           %[[VAL_12:.*]] = spirv.VectorExtractDynamic %[[VAL_11]]{{\[}}%[[VAL_0]]] : vector<3xi32>, i32
  %3 = gen.num_work_groups %dim
  func.return
}
