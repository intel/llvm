// RUN: polygeist-opt --loop-internalization --split-input-file -allow-unregistered-dialect %s | FileCheck %s --check-prefixes=CHECK,SIZE1
// RUN: polygeist-opt --loop-internalization --loop-internalization-tile-sizes=2 --split-input-file -allow-unregistered-dialect %s | FileCheck %s --check-prefixes=CHECK,SIZE2

// CHECK-DAG:   [[MAP1:#map.*]] = affine_map<()[s0] -> (256 ceildiv s0)>
// CHECK-DAG:   [[MAP2:#map.*]] = affine_map<(d0)[s0] -> (d0 * s0)>
// CHECK-DAG:   [[MAP3:#map.*]] = affine_map<(d0)[s0] -> (d0 * s0 + s0, 256)>
// CHECK-LABEL: func.func @affine_1d() {
// SIZE1-NEXT:    [[TILESIZE:%.*]] = arith.constant 1 : index
// SIZE2-NEXT:    [[TILESIZE:%.*]] = arith.constant 2 : index
// CHECK-NEXT:    affine.for %arg0 = 0 to [[MAP1]]()[[[TILESIZE]]] {
// CHECK-NEXT:      spirv.ControlBarrier <Workgroup>, <Workgroup>, <SequentiallyConsistent|WorkgroupMemory>
// CHECK-NEXT:      affine.for %arg1 = [[MAP2]](%arg0)[[[TILESIZE]]] to min [[MAP3]](%arg0)[[[TILESIZE]]] {
// CHECK-NEXT:        "test.foo"(%arg1) : (index) -> ()
// CHECK-NEXT:      }
// CHECK-NEXT:      spirv.ControlBarrier <Workgroup>, <Workgroup>, <SequentiallyConsistent|WorkgroupMemory>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @affine_1d() {
  affine.for %i = 0 to 256 {
    "test.foo"(%i) : (index) -> ()
  }
  return
}

// -----

// CHECK-DAG:   [[MAP1:#map.*]] = affine_map<()[s0] -> (256 ceildiv s0)>
// CHECK-DAG:   [[MAP2:#map.*]] = affine_map<()[s0] -> (511 ceildiv s0 + 1)>
// CHECK-DAG:   [[MAP3:#map.*]] = affine_map<(d0)[s0] -> (d0 * s0)>
// CHECK-DAG:   [[MAP4:#map.*]] = affine_map<(d0)[s0] -> (d0 * s0 + s0, 256)>
// CHECK-DAG:   [[MAP5:#map.*]] = affine_map<(d0)[s0] -> ((d0 - 1) * s0 + 1)>
// CHECK-DAG:   [[MAP6:#map.*]] = affine_map<(d0)[s0] -> ((d0 - 1) * s0 + s0 + 1, 512)>
// CHECK-LABEL: func.func @affine_2d() {
// SIZE1-NEXT:    [[TILESIZE:%.*]] = arith.constant 1 : index
// SIZE2-NEXT:    [[TILESIZE:%.*]] = arith.constant 2 : index
// SIZE2-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    affine.for %arg0 = 0 to [[MAP1]]()[[[TILESIZE]]] {
// CHECK-NEXT:      affine.for %arg1 = 1 to [[MAP2]]()[%c1] {
// CHECK-NEXT:        spirv.ControlBarrier <Workgroup>, <Workgroup>, <SequentiallyConsistent|WorkgroupMemory>
// CHECK-NEXT:        affine.for %arg2 = [[MAP3]](%arg0)[[[TILESIZE]]] to min [[MAP4]](%arg0)[[[TILESIZE]]] {
// CHECK-NEXT:          affine.for %arg3 = [[MAP5]](%arg1)[%c1] to min [[MAP6]](%arg1)[%c1] {
// CHECK-NEXT:            "test.foo"(%arg2, %arg3) : (index, index) -> ()
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        spirv.ControlBarrier <Workgroup>, <Workgroup>, <SequentiallyConsistent|WorkgroupMemory>
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @affine_2d() {
  affine.for %i = 0 to 256 {
    affine.for %j = 1 to 512 {
      "test.foo"(%i, %j) : (index, index) -> ()
    }
  }
  return
}

// -----

// CHECK-LABEL: func.func @scf_1d(%arg0: memref<?x?xf32>) {
// CHECK-DAG:     %c0 = arith.constant 0 : index
// CHECK-DAG:     %c1 = arith.constant 1 : index
// CHECK-DAG:     %c256 = arith.constant 256 : index
// SIZE1-DAG:     [[TILESIZE:%.*]] = arith.constant 1 : index
// SIZE2-DAG:     [[TILESIZE:%.*]] = arith.constant 2 : index
// CHECK-NEXT:    %0 = arith.muli %c1, [[TILESIZE]] : index
// CHECK-NEXT:    scf.for %arg1 = %c0 to %c256 step %0 {
// CHECK-NEXT:      spirv.ControlBarrier <Workgroup>, <Workgroup>, <SequentiallyConsistent|WorkgroupMemory>
// CHECK-NEXT:      [[VAL_1:%.*]] = arith.addi %arg1, %0 : index
// CHECK-NEXT:      [[VAL_2:%.*]] = arith.cmpi slt, %c256, [[VAL_1]] : index
// CHECK-NEXT:      [[VAL_3:%.*]] = arith.select [[VAL_2]], %c256, [[VAL_1]] : index
// CHECK-NEXT:      scf.for %arg2 = %arg1 to [[VAL_3]] step %c1 {
// CHECK-NEXT:        "test.foo"(%arg2) : (index) -> ()
// CHECK-NEXT:      }
// CHECK-NEXT:      spirv.ControlBarrier <Workgroup>, <Workgroup>, <SequentiallyConsistent|WorkgroupMemory>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @scf_1d(%arg0: memref<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c256 = arith.constant 256 : index
  scf.for %i = %c0 to %c256 step %c1 {
    "test.foo"(%i) : (index) -> ()
  }
  return
}

// -----

// CHECK-LABEL: func.func @scf_2d(%arg0: memref<?x?xf32>) {
// CHECK-DAG:     %c0 = arith.constant 0 : index
// CHECK-DAG:     %c1 = arith.constant 1 : index
// CHECK-DAG:     %c256 = arith.constant 256 : index
// CHECK-DAG:     %c512 = arith.constant 512 : index
// SIZE1-DAG:     [[TILESIZE:%.*]] = arith.constant 1 : index
// SIZE2-DAG:     [[TILESIZE:%.*]] = arith.constant 2 : index
// SIZE2-DAG:     %c1_0 = arith.constant 1 : index
// CHECK-NEXT:    %0 = arith.muli %c1, [[TILESIZE]] : index
// CHECK-NEXT:    scf.for %arg1 = %c0 to %c256 step %0 {
// CHECK-NEXT:      %1 = arith.muli %c1, %c1_0 : index
// CHECK-NEXT:      scf.for %arg2 = %c1 to %c512 step %1 {
// CHECK-NEXT:        spirv.ControlBarrier <Workgroup>, <Workgroup>, <SequentiallyConsistent|WorkgroupMemory>
// CHECK-NEXT:        [[VAL_2:%.*]] = arith.addi %arg1, %0 : index
// CHECK-NEXT:        [[VAL_3:%.*]] = arith.cmpi slt, %c256, [[VAL_2]] : index
// CHECK-NEXT:        [[VAL_4:%.*]] = arith.select [[VAL_3]], %c256, [[VAL_2]] : index
// CHECK-NEXT:        scf.for %arg3 = %arg1 to [[VAL_4]] step %c1 {
// CHECK-NEXT:          [[VAL_5:%.*]] = arith.addi %arg2, %1 : index
// CHECK-NEXT:          [[VAL_6:%.*]] = arith.cmpi slt, %c512, [[VAL_5]] : index
// CHECK-NEXT:          [[VAL_7:%.*]] = arith.select [[VAL_6]], %c512, [[VAL_5]] : index
// CHECK-NEXT:          scf.for %arg4 = %arg2 to [[VAL_7]] step %c1 {
// CHECK-NEXT:            "test.foo"(%arg3, %arg4) : (index, index) -> ()
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        spirv.ControlBarrier <Workgroup>, <Workgroup>, <SequentiallyConsistent|WorkgroupMemory>
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @scf_2d(%arg0: memref<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c256 = arith.constant 256 : index
  %c512 = arith.constant 512 : index
  scf.for %i = %c0 to %c256 step %c1 {
    scf.for %j = %c1 to %c512 step %c1 {
      "test.foo"(%i, %j) : (index, index) -> ()
    }
  }
  return
}
