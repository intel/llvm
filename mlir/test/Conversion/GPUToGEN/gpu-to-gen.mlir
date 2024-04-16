// RUN: mlir-opt -split-input-file -convert-gpu-to-gen %s | FileCheck %s

gpu.module @local_id_kernels {
  // CHECK-LABEL: gen_local_id_x
  gpu.func @gen_local_id_x() kernel {
    // CHECK: [[DIM:%.*]] = arith.constant 0 : i32
    // CHECK: gen.local_id [[DIM]]
    %0 = gpu.thread_id x
    gpu.return
  }

  // CHECK-LABEL: gen_local_id_y
  gpu.func @gen_local_id_y() kernel {
    // CHECK: [[DIM:%.*]] = arith.constant 1 : i32
    // CHECK: gen.local_id [[DIM]]
    %0 = gpu.thread_id y
    gpu.return
  }

  // CHECK-LABEL: gen_local_id_z
  gpu.func @gen_local_id_z() kernel {
    // CHECK: [[DIM:%.*]] = arith.constant 2 : i32
    // CHECK: gen.local_id [[DIM]]
    %0 = gpu.thread_id z
    gpu.return
  }
}

// -----


gpu.module @work_group_id_kernels {
  // CHECK-LABEL: gen_work_group_id_x
  gpu.func @gen_work_group_id_x() kernel {
    // CHECK: [[DIM:%.*]] = arith.constant 0 : i32
    // CHECK: gen.work_group_id [[DIM]]
    %0 = gpu.block_id x
    gpu.return
  }

  // CHECK-LABEL: gen_work_group_id_y
  gpu.func @gen_work_group_id_y() kernel {
    // CHECK: [[DIM:%.*]] = arith.constant 1 : i32
    // CHECK: gen.work_group_id [[DIM]]
    %0 = gpu.block_id y
    gpu.return
  }

  // CHECK-LABEL: gen_work_group_id_z
  gpu.func @gen_work_group_id_z() kernel {
    // CHECK: [[DIM:%.*]] = arith.constant 2 : i32
    // CHECK: gen.work_group_id [[DIM]]
    %0 = gpu.block_id z
    gpu.return
  }
}

// -----


gpu.module @work_group_size_kernels {
  // CHECK-LABEL: gen_work_group_size_x
  gpu.func @gen_work_group_size_x() kernel {
    // CHECK: [[DIM:%.*]] = arith.constant 0 : i32
    // CHECK: gen.work_group_size [[DIM]]
    %0 = gpu.block_dim x
    gpu.return
  }

  // CHECK-LABEL: gen_work_group_size_y
  gpu.func @gen_work_group_size_y() kernel {
    // CHECK: [[DIM:%.*]] = arith.constant 1 : i32
    // CHECK: gen.work_group_size [[DIM]]
    %0 = gpu.block_dim y
    gpu.return
  }

  // CHECK-LABEL: gen_work_group_size_z
  gpu.func @gen_work_group_size_z() kernel {
    // CHECK: [[DIM:%.*]] = arith.constant 2 : i32
    // CHECK: gen.work_group_size [[DIM]]
    %0 = gpu.block_dim z
    gpu.return
  }
}

// -----


gpu.module @num_work_groups_kernels {
  // CHECK-LABEL: gen_num_work_groups_x
  gpu.func @gen_num_work_groups_x() kernel {
    // CHECK: [[DIM:%.*]] = arith.constant 0 : i32
    // CHECK: gen.num_work_groups [[DIM]]
    %0 = gpu.grid_dim x
    gpu.return
  }

  // CHECK-LABEL: gen_num_work_groups_y
  gpu.func @gen_num_work_groups_y() kernel {
    // CHECK: [[DIM:%.*]] = arith.constant 1 : i32
    // CHECK: gen.num_work_groups [[DIM]]
    %0 = gpu.grid_dim y
    gpu.return
  }

  // CHECK-LABEL: gen_num_work_groups_z
  gpu.func @gen_num_work_groups_z() kernel {
    // CHECK: [[DIM:%.*]] = arith.constant 2 : i32
    // CHECK: gen.num_work_groups [[DIM]]
    %0 = gpu.grid_dim z
    gpu.return
  }
}

// -----

gpu.module @barrier_kernels {
  // CHECK-LABEL: gen_barrier
  gpu.func @gen_barrier() kernel {
    // CHECK: gen.barrier
    gpu.barrier
    gpu.return
  }
}

// -----

// CHECK-LABEL gpu.module @shuffle_kernels
gpu.module @shuffle_kernels {
  // CHECK: gpu.func @gen_shuffle_xor(%[[IN_XOR:.*]]: f32, %[[OFFSET_XOR:.*]]: i32) kernel {
  gpu.func @gen_shuffle_xor(%in : f32, %offset: i32) kernel {
    // CHECK: %{{.*}} = gen.sub_group_shuffle xor %[[IN_XOR]], %[[OFFSET_XOR]] : f32
    %width = arith.constant 32 : i32
    %0, %1 = gpu.shuffle xor %in, %offset, %width : f32
    gpu.return
  }
  // CHECK: gpu.func @gen_shuffle_up(%[[IN_UP:.*]]: f32, %[[OFFSET_UP:.*]]: i32) kernel {
  gpu.func @gen_shuffle_up(%in : f32, %offset: i32) kernel {
    // CHECK: %{{.*}} = gen.sub_group_shuffle up %[[IN_UP]], %[[OFFSET_UP]] : f32
    %width = arith.constant 32 : i32
    %0, %1 = gpu.shuffle up %in, %offset, %width : f32
    gpu.return
  }
  // CHECK: gpu.func @gen_shuffle_down(%[[IN_DOWN:.*]]: f32, %[[OFFSET_DOWN:.*]]: i32) kernel {
  gpu.func @gen_shuffle_down(%in : f32, %offset: i32) kernel {
    // CHECK: %{{.*}} = gen.sub_group_shuffle down %[[IN_DOWN]], %[[OFFSET_DOWN]] : f32
    %width = arith.constant 32 : i32
    %0, %1 = gpu.shuffle down %in, %offset, %width : f32
    gpu.return
  }
  // CHECK: gpu.func @gen_shuffle_idx(%[[IN_IDX:.*]]: f32, %[[OFFSET_IDX:.*]]: i32) kernel {
  gpu.func @gen_shuffle_idx(%in : f32, %offset: i32) kernel {
    // CHECK: %{{.*}} = gen.sub_group_shuffle idx %[[IN_IDX]], %[[OFFSET_IDX]] : f32
    %width = arith.constant 32 : i32
    %0, %1 = gpu.shuffle idx %in, %offset, %width : f32
    gpu.return
  }
}
