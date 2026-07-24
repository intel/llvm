// GroupNorm GPU benchmark: compute c = j / S per work-item (4 divisions × 100 iters)
// Shape: D=192, S=784, DS=150528, 1024 threads (32 subgroups of 32)
// Key runtime division: arith.divui(j+v, %S) — %S is a kernel scalar parameter

module @group_norm attributes {gpu.container_module} {
  gpu.module @gn_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Int16, Int64, Kernel], []>, api=OpenCL, #spirv.resource_limits<>>} {

    gpu.func @gn_kernel(%X: memref<150528xf16>, %Y: memref<150528xf16>,
                        %gamma: memref<192xf16>, %beta: memref<192xf16>,
                        %eps: f32, %D: i32, %S: i32, %DS: i32) kernel
    attributes {gpu.known_block_size = array<i32: 1024, 1, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>,
                gpu.known_grid_size = array<i32: 1, 1, 1>} {

      %tx_idx = gpu.thread_id x
      %tid = arith.index_castui %tx_idx : index to i32
      %c32 = arith.constant 32 : i32
      %sg_lid = arith.remui %tid, %c32 : i32

%c4 = arith.constant 4 : i32
      %cst_0 = arith.constant 0 : i32
      %cst_1 = arith.constant 1 : i32
      %c100 = arith.constant 100 : i32
      %c0_f32 = arith.constant 0.0 : f32
      %c100_idx = arith.constant 100 : index
      %c0_idx = arith.constant 0 : index
      %c1_idx = arith.constant 1 : index

      // Loop: 100 iterations of 4 divisions per work-item
      %result = scf.for %iter = %c0_idx to %c100_idx step %c1_idx iter_args(%acc = %c0_f32) -> (f32) {
        %j = arith.muli %sg_lid, %c4 : i32
        %c0 = arith.divui %j, %S : i32
        %j1 = arith.addi %j, %cst_1 : i32
        %c1 = arith.divui %j1, %S : i32
        %j2 = arith.addi %j1, %cst_1 : i32
        %c2 = arith.divui %j2, %S : i32
        %j3 = arith.addi %j2, %cst_1 : i32
        %c3 = arith.divui %j3, %S : i32
        %s0 = arith.addi %c0, %c1 : i32
        %s1 = arith.addi %c2, %c3 : i32
        %s2 = arith.addi %s0, %s1 : i32
        %acc_i = arith.sitofp %s2 : i32 to f32
        %acc_new = arith.addf %acc, %acc_i : f32
        scf.yield %acc_new : f32
      }

      %dummy = arith.truncf %result : f32 to f16
      %cst_0_store = arith.constant 0 : index
      memref.store %dummy, %Y[%cst_0_store] : memref<150528xf16>
      gpu.return
    }
  }

  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index

    %X = gpu.alloc host_shared () : memref<150528xf16>
    %Y = gpu.alloc host_shared () : memref<150528xf16>
    %gamma = gpu.alloc host_shared () : memref<192xf16>
    %beta = gpu.alloc host_shared () : memref<192xf16>

    %eps = arith.constant 1.000000e-05 : f32
    %D = arith.constant 192 : i32
    %S = arith.constant 784 : i32
    %DS = arith.constant 150528 : i32

    // WARMUP
    %c5 = arith.constant 5 : index
    scf.for %w = %c0 to %c5 step %c1 {
      gpu.launch_func @gn_kernel::@gn_kernel blocks in (%c1, %c1, %c1) threads in (%c1024, %c1, %c1)
        args(%X : memref<150528xf16>, %Y : memref<150528xf16>,
             %gamma : memref<192xf16>, %beta : memref<192xf16>,
             %eps : f32, %D : i32, %S : i32, %DS : i32)
    }

    // BENCHMARK
    %c50 = arith.constant 50 : index
    scf.for %b = %c0 to %c50 step %c1 {
      gpu.launch_func @gn_kernel::@gn_kernel blocks in (%c1, %c1, %c1) threads in (%c1024, %c1, %c1)
        args(%X : memref<150528xf16>, %Y : memref<150528xf16>,
             %gamma : memref<192xf16>, %beta : memref<192xf16>,
             %eps : f32, %D : i32, %S : i32, %DS : i32)
    }

    return
  }
}