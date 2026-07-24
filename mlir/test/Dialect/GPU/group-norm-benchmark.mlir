// GroupNorm benchmark: D=192, S=784, DS=150528 per group, 64 groups
// Each group launch: 1024 threads (32 SIMD × 32 subgroups)
// Total: 64 kernel launches per iteration
// Each thread: f16→f32 Welford scan + runtime int division c=j/S by scalar arg S

module @group_norm attributes {gpu.container_module} {
  gpu.module @gn_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Int16, Int64, Kernel], []>, api=OpenCL, #spirv.resource_limits<>>} {

    gpu.func @gn_kernel(%X: memref<150528xf16>, %Y: memref<150528xf16>,
                        %gamma: memref<192xf16>, %beta: memref<192xf16>,
                        %eps: f32, %D: i32, %S: i32, %DS: i32) kernel
    attributes {gpu.known_block_size = array<i32: 1024, 1, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>,
                gpu.known_grid_size = array<i32: 1, 1, 1>} {

      %tx = gpu.thread_id x
      %lid = arith.index_castui %tx : index to i32
      %c32 = arith.constant 32 : i32
      %c4 = arith.constant 4 : i32
      %c4_idx = arith.constant 4 : index
      %inv4 = arith.constant 0.25 : f32
      %c0_f32 = arith.constant 0.0 : f32
      %c4_f32 = arith.constant 4.0 : f32
      %c1024 = arith.constant 1024 : index
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32

      // DS_vec = DS / 4
      %DS_vec = arith.divui %DS, %c4 : i32
      %DS_vec_idx = arith.index_castui %DS_vec : i32 to index
      %c1024_idx = arith.constant 1024 : index
      %c0_idx = arith.constant 0 : index
      %c1_idx = arith.constant 1 : index

      // === Welford scan ===
      %final_mean, %final_m2, %final_nf = scf.for %vi = %c0_idx to %DS_vec_idx step %c1024_idx iter_args(
          %st_mean = %c0_f32, %st_m2 = %c0_f32, %st_nf = %c0_f32) -> (f32, f32, f32) {

        %vi1 = arith.addi %vi, %c1_idx : index
        %vi2 = arith.addi %vi1, %c1_idx : index
        %vi3 = arith.addi %vi2, %c1_idx : index

        %x0 = memref.load %X[%vi] : memref<150528xf16>
        %x1 = memref.load %X[%vi1] : memref<150528xf16>
        %x2 = memref.load %X[%vi2] : memref<150528xf16>
        %x3 = memref.load %X[%vi3] : memref<150528xf16>

        %f0 = arith.extf %x0 : f16 to f32
        %f1 = arith.extf %x1 : f16 to f32
        %f2 = arith.extf %x2 : f16 to f32
        %f3 = arith.extf %x3 : f16 to f32

        // === TARGET: c = j/S where j=vi*4 ===
        %j = arith.muli %vi, %c4_idx : index
        %j_i32 = arith.index_castui %j : index to i32
        %ch = arith.divui %j_i32, %S : i32
        %ch_f = arith.sitofp %ch : i32 to f32
        %f0_m = arith.mulf %f0, %ch_f : f32
        %f1_m = arith.mulf %f1, %ch_f : f32
        %f2_m = arith.mulf %f2, %ch_f : f32
        %f3_m = arith.mulf %f3, %ch_f : f32

        // Batch Welford for 4 elements
        %bs = arith.addf %f0_m, %f1_m : f32
        %bs2 = arith.addf %f2_m, %f3_m : f32
        %batch_sum = arith.addf %bs, %bs2 : f32
        %batch_mean = arith.mulf %batch_sum, %inv4 : f32
        %batch_mean_x_sum = arith.mulf %batch_sum, %batch_mean : f32
        %ss = arith.mulf %f0_m, %f0_m : f32
        %ss2 = arith.mulf %f1_m, %f1_m : f32
        %ss3 = arith.mulf %f2_m, %f2_m : f32
        %ss4 = arith.mulf %f3_m, %f3_m : f32
        %ssa = arith.addf %ss, %ss2 : f32
        %ssb = arith.addf %ss3, %ss4 : f32
        %batch_ss = arith.addf %ssa, %ssb : f32
        %batch_M2 = arith.subf %batch_ss, %batch_mean_x_sum : f32

        // Welford combine
        %delta = arith.subf %batch_mean, %st_mean : f32
        %delta_r = arith.mulf %delta, %c4_f32 : f32
        %total = arith.addf %st_nf, %c4_f32 : f32
        %r = arith.divf %delta_r, %total : f32
        %new_mean = arith.addf %st_mean, %r : f32
        %delta_sq = arith.mulf %delta, %delta : f32
        %delta_sq_nf = arith.mulf %delta_sq, %st_nf : f32
        %delta_sq_nf_r = arith.divf %delta_sq_nf, %total : f32
        %new_m2_a = arith.addf %st_m2, %batch_M2 : f32
        %new_m2 = arith.addf %new_m2_a, %delta_sq_nf_r : f32

        scf.yield %new_mean, %new_m2, %total : f32, f32, f32
      }

      // Store result of Welford (just to make sure it's computed)
      %c0_store = arith.constant 0 : index
      %dummy = arith.truncf %final_mean : f32 to f16
      memref.store %dummy, %Y[%c0_store] : memref<150528xf16>
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

    // 64 groups per iteration
    %c64 = arith.constant 64 : index
    %c5 = arith.constant 5 : index
    scf.for %w = %c0 to %c5 step %c1 {
      scf.for %g = %c0 to %c64 step %c1 {
        gpu.launch_func @gn_kernel::@gn_kernel blocks in (%c1, %c1, %c1) threads in (%c1024, %c1, %c1)
          args(%X : memref<150528xf16>, %Y : memref<150528xf16>,
               %gamma : memref<192xf16>, %beta : memref<192xf16>,
               %eps : f32, %D : i32, %S : i32, %DS : i32)
      }
    }

    %c20 = arith.constant 20 : index
    scf.for %b = %c0 to %c20 step %c1 {
      scf.for %g = %c0 to %c64 step %c1 {
        gpu.launch_func @gn_kernel::@gn_kernel blocks in (%c1, %c1, %c1) threads in (%c1024, %c1, %c1)
          args(%X : memref<150528xf16>, %Y : memref<150528xf16>,
               %gamma : memref<192xf16>, %beta : memref<192xf16>,
               %eps : f32, %D : i32, %S : i32, %DS : i32)
      }
    }

    return
  }
}