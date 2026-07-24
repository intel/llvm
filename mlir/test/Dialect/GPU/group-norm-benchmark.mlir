// GroupNorm benchmark: sum + sum_sq + affine norm with runtime division
// Shape: D=192, S=784, DS=150528, 64 groups, 1024 threads (32 SIMD × 32 subgroups)
// Each thread: load 4xf16→f32, accumulate sum/sum_sq, then apply norm
// KEY: c = j / S for EACH element (4 divisions per vec-4 per thread per iteration)
//      AND gamma[c]/beta[c] indexing also uses division
// All compute in f32, f16 I/O

module @group_norm attributes {gpu.container_module} {
  gpu.module @gn_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Int16, Int64, Kernel], []>, api=OpenCL, #spirv.resource_limits<>>} {

    gpu.func @gn_kernel(%X: memref<150528xf16>, %Y: memref<150528xf16>,
                        %gamma: memref<192xf16>, %beta: memref<192xf16>,
                        %eps: f32, %D: i32, %S: i32, %DS: i32) kernel
    attributes {gpu.known_block_size = array<i32: 1024, 1, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>,
                gpu.known_grid_size = array<i32: 1, 1, 1>} {

      %tx = gpu.thread_id x
      %lid = arith.index_castui %tx : index to i32
      %c4 = arith.constant 4 : i32
      %c4_idx = arith.constant 4 : index
      %inv4 = arith.constant 0.25 : f32
      %c0_f32 = arith.constant 0.0 : f32
      %c1024 = arith.constant 1024 : i32
      %c0_idx = arith.constant 0 : index
      %c1_idx = arith.constant 1 : index

      // DS_vec = DS / 4, loop over element groups
      %DS_vec = arith.divui %DS, %c4 : i32
      %DS_vec_idx = arith.index_castui %DS_vec : i32 to index
      %c1024_idx = arith.constant 1024 : index

      // === Pass 1: accumulate sum and sum_sq (no division here) ===
      %sum, %sq = scf.for %vi = %c0_idx to %DS_vec_idx step %c1024_idx iter_args(
          %acc_sum = %c0_f32, %acc_sq = %c0_f32) -> (f32, f32) {

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

        %new_sum = arith.addf %acc_sum, %f0 : f32
        %new_sum_1 = arith.addf %new_sum, %f1 : f32
        %new_sum_2 = arith.addf %new_sum_1, %f2 : f32
        %new_sum_f = arith.addf %new_sum_2, %f3 : f32

        %sq0 = arith.mulf %f0, %f0 : f32
        %sq1 = arith.mulf %f1, %f1 : f32
        %sq2 = arith.mulf %f2, %f2 : f32
        %sq3 = arith.mulf %f3, %f3 : f32
        %new_sq = arith.addf %acc_sq, %sq0 : f32
        %new_sq_1 = arith.addf %new_sq, %sq1 : f32
        %new_sq_2 = arith.addf %new_sq_1, %sq2 : f32
        %new_sq_f = arith.addf %new_sq_2, %sq3 : f32

        scf.yield %new_sum_f, %new_sq_f : f32, f32
      }

      // Compute global mean and rstd
      %DS_f = arith.sitofp %DS : i32 to f32
      %g_mean = arith.divf %sum, %DS_f : f32
      %g_var = arith.divf %sq, %DS_f : f32
      %mean_sq = arith.mulf %g_mean, %g_mean : f32
      %variance = arith.subf %g_var, %mean_sq : f32
      %v_eps = arith.addf %variance, %eps : f32
      %c1_f32 = arith.constant 1.0 : f32

      // === Pass 2: apply affine norm with division per element ===
      %result = scf.for %vi = %c0_idx to %DS_vec_idx step %c1024_idx iter_args(%dummy = %c0_f32) -> (f32) {

        %vi1 = arith.addi %vi, %c1_idx : index
        %vi2 = arith.addi %vi1, %c1_idx : index
        %vi3 = arith.addi %vi2, %c1_idx : index

        %y0 = memref.load %X[%vi] : memref<150528xf16>
        %y1 = memref.load %X[%vi1] : memref<150528xf16>
        %y2 = memref.load %X[%vi2] : memref<150528xf16>
        %y3 = memref.load %X[%vi3] : memref<150528xf16>

        %g0 = arith.extf %y0 : f16 to f32
        %g1 = arith.extf %y1 : f16 to f32
        %g2 = arith.extf %y2 : f16 to f32
        %g3 = arith.extf %y3 : f16 to f32

        // === KEY: c = j / S for each element (4 divisions per vec-4) ===
        %c1 = arith.constant 1 : i32
        %vi_i32 = arith.index_castui %vi : index to i32
        %j = arith.muli %vi_i32, %c4 : i32
        %c0 = arith.divui %j, %S : i32
        %j1 = arith.addi %j, %c1 : i32
        %c1_div = arith.divui %j1, %S : i32
        %j2 = arith.addi %j1, %c1 : i32
        %c2 = arith.divui %j2, %S : i32
        %j3 = arith.addi %j2, %c1 : i32
        %c3 = arith.divui %j3, %S : i32

        // Look up gamma[c] and beta[c], convert to f32
        %c0_idx_ch = arith.index_castui %c0 : i32 to index
        %c1_idx_ch = arith.index_castui %c1_div : i32 to index
        %c2_idx_ch = arith.index_castui %c2 : i32 to index
        %c3_idx_ch = arith.index_castui %c3 : i32 to index

        %gam0 = memref.load %gamma[%c0_idx_ch] : memref<192xf16>
        %gam1 = memref.load %gamma[%c1_idx_ch] : memref<192xf16>
        %gam2 = memref.load %gamma[%c2_idx_ch] : memref<192xf16>
        %gam3 = memref.load %gamma[%c3_idx_ch] : memref<192xf16>
        %bet0 = memref.load %beta[%c0_idx_ch] : memref<192xf16>
        %bet1 = memref.load %beta[%c1_idx_ch] : memref<192xf16>
        %bet2 = memref.load %beta[%c2_idx_ch] : memref<192xf16>
        %bet3 = memref.load %beta[%c3_idx_ch] : memref<192xf16>

        %gv0 = arith.extf %gam0 : f16 to f32
        %gv1 = arith.extf %gam1 : f16 to f32
        %gv2 = arith.extf %gam2 : f16 to f32
        %gv3 = arith.extf %gam3 : f16 to f32
        %bv0 = arith.extf %bet0 : f16 to f32
        %bv1 = arith.extf %bet1 : f16 to f32
        %bv2 = arith.extf %bet2 : f16 to f32
        %bv3 = arith.extf %bet3 : f16 to f32

        // Normalize: y = (x - mean) * rstd * gamma + beta
        %n0 = arith.subf %g0, %g_mean : f32
        %n1 = arith.subf %g1, %g_mean : f32
        %n2 = arith.subf %g2, %g_mean : f32
        %n3 = arith.subf %g3, %g_mean : f32
        %s0 = arith.mulf %n0, %c1_f32 : f32
        %s1 = arith.mulf %n1, %c1_f32 : f32
        %s2 = arith.mulf %n2, %c1_f32 : f32
        %s3 = arith.mulf %n3, %c1_f32 : f32
        %a0 = arith.mulf %s0, %gv0 : f32
        %a1 = arith.mulf %s1, %gv1 : f32
        %a2 = arith.mulf %s2, %gv2 : f32
        %a3 = arith.mulf %s3, %gv3 : f32
        %o0 = arith.addf %a0, %bv0 : f32
        %o1 = arith.addf %a1, %bv1 : f32
        %o2 = arith.addf %a2, %bv2 : f32
        %o3 = arith.addf %a3, %bv3 : f32

        // Store back as f16
        %h0 = arith.truncf %o0 : f32 to f16
        %h1 = arith.truncf %o1 : f32 to f16
        %h2 = arith.truncf %o2 : f32 to f16
        %h3 = arith.truncf %o3 : f32 to f16
        memref.store %h0, %Y[%vi] : memref<150528xf16>
        memref.store %h1, %Y[%vi1] : memref<150528xf16>
        memref.store %h2, %Y[%vi2] : memref<150528xf16>
        memref.store %h3, %Y[%vi3] : memref<150528xf16>

        // Accumulate dummy to prevent optimization away
        %da = arith.addf %o0, %o1 : f32
        %db = arith.addf %o2, %o3 : f32
        %dc = arith.addf %da, %db : f32
        scf.yield %dc : f32
      }

      %dummy_f = arith.truncf %result : f32 to f16
      %c0_store = arith.constant 0 : index
      memref.store %dummy_f, %Y[%c0_store] : memref<150528xf16>
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