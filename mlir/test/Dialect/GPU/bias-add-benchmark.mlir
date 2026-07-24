// BiasAdd GPU benchmark: dst[i] = src[i] + bias[c], c = (i % chw) / hw
// Shape: tot=67108864 (64M), N=16, C=16, HW=262144, chw=4194304
// 256 threads × 262144 blocks, each thread does: urem+udiv+load+fadd+store
// All values are f32 compute

module @bias_add attributes {gpu.container_module} {
  gpu.module @bias_add_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Int64, Kernel], []>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @bias_add_kernel(%src: memref<67108864xf32>, %bias: memref<16xf32>, %dst: memref<67108864xf32>,
                              %tot: i32, %chw: i32, %hw: i32) kernel
    attributes {gpu.known_block_size = array<i32: 256, 1, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %tx = gpu.thread_id x
      %bx = gpu.block_id x
      %bs = gpu.block_dim x
      %gid0 = arith.muli %bx, %bs : index
      %gid = arith.addi %tx, %gid0 : index
      %i = arith.index_castui %gid : index to i32
      %is_in = arith.cmpi ult, %i, %tot : i32
      scf.if %is_in {
        // === TARGET: runtime division by uniform scalar args ===
        %rem = arith.remui %i, %chw : i32
        %ch = arith.divui %rem, %hw : i32
        %ci = arith.index_castui %ch : i32 to index
        %idx = arith.index_castui %i : i32 to index
        %s = memref.load %src[%idx] : memref<67108864xf32>
        %b = memref.load %bias[%ci] : memref<16xf32>
        %r = arith.addf %s, %b : f32
        memref.store %r, %dst[%idx] : memref<67108864xf32>
      }
      gpu.return
    }
  }

  func.func @main() {
    %tot = arith.constant 67108864 : i32
    %chw = arith.constant 4194304 : i32
    %hw = arith.constant 262144 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %blocks = arith.constant 262144 : index

    %mem_src = gpu.alloc host_shared () : memref<67108864xf32>
    %mem_dst = gpu.alloc host_shared () : memref<67108864xf32>
    %mem_bias = gpu.alloc host_shared () : memref<16xf32>

    %v1 = arith.constant 1.0 : f32
    memref.store %v1, %mem_bias[%c0] : memref<16xf32>

    %c5 = arith.constant 5 : index
    scf.for %w = %c0 to %c5 step %c1 {
      gpu.launch_func @bias_add_kernel::@bias_add_kernel blocks in (%blocks, %c1, %c1) threads in (%c256, %c1, %c1)
        args(%mem_src : memref<67108864xf32>, %mem_bias : memref<16xf32>, %mem_dst : memref<67108864xf32>,
             %tot : i32, %chw : i32, %hw : i32)
    }

    %c100 = arith.constant 100 : index
    scf.for %b = %c0 to %c100 step %c1 {
      gpu.launch_func @bias_add_kernel::@bias_add_kernel blocks in (%blocks, %c1, %c1) threads in (%c256, %c1, %c1)
        args(%mem_src : memref<67108864xf32>, %mem_bias : memref<16xf32>, %mem_dst : memref<67108864xf32>,
             %tot : i32, %chw : i32, %hw : i32)
    }

    return
  }
}