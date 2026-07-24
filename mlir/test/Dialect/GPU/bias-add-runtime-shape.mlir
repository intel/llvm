module @bias_add attributes {gpu.container_module} {
  gpu.module @bias_add_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Int64, Kernel], []>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @bias_add_kernel(%src: memref<12xf32>, %bias: memref<3xf32>, %dst: memref<12xf32>,
                              %tot: index, %chw: index, %hw: index) kernel
    attributes {gpu.known_block_size = array<i32: 12, 1, 1>, gpu.known_grid_size = array<i32: 1, 1, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %tx = gpu.thread_id x
      %i = arith.index_castui %tx : index to i32
      %tot_i32 = arith.index_castui %tot : index to i32
      %is_in = arith.cmpi ult, %i, %tot_i32 : i32
      scf.if %is_in {
        %chw_i32 = arith.index_castui %chw : index to i32
        %hw_i32 = arith.index_castui %hw : index to i32
        %rem = arith.remui %i, %chw_i32 : i32
        %ch = arith.divui %rem, %hw_i32 : i32
        %ci = arith.index_castui %ch : i32 to index
        %idx = arith.index_castui %i : i32 to index
        %s = memref.load %src[%idx] : memref<12xf32>
        %b = memref.load %bias[%ci] : memref<3xf32>
        %r = arith.addf %s, %b : f32
        memref.store %r, %dst[%idx] : memref<12xf32>
      }
      gpu.return
    }
  }
  
  func.func @main() {
    %c12 = arith.constant 12 : index
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    
    %mem_src = gpu.alloc host_shared () : memref<12xf32>
    %mem_dst = gpu.alloc host_shared () : memref<12xf32>
    %mem_bias = gpu.alloc host_shared () : memref<3xf32>
    
    %v0 = arith.constant 0.0 : f32
    %v1 = arith.constant 1.0 : f32
    %v2 = arith.constant 2.0 : f32
    %v3 = arith.constant 3.0 : f32
    %v4 = arith.constant 4.0 : f32
    memref.store %v0, %mem_src[%c0] : memref<12xf32>
    memref.store %v1, %mem_src[%c1] : memref<12xf32>
    memref.store %v2, %mem_src[%c2] : memref<12xf32>
    memref.store %v3, %mem_src[%c3] : memref<12xf32>
    %c4_i = arith.constant 4 : index
    memref.store %v4, %mem_src[%c4_i] : memref<12xf32>
    
    memref.store %v1, %mem_bias[%c0] : memref<3xf32>
    memref.store %v2, %mem_bias[%c1] : memref<3xf32>
    memref.store %v3, %mem_bias[%c2] : memref<3xf32>
    
    %tot = arith.constant 12 : index
    %chw = arith.constant 12 : index
    %hw = arith.constant 4 : index
    
    gpu.launch_func @bias_add_kernel::@bias_add_kernel blocks in (%c1, %c1, %c1) threads in (%c12, %c1, %c1)
      args(%mem_src : memref<12xf32>, %mem_bias : memref<3xf32>, %mem_dst : memref<12xf32>,
           %tot : index, %chw : index, %hw : index)
    
    %cast = memref.cast %mem_dst : memref<12xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    return
  }
  func.func private @printMemrefF32(memref<*xf32>)
}
