// RUN: polygeist-opt --cpuify="method=distribute.mincut" --split-input-file %s | FileCheck %s

module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  func.func @_Z11bpnnwrapperiPfiS_iS_S_(%arg0: i32, %arg1: memref<?xf32>, %arg2: i32, %arg3: memref<?xf32>, %arg4: i32, %arg5: memref<?xf32>, %arg6: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c16 = arith.constant 16 : index
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 3.000000e-01 : f64
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = arith.index_cast %arg2 : i32 to index
    %1 = arith.index_cast %arg0 : i32 to index
    %2 = arith.addi %0, %c1 : index
    %3 = arith.muli %2, %c16 : index
    scf.parallel (%arg7, %arg8, %arg9) = (%c0, %c0, %c0) to (%1, %c16, %c16) step (%c1, %c1, %c1) {
      %4 = arith.index_cast %arg7 : index to i32
      %5 = arith.index_cast %arg9 : index to i32
      %6 = arith.muli %3, %arg7 : index
      %7 = arith.muli %2, %arg9 : index
      %8 = arith.addi %6, %7 : index
      %9 = arith.addi %8, %arg8 : index
      %10 = arith.addi %9, %c1 : index
      %11 = arith.muli %arg7, %c16 : index
      %12 = arith.addi %11, %arg9 : index
      %13 = arith.addi %10, %2 : index
      %14 = arith.addi %arg8, %c1 : index
      %15 = memref.load %arg1[%14] : memref<?xf32>
      %16 = arith.extf %15 : f32 to f64
      %17 = arith.mulf %cst, %16 : f64
      %18 = arith.addi %12, %c1 : index
      %19 = memref.load %arg3[%18] : memref<?xf32>
      %20 = arith.extf %19 : f32 to f64
      %21 = arith.mulf %17, %20 : f64
      %22 = memref.load %arg6[%13] : memref<?xf32>
      %23 = arith.extf %22 : f32 to f64
      %24 = arith.mulf %cst, %23 : f64
      %25 = arith.addf %21, %24 : f64
      %26 = memref.load %arg5[%13] : memref<?xf32>
      %27 = arith.truncf %25 : f64 to f32
      %28 = arith.addf %26, %27 : f32
      memref.store %28, %arg5[%13] : memref<?xf32>
      %29 = memref.load %arg1[%14] : memref<?xf32>
      %30 = arith.extf %29 : f32 to f64
      %31 = arith.mulf %cst, %30 : f64
      %32 = memref.load %arg3[%18] : memref<?xf32>
      %33 = arith.extf %32 : f32 to f64
      %34 = arith.mulf %31, %33 : f64
      %35 = memref.load %arg6[%13] : memref<?xf32>
      %36 = arith.extf %35 : f32 to f64
      %37 = arith.mulf %cst, %36 : f64
      %38 = arith.addf %34, %37 : f64
      %39 = arith.truncf %38 : f64 to f32
      memref.store %39, %arg6[%13] : memref<?xf32>
      "polygeist.barrier"(%arg8, %arg9, %c0) : (index, index, index) -> ()
      %40 = arith.cmpi eq, %5, %c0_i32 : i32
      scf.if %40 {
        %41 = arith.cmpi eq, %4, %c0_i32 : i32
        scf.if %41 {
          %42 = memref.load %arg1[%14] : memref<?xf32>
          %43 = arith.extf %42 : f32 to f64
          %44 = arith.mulf %cst, %43 : f64
          %45 = memref.load %arg6[%14] : memref<?xf32>
          %46 = arith.extf %45 : f32 to f64
          %47 = arith.mulf %cst, %46 : f64
          %48 = arith.addf %44, %47 : f64
          %49 = memref.load %arg5[%14] : memref<?xf32>
          %50 = arith.truncf %48 : f64 to f32
          %51 = arith.addf %49, %50 : f32
          memref.store %51, %arg5[%14] : memref<?xf32>
          %52 = memref.load %arg1[%14] : memref<?xf32>
          %53 = arith.extf %52 : f32 to f64
          %54 = arith.mulf %cst, %53 : f64
          %55 = memref.load %arg6[%14] : memref<?xf32>
          %56 = arith.extf %55 : f32 to f64
          %57 = arith.mulf %cst, %56 : f64
          %58 = arith.addf %54, %57 : f64
          %59 = arith.truncf %58 : f64 to f32
          memref.store %59, %arg6[%14] : memref<?xf32>
        }
      }
      scf.yield
    }
    return
  }
  func.func @_Z30bpnn_layerforward_CUDA_wrapperiPfS_S_S_ii(%arg0: i32, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>, %arg5: i32, %arg6: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c16 = arith.constant 16 : index
    %cst = arith.constant 2.000000e+00 : f32
    %cst_0 = arith.constant 4.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %0 = arith.index_cast %arg6 : i32 to index
    %1 = arith.index_cast %arg0 : i32 to index
    %2 = arith.addi %0, %c1 : index
    %3 = arith.muli %2, %c16 : index
    scf.parallel (%arg7) = (%c0) to (%1) step (%c1) {
      %4 = memref.alloca() : memref<16xf32>
      %5 = memref.alloca() : memref<16x16xf32>
      %6 = arith.muli %3, %arg7 : index
      %7 = arith.muli %arg7, %c16 : index
      scf.parallel (%arg8, %arg9) = (%c0, %c0) to (%c16, %c16) step (%c1, %c1) {
        %8 = arith.index_cast %arg8 : index to i32
        %9 = arith.index_cast %arg9 : index to i32
        %10 = arith.muli %2, %arg9 : index
        %11 = arith.addi %6, %10 : index
        %12 = arith.addi %11, %arg8 : index
        %13 = arith.addi %12, %c1 : index
        %14 = arith.addi %7, %arg9 : index
        %15 = arith.cmpi eq, %8, %c0_i32 : i32
        scf.if %15 {
          %23 = arith.addi %14, %c1 : index
          %24 = memref.load %arg1[%23] : memref<?xf32>
          memref.store %24, %4[%arg9] : memref<16xf32>
        }
        "polygeist.barrier"(%arg8, %arg9, %c0) : (index, index, index) -> ()
        %16 = arith.addi %13, %2 : index
        %17 = memref.load %arg3[%16] : memref<?xf32>
        memref.store %17, %5[%arg9, %arg8] : memref<16x16xf32>
        "polygeist.barrier"(%arg8, %arg9, %c0) : (index, index, index) -> ()
        %18 = memref.load %5[%arg9, %arg8] : memref<16x16xf32>
        %19 = memref.load %4[%arg9] : memref<16xf32>
        %20 = arith.mulf %18, %19 : f32
        memref.store %20, %5[%arg9, %arg8] : memref<16x16xf32>
        "polygeist.barrier"(%arg8, %arg9, %c0) : (index, index, index) -> ()
        %21 = scf.while (%arg10 = %c1_i32) : (i32) -> i32 {
          %23 = arith.sitofp %arg10 : i32 to f32
          %24 = arith.cmpf ule, %23, %cst_0 : f32
          scf.condition(%24) %arg10 : i32
        } do {
        ^bb0(%arg10: i32):  // no predecessors
          %23 = arith.sitofp %arg10 : i32 to f32
          %24 = math.powf %cst, %23 : f32
          %25 = arith.fptosi %24 : f32 to i32
          %26 = arith.index_cast %25 : i32 to index
          %27 = arith.remsi %9, %25 : i32
          %28 = arith.cmpi eq, %27, %c0_i32 : i32
          scf.if %28 {
            %30 = memref.load %5[%arg9, %arg8] : memref<16x16xf32>
            %31 = arith.divsi %26, %c2 : index
            %32 = arith.addi %arg9, %31 : index
            %33 = memref.load %5[%32, %arg8] : memref<16x16xf32>
            %34 = arith.addf %30, %33 : f32
            memref.store %34, %5[%arg9, %arg8] : memref<16x16xf32>
          }
          "polygeist.barrier"(%arg8, %arg9, %c0) : (index, index, index) -> ()
          %29 = arith.addi %arg10, %c1_i32 : i32
          scf.yield %29 : i32
        }
        %22 = memref.load %5[%arg9, %arg8] : memref<16x16xf32>
        memref.store %22, %arg3[%16] : memref<?xf32>
        "polygeist.barrier"(%arg8, %arg9, %c0) : (index, index, index) -> ()
        scf.if %15 {
          %23 = memref.load %5[%arg8, %arg9] : memref<16x16xf32>
          %24 = arith.muli %arg7, %0 : index
          %25 = arith.addi %arg9, %24 : index
          memref.store %23, %arg4[%25] : memref<?xf32>
        }
        scf.yield
      }
      scf.yield
    }
    return
  }
}

// CHECK:  func.func @_Z11bpnnwrapperiPfiS_iS_S_(%arg0: i32, %arg1: memref<?xf32>, %arg2: i32, %arg3: memref<?xf32>, %arg4: i32, %arg5: memref<?xf32>, %arg6: memref<?xf32>)
// CHECK-NEXT:    %c16 = arith.constant 16 : index
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    %cst = arith.constant 3.000000e-01 : f64
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %0 = arith.index_cast %arg2 : i32 to index
// CHECK-NEXT:    %1 = arith.index_cast %arg0 : i32 to index
// CHECK-NEXT:    %2 = arith.addi %0, %c1 : index
// CHECK-NEXT:    %3 = arith.muli %2, %c16 : index
// CHECK-NEXT:    scf.parallel (%arg7) = (%c0) to (%1) step (%c1) {
// CHECK-NEXT:      scf.parallel (%arg8, %arg9) = (%c0, %c0) to (%c16, %c16) step (%c1, %c1) {
// CHECK-NEXT:        %4 = arith.muli %3, %arg7 : index
// CHECK-NEXT:        %5 = arith.muli %2, %arg9 : index
// CHECK-NEXT:        %6 = arith.addi %4, %5 : index
// CHECK-NEXT:        %7 = arith.addi %6, %arg8 : index
// CHECK-NEXT:        %8 = arith.addi %7, %c1 : index
// CHECK-NEXT:        %9 = arith.muli %arg7, %c16 : index
// CHECK-NEXT:        %10 = arith.addi %9, %arg9 : index
// CHECK-NEXT:        %11 = arith.addi %8, %2 : index
// CHECK-NEXT:        %12 = arith.addi %arg8, %c1 : index
// CHECK-NEXT:        %13 = memref.load %arg1[%12] : memref<?xf32>
// CHECK-NEXT:        %14 = arith.extf %13 : f32 to f64
// CHECK-NEXT:        %15 = arith.mulf %14, %cst : f64
// CHECK-NEXT:        %16 = arith.addi %10, %c1 : index
// CHECK-NEXT:        %17 = memref.load %arg3[%16] : memref<?xf32>
// CHECK-NEXT:        %18 = arith.extf %17 : f32 to f64
// CHECK-NEXT:        %19 = arith.mulf %15, %18 : f64
// CHECK-NEXT:        %20 = memref.load %arg6[%11] : memref<?xf32>
// CHECK-NEXT:        %21 = arith.extf %20 : f32 to f64
// CHECK-NEXT:        %22 = arith.mulf %21, %cst : f64
// CHECK-NEXT:        %23 = arith.addf %19, %22 : f64
// CHECK-NEXT:        %24 = memref.load %arg5[%11] : memref<?xf32>
// CHECK-NEXT:        %25 = arith.truncf %23 : f64 to f32
// CHECK-NEXT:        %26 = arith.addf %24, %25 : f32
// CHECK-NEXT:        memref.store %26, %arg5[%11] : memref<?xf32>
// CHECK-NEXT:        %27 = memref.load %arg1[%12] : memref<?xf32>
// CHECK-NEXT:        %28 = arith.extf %27 : f32 to f64
// CHECK-NEXT:        %29 = arith.mulf %28, %cst : f64
// CHECK-NEXT:        %30 = memref.load %arg3[%16] : memref<?xf32>
// CHECK-NEXT:        %31 = arith.extf %30 : f32 to f64
// CHECK-NEXT:        %32 = arith.mulf %29, %31 : f64
// CHECK-NEXT:        %33 = memref.load %arg6[%11] : memref<?xf32>
// CHECK-NEXT:        %34 = arith.extf %33 : f32 to f64
// CHECK-NEXT:        %35 = arith.mulf %34, %cst : f64
// CHECK-NEXT:        %36 = arith.addf %32, %35 : f64
// CHECK-NEXT:        %37 = arith.truncf %36 : f64 to f32
// CHECK-NEXT:        memref.store %37, %arg6[%11] : memref<?xf32>
// CHECK-NEXT:        scf.yield
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.parallel (%arg8, %arg9) = (%c0, %c0) to (%c16, %c16) step (%c1, %c1) {
// CHECK-NEXT:        %4 = arith.addi %arg8, %c1 : index
// CHECK-NEXT:        %5 = arith.index_cast %arg9 : index to i32
// CHECK-NEXT:        %6 = arith.index_cast %arg7 : index to i32
// CHECK-NEXT:        %7 = arith.cmpi eq, %5, %c0_i32 : i32
// CHECK-NEXT:        scf.if %7 {
// CHECK-NEXT:          %8 = arith.cmpi eq, %6, %c0_i32 : i32
// CHECK-NEXT:          scf.if %8 {
// CHECK-NEXT:            %9 = memref.load %arg1[%4] : memref<?xf32>
// CHECK-NEXT:            %10 = arith.extf %9 : f32 to f64
// CHECK-NEXT:            %11 = arith.mulf %10, %cst : f64
// CHECK-NEXT:            %12 = memref.load %arg6[%4] : memref<?xf32>
// CHECK-NEXT:            %13 = arith.extf %12 : f32 to f64
// CHECK-NEXT:            %14 = arith.mulf %13, %cst : f64
// CHECK-NEXT:            %15 = arith.addf %11, %14 : f64
// CHECK-NEXT:            %16 = memref.load %arg5[%4] : memref<?xf32>
// CHECK-NEXT:            %17 = arith.truncf %15 : f64 to f32
// CHECK-NEXT:            %18 = arith.addf %16, %17 : f32
// CHECK-NEXT:            memref.store %18, %arg5[%4] : memref<?xf32>
// CHECK-NEXT:            %19 = memref.load %arg1[%4] : memref<?xf32>
// CHECK-NEXT:            %20 = arith.extf %19 : f32 to f64
// CHECK-NEXT:            %21 = arith.mulf %20, %cst : f64
// CHECK-NEXT:            %22 = memref.load %arg6[%4] : memref<?xf32>
// CHECK-NEXT:            %23 = arith.extf %22 : f32 to f64
// CHECK-NEXT:            %24 = arith.mulf %23, %cst : f64
// CHECK-NEXT:            %25 = arith.addf %21, %24 : f64
// CHECK-NEXT:            %26 = arith.truncf %25 : f64 to f32
// CHECK-NEXT:            memref.store %26, %arg6[%4] : memref<?xf32>
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        scf.yield
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.yield
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }


// CHECK:   func.func @_Z30bpnn_layerforward_CUDA_wrapperiPfS_S_S_ii(%arg0: i32, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>, %arg5: i32, %arg6: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:    %c16 = arith.constant 16 : index
// CHECK-NEXT:    %cst = arith.constant 2.000000e+00 : f32
// CHECK-NEXT:    %cst_0 = arith.constant 4.000000e+00 : f32
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %c2 = arith.constant 2 : index
// CHECK-NEXT:    %0 = arith.index_cast %arg6 : i32 to index
// CHECK-NEXT:    %1 = arith.index_cast %arg0 : i32 to index
// CHECK-NEXT:    %2 = arith.addi %0, %c1 : index
// CHECK-NEXT:    %3 = arith.muli %2, %c16 : index
// CHECK-NEXT:    scf.parallel (%arg7) = (%c0) to (%1) step (%c1) {
// CHECK-NEXT:      %4 = memref.alloca() : memref<16xf32>
// CHECK-NEXT:      %5 = memref.alloca() : memref<16x16xf32>
// CHECK-NEXT:      %6 = arith.muli %3, %arg7 : index
// CHECK-NEXT:      %7 = arith.muli %arg7, %c16 : index
// CHECK-NEXT:      memref.alloca_scope  {
// CHECK-NEXT:        scf.parallel (%arg8, %arg9) = (%c0, %c0) to (%c16, %c16) step (%c1, %c1) {
// CHECK-NEXT:          %8 = arith.index_cast %arg8 : index to i32
// CHECK-NEXT:          %9 = arith.addi %7, %arg9 : index
// CHECK-NEXT:          %10 = arith.cmpi eq, %8, %c0_i32 : i32
// CHECK-NEXT:          scf.if %10 {
// CHECK-NEXT:            %11 = arith.addi %9, %c1 : index
// CHECK-NEXT:            %12 = memref.load %arg1[%11] : memref<?xf32>
// CHECK-NEXT:            memref.store %12, %4[%arg9] : memref<16xf32>
// CHECK-NEXT:          }
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }
// CHECK-NEXT:        memref.alloca_scope  {
// CHECK-NEXT:          %8 = memref.alloca(%c16, %c16) : memref<?x?xi32>
// CHECK-NEXT:          %9 = memref.alloca(%c16, %c16) : memref<?x?xi32>
// CHECK-NEXT:          memref.alloca_scope  {
// CHECK-NEXT:            scf.parallel (%arg8, %arg9) = (%c0, %c0) to (%c16, %c16) step (%c1, %c1) {
// CHECK-NEXT:              %10 = arith.muli %2, %arg9 : index
// CHECK-NEXT:              %11 = arith.addi %6, %10 : index
// CHECK-NEXT:              %12 = arith.addi %11, %arg8 : index
// CHECK-NEXT:              %13 = arith.addi %12, %c1 : index
// CHECK-NEXT:              %14 = arith.addi %13, %2 : index
// CHECK-NEXT:              %15 = memref.load %arg3[%14] : memref<?xf32>
// CHECK-NEXT:              memref.store %15, %5[%arg9, %arg8] : memref<16x16xf32>
// CHECK-NEXT:              scf.yield
// CHECK-NEXT:            }
// CHECK-NEXT:            scf.parallel (%arg8, %arg9) = (%c0, %c0) to (%c16, %c16) step (%c1, %c1) {
// CHECK-NEXT:              %10 = memref.load %5[%arg9, %arg8] : memref<16x16xf32>
// CHECK-NEXT:              %11 = memref.load %4[%arg9] : memref<16xf32>
// CHECK-NEXT:              %12 = arith.mulf %10, %11 : f32
// CHECK-NEXT:              memref.store %12, %5[%arg9, %arg8] : memref<16x16xf32>
// CHECK-NEXT:              %13 = "polygeist.subindex"(%8, %arg8) : (memref<?x?xi32>, index) -> memref<?xi32>
// CHECK-NEXT:              %14 = "polygeist.subindex"(%13, %arg9) : (memref<?xi32>, index) -> memref<i32>
// CHECK-NEXT:              memref.store %c1_i32, %14[] : memref<i32>
// CHECK-NEXT:              scf.yield
// CHECK-NEXT:            }
// CHECK-NEXT:          }
// CHECK-NEXT:          memref.alloca_scope  {
// CHECK-NEXT:            scf.while : () -> () {
// CHECK-NEXT:              %10 = memref.alloca() : memref<i1>
// CHECK-NEXT:              scf.parallel (%arg8, %arg9) = (%c0, %c0) to (%c16, %c16) step (%c1, %c1) {
// CHECK-NEXT:                %12 = "polygeist.subindex"(%8, %arg8) : (memref<?x?xi32>, index) -> memref<?xi32>
// CHECK-NEXT:                %13 = "polygeist.subindex"(%12, %arg9) : (memref<?xi32>, index) -> memref<i32>
// CHECK-NEXT:                %14 = memref.load %13[] : memref<i32>
// CHECK-NEXT:                %15 = arith.sitofp %14 : i32 to f32
// CHECK-NEXT:                %16 = arith.cmpf ule, %15, %cst_0 : f32
// CHECK-NEXT:                %17 = arith.cmpi eq, %arg8, %c0 : index
// CHECK-NEXT:                %18 = arith.cmpi eq, %arg9, %c0 : index
// CHECK-NEXT:                %19 = arith.andi %18, %17 : i1
// CHECK-NEXT:                scf.if %19 {
// CHECK-NEXT:                  memref.store %16, %10[] : memref<i1>
// CHECK-NEXT:                }
// CHECK-NEXT:                %20 = "polygeist.subindex"(%9, %arg8) : (memref<?x?xi32>, index) -> memref<?xi32>
// CHECK-NEXT:                %21 = "polygeist.subindex"(%20, %arg9) : (memref<?xi32>, index) -> memref<i32>
// CHECK-NEXT:                memref.store %14, %21[] : memref<i32>
// CHECK-NEXT:                scf.yield
// CHECK-NEXT:              }
// CHECK-NEXT:              %11 = memref.load %10[] : memref<i1>
// CHECK-NEXT:              scf.condition(%11)
// CHECK-NEXT:            } do {
// CHECK-NEXT:              memref.alloca_scope  {
// CHECK-NEXT:                scf.parallel (%arg8, %arg9) = (%c0, %c0) to (%c16, %c16) step (%c1, %c1) {
// CHECK-NEXT:                  %10 = arith.index_cast %arg9 : index to i32
// CHECK-NEXT:                  %11 = "polygeist.subindex"(%9, %arg8) : (memref<?x?xi32>, index) -> memref<?xi32>
// CHECK-NEXT:                  %12 = "polygeist.subindex"(%11, %arg9) : (memref<?xi32>, index) -> memref<i32>
// CHECK-NEXT:                  %13 = memref.load %12[] : memref<i32>
// CHECK-NEXT:                  %14 = arith.sitofp %13 : i32 to f32
// CHECK-NEXT:                  %15 = math.powf %cst, %14 : f32
// CHECK-NEXT:                  %16 = arith.fptosi %15 : f32 to i32
// CHECK-NEXT:                  %17 = arith.index_cast %16 : i32 to index
// CHECK-NEXT:                  %18 = arith.remsi %10, %16 : i32
// CHECK-NEXT:                  %19 = arith.cmpi eq, %18, %c0_i32 : i32
// CHECK-NEXT:                  scf.if %19 {
// CHECK-NEXT:                    %20 = memref.load %5[%arg9, %arg8] : memref<16x16xf32>
// CHECK-NEXT:                    %21 = arith.divsi %17, %c2 : index
// CHECK-NEXT:                    %22 = arith.addi %arg9, %21 : index
// CHECK-NEXT:                    %23 = memref.load %5[%22, %arg8] : memref<16x16xf32>
// CHECK-NEXT:                    %24 = arith.addf %20, %23 : f32
// CHECK-NEXT:                    memref.store %24, %5[%arg9, %arg8] : memref<16x16xf32>
// CHECK-NEXT:                  }
// CHECK-NEXT:                  scf.yield
// CHECK-NEXT:                }
// CHECK-NEXT:                scf.parallel (%arg8, %arg9) = (%c0, %c0) to (%c16, %c16) step (%c1, %c1) {
// CHECK-NEXT:                  %10 = "polygeist.subindex"(%9, %arg8) : (memref<?x?xi32>, index) -> memref<?xi32>
// CHECK-NEXT:                  %11 = "polygeist.subindex"(%10, %arg9) : (memref<?xi32>, index) -> memref<i32>
// CHECK-NEXT:                  %12 = memref.load %11[] : memref<i32>
// CHECK-NEXT:                  %13 = arith.addi %12, %c1_i32 : i32
// CHECK-NEXT:                  %14 = "polygeist.subindex"(%8, %arg8) : (memref<?x?xi32>, index) -> memref<?xi32>
// CHECK-NEXT:                  %15 = "polygeist.subindex"(%14, %arg9) : (memref<?xi32>, index) -> memref<i32>
// CHECK-NEXT:                  memref.store %13, %15[] : memref<i32>
// CHECK-NEXT:                  scf.yield
// CHECK-NEXT:                }
// CHECK-NEXT:              }
// CHECK-NEXT:              scf.yield
// CHECK-NEXT:            }
// CHECK-NEXT:            memref.alloca_scope  {
// CHECK-NEXT:              scf.parallel (%arg8, %arg9) = (%c0, %c0) to (%c16, %c16) step (%c1, %c1) {
// CHECK-NEXT:                %10 = arith.muli %2, %arg9 : index
// CHECK-NEXT:                %11 = arith.addi %6, %10 : index
// CHECK-NEXT:                %12 = arith.addi %11, %arg8 : index
// CHECK-NEXT:                %13 = arith.addi %12, %c1 : index
// CHECK-NEXT:                %14 = arith.addi %13, %2 : index
// CHECK-NEXT:                %15 = memref.load %5[%arg9, %arg8] : memref<16x16xf32>
// CHECK-NEXT:                memref.store %15, %arg3[%14] : memref<?xf32>
// CHECK-NEXT:                scf.yield
// CHECK-NEXT:              }
// CHECK-NEXT:              scf.parallel (%arg8, %arg9) = (%c0, %c0) to (%c16, %c16) step (%c1, %c1) {
// CHECK-NEXT:                %10 = arith.index_cast %arg8 : index to i32
// CHECK-NEXT:                %11 = arith.cmpi eq, %10, %c0_i32 : i32
// CHECK-NEXT:                scf.if %11 {
// CHECK-NEXT:                  %12 = memref.load %5[%arg8, %arg9] : memref<16x16xf32>
// CHECK-NEXT:                  %13 = arith.muli %arg7, %0 : index
// CHECK-NEXT:                  %14 = arith.addi %arg9, %13 : index
// CHECK-NEXT:                  memref.store %12, %arg4[%14] : memref<?xf32>
// CHECK-NEXT:                }
// CHECK-NEXT:                scf.yield
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.yield
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
