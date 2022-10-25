// RUN: polygeist-opt --cpuify="method=distribute.mincut" --split-input-file %s | FileCheck %s
module {
  func.func @t(%arg0: memref<?xf32>, %arg1: memref<?xmemref<?xf32>>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %true = arith.constant true
    %cst = arith.constant 1.000000e+00 : f32
    %c2_i32 = arith.constant 2 : i32
    %c15_i32 = arith.constant 15 : i32
    %c14_i32 = arith.constant 14 : i32
    %c0_i8 = arith.constant 0 : i8
    %c1_i8 = arith.constant 1 : i8
    %cst_0 = arith.constant 2.000000e+00 : f64
    %cst_1 = arith.constant 8.000000e+01 : f64
    %c16 = arith.constant 16 : index
    %c-1_i32 = arith.constant -1 : i32
    %cst_2 = arith.constant 0.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %cst_3 = arith.constant 1.000000e-03 : f64
    %cst_4 = arith.constant 6857.1428571428569 : f64
    %cst_5 = arith.constant 1.000000e+02 : f32
    %cst_6 = arith.constant 5.000000e-04 : f64
    %cst_7 = arith.constant 1.000000e-01 : f64
    %cst_8 = arith.constant 4.375000e+02 : f64
    %cst_9 = arith.constant 1.600000e-02 : f64
    %c1_i32 = arith.constant 1 : i32
    %c16_i32 = arith.constant 16 : i32
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %0 = arith.sitofp %arg3 : i32 to f64
    %1 = arith.divf %cst_9, %0 : f64
    %2 = arith.truncf %1 : f64 to f32
    %3 = arith.sitofp %arg2 : i32 to f64
    %4 = arith.divf %cst_9, %3 : f64
    %5 = arith.truncf %4 : f64 to f32
    %6 = arith.extf %5 : f32 to f64
    %7 = arith.mulf %6, %cst_8 : f64
    %8 = arith.extf %2 : f32 to f64
    %9 = arith.mulf %7, %8 : f64
    %10 = arith.truncf %9 : f64 to f32
    %11 = arith.mulf %8, %cst_7 : f64
    %12 = arith.divf %6, %11 : f64
    %13 = arith.truncf %12 : f64 to f32
    %14 = arith.mulf %6, %cst_7 : f64
    %15 = arith.divf %8, %14 : f64
    %16 = arith.truncf %15 : f64 to f32
    %17 = arith.mulf %2, %cst_5 : f32
    %18 = arith.mulf %17, %5 : f32
    %19 = arith.extf %18 : f32 to f64
    %20 = arith.divf %cst_6, %19 : f64
    %21 = arith.truncf %20 : f64 to f32
    %22 = arith.truncf %cst_4 : f64 to f32
    %23 = arith.extf %22 : f32 to f64
    %24 = arith.divf %cst_3, %23 : f64
    %25 = arith.truncf %24 : f64 to f32
    %26 = arith.sitofp %arg4 : i32 to f32
    %27 = arith.sitofp %arg5 : i32 to f32
    %28 = arith.index_cast %arg6 : i32 to index
    %29 = arith.index_cast %arg7 : i32 to index
    %30 = arith.index_cast %arg8 : i32 to index
    %31 = arith.index_cast %arg9 : i32 to index
    %32 = arith.index_cast %arg2 : i32 to index
    %33 = llvm.mlir.undef : i8
    %34 = arith.divf %25, %10 : f32
    %35 = arith.divf %cst, %13 : f32
    %36 = arith.divf %cst, %16 : f32
    %37 = arith.divf %cst, %21 : f32
    %38 = arith.addi %arg3, %c-1_i32 : i32
    %39 = arith.addi %arg2, %c-1_i32 : i32
    %40:3 = scf.while (%arg10 = %c0_i32, %arg11 = %c1_i32, %arg12 = %cst_2) : (i32, i32, f32) -> (i32, i32, f32) {
      %41 = arith.cmpf ult, %arg12, %26 : f32
      scf.condition(%41) %arg10, %arg11, %arg12 : i32, i32, f32
    } do {
    ^bb0(%arg10: i32, %arg11: i32, %arg12: f32):
      %41 = arith.index_cast %arg10 : i32 to index
      %42 = arith.index_cast %arg11 : i32 to index
      %43 = arith.subf %26, %arg12 : f32
      %44 = arith.cmpf ule, %27, %43 : f32
      %45 = arith.select %44, %27, %43 : f32
      %46 = arith.fptosi %45 : f32 to i32
      %47 = memref.load %arg1[%41] : memref<?xmemref<?xf32>>
      %48 = memref.load %arg1[%42] : memref<?xmemref<?xf32>>
      %49 = arith.index_cast %46 : i32 to index
      %50 = arith.muli %46, %c2_i32 : i32
      %51 = arith.muli %49, %c2 : index
      %52 = arith.subi %c16_i32, %50 : i32
      %53 = arith.subi %c16, %51 : index
      %54 = arith.addi %46, %c-1_i32 : i32
      scf.parallel (%arg13, %arg14) = (%c0, %c0) to (%28, %29) step (%c1, %c1) {
        %56 = memref.alloca() : memref<16x16xf32>
        %57 = memref.alloca() : memref<16x16xf32>
        %58 = memref.alloca() : memref<16x16xf32>
        %59 = arith.index_cast %arg13 : index to i32
        %60 = arith.index_cast %arg14 : index to i32
        %61 = arith.muli %52, %60 : i32
        %62 = arith.muli %53, %arg14 : index
        %63 = arith.subi %61, %arg9 : i32
        %64 = arith.subi %62, %31 : index
        %65 = arith.muli %52, %59 : i32
        %66 = arith.muli %53, %arg13 : index
        %67 = arith.subi %65, %arg8 : i32
        %68 = arith.subi %66, %30 : index
        %69 = arith.addi %63, %c15_i32 : i32
        %70 = arith.addi %67, %c15_i32 : i32
        %71 = arith.cmpi slt, %63, %c0_i32 : i32
        %72 = arith.cmpi sgt, %69, %38 : i32
        %73 = arith.cmpi slt, %67, %c0_i32 : i32
        %74 = arith.cmpi sgt, %70, %39 : i32
        scf.parallel (%arg15, %arg16) = (%c0, %c0) to (%c16, %c16) step (%c1, %c1) {
          %75 = arith.index_cast %arg15 : index to i32
          %76 = arith.index_cast %arg16 : index to i32
          %77 = arith.addi %63, %76 : i32
          %78 = arith.addi %64, %arg16 : index
          %79 = arith.addi %67, %75 : i32
          %80 = arith.addi %68, %arg15 : index
          %81 = arith.muli %32, %78 : index
          %82 = arith.cmpi sge, %77, %c0_i32 : i32
          scf.if %82 {
            %105 = arith.cmpi sle, %77, %38 : i32
            scf.if %105 {
              %106 = arith.cmpi sge, %79, %c0_i32 : i32
              scf.if %106 {
                %107 = arith.cmpi sle, %79, %39 : i32
                scf.if %107 {
                  %108 = arith.addi %81, %80 : index
                  %109 = memref.load %47[%108] : memref<?xf32>
                  memref.store %109, %56[%arg16, %arg15] : memref<16x16xf32>
                  %110 = memref.load %arg0[%108] : memref<?xf32>
                  memref.store %110, %57[%arg16, %arg15] : memref<16x16xf32>
                }
              }
            }
          }
          "polygeist.barrier"(%arg15, %arg16, %c0) : (index, index, index) -> ()
          %83 = scf.if %71 -> (i32) {
            %105 = arith.subi %c0_i32, %63 : i32
            scf.yield %105 : i32
          } else {
            scf.yield %c0_i32 : i32
          }
          %84 = scf.if %72 -> (i32) {
            %105 = arith.subi %69, %arg3 : i32
            %106 = arith.subi %c14_i32, %105 : i32
            scf.yield %106 : i32
          } else {
            scf.yield %c15_i32 : i32
          }
          %85 = scf.if %73 -> (i32) {
            %105 = arith.subi %c0_i32, %67 : i32
            scf.yield %105 : i32
          } else {
            scf.yield %c0_i32 : i32
          }
          %86 = scf.if %74 -> (i32) {
            %105 = arith.subi %70, %arg2 : i32
            %106 = arith.subi %c14_i32, %105 : i32
            scf.yield %106 : i32
          } else {
            scf.yield %c15_i32 : i32
          }
          %87 = arith.addi %76, %c-1_i32 : i32
          %88 = arith.addi %76, %c1_i32 : i32
          %89 = arith.addi %75, %c-1_i32 : i32
          %90 = arith.addi %75, %c1_i32 : i32
          %91 = arith.cmpi slt, %87, %83 : i32
          %92 = arith.select %91, %83, %87 : i32
          %93 = arith.index_cast %92 : i32 to index
          %94 = arith.cmpi sgt, %88, %84 : i32
          %95 = arith.select %94, %84, %88 : i32
          %96 = arith.index_cast %95 : i32 to index
          %97 = arith.cmpi slt, %89, %85 : i32
          %98 = arith.select %97, %85, %89 : i32
          %99 = arith.index_cast %98 : i32 to index
          %100 = arith.cmpi sgt, %90, %86 : i32
          %101 = arith.select %100, %86, %90 : i32
          %102 = arith.index_cast %101 : i32 to index
          %103:2 = scf.while (%arg17 = %c0_i32, %arg18 = %33, %arg19 = %true) : (i32, i8, i1) -> (i8, i32) {
            %105 = arith.cmpi slt, %arg17, %46 : i32
            %106 = arith.andi %105, %arg19 : i1
            scf.condition(%106) %arg18, %arg17 : i8, i32
          } do {
          ^bb0(%arg17: i8, %arg18: i32):
            %105 = arith.addi %arg18, %c1_i32 : i32
            %106 = arith.cmpi sge, %75, %105 : i32
            %107 = scf.if %106 -> (i8) {
              %110 = arith.subi %c14_i32, %arg18 : i32
              %111 = arith.cmpi sle, %75, %110 : i32
              %112 = scf.if %111 -> (i8) {
                %113 = arith.cmpi sge, %76, %105 : i32
                %114 = scf.if %113 -> (i8) {
                  %115 = arith.cmpi sle, %76, %110 : i32
                  %116 = scf.if %115 -> (i8) {
                    %117 = arith.cmpi sge, %75, %85 : i32
                    %118 = scf.if %117 -> (i8) {
                      %119 = arith.cmpi sle, %75, %86 : i32
                      %120 = scf.if %119 -> (i8) {
                        %121 = arith.cmpi sge, %76, %83 : i32
                        %122 = scf.if %121 -> (i8) {
                          %123 = arith.cmpi sle, %76, %84 : i32
                          %124 = scf.if %123 -> (i8) {
                            %125 = memref.load %56[%arg16, %arg15] : memref<16x16xf32>
                            %126 = arith.extf %125 : f32 to f64
                            %127 = arith.extf %34 : f32 to f64
                            %128 = memref.load %57[%arg16, %arg15] : memref<16x16xf32>
                            %129 = arith.extf %128 : f32 to f64
                            %130 = memref.load %56[%96, %arg15] : memref<16x16xf32>
                            %131 = memref.load %56[%93, %arg15] : memref<16x16xf32>
                            %132 = arith.addf %130, %131 : f32
                            %133 = arith.extf %132 : f32 to f64
                            %134 = arith.mulf %126, %cst_0 : f64
                            %135 = arith.subf %133, %134 : f64
                            %136 = arith.extf %36 : f32 to f64
                            %137 = arith.mulf %135, %136 : f64
                            %138 = arith.addf %129, %137 : f64
                            %139 = memref.load %56[%arg16, %102] : memref<16x16xf32>
                            %140 = memref.load %56[%arg16, %99] : memref<16x16xf32>
                            %141 = arith.addf %139, %140 : f32
                            %142 = arith.extf %141 : f32 to f64
                            %143 = arith.subf %142, %134 : f64
                            %144 = arith.extf %35 : f32 to f64
                            %145 = arith.mulf %143, %144 : f64
                            %146 = arith.addf %138, %145 : f64
                            %147 = arith.subf %cst_1, %126 : f64
                            %148 = arith.extf %37 : f32 to f64
                            %149 = arith.mulf %147, %148 : f64
                            %150 = arith.addf %146, %149 : f64
                            %151 = arith.mulf %127, %150 : f64
                            %152 = arith.addf %126, %151 : f64
                            %153 = arith.truncf %152 : f64 to f32
                            memref.store %153, %58[%arg16, %arg15] : memref<16x16xf32>
                            scf.yield %c1_i8 : i8
                          } else {
                            scf.yield %c0_i8 : i8
                          }
                          scf.yield %124 : i8
                        } else {
                          scf.yield %c0_i8 : i8
                        }
                        scf.yield %122 : i8
                      } else {
                        scf.yield %c0_i8 : i8
                      }
                      scf.yield %120 : i8
                    } else {
                      scf.yield %c0_i8 : i8
                    }
                    scf.yield %118 : i8
                  } else {
                    scf.yield %c0_i8 : i8
                  }
                  scf.yield %116 : i8
                } else {
                  scf.yield %c0_i8 : i8
                }
                scf.yield %114 : i8
              } else {
                scf.yield %c0_i8 : i8
              }
              scf.yield %112 : i8
            } else {
              scf.yield %c0_i8 : i8
            }
            "polygeist.barrier"(%arg15, %arg16, %c0) : (index, index, index) -> ()
            %108 = arith.cmpi ne, %arg18, %54 : i32
            %109 = scf.if %108 -> (i32) {
              %110 = arith.cmpi ne, %107, %c0_i8 : i8
              scf.if %110 {
                %111 = memref.load %58[%arg16, %arg15] : memref<16x16xf32>
                memref.store %111, %56[%arg16, %arg15] : memref<16x16xf32>
              }
              "polygeist.barrier"(%arg15, %arg16, %c0) : (index, index, index) -> ()
              scf.yield %105 : i32
            } else {
              scf.yield %arg18 : i32
            }
            scf.yield %109, %107, %108 : i32, i8, i1
          }
          %104 = arith.cmpi ne, %103#0, %c0_i8 : i8
          scf.if %104 {
            %105 = memref.load %58[%arg16, %arg15] : memref<16x16xf32>
            %106 = arith.addi %81, %80 : index
            memref.store %105, %48[%106] : memref<?xf32>
          }
          scf.yield
        }
        scf.yield
      }
      %55 = arith.addf %arg12, %27 : f32
      scf.yield %arg11, %arg10, %55 : i32, i32, f32
    }
    return %40#0 : i32
  }
}

// CHECK:   func.func @t(%arg0: memref<?xf32>, %arg1: memref<?xmemref<?xf32>>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) -> i32 
// CHECK-NEXT:    %true = arith.constant true
// CHECK-NEXT:    %cst = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:    %c2_i32 = arith.constant 2 : i32
// CHECK-NEXT:    %c15_i32 = arith.constant 15 : i32
// CHECK-NEXT:    %c14_i32 = arith.constant 14 : i32
// CHECK-NEXT:    %c0_i8 = arith.constant 0 : i8
// CHECK-NEXT:    %c1_i8 = arith.constant 1 : i8
// CHECK-NEXT:    %cst_0 = arith.constant 2.000000e+00 : f64
// CHECK-NEXT:    %cst_1 = arith.constant 8.000000e+01 : f64
// CHECK-NEXT:    %c16 = arith.constant 16 : index
// CHECK-NEXT:    %c-1_i32 = arith.constant -1 : i32
// CHECK-NEXT:    %cst_2 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    %cst_3 = arith.constant 1.000000e-03 : f64
// CHECK-NEXT:    %cst_4 = arith.constant 6857.1428571428569 : f64
// CHECK-NEXT:    %cst_5 = arith.constant 1.000000e+02 : f32
// CHECK-NEXT:    %cst_6 = arith.constant 5.000000e-04 : f64
// CHECK-NEXT:    %cst_7 = arith.constant 1.000000e-01 : f64
// CHECK-NEXT:    %cst_8 = arith.constant 4.375000e+02 : f64
// CHECK-NEXT:    %cst_9 = arith.constant 1.600000e-02 : f64
// CHECK-NEXT:    %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:    %c16_i32 = arith.constant 16 : i32
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %c2 = arith.constant 2 : index
// CHECK-NEXT:    %0 = arith.sitofp %arg3 : i32 to f64
// CHECK-NEXT:    %1 = arith.divf %cst_9, %0 : f64
// CHECK-NEXT:    %2 = arith.truncf %1 : f64 to f32
// CHECK-NEXT:    %3 = arith.sitofp %arg2 : i32 to f64
// CHECK-NEXT:    %4 = arith.divf %cst_9, %3 : f64
// CHECK-NEXT:    %5 = arith.truncf %4 : f64 to f32
// CHECK-NEXT:    %6 = arith.extf %5 : f32 to f64
// CHECK-NEXT:    %7 = arith.mulf %6, %cst_8 : f64
// CHECK-NEXT:    %8 = arith.extf %2 : f32 to f64
// CHECK-NEXT:    %9 = arith.mulf %7, %8 : f64
// CHECK-NEXT:    %10 = arith.truncf %9 : f64 to f32
// CHECK-NEXT:    %11 = arith.mulf %8, %cst_7 : f64
// CHECK-NEXT:    %12 = arith.divf %6, %11 : f64
// CHECK-NEXT:    %13 = arith.truncf %12 : f64 to f32
// CHECK-NEXT:    %14 = arith.mulf %6, %cst_7 : f64
// CHECK-NEXT:    %15 = arith.divf %8, %14 : f64
// CHECK-NEXT:    %16 = arith.truncf %15 : f64 to f32
// CHECK-NEXT:    %17 = arith.mulf %2, %cst_5 : f32
// CHECK-NEXT:    %18 = arith.mulf %17, %5 : f32
// CHECK-NEXT:    %19 = arith.extf %18 : f32 to f64
// CHECK-NEXT:    %20 = arith.divf %cst_6, %19 : f64
// CHECK-NEXT:    %21 = arith.truncf %20 : f64 to f32
// CHECK-NEXT:    %22 = arith.truncf %cst_4 : f64 to f32
// CHECK-NEXT:    %23 = arith.extf %22 : f32 to f64
// CHECK-NEXT:    %24 = arith.divf %cst_3, %23 : f64
// CHECK-NEXT:    %25 = arith.truncf %24 : f64 to f32
// CHECK-NEXT:    %26 = arith.sitofp %arg4 : i32 to f32
// CHECK-NEXT:    %27 = arith.sitofp %arg5 : i32 to f32
// CHECK-NEXT:    %28 = arith.index_cast %arg6 : i32 to index
// CHECK-NEXT:    %29 = arith.index_cast %arg7 : i32 to index
// CHECK-NEXT:    %30 = arith.index_cast %arg8 : i32 to index
// CHECK-NEXT:    %31 = arith.index_cast %arg9 : i32 to index
// CHECK-NEXT:    %32 = arith.index_cast %arg2 : i32 to index
// CHECK-NEXT:    %33 = arith.divf %25, %10 : f32
// CHECK-NEXT:    %34 = arith.divf %cst, %13 : f32
// CHECK-NEXT:    %35 = arith.divf %cst, %16 : f32
// CHECK-NEXT:    %36 = arith.divf %cst, %21 : f32
// CHECK-NEXT:    %37 = arith.addi %arg3, %c-1_i32 : i32
// CHECK-NEXT:    %38 = arith.addi %arg2, %c-1_i32 : i32
// CHECK-NEXT:    %39:3 = scf.while (%arg10 = %c0_i32, %arg11 = %c1_i32, %arg12 = %cst_2) : (i32, i32, f32) -> (i32, i32, f32) {
// CHECK-NEXT:      %40 = arith.cmpf ult, %arg12, %26 : f32
// CHECK-NEXT:      scf.condition(%40) %arg10, %arg11, %arg12 : i32, i32, f32
// CHECK-NEXT:    } do {
// CHECK-NEXT:    ^bb0(%arg10: i32, %arg11: i32, %arg12: f32):
// CHECK-NEXT:      %40 = arith.index_cast %arg10 : i32 to index
// CHECK-NEXT:      %41 = arith.index_cast %arg11 : i32 to index
// CHECK-NEXT:      %42 = arith.subf %26, %arg12 : f32
// CHECK-NEXT:      %43 = arith.cmpf ule, %27, %42 : f32
// CHECK-NEXT:      %44 = arith.select %43, %27, %42 : f32
// CHECK-NEXT:      %45 = arith.fptosi %44 : f32 to i32
// CHECK-NEXT:      %46 = memref.load %arg1[%40] : memref<?xmemref<?xf32>>
// CHECK-NEXT:      %47 = memref.load %arg1[%41] : memref<?xmemref<?xf32>>
// CHECK-NEXT:      %48 = arith.index_cast %45 : i32 to index
// CHECK-NEXT:      %49 = arith.muli %45, %c2_i32 : i32
// CHECK-NEXT:      %50 = arith.muli %48, %c2 : index
// CHECK-NEXT:      %51 = arith.subi %c16_i32, %49 : i32
// CHECK-NEXT:      %52 = arith.subi %c16, %50 : index
// CHECK-NEXT:      %53 = arith.addi %45, %c-1_i32 : i32
// CHECK-NEXT:      scf.parallel (%arg13, %arg14) = (%c0, %c0) to (%28, %29) step (%c1, %c1) {
// CHECK-NEXT:        %alloca = memref.alloca() : memref<16x16xf32>
// CHECK-NEXT:        %alloca_10 = memref.alloca() : memref<16x16xf32>
// CHECK-NEXT:        %alloca_11 = memref.alloca() : memref<16x16xf32>
// CHECK-NEXT:        %55 = arith.index_cast %arg13 : index to i32
// CHECK-NEXT:        %56 = arith.index_cast %arg14 : index to i32
// CHECK-NEXT:        %57 = arith.muli %51, %56 : i32
// CHECK-NEXT:        %58 = arith.muli %52, %arg14 : index
// CHECK-NEXT:        %59 = arith.subi %57, %arg9 : i32
// CHECK-NEXT:        %60 = arith.subi %58, %31 : index
// CHECK-NEXT:        %61 = arith.muli %51, %55 : i32
// CHECK-NEXT:        %62 = arith.muli %52, %arg13 : index
// CHECK-NEXT:        %63 = arith.subi %61, %arg8 : i32
// CHECK-NEXT:        %64 = arith.subi %62, %30 : index
// CHECK-NEXT:        %65 = arith.addi %59, %c15_i32 : i32
// CHECK-NEXT:        %66 = arith.addi %63, %c15_i32 : i32
// CHECK-NEXT:        %67 = arith.cmpi slt, %59, %c0_i32 : i32
// CHECK-NEXT:        %68 = arith.cmpi sgt, %65, %37 : i32
// CHECK-NEXT:        %69 = arith.cmpi slt, %63, %c0_i32 : i32
// CHECK-NEXT:        %70 = arith.cmpi sgt, %66, %38 : i32
// CHECK-NEXT:        memref.alloca_scope  {
// CHECK-NEXT:          scf.parallel (%arg15, %arg16) = (%c0, %c0) to (%c16, %c16) step (%c1, %c1) {
// CHECK-NEXT:            %71 = arith.index_cast %arg15 : index to i32
// CHECK-NEXT:            %72 = arith.index_cast %arg16 : index to i32
// CHECK-NEXT:            %73 = arith.addi %59, %72 : i32
// CHECK-NEXT:            %74 = arith.addi %60, %arg16 : index
// CHECK-NEXT:            %75 = arith.addi %63, %71 : i32
// CHECK-NEXT:            %76 = arith.addi %64, %arg15 : index
// CHECK-NEXT:            %77 = arith.muli %32, %74 : index
// CHECK-NEXT:            %78 = arith.cmpi sge, %73, %c0_i32 : i32
// CHECK-NEXT:            scf.if %78 {
// CHECK-NEXT:              %79 = arith.cmpi sle, %73, %37 : i32
// CHECK-NEXT:              scf.if %79 {
// CHECK-NEXT:                %80 = arith.cmpi sge, %75, %c0_i32 : i32
// CHECK-NEXT:                scf.if %80 {
// CHECK-NEXT:                  %81 = arith.cmpi sle, %75, %38 : i32
// CHECK-NEXT:                  scf.if %81 {
// CHECK-NEXT:                    %82 = arith.addi %77, %76 : index
// CHECK-NEXT:                    %83 = memref.load %46[%82] : memref<?xf32>
// CHECK-NEXT:                    memref.store %83, %alloca[%arg16, %arg15] : memref<16x16xf32>
// CHECK-NEXT:                    %84 = memref.load %arg0[%82] : memref<?xf32>
// CHECK-NEXT:                    memref.store %84, %alloca_10[%arg16, %arg15] : memref<16x16xf32>
// CHECK-NEXT:                  }
// CHECK-NEXT:                }
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:            scf.yield
// CHECK-NEXT:          }
// CHECK-NEXT:          memref.alloca_scope  {
// CHECK-NEXT:            %alloca_12 = memref.alloca(%c16, %c16) : memref<?x?xi32>
// CHECK-NEXT:            %alloca_13 = memref.alloca(%c16, %c16) : memref<?x?xi1>
// CHECK-NEXT:            %alloca_14 = memref.alloca(%c16, %c16) : memref<?x?xi32>
// CHECK-NEXT:            %alloca_15 = memref.alloca(%c16, %c16) : memref<?x?xi8>
// CHECK-NEXT:            %alloca_16 = memref.alloca(%c16, %c16) : memref<?x?xi8>
// CHECK-NEXT:            scf.parallel (%arg15, %arg16) = (%c0, %c0) to (%c16, %c16) step (%c1, %c1) {
// CHECK-NEXT:              %71 = "polygeist.subindex"(%alloca_12, %arg15) : (memref<?x?xi32>, index) -> memref<?xi32>
// CHECK-NEXT:              %72 = "polygeist.subindex"(%71, %arg16) : (memref<?xi32>, index) -> memref<i32>
// CHECK-NEXT:              memref.store %c0_i32, %72[] : memref<i32>
// CHECK-NEXT:              %73 = "polygeist.subindex"(%alloca_13, %arg15) : (memref<?x?xi1>, index) -> memref<?xi1>
// CHECK-NEXT:              %74 = "polygeist.subindex"(%73, %arg16) : (memref<?xi1>, index) -> memref<i1>
// CHECK-NEXT:              memref.store %true, %74[] : memref<i1>
// CHECK-NEXT:              scf.yield
// CHECK-NEXT:            }
// CHECK-NEXT:            memref.alloca_scope  {
// CHECK-NEXT:              scf.while : () -> () {
// CHECK-NEXT:                %alloca_17 = memref.alloca() : memref<i1>
// CHECK-NEXT:                scf.parallel (%arg15, %arg16) = (%c0, %c0) to (%c16, %c16) step (%c1, %c1) {
// CHECK-NEXT:                  %72 = "polygeist.subindex"(%alloca_12, %arg15) : (memref<?x?xi32>, index) -> memref<?xi32>
// CHECK-NEXT:                  %73 = "polygeist.subindex"(%72, %arg16) : (memref<?xi32>, index) -> memref<i32>
// CHECK-NEXT:                  %74 = memref.load %73[] : memref<i32>
// CHECK-NEXT:                  %75 = "polygeist.subindex"(%alloca_16, %arg15) : (memref<?x?xi8>, index) -> memref<?xi8>
// CHECK-NEXT:                  %76 = "polygeist.subindex"(%75, %arg16) : (memref<?xi8>, index) -> memref<i8>
// CHECK-NEXT:                  %77 = memref.load %76[] : memref<i8>
// CHECK-NEXT:                  %78 = "polygeist.subindex"(%alloca_13, %arg15) : (memref<?x?xi1>, index) -> memref<?xi1>
// CHECK-NEXT:                  %79 = "polygeist.subindex"(%78, %arg16) : (memref<?xi1>, index) -> memref<i1>
// CHECK-NEXT:                  %80 = memref.load %79[] : memref<i1>
// CHECK-NEXT:                  %81 = arith.cmpi slt, %74, %45 : i32
// CHECK-NEXT:                  %82 = arith.andi %81, %80 : i1
// CHECK-NEXT:                  %83 = arith.cmpi eq, %arg15, %c0 : index
// CHECK-NEXT:                  %84 = arith.cmpi eq, %arg16, %c0 : index
// CHECK-NEXT:                  %85 = arith.andi %84, %83 : i1
// CHECK-NEXT:                  scf.if %85 {
// CHECK-NEXT:                    memref.store %82, %alloca_17[] : memref<i1>
// CHECK-NEXT:                  }
// CHECK-NEXT:                  %86 = "polygeist.subindex"(%alloca_15, %arg15) : (memref<?x?xi8>, index) -> memref<?xi8>
// CHECK-NEXT:                  %87 = "polygeist.subindex"(%86, %arg16) : (memref<?xi8>, index) -> memref<i8>
// CHECK-NEXT:                  memref.store %77, %87[] : memref<i8>
// CHECK-NEXT:                  %88 = "polygeist.subindex"(%alloca_14, %arg15) : (memref<?x?xi32>, index) -> memref<?xi32>
// CHECK-NEXT:                  %89 = "polygeist.subindex"(%88, %arg16) : (memref<?xi32>, index) -> memref<i32>
// CHECK-NEXT:                  memref.store %74, %89[] : memref<i32>
// CHECK-NEXT:                  scf.yield
// CHECK-NEXT:                }
// CHECK-NEXT:                %71 = memref.load %alloca_17[] : memref<i1>
// CHECK-NEXT:                scf.condition(%71)
// CHECK-NEXT:              } do {
// CHECK-NEXT:                memref.alloca_scope  {
// CHECK-NEXT:                  %alloca_17 = memref.alloca(%c16, %c16) : memref<?x?xi8>
// CHECK-NEXT:                  scf.parallel (%arg15, %arg16) = (%c0, %c0) to (%c16, %c16) step (%c1, %c1) {
// CHECK-NEXT:                    %71 = arith.index_cast %arg15 : index to i32
// CHECK-NEXT:                    %72 = arith.addi %71, %c1_i32 : i32
// CHECK-NEXT:                    %73 = scf.if %70 -> (i32) {
// CHECK-NEXT:                      %99 = arith.subi %66, %arg2 : i32
// CHECK-NEXT:                      %100 = arith.subi %c14_i32, %99 : i32
// CHECK-NEXT:                      scf.yield %100 : i32
// CHECK-NEXT:                    } else {
// CHECK-NEXT:                      scf.yield %c15_i32 : i32
// CHECK-NEXT:                    }
// CHECK-NEXT:                    %74 = arith.cmpi sgt, %72, %73 : i32
// CHECK-NEXT:                    %75 = arith.select %74, %73, %72 : i32
// CHECK-NEXT:                    %76 = arith.index_cast %75 : i32 to index
// CHECK-NEXT:                    %77 = arith.addi %71, %c-1_i32 : i32
// CHECK-NEXT:                    %78 = scf.if %69 -> (i32) {
// CHECK-NEXT:                      %99 = arith.subi %c0_i32, %63 : i32
// CHECK-NEXT:                      scf.yield %99 : i32
// CHECK-NEXT:                    } else {
// CHECK-NEXT:                      scf.yield %c0_i32 : i32
// CHECK-NEXT:                    }
// CHECK-NEXT:                    %79 = arith.cmpi slt, %77, %78 : i32
// CHECK-NEXT:                    %80 = arith.select %79, %78, %77 : i32
// CHECK-NEXT:                    %81 = arith.index_cast %80 : i32 to index
// CHECK-NEXT:                    %82 = arith.index_cast %arg16 : index to i32
// CHECK-NEXT:                    %83 = arith.addi %82, %c1_i32 : i32
// CHECK-NEXT:                    %84 = scf.if %68 -> (i32) {
// CHECK-NEXT:                      %99 = arith.subi %65, %arg3 : i32
// CHECK-NEXT:                      %100 = arith.subi %c14_i32, %99 : i32
// CHECK-NEXT:                      scf.yield %100 : i32
// CHECK-NEXT:                    } else {
// CHECK-NEXT:                      scf.yield %c15_i32 : i32
// CHECK-NEXT:                    }
// CHECK-NEXT:                    %85 = arith.cmpi sgt, %83, %84 : i32
// CHECK-NEXT:                    %86 = arith.select %85, %84, %83 : i32
// CHECK-NEXT:                    %87 = arith.index_cast %86 : i32 to index
// CHECK-NEXT:                    %88 = arith.addi %82, %c-1_i32 : i32
// CHECK-NEXT:                    %89 = scf.if %67 -> (i32) {
// CHECK-NEXT:                      %99 = arith.subi %c0_i32, %59 : i32
// CHECK-NEXT:                      scf.yield %99 : i32
// CHECK-NEXT:                    } else {
// CHECK-NEXT:                      scf.yield %c0_i32 : i32
// CHECK-NEXT:                    }
// CHECK-NEXT:                    %90 = arith.cmpi slt, %88, %89 : i32
// CHECK-NEXT:                    %91 = arith.select %90, %89, %88 : i32
// CHECK-NEXT:                    %92 = arith.index_cast %91 : i32 to index
// CHECK-NEXT:                    %93 = "polygeist.subindex"(%alloca_14, %arg15) : (memref<?x?xi32>, index) -> memref<?xi32>
// CHECK-NEXT:                    %94 = "polygeist.subindex"(%93, %arg16) : (memref<?xi32>, index) -> memref<i32>
// CHECK-NEXT:                    %95 = memref.load %94[] : memref<i32>
// CHECK-NEXT:                    %96 = arith.addi %95, %c1_i32 : i32
// CHECK-NEXT:                    %97 = arith.cmpi sge, %71, %96 : i32
// CHECK-NEXT:                    %98 = scf.if %97 -> (i8) {
// CHECK-NEXT:                      %99 = arith.subi %c14_i32, %95 : i32
// CHECK-NEXT:                      %100 = arith.cmpi sle, %71, %99 : i32
// CHECK-NEXT:                      %101 = scf.if %100 -> (i8) {
// CHECK-NEXT:                        %102 = arith.cmpi sge, %82, %96 : i32
// CHECK-NEXT:                        %103 = scf.if %102 -> (i8) {
// CHECK-NEXT:                          %104 = arith.cmpi sle, %82, %99 : i32
// CHECK-NEXT:                          %105 = scf.if %104 -> (i8) {
// CHECK-NEXT:                            %106 = arith.cmpi sge, %71, %78 : i32
// CHECK-NEXT:                            %107 = scf.if %106 -> (i8) {
// CHECK-NEXT:                              %108 = arith.cmpi sle, %71, %73 : i32
// CHECK-NEXT:                              %109 = scf.if %108 -> (i8) {
// CHECK-NEXT:                                %110 = arith.cmpi sge, %82, %89 : i32
// CHECK-NEXT:                                %111 = scf.if %110 -> (i8) {
// CHECK-NEXT:                                  %112 = arith.cmpi sle, %82, %84 : i32
// CHECK-NEXT:                                  %113 = scf.if %112 -> (i8) {
// CHECK-NEXT:                                    %114 = memref.load %alloca[%arg16, %arg15] : memref<16x16xf32>
// CHECK-NEXT:                                    %115 = arith.extf %114 : f32 to f64
// CHECK-NEXT:                                    %116 = arith.extf %33 : f32 to f64
// CHECK-NEXT:                                    %117 = memref.load %alloca_10[%arg16, %arg15] : memref<16x16xf32>
// CHECK-NEXT:                                    %118 = arith.extf %117 : f32 to f64
// CHECK-NEXT:                                    %119 = memref.load %alloca[%87, %arg15] : memref<16x16xf32>
// CHECK-NEXT:                                    %120 = memref.load %alloca[%92, %arg15] : memref<16x16xf32>
// CHECK-NEXT:                                    %121 = arith.addf %119, %120 : f32
// CHECK-NEXT:                                    %122 = arith.extf %121 : f32 to f64
// CHECK-NEXT:                                    %123 = arith.mulf %115, %cst_0 : f64
// CHECK-NEXT:                                    %124 = arith.subf %122, %123 : f64
// CHECK-NEXT:                                    %125 = arith.extf %35 : f32 to f64
// CHECK-NEXT:                                    %126 = arith.mulf %124, %125 : f64
// CHECK-NEXT:                                    %127 = arith.addf %118, %126 : f64
// CHECK-NEXT:                                    %128 = memref.load %alloca[%arg16, %76] : memref<16x16xf32>
// CHECK-NEXT:                                    %129 = memref.load %alloca[%arg16, %81] : memref<16x16xf32>
// CHECK-NEXT:                                    %130 = arith.addf %128, %129 : f32
// CHECK-NEXT:                                    %131 = arith.extf %130 : f32 to f64
// CHECK-NEXT:                                    %132 = arith.subf %131, %123 : f64
// CHECK-NEXT:                                    %133 = arith.extf %34 : f32 to f64
// CHECK-NEXT:                                    %134 = arith.mulf %132, %133 : f64
// CHECK-NEXT:                                    %135 = arith.addf %127, %134 : f64
// CHECK-NEXT:                                    %136 = arith.subf %cst_1, %115 : f64
// CHECK-NEXT:                                    %137 = arith.extf %36 : f32 to f64
// CHECK-NEXT:                                    %138 = arith.mulf %136, %137 : f64
// CHECK-NEXT:                                    %139 = arith.addf %135, %138 : f64
// CHECK-NEXT:                                    %140 = arith.mulf %116, %139 : f64
// CHECK-NEXT:                                    %141 = arith.addf %115, %140 : f64
// CHECK-NEXT:                                    %142 = arith.truncf %141 : f64 to f32
// CHECK-NEXT:                                    memref.store %142, %alloca_11[%arg16, %arg15] : memref<16x16xf32>
// CHECK-NEXT:                                    scf.yield %c1_i8 : i8
// CHECK-NEXT:                                  } else {
// CHECK-NEXT:                                    scf.yield %c0_i8 : i8
// CHECK-NEXT:                                  }
// CHECK-NEXT:                                  scf.yield %113 : i8
// CHECK-NEXT:                                } else {
// CHECK-NEXT:                                  scf.yield %c0_i8 : i8
// CHECK-NEXT:                                }
// CHECK-NEXT:                                scf.yield %111 : i8
// CHECK-NEXT:                              } else {
// CHECK-NEXT:                                scf.yield %c0_i8 : i8
// CHECK-NEXT:                              }
// CHECK-NEXT:                              scf.yield %109 : i8
// CHECK-NEXT:                            } else {
// CHECK-NEXT:                              scf.yield %c0_i8 : i8
// CHECK-NEXT:                            }
// CHECK-NEXT:                            scf.yield %107 : i8
// CHECK-NEXT:                          } else {
// CHECK-NEXT:                            scf.yield %c0_i8 : i8
// CHECK-NEXT:                          }
// CHECK-NEXT:                          scf.yield %105 : i8
// CHECK-NEXT:                        } else {
// CHECK-NEXT:                          scf.yield %c0_i8 : i8
// CHECK-NEXT:                        }
// CHECK-NEXT:                        scf.yield %103 : i8
// CHECK-NEXT:                      } else {
// CHECK-NEXT:                        scf.yield %c0_i8 : i8
// CHECK-NEXT:                      }
// CHECK-NEXT:                      scf.yield %101 : i8
// CHECK-NEXT:                    } else {
// CHECK-NEXT:                      scf.yield %c0_i8 : i8
// CHECK-NEXT:                    }
// CHECK-NEXT:                    memref.store %98, %alloca_17[%arg15, %arg16] : memref<?x?xi8>
// CHECK-NEXT:                    scf.yield
// CHECK-NEXT:                  }
// CHECK-NEXT:                  scf.parallel (%arg15, %arg16) = (%c0, %c0) to (%c16, %c16) step (%c1, %c1) {
// CHECK-NEXT:                    %71 = memref.load %alloca_17[%arg15, %arg16] : memref<?x?xi8>
// CHECK-NEXT:                    %72 = "polygeist.subindex"(%alloca_14, %arg15) : (memref<?x?xi32>, index) -> memref<?xi32>
// CHECK-NEXT:                    %73 = "polygeist.subindex"(%72, %arg16) : (memref<?xi32>, index) -> memref<i32>
// CHECK-NEXT:                    %74 = memref.load %73[] : memref<i32>
// CHECK-NEXT:                    %75 = arith.addi %74, %c1_i32 : i32
// CHECK-NEXT:                    %76 = arith.cmpi ne, %74, %53 : i32
// CHECK-NEXT:                    scf.if %76 {
// CHECK-NEXT:                      %81 = arith.cmpi ne, %71, %c0_i8 : i8
// CHECK-NEXT:                      scf.if %81 {
// CHECK-NEXT:                        %84 = memref.load %alloca_11[%arg16, %arg15] : memref<16x16xf32>
// CHECK-NEXT:                        memref.store %84, %alloca[%arg16, %arg15] : memref<16x16xf32>
// CHECK-NEXT:                      }
// CHECK-NEXT:                      %82 = "polygeist.subindex"(%alloca_12, %arg15) : (memref<?x?xi32>, index) -> memref<?xi32>
// CHECK-NEXT:                      %83 = "polygeist.subindex"(%82, %arg16) : (memref<?xi32>, index) -> memref<i32>
// CHECK-NEXT:                      memref.store %75, %83[] : memref<i32>
// CHECK-NEXT:                    } else {
// CHECK-NEXT:                      %81 = "polygeist.subindex"(%alloca_12, %arg15) : (memref<?x?xi32>, index) -> memref<?xi32>
// CHECK-NEXT:                      %82 = "polygeist.subindex"(%81, %arg16) : (memref<?xi32>, index) -> memref<i32>
// CHECK-NEXT:                      memref.store %74, %82[] : memref<i32>
// CHECK-NEXT:                    }
// CHECK-NEXT:                    %77 = "polygeist.subindex"(%alloca_16, %arg15) : (memref<?x?xi8>, index) -> memref<?xi8>
// CHECK-NEXT:                    %78 = "polygeist.subindex"(%77, %arg16) : (memref<?xi8>, index) -> memref<i8>
// CHECK-NEXT:                    memref.store %71, %78[] : memref<i8>
// CHECK-NEXT:                    %79 = "polygeist.subindex"(%alloca_13, %arg15) : (memref<?x?xi1>, index) -> memref<?xi1>
// CHECK-NEXT:                    %80 = "polygeist.subindex"(%79, %arg16) : (memref<?xi1>, index) -> memref<i1>
// CHECK-NEXT:                    memref.store %76, %80[] : memref<i1>
// CHECK-NEXT:                    scf.yield
// CHECK-NEXT:                  }
// CHECK-NEXT:                }
// CHECK-NEXT:                scf.yield
// CHECK-NEXT:              }
// CHECK-NEXT:              scf.parallel (%arg15, %arg16) = (%c0, %c0) to (%c16, %c16) step (%c1, %c1) {
// CHECK-NEXT:                %71 = arith.addi %60, %arg16 : index
// CHECK-NEXT:                %72 = arith.muli %32, %71 : index
// CHECK-NEXT:                %73 = arith.addi %64, %arg15 : index
// CHECK-NEXT:                %74 = "polygeist.subindex"(%alloca_15, %arg15) : (memref<?x?xi8>, index) -> memref<?xi8>
// CHECK-NEXT:                %75 = "polygeist.subindex"(%74, %arg16) : (memref<?xi8>, index) -> memref<i8>
// CHECK-NEXT:                %76 = memref.load %75[] : memref<i8>
// CHECK-NEXT:                %77 = arith.cmpi ne, %76, %c0_i8 : i8
// CHECK-NEXT:                scf.if %77 {
// CHECK-NEXT:                  %78 = memref.load %alloca_11[%arg16, %arg15] : memref<16x16xf32>
// CHECK-NEXT:                  %79 = arith.addi %72, %73 : index
// CHECK-NEXT:                  memref.store %78, %47[%79] : memref<?xf32>
// CHECK-NEXT:                }
// CHECK-NEXT:                scf.yield
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        scf.yield
// CHECK-NEXT:      }
// CHECK-NEXT:      %54 = arith.addf %arg12, %27 : f32
// CHECK-NEXT:      scf.yield %arg11, %arg10, %54 : i32, i32, f32
// CHECK-NEXT:    }
// CHECK-NEXT:    return %39#0 : i32
// CHECK-NEXT:  }
