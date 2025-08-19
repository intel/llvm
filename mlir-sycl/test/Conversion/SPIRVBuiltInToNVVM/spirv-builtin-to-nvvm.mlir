// RUN: sycl-mlir-opt -convert-spirv-builtin-to-nvvm -canonicalize %s -o - | FileCheck %s

llvm.mlir.global external constant @__spirv_BuiltInGlobalOffset() {addr_space = 1 : i32} : vector<3xi64>
llvm.mlir.global external constant @__spirv_BuiltInLocalInvocationId() {addr_space = 1 : i32} : vector<3xi64>
llvm.mlir.global external constant @__spirv_BuiltInGlobalInvocationId() {addr_space = 1 : i32} : vector<3xi64>
llvm.mlir.global external constant @__spirv_BuiltInWorkgroupId() {addr_space = 1 : i32} : vector<3xi64>
llvm.mlir.global external constant @__spirv_BuiltInNumWorkgroups() {addr_space = 1 : i32} : vector<3xi64>
llvm.mlir.global external constant @__spirv_BuiltInWorkgroupSize() {addr_space = 1 : i32} : vector<3xi64>
llvm.mlir.global external constant @__spirv_BuiltInGlobalSize() {addr_space = 1 : i32} : vector<3xi64>

llvm.func @use_value(i64)

// CHECK-LABEL:   llvm.func @test_spirv_BuiltInGlobalSize_calls() {
// CHECK-NEXT:      %0 = llvm.call @_Z20__spirv_GlobalSize_xv() : () -> i64
// CHECK-NEXT:      llvm.call @use_value(%0) : (i64) -> ()
// CHECK-NEXT:      %1 = llvm.call @_Z20__spirv_GlobalSize_yv() : () -> i64
// CHECK-NEXT:      llvm.call @use_value(%1) : (i64) -> ()
// CHECK-NEXT:      %2 = llvm.call @_Z20__spirv_GlobalSize_zv() : () -> i64
// CHECK-NEXT:      llvm.call @use_value(%2) : (i64) -> ()
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }
llvm.func @test_spirv_BuiltInGlobalSize_calls() {
  %0 = llvm.mlir.constant(0 : i32) : i32
  %1 = llvm.mlir.constant(1 : i32) : i32
  %2 = llvm.mlir.constant(2 : i32) : i32
  %3 = llvm.mlir.addressof @__spirv_BuiltInGlobalSize : !llvm.ptr<1>
  %4 = llvm.load %3 : !llvm.ptr<1> -> vector<3xi64>
  %5 = llvm.extractelement %4[%0 : i32] : vector<3xi64>
  llvm.call @use_value(%5) : (i64) -> ()
  %6 = llvm.mlir.addressof @__spirv_BuiltInGlobalSize : !llvm.ptr<1>
  %7 = llvm.load %6 : !llvm.ptr<1> -> vector<3xi64>
  %8 = llvm.extractelement %7[%1 : i32] : vector<3xi64>
  llvm.call @use_value(%8) : (i64) -> ()
  %9 = llvm.mlir.addressof @__spirv_BuiltInGlobalSize : !llvm.ptr<1>
  %10 = llvm.load %9 : !llvm.ptr<1> -> vector<3xi64>
  %11 = llvm.extractelement %10[%2 : i32] : vector<3xi64>
  llvm.call @use_value(%11) : (i64) -> ()
  llvm.return
}

// CHECK-LABEL:   llvm.func @test_spirv_BuiltInGlobalOffset_calls() {
// CHECK-NEXT:      %0 = llvm.call @_Z22__spirv_GlobalOffset_xv() : () -> i64
// CHECK-NEXT:      llvm.call @use_value(%0) : (i64) -> ()
// CHECK-NEXT:      %1 = llvm.call @_Z22__spirv_GlobalOffset_yv() : () -> i64
// CHECK-NEXT:      llvm.call @use_value(%1) : (i64) -> ()
// CHECK-NEXT:      %2 = llvm.call @_Z22__spirv_GlobalOffset_zv() : () -> i64
// CHECK-NEXT:      llvm.call @use_value(%2) : (i64) -> ()
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }
llvm.func @test_spirv_BuiltInGlobalOffset_calls() {
  %0 = llvm.mlir.constant(0 : i32) : i32
  %1 = llvm.mlir.constant(1 : i32) : i32
  %2 = llvm.mlir.constant(2 : i32) : i32
  %3 = llvm.mlir.addressof @__spirv_BuiltInGlobalOffset : !llvm.ptr<1>
  %4 = llvm.load %3 : !llvm.ptr<1> -> vector<3xi64>
  %5 = llvm.extractelement %4[%0 : i32] : vector<3xi64>
  llvm.call @use_value(%5) : (i64) -> ()
  %6 = llvm.mlir.addressof @__spirv_BuiltInGlobalOffset : !llvm.ptr<1>
  %7 = llvm.load %6 : !llvm.ptr<1> -> vector<3xi64>
  %8 = llvm.extractelement %7[%1 : i32] : vector<3xi64>
  llvm.call @use_value(%8) : (i64) -> ()
  %9 = llvm.mlir.addressof @__spirv_BuiltInGlobalOffset : !llvm.ptr<1>
  %10 = llvm.load %9 : !llvm.ptr<1> -> vector<3xi64>
  %11 = llvm.extractelement %10[%2 : i32] : vector<3xi64>
  llvm.call @use_value(%11) : (i64) -> ()
  llvm.return
}

// CHECK-LABEL:   llvm.func @test_spirv_BuiltInGlobalInvocationId_calls() {
// CHECK-NEXT:      %0 = llvm.call @_Z28__spirv_GlobalInvocationId_xv() : () -> i64
// CHECK-NEXT:      llvm.call @use_value(%0) : (i64) -> ()
// CHECK-NEXT:      %1 = llvm.call @_Z28__spirv_GlobalInvocationId_yv() : () -> i64
// CHECK-NEXT:      llvm.call @use_value(%1) : (i64) -> ()
// CHECK-NEXT:      %2 = llvm.call @_Z28__spirv_GlobalInvocationId_zv() : () -> i64
// CHECK-NEXT:      llvm.call @use_value(%2) : (i64) -> ()
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }
llvm.func @test_spirv_BuiltInGlobalInvocationId_calls() {
  %0 = llvm.mlir.constant(0 : i32) : i32
  %1 = llvm.mlir.constant(1 : i32) : i32
  %2 = llvm.mlir.constant(2 : i32) : i32
  %3 = llvm.mlir.addressof @__spirv_BuiltInGlobalInvocationId : !llvm.ptr<1>
  %4 = llvm.load %3 : !llvm.ptr<1> -> vector<3xi64>
  %5 = llvm.extractelement %4[%0 : i32] : vector<3xi64>
  llvm.call @use_value(%5) : (i64) -> ()
  %6 = llvm.mlir.addressof @__spirv_BuiltInGlobalInvocationId : !llvm.ptr<1>
  %7 = llvm.load %6 : !llvm.ptr<1> -> vector<3xi64>
  %8 = llvm.extractelement %7[%1 : i32] : vector<3xi64>
  llvm.call @use_value(%8) : (i64) -> ()
  %9 = llvm.mlir.addressof @__spirv_BuiltInGlobalInvocationId : !llvm.ptr<1>
  %10 = llvm.load %9 : !llvm.ptr<1> -> vector<3xi64>
  %11 = llvm.extractelement %10[%2 : i32] : vector<3xi64>
  llvm.call @use_value(%11) : (i64) -> ()
  llvm.return
}

// CHECK-LABEL:   llvm.func @test_spirv_BuiltInLocalInvocationId_calls() {
// CHECK-NEXT:      %0 = llvm.call @_Z27__spirv_LocalInvocationId_xv() : () -> i64
// CHECK-NEXT:      llvm.call @use_value(%0) : (i64) -> ()
// CHECK-NEXT:      %1 = llvm.call @_Z27__spirv_LocalInvocationId_yv() : () -> i64
// CHECK-NEXT:      llvm.call @use_value(%1) : (i64) -> ()
// CHECK-NEXT:      %2 = llvm.call @_Z27__spirv_LocalInvocationId_zv() : () -> i64
// CHECK-NEXT:      llvm.call @use_value(%2) : (i64) -> ()
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }
llvm.func @test_spirv_BuiltInLocalInvocationId_calls() {
  %0 = llvm.mlir.constant(0 : i32) : i32
  %1 = llvm.mlir.constant(1 : i32) : i32
  %2 = llvm.mlir.constant(2 : i32) : i32
  %3 = llvm.mlir.addressof @__spirv_BuiltInLocalInvocationId : !llvm.ptr<1>
  %4 = llvm.load %3 : !llvm.ptr<1> -> vector<3xi64>
  %5 = llvm.extractelement %4[%0 : i32] : vector<3xi64>
  llvm.call @use_value(%5) : (i64) -> ()
  %6 = llvm.mlir.addressof @__spirv_BuiltInLocalInvocationId : !llvm.ptr<1>
  %7 = llvm.load %6 : !llvm.ptr<1> -> vector<3xi64>
  %8 = llvm.extractelement %7[%1 : i32] : vector<3xi64>
  llvm.call @use_value(%8) : (i64) -> ()
  %9 = llvm.mlir.addressof @__spirv_BuiltInLocalInvocationId : !llvm.ptr<1>
  %10 = llvm.load %9 : !llvm.ptr<1> -> vector<3xi64>
  %11 = llvm.extractelement %10[%2 : i32] : vector<3xi64>
  llvm.call @use_value(%11) : (i64) -> ()
  llvm.return
}

// CHECK-LABEL:   llvm.func @test_spirv_BuiltInWorkgroupId_calls() {
// CHECK-NEXT:      %0 = llvm.call @_Z21__spirv_WorkgroupId_xv() : () -> i64
// CHECK-NEXT:      llvm.call @use_value(%0) : (i64) -> ()
// CHECK-NEXT:      %1 = llvm.call @_Z21__spirv_WorkgroupId_yv() : () -> i64
// CHECK-NEXT:      llvm.call @use_value(%1) : (i64) -> ()
// CHECK-NEXT:      %2 = llvm.call @_Z21__spirv_WorkgroupId_zv() : () -> i64
// CHECK-NEXT:      llvm.call @use_value(%2) : (i64) -> ()
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }
llvm.func @test_spirv_BuiltInWorkgroupId_calls() {
  %0 = llvm.mlir.constant(0 : i32) : i32
  %1 = llvm.mlir.constant(1 : i32) : i32
  %2 = llvm.mlir.constant(2 : i32) : i32
  %3 = llvm.mlir.addressof @__spirv_BuiltInWorkgroupId : !llvm.ptr<1>
  %4 = llvm.load %3 : !llvm.ptr<1> -> vector<3xi64>
  %5 = llvm.extractelement %4[%0 : i32] : vector<3xi64>
  llvm.call @use_value(%5) : (i64) -> ()
  %6 = llvm.mlir.addressof @__spirv_BuiltInWorkgroupId : !llvm.ptr<1>
  %7 = llvm.load %6 : !llvm.ptr<1> -> vector<3xi64>
  %8 = llvm.extractelement %7[%1 : i32] : vector<3xi64>
  llvm.call @use_value(%8) : (i64) -> ()
  %9 = llvm.mlir.addressof @__spirv_BuiltInWorkgroupId : !llvm.ptr<1>
  %10 = llvm.load %9 : !llvm.ptr<1> -> vector<3xi64>
  %11 = llvm.extractelement %10[%2 : i32] : vector<3xi64>
  llvm.call @use_value(%11) : (i64) -> ()
  llvm.return
}

// CHECK-LABEL:   llvm.func @test_spirv_BuiltInNumWorkgroups_calls() {
// CHECK-NEXT:      %0 = llvm.call @_Z23__spirv_NumWorkgroups_xv() : () -> i64
// CHECK-NEXT:      llvm.call @use_value(%0) : (i64) -> ()
// CHECK-NEXT:      %1 = llvm.call @_Z23__spirv_NumWorkgroups_yv() : () -> i64
// CHECK-NEXT:      llvm.call @use_value(%1) : (i64) -> ()
// CHECK-NEXT:      %2 = llvm.call @_Z23__spirv_NumWorkgroups_zv() : () -> i64
// CHECK-NEXT:      llvm.call @use_value(%2) : (i64) -> ()
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }
llvm.func @test_spirv_BuiltInNumWorkgroups_calls() {
  %0 = llvm.mlir.constant(0 : i32) : i32
  %1 = llvm.mlir.constant(1 : i32) : i32
  %2 = llvm.mlir.constant(2 : i32) : i32
  %3 = llvm.mlir.addressof @__spirv_BuiltInNumWorkgroups : !llvm.ptr<1>
  %4 = llvm.load %3 : !llvm.ptr<1> -> vector<3xi64>
  %5 = llvm.extractelement %4[%0 : i32] : vector<3xi64>
  llvm.call @use_value(%5) : (i64) -> ()
  %6 = llvm.mlir.addressof @__spirv_BuiltInNumWorkgroups : !llvm.ptr<1>
  %7 = llvm.load %6 : !llvm.ptr<1> -> vector<3xi64>
  %8 = llvm.extractelement %7[%1 : i32] : vector<3xi64>
  llvm.call @use_value(%8) : (i64) -> ()
  %9 = llvm.mlir.addressof @__spirv_BuiltInNumWorkgroups : !llvm.ptr<1>
  %10 = llvm.load %9 : !llvm.ptr<1> -> vector<3xi64>
  %11 = llvm.extractelement %10[%2 : i32] : vector<3xi64>
  llvm.call @use_value(%11) : (i64) -> ()
  llvm.return
}

// CHECK-LABEL:   llvm.func @test_spirv_BuiltInWorkgroupSize_calls() {
// CHECK-NEXT:      %0 = llvm.call @_Z23__spirv_WorkgroupSize_xv() : () -> i64
// CHECK-NEXT:      llvm.call @use_value(%0) : (i64) -> ()
// CHECK-NEXT:      %1 = llvm.call @_Z23__spirv_WorkgroupSize_yv() : () -> i64
// CHECK-NEXT:      llvm.call @use_value(%1) : (i64) -> ()
// CHECK-NEXT:      %2 = llvm.call @_Z23__spirv_WorkgroupSize_zv() : () -> i64
// CHECK-NEXT:      llvm.call @use_value(%2) : (i64) -> ()
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }
llvm.func @test_spirv_BuiltInWorkgroupSize_calls() {
  %0 = llvm.mlir.constant(0 : i32) : i32
  %1 = llvm.mlir.constant(1 : i32) : i32
  %2 = llvm.mlir.constant(2 : i32) : i32
  %3 = llvm.mlir.addressof @__spirv_BuiltInWorkgroupSize : !llvm.ptr<1>
  %4 = llvm.load %3 : !llvm.ptr<1> -> vector<3xi64>
  %5 = llvm.extractelement %4[%0 : i32] : vector<3xi64>
  llvm.call @use_value(%5) : (i64) -> ()
  %6 = llvm.mlir.addressof @__spirv_BuiltInWorkgroupSize : !llvm.ptr<1>
  %7 = llvm.load %6 : !llvm.ptr<1> -> vector<3xi64>
  %8 = llvm.extractelement %7[%1 : i32] : vector<3xi64>
  llvm.call @use_value(%8) : (i64) -> ()
  %9 = llvm.mlir.addressof @__spirv_BuiltInWorkgroupSize : !llvm.ptr<1>
  %10 = llvm.load %9 : !llvm.ptr<1> -> vector<3xi64>
  %11 = llvm.extractelement %10[%2 : i32] : vector<3xi64>
  llvm.call @use_value(%11) : (i64) -> ()
  llvm.return
}

// CHECK-LABEL:   llvm.func @_Z20__spirv_GlobalSize_xv() -> i64 {
// CHECK-NEXT:      %0 = llvm.call @llvm.nvvm.read.ptx.sreg.nctaid.x() : () -> i32
// CHECK-NEXT:      %1 = llvm.call @llvm.nvvm.read.ptx.sreg.ntid.x() : () -> i32
// CHECK-NEXT:      %2 = llvm.zext %0 : i32 to i64
// CHECK-NEXT:      %3 = llvm.zext %1 : i32 to i64
// CHECK-NEXT:      %4 = llvm.mul %2, %3  : i64
// CHECK-NEXT:      llvm.return %4 : i64
// CHECK-NEXT:    }
llvm.func @_Z20__spirv_GlobalSize_xv() -> i64 {}

// CHECK-LABEL:   llvm.func @_Z20__spirv_GlobalSize_yv() -> i64 {
// CHECK-NEXT:      %0 = llvm.call @llvm.nvvm.read.ptx.sreg.nctaid.y() : () -> i32
// CHECK-NEXT:      %1 = llvm.call @llvm.nvvm.read.ptx.sreg.ntid.y() : () -> i32
// CHECK-NEXT:      %2 = llvm.zext %0 : i32 to i64
// CHECK-NEXT:      %3 = llvm.zext %1 : i32 to i64
// CHECK-NEXT:      %4 = llvm.mul %2, %3  : i64
// CHECK-NEXT:      llvm.return %4 : i64
// CHECK-NEXT:    }
llvm.func @_Z20__spirv_GlobalSize_yv() -> i64 {}

// CHECK-LABEL:   llvm.func @_Z20__spirv_GlobalSize_zv() -> i64 {
// CHECK-NEXT:      %0 = llvm.call @llvm.nvvm.read.ptx.sreg.nctaid.z() : () -> i32
// CHECK-NEXT:      %1 = llvm.call @llvm.nvvm.read.ptx.sreg.ntid.z() : () -> i32
// CHECK-NEXT:      %2 = llvm.zext %0 : i32 to i64
// CHECK-NEXT:      %3 = llvm.zext %1 : i32 to i64
// CHECK-NEXT:      %4 = llvm.mul %2, %3  : i64
// CHECK-NEXT:      llvm.return %4 : i64
// CHECK-NEXT:    }
llvm.func @_Z20__spirv_GlobalSize_zv() -> i64 {}

// CHECK-LABEL:   llvm.func @_Z22__spirv_GlobalOffset_xv() -> i64 {
// CHECK-NEXT:      %0 = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:      llvm.return %0 : i64
// CHECK-NEXT:    }
llvm.func @_Z22__spirv_GlobalOffset_xv() -> i64 {}

// CHECK-LABEL:   llvm.func @_Z22__spirv_GlobalOffset_yv() -> i64 {
// CHECK-NEXT:      %0 = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:      llvm.return %0 : i64
// CHECK-NEXT:    }
llvm.func @_Z22__spirv_GlobalOffset_yv() -> i64 {}

// CHECK-LABEL:   llvm.func @_Z22__spirv_GlobalOffset_zv() -> i64 {
// CHECK-NEXT:      %0 = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:      llvm.return %0 : i64
// CHECK-NEXT:    }
llvm.func @_Z22__spirv_GlobalOffset_zv() -> i64 {}

// CHECK-LABEL:   llvm.func @_Z28__spirv_GlobalInvocationId_xv() -> i64 {
// CHECK-NEXT:      %0 = llvm.call @llvm.nvvm.read.ptx.sreg.tid.x() : () -> i32
// CHECK-NEXT:      %1 = llvm.call @llvm.nvvm.read.ptx.sreg.ntid.x() : () -> i32
// CHECK-NEXT:      %2 = llvm.call @llvm.nvvm.read.ptx.sreg.ctaid.x() : () -> i32
// CHECK-NEXT:      %3 = llvm.zext %0 : i32 to i64
// CHECK-NEXT:      %4 = llvm.zext %1 : i32 to i64
// CHECK-NEXT:      %5 = llvm.zext %2 : i32 to i64
// CHECK-NEXT:      %6 = llvm.mul %5, %4  : i64
// CHECK-NEXT:      %7 = llvm.add %6, %3  : i64
// CHECK-NEXT:      llvm.return %7 : i64
// CHECK-NEXT:    }
llvm.func @_Z28__spirv_GlobalInvocationId_xv() -> i64 {}

// CHECK-LABEL:   llvm.func @_Z28__spirv_GlobalInvocationId_yv() -> i64 {
// CHECK-NEXT:      %0 = llvm.call @llvm.nvvm.read.ptx.sreg.tid.y() : () -> i32
// CHECK-NEXT:      %1 = llvm.call @llvm.nvvm.read.ptx.sreg.ntid.y() : () -> i32
// CHECK-NEXT:      %2 = llvm.call @llvm.nvvm.read.ptx.sreg.ctaid.y() : () -> i32
// CHECK-NEXT:      %3 = llvm.zext %0 : i32 to i64
// CHECK-NEXT:      %4 = llvm.zext %1 : i32 to i64
// CHECK-NEXT:      %5 = llvm.zext %2 : i32 to i64
// CHECK-NEXT:      %6 = llvm.mul %5, %4  : i64
// CHECK-NEXT:      %7 = llvm.add %6, %3  : i64
// CHECK-NEXT:      llvm.return %7 : i64
// CHECK-NEXT:    }
llvm.func @_Z28__spirv_GlobalInvocationId_yv() -> i64 {}

// CHECK-LABEL:   llvm.func @_Z28__spirv_GlobalInvocationId_zv() -> i64 {
// CHECK-NEXT:      %0 = llvm.call @llvm.nvvm.read.ptx.sreg.tid.z() : () -> i32
// CHECK-NEXT:      %1 = llvm.call @llvm.nvvm.read.ptx.sreg.ntid.z() : () -> i32
// CHECK-NEXT:      %2 = llvm.call @llvm.nvvm.read.ptx.sreg.ctaid.z() : () -> i32
// CHECK-NEXT:      %3 = llvm.zext %0 : i32 to i64
// CHECK-NEXT:      %4 = llvm.zext %1 : i32 to i64
// CHECK-NEXT:      %5 = llvm.zext %2 : i32 to i64
// CHECK-NEXT:      %6 = llvm.mul %5, %4  : i64
// CHECK-NEXT:      %7 = llvm.add %6, %3  : i64
// CHECK-NEXT:      llvm.return %7 : i64
// CHECK-NEXT:    }
llvm.func @_Z28__spirv_GlobalInvocationId_zv() -> i64 {}

// CHECK-LABEL:   llvm.func @_Z27__spirv_LocalInvocationId_xv() -> i64 {
// CHECK-NEXT:      %0 = llvm.call @llvm.nvvm.read.ptx.sreg.tid.x() : () -> i32
// CHECK-NEXT:      %1 = llvm.zext %0 : i32 to i64
// CHECK-NEXT:      llvm.return %1 : i64
// CHECK-NEXT:    }
llvm.func @_Z27__spirv_LocalInvocationId_xv() -> i64 {}

// CHECK-LABEL:   llvm.func @_Z27__spirv_LocalInvocationId_yv() -> i64 {
// CHECK-NEXT:      %0 = llvm.call @llvm.nvvm.read.ptx.sreg.tid.y() : () -> i32
// CHECK-NEXT:      %1 = llvm.zext %0 : i32 to i64
// CHECK-NEXT:      llvm.return %1 : i64
// CHECK-NEXT:    }
llvm.func @_Z27__spirv_LocalInvocationId_yv() -> i64 {}

// CHECK-LABEL:   llvm.func @_Z27__spirv_LocalInvocationId_zv() -> i64 {
// CHECK-NEXT:      %0 = llvm.call @llvm.nvvm.read.ptx.sreg.tid.z() : () -> i32
// CHECK-NEXT:      %1 = llvm.zext %0 : i32 to i64
// CHECK-NEXT:      llvm.return %1 : i64
// CHECK-NEXT:    }
llvm.func @_Z27__spirv_LocalInvocationId_zv() -> i64 {}

// CHECK-LABEL:   llvm.func @_Z21__spirv_WorkgroupId_xv() -> i64 {
// CHECK-NEXT:      %0 = llvm.call @llvm.nvvm.read.ptx.sreg.ctaid.x() : () -> i32
// CHECK-NEXT:      %1 = llvm.zext %0 : i32 to i64
// CHECK-NEXT:      llvm.return %1 : i64
// CHECK-NEXT:    }
llvm.func @_Z21__spirv_WorkgroupId_xv() -> i64 {}

// CHECK-LABEL:   llvm.func @_Z21__spirv_WorkgroupId_yv() -> i64 {
// CHECK-NEXT:      %0 = llvm.call @llvm.nvvm.read.ptx.sreg.ctaid.y() : () -> i32
// CHECK-NEXT:      %1 = llvm.zext %0 : i32 to i64
// CHECK-NEXT:      llvm.return %1 : i64
// CHECK-NEXT:    }
llvm.func @_Z21__spirv_WorkgroupId_yv() -> i64 {}

// CHECK-LABEL:   llvm.func @_Z21__spirv_WorkgroupId_zv() -> i64 {
// CHECK-NEXT:      %0 = llvm.call @llvm.nvvm.read.ptx.sreg.ctaid.z() : () -> i32
// CHECK-NEXT:      %1 = llvm.zext %0 : i32 to i64
// CHECK-NEXT:      llvm.return %1 : i64
// CHECK-NEXT:    }
llvm.func @_Z21__spirv_WorkgroupId_zv() -> i64 {}

// CHECK-LABEL:   llvm.func @_Z23__spirv_NumWorkgroups_xv() -> i64 {
// CHECK-NEXT:      %0 = llvm.call @llvm.nvvm.read.ptx.sreg.nctaid.x() : () -> i32
// CHECK-NEXT:      %1 = llvm.zext %0 : i32 to i64
// CHECK-NEXT:      llvm.return %1 : i64
// CHECK-NEXT:    }
llvm.func @_Z23__spirv_NumWorkgroups_xv() -> i64 {}

// CHECK-LABEL:   llvm.func @_Z23__spirv_NumWorkgroups_yv() -> i64 {
// CHECK-NEXT:      %0 = llvm.call @llvm.nvvm.read.ptx.sreg.nctaid.y() : () -> i32
// CHECK-NEXT:      %1 = llvm.zext %0 : i32 to i64
// CHECK-NEXT:      llvm.return %1 : i64
// CHECK-NEXT:    }
llvm.func @_Z23__spirv_NumWorkgroups_yv() -> i64 {}

// CHECK-LABEL:   llvm.func @_Z23__spirv_NumWorkgroups_zv() -> i64 {
// CHECK-NEXT:      %0 = llvm.call @llvm.nvvm.read.ptx.sreg.nctaid.z() : () -> i32
// CHECK-NEXT:      %1 = llvm.zext %0 : i32 to i64
// CHECK-NEXT:      llvm.return %1 : i64
// CHECK-NEXT:    }
llvm.func @_Z23__spirv_NumWorkgroups_zv() -> i64 {}

// CHECK-LABEL:   llvm.func @_Z23__spirv_WorkgroupSize_xv() -> i64 {
// CHECK-NEXT:      %0 = llvm.call @llvm.nvvm.read.ptx.sreg.ntid.x() : () -> i32
// CHECK-NEXT:      %1 = llvm.zext %0 : i32 to i64
// CHECK-NEXT:      llvm.return %1 : i64
// CHECK-NEXT:    }
llvm.func @_Z23__spirv_WorkgroupSize_xv() -> i64 {}

// CHECK-LABEL:   llvm.func @_Z23__spirv_WorkgroupSize_yv() -> i64 {
// CHECK-NEXT:      %0 = llvm.call @llvm.nvvm.read.ptx.sreg.ntid.y() : () -> i32
// CHECK-NEXT:      %1 = llvm.zext %0 : i32 to i64
// CHECK-NEXT:      llvm.return %1 : i64
// CHECK-NEXT:    }
llvm.func @_Z23__spirv_WorkgroupSize_yv() -> i64 {}

// CHECK-LABEL:   llvm.func @_Z23__spirv_WorkgroupSize_zv() -> i64 {
// CHECK-NEXT:      %0 = llvm.call @llvm.nvvm.read.ptx.sreg.ntid.z() : () -> i32
// CHECK-NEXT:      %1 = llvm.zext %0 : i32 to i64
// CHECK-NEXT:      llvm.return %1 : i64
// CHECK-NEXT:    }
llvm.func @_Z23__spirv_WorkgroupSize_zv() -> i64 {}
