// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

llvm.func @__gxx_personality_v0(...) -> i32

// CHECK: declare fastcc void @cconv_fastcc()
// CHECK: declare        void @cconv_ccc()
// CHECK: declare tailcc void @cconv_tailcc()
// CHECK: declare ghccc  void @cconv_ghccc()
llvm.func fastcc @cconv_fastcc()
llvm.func ccc    @cconv_ccc()
llvm.func tailcc @cconv_tailcc()
llvm.func cc_10  @cconv_ghccc()

// CHECK-LABEL: @test_ccs
llvm.func @test_ccs() {
  // CHECK-NEXT: call fastcc void @cconv_fastcc()
  // CHECK-NEXT: call        void @cconv_ccc()
  // CHECK-NEXT: call        void @cconv_ccc()
  // CHECK-NEXT: call tailcc void @cconv_tailcc()
  // CHECK-NEXT: call ghccc  void @cconv_ghccc()
  // CHECK-NEXT: ret void
  llvm.call fastcc @cconv_fastcc() : () -> ()
  llvm.call ccc    @cconv_ccc()    : () -> ()
  llvm.call        @cconv_ccc()    : () -> ()
  llvm.call tailcc @cconv_tailcc() : () -> ()
  llvm.call cc_10  @cconv_ghccc()  : () -> ()
  llvm.return
}
