// RUN: polygeist-opt %s --convert-polygeist-to-llvm --split-input-file --verify-diagnostics 2>&1

// expected-error @below {{'func.func' op could break ABI when converting to LLVM}}
func.func @bare_abi_break(%arg: memref<?x?xf32>) {
  return
}

// -----

// expected-error @below {{'func.func' op could break ABI when converting to LLVM}}
func.func @bare_abi_break(%arg: memref<?x?xf32>) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
  return
}
