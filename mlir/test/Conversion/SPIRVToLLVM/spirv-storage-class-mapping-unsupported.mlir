// RUN: mlir-opt -convert-spirv-to-llvm='client-api-for-address-space-mapping=Metal' -verify-diagnostics %s
// RUN: mlir-opt -convert-spirv-to-llvm='client-api-for-address-space-mapping=Vulkan' -verify-diagnostics %s
// RUN: mlir-opt -convert-spirv-to-llvm='client-api-for-address-space-mapping=WebGPU' -verify-diagnostics %s

module {}  // expected-warning-re {{address space mapping for client {{.*}} not implemented}}
