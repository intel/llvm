// RUN: rm -rf %t.spvdir && mkdir %t.spvdir

// Range rounding disabled so each lambda kernel yields a single entry point.
// -fsycl-device-code-split=off with -fno-sycl-device-code-split-esimd keeps all
// kernels (including ESIMD) in one dumped SPIR-V module, so a single .spv is
// produced and can be renamed to a known name and converted to text.
// RUN: %clangxx -fsycl -fsycl-range-rounding=disable -fsycl-device-code-split=off -fno-sycl-device-code-split-esimd -o %t.out -save-offload-code=%t.spvdir %s
// RUN: mv %t.spvdir/*.spv %t.spvdir/dump.spv
// RUN: llvm-spirv -to-text %t.spvdir/dump.spv

// RUN: FileCheck %s < %t.spvdir/dump.spt

// CHECK-DAG: EntryPoint {{[0-9]+}} [[#SyclFreeFunctionSpecifiedId:]] "{{.*}}sycl_kernel_free_function_kernel_specified{{.*}}"
// CHECK-DAG: ExecutionMode [[#SyclFreeFunctionSpecifiedId]] [[MAXIMUM_REGISTERS_INTEL:6461]] 256
// CHECK-DAG: EntryPoint {{[0-9]+}} [[#SyclFreeFunctionAutomaticId:]] "{{.*}}sycl_kernel_free_function_kernel_automatic{{.*}}"
// CHECK-DAG: ExecutionMode [[#SyclFreeFunctionAutomaticId]] [[NAMED_MAXIMUM_REGISTERS_INTEL:6463]] [[AUTO_INTEL:0]]
// CHECK-DAG: EntryPoint {{[0-9]+}} [[#EsimdFreeFunctionSpecifiedId:]] "{{.*}}sycl_kernel_esimd_free_function_kernel_specified{{.*}}"
// CHECK-DAG: ExecutionMode [[#EsimdFreeFunctionSpecifiedId]] [[MAXIMUM_REGISTERS_INTEL]] 256
// CHECK-DAG: EntryPoint {{[0-9]+}} [[#EsimdFreeFunctionAutomaticId:]] "{{.*}}sycl_kernel_esimd_free_function_kernel_automatic{{.*}}"
// CHECK-DAG: ExecutionMode [[#EsimdFreeFunctionAutomaticId]] [[NAMED_MAXIMUM_REGISTERS_INTEL]] [[AUTO_INTEL]]
// CHECK-DAG: EntryPoint {{[0-9]+}} [[#SyclLambdaSpecifiedId:]] "{{.*}}runLambdaSYCL{{.*}}maximum_registers_key{{.*}}"
// CHECK-DAG: ExecutionMode [[#SyclLambdaSpecifiedId]] [[MAXIMUM_REGISTERS_INTEL]] 256
// CHECK-DAG: EntryPoint {{[0-9]+}} [[#SyclLambdaAutomaticId:]] "{{.*}}runLambdaSYCL{{.*}}maximum_registers_automatic_key{{.*}}"
// CHECK-DAG: ExecutionMode [[#SyclLambdaAutomaticId]] [[NAMED_MAXIMUM_REGISTERS_INTEL]] [[AUTO_INTEL]]
// CHECK-DAG: EntryPoint {{[0-9]+}} [[#EsimdLambdaSpecifiedId:]] "{{.*}}runLambdaESIMD{{.*}}maximum_registers_key{{.*}}"
// CHECK-DAG: ExecutionMode [[#EsimdLambdaSpecifiedId]] [[MAXIMUM_REGISTERS_INTEL]] 256
// CHECK-DAG: EntryPoint {{[0-9]+}} [[#EsimdLambdaAutomaticId:]] "{{.*}}runLambdaESIMD{{.*}}maximum_registers_automatic_key{{.*}}"
// CHECK-DAG: ExecutionMode [[#EsimdLambdaAutomaticId]] [[NAMED_MAXIMUM_REGISTERS_INTEL]] [[AUTO_INTEL]]

#include "../../../test-e2e/Properties/maximum_registers.cpp"
