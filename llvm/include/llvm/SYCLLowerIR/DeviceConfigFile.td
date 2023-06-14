//===-DeviceConfigFile.td - Device Config File for SYCL ---*--- tablegen -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/TableGen/DynamicTable.td"

class Aspect<string name> {
    string Name = name;
}

def AspectCpu : Aspect<"cpu">;
def AspectGpu : Aspect<"gpu">;
def AspectAccelerator : Aspect<"accelerator">;
def AspectCustom : Aspect<"custom">;
def AspectFp16 : Aspect<"fp16">;
def AspectFp64 : Aspect<"fp64">;
def AspectImage : Aspect<"image">;
def AspectOnline_compiler : Aspect<"online_compiler">;
def AspectOnline_linker : Aspect<"online_linker">;
def AspectQueue_profiling : Aspect<"queue_profiling">;
def AspectUsm_device_allocations : Aspect<"usm_device_allocations">;
def AspectUsm_host_allocations : Aspect<"usm_host_allocations">;
def AspectUsm_shared_allocations : Aspect<"usm_shared_allocations">;
def AspectUsm_system_allocations : Aspect<"usm_system_allocations">;
def AspectExt_intel_pci_address : Aspect<"ext_intel_pci_address">;
def AspectExt_intel_gpu_eu_count : Aspect<"ext_intel_gpu_eu_count">;
def AspectExt_intel_gpu_eu_simd_width : Aspect<"ext_intel_gpu_eu_simd_width">;
def AspectExt_intel_gpu_slices : Aspect<"ext_intel_gpu_slices">;
def AspectExt_intel_gpu_subslices_per_slice : Aspect<"ext_intel_gpu_subslices_per_slice">;
def AspectExt_intel_gpu_eu_count_per_subslice : Aspect<"ext_intel_gpu_eu_count_per_subslice">;
def AspectExt_intel_max_mem_bandwidth : Aspect<"ext_intel_max_mem_bandwidth">;
def AspectExt_intel_mem_channel : Aspect<"ext_intel_mem_channel">;
def AspectUsm_atomic_host_allocations : Aspect<"usm_atomic_host_allocations">;
def AspectUsm_atomic_shared_allocations : Aspect<"usm_atomic_shared_allocations">;
def AspectAtomic64 : Aspect<"atomic64">;
def AspectExt_intel_device_info_uuid : Aspect<"ext_intel_device_info_uuid">;
def AspectExt_oneapi_srgb : Aspect<"ext_oneapi_srgb">;
def AspectExt_oneapi_native_assert : Aspect<"ext_oneapi_native_assert">;
def AspectHost_debuggable : Aspect<"host_debuggable">;
def AspectExt_intel_gpu_hw_threads_per_eu : Aspect<"ext_intel_gpu_hw_threads_per_eu">;
def AspectExt_oneapi_cuda_async_barrier : Aspect<"ext_oneapi_cuda_async_barrier">;
def AspectExt_oneapi_bfloat16_math_functions : Aspect<"ext_oneapi_bfloat16_math_functions">;
def AspectExt_intel_free_memory : Aspect<"ext_intel_free_memory">;
def AspectExt_intel_device_id : Aspect<"ext_intel_device_id">;
def AspectExt_intel_memory_clock_rate : Aspect<"ext_intel_memory_clock_rate">;
def AspectExt_intel_memory_bus_width : Aspect<"ext_intel_memory_bus_width">;
def AspectEmulated : Aspect<"emulated">;


def TargetTable : DynamicTable {
  let FilterClass = "TargetInfo";
  let Fields = ["DeviceId", "maySupportOtherAspects", "aspects",
                "subGroupSizes", "aotToolchain", "aotToolchainOptions"];
  string TypeOf_aspects = "list<Aspect>";
  string TypeOf_subGroupSizes = "list<int>";
}

class TargetInfo<string str, bit val1, list<Aspect> aspectList, list<int> subGroupSizesList, string toolchain, string options> {
  string DeviceId = str;
  bits<1> maySupportOtherAspects = val1;
  list<Aspect> aspects = aspectList;
  list<int> subGroupSizes = subGroupSizesList;
  string aotToolchain = toolchain;
  string aotToolchainOptions = options;
}

def : TargetInfo<"TargetA", 1, [AspectCpu, AspectAtomic64], [8, 16], "ocloc", "devA">;
def : TargetInfo<"TargetB", 0, [AspectGpu, AspectFp16], [8, 16], "ocloc", "devB">;
def : TargetInfo<"TargetC", 0, [AspectEmulated, AspectImage], [8, 16], "ocloc", "devC">;
def : TargetInfo<"TargetD", 1, [AspectCpu, AspectFp64, AspectHost_debuggable], [8, 16], "ocloc", "devD">;
