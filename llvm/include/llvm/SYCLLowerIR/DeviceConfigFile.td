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
def AspectExt_intel_legacy_image : Aspect<"ext_intel_legacy_image">;
def AspectExt_oneapi_bindless_images : Aspect<"ext_oneapi_bindless_images">;
def AspectExt_oneapi_bindless_images_shared_usm : Aspect<"ext_oneapi_bindless_images_shared_usm">;
def AspectExt_oneapi_bindless_images_1d_usm : Aspect<"ext_oneapi_bindless_images_1d_usm">;
def AspectExt_oneapi_bindless_images_2d_usm : Aspect<"ext_oneapi_bindless_images_2d_usm">;
def AspectExt_oneapi_interop_memory_import : Aspect<"ext_oneapi_interop_memory_import">;
def AspectExt_oneapi_interop_memory_export : Aspect<"ext_oneapi_interop_memory_export">;
def AspectExt_oneapi_interop_semaphore_import : Aspect<"ext_oneapi_interop_semaphore_import">;
def AspectExt_oneapi_interop_semaphore_export : Aspect<"ext_oneapi_interop_semaphore_export">;
def AspectExt_oneapi_mipmap : Aspect<"ext_oneapi_mipmap">;
def AspectExt_oneapi_mipmap_anisotropy : Aspect<"ext_oneapi_mipmap_anisotropy">;
def AspectExt_oneapi_mipmap_level_reference : Aspect<"ext_oneapi_mipmap_level_reference">;
def AspectExt_oneapi_bindless_sampled_image_fetch_1d_usm : Aspect<"ext_oneapi_bindless_sampled_image_fetch_1d_usm">;
def AspectExt_oneapi_bindless_sampled_image_fetch_1d : Aspect<"ext_oneapi_bindless_sampled_image_fetch_1d">;
def AspectExt_oneapi_bindless_sampled_image_fetch_2d_usm : Aspect<"ext_oneapi_bindless_sampled_image_fetch_2d_usm">;
def AspectExt_oneapi_bindless_sampled_image_fetch_2d : Aspect<"ext_oneapi_bindless_sampled_image_fetch_2d">;
def AspectExt_oneapi_bindless_sampled_image_fetch_3d_usm : Aspect<"ext_oneapi_bindless_sampled_image_fetch_3d_usm">;
def AspectExt_oneapi_bindless_sampled_image_fetch_3d : Aspect<"ext_oneapi_bindless_sampled_image_fetch_3d">;
def AspectExt_intel_esimd : Aspect<"ext_intel_esimd">;
def AspectExt_oneapi_ballot_group : Aspect<"ext_oneapi_ballot_group">;
def AspectExt_oneapi_fixed_size_group : Aspect<"ext_oneapi_fixed_size_group">;
def AspectExt_oneapi_opportunistic_group : Aspect<"ext_oneapi_opportunistic_group">;
def AspectExt_oneapi_tangle_group : Aspect<"ext_oneapi_tangle_group">;
def AspectExt_intel_matrix : Aspect<"ext_intel_matrix">;
def AspectExt_oneapi_is_composite : Aspect<"ext_oneapi_is_composite">;
def AspectExt_oneapi_is_component : Aspect<"ext_oneapi_is_component">;
// Deprecated aspects
def AspectInt64_base_atomics : Aspect<"int64_base_atomics">;
def AspectInt64_extended_atomics : Aspect<"int64_extended_atomics">;
def AspectUsm_system_allocator : Aspect<"usm_system_allocator">;
def AspectUsm_restricted_shared_allocations : Aspect<"usm_restricted_shared_allocations">;
def AspectHost : Aspect<"host">;
defvar AllUSMAspects = [AspectUsm_device_allocations, AspectUsm_host_allocations,
    AspectUsm_shared_allocations, AspectUsm_system_allocations, AspectUsm_atomic_host_allocations,
    AspectUsm_atomic_shared_allocations];


def TargetTable : DynamicTable {
  let FilterClass = "TargetInfo";
  let Fields = ["DeviceId", "maySupportOtherAspects", "aspects",
                "subGroupSizes", "aotToolchain", "aotToolchainOptions"];
  string TypeOf_aspects = "list<Aspect>";
  string TypeOf_subGroupSizes = "list<int>";
}

// TargetInfo is a cpp struct defined in DeviceConfigFile.hpp. Both the fields
// and the name must match those in DeviceConfigFile.hpp. Thus, any
// modification to this definition in this file must be mirrored in
// DeviceConfigFile.hpp.
class TargetInfo<string targetName, list<Aspect> aspectList, list<int> subGroupSizesList, string toolchain = "", string options = "", bit maySupportOtherAspects_init = 0> {
  string DeviceId = targetName;
  bits<1> maySupportOtherAspects = maySupportOtherAspects_init;
  list<Aspect> aspects = aspectList;
  list<int> subGroupSizes = subGroupSizesList;
  string aotToolchain = toolchain;
  string aotToolchainOptions = options;
}

// This definition serves the only purpose of testing whether the aspect list defined in here and in SYCL RT match.
def : TargetInfo<"__TestAspectList",
    [AspectCpu, AspectGpu, AspectAccelerator, AspectCustom, AspectFp16, AspectFp64, AspectImage, AspectOnline_compiler, 
    AspectOnline_linker, AspectQueue_profiling, AspectUsm_device_allocations, AspectUsm_host_allocations,
    AspectUsm_shared_allocations, AspectUsm_system_allocations, AspectExt_intel_pci_address,
    AspectExt_intel_gpu_eu_count, AspectExt_intel_gpu_eu_simd_width, AspectExt_intel_gpu_slices,
    AspectExt_intel_gpu_subslices_per_slice, AspectExt_intel_gpu_eu_count_per_subslice,
    AspectExt_intel_max_mem_bandwidth, AspectExt_intel_mem_channel, AspectUsm_atomic_host_allocations,
    AspectUsm_atomic_shared_allocations, AspectAtomic64, AspectExt_intel_device_info_uuid, AspectExt_oneapi_srgb,
    AspectExt_oneapi_native_assert, AspectHost_debuggable, AspectExt_intel_gpu_hw_threads_per_eu,
    AspectExt_oneapi_cuda_async_barrier, AspectExt_oneapi_bfloat16_math_functions, AspectExt_intel_free_memory,
    AspectExt_intel_device_id, AspectExt_intel_memory_clock_rate, AspectExt_intel_memory_bus_width, AspectEmulated,
    AspectExt_intel_legacy_image, AspectExt_oneapi_bindless_images,
    AspectExt_oneapi_bindless_images_shared_usm, AspectExt_oneapi_bindless_images_1d_usm, AspectExt_oneapi_bindless_images_2d_usm,
    AspectExt_oneapi_interop_memory_import, AspectExt_oneapi_interop_memory_export,
    AspectExt_oneapi_interop_semaphore_import, AspectExt_oneapi_interop_semaphore_export,
    AspectExt_oneapi_mipmap, AspectExt_oneapi_mipmap_anisotropy, AspectExt_oneapi_mipmap_level_reference,
    AspectExt_oneapi_bindless_sampled_image_fetch_1d_usm, AspectExt_oneapi_bindless_sampled_image_fetch_1d,
    AspectExt_oneapi_bindless_sampled_image_fetch_2d_usm, AspectExt_oneapi_bindless_sampled_image_fetch_2d,
    AspectExt_oneapi_bindless_sampled_image_fetch_3d_usm, AspectExt_oneapi_bindless_sampled_image_fetch_3d,
    AspectExt_intel_esimd, AspectExt_oneapi_ballot_group, AspectExt_oneapi_fixed_size_group, 
    AspectExt_oneapi_opportunistic_group, AspectExt_oneapi_tangle_group, AspectExt_intel_matrix, 
    AspectExt_oneapi_is_composite, AspectExt_oneapi_is_component],
    []>;
// This definition serves the only purpose of testing whether the deprecated aspect list defined in here and in SYCL RT
// match.
def : TargetInfo<"__TestDeprecatedAspectList",
    [AspectInt64_base_atomics, AspectInt64_extended_atomics, AspectUsm_system_allocator,
    AspectUsm_restricted_shared_allocations, AspectHost],
    []>;

def : TargetInfo<"spir64", [], [], "", "", 1>;
def : TargetInfo<"spir64_gen", [], [], "", "", 1>;
def : TargetInfo<"spir64_x86_64", [], [], "", "", 1>;
def : TargetInfo<"spir64_fpga", [], [], "", "", 1>;
def : TargetInfo<"x86_64", [], [], "", "", 1>;
// Examples of how to use a combination of explicitly specified values + predefined lists
//defvar AspectList = [AspectCpu] # AllUSMAspects;
//def : TargetInfo<"Test", AspectList, []>;
//def : TargetInfo<"Test2", [AspectCpu] # AllUSMAspects, []>;
