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
def AspectExt_oneapi_cubemap : Aspect<"ext_oneapi_cubemap">;
def AspectExt_oneapi_cubemap_seamless_filtering : Aspect<"ext_oneapi_cubemap_seamless_filtering">;
def AspectExt_intel_esimd : Aspect<"ext_intel_esimd">;
def AspectExt_oneapi_ballot_group : Aspect<"ext_oneapi_ballot_group">;
def AspectExt_oneapi_fixed_size_group : Aspect<"ext_oneapi_fixed_size_group">;
def AspectExt_oneapi_opportunistic_group : Aspect<"ext_oneapi_opportunistic_group">;
def AspectExt_oneapi_tangle_group : Aspect<"ext_oneapi_tangle_group">;
def AspectExt_intel_matrix : Aspect<"ext_intel_matrix">;
def AspectExt_oneapi_is_composite : Aspect<"ext_oneapi_is_composite">;
def AspectExt_oneapi_is_component : Aspect<"ext_oneapi_is_component">;
def AspectExt_oneapi_graph : Aspect<"ext_oneapi_graph">;
def AspectExt_oneapi_limited_graph : Aspect<"ext_oneapi_limited_graph">;
def AspectExt_oneapi_private_alloca : Aspect<"ext_oneapi_private_alloca">;
def AspectExt_oneapi_queue_profiling_tag : Aspect<"ext_oneapi_queue_profiling_tag">;
def AspectExt_oneapi_virtual_mem : Aspect<"ext_oneapi_virtual_mem">;
def AspectExt_oneapi_cuda_cluster_group : Aspect<"ext_oneapi_cuda_cluster_group">;
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
    AspectExt_oneapi_cuda_async_barrier, AspectExt_intel_free_memory,
    AspectExt_intel_device_id, AspectExt_intel_memory_clock_rate, AspectExt_intel_memory_bus_width, AspectEmulated,
    AspectExt_intel_legacy_image, AspectExt_oneapi_bindless_images,
    AspectExt_oneapi_bindless_images_shared_usm, AspectExt_oneapi_bindless_images_1d_usm, AspectExt_oneapi_bindless_images_2d_usm,
    AspectExt_oneapi_interop_memory_import, AspectExt_oneapi_interop_memory_export,
    AspectExt_oneapi_interop_semaphore_import, AspectExt_oneapi_interop_semaphore_export,
    AspectExt_oneapi_mipmap, AspectExt_oneapi_mipmap_anisotropy, AspectExt_oneapi_mipmap_level_reference, AspectExt_oneapi_cubemap, 
    AspectExt_oneapi_cubemap_seamless_filtering, 
    AspectExt_oneapi_bindless_sampled_image_fetch_1d_usm, AspectExt_oneapi_bindless_sampled_image_fetch_1d,
    AspectExt_oneapi_bindless_sampled_image_fetch_2d_usm, AspectExt_oneapi_bindless_sampled_image_fetch_2d,
    AspectExt_oneapi_bindless_sampled_image_fetch_3d_usm, AspectExt_oneapi_bindless_sampled_image_fetch_3d,
    AspectExt_intel_esimd,
    AspectExt_oneapi_ballot_group, AspectExt_oneapi_fixed_size_group, AspectExt_oneapi_opportunistic_group,
    AspectExt_oneapi_tangle_group, AspectExt_intel_matrix, AspectExt_oneapi_is_composite, AspectExt_oneapi_is_component, 
    AspectExt_oneapi_graph, AspectExt_oneapi_limited_graph,
    AspectExt_oneapi_private_alloca, AspectExt_oneapi_queue_profiling_tag, AspectExt_oneapi_virtual_mem, AspectExt_oneapi_cuda_cluster_group],
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

// TODO: The aspects listed for the intel_gpu targets right now are incomplete; 
// only the fp16/fp64/atomic64 aspects are listed.
defvar Fp16Fp64Atomic64 = [AspectFp16, AspectFp64, AspectAtomic64];
defvar Fp16Atomic64 = [AspectFp16, AspectAtomic64];
defvar Sg8_16_32 = [8, 16, 32];
defvar Sg16_32 = [16, 32];
defvar IntelBaseAspects = [AspectExt_intel_esimd];
class IntelTargetInfo<string Name, list<Aspect> Aspects, list<int> subGroupSizesList>
: TargetInfo<Name, IntelBaseAspects # Aspects, subGroupSizesList>;
// Note: only the "canonical" target names are listed here - see
// SYCL::gen::resolveGenDevice().
def : IntelTargetInfo<"intel_gpu_arl_h", Fp16Fp64Atomic64, Sg8_16_32>;
def : IntelTargetInfo<"intel_gpu_mtl_h", Fp16Fp64Atomic64, Sg8_16_32>;
def : IntelTargetInfo<"intel_gpu_mtl_u", Fp16Fp64Atomic64, Sg8_16_32>;
def : IntelTargetInfo<"intel_gpu_pvc_vg", Fp16Fp64Atomic64, Sg16_32>;
def : IntelTargetInfo<"intel_gpu_pvc", Fp16Fp64Atomic64, Sg16_32>;
def : IntelTargetInfo<"intel_gpu_acm_g12", Fp16Atomic64, Sg8_16_32>;
def : IntelTargetInfo<"intel_gpu_acm_g11", Fp16Atomic64, Sg8_16_32>;
def : IntelTargetInfo<"intel_gpu_acm_g10", Fp16Atomic64, Sg8_16_32>;
def : IntelTargetInfo<"intel_gpu_dg1", Fp16Atomic64, Sg8_16_32>;
def : IntelTargetInfo<"intel_gpu_adl_n", Fp16Atomic64, Sg8_16_32>;
def : IntelTargetInfo<"intel_gpu_adl_p", Fp16Atomic64, Sg8_16_32>;
def : IntelTargetInfo<"intel_gpu_adl_s", Fp16Atomic64, Sg8_16_32>;
def : IntelTargetInfo<"intel_gpu_rkl", Fp16Atomic64, Sg8_16_32>;
def : IntelTargetInfo<"intel_gpu_tgllp", Fp16Atomic64, Sg8_16_32>;
def : IntelTargetInfo<"intel_gpu_ehl", Fp16Atomic64, Sg8_16_32>;
def : IntelTargetInfo<"intel_gpu_icllp", Fp16Atomic64, Sg8_16_32>;
def : IntelTargetInfo<"intel_gpu_cml", Fp16Fp64Atomic64, Sg8_16_32>;
def : IntelTargetInfo<"intel_gpu_aml", Fp16Fp64Atomic64, Sg8_16_32>;
def : IntelTargetInfo<"intel_gpu_whl", Fp16Fp64Atomic64, Sg8_16_32>;
def : IntelTargetInfo<"intel_gpu_glk", Fp16Fp64Atomic64, Sg8_16_32>;
def : IntelTargetInfo<"intel_gpu_apl", Fp16Fp64Atomic64, Sg8_16_32>;
def : IntelTargetInfo<"intel_gpu_cfl", Fp16Fp64Atomic64, Sg8_16_32>;
def : IntelTargetInfo<"intel_gpu_kbl", Fp16Fp64Atomic64, Sg8_16_32>;
def : IntelTargetInfo<"intel_gpu_skl", Fp16Fp64Atomic64, Sg8_16_32>;
def : IntelTargetInfo<"intel_gpu_bdw", Fp16Fp64Atomic64, Sg8_16_32>;

//
// CUDA / NVPTX device aspects
//

// Specialize the TargetInfo class for CUDA:
// 'maySupportOtherAspects' is set to 1 because any Cuda device with the same
// or newer capabilities will support the aspects of the least capable device.
class CudaTargetInfo<string targetName, list<Aspect> aspectList, int subGroupSize = 32, string toolchain = "", string options = "">
    : TargetInfo<targetName, aspectList, [subGroupSize], toolchain, options, /*maySupportOtherAspects*/1> {
  assert !eq(subGroupSize, 32), "sub-group size for Cuda must be equal to 32 and not " # subGroupSize # ".";
}

defvar CudaMinAspects = !listconcat(AllUSMAspects, [AspectGpu, AspectFp64, AspectOnline_compiler, AspectOnline_linker,
    AspectQueue_profiling, AspectExt_intel_pci_address, AspectExt_intel_max_mem_bandwidth, AspectExt_intel_memory_bus_width,
    AspectExt_intel_device_info_uuid, AspectExt_oneapi_native_assert, AspectExt_intel_free_memory, AspectExt_intel_device_id,
    AspectExt_intel_memory_clock_rate, AspectExt_oneapi_ballot_group, AspectExt_oneapi_fixed_size_group,
    AspectExt_oneapi_opportunistic_group, AspectExt_oneapi_graph, AspectExt_oneapi_limited_graph]);
// Bindless images aspects are partially supported on CUDA and disabled by default at the moment.
defvar CudaBindlessImagesAspects = [AspectExt_oneapi_bindless_images, AspectExt_oneapi_bindless_images_shared_usm,
    AspectExt_oneapi_bindless_images_1d_usm, AspectExt_oneapi_bindless_images_2d_usm, AspectExt_oneapi_interop_memory_import,
    AspectExt_oneapi_interop_semaphore_import, AspectExt_oneapi_mipmap, AspectExt_oneapi_mipmap_anisotropy,
    AspectExt_oneapi_mipmap_level_reference, AspectExt_oneapi_cubemap, AspectExt_oneapi_cubemap_seamless_filtering];

def : CudaTargetInfo<"nvidia_gpu_sm_50", !listconcat(CudaMinAspects, CudaBindlessImagesAspects)>;
def : CudaTargetInfo<"nvidia_gpu_sm_52", !listconcat(CudaMinAspects, CudaBindlessImagesAspects)>;
def : CudaTargetInfo<"nvidia_gpu_sm_53", !listconcat(CudaMinAspects, CudaBindlessImagesAspects, [AspectFp16])>;
def : CudaTargetInfo<"nvidia_gpu_sm_60", !listconcat(CudaMinAspects, CudaBindlessImagesAspects, [AspectFp16, AspectAtomic64])>;
def : CudaTargetInfo<"nvidia_gpu_sm_61", !listconcat(CudaMinAspects, CudaBindlessImagesAspects, [AspectFp16, AspectAtomic64])>;
def : CudaTargetInfo<"nvidia_gpu_sm_62", !listconcat(CudaMinAspects, CudaBindlessImagesAspects, [AspectFp16, AspectAtomic64])>;
def : CudaTargetInfo<"nvidia_gpu_sm_70", !listconcat(CudaMinAspects, CudaBindlessImagesAspects, [AspectFp16, AspectAtomic64])>;
def : CudaTargetInfo<"nvidia_gpu_sm_72", !listconcat(CudaMinAspects, CudaBindlessImagesAspects, [AspectFp16, AspectAtomic64])>;
def : CudaTargetInfo<"nvidia_gpu_sm_75", !listconcat(CudaMinAspects, CudaBindlessImagesAspects, [AspectFp16, AspectAtomic64])>;
def : CudaTargetInfo<"nvidia_gpu_sm_80", !listconcat(CudaMinAspects, CudaBindlessImagesAspects,
    [AspectFp16, AspectAtomic64, AspectExt_oneapi_cuda_async_barrier])>;
def : CudaTargetInfo<"nvidia_gpu_sm_86", !listconcat(CudaMinAspects, CudaBindlessImagesAspects,
    [AspectFp16, AspectAtomic64, AspectExt_oneapi_cuda_async_barrier])>;
def : CudaTargetInfo<"nvidia_gpu_sm_87", !listconcat(CudaMinAspects, CudaBindlessImagesAspects,
    [AspectFp16, AspectAtomic64, AspectExt_oneapi_cuda_async_barrier])>;
def : CudaTargetInfo<"nvidia_gpu_sm_89", !listconcat(CudaMinAspects, CudaBindlessImagesAspects,
    [AspectFp16, AspectAtomic64, AspectExt_oneapi_cuda_async_barrier])>;
def : CudaTargetInfo<"nvidia_gpu_sm_90", !listconcat(CudaMinAspects, CudaBindlessImagesAspects,
    [AspectFp16, AspectAtomic64, AspectExt_oneapi_cuda_async_barrier, AspectExt_oneapi_cuda_cluster_group])>;
def : CudaTargetInfo<"nvidia_gpu_sm_90a", !listconcat(CudaMinAspects, CudaBindlessImagesAspects,
    [AspectFp16, AspectAtomic64, AspectExt_oneapi_cuda_async_barrier, AspectExt_oneapi_cuda_cluster_group])>;

//
// HIP / AMDGPU device aspects
//

// Specialize the TargetInfo class for HIP:
// 'maySupportOtherAspects' is set to 0 because there is no guarantee that any
// aspect support is compatible between older and newer gfx architectures.
class HipTargetInfo<string targetName, list<Aspect> aspectList, list<int> subGroupSizesList, string toolchain = "", string options = "">
    : TargetInfo<targetName, aspectList, subGroupSizesList, toolchain, options, /*maySupportOtherAspects*/0>;

// DPCPP does not support AMD targets prior to the gfx7 (GCN2) family.
defvar HipSubgroupSizesGCN2 = [16];     // gfx7
defvar HipSubgroupSizesGCN3 = [16];     // gfx8, GCN 3rd gen and 4th gen have the same subgroup sizes
defvar HipSubgroupSizesGCN5 = [64];     // gfx900-gfx906 GCN5.0 (known as "Vega"), gfx90c GCN5.1 (known as "Vega 7nm")
defvar HipSubgroupSizesRDNA = [32, 64]; // gfxX10-gfx11 (encapsulates RDNA1..3), natively 32 (64-waves mode available)
defvar HipSubgroupSizesCDNA = [64];     // gfx908, gfx90a (encapsulates CDNA1..2)

defvar HipMinAspects = [AspectGpu, AspectFp64, AspectOnline_compiler, AspectOnline_linker, AspectQueue_profiling,
    AspectExt_intel_pci_address, AspectExt_intel_max_mem_bandwidth, AspectExt_intel_device_id,
    AspectExt_intel_memory_clock_rate, AspectExt_intel_memory_bus_width, AspectExt_intel_free_memory];

// The following AMDGCN targets are ordered based on their ROCm driver support:
//
// Officially supported:
def : HipTargetInfo<"amd_gpu_gfx908", !listconcat(HipMinAspects, AllUSMAspects,
    [AspectExt_intel_device_info_uuid, AspectExt_oneapi_graph, AspectExt_oneapi_limited_graph]), HipSubgroupSizesCDNA>;
def : HipTargetInfo<"amd_gpu_gfx90a", !listconcat(HipMinAspects, AllUSMAspects,
    [AspectExt_intel_device_info_uuid, AspectExt_oneapi_graph, AspectExt_oneapi_limited_graph, AspectExt_oneapi_native_assert]),
    HipSubgroupSizesCDNA>;
def : HipTargetInfo<"amd_gpu_gfx940", !listconcat(HipMinAspects, AllUSMAspects,
    [AspectExt_intel_device_info_uuid, AspectExt_oneapi_graph, AspectExt_oneapi_limited_graph]),
    HipSubgroupSizesCDNA>;
def : HipTargetInfo<"amd_gpu_gfx1030", !listconcat(HipMinAspects, AllUSMAspects,
    [AspectExt_intel_device_info_uuid, AspectExt_oneapi_graph, AspectExt_oneapi_limited_graph]),
    HipSubgroupSizesRDNA>;
def : HipTargetInfo<"amd_gpu_gfx1100", !listconcat(HipMinAspects, AllUSMAspects,
    [AspectExt_intel_device_info_uuid, AspectExt_oneapi_graph, AspectExt_oneapi_limited_graph]),
    HipSubgroupSizesRDNA>;
// Deprecated support:
def : HipTargetInfo<"amd_gpu_gfx906", !listconcat(HipMinAspects, AllUSMAspects), HipSubgroupSizesGCN5>;
// Unsupported (or unofficially supported):
def : HipTargetInfo<"amd_gpu_gfx700", HipMinAspects, HipSubgroupSizesGCN2>;
def : HipTargetInfo<"amd_gpu_gfx701", HipMinAspects, HipSubgroupSizesGCN2>;
def : HipTargetInfo<"amd_gpu_gfx702", HipMinAspects, HipSubgroupSizesGCN2>;
def : HipTargetInfo<"amd_gpu_gfx801", HipMinAspects, HipSubgroupSizesGCN3>;
def : HipTargetInfo<"amd_gpu_gfx802", HipMinAspects, HipSubgroupSizesGCN3>;
def : HipTargetInfo<"amd_gpu_gfx803", HipMinAspects, HipSubgroupSizesGCN3>;
def : HipTargetInfo<"amd_gpu_gfx805", HipMinAspects, HipSubgroupSizesGCN3>;
def : HipTargetInfo<"amd_gpu_gfx810", HipMinAspects, HipSubgroupSizesGCN3>;
def : HipTargetInfo<"amd_gpu_gfx900", HipMinAspects, HipSubgroupSizesGCN5>;
def : HipTargetInfo<"amd_gpu_gfx902", HipMinAspects, HipSubgroupSizesGCN5>;
def : HipTargetInfo<"amd_gpu_gfx904", HipMinAspects, HipSubgroupSizesGCN5>;
def : HipTargetInfo<"amd_gpu_gfx909", HipMinAspects, HipSubgroupSizesGCN5>;
def : HipTargetInfo<"amd_gpu_gfx90c", !listconcat(HipMinAspects, AllUSMAspects), HipSubgroupSizesGCN5>;
def : HipTargetInfo<"amd_gpu_gfx1010", !listconcat(HipMinAspects, AllUSMAspects), HipSubgroupSizesRDNA>;
def : HipTargetInfo<"amd_gpu_gfx1011", !listconcat(HipMinAspects, AllUSMAspects), HipSubgroupSizesRDNA>;
def : HipTargetInfo<"amd_gpu_gfx1012", !listconcat(HipMinAspects, AllUSMAspects), HipSubgroupSizesRDNA>;
def : HipTargetInfo<"amd_gpu_gfx1013", !listconcat(HipMinAspects, AllUSMAspects), HipSubgroupSizesRDNA>;
def : HipTargetInfo<"amd_gpu_gfx1031", !listconcat(!listremove(HipMinAspects, [AspectExt_intel_free_memory]), AllUSMAspects),
    HipSubgroupSizesRDNA>;
def : HipTargetInfo<"amd_gpu_gfx1032", !listconcat(HipMinAspects, AllUSMAspects), HipSubgroupSizesRDNA>;
def : HipTargetInfo<"amd_gpu_gfx1033", !listconcat(HipMinAspects, AllUSMAspects), HipSubgroupSizesRDNA>;
def : HipTargetInfo<"amd_gpu_gfx1034", !listconcat(HipMinAspects, AllUSMAspects), HipSubgroupSizesRDNA>;
def : HipTargetInfo<"amd_gpu_gfx1035", !listconcat(HipMinAspects, AllUSMAspects), HipSubgroupSizesRDNA>;
def : HipTargetInfo<"amd_gpu_gfx1036", !listconcat(HipMinAspects, AllUSMAspects), HipSubgroupSizesRDNA>;
def : HipTargetInfo<"amd_gpu_gfx1101", !listconcat(HipMinAspects, AllUSMAspects), HipSubgroupSizesRDNA>;
def : HipTargetInfo<"amd_gpu_gfx1102", !listconcat(HipMinAspects, AllUSMAspects), HipSubgroupSizesRDNA>;
def : HipTargetInfo<"amd_gpu_gfx1103", !listconcat(HipMinAspects, AllUSMAspects), HipSubgroupSizesRDNA>;
def : HipTargetInfo<"amd_gpu_gfx1150", !listconcat(HipMinAspects, AllUSMAspects), HipSubgroupSizesRDNA>;
def : HipTargetInfo<"amd_gpu_gfx1151", !listconcat(HipMinAspects, AllUSMAspects), HipSubgroupSizesRDNA>;
// TBA
def : HipTargetInfo<"amd_gpu_gfx941", [], []>;  // CDNA 3
def : HipTargetInfo<"amd_gpu_gfx942", [], []>;  // CDNA 3
def : HipTargetInfo<"amd_gpu_gfx1200", [], []>; // RDNA 4
def : HipTargetInfo<"amd_gpu_gfx1201", [], []>; // RDNA 4
