# Implementation Design for Device Configuration File
This design document describes the implementation of the DPC++ Device
Configuration File.

In summary, there several scenarios where we need to know information about a
target at compile-time, which is the main purpose of this Device Configuration
File. Examples are `any_device_has/all_devices_have` which defines macros
depending on the optional features supported by a target; or conditional AOT
compilation based on optional features used in kernels and supported by targets.

## Requirements
We need a default Device Configuration File embedded in the compiler describing
the well known targets at the time of building the compiler.  This embedded
knowledge must be extendable, since our AOT toolchain allows compiling for
targets not known at the time of building the compiler so long as the
appropriate toolchain --AOT compiler and driver-- support such targets. In
other words, we need to provide a way for users to add entries for new targets
at application compile time.

An entry of the Device Configuration File should include:
- Name of the target.
- List of supported aspects.
- List of supported sub-group sizes.
- [Optional] `aot-toolchain` command to be called when compiling for this
target. This information is optional because we plan to implement an
auto-detection mechanism that is able to infer the `aot-toolchain` from the
target name for well known targets.
- [Optional] `ocloc-device` information to be passed to the `aot-toolchain`
command so that it knows what the target device is. This information is optional
because we plan to implement an auto-detection mechanism that is able to infer
the `ocloc-device` from the target name for well known targets.

The information provided in the Device Configuration File is required from
different tools and compiler modules:
- Compiler driver:
    - `any_device_has/all_devices_have` requires compiler driver to read the
    config file and define corresponding macros.
    [[DeviceAspectTraitDesign](https://github.com/intel/llvm/blob/sycl/sycl/doc/design/DeviceAspectTraitDesign.md)]
    - Compiler driver requires `aot-toolchain` and `ocloc-device` to trigger the
    compilation for the required targets.
    [https://github.com/intel/llvm/pull/6775/files]
- `sycl-aspect-filter`:
https://github.com/intel/llvm/blob/sycl/sycl/doc/design/OptionalDeviceFeatures.md#aspect-filter-tool

Finally, overhead should be minimal. Particularly, users should not pay for what
they do not use. This motivates our decision to embed the default Device
Configuration File rather than releasing it as a separate file. 

## High-Level Design
The default Device Configuration File is a `.td` file located in the compiler
source code. `.td` is the file extension for [LLVM
TableGen](https://llvm.org/docs/TableGen/). This default file will include all
the devices known by the developers at the time of the release. During the
build process, using a custom TableGen backend, we generate a `.inc` C++ file
containing a `std::map` with one key/value element for each entry in the `.td`
file. Using a map we can later update or add new elements if the user provides
new targets at application compile time. Finally, the tools and compiler
modules that need information about the targets can simply query the map to get
it.

### New `TableGen` backend
Note: This [guide](https://llvm.org/docs/TableGen/BackGuide.html) details how
to implement new TableGen backends. Also, the [Search
Indexes](https://llvm.org/docs/TableGen/BackEnds.html#search-indexes) backend
already does something very similar to what we seek. It generates a table that
provides a lookup function, but it cannot be extended with new entries. We can
use _Search Indexes_ backend as inspiration for ours. 

Our backend should generate a map where the key is the target name and the value
is an object of a custom class/struct including all the information required. 

The original `.td` file should look like the one below: 
``` 
include "llvm/TableGen/SearchableTable.td"
// AspectEnum, AspectEntry, and all the aspects definitions could be outlined
// to another .td file that could be included into this file
def AspectEnum : GenericEnum {
  let FilterClass = "AspectEntry";
  let NameField = "Name";
  let ValueField = "Encoding";
}

class AspectEntry<bits<16> enc> {
  string Name = NAME;
  bits<16> Encoding = enc;
}

def AspectCpu : AspectEntry<1>;
def AspectGpu : AspectEntry<2>;
def AspectAccelerator : AspectEntry<3>;
def AspectCustom : AspectEntry<4>;
def AspectFp16 : AspectEntry<5>;
def AspectFp64 : AspectEntry<6>;
def AspectImage : AspectEntry<9>;
def AspectOnline_compiler : AspectEntry<10>;
def AspectOnline_linker : AspectEntry<11>;
def AspectQueue_profiling : AspectEntry<12>;
def AspectUsm_device_allocations : AspectEntry<13>;
def AspectUsm_host_allocations : AspectEntry<14>;
def AspectUsm_shared_allocations : AspectEntry<15>;
def AspectUsm_system_allocations : AspectEntry<17>;
def AspectExt_intel_pci_address : AspectEntry<18>;
def AspectExt_intel_gpu_eu_count : AspectEntry<19>;
def AspectExt_intel_gpu_eu_simd_width : AspectEntry<20>;
def AspectExt_intel_gpu_slices : AspectEntry<21>;
def AspectExt_intel_gpu_subslices_per_slice : AspectEntry<22>;
def AspectExt_intel_gpu_eu_count_per_subslice : AspectEntry<23>;
def AspectExt_intel_max_mem_bandwidth : AspectEntry<24>;
def AspectExt_intel_mem_channel : AspectEntry<25>;
def AspectUsm_atomic_host_allocations : AspectEntry<26>;
def AspectUsm_atomic_shared_allocations : AspectEntry<27>;
def AspectAtomic64 : AspectEntry<28>;
def AspectExt_intel_device_info_uuid : AspectEntry<29>;
def AspectExt_oneapi_srgb : AspectEntry<30>;
def AspectExt_oneapi_native_assert : AspectEntry<31>;
def AspectHost_debuggable : AspectEntry<32>;
def AspectExt_intel_gpu_hw_threads_per_eu : AspectEntry<33>;
def AspectExt_oneapi_cuda_async_barrier : AspectEntry<34>;
def AspectExt_oneapi_bfloat16_math_functions : AspectEntry<35>;
def AspectExt_intel_free_memory : AspectEntry<36>;
def AspectExt_intel_device_id : AspectEntry<37>;
def AspectExt_intel_memory_clock_rate : AspectEntry<38>;
def AspectExt_intel_memory_bus_width : AspectEntry<39>;
def AspectEmulated : AspectEntry<40>;
    
def TargetTable : DynamicTable { 
    let FilterClass = "TargetInfo";
    let Fields = ["TargetName", "aspects", "maySupportOtherAspects",
                  "subGroupSizes", "aotToolchain", "oclocDevice"];
    let PrimaryKey = ["TargetName"];
}

class TargetInfo <string tgtName, list<AspectEntry> listAspects, bit otherAspects,
                  list<int> listSubGroupSizes, string toolchain, string device>
{
    list<AspectEntry> aspects = listAspects;
    bits<1> maySupportOtherAspects = otherAspects;
    list<int> subGroupSizes = listSubGroupSizes;
    string aotToolchain = toolchain;
    string oclocDevice = device;
}

def : TargetInfo<"TargetA", [AspectCpu, AspectAtomic64], 
                                  0, [8, 16], "ocloc", "tgtA">; 
def : TargetInfo<"TargetB", [AspectGpu, AspectFp16],
                                  0, [8, 16], "ocloc", "tgtB">;
def : TargetInfo<"TargetC", [AspectEmulated, AspectImage],
                                   0, [8, 32], "ocloc", "tgtC">;
```
Note: backends tested don't allow lists within `TargetInfo` class. I
_think_ this is a backend limitation, rather than a TableGen limitation.
Thus, we should be able to lift this limitation in our own backend.

The generated `.inc` file should look like the example below: 
```c++
enum AspectEnum {
  AspectAccelerator = 3,
  AspectAtomic64 = 28,
  AspectCpu = 1,
  AspectCustom = 4,
  AspectEmulated = 40,
  AspectExt_intel_device_id = 37,
  AspectExt_intel_device_info_uuid = 29,
  AspectExt_intel_free_memory = 36,
  AspectExt_intel_gpu_eu_count = 19,
  AspectExt_intel_gpu_eu_count_per_subslice = 23,
  AspectExt_intel_gpu_eu_simd_width = 20,
  AspectExt_intel_gpu_hw_threads_per_eu = 33,
  AspectExt_intel_gpu_slices = 21,
  AspectExt_intel_gpu_subslices_per_slice = 22,
  AspectExt_intel_max_mem_bandwidth = 24,
  AspectExt_intel_mem_channel = 25,
  AspectExt_intel_memory_bus_width = 39,
  AspectExt_intel_memory_clock_rate = 38,
  AspectExt_intel_pci_address = 18,
  AspectExt_oneapi_bfloat16_math_functions = 35,
  AspectExt_oneapi_cuda_async_barrier = 34,
  AspectExt_oneapi_native_assert = 31,
  AspectExt_oneapi_srgb = 30,
  AspectFp16 = 5,
  AspectFp64 = 6,
  AspectGpu = 2,
  AspectHost_debuggable = 32,
  AspectImage = 9,
  AspectOnline_compiler = 10,
  AspectOnline_linker = 11,
  AspectQueue_profiling = 12,
  AspectUsm_atomic_host_allocations = 26,
  AspectUsm_atomic_shared_allocations = 27,
  AspectUsm_device_allocations = 13,
  AspectUsm_host_allocations = 14,
  AspectUsm_shared_allocations = 15,
  AspectUsm_system_allocations = 17,
};
    
struct TargetInfo {
  bool maySupportOtherAspects;
  std::vector<unsigned> aspects;
  std::vector<unsigned> subGroupSizes;
  std::string aotToolchain;
  std::string oclocDevice;
};

std::map<std::string, TargetInfo> targets = {
    {"TargetA",
     {{AspectCpu, AspectAtomic64}, 0, {8, 16}, "ocloc", "tgtA"}},
    {"TargetB",
     {{AspectGpu, AspectFp16}, 0, {8, 16}, "ocloc", "tgtB"}},
    {"TargetC",
     {{AspectEmulated, AspectImage}, 0, {8, 32}, "ocloc", "tgtC"}}};
```

We also need a header file that includes the `.inc` file generated by the
TableGen backend. Other backends don't generate the definition of `struct
TargetInfo`. This might be a limitation of the backends, or a limitation of
TableGen. In case it cannot be autogenerated by the backend, a possible
workaround is simply to define the struct in this header file. This header
file should look like the code below:
```c++
namespace DeviceConfigFile {
// In case the backend cannot generate a definition for TargetInfo
struct TargetInfo {
  bool maySupportOtherAspects;
  std::vector<unsigned> aspects;
  std::vector<unsigned> subGroupSizes;
  std::string aotToolchain;
  std::string oclocDevice;
};

#include "device_config_file.inc"
}; // namespace DeviceConfigFile
```

Other modules can query the map to get the information like in the example
below:
```c++ 
DeviceConfigFile::TargetInfo info = DeviceConfigFile::targets.find("TargetA");
if (info == DeviceConfigFile::targets.end()) {
  /* Target not found */
  ...
} else {
  auto aspects = info.aspects;
  auto maySupportOtherAspects = info.maySupportOtherAspects;
  auto subGroupSizes = info.subGroupSizes;
  ...
}
```

### Changes to Build Infrastructure
We need the information about the targets in multiple tools and compiler
modules listed in [Requirements](#Requirements).  Thus, we need to make sure
that the generation of the `.inc` file out of the `.td` file is done in time
for all the consumers. The command we need to run for TableGen is `llvm-tblgen
-gen-dynamic-tables -I /llvm-root/llvm/include/ input.td -o output.inc`.
Additionally, we need to set dependencies adequately so that this command is
run before any of the consumers need it.

### Changes to the DPC++ Frontend
To allow users to add new targets we provide a new flag:
`fsycl-device-config-file=/path/to/file.yaml`. Users can pass a `.yaml` file
describing the targets to be added/updated. An example of how such .yaml` file
should look like is shown below.
```
intel_gpu_skl:
    aspects: [aspect_name1, aspect_name2]
    may_support_other_aspects: true/false
    sub-group-sizes: [1, 2, 4, 8]
    aot-toolchain: ocloc
    ocloc-device: skl
```
The frontend module should parse the user-provided `.yaml` file and update the
map with the new information about targets. LLVM provides
[YAML/IO](https://llvm.org/docs/YamlIO.html) library to easily parse `.yaml`
files.

As mentioned in [Requirements](#Requirements), there is an auto-detection
mechanism for `aot-toolchain` and `ocloc-device` inferring these from the target
name. In the `.yaml` example shown above the target name is `intel_gpu_skl`.
From that name, we can infer that `aot-toolchain` is `ocloc` because the name
starts with `intel_gpu`. Also, we can infer that `ocloc-device` is `skl` just by
keeping what is left after the prefix `intel_gpu`.

#### Auto-detection Potential Issues/Limitations
The auto-detection mechanism is a best effort to relieve users from specifying
`aot-toolchain` and `ocloc-device` from well known devices. However, it has
limitations and potential issues:
- Rules for target names: **TODO: Define rules for names so that they can be
auto-detected.**
- Multiple targets with the same name: On the one hand, the compiler emits a
warning so that the user is aware that multiple targets share the same name. On
the other hand, it simply processes each new entry and updates the map with the
latest information found.
- A target name that can be auto-detected specifies `aot-toolchain` and
`ocloc-device`: user-specified information has precedence over auto-detected
information.
