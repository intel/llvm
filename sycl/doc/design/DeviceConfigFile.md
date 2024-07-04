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
other words, we need to provide a way for users to add entries for new targets or
update existing targets at application compile time.

An entry of the Device Configuration File should include:
- Name of the target. Target names should be spelled exactly as expected in
`-fsycl-targets`, since these are going to be used to implement validation of
supported targets.
- List of supported aspects.
- List of supported sub-group sizes.
- [Optional] `aot-toolchain` name/identifier describing the toolchain used to compile
for this target. This information is optional because we plan to implement an
auto-detection mechanism that is able to infer the `aot-toolchain` from the
target name for well known targets.
- [Optional] `aot-toolchain-%option_name` information to be passed to the
`aot-toolchain` command. This information is optional. For some targets, the 
auto-detection mechanism might be able to infer values for this. One example of this
information would be `ocloc-device %device_id`.

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

Further information about TableGen can be found in [TableGenFundamentals](https://releases.llvm.org/1.9/docs/TableGenFundamentals.html).

### New `TableGen` backend
Note: This [guide](https://llvm.org/docs/TableGen/BackGuide.html) details how
to implement new TableGen backends. Also, the [Search
Indexes](https://llvm.org/docs/TableGen/BackEnds.html#search-indexes) backend
already does something very similar to what we seek. It generates a table that
provides a lookup function, but it cannot be extended with new entries. We can
use _Search Indexes_ backend as inspiration for ours. 

Our backend should generate a map where the key is the target name and the value
is an object of a custom class/struct including all the information required. 

Firstly, we need to provide a file describing the `DynamicTable` class. An
example for this is `SearchableTable.td`, which describes `GenericEnum`, and
`GenericTable` classes for `gen-searchable-tables` backend. File
`llvm/include/llvm/TableGen/DynamicTable.td` should look like the one below:
```
//===- DynamicTable.td ----------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the key top-level classes needed to produce a reasonably
// generic dynamic table that can be updated in runtime. DynamicTable objects
// can be defined using the class in this file:
// 1. (Dynamic) Tables. By instantiating the DynamicTable
// class once, a table with the name of the instantiating def is generated and
// guarded by the GET_name_IMPL preprocessor guard.
//
//===----------------------------------------------------------------------===//
// Define a record derived from this class to generate a dynamic table. This
// table resembles a hashtable with a key-value pair, and can updated in runtime.
//
// The name of the record is used as the name of the global primary array of
// entries of the table in C++.
class DynamicTable {
  // Name of a class. The table will have one entry for each record that
  // derives from that class.
  string FilterClass;

  // Name of the C++ struct/class type that holds table entries. The
  // declaration of this type is not generated automatically.
  string CppTypeName = FilterClass;

  // List of the names of fields of collected records that contain the data for
  // table entries, in the order that is used for initialization in C++.
  //
  // TableGen needs to know the type of the fields so that it can format
  // the initializers correctly.
  //
  // For each field of the table named xxx, TableGen will look for a field
  // named TypeOf_xxx and use that as a more detailed description of the
  // type of the field.

  //   class MyTableEntry {
  //     MyEnum V;
  //     ...
  //   }
  //
  //   def MyTable : DynamicTable {
  //     let FilterClass = "MyTableEntry";
  //     let Fields = ["V", ...];
  //     string TypeOf_V = "list<int>";
  //   }
  list<string> Fields;
}
```

This file should be included --either directly or indirectly-- in any other
`.td` file that uses `DynamicTable` class.

The default device configuration `.td` file should look like the one below: 
``` 
include "llvm/TableGen/DynamicTable.td"

// Aspect and all the aspects definitions could be outlined
// to another .td file that could be included into this file
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
def AspectExt_oneapi_cuda_cluster_group : Aspect<"ext_oneapi_cuda_cluster_group">;
def AspectEmulated : Aspect<"emulated">;
    
def TargetTable : DynamicTable { 
    let FilterClass = "TargetInfo";
    let Fields = ["TargetName", "aspects", "maySupportOtherAspects",
                  "subGroupSizes", "aotToolchain", "aotToolchainOptions"];
    string TypeOf_aspects = "list<Aspect>";
    string TypeOf_subGroupSizes = "list<int>"
}

class TargetInfo <string tgtName, list<Aspect> aspectList, bit otherAspects,
                  list<int> listSubGroupSizes, string toolchain, string options>
{
    list<Aspect> aspects = aspectList;
    bits<1> maySupportOtherAspects = otherAspects;
    list<int> subGroupSizes = listSubGroupSizes;
    string aotToolchain = toolchain;
    string aotToolchainOptions = options;
}

def : TargetInfo<"TargetA", [AspectCpu, AspectAtomic64], 
                                  0, [8, 16], "ocloc", "-device tgtA">; 
def : TargetInfo<"TargetB", [AspectGpu, AspectFp16],
                                  0, [8, 16], "ocloc", "-device tgtB">;
def : TargetInfo<"TargetC", [AspectEmulated, AspectImage],
                                   0, [8, 32], "ocloc", "-device tgtC, -option2 val">;
```
Note: backends tested don't allow lists within `TargetInfo` class. This is a 
backend limitation, rather than a TableGen limitation. Thus, we should be able
to lift this limitation in our own backend, as shown in the initial prototype
implemented to drive the design.

The generated `.inc` file should look like the example below: 
```c++
std::map<std::string, TargetInfo> TargetTable = {
    {"TargetA",
     {{"cpu", "atomic64"}, 0, {8, 16}, "ocloc", "-device tgtA"}},
    {"TargetB",
     {{"gpu", "fp16"}, 0, {8, 16}, "ocloc", "-device tgtB"}},
    {"TargetC",
     {{"emulated", "image"}, 0, {8, 32}, "ocloc", "-device tgtC, -option2 val"}}};
```

We also need a header file that includes the `.inc` file generated by the
TableGen backend. Other backends don't generate the definition of `struct
TargetInfo`, and this seems a good idea to me: it simplifies the backend
implementation, and it is easier for developers to check the data structure
to understand how to work with it. The idea is simply to define the struct 
in this header file. This header file should look like the code below:
```c++
namespace DeviceConfigFile {
struct TargetInfo {
  bool maySupportOtherAspects;
  std::vector<std::string> aspects;
  std::vector<unsigned> subGroupSizes;
  std::string aotToolchain;
  std::string aotToolchainOptions;
};

#include "device_config_file.inc"
using TargetTable_t = std::map<std::string, TargetInfo>;
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

## Tools and Modules Interacting with Device Config File
This is a list of the tools and compiler modules that require using the file:
- The *compiler driver* needs the file to determine the set of legal values for 
`-fsycl-targets`.
- The *compiler driver* needs the file to define macros for `any_device_has/all_devices_have`.
- *Clang* needs the file to emit diagnostics related to `-fsycl-fixed-targets.`
- `sycl-post-link` needs the file to filter kernels in device images when doing AOT
compilation.

Following, you can find the changes required in different parts of the project
in more detail.

### Changes to Build Infrastructure
We need the information about the targets in multiple tools and compiler
modules listed in [Requirements](#requirements).  Thus, we need to make sure
that the generation of the `.inc` file out of the `.td` file is done in time
for all the consumers. The command we need to run for TableGen is `llvm-tblgen
-gen-dynamic-tables -I /llvm-root/llvm/include/ input.td -o output.inc`.
Additionally, we need to set dependencies adequately so that this command is
run before any of the consumers need it.

### Changes to the DPC++ Frontend
To allow users to add new targets we provide a new flag:
`fsycl-device-config-file=/path/to/file.yaml`. Users can pass a `.yaml` file
describing the targets to be added/updated. An example of how such `.yaml` file
should look like is shown below.
```
intel_gpu_skl:
    aspects: [aspect_name1, aspect_name2]
    may_support_other_aspects: true/false
    sub-group-sizes: [1, 2, 4, 8]
    aot-toolchain: ocloc
    aot-toolchain-options: -device skl
```
The frontend module should parse the user-provided `.yaml` file and update the
map with the new information about targets. LLVM provides
[YAML/IO](https://llvm.org/docs/YamlIO.html) library to easily parse `.yaml`
files. The driver should propagate this option to all the tools that require
the Device Configuration File (e.g. `sycl-post-link`) so that each of the
tools can modify the map according to the user extensions described in the 
`.yaml` file. 

As mentioned in [Requirements](#requirements), there is an auto-detection
mechanism for `aot-toolchain` and `aot-toolchain-options` that is able to
infer these from the target name. In the `.yaml` example shown above the target
name is `intel_gpu_skl`. From that name, we can infer that `aot-toolchain` is
`ocloc` because the name starts with `intel_gpu`. Also, we can infer that it needs
`aot-toolchain-options` set to `-device skl` just by keeping what is left after the
prefix `intel_gpu`.

#### Potential Issues/Limitations
- Multiple targets with the same name: On the one hand, the compiler emits a
warning so that the user is aware that multiple targets share the same name. On
the other hand, it simply processes each new entry and updates the map with the
latest information found.

The auto-detection mechanism is a best effort to relieve users from specifying
`aot-toolchain` and `aot-toolchain-options` from well known devices. However, 
it has its own limitations and potential issues:
- Rules for target names: As of now, auto-detection is only available for Intel GPU
targets. All targets starting with `intel_gpu_` will automatically set
`aot-toolchain=ocloc` and `aot-toolchain-options=-device suffix` being suffix the part
left after `intel_gpu_` prefix.
- User specifies `aot-toolchain` and `aot-toolchain-options` for a target name 
that can be auto-detected: user-specified information has precedence over auto-detected
information.

## Testing
There is a danger that the device configuration file will get out-of-sync with the
actual device capabilities. In order to prevent that, we need testing to validate
that the device config file does not go out-of-sync. There are two tests that we
should include:
- A test that compares the list of aspects known to SYCL RT (defined in `aspects.def`)
with the list of aspects defined in the `.td` file describing the default configuration.
This will be useful to detect new aspects added to SYCL RT that have not been added in
the `.td` file.
- A test that compares the aspects listed in the `.td` file with the aspects reported
via `device::has` for each device listed in the `.td` file. Both lists should match.
This test could copy the mechanism of the test for `any_device_has` that goes over each
item in `aspects.def` and tries to instantiate `any_device_has` with that enumerator.

Neither of the tests provides guarantees that nothing went out-of-sync *per se*, we 
would require running the second test in all the targets described in the `.td` file
for such guarantees, but at least provides the mechanism to detect potential desyncs.

