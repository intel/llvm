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

**Experimental**: the Device Config File should be extendable even at runtime
for experimental purposes and when dealing with new HW which may not be
supported by older compilers. **Question: Why is not enough compile time for new
HW?**

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

The original `.td` file should look like the one below: **Question: investigate
if we can have enums in .td files. Thus we can have list<int> rather than
list<string> for aspects.** 
``` 
def TargetTable : DynamicTable { 
    let FilterClass = "TargetInfo";
    let Fields = ["TargetName", "aspects", "maySupportOtherAspects",
                  "subGroupSizes", "aotToolchain", "oclocDevice"];
    let PrimaryKey = ["TargetName"];
}

class TargetInfo <string tgtName, list<string> listAspects, bit otherAspects,
                  list<int> listSubGroupSizes, string toolchain, string device>
{
          list<string> aspects = listAspects;
          bits<1> maySupportOtherAspects = otherAspects;
          list<int> subGroupSizes = listSubGroupSizes;
          string aotToolchain = toolchain;
          string oclocDevice = device;
}

def : TargetInfo<"TargetA", {"aspect_name_1", "aspect_name_2"}, 
                                  0, {8, 16}, "ocloc", "tgtA">; 
def : TargetInfo<"TargetB", {"aspect_name_4", "aspect_name_8"},
                                  0, {8, 16}, "ocloc", "tgtB">;
def : TargetInfo<"TargetC", {"aspect_name_6", "aspect_name_11"},
                                   0, {8, 32}, "ocloc", "tgtC">;
```

The generated `.inc` file should look like the example below: 
```c++
struct TargetInfo {
  bool maySupportOtherAspects;
  std::vector<std::string> aspects;
  std::vector<int> subGroupSizes;
  std::string aotToolchain;
  std::string oclocDevice;
};

std::map<std::string, TargetInfo> targets = {
    {"TargetA",
     {{"aspect_name_1", "aspect_name_2"}, 0, {8, 16}, "ocloc", "tgtA"}},
    {"TargetB",
     {{"aspect_name_4", "aspect_name_8"}, 0, {8, 16}, "ocloc", "tgtB"}},
    {"TargetC",
     {{"aspect_name_6", "aspect_name_11"}, 0, {8, 32}, "ocloc", "tgtC"}}};
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

**TODO: extend to explain how we should create a header file declaring the
namespace and including the .inc file within.**

### Changes to Build Infrastructure
We need the information about the targets in multiple tools and compiler
modules listed in [Requirements](#Requirements).  Thus, we need to make sure
that the generation of the `.inc` file out of the `.td` file is done in time
for all the consumers. The command we need to run for TableGen is `llvm-tblgen
-gen-[custom-backend] -I /llvm-root/llvm/include/ input.td -o output.inc`.
Additionally, we need to set dependencies adequately so that this command is
run before any of the consumers need it.

**Question: can we set the flag ourselves? Is there any convention we need to
follow?**

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
