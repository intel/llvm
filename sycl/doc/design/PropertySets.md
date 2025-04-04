# SYCL binary property sets

To communicate information about SYCL binaries to the SYCL runtime, the
implementation produces sets of properties. The intention of this design
document is to describe the structure of the property sets and define the
representation and meaning of pre-defined property set names.


## Property sets structure

A property set consists of a reserved name, enclosed in square brackets,
followed by a series of string key and value pairs. The set name and each entry
in the set are separated by a newline.

The string key and value pairs have the following format:
```
<string key>=<value type>|<value>
```

The value type is a string and the value of it has the following meaning for the
corresponding value:

| Value type | Description                               |
| ---------- | ----------------------------------------- |
| "1"        | The value is a 32 bit integer.            |
| "2"        | The value is a base64 encoded byte array. |

__Note:__ Whitespaces are __not__ ignored and are treated like any other
characters. As such, `some_key=1|1` is not the same as `some_key = 1 | 1` and
`[some property set]` is not the same as `[ some property set ]`.


## Property sets

This section describes the known property sets.


### [SYCL/specialization constants]

__Key:__ Specialization constant name.

__Value type:__ Byte array. ("2")

__Value:__ Information about the specialization constant with the following
fields: 

```c++
// Encodes ID of a scalar specialization constants which is a leaf of some
// composite specialization constant.
unsigned ID;
// Encodes offset from the beginning of composite, where scalar resides, i.e.
// location of the scalar value within a byte-array containing the whole
// composite specialization constant. If descriptor is used to represent a
// whole scalar specialization constant instead of an element of a composite,
// this field should be contain zero.
unsigned Offset;
// Encodes size of scalar specialization constant.
unsigned Size;
```

See also [SYCL2020-SpecializationConstants.md](./SYCL2020-SpecializationConstants.md).


### [SYCL/specialization constants default values]

__Key:__ Specialization constant name.

__Value type:__ Byte array. ("2")

__Value:__ Byte representation of the default value for the specialization
constant.

See also [SYCL2020-SpecializationConstants.md](./SYCL2020-SpecializationConstants.md).


### [SYCL/devicelib req mask]

__Key:__ At most one entry with "DeviceLibReqMask".

__Value type:__ 32 bit integer. ("1")

__Value:__ A bitmask of which device libraries the binary uses.

__Notes:__

1. If this property set is missing, no device libraries are used by the binary.


### [SYCL/kernel param opt]

__Key:__ Kernel name.

__Value type:__ Byte array. ("2")

__Value:__ A bitmask identifying the arguments of the kernel that have been
removed by the dead-argument-elimination optimization pass.

__Notes:__

1. If no entry is present for a given kernel in the binary, no arguments have
been eliminated.
2. If this property set is missing, no kernels in the binary have any eliminated
arguments.


### [SYCL/program metadata]

Program metadata properties:

| Key                                      | Value type            | Value                                                                                                            |
| ---------------------------------------- | --------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `kernel` + "@reqd_work_group_size"       | Byte array. ("2")     | Specifies the required work-group size for the kernel identified by the name `kernel`.                           |
| `kernel` + "@work_group_num_dim"         | Byte array. ("2")     | Specifies the work-group dimensionality of the kernel identified by the name `kernel`.                           |
| `kernel` + "@max_work_group_size"        | Byte array. ("2")     | Specifies the max work-group size for the kernel identified by the name `kernel`.                                |
| `kernel` + "@max_linear_work_group_size" | Byte array. ("2")     | Specifies the max linear work-group size for the kernel identified by the name `kernel`.                         |
| `variable` + "@global_id_mapping"        | Byte array. ("2")     | Specifies the mapping between the global variable with unique identifier `variable` and its name in the binary.  |


### [SYCL/misc properties]

Miscellaneous properties:

| Key                             | Value type            | Value                                                                                                            |
| ------------------------------- | --------------------- | ---------------------------------------------------------------------------------------------------------------- |
| "isEsimdImage"                  | 32 bit integer. ("1") | 1 if the image is ESIMD and 0 or missing otherwise.                                                              |
| "sycl-register-alloc-mode"      | 32 bit integer. ("1") | The register allocation mode: 2 for large and 0 or missing for automatic.                                        |
| "sycl-grf-size"                 | 32 bit integer. ("1") | The GRF size. Automatic if 0 or missing.                                                                         |
| "optLevel"                      | 32 bit integer. ("1") | Optimization level, corresponding to the `-O` option used during compilation.                                    |
| "sanUsed"                       | Byte array. ("2")     | Specifying if address sanitization ("asan") or memory sanitization ("msan") is used. Missing if neither is used. |
| "specConstsReplacedWithDefault" | 32 bit integer. ("1") | 1 if the specialization constants have been replaced by their default values and 0 or missing otherwise.         |


### [SYCL/assert used]

__Key:__ Kernel name.

__Value type:__ 32 bit integer. ("1")

__Value:__ 1 if the kernel uses assertions and 0 or missing otherwise.


### [SYCL/exported symbols]

__Key:__ Symbol name.

__Value type:__ 32 bit integer. ("1")

__Value:__ 1 if the symbol is exported by the binary and 0 or missing otherwise.

See also [SharedLibraries.md](SharedLibraries.md).


### [SYCL/imported symbols]

__Key:__ Symbol name.

__Value type:__ 32 bit integer. ("1")

__Value:__ 1 if the symbol is imported by the binary and 0 or missing otherwise.

See also [SharedLibraries.md](SharedLibraries.md).


### [SYCL/device globals]

__Key:__ Device global variable name.

__Value type:__ Byte array. ("2")

__Value:__ Information about the device global variable with the following
fields: 

```c++
  // Encodes size of the underlying type T of the device global variable.
  uint32_t Size;

  // Either 1 (true) or 0 (false), telling whether the device global variable
  // was declared with the device_image_scope property.
  // We use uint32_t for a boolean value to eliminate padding after the field
  // and suppress false positive reports from MemorySanitizer.
  uint32_t DeviceImageScope;
```

__Notes:__

1. If this property set is missing, the binary does not contain any device
global variables.

See also [DeviceGlobal.md](./DeviceGlobal.md).


### [SYCL/device requirements]

Set of device requirements for the entire module:

| Key                             | Value type        | Value                                                                                                                                                         |
| ------------------------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| "aspects"                       | Byte array. ("2") | A collection of 32 bit integers representing the SYCL aspects used. These correspond 1:1 with the enum values of `sycl::aspect`.                              |
| "fixed_target"                  | Byte array. ("2") | The string literals specified in `-fsycl-fixed-targets`.                                                                                                      |
| "reqd_work_group_size_uint64_t" | Byte array. ("2") | At most three 64 bit unsigned integers representing the required work-group size. If this entry is missing, there is no work-group size requirement.          |
| "joint_matrix"                  | Byte array. ("2") | A string containing a semi-colon-separated list of comma-separated descriptors for used matrices. The descriptors in the order they appear are: <ul><li>sycl-joint-matrix-type</li><li>sycl-joint-matrix-use</li><li>sycl-joint-matrix-rows</li><li>sycl-joint-matrix-cols</li></ul> |
| "joint_matrix_mad"              | Byte array. ("2") | A string containing a semi-colon-separated list of comma-separated descriptors for used matrix MAD operations. The descriptors in the order they appear are: <ul><li>sycl-joint-matrix-mad-type-A</li><li>sycl-joint-matrix-mad-type-B</li><li>sycl-joint-matrix-mad-type-C</li><li>sycl-joint-matrix-mad-type-D</li><li>sycl-joint-matrix-mad-size-M</li><li>sycl-joint-matrix-mad-size-K</li><li>sycl-joint-matrix-mad-size-N</li></ul> |
| "reqd_sub_group_size"           | Byte array. ("2") | At most three 32 bit unsigned integers representing the required sub-group size. If this entry is missing, there is no sub-group size requirement.            |
| "work_group_num_dim"            | Byte array. ("2") | At most three 32 bit unsigned integers representing the work-group dimensionality. If this entry is missing, there is no specified work-group dimensionality. |


See also [OptionalDeviceFeatures.md](OptionalDeviceFeatures.md).


### [SYCL/host pipes]

__Key:__ Host pipe variable name.

__Value type:__ Byte array. ("2")

__Value:__ Information about the host pipe variable with the following
fields: 

```c++
  // Encodes size of the underlying type T of the host pipe variable.
  uint32_t Size;
```

__Notes:__

1. If this property set is missing, the binary does not contain any host pipe
variables.


### [SYCL/virtual functions]

Set of information about virtual function usage in the module.

| Key                          | Value type        | Value                                                                                                                                                              |
| ---------------------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| "virtual-functions-set"      | Byte array. ("2") | A string identifying the set of virtual functions contained in the module. If this is missing, the module does not contain any virtual function sets.              |
| "uses-virtual-functions-set" | Byte array. ("2") | A string containing a comma-separated list of sets of virtual functions used by the module. If this is missing, the module does not use any virtual function sets. |


### [SYCL/implicit local arg]

__Key:__ Kernel name.

__Value type:__ 32 bit integer. ("1")

__Value:__ Index of the implicit local memory argument.

__Notes:__

1. If no entry is present for a given kernel in the binary, the kernel does not
have an implicit local memory argument.
2. If this property set is missing, no kernels in the binary have an implicit
local memory argument.


### [SYCL/registered kernels]

__Key:__ "Registered" kernel name.

__Value type:__ Byte array. ("2")

__Value:__ The name of the kernel corresponding to the registered kernel name.

__Notes:__

1. If this property set is missing, the binary does not have any registered
kernel names.


### [SYCLBIN/global metadata]

Set of global information about a SYCLBIN file.

| Key     | Value type            | Value |
| ------- | --------------------- | ----- |
| "state" | 32 bit integer. ("1") | Integer representation of one of the possible states of the file, corresponding to the `sycl::bundle_state` enum. It must be one of the following:<ol start="0"><li>`sycl::bundle_state::input`</li><li>`sycl::bundle_state::object`</li><li>`sycl::bundle_state::executable`</li></ol> |


### [SYCLBIN/ir module metadata]

Set of information about an IR module in a SYCLBIN file.

| Key      | Value type            | Value |
| -------- | --------------------- | ----- |
| "type"   | 32 bit integer. ("1") | Integer representation of one of the pre-defined IR types. It must be one of the following:<ol start="0"><li>SPIR-V</li><li>PTX</li><li>AMDGCN</li></ol> |
| "target" | Byte array. ("2")     | A string representing the architecture of the binary, corresponding to the value of `-fsycl-targets` option used when compiling this binary. This may be missing if no part of `-fsycl-targets` was used during the compilation of this binary or if `-fsycl-targets` was not used at all. |


### [SYCLBIN/native device code image metadata]

Set of information about an native device code image in a SYCLBIN file.

| Key    | Value type        | Value |
| ------ | ----------------- | ----- |
| "arch" | Byte array. ("2") | A string representing the architecture of the binary, corresponding to the value of `-fsycl-targets` option used when compiling this binary. |

