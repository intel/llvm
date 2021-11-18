# SYCL(TM) Proposal: Intel's Extensions for Device Information


**NOTE**: Khronos(R) is a registered trademark and SYCL(TM) is a trademark of the Khronos Group, Inc.

New device descriptors will be added to provide access to low-level hardware details about Intel GPU devices.  This information will be useful to developers tuning specifically for those devices.

This proposal details what is required to provide this information as a SYCL extensions.

## Feature Test Macro ##

The Feature Test Macro SYCL\_EXT\_INTEL\_DEVICE\_INFO will be defined as one of the values defined in the table below. The existence of this macro can be tested to determine if the implementation supports this feature, or applications can test the macro's value to determine which of the extension's APIs the implementation supports.

| Value | Description |
| ----- | ----------- |
| 1     | Initial extension version\. Base features are supported |
| 2     | Device UUID is supported |
| 3     | HW threads per EU device query is supported |


# Device UUID #

A new device descriptor will be added which will provide the device Universal Unique ID (UUID).

This new device descriptor is currently only available for devices in the Level Zero platform, and the matching aspect is only true for those devices. The DPC++ default behavior would be to expose the UUIDs of all supported devices which enables detection of total number of unique devices.


## Version ##

The extension supports this query in version 2 and later.


## Device Information Descriptors ##

| Device Descriptors | Return Type | Description |
| ------------------ | ----------- | ----------- |
| info\:\:device\:\:ext\_intel\_device\_info\_uuid | unsigned char | Returns the device UUID|


## Aspects ##

A new aspect, ext\_intel\_device\_info\_uuid, will be added.

## Error Condition ##

An invalid object runtime error will be thrown if the device does not support aspect\:\:ext\_intel\_device\_info\_uuid.


## Example Usage ##

The UUID can be obtained using the standard get\_info() interface.

    if (dev.has(aspect::ext_intel_device_info_uuid)) {
      auto UUID = dev.get_info<info::device::ext_intel_device_info_uuid>();
    }



# PCI Address #

A new device descriptor will be added which will provide the PCI address in BDF format.  BDF format contains the address as: `domain:bus:device.function`.

This new device descriptor is only available for devices in the Level Zero platform, and the matching aspect is only true for those devices. The DPC++ default behavior is to expose GPU devices through the Level Zero platform.

**Note:** The environment variable SYCL\_ENABLE\_PCI must be set to 1 to obtain the PCI address.


## Version ##

All versions of the extension support this query.


## Device Information Descriptors ##

| Device Descriptors | Return Type | Description |
| ------------------ | ----------- | ----------- |
| info\:\:device\:\:ext\_intel\_pci\_address | std\:\:string | For Level Zero BE, returns the PCI address in BDF format: `domain:bus:device.function`.|


## Aspects ##

A new aspect, ext\_intel\_pci\_address, will be added.

## Error Condition ##

An invalid object runtime error will be thrown if the device does not support aspect\:\:ext\_intel\_pci\_address.


## Example Usage ##

The PCI address can be obtained using the standard get\_info() interface.

    if (dev.has(aspect::ext_intel_pci_address)) {
      auto BDF = dev.get_info<info::device::ext_intel_pci_address>();
    }



# Intel GPU Execution Unit SIMD Width #

A new device descriptor will be added which will provide the physical SIMD width of an execution unit on an Intel GPU.  This data will be used to calculate the computational capabilities of the device.

This new device descriptor is only available for devices in the Level Zero platform, and the matching aspect is only true for those devices. The DPC++ default behavior is to expose GPU devices through the Level Zero platform.


## Version ##

All versions of the extension support this query.


## Device Information Descriptors ##

| Device Descriptors | Return Type | Description |
| ------------------ | ----------- | ----------- |
| info\:\:device\:\:ext\_intel\_gpu\_eu\_simd\_width | uint32\_t| Returns the physical SIMD width of the  execution unit (EU).|


## Aspects ##

A new aspect, ext\_intel\_gpu\_eu\_simd\_width, will be added.


## Error Condition ##

An invalid object runtime error will be thrown if the device does not support aspect\:\:ext\_intel\_gpu\_eu\_simd\_width.

## Example Usage ##

The physical EU SIMD width can be obtained using the standard get\_info() interface.

    if (dev.has(aspect::ext_intel_gpu_eu_simd_width)) {
        auto euSimdWidth = dev.get_info<info::device::ext_intel_gpu_eu_simd_width>();
    }



# Intel GPU Execution Unit Count #

A new device descriptor will be added which will provide the number of execution units on an Intel GPU.  If the device is a subdevice, then the number of EUs in the subdevice is returned.

This new device descriptor will provide the same information as "max\_compute\_units" does today.  We would like to have an API which is specific for Intel GPUs.

This new device descriptor is only available for devices in the Level Zero platform, and the matching aspect is only true for those devices. The DPC++ default behavior is to expose GPU devices through the Level Zero platform.


## Version ##

All versions of the extension support this query.


## Device Information Descriptors ##

| Device Descriptors | Return Type | Description |
| ------------------ | ----------- | ----------- |
| info\:\:device\:\:ext\_intel\_gpu\__eu\_count | uint32\_t| Returns the number of execution units (EUs) associated with the Intel GPU.|


## Aspects ##

A new aspect, ext\_intel\_gpu\_eu\_count, will be added.


## Error Condition ##

An invalid object runtime error will be thrown if the device does not support aspect\:\:ext\_intel\_gpu\_eu\_count.

## Example Usage ##

Then the number of EUs can be obtained using the standard get\_info() interface.

    if (dev.has(aspect::ext_intel_gpu_eu_count)) {
      auto euCount = dev.get_info<info::device::ext_intel_gpu_eu_count>();
    }



# Intel GPU Number of Slices #

A new device descriptor will be added which will provide the number of slices on an Intel GPU.  If the device is a subdevice, then the number of slices in the subdevice is returned.


## Version ##

All versions of the extension support this query.


## Device Information Descriptors ##

| Device Descriptors | Return Type | Description |
| ------------------ | ----------- | ----------- |
| info\:\:device\:\:ext\_intel\_gpu\_slices | uint32\_t| Returns the number of slices.|


## Aspects ##

A new aspect, ext\_intel\_gpu\_slices, will be added.


## Error Condition ##

An invalid object runtime error will be thrown if the device does not support aspect\:\:ext\_intel\_gpu\_slices.

## Example Usage ##

Then the number of slices can be obtained using the standard get\_info() interface.

    if (dev.has(aspect::ext_intel_gpu_slices)) {
      auto slices = dev.get_info<info::device::ext_intel_gpu_slices>();
    }


# Intel GPU Number of Subslices per Slice #

A new device descriptor will be added which will provide the number of subslices per slice on an Intel GPU.  If the device is a subdevice, then the number of subslices per slice in the subdevice is returned.


## Version ##

All versions of the extension support this query.


## Device Information Descriptors ##

| Device Descriptors | Return Type | Description |
| ------------------ | ----------- | ----------- |
| info\:\:device\:\:ext\_intel\_gpu\_subslices\_per\_slice | uint32\_t| Returns the number of subslices per slice.|


## Aspects ##

A new aspect, ext\_intel\_gpu\_subslices\_per\_slice, will be added.


## Error Condition ##

An invalid object runtime error will be thrown if the device does not support aspect\:\:ext\_intel\_gpu\_subslices\_per\_slice.

## Example Usage ##

Then the number of subslices per slice can be obtained using the standard get\_info() interface.

    if (dev.has(aspect::ext_intel_gpu_subslices_per_slice)) {
      auto subslices = dev.get_info<info::device::ext_intel_gpu_subslices_per_slice>();
    }


# Intel GPU Number of Execution Units (EUs) per Subslice #

A new device descriptor will be added which will provide the number of EUs per subslice on an Intel GPU.  If the device is a subdevice, then the number of EUs per subslice in the subdevice is returned.


## Version ##

All versions of the extension support this query.


## Device Information Descriptors ##

| Device Descriptors | Return Type | Description |
| ------------------ | ----------- | ----------- |
| info\:\:device\:\:ext\_intel\_gpu\_eu\_count\_per\_subslice | uint32\_t| Returns the number of EUs in a subslice.|


## Aspects ##

A new aspect, ext\_intel\_gpu\_eu\_count\_per\_subslice, will be added.


## Error Condition ##

An invalid object runtime error will be thrown if the device does not support aspect\:\:ext\_intel\_gpu\_eu\_count\_per\_subslice.

## Example Usage ##

Then the number of EUs per subslice can be obtained using the standard get\_info() interface.

    if (dev.has(aspect::ext_intel_gpu_eu_count_per_subslice)) {
      auto euCount = dev.get_info<info::device::ext_intel_gpu_eu_count_per_subslice>();
    }

# Intel GPU Number of hardware threads per EU #

A new device descriptor will be added which will provide the number of hardware threads per EU on an Intel GPU. If the device is a subdevice, then the number of hardware threads per EU in the subdevice is returned.


## Version ##

The extension supports this query in version 3 and later.

## Device Information Descriptors ##

| Device Descriptors | Return Type | Description |
| ------------------ | ----------- | ----------- |
| info\:\:device\:\:ext\_intel\_gpu\_hw\_threads\_per\_eu | uint32\_t| Returns the number of hardware threads in EU.|


## Aspects ##

A new aspect, ext\_intel\_gpu\_hw\_threads\_per\_eu, will be added.


## Error Condition ##

An invalid object runtime error will be thrown if the device does not support aspect\:\:ext\_intel\_gpu\_hw\_threads\_per\_eu.

## Example Usage ##

Then the number of hardware threads per EU can be obtained using the standard get\_info() interface.

    if (dev.has(aspect::ext_intel_gpu_hw_threads_per_eu)) {
      auto threadsCount = dev.get_info<info::device::ext_intel_gpu_hw_threads_per_eu>();
    }

# Maximum Memory Bandwidth #

A new device descriptor will be added which will provide the maximum memory bandwidth.  If the device is a subdevice, then the maximum bandwidth of the subdevice is returned.

This new device descriptor is only available for devices in the Level Zero platform, and the matching aspect is only true for those devices. The DPC++ default behavior is to expose GPU devices through the Level Zero platform.


## Version ##

All versions of the extension support this query.


## Device Information Descriptors ##

| Device Descriptors | Return Type | Description |
| ------------------ | ----------- | ----------- |
| info\:\:device\:\:ext\_intel\_max\_mem\_bandwidth | uint64\_t| Returns the maximum memory bandwidth in units of bytes\/second.|


## Aspects ##

A new aspect, ext\_intel\_max\_mem\_bandwidth, will be added.


## Error Condition ##

An invalid object runtime error will be thrown if the device does not support aspect\:\:ext\_intel\_max\_mem\_bandwidth.


## Example Usage ##

Then the maximum memory bandwidth can be obtained using the standard get\_info() interface.

    if (dev.has(aspect::ext_intel_max_mem_bandwidth)) {
      auto maxBW = dev.get_info<info::device::ext_intel_max_mem_bandwidth>();
    }
