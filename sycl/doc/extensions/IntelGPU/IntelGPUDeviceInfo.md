# SYCL(TM) Proposal: New device descriptors for Intel GPUs

**IMPORTANT**: This specification is a draft.

**NOTE**: Khronos(R) is a registered trademark and SYCL(TM) is a trademark of the Khronos Group, Inc.

New device descriptors will be added to provide access to low-level hardware details about Intel GPU devices.  This information will be useful to developers tuning specifically for those devices.

This proposal details what is required to provide this information as a SYCL extensions.

## Feature Test Macro ##

The same Feature Test Macro will be used for the new device descriptor support.  It will be defined as:

    #define SYCL_EXT_INTEL_GPU_DEVICE_INFO 1



# Intel GPU PCI Address #

A new device descriptor will be added which will provide the PCI address of an Intel GPU in BDF format.  BDF format contains the address as: `domain:bus:device.function`.

This support can only be provided when using the Level Zero backend \(BE\).  This is the default for Intel GPUs.  The OpenCL BE does not provide the PCI address at this time.


## Device Information Descriptors ##

| Device Descriptors | Return Type | Description |
| ------------------ | ----------- | ----------- |
| info\:\:device\:\:ext\_intel\_gpu\_pci\_address | std\:\:string | For Level Zero BE, returns the PCI address in BDF format: `domain:bus:device.function`.|


## Aspects ##

A new aspect, ext\_intel\_gpu\_pci\_address, will be added.

## Error Condition ##

The function device\:\:get_info\(\) will return an empty string if the device does not support aspect\:\:ext\_intel\_gpu\_pci\_address.


## Example Usage ##

The PCI address can be obtained using the standard get\_info() interface.

    if (dev.has(aspect::ext_intel_gpu_pci_address) {
      auto BDF = dev.get_info<info::device::ext_intel_gpu_pci_address>();
    }



# Intel GPU Execution Unit SIMD Width #

A new device descriptor will be added which will provide the physical SIMD width of an execution unit on an Intel GPU.  This data will be used to calculate the computational capabilities of the device.

This support will only be provided when using the Level Zero backend \(BE\).  This is the default for Intel GPUs.  The OpenCL BE does not provide the physical SIMD width at this time.

## Device Information Descriptors ##

| Device Descriptors | Return Type | Description |
| ------------------ | ----------- | ----------- |
| info\:\:device\:\:ext\_intel\_gpu\_eu\_simd\_width | uint32\_t| Returns the physical SIMD width of the  execution unit (EU).|


## Aspects ##

A new aspect, ext\_intel\_gpu\_eu\_simd\_width, will be added.


## Error Condition ##

The function device\:\:get_info\(\) will return PI\_INVALID\_VALUE if the device does not support aspect\:\:ext\_intel\_gpu\_eu\_simd\_width.

## Example Usage ##

The physical EU SIMD width can be obtained using the standard get\_info() interface.

    if (dev.has(aspect::ext_intel_gpu_eu_simd_width) {
        auto euSimdWidth = dev.get_info<info::device::ext_intel_gpu_eu_simd_width>();
    }


# Intel GPU Execution Unit Count #

A new device descriptor will be added which will provide the number of execution units on an Intel GPU.  If the device is a subdevice, then the number of EUs in the subdevice is returned.

This new device descriptor will provide the same information as "max\_compute\_units" does today.  We would like to have an API which is specific for Intel GPUs.

This support will only be provided when using the Level Zero backend \(BE\).  This is the default for Intel GPUs.

## Device Information Descriptors ##

| Device Descriptors | Return Type | Description |
| ------------------ | ----------- | ----------- |
| info\:\:device\:\:ext\_intel\_gpu\__eu\_count | uint32\_t| Returns the number of execution units (EUs) associated with the Intel GPU.|


## Aspects ##

A new aspect, ext\_intel\_gpu\_eu\_count, will be added.


## Error Condition ##

The function device\:\:get_info\(\) will return PI\_INVALID\_VALUE if the device does not support aspect\:\:ext\_intel\_gpu\_eu\_count.

## Example Usage ##

Then the number of EUs can be obtained using the standard get\_info() interface.

    if (dev.has(aspect::ext_intel_gpu_eu_count) {
      auto euCount = dev.get_info<info::device::ext_intel_gpu_eu_count>();
    }
