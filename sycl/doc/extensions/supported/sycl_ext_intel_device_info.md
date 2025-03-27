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
| 4     | Free device memory query is supported |
| 5     | Device ID is supported |
| 6     | Memory clock rate and bus width queries are supported |
| 7     | Throttle reasons, fan speed and power limits queries are supported |



# Device ID #

A new device descriptor will be added which will provide the device ID.

If the implementation is driven primarily by a PCI device with a PCI device ID, the device ID must be that PCI device ID. Otherwise, the choice of what to return may be dictated by operating system or platform policies - but should uniquely identify both the device version and any major configuration options (for example, core count in the case of multi-core devices).

## Version ##

The extension supports this query in version 5 and later.

## Device Information Descriptors ##

| Device Descriptors | Return Type | Description |
| ------------------ | ----------- | ----------- |
| ext\:\:intel\:\:info\:\:device\:\:device\_id | uint32\_t| Returns the device ID.|


## Aspects ##

A new aspect, ext\_intel\_device\_id, will be added.


## Error Condition ##

Throws a synchronous `exception` with the `errc::feature_not_supported` error code if the device does not have `aspect::ext_intel_device_id`.

## Example Usage ##

The device ID can be obtained using the standard get\_info() interface.

    if (dev.has(aspect::ext_intel_device_id)) {
      auto deviceID = dev.get_info<ext::intel::info::device::device_id>();
    }



# Device UUID #

A new device descriptor will be added which will provide the device Universal Unique ID (UUID).

The DPC++ default behavior would be to expose the UUIDs of all supported devices which enables detection of total number of unique devices.


## Version ##

The extension supports this query in version 2 and later.


## Device Information Descriptors ##

| Device Descriptors | Return Type | Description |
| ------------------ | ----------- | ----------- |
| ext\:\:intel\:\:info\:\:device\:\:uuid | std::array<unsigned char, 16> | Returns the device UUID|


## Aspects ##

A new aspect, ext\_intel\_device\_info\_uuid, will be added.

## Error Condition ##

Throws a synchronous `exception` with the `errc::feature_not_supported` error code if the device does not have `aspect::ext_intel_device_info_uuid`.


## Example Usage ##

The UUID can be obtained using the standard get\_info() interface.

    if (dev.has(aspect::ext_intel_device_info_uuid)) {
      auto UUID = dev.get_info<ext::intel::info::device::uuid>();
    }



# PCI Address #

A new device descriptor will be added which will provide the PCI address in BDF format.  BDF format contains the address as: `domain:bus:device.function`.

## Version ##

All versions of the extension support this query.


## Device Information Descriptors ##

| Device Descriptors | Return Type | Description |
| ------------------ | ----------- | ----------- |
| ext\:\:intel\:\:info\:\:device\:\:pci\_address | std\:\:string | For Level Zero BE, returns the PCI address in BDF format: `domain:bus:device.function`.|


## Aspects ##

A new aspect, ext\_intel\_pci\_address, will be added.

## Error Condition ##

Throws a synchronous `exception` with the `errc::feature_not_supported` error code if the device does not have `aspect::ext_intel_pci_address`.


## Example Usage ##

The PCI address can be obtained using the standard get\_info() interface.

    if (dev.has(aspect::ext_intel_pci_address)) {
      auto BDF = dev.get_info<ext::intel::info::device::pci_address>();
    }



# Intel GPU Execution Unit SIMD Width #

A new device descriptor will be added which will provide the physical SIMD width of an execution unit on an Intel GPU.  This data will be used to calculate the computational capabilities of the device.


## Version ##

All versions of the extension support this query.


## Device Information Descriptors ##

| Device Descriptors | Return Type | Description |
| ------------------ | ----------- | ----------- |
| ext\:\:intel\:\:info\:\:device\:\:gpu\_eu\_simd\_width | uint32\_t| Returns the physical SIMD width of the  execution unit (EU).|


## Aspects ##

A new aspect, ext\_intel\_gpu\_eu\_simd\_width, will be added.


## Error Condition ##

Throws a synchronous `exception` with the `errc::feature_not_supported` error code if the device does not have `aspect::ext_intel_gpu_eu_simd_width`.

## Example Usage ##

The physical EU SIMD width can be obtained using the standard get\_info() interface.

    if (dev.has(aspect::ext_intel_gpu_eu_simd_width)) {
        auto euSimdWidth = dev.get_info<ext::intel::info::device::gpu_eu_simd_width>();
    }



# Intel GPU Execution Unit Count #

A new device descriptor will be added which will provide the number of execution units on an Intel GPU.  If the device is a subdevice, then the number of EUs in the subdevice is returned.

This new device descriptor will provide the same information as "max\_compute\_units" does today.  We would like to have an API which is specific for Intel GPUs.


## Version ##

All versions of the extension support this query.


## Device Information Descriptors ##

| Device Descriptors | Return Type | Description |
| ------------------ | ----------- | ----------- |
| ext\:\:intel\:\:info\:\:device\:\:gpu\__eu\_count | uint32\_t| Returns the number of execution units (EUs) associated with the Intel GPU.|


## Aspects ##

A new aspect, ext\_intel\_gpu\_eu\_count, will be added.


## Error Condition ##

Throws a synchronous `exception` with the `errc::feature_not_supported` error code if the device does not have `aspect::ext_intel_gpu_eu_count`.

## Example Usage ##

Then the number of EUs can be obtained using the standard get\_info() interface.

    if (dev.has(aspect::ext_intel_gpu_eu_count)) {
      auto euCount = dev.get_info<ext::intel::info::device::gpu_eu_count>();
    }



# Intel GPU Number of Slices #

A new device descriptor will be added which will provide the number of slices on an Intel GPU.  If the device is a subdevice, then the number of slices in the subdevice is returned.


## Version ##

All versions of the extension support this query.


## Device Information Descriptors ##

| Device Descriptors | Return Type | Description |
| ------------------ | ----------- | ----------- |
| ext\:\:intel\:\:info\:\:device\:\:gpu\_slices | uint32\_t| Returns the number of slices.|


## Aspects ##

A new aspect, ext\_intel\_gpu\_slices, will be added.


## Error Condition ##

Throws a synchronous `exception` with the `errc::feature_not_supported` error code if the device does not have `aspect::ext_intel_gpu_slices`.

## Example Usage ##

Then the number of slices can be obtained using the standard get\_info() interface.

    if (dev.has(aspect::ext_intel_gpu_slices)) {
      auto slices = dev.get_info<ext::intel::info::device::gpu_slices>();
    }


# Intel GPU Number of Subslices per Slice #

A new device descriptor will be added which will provide the number of subslices per slice on an Intel GPU.  If the device is a subdevice, then the number of subslices per slice in the subdevice is returned.


## Version ##

All versions of the extension support this query.


## Device Information Descriptors ##

| Device Descriptors | Return Type | Description |
| ------------------ | ----------- | ----------- |
| ext\:\:intel\:\:info\:\:device\:\:gpu\_subslices\_per\_slice | uint32\_t| Returns the number of subslices per slice.|


## Aspects ##

A new aspect, ext\_intel\_gpu\_subslices\_per\_slice, will be added.


## Error Condition ##

Throws a synchronous `exception` with the `errc::feature_not_supported` error code if the device does not have `aspect::ext_intel_gpu_subslices_per_slice`.

## Example Usage ##

Then the number of subslices per slice can be obtained using the standard get\_info() interface.

    if (dev.has(aspect::ext_intel_gpu_subslices_per_slice)) {
      auto subslices = dev.get_info<ext::intel::info::device::gpu_subslices_per_slice>();
    }


# Intel GPU Number of Execution Units (EUs) per Subslice #

A new device descriptor will be added which will provide the number of EUs per subslice on an Intel GPU.  If the device is a subdevice, then the number of EUs per subslice in the subdevice is returned.


## Version ##

All versions of the extension support this query.


## Device Information Descriptors ##

| Device Descriptors | Return Type | Description |
| ------------------ | ----------- | ----------- |
| ext\:\:intel\:\:info\:\:device\:\:gpu\_eu\_count\_per\_subslice | uint32\_t| Returns the number of EUs in a subslice.|


## Aspects ##

A new aspect, ext\_intel\_gpu\_eu\_count\_per\_subslice, will be added.


## Error Condition ##

Throws a synchronous `exception` with the `errc::feature_not_supported` error code if the device does not have `aspect::ext_intel_gpu_eu_count_per_subslice`.

## Example Usage ##

Then the number of EUs per subslice can be obtained using the standard get\_info() interface.

    if (dev.has(aspect::ext_intel_gpu_eu_count_per_subslice)) {
      auto euCount = dev.get_info<ext::intel::info::device::gpu_eu_count_per_subslice>();
    }

# Intel GPU Number of hardware threads per EU #

A new device descriptor will be added which will provide the number of hardware threads per EU on an Intel GPU. If the device is a subdevice, then the number of hardware threads per EU in the subdevice is returned.


## Version ##

The extension supports this query in version 3 and later.

## Device Information Descriptors ##

| Device Descriptors | Return Type | Description |
| ------------------ | ----------- | ----------- |
| ext\:\:intel\:\:info\:\:device\:\:gpu\_hw\_threads\_per\_eu | uint32\_t| Returns the number of hardware threads in EU.|


## Aspects ##

A new aspect, ext\_intel\_gpu\_hw\_threads\_per\_eu, will be added.


## Error Condition ##

Throws a synchronous `exception` with the `errc::feature_not_supported` error code if the device does not have `aspect::ext_intel_gpu_hw_threads_per_eu`.

## Example Usage ##

Then the number of hardware threads per EU can be obtained using the standard get\_info() interface.

    if (dev.has(aspect::ext_intel_gpu_hw_threads_per_eu)) {
      auto threadsCount = dev.get_info<ext::intel::info::device::gpu_hw_threads_per_eu>();
    }

# Maximum Memory Bandwidth #

A new device descriptor will be added which will provide the maximum memory bandwidth.  If the device is a subdevice, then the maximum bandwidth of the subdevice is returned.


## Version ##

All versions of the extension support this query.


## Device Information Descriptors ##

| Device Descriptors | Return Type | Description |
| ------------------ | ----------- | ----------- |
| ext\:\:intel\:\:info\:\:device\:\:max\_mem\_bandwidth | uint64\_t| Returns the maximum memory bandwidth in units of bytes\/second.|


## Aspects ##

A new aspect, ext\_intel\_max\_mem\_bandwidth, will be added.


## Error Condition ##

Throws a synchronous `exception` with the `errc::feature_not_supported` error code if the device does not have `aspect::ext_intel_max_mem_bandwidth`.


## Example Usage ##

Then the maximum memory bandwidth can be obtained using the standard get\_info() interface.

    if (dev.has(aspect::ext_intel_max_mem_bandwidth)) {
      auto maxBW = dev.get_info<ext::intel::info::device::max_mem_bandwidth>();
    }

# Free Global Memory #

A new device descriptor will be added which will provide the number of bytes of free global memory for the device.
The amount of free global memory may be affected by other processes on the
system that are also using this device.
Beware that when other processes or threads are using this device when this call
is made, the value it returns may be stale even before it is returned to the
caller.


## Version ##

The extension supports this query in version 4 and later.


## Device Information Descriptors ##

| Device Descriptors | Return Type | Description |
| ------------------ | ----------- | ----------- |
| ext\:\:intel\:\:info\:\:device\:\:free\_memory | uint64\_t| Returns the memory avialble on the device in units of bytes.|


## Aspects ##

A new aspect, ext\_intel\_free\_memory, will be added.


## Error Condition ##

Throws a synchronous `exception` with the `errc::feature_not_supported` error code if the device does not have `aspect::ext_intel_free_memory`.


## Example Usage ##

Then the free device memory  can be obtained using the standard get\_info() interface.

    if (dev.has(aspect::ext_intel_free_memory)) {
      auto FreeMemory = dev.get_info<ext::intel::info::device::free_memory>();
    }


# Memory Clock Rate #

A new device descriptor is added which provides the maximum clock rate of device's global memory.


## Version ##

The extension supports this query in version 6 and later.


## Device Information Descriptors ##

| Device Descriptors | Return Type | Description |
| ------------------ | ----------- | ----------- |
| ext\:\:intel\:\:info\:\:device\:\:memory\_clock\_rate | uint32\_t| Returns the maximum clock rate of device's global memory in MHz. If device doesn't have memory then returns 0. If there are several memories on the device then the minimum of the clock rate values is returned. |


## Aspects ##

A new aspect, ext\_intel\_memory\_clock\_rate, is added.


## Error Condition ##

Throws a synchronous `exception` with the `errc::feature_not_supported` error code if the device does not have `aspect::ext_intel_memory_clock_rate`.


## Example Usage ##

Then the memory clock rate can be obtained using the standard get\_info() interface.

    if (dev.has(aspect::ext_intel_memory_clock_rate)) {
      auto MemoryClockRate = dev.get_info<ext::intel::info::device::memory_clock_rate>();
    }


# Memory Bus Width #

A new device descriptor is added which provides the maximum bus width between device and memory.


## Version ##

The extension supports this query in version 6 and later.


## Device Information Descriptors ##

| Device Descriptors | Return Type | Description |
| ------------------ | ----------- | ----------- |
| ext\:\:intel\:\:info\:\:device\:\:memory\_bus\_width | uint32\_t| Returns the maximum bus width between device and memory in bits. If device doesn't have memory then returns 0. If there are several memories on the device then the minimum of the bus width values is returned. |


## Aspects ##

A new aspect, ext\_intel\_memory\_bus\_width, is added.


## Error Condition ##

Throws a synchronous `exception` with the `errc::feature_not_supported` error code if the device does not have `aspect::ext_intel_memory_bus_width`.


## Example Usage ##

Then the memory bus width can be obtained using the standard get\_info() interface.

    if (dev.has(aspect::ext_intel_memory_bus_width)) {
      auto MemoryBusWidth = dev.get_info<ext::intel::info::device::memory_bus_width>();
    }

# Throttle reason #

A new device descriptor is added which provides the current clock throttle reasons.
A new enum is added with the list of possible throttle reasons.

## Version ##

The extension supports this query in version 7 and later.

## Throttle reasons ##

| Reason             | Description |
| ------------------ | ----------- |
| `power_cap` | The clock frequency is throttled due to hitting the power limit. |
| `current_limit` | The clock frequency is throttled due to hitting the current limit. |
| `thermal_limit` | The clock frequency is throttled due to hitting the thermal limit. |
| `psu_alert` | The clock frequency is throttled due to power supply assertion. |
| `sw_range` | The clock frequency is throttled due to software supplied frequency range. |
| `hw_range` | The clock frequency is throttled because there is a sub block that has a lower frequency when it receives clocks. |
| `other` | The clock frequency is throttled due to other reason. |


```
namespace sycl::ext::intel {

  enum class throttle_reason {
    power_cap,
    current_limit,
    thermal_limit,
    psu_alert,
    sw_range,
    hw_range,
    other
  }

}
```

## Device Information Descriptors ##

| Device Descriptors | Return Type | Description |
| ------------------ | ----------- | ----------- |
| `ext::intel::info::device::current_clock_throttle_reasons` | `std::vector<ext::intel::throttle_reason>` | Returns the set of throttle reasons describing why the frequency is being limited by the hardware. Returns empty set if frequency is not throttled. |


## Aspects ##

A new aspect, `ext_intel_current_clock_throttle_reasons`, is added.


## Error Condition ##

Throws a synchronous `exception` with the `errc::feature_not_supported` error code if the device does not have `aspect::ext_intel_current_clock_throttle_reasons`.

## Example Usage ##

Then the current clock throttle reasons can be obtained using the standard `get_info()` interface.

```
if (dev.has(aspect::ext_intel_current_clock_throttle_reasons)) {
  std::vector<ext::inte::info::throttle_reason> Reasons = dev.get_info<ext::intel::info::device::current_clock_throttle_reasons<>();
}
```


# Fan speed #

A new device descriptor is added which provides the fan speed for the device.

## Version ##

The extension supports this query in version 7 and later.

## Device Information Descriptors ##

| Device Descriptors | Return Type | Description |
| ------------------ | ----------- | ----------- |
| `ext::intel::info::device::fan_speed` | `int32_t` | Returns the current speed of device's fan (as a percentage of the maximum speed of the fan). If fan speed can't be measured then returns -1. If there are multiple fans, then returns maximum value. |


## Aspects ##

A new aspect, `ext_intel_fan_speed`, is added.


## Error Condition ##

Throws a synchronous `exception` with the `errc::feature_not_supported` error code if the device does not have `aspect::ext_intel_fan_speed`.

## Example Usage ##

Then the fan speed can be obtained using the standard `get_info()` interface.

```
    if (dev.has(aspect::ext_intel_fan_speed)) {
      auto FanSpeed = dev.get_info<ext::intel::info::device::fan_speed>();
    }
```

# Power limits #

New device descriptors are added which provide the maximum and minimum power limits for the device.

## Version ##

The extension supports this query in version 7 and later.

## Device Information Descriptors ##

| Device Descriptors | Return Type | Description |
| ------------------ | ----------- | ----------- |
|`ext::intel::info::device::min_power_limit` |`int32_t` | Returns the minimum power limit of the device in milliwatts. Returns -1 if the limit is not known. |
|`ext::intel::info::device::max_power_limit` |`int32_t` | Returns the maximum power limit of the device in milliwatts. Returns -1 if the limit is not known. |


## Aspects ##

A new aspect, `ext_intel_power_limits`, is added.


## Error Condition ##

Throws a synchronous `exception` with the `errc::feature_not_supported` error code if the device does not have `aspect::ext_intel_power_limits`.

## Example Usage ##

Then the power limits can be obtained using the standard `get_info()` interface.

```
    if (dev.has(aspect::ext_intel_power_limits)) {
      auto Min = dev.get_info<ext::intel::info::device::min_power_limit>();
      auto Max = dev.get_info<ext::intel::info::device::max_power_limit>();
    }
```




# Deprecated queries #

The table below lists deprecated, that would soon be removed and their replacements:

|Deprecated Descriptors | Replacement Descriptors |
| ------------------------------- |--------------------------- |
| info\:\:device\:\:ext\_intel\_device\_info\_uuid  | ext\:\:intel\:\:info\:\:device\:\:uuid |
| info\:\:device\:\:ext\_intel\_pci\_address        | ext\:\:intel\:\:info\:\:device\:\:pci\_address |
| info\:\:device\:\:ext\_intel\_gpu\_eu\_simd\_width  | ext\:\:intel\:\:info\:\:device\:\:gpu\_eu\_simd\_width |
| info\:\:device\:\:ext\_intel\_gpu\__eu\_count       | ext\:\:intel\:\:info\:\:device\:\:gpu\__eu\_count      |
| info\:\:device\:\:ext\_intel\_gpu\_slices               | ext\:\:intel\:\:info\:\:device\:\:gpu\_slices      |
| info\:\:device\:\:ext\_intel\_gpu\_subslices\_per\_slice | ext\:\:intel\:\:info\:\:device\:\:gpu\_subslices\_per\_slice    |
|info\:\:device\:\:ext\_intel\_gpu\_eu\_count\_per\_subslice | ext\:\:intel\:\:info\:\:device\:\:gpu\_eu\_count\_per\_subslice |
| info\:\:device\:\:ext\_intel\_gpu\_hw\_threads\_per\_eu    | ext\:\:intel\:\:info\:\:device\:\:gpu\_hw\_threads\_per\_eu     |
| info\:\:device\:\:ext\_intel\_max\_mem\_bandwidth | ext\:\:intel\:\:info\:\:device\:\:max\_mem\_bandwidth |
