# Design for Device-side Code Coverage

## Overview

This document describes the design and implementation of device-side code coverage for SYCL, extending Clang's source-based code coverage to support device code. The approach leverages the existing SYCL device global infrastructure, as detailed in the [DeviceGlobal.md](DeviceGlobal.md) design document, to enable collection and aggregation of coverage data from device kernels.

## Design Details

### Profiling Counter Representation

Profiling counters for code coverage are lowered by the compiler as device globals. Specifically, the `InstrProfilingLoweringPass` is modified so that, when targeting SPIR-V, coverage counters are represented as pointers to USM buffers, matching the representation of other SYCL device globals. This indirection allows counters to be relocatable and managed consistently with other device-side global variables.

Each counter is annotated with a unique identifier (`sycl-unique-id`) of the form `__profc_<fn_hash>`, where `<fn_hash>` is a 64-bit unsigned integer uniquely identifying the instrumented function. The counter's size is also recorded via the `sycl-device-global-size` attribute. These attributes ensure that counters are discoverable and manageable by the SYCL runtime and integration headers/footers.

The profile counter device global is represented as an array of 8-byte integers (`std::uint64_t`). The number of elements in this array corresponds to the number of regions in the function being instrumented, where a region typically represents a distinct code branch or block. The size of the device global variable is therefore determined by multiplying the number of regions by eight bytes, and this value is recorded in the `sycl-device-global-size` attribute for use by the runtime and integration logic.

### Integration with Device Global Infrastructure

The device global infrastructure, as described in [DeviceGlobal.md](DeviceGlobal.md), provides mechanisms for mapping host and device instances of global variables, managing their lifetimes, and facilitating data transfer. Device-side coverage counters are treated as a special class of device globals:

- They use the shared allocation type rather than the device allocation type for the underlying USM memory.
- They do not have corresponding `device_global` declarations in host code.
- Their lifetime and cleanup are managed via the device global map, with integration footer code ensuring registration and deregistration.

### Runtime Handling and Data Aggregation

When a device global entry corresponding to a coverage counter is released (e.g., when a device image is unloaded), the SYCL runtime aggregates the values from the device-side counter into the equivalent host-side counter. Equivalence is determined by matching both the `<fn_hash>` and the number of counter regions. If no matching host-side counter exists—typically due to differences in code between host and device caused by the `__SYCL_DEVICE_ONLY__` macro—the device-side counter values are discarded.

The aggregation is performed by invoking a new function in the compiler runtime, `__sycl_increment_profile_counters`, which is weakly linked to accommodate optional runtime availability. This function accepts the `<fn_hash>`, the number of regions, and the increment values, and updates the host-side counters accordingly. At program exit, the final profile data reflects the sum of host and device coverage counters.

### Compiler and Runtime Changes

#### Compiler Frontend

- The lowering pass for coverage counters is updated to emit device globals with the appropriate attributes and indirection.
- Integration headers and footers are updated to register device global counters with the runtime, using the unique identifier and size.

#### SYCL Runtime

- Device globals with IDs matching the `__profc_<fn_hash>` pattern are recognized as coverage counters.
- USM allocation and management for counters is handled as for other device globals, but without host-side declarations.
- Upon cleanup, device-side counter values are aggregated into host-side counters via the runtime API.

#### Compiler Runtime

- The new function `__sycl_increment_profile_counters` is introduced to update host-side counters.
- The function is weakly linked to allow for optional inclusion.

### Limitations and Considerations

- The feature is currently implemented only for SPIR-V targets; CUDA and HIP backends are not supported.
- Devices lacking support for device globals cannot utilize device-side code coverage.
- Differences in code between host and device (e.g., due to `__SYCL_DEVICE_ONLY__`) may prevent aggregation of coverage data for some functions.
- The design relies on the robustness of the device global infrastructure for correct mapping and lifetime management.

## Relationship to Device Global Design

This feature is built upon the mechanisms described in [DeviceGlobal.md](DeviceGlobal.md), including:

- Use of unique string identifiers (`sycl-unique-id`) for mapping and management.
- USM-based allocation and zero-initialization of device-side storage.
- Integration header/footer registration for host-device correlation.
- Runtime database for device global management and lookup.

The code coverage counters are a specialized use case of device globals, with additional logic for aggregation and profile generation.

## References

- [Implementation design for SYCL device globals](DeviceGlobal.md)
- [Clang Source-based Code Coverage](https://clang.llvm.org/docs/SourceBasedCodeCoverage.html)
- [SYCL Specification](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html)
