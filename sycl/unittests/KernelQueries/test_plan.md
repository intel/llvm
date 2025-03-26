# Test plan for [`sycl ext intel kernel queries`][spec-link] extension

## Testing scope

### Device coverage

The unit tests should be launched on every supported device configuration we 
have.

### Type coverage

Each query is templated on a single template type argument `Param`
with some queries being template overloads of each other.
Each query has a restriction on the value of `Param` defined
in the relevant section of the spec.
Each `Param` defines a related sycl aspect that signals whether the device
supports it.

Param must be one of the following types defined in 
`sycl::ext::intel::info::kernel_device_specific` namespace:
- `spill_memory_size`

The tests should cover all of these types.

## Tests

### Unit tests

#### Interface tests

These tests are intended to check that all classes and methods defined by the
extension have correct implementation, i.e.: right signatures, right return
types, all necessary constraints are checked/enforced, etc.

These tests should check the following:

- that each query is not available when the template argument `Param` has
  a value different than the one in the spec.
- that each query can be called with the appropriate value for `Param` and the
  appropriate argument types as defined by its signature.
- the return types of all queries match the spec.

Tests in this category may not perform some useful actions to exercise the
extension functionality in full, but instead they are focused on making sure
that all APIs are consistent with respect to other APIs.

#### Check behavior in the case of unsupported aspects

Verigy that a synchronous `exception` with the error code
`errc::feature_not_supported` is thrown if an aspect is not supported by the
device.

[spec-link]: https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/supported/sycl_ext_intel_kernel_queries.asciidoc 
