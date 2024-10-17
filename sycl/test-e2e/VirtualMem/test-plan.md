# Test plan for [`sycl_ext_oneapi_virtual_mem`][spec-link] extension

## Testing scope

### Device coverage

Functionality provided by the extension is optional and it is only available if
corresponding aspect (`ext_oneapi_virtual_mem`) is supported by a device.

However, we still need to check that that we return a proper exception (with
`feature_not_supported` error code) if the extension is attempted to be used on
a device which doesn't support corresponding aspect.

Therefore, tests should be structured in a way that negative testing is
performed on all available devices, while positive tests can be skipped if
device doesn't support the extension.

### Type coverage

APIs for physical and virtual memory allocation are not dependent on types and
operate on opaque `void *` and `uintptr_t`. Therefore, there is no need for
tests to be dependent on data types and to be repeated for every different data
type there is.

## Tests

### Unit-tests

Tests in this category may not perform some useful actions to exercise the
extension functionality in full, but instead they are focused on making sure
that all APIs are consistent with respect to other APIs.

#### Memory granularity

The following checks should be performed for `get_mem_granularity` API:
- the returned value is always bigger than 0
- recommended granularity is not smaller than minimum granularity
- exception with `feature_not_supported` error code is thrown when `context`
  passed to it contains multiple devices and not every of them have
  `ext_oneapi_virtual_mem` aspect
- **TODO**: do we need to check that returned granularity is not bigger than
  maximum allocation size on a device?

#### Reserving virtual address ranges

A check should be performed that exception with `feature_not_supported` error
code is thrown when `context` passed to it contains multiple devices and not
every of them have `ext_oneapi_virtual_mem` aspect.

A check should be performed that we can successfully perform and immediately
release a valid reservation.

**TODO**: the spec currently says that passing any invalid arguments to those
APIs results in UB and therefore there is nothing to test. This is probably an
incorrect approach of the spec, because we should be able to perform some checks
and inform user about an error. If/when the spec is updated, this test plan
should be updated as well to cover those new scenarios.

#### Physical memory

A check should be performed that methods `get_context()`, `get_device()` and
`size()` return correct values (i.e. ones which were passed to `physical_mem`
constructor).

A check should be performed that a value returned from a valid call to `map()`
is the same as `reinterpret_cast<void *>(ptr)`.

**TODO**: the spec currently doesn't say what happens if `map` arguments are
invalid (misaligned `ptr`, or `offset + numBytes > size()`) and once that is
clarified the test plan should be updated to cover those scenarios as well.

A check should be performed that we can change access mode of a virtual memory
range and immediately see it changed.

A check should be performed that we can successfully map and immediately unmap
a virtual memory range.

### End-to-end tests

Tests in this category perform some meaningful actions with the extension to
see that the extension works in a scenarios which mimic real-life usage of the
extension.

#### Basic access from a kernel

A test should allocate some amount of a physical memory and then create a
virtual memory range mapped to that physical memory with access mode read-write.
Test should submit two kernels:
- the first kernel fills virtual memory with some data
- the second kernel reads from that virtual memory and copies it content into
  an accessor

Before the second kernel is submitted, we should change access mode of the
virtual memory range to be read-only.

The content of the accessor is then checked on host to ensure that data can be
correctly both written and read from virtual memory.

#### Ability to use virtual memory as USM memory

This test case is intended to check that a pointer produced by a virtual memory
range mapping can indeed be used in various APIs accepting a USM pointer.

A series of checks should be performed:
- that we can copy _from_ virtual memory _to_ a buffer via accessor
- that we can copy _from_ virtual memory _to_ a USM allocation
- that we can copy _from_ a USM allocation _to_ virtual memory
- that we can copy _from_ a buffer via accessor _to_ virtual memory
- that we can use `memset` on virtual memory
- that we can use `fill` on virtual memory

#### Remapping of virtual memory range

This test case is intended to check that we can correctly access virtual memory
range even if it was re-mapped to a different physical range.

A test should allocate some physical memory and create a virtual memory range
mapped to it. Then, a kernel should be submitted which will fill that virtual
memory range with data.
Once the kernel is complete, another physical memory region should be allocated
and existing virtual memory range should be re-mapped to it. Another kernel
should be submitted to fill that virtual memory range once again with some new
data (different from the data used in the first kernel).
After that we should check that virtual memory range contains the right data
(put there by the second kernel).

#### "Extending" virtual memory range

This test case is intended to check that memory accesses to contiguous
virtual memory ranges are performed correctly.

A test reserve a number of virtual memory ranges which comprsise a contiguous
memory range. Each virtual memory range should be mapped to separate physical
memory range.

Then this single huge virtual memory range (consisting of several smaller
ranges) should be used to perform various operations, like copying from/to that
range and accessing it from a kernel.

[spec-link]: https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_virtual_mem.asciidoc
