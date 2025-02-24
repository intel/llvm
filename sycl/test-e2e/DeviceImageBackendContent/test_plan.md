# Test plan for [`sycl_ext_oneapi_device_image_backend_content`][spec-link] 
# extension

## Testing scope

### Device coverage

The unit tests should be launched on every supported device configuration we
have.

The end-to-end tests, which consist of checking the interoperability with 
Level Zero, OpenCL and CUDA backends, should be run on devices that are 
exposed through these low-level interfaces.

### Type coverage

All of the APIs introduced by this extension are not templated and do not have
any arguments. 

There are, however, some requirements related to the value of the 
`State` template parameter of the `device_image` class on which these 
APIs are defined. In particular, the `ext_oneapi_get_backend_content` 
and `ext_oneapi_get_backend_content_view` functions are only 
available when `State == sycl::bundle_state::executable`. 
Tests should verify that these functions are indeed only 
available when this equality occurs.

## Tests

### Unit tests

#### Interface tests

These tests are intended to check that all classes and methods defined by the
extension have correct implementation, i.e.: right signatures, right return
types, all necessary constraints are checked/enforced, etc.

These tests should check the following:

- that diagnostic is emitted when `ext_oneapi_get_backend_content` or
  `ext_oneapi_get_backend_content_view` functions are called and 
  `State != sycl::bundle_state::executable`
- the return types of all functions match the spec

Tests in this category may not perform some useful actions to exercise the
extension functionality in full, but instead they are focused on making sure
that all APIs are consistent with respect to other APIs.

#### Consistency of backend API

The test needs to check that `ext_oneapi_get_backend` returns the 
same value as `sycl::kernel_bundle::get_backend` on the kernel bundle
that contains the image.

#### Consistency of image contents and a view of the image contents

The test needs to check that the values returned by 
`ext_oneapi_get_backend_content` and `ext_oneapi_get_backend_content_view` 
have the same contents.

### End-to-end tests

Tests in this category perform some meaningful actions with the extension to
see that the extension works in scenarios which mimic real-life usage of the
extension.

#### Level Zero and OpenCL interoperability

In general, it is not possible to interpret the device image contents returned
 by this API in such a way that allows us to retrieve specific kernels inside
the device image. However, under the conditions that the device be managed by 
an Level Zero or OpenCL backend and the kernel is defined using the
requirements [here][ref-link], it is possible to retrieve the kernel from the
 device image contents.

This test selects a Level Zero or OpenCL backend device and defines a simple
kernel using free function syntax adhering to the requirements mentioned in
the paragraph above. Then, using the device image contents and backend
specific API, it should create a Level Zero or OpenCL kernel object
corresponding to the kernel. Using interoperability API such as
`sycl::make_kernel`, this kernel object can be made into a high-level SYCL
kernel object. The test, therefore, has two versions of the same SYCL
kernel, one of which is constructed directly from the source code kernel
using only SYCL API and the other constructed using the
`sycl_ext_oneapi_device_image_backend_content` API, backend specific API and
the SYCL interoperability API. The test should run both kernels and verify
that they have the same effect, for example, by having the kernel write a
specific value to a memory location passed as a kernel argument and
checking that after both kernels have run, both memory locations have the
same value.

The test requires either Level Zero or OpenCL backend and development kits
to be available in the testing environment.

### CUDA interoperability

Unlike above, for CUDA, there is not a portable way to retrieve a SYCL
kernel from a CUBIN module. This test, therefore, will simply get the
contents of the CUBIN module and use `cuModuleLoadData` to create a module
object out of the image contents and verify that `CUDA_SUCCESS` is returned.

The test requires CUDA backend and development kit to be available in the
testing environment.

[ref-link]: ../proposed/sycl_ext_oneapi_free_function_kernels.asciidoc#level-zero-and-opencl-compatibility
[spec-link]: https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/proposed/sycl_ext_oneapi_device_image_backend_content.asciidoc
