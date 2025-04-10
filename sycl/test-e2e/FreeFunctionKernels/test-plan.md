# Test plan for [`sycl_ext_oneapi_free_function_kernels`][spec-link] extension

## Testing scope

### Device coverage

The tests should be launched on every supported device configuration we have.

### Type coverage
New APIs for new way to define a kernel as a simple C++ function, where the kernel arguments are parameters to this function described by the extension can take only allowed kernel parameter type as specified in section 4.12.4 "Rules for parameter passing to kernels" of the core SYCL specification with exceptions to parameters of type `reducer` or `kernel_handler`. 

Therefore those tests should be structured in a way that checks are performed on all allowed kernel parameter types to ensure that everything works correctly.

## Tests

### Unit tests

Tests in this category may not fully exercise the extension functionality, but are instead they are focused on making sure that all APIs are consistent with respect to other APIs.


Perform tests on free function kernels requirments which should check the following:
 - that compiler will emit diagnostic when free function kernel is declared with reference types as parameters.
  
 - that compiler will emit diagnostic when free function kernel is declared with variadic arguments.

 - that compiler will emit diagnostic when free function kernel provides default parameter values.

 - that compiler will emit diagnostic when free function kernel return type is not `void`.

Perform tests on new traits for free function kernels which should check the following:
 - that `is_nd_range_kernel_v` trait returns true if function declaration is decorated with `nd_range_kernel` property and false if it is not.

 - that `is_single_task_kernel_v` trait returns true function if declaration is decorated with `single_task_kernel` and false if it is not.

- that `is_kernel_v` trait returns true for function whose declaration is decorated with either the `nd_range_kernel` property or the `single_task_kernel` property when it is not then it returns false.

Perform tests on new kernel bundle member functions for free function kernels which should check the following:

- that `get_kernel_id` member function returns valid kernel identifier that is associated with that kernel.

- that `get_kernel_bundle(const context& ctxt)` member function returns kernel bundle in which associated free function kernel can be found.

- that `get_kernel_bundle(const context& ctxt, const std::vector<device>& devs)` member function returns kernel bundle in which associated free function kernel can be found.

- that `has_kernel_bundle(const context& ctxt)` returns true when free function kernel can be represent in a device image of state and free function kernel is compatible with at least one of the devices in context.

- that `has_kernel_bundle(const context& ctxt, const std::vector<device>& devs)` returns true when free function kernel can be represent in a device image of state and free function kernel is compatible with at least one of the devices.

- that `has_kernel_bundle(const context& ctxt, const std::vector<device>& devs)` returns true when free function kernel can be represent in a device image of state and free function kernel is compatible with at least one of the devices.

- that `is_compatible(const device& dev)` returns true when free function kernel is compatible with the device.

- that `ext_oneapi_has_kernel()` returns true only if the kernel bundle contains the free function kernel.

- that `ext_oneapi_has_kernel(const device &dev)` returns true when kernel bundle contains the free function kernel and if that kernel is compatible with the device.

- that `ext_oneapi_get_kernel` returns the kernel object representing that kernel if the free function kernel resides in this kernel bundle. 

- that `ext_oneapi_get_kernel` throws exception with the error code `errc::invalid` if the free function kernel does not reside in this kernel bundle.


Perform tests on new free functions to query kernel information descriptors which should check the following:

- that `get_kernel_info(const context& ctxt)` produces the same result as would be computed by 
    ```
    auto bundle = sycl::get_kernel_bundle<Func, sycl::bundle_state::executable>(ctxt);
    auto ret = bundle.ext_oneapi_get_kernel<Func>().get_info<Param>()`
    ```

- that `get_kernel_info(const context& ctxt, const device& dev)` produces the same result as would be computed by 
    ```
    auto bundle =
      sycl::get_kernel_bundle<Func, sycl::bundle_state::executable>(ctxt);
    auto ret = bundle.ext_oneapi_get_kernel<Func>().get_info<Param>(dev);
    ```

- that ` get_kernel_info(const queue& q)` produces the same result as would be computed by 
    ```
    sycl::context ctxt = q.get_context();
    sycl::device dev = q.get_device();
    auto bundle =
    sycl::get_kernel_bundle<Func, sycl::bundle_state::executable>(ctxt);
    auto ret = bundle.ext_oneapi_get_kernel<Func>().get_info<Param>(dev);
    ```

### End-to-end tests

Tests in this category perform some meaningful actions with the extension to
see that the extension works in a scenarios which mimic real-life usage of the
extension.

With the exception of the`single_task_kernel` free function kernels, all subsequent tests are executed with Dimensions = 1, 2, 3. 

In all subsequent tests, free function kernels should be declared within a namespace, as static member functions of a class, or in the global namespace.

#### Test `accessor` as kernel parameter:
A series of tests should be performed that `accessor` is supported when
templated with `target::device`, inside free function kernel when passed as kernel parameter.

#### Test `USM` pointer as kernel parameter:
A series of checks should be performed that USM memory with three types of memory allocations `host`, `device` and `shared` is supported inside free function kernel when passed as kernel parameter.

#### Test `id` as kernel parameter:
A series of checks should be performed that we can pass `id` with different dimensions = 1, 2, 3 as kernel parameter to free function kernel and use it within kernel. 

#### Test `range` as kernel parameter:
A series of checks should be performed that we can pass `range` with different dimensions = 1, 2, 3 as kernel parameter to free function kernel and use it within kernel.

#### Test `marray<T, NumElements>` when `T` is device copyable as kernel parameter:
A series of checks should be performed that we can pass `marray<T, NumElements>` as kernel parameter to free function kernel and use it within kernel.

#### Test `vec<T, NumElements>` when `T` is device copyable as kernel parameter:
A series of checks should be performed that we can pass `vec<T, NumElements>` as kernel parameter to free function kernel and use it within kernel.

#### Test `sampled_image_accessor` as kernel parameter:
A series of checks should be performed that we can pass `sampled_image_accessor` as kernel parameter to free function kernel and use it within kernel.

#### Test `unsampled_image_accessor` as kernel parameter:
A series of checks should be performed that we can pass `unsampled_image_accessor` as kernel parameter to free function kernel and use it within kernel.

#### Test `local_accessor` as kernel parameter: 
A series of checks should be performed that we can pass `local_accessor` as kernel parameter to free function kernel and use it within kernel

#### Interaction with additional kernel properties
A series of checks should be performed to check that to the free function kernels may also be decorated with any of the properties defined in `sycl_ext_oneapi_kernel_properties`. This test should perform simple checks verifying if applied kernel_properties work within defined kernels.

#### Free function kernels compatibility with L0 backend
A series of checks should be performed to check compatibility of free function kernels with Level Zero Backend without going through the SYCL host runtime.

#### Free function kernels compatibility with OpenCL backend
A series of checks should be performed to check compatibility of free function kernels with OpenCL Backend without going through the SYCL host runtime.

[spec-link]: https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/proposed/sycl_ext_oneapi_free_function_kernels.asciidoc