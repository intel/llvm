# Test plan for [`sycl_ext_oneapi_free_function_kernels`][spec-link] extension

## Testing scope

### Device coverage

The tests should be launched on every supported device configuration we have.

### Type coverage
New APIs for new way to define a kernel as a simple C++ function, 
where the kernel arguments are parameters to this function described by 
the extension can take only allowed kernel parameter type as specified in 
section [4.12.4 of the SYCL 2020 specification](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:kernel.parameter.passing) 
with exceptions to parameters of type `reducer` or `kernel_handler`. 

Therefore those tests should be structured in a way that checks are performed on 
set of types that satisfy the various requirements for kernel arguments to 
ensure that everything works correctly.

## Tests

### Unit tests

Tests in this category may not fully exercise the extension functionality, but 
are instead focused on making sure that all APIs are consistent with respect to 
existing APIs.


#### Perform tests on free function kernels requirements which should check that:
 - the compiler will emit diagnostic when free function kernel is declared with 
 reference types as parameters.
 - the compiler will emit diagnostic when free function kernel is declared with 
 variadic arguments.
 - the compiler will emit diagnostic when free function kernel provides 
 default parameter values.
 - the compiler will emit diagnostic when free function kernel return type 
 is not `void`.
 - the compiler will emit diagnostic when a non-static member function is used 
 as a kernel. Only static member function at class scope are allowed as 
 free function kernel.

#### Perform tests on free function kernel declaration with properties `nd_range_kernel` and `single_task_kernel` which should check the following:
 - that if the property does not appear on the first declaration of the 
function in the translation unit, it will result in a compilation error.
 - that if the property appears on the first declaration of the function, and 
the following redeclarations do not have this property, it will not result in 
a compilation error.
 - that if a function is decorated with more than one of these properties, it 
will result in a compilation error.
 - that if a redeclaration of a function is decorated with the same property 
but with different arguments, the program should result in a compilation error.

#### Perform tests on new traits for free function kernels which should check the following:
- that `is_nd_range_kernel_v` trait should be a subclass of `true_type` if 
function declaration is decorated with `nd_range_kernel` property or a 
subclass of `false_type` if it is not.
- that `is_single_task_kernel_v` trait should be a subclass of `true_type` if 
declaration is decorated with `single_task_kernel` or a subclass of `false_type`
if it is not.
- that `is_kernel_v` trait should be a subclass of `true_type` for function 
whose declaration is decorated with either the `nd_range_kernel` property or 
the `single_task_kernel` property when it is not then it should be a subclass 
of `false_type`.

#### Perform tests on new `kernel_bundle` member functions for free function kernels by declaring `nd_range_kernel` and `single_task_kernel` and verifying that:

- the `get_kernel_id` member function returns a valid kernel identifier 
associated with free function kernel which can be found within all kernel 
identifiers for any free function kernels defined by the application which can 
be queried using `get_kernel_ids()`.
- the `get_kernel_bundle(const context& ctxt)` member function returns a 
kernel bundle that contains the corresponding free function kernel.
- the `get_kernel_bundle(const context& ctxt, const std::vector<device>& devs)` 
member function returns a kernel bundle that contains the corresponding 
free function kernel.
- the `has_kernel_bundle(const context& ctxt)` returns true when a free function 
kernel can be represented in a device image in the corresponding state and the 
associated free function kernel is compatible with at least one of the devices 
in `ctxt`.
- the `has_kernel_bundle(const context& ctxt, const std::vector<device>& devs)` 
returns true when declared free function kernel can be represented in a device 
image in the corresponding state and that free function kernel is compatible 
with at least one of the devices in `devs`.
- the `is_compatible(const device& dev)` returns true when the associated free 
function kernel is compatible with `dev`.
- the `ext_oneapi_has_kernel()` returns true only if the kernel bundle contains 
the associated free function kernel.
- the `ext_oneapi_has_kernel(const device &dev)` returns true when kernel 
bundle contains the associated free function kernel and if that kernel is 
compatible with `dev`.
- the `ext_oneapi_get_kernel` returns the kernel object representing that 
kernel if the free function kernel resides in this kernel bundle. 
- the `ext_oneapi_get_kernel` throws exception with the error code 
`errc::invalid` if the associated free function kernel does not reside in this 
kernel bundle.
- the `get_kernel_ids()` returns all of the kernels defined in the source, 
whether they were defined as free function kernels, lambda expressions or 
named kernel objects.
- the `info::kernel::num_args` returns the number of parameters in the function 
definition of the associated free function kernel.

Write test that perform all the checks mentioned above on `nd_range_kernel` 
and `single_task_kernel` free functions, which are declared in one translation 
unit and defined in another.

#### Perform tests on new free functions to query kernel information descriptors which should check the following:

- that `get_kernel_info(const context& ctxt)` produces the same result as would 
be computed by 
  ```
  auto bundle = sycl::get_kernel_bundle<Func, sycl::bundle_state::executable>(ctxt);
  auto ret = bundle.ext_oneapi_get_kernel<Func>().get_info<Param>();
  ```

- that `get_kernel_info(const context& ctxt, const device& dev)` produces the 
same result as would be computed by 
  ```
  auto bundle =
    sycl::get_kernel_bundle<Func, sycl::bundle_state::executable>(ctxt);
  auto ret = bundle.ext_oneapi_get_kernel<Func>().get_info<Param>(dev);
  ```

- that ` get_kernel_info(const queue& q)` produces the same result as would 
be computed by 
  ```
  sycl::context ctxt = q.get_context();
  sycl::device dev = q.get_device();
  auto bundle =
    sycl::get_kernel_bundle<Func, sycl::bundle_state::executable>(ctxt);
  auto ret = bundle.ext_oneapi_get_kernel<Func>().get_info<Param>(dev);
  ```

#### Perform tests on use of illegal types for kernel paramters:
- that a class type `S` with a virtual base class of type `T` can not be used 
as free function kernel parameter type.
- that a class type `S` with a virtual member function can not be used as free 
function kernel parameter type.

#### Test type aliases to allowed kernel paramater types as kernel parameter:
A series of checks should be performed that we can pass type aliases to allowed 
kernel paramater types as kernel parameter and use it within kernel.

### End-to-end tests

Tests in this category perform some meaningful actions with the extension to
see that the extension works in a scenarios which mimic real-life usage of the
extension.

With the exception of the `single_task_kernel` free function kernels, all 
subsequent tests are executed with `Dimensions` $$\in \{1, 2, 3\}$$.

In all subsequent tests, free function kernels should be declared within a 
namespace, as static member functions of a class, or in the global namespace.

#### Perform test that free function kernel can be used as device function within another kernel:
A series of checks should be performed that free function kernel can be used 
within another kernel and behave as device function.

#### Perform test that free function kernel can be used as normal host function:
A series of checks should be performed that free function kernel can be used 
within another host function and behave as regular host function.

#### Test `accessor` as kernel parameter:
A series of tests should be performed that `accessor` is supported when
templated with `target::device`, inside free function kernel when passed as 
kernel parameter.

#### Test `USM` pointer as kernel parameter:
A series of checks should be performed that USM memory with three types of 
memory allocations `host`, `device` and `shared` is supported inside 
free function kernel when passed as kernel parameter.

#### Test `id` as kernel parameter:
A series of checks should be performed that we can pass `id<Dimensions>` where 
`Dimensions` is in $$\in \{1, 2, 3\}$$ as kernel parameter to free function 
kernel and use it within kernel. 

#### Test `range` as kernel parameter:
A series of checks should be performed that we can pass `range` where 
`Dimensions` is in $$\in \{1, 2, 3\}$$ as kernel parameter to free function 
kernel and use it within kernel.

#### Test `marray<T, NumElements>` when `T` is device copyable as kernel parameter:
A series of checks should be performed that we can pass `marray<T, NumElements>` 
as kernel parameter to free function kernel and use it within kernel.

#### Test `vec<T, NumElements>` when `T` is device copyable as kernel parameter:
A series of checks should be performed that we can pass `vec<T, NumElements>` 
as kernel parameter to free function kernel and use it within kernel.

#### Test `sampled_image_accessor` as kernel parameter:
A series of checks should be performed that we can pass `sampled_image_accessor` 
as kernel parameter to free function kernel and use it within kernel.

#### Test `unsampled_image_accessor` as kernel parameter:
A series of checks should be performed that we can pass 
`unsampled_image_accessor` as kernel parameter to free function kernel and 
use it within kernel.

#### Test `local_accessor` as kernel parameter: 
A series of checks should be performed that we can pass `local_accessor` 
as kernel parameter to free function kernel and use it within kernel.

#### Test structs that contain one of the following `accessor`, `local_accessor`, `sampled_image_accessor` or `unsampled_image_accessor` types when used as kernel parameter:
A series of checks should be performed that we can pass struct that contain 
one of the following `accessor`, `local_accessor`, `sampled_image_accessor` or 
`unsampled_image_accessor` types as kernel parameter to free function kernel 
and use it within kernel.

#### Test `struct` defined at namespace scope as kernel parameter:
A series of checks should be performed that we can pass `struct` as kernel 
parameter and use it within kernel.

#### Test `class` defined at namespace scope as kernel parameter:
A series of checks should be performed that we can pass `class` as kernel 
parameter and use it within kernel.

#### Test scoped enumeration defined at namespace scope as kernel parameter:
A series of checks should be performed that we can pass scoped enumeration as 
kernel parameter and use it within kernel.

#### Test unscoped enumeration that has an explicit underlying type defined at namespace scope as kernel parameter:
A series of checks should be performed that we can pass unscoped enumeration 
that has an explicit underlying type as kernel parameter and use it 
within kernel.

#### Interaction with additional kernel properties:
A series of checks should be performed to check that to the free function 
kernels may also be decorated with the properties defined in 
`sycl_ext_oneapi_kernel_properties`. This test should check that properties can 
be queried via `sycl::ext::oneapi::experimental::get_kernel_info` using either 
the `info::kernel::attributes` information descriptor or the 
`info::kernel_device_specific` information descriptor and it would produce the 
same result as would be computed by `kernel::get_info` using same information 
descriptors.
   
#### Free function kernels compatibility with L0 backend:
A series of checks should be performed to check compatibility of free function 
kernels with Level Zero Backend without going through the SYCL host runtime.

#### Free function kernels compatibility with OpenCL backend:
A series of checks should be performed to check compatibility of free function 
kernels with OpenCL Backend without going through the SYCL host runtime.

#### Test template support in free function kernels:
A series of checks should be performed to check compatibility of free function 
kernels with templateed kernel parameters.

[spec-link]: https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/proposed/sycl_ext_oneapi_free_function_kernels.asciidoc