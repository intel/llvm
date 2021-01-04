# SYCL(TM) Proposal: Intel's Extension to Set Kernel Cache Configuration


**NOTE**: Khronos(R) is a registered trademark and SYCL(TM) is a trademark of the Khronos Group, Inc.

A new API will be added to set the cache configuration in the kernel.  This information will be useful to developers tuning specifically for those devices.

This proposal details what is required to provide this API as a SYCL extensions.

## Feature Test Macro ##

The Feature Test Macro will be defined as:

    #define SYCL_EXT_INTEL_KERNEL_CACHE_CONFIG 1


# Setting the Kernel Cache Configuration #

The cache configuration of the kernel can be modified to favor large shared local memory (SLM) or large data.  This API gives users a way to do request this change.

The new API is only available when using the Level Zero platform.  The DPC++ default behavior is to expose GPU devices through the Level Zero platform.

``` C++
enum class kernel_cache_config {
  cache_large_slm,
  cache_large_data
};

cl::sycl::kernel::set_cache_config(const kernel_cache_config);
```

**Note:** The environment variable SYCL\_ENABLE\_KERNEL\_CACHE\_CONFIG must be set to 1 to modify the kernel cache configuration.


## Error Condition ##
An invalid object runtime error will be thrown if the configuration cannot be modified.


## Example Usage ##

``` C++
  // Must be enabled at the beginning of the application
  // to modify the kernel cache configuration
  setenv("SYCL_ENABLE_KERNEL_CACHE_CONFIG", "1", 0);
              .
              .
              .
  // Use Feature Test macro to see if this extension is supported.
  if (SYCL_EXT_INTEL_KERNEL_CACHE_CONFIG >= 1) {
    // Configure the kernel to use large SLM.
    krn.set_cache_config(cache_large_slm);
    }

```
