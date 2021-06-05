# SYCL(TM) Proposal: Intel's Extension to Set Kernel Cache Configuration


**NOTE**: Khronos(R) is a registered trademark and SYCL(TM) is a trademark of the Khronos Group, Inc.

A new API will be added to set the cache configuration in the kernel.  This information will be useful to developers tuning specifically for those devices.

This proposal details what is required to provide this API as a SYCL extensions.

## Feature Test Macro ##

The Feature Test Macro will be defined as:

    #define SYCL_EXT_INTEL_KERNEL_CACHE_CONFIG 1


# Setting the Kernel Cache Configuration #

The cache configuration of the kernel can be modified to favor large shared local memory (SLM) or large data.  This API gives users a way to do request this change.

The new API is only available when using the Level Zero platform.

``` C++
enum class kernel_cache_config {
  cache_large_slm,
  cache_large_data
};

cl::sycl::kernel::set_cache_config(const kernel_cache_config);
```


## Example Usage ##

``` C++
  // Use Feature Test macro to see if this extension is supported.
#ifdef SYCL_EXT_INTEL_KERNEL_CACHE_CONFIG
    // Configure the kernel to use large SLM.
    krn.set_cache_config(cache_large_slm);
#endif

```
