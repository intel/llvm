# Level-Zero backend specification

## 1. Introduction

This extension introduces Level-Zero backend for SYCL.
It is built on top of Level-Zero runtime enabled with [Level-Zero API](https://spec.oneapi.com/level-zero/latest/index.html).
The Level-Zero backend is aimed to provide the best possible performance of SYCL application on a variety of targets supported.
The currently supported targets are all Intel GPUs starting with Gen9.

This extension provides a feature-test macro as described in the core SYCL specification section 6.3.3 "Feature test macros". Therefore, an implementation supporting this extension must predefine the macro SYCL_EXT_ONEAPI_BACKEND_LEVEL_ZERO to one of the values defined in the table below. Applications can test for the existence of this macro to determine if the implementation supports this feature, or applications can test the macro’s value to determine which of the extension’s APIs the implementation supports.

NOTE: By necessity, this specification exposes some details about the way SYCL is layered on top of the Level Zero backend.  Although the DPC++/SYCL implementers make every effort to retain backward compatibility for changes to this specification, it is occasionally necessary to change an API in this specification in a way that is not compatible with previous versions. For example, this sometimes happens when DPC++ changes the way it layers SYCL on top of Level Zero.  When an API breaking change like this occurs, we update the version of the SYCL_EXT_ONEAPI_BACKEND_LEVEL_ZERO feature-test macro and describe the API breakage as part of the description for that API version.

|Value|Description|
|---|:---|
|1|Initial extension version.
|2|Added support for the make_buffer() API.
|3|Added device member to backend_input_t<backend::ext_oneapi_level_zero, queue>.
|4|Change the definition of backend_input_t and backend_return_t for the queue object, which changes the API for make_queue and get_native (when applied to queue).
|5|Added support for make_image() API.

NOTE: This extension is following SYCL 2020 backend specification. Prior API for interoperability with Level-Zero is marked
      as deprecated and will be removed in the next release.


## 2. Prerequisites

The Level-Zero loader and drivers need to be installed on the system for SYCL runtime to recognize and enable the Level-Zero backend.
For further details see <https://software.intel.com/content/www/us/en/develop/articles/intel-oneapi-dpcpp-system-requirements.html>.
                  
## 3. User-visible Level-Zero backend selection (and the default backend)

The Level-Zero backend is added to the sycl::backend enumeration:

``` C++
enum class backend {
  // ...
  ext_oneapi_level_zero,
  // ...
};
```

There are multiple ways in which the Level-Zero backend can be selected by the user.
        
### 3.1 Through an environment variable
        
The ONEAPI_DEVICE_SELECTOR environment variable limits the SYCL runtime to use only a subset of the system's devices.
By using ```level_zero``` for backend in ONEAPI_DEVICE_SELECTOR you can select the use of Level-Zero as a SYCL backend.
For further details see here: <https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md>.
        
### 3.2 Through a programming API
        
There is an extension that introduces a filtering device selection to SYCL described in
[sycl\_ext\_oneapi\_filter\_selector](../supported/sycl_ext_oneapi_filter_selector.asciidoc).
Similar to how ONEAPI_DEVICE_SELECTOR applies filtering to the entire process this device selector can be used to
programmatically select the Level-Zero backend.
                
When neither the environment variable nor the filtering device selector are used, the implementation chooses
the Level-Zero backend for GPU devices supported by the installed Level-Zero runtime.
The serving backend for a SYCL platform can be queried with the ```get_backend()``` member function of ```sycl::platform```.

## 4. Interoperability with the Level-Zero API

This chapter describes the various interoperability possible between SYCL and Level-Zero.
The application must include both of the following headers in order to use any of the interoperation APIs described in this section,
and they must be included in the order shown:

``` C++
  #include "level_zero/ze_api.h"
  #include "sycl/ext/oneapi/backend/level_zero.hpp"
```
### 4.1 Mapping of SYCL objects to Level-Zero handles

SYCL objects encapsulate the corresponding Level-Zero handles and this extension provides the following specialization of the interoperability types:

<table>
<tr>
<th>SyclType</th>
<th>

``` C++
backend_return_t<backend::ext_oneapi_level_zero,
                 SyclType>
```
</th>
<th>

``` C++
backend_input_t<backend::ext_oneapi_level_zero,
                SyclType>
```
</th>
</tr><tr>
<td>platform</td>
<td><pre>ze_driver_handle_t</pre></td>
<td><pre>ze_driver_handle_t</pre></td>
</tr><tr>
<td>device</td>
<td><pre>ze_device_handle_t</pre></td>
<td><pre>ze_device_handle_t</pre></td>
</tr><tr>
<td>context</td>
<td><pre>ze_context_handle_t</pre></td>
<td>

``` C++
struct {
  ze_context_handle_t NativeHandle;
  std::vector<device> DeviceList;
  ext::oneapi::level_zero::ownership Ownership{
      ext::oneapi::level_zero::ownership::transfer};
}
```
</td>
</tr>
<tr>
<td rowspan="3">queue</td>
<td rowspan="2">

``` C++
ze_command_queue_handle_t
```

Prior to version 4 of this specification.
</td>

<td>

``` C++
struct {
  ze_command_queue_handle_t NativeHandle;
  ext::oneapi::level_zero::ownership Ownership{
      ext::oneapi::level_zero::ownership::transfer};
}
```

Prior to version 3 of this specification.
</td>
</tr>
<tr>
<td>

``` C++
struct {
  ze_command_queue_handle_t NativeHandle;
  device Device;
  ext::oneapi::level_zero::ownership Ownership{
      ext::oneapi::level_zero::ownership::transfer};
}
```

Starting in version 3 of this specification.
</td>
</tr>
<td rowspan="1">

``` C++
std::variant<ze_command_queue_handle_t,
             ze_command_list_handle_t>
```

Starting in version 4 of this specification.
</td>
<td>

``` C++
struct {
  std::variant<ze_command_queue_handle_t,
               ze_command_list_handle_t>
    NativeHandle;
  device Device;
  ext::oneapi::level_zero::ownership Ownership{
      ext::oneapi::level_zero::ownership::transfer};
  property_list Properties{};
}
```

Starting in version 4 of this specification.
</td>
</tr>

<tr>
<td>event</td>
<td><pre>ze_event_handle_t</pre></td>
<td>

``` C++
struct {
  ze_event_handle_t NativeHandle;
  ext::oneapi::level_zero::ownership Ownership{
      ext::oneapi::level_zero::ownership::transfer};
}
```
</td>
</tr><tr>
<td>kernel_bundle</td>
<td>

``` C++
std::vector<ze_module_handle_t>
```
</td>
<td>

``` C++
struct {
  ze_module_handle_t NativeHandle;
  ext::oneapi::level_zero::ownership Ownership{
      ext::oneapi::level_zero::ownership::transfer};
}
```
</td>
</tr><tr>
<td>kernel</td>
<td>

``` C++
ze_kernel_handle_t
```
</td>
<td>

``` C++
struct {
  kernel_bundle<bundle_state::executable> KernelBundle;
  ze_kernel_handle_t NativeHandle;
  ext::oneapi::level_zero::ownership Ownership{
      ext::oneapi::level_zero::ownership::transfer};
}
```
</td>
</tr><tr>
<td>buffer</td>
<td>

``` C++
void *
```
</td>
<td>

``` C++
struct {
  void *NativeHandle;
  ext::oneapi::level_zero::ownership Ownership{
      ext::oneapi::level_zero::ownership::transfer};
}
```
</td>
</tr>
<tr>
<td>image</td>
<td>

``` C++
ze_image_handle_t
```
</td>
<td>

``` C++
struct {
    ze_image_handle_t ZeImageHandle;
    sycl::image_channel_order ChanOrder;
    sycl::image_channel_type ChanType;
    sycl::range<Dimensions> Range;
    ext::oneapi::level_zero::ownership Ownership{
        ext::oneapi::level_zero::ownership::transfer};
  }
```
</td>
</tr>
</table>

### 4.2 Obtaining of native Level-Zero handles from SYCL objects
                
The ```sycl::get_native<backend::ext_oneapi_level_zero>``` free-function is how a raw native Level-Zero handle can be obtained
for a specific SYCL object.
``` C++
template <backend BackendName, class SyclObjectT>
auto get_native(const SyclObjectT &Obj)
    -> backend_return_t<BackendName, SyclObjectT>
```
It is currently supported for SYCL ```platform```, ```device```, ```context```, ```queue```, ```event```,
```kernel_bundle```, and ```kernel``` classes. 

The ```get_native(queue)``` function returns either ```ze_command_queue_handle_t``` or ```ze_command_list_handle_t``` depending on the manner in which the input argument ```queue``` had been created. Queues created with the SYCL ```queue``` constructors have a default setting for whether they use command queues or command lists. The default and how it may be changed is documented in the description for the environment variable ```SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS```. Queues created using ```make_queue()``` use either a command list or command queue depending on the input argument to ```make_queue``` and are not affected by the default for SYCL queues or the environment variable.

The ```sycl::get_native<backend::ext_oneapi_level_zero>```
free-function is not supported for SYCL ```buffer``` or ```image``` class. The native backend object associated with the
buffer can be obtained using interop_hande class as described in the core SYCL specification section
4.10.2, "Class interop_handle". 
The pointer returned by ```get_native_mem<backend::ext_oneapi_level_zero>``` method of the ```interop_handle```
class is the value returned from a call to <code>zeMemAllocShared()</code>, <code>zeMemAllocDevice()</code>,
or <code>zeMemAllocHost()</code> and not necessarily directly accessible from the host.  Users may need to copy
data to the host to access the data. Users can get type of the allocation using ```type``` data member of the 
```ze_memory_allocation_properties_t``` struct returned by ```zeMemGetAllocProperties```.

``` C++
    Queue.submit([&](handler &CGH) {
        auto BufferAcc = Buffer.get_access<access::mode::write>(CGH);
        CGH.host_task([=](const interop_handle &IH) {
            void *DevicePtr =
                IH.get_native_mem<backend::ext_oneapi_level_zero>(BufferAcc);
            ze_memory_allocation_properties_t MemAllocProperties{};
            ze_result_t Res = zeMemGetAllocProperties(
                ZeContext, DevicePtr, &MemAllocProperties, nullptr);
            ze_memory_type_t ZeMemType = MemAllocProperties.type;
        });
    }).wait();
```
### 4.3 Construct a SYCL object from a Level-Zero handle
        
The following free functions defined in the ```sycl``` namespace are specialized for Level-Zero backend to allow
an application to create a SYCL object that encapsulates a corresponding Level-Zero object:

<table>
<tr>
<th>Level-Zero interoperability function</th>
<th style="text-align:left"> Description</th>
</tr><tr>
<td>

``` C++
template <backend Backend>
platform make_platform(
    const backend_input_t<Backend, platform> &)
```
</td>
<td>Constructs a SYCL platform instance from a Level-Zero <code>ze_driver_handle_t</code>. The SYCL execution environment contains a fixed number of platforms that are enumerated via <code>sycl::platform::get_platforms()</code>. Calling this function does not create a new platform. Rather it merely creates a <code>sycl::platform</code> object that is a copy of one of the platforms from that enumeration.</td>
</tr><tr>
<td>

``` C++
template <backend Backend>
device make_device(
    const backend_input_t<Backend, device> &)
```
</td>
<td>Constructs a SYCL device instance from a Level-Zero <code>ze_device_handle_t</code>. The SYCL execution environment for the Level Zero backend contains a fixed number of devices that are enumerated via <code>sycl::device::get_devices()</code> and a fixed number of sub-devices that are enumerated via <code>sycl::device::create_sub_devices(...)</code>. Calling this function does not create a new device. Rather it merely creates a <code>sycl::device</code> object that is a copy of one of the devices from those enumerations.</td>
</tr><tr>
<td>

``` C++
template <backend Backend>
context make_context(
    const backend_input_t<Backend, context> &)
```
</td>
<td>Constructs a SYCL context instance from a Level-Zero <code>ze_context_handle_t</code>. The context is created against the devices passed in <code>DeviceList</code> structure member. There must be at least one device given and all the devices must be from the same SYCL platform and thus from the same Level-Zero driver. The <code>Ownership</code> input structure member specifies if the SYCL runtime should take ownership of the passed native handle. The default behavior is to transfer the ownership to the SYCL runtime. See section 4.4 for details.</td>
</tr><tr>
<td>

``` C++
template <backend Backend>
queue make_queue(
    const backend_input_t<Backend, queue> &,
    const context &Context)
```
</td>
<td>Constructs a SYCL queue instance from a Level-Zero <code>ze_command_queue_handle_t</code>. The <code>Context</code> argument must be a valid SYCL context encapsulating a Level-Zero context. The <code>Device</code> input structure member specifies the device to create the <code>queue</code> against and must be in <code>Context</code>. The <code>Ownership</code> input structure member specifies if the SYCL runtime should take ownership of the passed native handle. The default behavior is to transfer the ownership to the SYCL runtime. See section 4.4 for details.

If the deprecated variant of <code>backend_input_t<backend::ext_oneapi_level_zero, queue></code> is passed to <code>make_queue</code> the queue is attached to the first device in <code>Context</code>.

Starting in version 4 of this specification, ```make_queue()``` can be called by passing either a Level Zero ```ze_command_queue_handle_t``` or a Level Zero ```ze_command_list_handle_t```. Queues created from a Level Zero immediate command list (```ze_command_list_handle_t```) generally perform better than queues created from a standard Level Zero ```ze_command_queue_handle_t```. See the Level Zero documentation of these native handles for more details. Also starting in version 4 the ```make_queue()``` function accepts a ```Properties``` member variable. This can contain any of the SYCL properties that are accepted by the SYCL queue constructor, except
the ```compute_index``` property which is built into the command queue or command list.
</td>
</tr><tr>
<td>

``` C++
template <backend Backend>
event make_event(
    const backend_input_t<Backend, event> &,
    const context &Context)
```
</td>
<td>Constructs a SYCL event instance from a Level-Zero <code>ze_event_handle_t</code>. The <code>Context</code> argument must be a valid SYCL context encapsulating a Level-Zero context. The Level-Zero event should be allocated from an event pool created in the same context. The <code>Ownership</code> input structure member specifies if the SYCL runtime should take ownership of the passed native handle. The default behavior is to transfer the ownership to the SYCL runtime. See section 4.4 for details.</td>
</tr><tr>
<td>

``` C++
// State must be bundle_state::executable
template <backend Backend, bundle_state State>
kernel_bundle<State> make_kernel_bundle(
    const backend_input_t<Backend,
        kernel_bundle<State>> &,
    const context &Context)
```
</td>
<td>Constructs a SYCL kernel_bundle instance from a Level-Zero
<code>ze_module_handle_t</code>. The <code>Context</code> argument must be a
valid SYCL context encapsulating a Level-Zero context, and the Level-Zero
module must be created on the same context. The Level-Zero module must be
fully linked (i.e. not require further linking through <a href="https://spec.oneapi.com/level-zero/latest/core/api.html?highlight=zemoduledynamiclink#_CPPv419zeModuleDynamicLink8uint32_tP18ze_module_handle_tP28ze_module_build_log_handle_t">
<code>zeModuleDynamicLink</code></a>), and thus the SYCL kernel_bundle is
created in the "executable" state. The <code>Ownership</code> input structure
member specifies if the SYCL runtime should take ownership of the passed
native handle. The default behavior is to transfer the ownership to the SYCL
runtime. See section 4.4 for details. If the behavior is "transfer" then the
runtime is going to destroy the input Level-Zero module, and hence the
application must not to have any outstanding <code>ze_kernel_handle_t</code>
handles to the underlying <code>ze_module_handle_t</code> by the time this
interoperability <code>kernel_bundle</code> destructor is called.</td>
</tr><tr>
<td>

``` C++
template <backend Backend>
kernel make_kernel(
    const backend_input_t<Backend, kernel> &,
    const context &Context)
```
</td>
<td>Constructs a SYCL kernel instance from a Level-Zero
<code>ze_kernel_handle_t</code>. The <code>KernelBundle</code> input structure
specifies the <code>kernel_bundle<bundle_state::executable></code> corresponding
to the Level-Zero module from which the kernel is created. There must be exactly
one Level-Zero module in the <code>KernelBundle</code>. The <code>Context</code>
argument must be a valid SYCL context encapsulating a Level-Zero context, and
the Level-Zero module must be created on the same context.
The <code>Ownership</code> input structure member specifies if the SYCL runtime
should take ownership of the passed native handle. The default behavior is to
transfer the ownership to the SYCL runtime. See section 4.4 for details. If
the behavior is "transfer" then the runtime is going to destroy the input
Level-Zero kernel</td>
</tr><tr>
<td>

``` C++
template <backend Backend, 
          typename T, int Dimensions = 1,
          typename AllocatorT = 
            buffer_allocator<std::remove_const_t<T>>>
buffer<T, Dimensions, AllocatorT> make_buffer(
    const backend_input_t<Backend,
                          buffer<T, 
                                 Dimensions, 
                                 AllocatorT>> &,
    const context &Context)
```
</td>
<td>This API is available starting with revision 2 of this specification.

Construct a SYCL buffer instance from a pointer to a Level Zero memory allocation. The pointer must be the value returned from a previous call to <code>zeMemAllocShared()</code>, <code>zeMemAllocDevice()</code>, or <code>zeMemAllocHost()</code>. The input SYCL context <code>Context</code> must be associated with a single device, matching the device used at the prior allocation.
The <code>Context</code> argument must be a valid SYCL context encapsulating a Level-Zero context, and the Level-Zero memory must be allocated on the same context. Created SYCL buffer can be accessed in another contexts, not only in the provided input context.
The <code>Ownership</code> input structure member specifies if the SYCL runtime should take ownership of the passed native handle. The default behavior is to transfer the ownership to the SYCL runtime. See section 4.4 for details. If the behavior is "transfer" then the runtime is going to free the input Level-Zero memory allocation. 
Synchronization rules for a buffer that is created with this API are described in Section 4.5</td>
</tr><tr>
<td>

``` C++
template <backend Backend, 
          typename T, int Dimensions = 1,
          typename AllocatorT = 
            buffer_allocator<std::remove_const_t<T>>>
buffer<T, Dimensions, AllocatorT> make_buffer(
    const backend_input_t<Backend,
                          buffer<T, 
                                 Dimensions, 
                                 AllocatorT>> &,
    const context &Context, event AvailableEvent)
```
</td>
<td>This API is available starting with revision 2 of this specification.

Construct a SYCL buffer instance from a pointer to a Level Zero memory allocation. Please refer to <code>make_buffer</code>
description above for semantics and restrictions.
The additional <code>AvailableEvent</code> argument must be a valid SYCL event. The instance of the SYCL buffer class template being constructed must wait for the SYCL event parameter to signal that the memory native handle is ready to be used.
</tr>

<tr>
<td>

``` C++
template<backend Backend, int Dimensions = 1, 
         typename AllocrT = sycl::image_allocator>
image<Dimensions, AllocrT> make_image(
    const backend_input_t<Backend, 
                          image<Dimensions, 
                            AllocrT>> &backendObject,
    const context &targetContext);
```
</td>
<td>This API is available starting with revision 5 of this specification.

Construct a SYCL image instance from a ze_image_handle_t. 

Because LevelZero has no way of getting image information from an image, it must be provided.  The  <code>backend_input_t</code> is a struct type like so:
``` C++
struct type {
    ze_image_handle_t ZeImageHandle;
    sycl::image_channel_order ChanOrder;
    sycl::image_channel_type ChanType;
    sycl::range<Dimensions> Range;
    ext::oneapi::level_zero::ownership Ownership{
        ext::oneapi::level_zero::ownership::transfer};
  };
```
where the Range should be ordered (width), (width, height), or (width, height, depth) for 1D, 2D and 3D images respectively, 
with those values matching the dimensions used in the `ze_image_desc` that was used to create the `ze_image_handle_t` initially. 
Note that the range term ordering (width first, depth last) is true for SYCL 1.2.1 images that are supported here. But future classes like
sampled_image and unsampled_image might have a different ordering.

Example Usage
``` C++
ze_image_handle_t ZeHImage; 
// ... user provided LevelZero ZeHImage image 
// handle gotten somehow (possibly zeImageCreate)

// the informational data that matches ZeHImage
sycl::image_channel_order ChanOrder 
     = sycl::image_channel_order::rgba;
sycl::image_channel_type ChanType 
     = sycl::image_channel_type::unsigned_int8;
size_t width  = 4;
size_t height = 2;
sycl::range<2> ImgRange_2D(width, height);

constexpr sycl::backend BE 
       = sycl::backend::ext_oneapi_level_zero;
sycl::backend_input_t<BE, sycl::image<2>> ImageInteropInput{ 
    ZeHImage, 
    ChanOrder,
    ChanType, 
    ImgRange_2D, 
    sycl::ext::oneapi::level_zero::ownership::transfer };      
    
sycl::image<2> Image_2D  
  = sycl::make_image<BE, 2>(ImageInteropInput, Context);
```

The image can only be used on the single device where it was created. This limitation may be relaxed in the future.
The <code>Context</code> argument must be a valid SYCL context encapsulating a Level-Zero context, and the Level-Zero image must have been created on the same context. The created SYCL image can only be accessed from kernels that are submitted to a queue using this same context.
The <code>Ownership</code> input structure member specifies if the SYCL runtime should take ownership of the passed native handle. The default behavior is to transfer the ownership to the SYCL runtime. See section 4.4 for details. If the behavior is "transfer" then the SYCL runtime is going to free the input Level-Zero memory allocation, meaning the memory will be freed when the ~image destructor fires. When using "transfer" the ~image destructor may not need to block.  If the behavior is "keep", then the memory will not be freed by the ~image destructor, and the ~image destructor blocks until all work in the queues on the image have been completed. When using "keep" it is the responsibility of the caller to free the memory appropriately. 
</td>
</tr>

<tr>
<td>

``` C++
template<backend Backend, int Dimensions = 1, 
         typename AllocrT = sycl::image_allocator>
image<Dimensions, AllocrT> make_image(
    const backend_input_t<Backend, 
                          image<Dimensions, 
                            AllocrT>> &backendObject,
    const context &targetContext, event availableEvent);
```
</td>
<td>This API is available starting with revision 5 of this specification.

Construct a SYCL image instance from a pointer to a Level Zero memory allocation. Please refer to <code>make_image</code>
description above for semantics and restrictions.
The additional <code>AvailableEvent</code> argument must be a valid SYCL event. The instance of the SYCL image class template being constructed must wait for the SYCL event parameter to signal that the memory native handle is ready to be used.
</td>
</tr>
</table>

NOTE: We shall consider adding other interoperability as needed, if possible.
                
### 4.4 Level-Zero handles' ownership and thread-safety
        
The Level-Zero runtime doesn't do reference-counting of its objects, so it is crucial to adhere to these
practices of how Level-Zero handles are managed. By default, the ownership is transferred to the SYCL runtime, but
some interoperability API supports overriding this behavior and keep the ownership in the application.
Use this enumeration for explicit specification of the ownership:
``` C++
namespace sycl {
namespace ext {
namespace oneapi {
namespace level_zero {

enum class ownership { transfer, keep };

} // namespace level_zero
} // namespace oneapi
} // namespace ext
} // namespace sycl
```
                
#### 4.4.1 SYCL runtime takes ownership (default)
                
Whenever the application creates a SYCL object from the corresponding Level-Zero handle via one of the ```make_*``` functions,
the SYCL runtime takes ownership of the Level-Zero handle, if no explicit ```ownership::keep``` was specified.
The application must not use the Level-Zero handle after the last host copy of the SYCL object is destroyed (
as described in the core SYCL specification under "Common reference semantics"), and the application must not
destroy the Level-Zero handle itself.

#### 4.4.2 Application keeps ownership (explicit)

If SYCL object is created with an interoperability API explicitly asking to keep the native handle ownership in the application with
```ownership::keep``` then the SYCL runtime does not take the ownership and will not destroy the Level-Zero handle at the destruction of the SYCL object.
The application is responsible for destroying the native handle when it no longer needs it, but it must not destroy the
handle before the last host copy of the SYCL object is destroyed (as described in the core SYCL specification under
"Common reference semantics").
                                                                
#### 4.4.3 Obtaining native handle does not change ownership

The application may call the ```get_native<backend::ext_oneapi_level_zero>``` free function on a SYCL object to retrieve the underlying Level-Zero handle.
Doing so does not change the ownership of the the Level-Zero handle.  Therefore, the application may not use this
handle after the last host copy of the SYCL object is destroyed (as described in the core SYCL specification under
"Common reference semantics") unless the SYCL object was created by the application with ```ownership::keep```.

#### 4.4.4 Considerations for multi-threaded environment

The Level-Zero API is not thread-safe, refer to <https://spec.oneapi.com/level-zero/latest/core/INTRO.html#multithreading-and-concurrency>.
Applications must make sure that the Level-Zero handles themselves aren't used simultaneously from different threads.
Practically speaking, and taking into account that SYCL runtime takes ownership of the Level-Zero handles,
the application should not attempt further direct use of those handles.

### 4.5 Interoperability buffer synchronization rules

A SYCL buffer that is constructed with this interop API uses the Level Zero memory allocation for its full lifetime, and the contents of the Level Zero memory allocation are unspecified for the lifetime of the SYCL buffer. If the application modifies the contents of that Level Zero memory allocation during the lifetime of the SYCL buffer, the behavior is undefined. The initial contents of the SYCL buffer will be the initial contents of the Level Zero memory allocation at the time of the SYCL buffer's construction.

The behavior of the SYCL buffer destructor depends on the Ownership flag. As with other SYCL buffers, this behavior is triggered only when the last reference count to the buffer is dropped, as described in the core SYCL specification section 4.7.2.3, "Buffer synchronization rules".

* If the ownership is keep (i.e. the application retains ownership of the Level Zero memory allocation), then the SYCL buffer destructor blocks until all work in queues on the buffer have completed. The buffer's contents is not copied back to the Level Zero memory allocation.
* If the ownership is transfer (i.e. the SYCL runtime has ownership of the Level Zero memory allocation), then the SYCL buffer destructor does not need to block even if work on the buffer has not completed. The SYCL runtime frees the Level Zero memory allocation asynchronously when it is no longer in use in queues.

## 5. Device information descriptors

|Device descriptors|Value|
|-------------|:------------|
|`info::device::backend_version`|The Level Zero backend does not define the value of this information descriptor. Applications should not use it, and implementations are encouraged to return the empty string.

## Revision History
|Rev|Date|Author|Changes|
|-------------|:------------|:------------|:------------|
|1|2021-01-26|Sergey Maslov|Initial public working draft
|2|2021-02-22|Sergey Maslov|Introduced explicit ownership for context
|3|2021-04-13|James Brodman|Free Memory Query
|4|2021-07-06|Rehana Begam|Introduced explicit ownership for queue
|5|2021-07-25|Sergey Maslov|Introduced SYCL interop for events
|6|2021-08-30|Dmitry Vodopyanov|Updated according to SYCL 2020 reqs for extensions
|7|2021-09-13|Sergey Maslov|Updated according to SYCL 2020 standard
|8|2022-01-06|Artur Gainullin|Introduced make_buffer() API
|9|2022-05-12|Steffen Larsen|Added device member to queue input type
|10|2022-08-18|Sergey Maslov|Moved free_memory device info query to be sycl_ext_intel_device_info extension
|11|2023-03-14|Rajiv Deodhar|Added support for Level Zero immediate command lists
|12|2023-04-06|Chris Perkins|Introduced make_image() API
