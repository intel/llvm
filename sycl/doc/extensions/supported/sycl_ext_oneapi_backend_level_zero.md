# Level-Zero backend specification

## 1. Introduction

This extension introduces Level-Zero backend for SYCL.
It is built on top of Level-Zero runtime enabled with [Level-Zero API](https://spec.oneapi.com/level-zero/latest/index.html).
The Level-Zero backend is aimed to provide the best possible performance of SYCL application on a variety of targets supported.
The currently supported targets are all Intel GPUs starting with Gen9.

This extension provides a feature-test macro as described in the core SYCL specification section 6.3.3 "Feature test macros". Therefore, an implementation supporting this extension must predefine the macro SYCL_EXT_ONEAPI_BACKEND_LEVEL_ZERO to one of the values defined in the table below. Applications can test for the existence of this macro to determine if the implementation supports this feature, or applications can test the macro’s value to determine which of the extension’s APIs the implementation supports.


|Value|Description|
|---|:---|
|1|Initial extension version.

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
        
The SYCL_DEVICE_FILTER environment variable limits the SYCL runtime to use only a subset of the system's devices.
By using ```level_zero``` for backend in SYCL_DEVICE_FILTER you can select the use of Level-Zero as a SYCL backend.
For further details see here: <https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md>.
        
### 3.2 Through a programming API
        
There is an extension that introduces a filtering device selection to SYCL described in
[sycl\_ext\_oneapi\_filter\_selector](../supported/sycl_ext_oneapi_filter_selector.asciidoc).

Similar to how SYCL_DEVICE_FILTER applies filtering to the entire process this device selector can be used to
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
</tr><tr>
<td>queue</td>
<td><pre>ze_command_queue_handle_t</pre></td>
<td>

``` C++
struct {
  ze_command_queue_handle_t NativeHandle;
  ext::oneapi::level_zero::ownership Ownership{
      ext::oneapi::level_zero::ownership::transfer};
}
```
</td>
</tr><tr>
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
make_platform<backend::ext_oneapi_level_zero>(
    const backend_input_t<
        backend::ext_oneapi_level_zero, platform> &)
```
</td>
<td>Constructs a SYCL platform instance from a Level-Zero <code>ze_driver_handle_t</code>. The SYCL execution environment contains a fixed number of platforms that are enumerated via <code>sycl::platform::get_platforms()</code>. Calling this function does not create a new platform. Rather it merely creates a <code>sycl::platform</code> object that is a copy of one of the platforms from that enumeration.</td>
</tr><tr>
<td>

``` C++
make_device<backend::ext_oneapi_level_zero>(
    const backend_input_t<
        backend::ext_oneapi_level_zero, device> &)
```
</td>
<td>Constructs a SYCL device instance from a Level-Zero <code>ze_device_handle_t</code>. The SYCL execution environment for the Level Zero backend contains a fixed number of devices that are enumerated via <code>sycl::device::get_devices()</code> and a fixed number of sub-devices that are enumerated via <code>sycl::device::create_sub_devices(...)</code>. Calling this function does not create a new device. Rather it merely creates a <code>sycl::device</code> object that is a copy of one of the devices from those enumerations.</td>
</tr><tr>
<td>

``` C++
make_context<backend::ext_oneapi_level_zero>(
    const backend_input_t<
        backend::ext_oneapi_level_zero, context> &)
```
</td>
<td>Constructs a SYCL context instance from a Level-Zero <code>ze_context_handle_t</code>. The context is created against the devices passed in <code>DeviceList</code> structure member. There must be at least one device given and all the devices must be from the same SYCL platform and thus from the same Level-Zero driver. The <code>Ownership</code> input structure member specifies if the SYCL runtime should take ownership of the passed native handle. The default behavior is to transfer the ownership to the SYCL runtime. See section 4.4 for details.</td>
</tr><tr>
<td>

``` C++
make_queue<backend::ext_oneapi_level_zero>(
    const backend_input_t<
        backend::ext_oneapi_level_zero, queue> &,
    const context &Context)
```
</td>
<td>Constructs a SYCL queue instance from a Level-Zero <code>ze_command_queue_handle_t</code>. The <code>Context</code> argument must be a valid SYCL context encapsulating a Level-Zero context. The queue is attached to the first device in the passed SYCL context. The <code>Ownership</code> input structure member specifies if the SYCL runtime should take ownership of the passed native handle. The default behavior is to transfer the ownership to the SYCL runtime. See section 4.4 for details.</td>
</tr><tr>
<td>

``` C++
make_event<backend::ext_oneapi_level_zero>(
    const backend_input_t<
        backend::ext_oneapi_level_zero, event> &,
    const context &Context)
```
</td>
<td>Constructs a SYCL event instance from a Level-Zero <code>ze_event_handle_t</code>. The <code>Context</code> argument must be a valid SYCL context encapsulating a Level-Zero context. The Level-Zero event should be allocated from an event pool created in the same context. The <code>Ownership</code> input structure member specifies if the SYCL runtime should take ownership of the passed native handle. The default behavior is to transfer the ownership to the SYCL runtime. See section 4.4 for details.</td>
</tr><tr>
<td>

``` C++
make_kernel_bundle<backend::ext_oneapi_level_zero,
                   bundle_state::executable>(
    const backend_input_t<
        backend::ext_oneapi_level_zero,
        kernel_bundle<bundle_state::executable>> &,
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
make_kernel<backend::ext_oneapi_level_zero>(
    const backend_input_t<
        backend::ext_oneapi_level_zero, kernel> &,
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

## 5 Level-Zero additional functionality

### 5.1 Device Information Descriptors
The Level Zero backend provides the following device information descriptors
that an application can use to query information about a Level Zero device.
Applications use these queries via the `device::get_backend_info<>()` member
function as shown in the example below (which illustrates the `free_memory`
query):

``` C++
sycl::queue Queue;
auto Device = Queue.get_device();

size_t freeMemory =
  Device.get_backend_info<sycl::ext::oneapi::level_zero::info::device::free_memory>();
```

New descriptors added as part of this specification are described in the table below and in the subsequent synopsis.

| Descriptor | Description |
| ---------- | ----------- |
| `sycl::ext::oneapi::level_zero::info::device::free_memory` | Returns the number of bytes of free memory for the device. |


``` C++
namespace sycl{
namespace ext {
namespace oneapi {
namespace level_zero {
namespace info {
namespace device {

struct free_memory {
    using return_type = size_t;
};

} // namespace device;
} // namespace info
} // namespace level_zero
} // namespace oneapi
} // namespace ext
} // namespace sycl
```

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
