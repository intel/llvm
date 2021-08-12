# Level-Zero backend specification

## 1. Introduction

This extension introduces Level-Zero backend for SYCL.
It is built on top of Level-Zero runtime enabled with [Level-Zero API](https://spec.oneapi.com/level-zero/latest/index.html).
The Level-Zero backend is aimed to provide the best possible performance of SYCL application on a variety of targets supported.
The currently supported targets are all Intel GPUs starting with Gen9.

NOTE: This specification is a draft. While describing the currently implemented behaviors it is known to be not complete nor exhaustive.
      We shall continue to add more information, e.g. explain general mapping of SYCL programming model to Level-Zero API.
      It will also be gradually changing to a SYCL-2020 conforming implementation.

## 2. Prerequisites

The Level-Zero loader and drivers need to be installed on the system for SYCL runtime to recognize and enable the Level-Zero backend.
For further details see <https://software.intel.com/content/www/us/en/develop/articles/intel-oneapi-dpcpp-system-requirements.html>.
                  
## 3. User-visible Level-Zero backend selection (and the default backend)

The Level-Zero backend is added to the cl::sycl::backend enumeration:

``` C++
enum class backend {
  // ...
  level_zero,
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
<https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/FilterSelector/FilterSelector.adoc>.
Similar to how SYCL_DEVICE_FILTER applies filtering to the entire process this device selector can be used to
programmatically select the Level-Zero backend.
                
When neither the environment variable nor the filtering device selector are used, the implementation chooses
the Level-Zero backend for GPU devices supported by the installed Level-Zero runtime.
The serving backend for a SYCL platform can be queried with the ```get_backend()``` member function of ```cl::sycl::platform```.

## 4. Interoperability with the Level-Zero API

This chapter describes the various interoperability possible between SYCL and Level-Zero.
The application must include both of the following headers in order to use any of the interoperation APIs described in this section,
and they must be included in the order shown:

``` C++
  #include "level_zero/ze_api.h"
  #include "sycl/backend/level_zero.hpp"
```
### 4.1 Mapping of SYCL objects to Level-Zero handles

These SYCL objects encapsulate the corresponding Level-Zero handles:
|  SYCL object  |    Level-Zero handle   |
|-------------|:------------|
|platform |ze_driver_handle_t|
|device   |ze_device_handle_t|
|context  |ze_context_handle_t|
|queue    |ze_command_queue_handle_t|
|event    |ze_event_handle_t|
|program  |ze_module_handle_t|

### 4.2 Obtaining of native Level-Zero handles from SYCL objects
                
The ```get_native<cl::sycl::backend::level_zero>()``` member function is how a raw native Level-Zero handle can be obtained
for a specific SYCL object. It is currently supported for SYCL ```platform```, ```device```, ```context```, ```queue```, ```event```
and ```program``` classes. There is also a free-function defined in ```cl::sycl``` namespace that can be used instead of the member function:
``` C++
template <backend BackendName, class SyclObjectT>
auto get_native(const SyclObjectT &Obj) ->
	typename interop<BackendName, SyclObjectT>::type;
```
### 4.3 Construct a SYCL object from a Level-Zero handle
        
The following free functions defined in the ```cl::sycl::level_zero``` namespace allow an application to create
a SYCL object that encapsulates a corresponding Level-Zero object:

| Level-Zero interoperability function |Description|
|-------------|:------------|
|``` make<platform>(ze_driver_handle_t);```|Constructs a SYCL platform instance from a Level-Zero ```ze_driver_handle_t```.|
|``` make<device>(const platform &, ze_device_handle_t);```|Constructs a SYCL device instance from a Level-Zero ```ze_device_handle_t```. The platform argument gives a SYCL platform, encapsulating a Level-Zero driver supporting the passed Level-Zero device.|
|``` make<context>(const vector_class<device> &, ze_context_handle_t, ownership = transfer);```| Constructs a SYCL context instance from a Level-Zero ```ze_context_handle_t```. The context is created against the devices passed in. There must be at least one device given and all the devices must be from the same SYCL platform and thus from the same Level-Zero driver. The ```ownership``` argument specifies if the SYCL runtime should take ownership of the passed native handle. The default behavior is to transfer the ownership to the SYCL runtime. See section 4.4 for details.|
|``` make<queue>(const context &, ze_command_queue_handle_t, ownership = transfer);```| Constructs a SYCL queue instance from a Level-Zero ```ze_command_queue_handle_t```. The context argument must be a valid SYCL context encapsulating a Level-Zero context. The queue is attached to the first device in the passed SYCL context. The ```ownership``` argument specifies if the SYCL runtime should take ownership of the passed native handle. The default behavior is to transfer the ownership to the SYCL runtime. See section 4.4 for details.|
|``` make<event>(const context &, ze_event_handle_t, ownership = transfer);```| Constructs a SYCL event instance from a Level-Zero ```ze_event_handle_t```. The context argument must be a valid SYCL context encapsulating a Level-Zero context. The Level-Zero event should be allocated from an event pool created in the same context. The ```ownership``` argument specifies if the SYCL runtime should take ownership of the passed native handle. The default behavior is to transfer the ownership to the SYCL runtime. See section 4.4 for details.|
|``` make<program>(const context &, ze_module_handle_t);```| Constructs a SYCL program instance from a Level-Zero ```ze_module_handle_t```. The context argument must be a valid SYCL context encapsulating a Level-Zero context. The Level-Zero module must be fully linked (i.e. not require further linking through [```zeModuleDynamicLink```](https://spec.oneapi.com/level-zero/latest/core/api.html?highlight=zemoduledynamiclink#_CPPv419zeModuleDynamicLink8uint32_tP18ze_module_handle_tP28ze_module_build_log_handle_t)), and thus the SYCL program is created in the "linked" state.|

NOTE: We shall consider adding other interoperability as needed, if possible.
                
### 4.4 Level-Zero handles' ownership and thread-safety
        
The Level-Zero runtime doesn't do reference-counting of its objects, so it is crucial to adhere to these
practices of how Level-Zero handles are managed. By default, the ownership is transferred to the SYCL runtime, but
some interoperability API supports overriding this behavior and keep the ownership in the application.
Use this enumeration for explicit specification of the ownership:
``` C++
namespace sycl {
namespace level_zero {

enum class ownership { transfer, keep };

} // namespace level_zero
} // namespace sycl
```
                
#### 4.4.1 SYCL runtime takes ownership (default)
                
Whenever the application creates a SYCL object from the corresponding Level-Zero handle via one of the ```make<T>()``` functions,
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

The application may call the ```get_native<T>()``` member function of a SYCL object to retrieve the underlying Level-Zero handle.
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
