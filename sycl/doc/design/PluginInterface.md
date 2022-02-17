# The DPC++ Runtime Plugin Interface.

## Overview
The DPC++ Runtime Plugin Interface (PI) is an interface layer between the
device-agnostic part of DPC++ runtime and the device-specific runtime layers
which control execution on devices. It employs the “plugin” mechanism to bind
to the device specific runtime layers similar to what is used by libomptarget
or OpenCL.

The picture below illustrates the placement of PI within the overall DPC++
runtime stack. Dotted lines show components or paths which are not yet available
in the runtime, but are likely to be developed.
![PI in DPC++ runtime architecture](images/RuntimeArchitecture.svg)

The plugin interface and the discovery process behind it allows to dynamically
plug in implementations based on OpenCL and “native” runtime for a particular
device – such as OpenCL for
FPGA devices or native runtimes for GPUs. Implementations of the PI are
“plugins” - dynamic libraries or shared objects which expose a number of entry
points implementing the PI interface. The DPC++ runtime collects those function
pointers into a PI interface dispatch table - one per plugin - and uses this
table to dispatch to the device(s) covered by the corresponding plugin.

PI is based on a subset of OpenCL 1.2 runtime specification, it follows OpenCL's
platform, execution and memory models in all aspects except for those explicitly
mentioned in this document. Some of PI API types and functions have exact
matches in OpenCL. Whenever there is such a match, the semantics also fully
match unless the differences are explicitly specified in this document. While
PI has roots in OpenCL, it does have many differences, and the gap is likely
to grow, for example in areas of memory model and management, program
management.

## Discovery and linkage of PI implementations

![PI implementation discovery](images/PluginDiscovery.svg)

Device discovery phase enumerates all available devices and their features by
querying underlying plugins found in the system. This process is performed when
all attached platforms or devices are queried in an application; for example,
during device selection.

### Plugin discovery

Plugins are physically dynamic libraries or shared objects.
The process to discover plugins follows the following guidelines.

The DPC++ Runtime reads the names of the plugins from a configuration file 
at a predetermined location (TBD - Add this location). These plugins are
searched at locations in env LD_LIBRARY_PATH on Linux and env PATH on Windows.
(TBD - Extend to search the plugins at a path relative to the SYCL Runtime
installation directory by using DT_RPATH on Linux. Similar functionality can be
achieved on Windows using SetDllDirectory. This will help avoiding extra setting
of LD_LIBRARY_PATH.)
To avoid any issues with read-only access, an environment variable
SYCL_PI_CONFIG can be set to point to the configuration file which lists the
Plugin names. The enviroment variable if set overrides the predetermined
location's config file. These Plugins are then be searched in LD_LIBRARY_PATH
locations. It is the developer's responsibility to include the plugin names from
the predetermined location's config file to enable discovery of all plugins.
(TBD - Extend to support search in DT_RPATH as above.)
In the current implementation the plugin names are hardcoded in the library.
Configuration file or env SYCL_PI_CONFIG is currently not being considered.

A trace mechanism is provided using env SYCL_PI_TRACE to log the discovery/
binding/ device enumeration process. Different levels of tracing can be achieved
with different values of SYCL_PI_TRACE.
SYCL_PI_TRACE=0x01 provides basic trace of plugins discovered and bound. It also
lists the device selector's selected device information.
SYCL_PI_TRACE=0x02 provides trace of all PI calls made from the DPC++ runtime
with arguments and returned values.
SYCL_PI_TRACE=-1 lists all PI Traces above and more debug messages.

#### Plugin binary interface
Plugins should implement all the Interface APIs required for the PI Version
it supports. There is [pi.def](../../include/CL/sycl/detail/pi.def)/
[pi.h](../../include/CL/sycl/detail/pi.h) file listing all PI API names that
can be called by the specific version of Plugin Interface.
It exports a function - "piPluginInit" that returns the plugin details and
function pointer table containing the list of pointers to implemented Interface
Functions defined in pi.h.
In the future, this document will list the minimum set of Interface APIs
to be supported by Plugins. This will also require adding functionality to SYCL
Runtime to work with such limited functionality plugins.

(TBD - list and describe the symbols that a plugin must implement in order to
be picked up by the DPC++ runtime for offload.)

#### Binding a Plugin
The DPC++ Runtime loads all discovered Plugins and tries to bind them by calling
piPluginInit API for each loaded Plugin. The Plugins return the information of
supported PI version and the list of implemented PI API Function pointers.
(TBD - Use the PI API Version information and check for compatibility.
Extend to support version compatibility checks without loading the library.
Eg:Changing the plugin name to reflect the supported Plugin Interface version.)
The information of compatible plugins (with the Function Pointer Table) is
stored in the associated platforms during platform object construction.
The PI API calls are later forwarded using this information.
A plugin is said to "bind" after this process completes with no errors.
During device selection, the user can prefer selection of a device from a
specific Plugin or Backend using the env SYCL_BE. The correspondence between
a plugin and a SYCL_BE value is currently hardcoded in the runtime.
( TBD: Make this a part of configuration file).
Eg: SYCL_BE=PI_OPENCL corresponds to OpenCL Plugin.

#### OpenCL plugin

OpenCL plugin is a usual plugin from DPC++ runtime standpoint, but its loading
and initialization involves a nested discovery process which finds out available
OpenCL implementations. They can be installed either in the standard Khronos
ICD-compatible way (e.g. listed in files under /etc/OpenCL/vendors on
Linux) or not, and the OpenCL plugin can hook up with both.

TBD - implement and describe the nested OpenCL implementation discovery process
performed by the OpenCL plugin

### Device enumeration by plugins
Devices from all bound plugins are queried and listed as and when required, eg:
during device selection in device_selector.
The trace shows the PI API calls made when using SYCL_PI_TRACE=-1.
(TBD - Add the trace to list all available devices when plugins are successfully
bound.)

### Plugin Unloading
The plugins not chosen to be connected to should be unloaded. piInitializePlugins()
can be called to load and bound the necessary plugins. In addition, piTearDown()
can be called when plugins are not needed any more. It notifies each
plugin to start performing its own tear-down process such as global memory
deallocation. In the future, piTearDown() can include any other jobs that need to
be done before the plugin is unloaded from memory. Possibly, a
notification of the plugin unloading to lower-level plugins can be added so that
they can clean up their own memory [TBD].
After piTearDown() is called, the plugin can be safely unloaded by calling unload(),
which is going to invoke OS-specific system calls to remove the dynamic library
from memory.

Each plugin should not create global variables that require non-trivial
destructor. Pointer variables with heap memory allocation is a good example
to be created at the global scope. A std::vector object is not. piTearDown
will take care of deallocation of these global variables safely.

## PI API Specification

PI interface is logically divided into few subsets:
- **Core API** which must be implemented by all plugins for DPC++ runtime to be
able to operate on the corresponding device. The core API further breaks down
into
  - **OpenCL-based** APIs which have OpenCL origin and semantics
  - **Extension** APIs which don't have counterparts in the OpenCL
- **Interoperability API** which allows interoperability with underlying
runtimes such as OpenCL.

See [pi.h](../../include/CL/sycl/detail/pi.h) header for the full list and
descriptions of PI APIs.

### The Core OpenCL-based PI APIs

This subset defines functions representing core functionality,
such as device memory management, kernel creation and parameter setting,
enqueuing kernel for execution, etc. Functions in this subset fully match
semantics of the corresponding OpenCL functions, for example:

    piKernelCreate
    piKernelRelease
    piKernelSetArg

### The Extension PI APIs

Those APIs don't have OpenCL counter parts and require full specification. For
example, the function below selects the most appropriate device binary based
on runtime information and the binary's characteristics
```
pi_result piextDeviceSelectBinary(
  pi_device           device,
  pi_device_binary *  binaries,
  pi_uint32           num_binaries,
  pi_device_binary *  selected_binary);
```

PI also defines few types and string tags to describe a device binary image.
Those are used to communicate to plugins information about the images where it
is needed, currently only in the above function. The main
type is ```pi_device_binary```, whose detailed description can also be found
in the header.  The layout of this type strictly matches the layout of the
corresponding device binary descriptor type defined in the
```clang-offload-wrapper``` tool which wraps device binaries into a host
object for further linkage. The wrapped binaries reside inside this descriptor
in a data section.

### The Interoperability PI APIs

These are APIs needed to implement DPC++ runtime interoperability with
underlying "native" device runtimes such as OpenCL.
There are some OpenCL interoperability APIs, which are to be implemented
by the OpenCL PI plugin only. These APIs match semantics of the corresponding
OpenCL APIs exactly.
For example:

```
pi_result piclProgramCreateWithSource(
  pi_context        context,
  pi_uint32         count,
  const char **     strings,
  const size_t *    lengths,
  pi_program *      ret_program);
```

Some interoperability extension APIs have been added to get native runtime
handles from the backend-agnostic PI Objects or to create PI Objects using the
native handles. Eg:

```
pi_result piextDeviceGetNativeHandle(
  pi_device device,
  pi_native_handle *nativeHandle);

pi_result piextDeviceCreateWithNativeHandle(
  pi_native_handle nativeHandle,
  pi_device *device);

```

### PI Extension mechanism

TBD This section describes a mechanism for DPC++ or other runtimes to detect
availability of and obtain interfaces beyond those defined by the PI dispatch.

TBD Add API to query PI version supported by plugin at runtime.
