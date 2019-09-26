# The DPC++ Runtime Plugin Interface.

## Overview
The DPC++ Runtime Plugin Interface (PI) is the interface layer between
device-agnostic part of the DPC++ runtime and the device-specific runtime layers
which control execution on devices. It employs the “plugin” mechanism to bind
to the device specific runtime layers similarly to what is used by libomptarget
or OpenCL.

The picture below illustrates the placement of the PI within the overall DPC++
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

PI is based on a subset of OpenCL 1.2 runtime specification, it follows its
platform, execution and memory models in all aspects except those explicitly
mentioned in this document. A part of PI API types and functions have exact
matches in OpenCL. Whenever there is such a match, the semantics also fully
matches unless the differences are explicitly specified in this document. While
PI has roots in OpenCL, it does have many differences, and the gap is likely
to grow, for example in the areas of memory model and management, program
management.

## Discovery and linkage of PI implementations

![PI implementation discovery](images/PluginDiscovery.svg)

Device discovery phase enumerates all available devices and their features by
querying underlying plugins found in the system. This process is only performed
once before any actual offload is attempted.

### Plugin discovery

Plugins are physically dynamic libraries or shared objects.
The process to discover plugins will follow the following guidelines.

The SYCL Runtime will read the names of the plugins from a configuration file 
at a predetermined location (TBD - Add this location). These plugins are
searched at locations in env LD_LIBRARY_PATH on Linux and env PATH on Windows.
(TBD - Extend to search the plugins at a path relative to the SYCL Runtime
installation directory by using DT_RPATH on Linux. Similar functionality can be
achieved on Windows using SetDllDirectory. This will help avoiding extra setting
of LD_LIBRARY_PATH.)
To avoid any issues with read-only access, an environment variable SYCL_PI_CONFIG
can be set to point to the configuration file which lists the Plugin names. The
enviroment variable if set overrides the predetermined location's config file.
These Plugins will then be searched in LD_LIBRARY_PATH locations.
It is the developer's responsibility to include the plugin names from the
predetermined location's config file to enable discovery of all plugins.
(TBD - Extend to support search in DT_RPATH as above.)
A trace mechanism is provided to log the discovery/ binding/ device
enumeration process. Eg: Display all the plugins being discovered, their
information and supported PI version. List attached devices and their properties.

 TBD - design and describe the process in detail.

#### Plugin binary interface
Plugins should implement all the Interface APIs required for the PI Version
it supports. It will export a function that will return the function pointer
table that contains the list of implemented Interface Function pointers in a
predetermined order defined in pi.h.
In the future, this document will list the minimum set of Interface APIs
to be supported by Plugins. This will also require adding functionality to SYCL
Runtime to work with such limited functionality plugins.

 TBD - list and describe the symbols that a plugin must implement in order to
 be picked up by the SYCL runtime for offload.

#### Binding a Plugin
Plugins expose the information of supported PI API version.
The Plugin Interface queries the plugins on the supported PI version and checks
for compatibility.(TBD - Extend to support version compatibility checks without
loading the library. Eg: Changing the plugin name to reflect the supported
Plugin Interface version.)
The Plugin Loader then queries each plugin for the Function Pointer Table
and populates a list of the PI API Function Pointers for each plugin.
The user can select/disable a specific plugin with an environment variable,
SYCL_PI_USE. (TBD - Describe the semantics in a separate section for EV and
trace.)
The information of compatible plugins (with the Function Pointer Table) is
stored in the associated platforms during platform object construction.
The PI API calls are forwarded using this information.
There is pi.def/pi.h file that lists all PI API names that can be called by the
Plugin Interface.

#### OpenCL plugin

OpenCL plugin is a usual plugin from DPC++ runtime standpoint, but its loading
and initialization involves a nested discovery process which finds out available
OpenCL implementations. They can be installed either in the standard Khronos
ICD-compatible way (e.g. listed in files under /etc/OpenCL/vendors on
Linux) or not, and the OpenCL plugin can hook up with both.

TBD - describe the nested OpenCL implementation discovery process performed by
the OpenCL plugin

### Device enumeration by plugins
After the compatible plugins are loaded, the trace will show all available
devices from each plugin. Similarly, the trace can be extended to show the
underlying API calls that each PI plugin call is being directed to.

TBD - Describe the exact API calls to enable device enumeration feature.

### Plugin Unloading
The plugins not chosen to be connected to will be unloaded.

TBD - Unloading a bound plugin.

## PI API Specification

PI interface is logically divided into few subsets:
- **Core API** which must be implemented by all plugins for DPC++ runtime to be
able to operate on the corresponding device. The core API further breaks down
into
  - **OpenCL-based** APIs which have OpenCL origin and semantics
  - **Extension** APIs which don't have counterparts in the OpenCL
- **Interoperability API** which allows interoperability with underlying APIs
such as OpenCL.

See [pi.h](../include/CL/sycl/detail/pi.h) header for the full list and
descriptions of PI APIs. [TBD: link to pi.h doxygen here]

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
underlying "native" device runtimes such as OpenCL. Currently there are only
OpenCL interoperability APIs, which is to be implemented by the OpenCL PI
plugin only.  These APIs match semantics of the corresponding OpenCL APIs
exactly.
For example:

```
pi_result piclProgramCreateWithSource(
  pi_context        context,
  pi_uint32         count,
  const char **     strings,
  const size_t *    lengths,
  pi_program *      ret_program);
```

### PI Extension mechanism

TBD This section describes a mechanism for DPC++ or other runtimes to detect
availability of and obtain interfaces beyond those defined by the PI dispatch.

TBD Add API to query PI version supported by plugin at runtime.
