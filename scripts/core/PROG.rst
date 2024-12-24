
<%
    OneApi=tags['$OneApi']
    x=tags['$x']
    X=x.upper()
%>
.. _core-programming-guide:

========================
 Programming Guide
========================

Platforms and Devices
======================

The oneAPI Unified Runtime API architecture exposes both physical and logical abstraction of the underlying devices capabilities.
The device, sub-device and memory are exposed at physical level while command queues, events and
synchronization methods are defined as logical entities.
All logical entities will be bound to device level physical capabilities.

Device discovery APIs enumerate the accelerators functional features.
These APIs provide interface to query information like compute unit count within the device or sub device, 
available memory and affinity to the compute, user managed cache size and work submission command queues.

Platforms
---------

A platform object represents a collection of physical devices in the system accessed by the same driver.

- The application may query the number of platforms installed on the system, and their respective handles, using ${x}PlatformGet.
- More than one platform may be available in the system. For example, one platform may support two GPUs from one vendor, another platform supports a GPU from a different vendor, and finally a different platform may support an FPGA.
- Platform objects are read-only, global constructs. i.e. multiple calls to ${x}PlatformGet will return identical platform handles.
- A platform handle is primarily used during device discovery and during creation and management of contexts.

Device
------

A device object represents a physical device in the system that supports the platform.

- The application may query the number devices supported by a platform, and their respective handles, using ${x}DeviceGet.
- Device objects are read-only, global constructs. i.e. multiple calls to ${x}DeviceGet will return identical device handles.
- A device handle is primarily used during creation and management of resources that are specific to a device.
- Device may expose sub-devices that allow finer-grained control of physical or logical partitions of a device.

The following diagram illustrates the relationship between the platform, device, context and other objects described in this document.

.. image:: ../images/runtime_object_hier.png

Initialization and Discovery
----------------------------

.. parsed-literal::

    // Discover all available adapters
    uint32_t adapterCount = 0;
    ${x}AdapterGet(0, nullptr, &adapterCount);
    std::vector<${x}_adapter_handle_t> adapters(adapterCount);
    ${x}AdapterGet(adapterCount, adapters.data(), nullptr);

    // Discover all the platform instances
    uint32_t platformCount = 0;
    ${x}PlatformGet(adapters.data(), adapterCount, 0, nullptr, &platformCount);

    std::vector<${x}_platform_handle_t> platforms(platformCount);
    ${x}PlatformGet(adapters.data(), adapterCount, platform.size(), platforms.data(), &platformCount);

    // Get number of total GPU devices in the platform
    uint32_t deviceCount = 0;
    ${x}DeviceGet(platforms[0], ${X}_DEVICE_TYPE_GPU, &deviceCount, nullptr, 
                  nullptr);

    // Get handles of all GPU devices in the platform
    std::vector<${x}_device_handle_t> devices(deviceCount);
    ${x}DeviceGet(platforms[0], ${X}_DEVICE_TYPE_GPU, &deviceCount, 
                  devices.data(), devices.size());

Device handle lifetime
----------------------

Device objects are reference-counted, using ${x}DeviceRetain and ${x}DeviceRelease.
The ref-count of a device is automatically incremented when a device is obtained by ${x}DeviceGet.
After a device is no longer needed by the application it must call ${x}DeviceRelease.
When the ref-count of the underlying device handle becomes zero then that device object is deleted.
Note that a Unified Runtime adapter may internally increment and decrement a device's ref-count.
So after the call to ${x}DeviceRelease below, the device may stay active until other
objects using it, such as a command-queue, are deleted. However, an application
may not use the device after it releases its last reference.

.. parsed-literal::

    // Get the handle of the first GPU device in the platform
    ${x}_device_handle_t hDevice;
    uint32_t deviceCount = 1;
    ${x}DeviceGet(hPlatforms[0], ${X}_DEVICE_TYPE_GPU, &deviceCount, &hDevice, 1);
    ${x}DeviceRelease(hDevice);


Retrieve info about device
--------------------------

The ${x}DeviceGetInfo can return various information about the device.
In case where the info size is only known at runtime then two calls are needed, where first will retrieve the size.

.. parsed-literal::

    // Size is known beforehand
    ${x}_device_type_t deviceType;
    ${x}DeviceGetInfo(hDevice, ${X}_DEVICE_INFO_TYPE, 
                      sizeof(${x}_device_type_t), &deviceType, nullptr);

    // Size is only known at runtime
    size_t infoSize;
    ${x}DeviceGetInfo(hDevice, ${X}_DEVICE_INFO_NAME, 0, &infoSize, nullptr);
    
    std::string deviceName;
    DeviceName.resize(infoSize);
    ${x}DeviceGetInfo(hDevice, ${X}_DEVICE_INFO_NAME, infoSize, 
                      deviceName.data(), nullptr);

Device partitioning into sub-devices
------------------------------------

${x}DevicePartition partitions a device into a sub-device. The exact representation and
characteristics of the sub-devices are device specific, but normally they each represent a
fixed part of the parent device, which can explicitly be programmed individually.

.. parsed-literal::

    ${x}_device_handle_t hDevice;
    ${x}_device_partition_property_t prop;
    prop.value.affinity_domain = ${X}_DEVICE_AFFINITY_DOMAIN_FLAG_NEXT_PARTITIONABLE;

    ur_device_partition_properties_t properties{
        ${X}_STRUCTURE_TYPE_DEVICE_PARTITION_PROPERTIES,
        nullptr,
        &prop,
        1,
    };

    uint32_t count = 0;
    std::vector<${x}_device_handle_t> subDevices;
    ${x}DevicePartition(hDevice, &properties, 0, nullptr, &count);

    if (count > 0) {
        subDevices.resize(count);
        ${x}DevicePartition(Device, &properties, count, &subDevices.data(), 
                            nullptr);
    }

The returned sub-devices may be requested for further partitioning into sub-sub-devices, and so on.
An implementation will return "0" in the count if no further partitioning is supported.

.. parsed-literal::

    uint32_t count;
    ${x}DevicePartition(subDevices[0], &properties, 0, nullptr, &count);
    if(count == 0){
        // no further partitioning allowed
    }
    
Contexts
========

Contexts serve the purpose of resource sharing (between devices in the same context),
and resource isolation (ensuring that resources do not cross context
boundaries). Resources such as memory allocations, events, and programs are
explicitly created against a context.

.. parsed-literal::

    uint32_t deviceCount = 1;
    ${x}_device_handle_t hDevice;
    ${x}DeviceGet(hPlatform, ${X}_DEVICE_TYPE_GPU, &deviceCount, &hDevice, 
                  nullptr);

    // Create a context
    ${x}_context_handle_t hContext;
    ${x}ContextCreate(1, &hDevice, nullptr, &hContext);

    // Operations on this context
    // ...

    // Release the context handle
    ${x}ContextRelease(hContext);    

Object Queries
==============

Queries to get information from API objects follow a common pattern. The entry
points for this are generally of the form:

.. code-block::

   ObjectGetInfo(ur_object_handle_t hObject, ur_object_info_t propName,
                 size_t propSize, void *pPropValue, size_t *pPropSizeRet)

where ``propName`` selects the information to query out. The object info enum
representing possible queries will generally be found in the enums section of
the relevant object. Some info queries would be difficult or impossible to
support for certain backends, these are denoted with [optional-query] in the
enum description. Using any enum marked optional in this way may result in
${X}_RESULT_ERROR_UNSUPPORTED_ENUMERATION if the adapter doesn't support it.

Programs and Kernels
====================

There are two constructs we need to prepare code for execution on the device:

* Programs serve as containers for device code. They typically encapsulate a
  collection of functions and global variables represented in an intermediate
  language, and one or more device-native binaries compiled from that
  collection.
* Kernels represent a handle to a function within a program that can be
  launched on a device.


Programs
--------

Programs can be constructed with an intermediate language binary or a
device-native binary. Programs constructed with IL must be further compiled
through either ${x}ProgramCompile and ${x}ProgramLink or ${x}ProgramBuild
before they can be used to create a kernel object.

.. parsed-literal::

    // Create a program with IL
    ${x}_program_handle_t hProgram;
    ${x}ProgramCreateWithIL(hContext, ILBin, ILBinSize, nullptr, &hProgram);

    // Build the program.
    ${x}ProgramBuild(hContext, hProgram, nullptr);

The diagram below shows the possible paths to obtaining a program that can be
used to create a kernel:

.. image:: ../images/programs.png

Kernels
-------

A Kernel is a reference to a kernel within a program and it supports both
explicit and implicit kernel arguments along with data needed for launch.

.. parsed-literal::

    // Create kernel object from program
    ${x}_kernel_handle_t hKernel;
    ${x}KernelCreate(hProgram, "addVectors", &hKernel);
    ${x}KernelSetArgMemObj(hKernel, 0, nullptr, A);
    ${x}KernelSetArgMemObj(hKernel, 1, nullptr, B);
    ${x}KernelSetArgMemObj(hKernel, 2, nullptr, C);

Queue and Enqueue
=================

Queue objects are used to submit work to a given device. Kernels
and commands are submitted to queue for execution using Enqueue commands:
such as ${x}EnqueueKernelLaunch, ${x}EnqueueMemBufferWrite. Enqueued kernels
and commands can be executed in order or out of order depending on the
queue's property ${X}_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE when the
queue is created. If a queue is out of order, the queue may internally do some
scheduling of work to achieve concurrency on the device, while honouring the
event dependencies that are passed to each Enqueue command.

.. parsed-literal::

    // Create an out of order queue for hDevice in hContext
    ${x}_queue_handle_t hQueue;
    ${x}QueueCreate(hContext, hDevice,
                    ${X}_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE, &hQueue);

    // Launch a kernel with 3D workspace partitioning
    const uint32_t nDim = 3;
    const size_t gWorkOffset = {0, 0, 0};
    const size_t gWorkSize = {128, 128, 128};
    const size_t lWorkSize = {1, 8, 8}; 
    ${x}EnqueueKernelLaunch(hQueue, hKernel, nDim, gWorkOffset, gWorkSize, 
                            lWorkSize, 0, nullptr, nullptr);

Queue object lifetime
---------------------

Queue objects are reference-counted. If an application or thread needs to
retain access to a queue created by another application or thread, it can call
${x}QueueRetain. An application must call ${x}QueueRelease
when a queue object is no longer needed. When a queue object's reference count becomes
zero, it is deleted by the runtime.

Native Driver Access
----------------------------------

The runtime API provides accessors for native handles.
For example, given a ${x}_program_handle_t, we can
call ${x}ProgramGetNativeHandle to retrieve a ${x}_native_handle_t.
We can then leverage a platform extension to convert the
native handle to a driver handle. For example, OpenCL platform
may expose an extension ${x}ProgramCreateWithNativeHandle to retrieve
a cl_program.

Memory
======

UR Mem Handles
--------------

A ${x}_mem_handle_t can represent an untyped memory buffer object, created by
${x}MemBufferCreate, or a memory image object, created by ${x}MemImageCreate.
A ${x}_mem_handle_t manages the internal allocation and deallocation of native
memory objects across all devices in a ${x}_context_handle_t. A
${x}_mem_handle_t may only be used by queues that share the same
${x}_context_handle_t. 

If multiple queues in the same ${x}_context_handle_t use the same
${x}_mem_handle_t across dependent commands, a dependency must be defined by the
user using the enqueue entry point's phEventWaitList parameter. Provided that
dependencies are explicitly passed to UR entry points, a UR adapter will manage
memory migration of native memory objects across all devices in a context, if
memory migration is indeed necessary in the backend API.

.. parsed-literal::

    // Q1 and Q2 are both in hContext
    ${x}_mem_handle_t hBuffer;
    ${x}MemBufferCreate(hContext,,,,&hBuffer);
    ${x}EnqueueMemBufferWrite(Q1, hBuffer,,,,,,, &outEv);
    ${x}EnqueueMemBufferRead(Q2, hBuffer,,,,, 1, &outEv /*phEventWaitList*/, );

As such, the buffer written to in ${x}EnqueueMemBufferWrite can be
successfully read using ${x}EnqueueMemBufferRead from another queue in the same
context, since the event associated with the write operation has been passed as
a dependency to the read operation.

Memory Pooling
----------------------------------

The ${x}USMPoolCreate function explicitly creates memory pools and returns ${x}_usm_pool_handle_t.
${x}_usm_pool_handle_t can be passed to ${x}USMDeviceAlloc, ${x}USMHostAlloc and ${x}USMSharedAlloc
through ${x}_usm_desc_t structure. Allocations that specify different pool handles must be
isolated and not reside on the same page. Memory pool is subject to limits specified during pool creation.

Even if no ${x}_usm_pool_handle_t is provided to an allocation function, each adapter may still perform memory pooling.
