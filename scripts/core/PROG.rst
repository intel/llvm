
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

    // Discover all the platform instances
    uint32_t platformCount = 0;
    ${x}PlatformGet(0, nullptr, &platformCount);

    std::vector<${x}_platform_handle_t> platforms(platformCount);
    ${x}PlatformGet(platform.size(), platforms.data(), &platformCount);

    // Get number of total GPU devices in the platform
    uint32_t deviceCount = 0;
    ${x}DeviceGet(platforms[0], ${X}_DEVICE_TYPE_GPU, &deviceCount, nullptr, nullptr);

    // Get handles of all GPU devices in the platform
    std::vector<${x}_device_handle_t> devices(deviceCount);
    ${x}DeviceGet(platforms[0], ${X}_DEVICE_TYPE_GPU, &deviceCount, devices.data(), devices.size());

Device handle lifetime
----------------------

The device objects are reference-counted, and there are ${x}DeviceRetain and ${x}DeviceRelease.
The ref-count of a device is automatically incremented when device is obtained by ${x}DeviceGet.
After device is no longer needed to the application it must call to ${x}DeviceRelease.
When ref-count of the underlying device handle becomes zero then that device object is deleted.
Note, that besides the application itself, the Unified Runtime may increment and decrement ref-count on its own.
So, after the call to ${x}DeviceRelease below, the device may stay alive until other
objects attached to it, like command-queues, are deleted. But application may not use the device
after it released its own reference.

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
    ${x}DeviceGetInfo(hDevice, ${X}_DEVICE_INFO_TYPE, sizeof(${x}_device_type_t), &deviceType, nullptr);

    // Size is only known at runtime
    size_t infoSize;
    ${x}DeviceGetInfo(hDevice, ${X}_DEVICE_INFO_NAME, 0, &infoSize, nullptr);
    
    std::string deviceName;
    DeviceName.resize(infoSize);
    ${x}DeviceGetInfo(hDevice, ${X}_DEVICE_INFO_NAME, infoSize, deviceName.data(), nullptr);

Device partitioning into sub-devices
------------------------------------

The ${x}DevicePartition could partition a device into sub-device. The exact representation and
characteristics of the sub-devices are device specific, but normally they each represent a
fixed part of the parent device, which can explicitly be programmed individually.

.. parsed-literal::

    ${x}_device_handle_t hDevice;
    ${x}_device_partition_property_t properties[] = { 
               ${X}_DEVICE_PARTITION_BY_AFFINITY_DOMAIN,
               ${X}_DEVICE_AFFINITY_DOMAIN_FLAG_NEXT_PARTITIONABLE,
               0};

    uint32_t count = 0;
    std::vector<${x}_device_handle_t> subDevices;
    ${x}DevicePartition(hDevice, &properties, &count, nullptr, nullptr);

    if (count > 0) {
        subDevices.resize(count);
        ${x}DevicePartition(Device, &properties, &count, &subDevices.data(), nullptr);
    }

The returned sub-devices may be requested for further partitioning into sub-sub-devices, and so on.
An implementation would return "0" in the count if no further partitioning is supported.

.. parsed-literal::

    uint32_t count = 1;
    ${x}_device_handle_t hSubSubDevice;
    ${x}DevicePartition(subDevices[0], properties, &count, &hSubSubDevice, nullptr);

Contexts
========

Contexts are serving the purpose of resources sharing (between devices in the same context),
and resources isolation (resources do not cross context boundaries). Resources such as memory allocations,
events, and programs are explicitly created against a context. A trivial work with context looks like this:

.. parsed-literal::

    uint32_t deviceCount = 1;
    ${x}_device_handle_t hDevice;
    ${x}DeviceGet(hPlatform, ${X}_DEVICE_TYPE_GPU, &deviceCount, &hDevice, nullptr);

    // Create a context
    ${x}_context_handle_t hContext;
    ${x}ContextCreate(1, &hDevice, &hContext);

    // Operations on this context
    // ...

    // Release the context handle
    ${x}ContextRelease(hContext);    

Modules and Programs
====================

There are multiple levels of constructs needed for executing kernels on the device:

* Modules represent a single translation unit that consists of kernels and globals that have been compiled together.
* Programs represent one or more modules that have been linked together.
* Kernels represent the kernel within a program that will be launched onto the device.

.. image:: ../images/modules_programs.png

Modules and Programs
--------------------

A module is the compiled code or object for a single compilation unit. Modules can be created from a SPIR-V module. A program
are a collection of modules that are linked together.

.. parsed-literal::

    // Create module
    ${x}_module_handle_t hModule;
    ${x}ModuleCreate(hContext, (const void*)pIL, length, nullptr, nullptr, nullptr, hModule);

    // Create program from module
    ${x}_program_handle_t hProgram;
    ${x}ProgramCreate(hContext, 1, &hModule, nullptr, hProgram);


Kernels
-------

A Kernel is a reference to a kernel within a module and it supports both explicit and implicit kernel
arguments along with data needed for launch.

.. parsed-literal::

    // Create kernel object from program
    ${x}_kernel_handle_t hKernel;
    ${x}KernelCreate(hProgram, "addVectors", &hKernel);
    ${x}KernelSetArgMemObj(hKernel, 0, A);
    ${x}KernelSetArgMemObj(hKernel, 1, B);
    ${x}KernelSetArgMemObj(hKernel, 2, C);

Queue and Enqueue
=================

A queue object represents a logic input stream to a device. Kernels 
and commands are submitted to queue for execution using Equeue commands:
such as ${x}EnqueueKernelLaunch, ${x}EnqueueMemBufferWrite. Enqueued kernels
and commands can be executed in order or out of order depending on the
queue's property ${X}_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE when the
queue is created.

.. parsed-literal::

    // Create an out of order queue for hDevice in hContext
    ${x}_queue_handle_t hQueue;
    ${x}QueueCreate(hContext, hDevice, ${X}_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE, &hQueue);

    // Lanuch a kernel with 3D workspace partitioning
    const uint32_t nDim = 3;
    const size_t gWorkOffset = {0, 0, 0};
    const size_t gWorkSize = {128, 128, 128};
    const size_t lWorkSize = {1, 8, 8}; 
    ${x}EnqueueKernelLaunch(hQueue, hKernel, nDim, gWorkOffset, gWorkSize, lWorkSize, 0, nullptr, nullptr);

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

