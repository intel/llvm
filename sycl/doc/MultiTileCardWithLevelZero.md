# Considerations for programming to multi-tile and multi-card under Level-Zero backend

## 1 Devices discovery
### 1.1 Root-devices

Intel GPUs are represented as SYCL GPU devices, root-devices.
The discovery of root-devices is best with "sycl-ls" tool, for example:
	
```
$ sycl-ls
[opencl:0] GPU : Intel(R) OpenCL HD Graphics 3.0 [21.19.19792]
[opencl:0] CPU : Intel(R) OpenCL 2.1 [2020.11.11.0.03_160000]
[level_zero:0] GPU : Intel(R) Level-Zero 1.1 [1.1.19792]
[host:0] HOST: SYCL host platform 1.2 [1.2]
```

Note that "sycl-ls" shows all devices from all platforms of all SYCL backends that are seen
by SYCL runtime. Thus in the example above there is CPU (managed by OpenCL backend) and 2!
GPUs corresponding to the single physical GPU (managed by either OpenCL or Level-Zero backend).
There are few ways to filter observable root-devices.
	
One is using environment variable SYCL_DEVICE_FILTER described in [EnvironmentVariables.md](https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md)
```
$ SYCL_DEVICE_FILTER=level_zero sycl-ls
[level_zero:0] GPU : Intel(R) Level-Zero 1.1 [1.1.19792]
```
Another way is to use similar SYCL API described in [SYCL\_EXT\_ONEAPI\_FILTER\_SELECTOR](extensions/supported/SYCL_EXT_ONEAPI_FILTER_SELECTOR.asciidoc)
E.g. `filter_selector("level_zero")` will only see Level-Zero operated devices.

If there are multiple GPUs in a system then they will be seen as multiple different root-devices.
On Linux these would be multiple SYCL root-devices of the same SYCL platform (representing Level-Zero driver).
On Windows these would appear as root-devices of multiple different SYCL platforms (Level-Zero drivers).

`CreateMultipleRootDevices=N NEOReadDebugKeys=1` evironment variables can be used to emulate multiple GPU cards, e.g.
```
$ CreateMultipleRootDevices=2 NEOReadDebugKeys=1 SYCL_DEVICE_FILTER=level_zero sycl-ls
[level_zero:0] GPU : Intel(R) Level-Zero 1.1 [1.1.19792]
[level_zero:1] GPU : Intel(R) Level-Zero 1.1 [1.1.19792]
```
	
### 1.2 Sub-devices
	
Some Intel GPU HW is composed of multiple tiles, e.g. 4 tile ATS.
The root-device in such cases can be partitioned to sub-devices, each corresponding to the physical tiles.

``` C++	
try {
  vector<device> SubDevices = RootDevice.create_sub_devices<
  cl::sycl::info::partition_property::partition_by_affinity_domain>(
  cl::sycl::info::partition_affinity_domain::next_partitionable);
}
```

Each call to `create_sub_devices` will return exactly the same sub-devices and in the persistent order.
To control what sub-devices are exposed by Level-Zero UMD one can use ZE_AFFINITY_MASK environment variable.

NOTE: The `partition_by_affinity_domain` is the only partitioning supported for Intel GPU.
Similar `next_partitionable` and `numa` are the only partitioning properties supported (both doing the same thing).

`CreateMultipleSubDevices=N NEOReadDebugKeys=1` environment variables can be used to emulate multiple tiles of a GPU.

## 2 Context	
	
Contexts are used for resources isolation and sharing. A SYCL context may consist of one or multiple devices.
Both root-devices and sub-devices can be within single context, but they all should be of the same SYCL platform.
A SYCL program (kernel_bundle) created against a context with multiple devices will be built to each of the root-devices in the context.
For context that consists of multiple sub-devices of the same root-device only single build (to that root-device) is needed.
	
## 3 Memory
### 3.1 USM

There are multiple ways to allocate memory:

`malloc_device`:
- Allocation can only be accessed by the specified device but not by other devices in the context nor by host.
- The data stays on the device all the time and thus is the fastest available for kernel execution.
- Explicit copy is needed for transfer data to the host or other devices in the context.
				   
`malloc_host`:
- Allocation can be accessed by the host and any other device in the context.
- The data stays on the host all the time and is accessed via PCI from the devices.
- No explicit copy is needed for synchronizing of the data with the host or devices.
			   
`malloc_shared`:
- Allocation can be accessed by the host and the specified device only.
- The data can migrate (operated by the Level-Zero driver) between the host and the device for faster access.
- No explicit copy is necessary for synchronizing between the host and the device, but it is needed for other devices in the context.
				   
NOTE: Memory allocated against a root-device is accessible by all of its sub-devices (tiles).
So if operating on a context with multiple sub-devices of the same root-device then you can use `malloc_device` on that root-device instead of using the slower `malloc_host`.
Remember that if using `malloc_device` you'd need an explicit copy out to the host if it necessary to see data there.
					   
### 3.2 Buffer
	
SYCL buffers are also created against a context and under the hood are mapped to the Level-Zero USM allocation discussed above.
The current mapping is following:

- For integrated device the allocations are made on host, and are accessible by the host and the device without any copying.
- Memory buffers for context with sub-devices of the same root-device (possibly including the root-device itself) are allocated on that root-device.
   Thus they are readily accessible by all the devices in such context. The synchronization with the host is performed by SYCL RT with map/unmap doing implicit copies when necessary.
- Memory buffers for context with devices from different root-devices in it are allocated on host (thus made accessible to all devices).
	
## 4 Queue

SYCL queue is always attached to a single device in a possibly multi-device context.
Some typical scenarios are the following (from most performant to least performant):

**A.** Context with a single sub-device in it and the queue is attached to that sub-device (tile)
- The execution/visibility is limited to the single sub-device only
- Expected to offer the best performance per tile
- Example:
``` C++	
try {
  vector<device> SubDevices = ...;
  for (auto &D : SubDevices) {
    // Each queue is in its own context, no data sharing across them.
    auto Q = queue(D);
    Q.submit([&](handler& cgh) {...});
  }
}
```

**B.** Context with multiple sub-devices of the same root-device (multi-tile)
- Queues are to be attached to the sub-devices effectively implementing "explicit scaling"
- The root-device should not be passed to such context for better performance
- Example:
``` C++	
try {
  vector<device> SubDevices = ...;
  auto C = context(SubDevices);
  for (auto &D : SubDevices) {
    // All queues share the same context, data can be shared across queues.
    auto Q = queue(C, D);
    Q.submit([&](handler& cgh) {...});
  }
}
```
	
**C.** Context with a single root-device in it and the queue is attached to that root-device
- The work will be automatically distributed across all sub-devices/tiles via "implicit scaling" by the driver
- The most simple way to enable multi-tile HW but doesn't offer possibility to target specific tiles
- Example:
``` C++	
try {
  // The queue is attached to the root-device, driver distributes to sub-devices, if any.
  auto D = device(gpu_selector{});
  auto Q = queue(D);
  Q.submit([&](handler& cgh) {...});
}
```
		
**D.** Contexts with multiple root-devices (multi-card)
- The most unrestrictive context with queues attached to different root-devices
- Offers most sharing possibilities at the cost of slow access through host memory or explicit copies needed
- Example:
``` C++	
try {
  auto P = platform(gpu_selector{});
  auto RootDevices = P.get_devices();
  auto C = context(RootDevices);
  for (auto &D : RootDevices) {
    // Context has multiple root-devices, data can be shared across multi-card (requires explict copying)
    auto Q = queue(C, D);
    Q.submit([&](handler& cgh) {...});
  }
}
```

Depending on the chosen programming model (A,B,C,D) and algorithm used make sure to do proper memory allocation/synchronization.
				
## 5 Examples
	
These are few examples of programming to multiple tiles and multiple cards:
- https://github.com/jeffhammond/PRK/blob/dpct/Cxx11/dgemm-multigpu-onemkl.cc
- https://github.com/pvelesko/PPP/tree/master/languages/c%2B%2B/sycl/gpu2gpu
