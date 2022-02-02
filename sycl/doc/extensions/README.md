# Extensions

The directory contains documents that describe DPC++ extensions to SYCL
specification.

DPC++ extensions status:

|  Extension  |    Status   |   Comment   |
|-------------|:------------|:------------|
| [SYCL_INTEL_group_algorithms](GroupAlgorithms/SYCL_INTEL_group_algorithms.asciidoc)                                         | Deprecated                                | |
| [GroupMask](GroupMask/GroupMask.asciidoc)                                                                                   | Proposal                                  | |
| [Reductions for ND-Range Parallelism](Reduction/Reduction.md)                                                               | Partially supported(OpenCL: CPU, GPU; CUDA) | Not supported: multiple reduction vars, multi-dimensional reduction vars |
| [SPV_INTEL_function_pointers](SPIRV/SPV_INTEL_function_pointers.asciidoc)                                                   | Supported(OpenCL: CPU, GPU; HOST)         | |
| [SPV_INTEL_inline_assembly](SPIRV/SPV_INTEL_inline_assembly.asciidoc)                                                       | Supported(OpenCL: GPU)                    | |
| [SYCL_INTEL_static_local_memory_query](StaticLocalMemoryQuery/SYCL_INTEL_static_local_memory_query.asciidoc)                | Proposal                                  | |
| [Sub-groups for NDRange Parallelism](SubGroupNDRange/SubGroupNDRange.md)                                                    | Deprecated(OpenCL: CPU, GPU)              | |
| [Sub-groups](SubGroup/SYCL_INTEL_sub_group.asciidoc)                                                                        | Deprecated                                | |
| [Property List](PropertyList/SYCL_EXT_ONEAPI_property_list.asciidoc)                                                        | Proposal                                  | |
| [DiscardQueueEvents](DiscardQueueEvents/SYCL_EXT_ONEAPI_DISCARD_QUEUE_EVENTS.asciidoc) | Proposal | |

Legend:

|  Keyword    |   Meaning   |
|-------------|:------------|
|  Proposal                        | A document describing an extension is published, but the extension is not supported yet |
|  Supported                       | An extension is supported |
|  Partially supported             | An extension is partially supported, see comments column for more info |
|  Deprecated                      | An extension is deprecated and can be removed in future versions |
|  (API: DeviceType1, DeviceType2) | An extension is supported when specific combination of API and device types are used. If device type or API are not mentioned then an extension is supported on any device type or API. API can be OpenCL, CUDA, HOST. DeviceType can be CPU, GPU, ACCELERATOR |


See [User Manual](../UsersManual.md) to find information how to enable extensions.
