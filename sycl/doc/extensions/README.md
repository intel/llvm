# Extensions

The directory contains documents that describe DPC++ extensions to SYCL
specification.

DPC++ extensions status:

|  Extension  |    Status   |   Comment   |
|-------------|:------------|:------------|
| [SYCL_INTEL_bitcast](Bitcast/SYCL_INTEL_bitcast.asciidoc)                                                                   | Supported                                 | As sycl::detail::bit_cast |
| [C and C++ Standard libraries support](C-CXX-StandardLibrary/C-CXX-StandardLibrary.rst)                                     | Partially supported(OpenCL: CPU, GPU)     | |
| [SYCL_INTEL_data_flow_pipes](DataFlowPipes/data_flow_pipes.asciidoc)                                                        | Partially supported(OpenCL: ACCELERATOR)  | kernel_host_pipe_support part is not implemented |
| [SYCL_INTEL_deduction_guides](deduction_guides/SYCL_INTEL_deduction_guides.asciidoc)                                        | Supported                                 | |
| [SYCL_INTEL_device_specific_kernel_queries](DeviceSpecificKernelQueries/SYCL_INTEL_device_specific_kernel_queries.asciidoc) | Proposal                                  | |
| [SYCL_INTEL_enqueue_barrier](EnqueueBarrier/enqueue_barrier.asciidoc)                                                       | Supported(OpenCL, Level Zero)             | |
| [SYCL_INTEL_extended_atomics](ExtendedAtomics/SYCL_INTEL_extended_atomics.asciidoc)                                         | Supported(OpenCL: CPU, GPU)               | |
| [SYCL_INTEL_group_algorithms](GroupAlgorithms/SYCL_INTEL_group_algorithms.asciidoc)                                         | Supported(OpenCL; CUDA)                   | |
| [GroupMask](GroupMask/GroupMask.asciidoc)                                                                                   | Proposal                                  | |
| [FPGA selector](IntelFPGA/FPGASelector.md)                                                                                  | Supported                                 | |
| [FPGA reg](IntelFPGA/FPGAReg.md)                                                                                            | Supported(OpenCL: ACCELERATOR)            | |
| [SYCL_INTEL_kernel_restrict_all](KernelRestrictAll/SYCL_INTEL_kernel_restrict_all.asciidoc)                                 | Supported(OpenCL)                         | |
| [SYCL_INTEL_attribute_style](KernelRHSAttributes/SYCL_INTEL_attribute_style.asciidoc)                                       | Proposal                                  | |
| [Queue Order Properties](OrderedQueue/OrderedQueue_v2.adoc)                                                                 | Supported                                 | |
| [Queue Shortcuts](QueueShortcuts/QueueShortcuts.adoc)                                                                       | Supported                                 | |
| [Reductions for ND-Range Parallelism](Reduction/Reduction.md)                                                               | Partially supported(OpenCL: CPU, GPU; CUDA) | Not supported: multiple reduction vars, multi-dimensional reduction vars |
| [SYCL_INTEL_relax_standard_layout](RelaxStdLayout/SYCL_INTEL_relax_standard_layout.asciidoc)                                | Supported                                 | |
| [SYCL_INTEL_reqd_work_group_size](ReqdWorkGroupSize/SYCL_INTEL_reqd_work_group_size.asciidoc)                               | Supported(OpenCL: CPU, GPU)               | |
| [SPV_INTEL_function_pointers](SPIRV/SPV_INTEL_function_pointers.asciidoc)                                                   | Supported(OpenCL: CPU, GPU; HOST)         | |
| [SPV_INTEL_inline_assembly](SPIRV/SPV_INTEL_inline_assembly.asciidoc)                                                       | Supported(OpenCL: GPU)                    | |
| [LocalMemory](LocalMemory/LocalMemory.asciidoc)                                                                             | Supported(OpenCL; CUDA)                   | Revision 1 of the spec is fully supported, future revisions are expected to expand the functionality |
| [SYCL_INTEL_static_local_memory_query](StaticLocalMemoryQuery/SYCL_INTEL_static_local_memory_query.asciidoc)                | Proposal                                  | |
| [SYCL_INTEL_sub_group_algorithms](SubGroupAlgorithms/SYCL_INTEL_sub_group_algorithms.asciidoc)                              | Partially supported(OpenCL: CPU, GPU)     | Features from SYCL_INTEL_group_algorithms extended to sub-groups |
| [Sub-groups for NDRange Parallelism](SubGroupNDRange/SubGroupNDRange.md)                                                    | Deprecated(OpenCL: CPU, GPU)              | |
| [Sub-groups](SubGroup/SYCL_INTEL_sub_group.asciidoc)                                                                        | Partially supported(OpenCL)               | Not supported: auto/stable sizes, stable query, compiler flags |
| [SYCL_INTEL_unnamed_kernel_lambda](UnnamedKernelLambda/SYCL_INTEL_unnamed_kernel_lambda.asciidoc)                           | Supported(OpenCL)                         | |
| [Unified Shared Memory](USM/USM.adoc)                                                                                       | Supported(OpenCL)                         | |
| [Use Pinned Memory Property](UsePinnedMemoryProperty/UsePinnedMemoryPropery.adoc)                                           | Supported                                 | |
| [Level-Zero backend specification](LevelZeroBackend/LevelZeroBackend.md)                                           	      | Supported                                 | |
| [ITT annotations support](ITTAnnotations/ITTAnnotations.rst) | Supported | |
| [SYCL_EXT_ONEAPI_DEVICE_IF](DeviceIf/device_if.asciidoc) | Proposal | |
| [SYCL_INTEL_group_sort](GroupAlgorithms/SYCL_INTEL_group_sort.asciidoc)                                                     | Proposal                                  | |
| [Invoke SIMD](InvokeSIMD/InvokeSIMD.asciidoc)                                                                               | Proposal                                  | |
| [Uniform](Uniform/Uniform.asciidoc)                                                                                         | Proposal                                  | |
| [Assert](Assert/SYCL_ONEAPI_ASSERT.asciidoc) | Proposal | |
| [Matrix](Matrix/dpcpp-joint-matrix.asciidoc)                                                                        | Partially supported(AMX AOT)               | Not supported: dynamic-extent, wg and wi scopes, layouts other than packed|
| [SYCL_INTEL_free_function_queries](FreeFunctionQueries/SYCL_INTEL_free_function_queries.asciidoc)                           | Supported (experimental)                  | |

Legend:

|  Keyword    |   Meaning   |
|-------------|:------------|
|  Proposal                        | A document describing an extension is published, but the extension is not supported yet |
|  Supported                       | An extension is supported |
|  Partially supported             | An extension is partially supported, see comments column for more info |
|  Deprecated                      | An extension is deprecated and can be removed in future versions |
|  (API: DeviceType1, DeviceType2) | An extension is supported when specific combination of API and device types are used. If device type or API are not mentioned then an extension is supported on any device type or API. API can be OpenCL, CUDA, HOST. DeviceType can be CPU, GPU, ACCELERATOR |


See [User Manual](../UsersManual.md) to find information how to enable extensions.
