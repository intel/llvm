# User driven Kernel Fusion

## Context

To support the [user driven kernel fusion extension](https://github.com/intel/llvm/pull/7098) (a presentation can be found [here](https://github.com/oneapi-src/oneAPI-tab/blob/main/tab-dpcpp-onedpl/presentations/oneAPI-TAB-20220727-Kernel-Fusion.pdf)).
Currently, only targets able to consume SPIR-V are supported.

The basic workflow is shown in the diagram below

![Fusion workflow](images/KernelFusionRuntimeWorkflow.svg)

During the SYCL program execution, kernels to be fused are collected in a list in the queue (see [Scheduler Integration](#scheduler-integration) section).
When the fusion of kernels is requested, the SPIR-V modules containing them are passed to the JIT that will perform the fusion process (see [Fusion Process](#fusion-process)).
The fusion JIT will generate a new entry point containing all kernels to fuse and run the optimization pipeline.
The result is a new SPIR-V module that is injected back into the SYCL runtime for execution.

The feature is a pure runtime process, no action at compile time is required at the moment.

### Fusion JIT integration

![Fusion via plugin](images/RuntimeArchitecture-with-fusion.svg)

The JIT is integrated into the SYCL runtime via a companion library to `libsycl.so` (Fusion plugin box).
The runtime communicates with the JIT via 2 main entry points: a context object (`JITContext`) and a `fuseKernels` function.


## Scheduler Integration

### Tasks

The integration of kernel fusion into the scheduling process needs to fulfill three main tasks:
1. Maintain the list of kernels submitted for fusion (the *fusion list*) for each queue.
2. Make sure that the kernel(s) submitted at the end of the fusion process, i.e., either the individual kernels (in case fusion is aborted or `cancel_fusion` is called) or the fused kernel (in case of `complete_fusion`), have the correct requirements & dependencies.
3. Synchronization - the extension proposal outlines a number of scenarios in which kernel fusion must be aborted early for synchronization and to avoid circular dependencies in the SYCL RT execution graph.

To achieve these tasks, the command graph gained a new command node (`KernelFusionCommand`) and the behavior of the queue is modified accordingly.

### KernelFusionCommand

A part of the necessary functionality is implemented as part of a new `KernelFusionCommand` as a sub-class of the `Command` class. This class fulfills two of the tasks listed above, as it maintains the fusion list for a queue (1) and allows the detection of scenarios in the scheduler where synchronization is necessary (3), as set out by the kernel fusion extension proposal.

For each `queue` for which `start_fusion` has been called, the `GraphBuilder` maintains one `KernelFusionCommand` in a map, with the map key being the unique numeric ID introduced for each `queue_impl`.

The execution of the `KernelFusionCommand` (`enqueueImp`) is similar to the `EmptyCommand`, as it simply waits for all its dependencies and then completes.

### Behavior on `queue::ext_codeplay_start_fusion`

On a call to `ext_codeplay_start_fusion`, a new `KernelFusionCommand` with status `ACTIVE` is created and inserted into the map maintained by the `GraphBuilder`, effectively putting the queue into fusion mode.
If a previous `KernelFusionCommand` for this queue is still present, its status is now set to `DELETE` and it is processed for clean-up (more details on why this is necessary can be found in the section on event handling).

### Behavior on `queue::submit`

If the queue is not in fusion mode, the behavior of this call remains unchanged. In case the submitted command is not a device kernel, the synchronization rules detailed below and in the kernel fusion extension proposal apply.

In all other cases, the kernel is added to the graph as usual, setting up the necessary dependency edges for all requirements, adding the kernel to the leaves of the memory records, and connecting event dependencies specified for the kernel. However, in contrast to the regular process, the kernel command and potential auxiliary commands (e.g., connection commands for event dependencies across different contexts) are **not** passed to the `GraphProcessor` for enqueueing right away, but rather stored in the fusion list of the `KernelFusionCommand`. Also, an event dependency between the `KernelFusionCommand` and the newly added kernel command is added to the graph.

### Behavior on `queue::ext_codeplay_cancel_fusion`

If the queue is not in fusion mode, a warning is printed if `SYCL_RT_WARNING_LEVEL` is set to 1 otherwise nothing needs to be done.

Otherwise, as the individual kernels which need to be executed after `ext_codeplay_cancel_fusion` are already correctly added to the graph with all their requirements and event dependencies, the process for `ext_codeplay_cancel_fusion` is comparably simple: The fusion command's status is changed to `CANCELLED` and the fusion command, all the kernels in the fusion list and all the auxiliary commands associated with them are enqueued to the `GraphProcessor`.

### Behavior on `queue::ext_codeplay_complete_fusion`

If the queue is not in fusion mode (this might be due to an earlier cancellation on a synchronization event, see section on [synchronization](#synchronization-behavior)), `ext_codeplay_complete_fusion` still needs to return a valid event. More details on how this case is handled can be found in the section on event handling.

If the queue is still in fusion mode, the `GraphBuilder` will call into the `jit_compiler` to try and fuse the fusion list associated with the fusion command (but not the auxiliary commands) into a single command-group.

In case the fusion process in the JIT compiler fails, the fusion will be aborted by calling `ext_codeplay_cancel_fusion`, with the effects described in the corresponding section above.

If the fusion process completes successfully, the event dependencies of the kernels in the fusion list are filtered to remove any event dependencies that are internal to the fused kernel, i.e., dependencies from one kernel in the fusion list to another kernel in the fusion list.

After that, a new `ExecCGCommand` is constructed and placed in the graph instead of the individual kernels in the fusion list. This is implemented by first removing all the individual kernels from the graph, including their requirement and dependency edges, and restoring the leaves for all memory records that the individual kernels had requirements on. Now that the graph state is restored, the new fused kernel command can be added to the graph, using the union of the requirements and event dependencies of the individual kernels, to create all necessary requirements and dependency edges.
Additionally, an event dependency between the `KernelFusionCommand` and the fused kernel `ExecCGCommand` is added to the graph before all auxiliary commands.
The fused kernel and the `KernelFusionCommand` are eventually enqueued to the `GraphProcessor`.
The `KernelFusionCommand` status is set to `COMPLETED`.

### Synchronization Behavior

As described in the [kernel fusion extension proposal](https://github.com/intel/llvm/pull/7098), several scenarios require aborting the fusion early to avoid semantic violations or circular dependencies in the execution graph. Essentially, this affects all commands that do not become part of the fusion process, e.g., kernels on other queues, host tasks, or explicit memory operations, that have a dependency on at least one of the kernels in the current fusion list due to a requirement or event dependency.

The `GraphProcessor` is able to detect such scenarios. Independent of the actual command requiring synchronization, all execution paths end up enqueueing the command itself and/or its dependencies to  `GraphProcessor::enqueueCommand` . This member function is responsible for detecting if any of the commands enqueued is associated with an active `KernelFusionCommand`. If that is the case, the enqueue process is paused and the fusion on the queue associated with the `KernelFusionCommand` is canceled, identical to an explicit call to `queue::ext_codeplay_cancel_fusion` as described above.

The detection at this stage is possible because even if the queue is in fusion mode, the individual kernels in the fusion list are added to the graph (and the leaves of memory records) such that other commands not part of the fusion process can correctly register them in their dependencies.

Special case treatment and earlier detection during graph-construction in the `GraphBuilder` are necessary for the case where commands submitted to one queue have requirements or dependencies on commands submitted to another queue in fusion mode.

If the fusion for a queue is aborted early, a warning can be printed to inform the user. This warning can be enabled by setting the environment variable `SYCL_RT_WARNING_LEVEL` to a value of  `1` or higher.

### Event Handling

The calls to `queue::submit` as well as to `queue::ext_codeplay_complete_fusion` need to return an event to allow synchronization with the execution of submitted kernels or the fused kernel.

Returning the event associated with the individual kernels (`ExecCGCommand`) from `queue::submit` while in fusion mode would mean that these events become useless if the fusion mode is later on completed successfully because the individual kernels are removed from the graph and never executed in this case.

To overcome this problem, a call to `queue::submit` while in fusion mode will instead return the event associated with the `KernelFusionCommand`.
This event will always remain valid and allow for synchronization, independent of whether the fusion mode was terminated through early cancellation, `ext_codeplay_cancel_fusion`, or `ext_codeplay_complete_fusion`, as the `KernelFusionCommand` in all these scenarios has event dependencies on all relevant commands.

Another important scenario is a call to `ext_codeplay_complete_fusion` after the fusion for this queue has been aborted early (see section on [Synchronization Behavior](#synchronization-behavior)).
In this case, the individual kernels and the `KernelFusionCommand` could have been enqueued and even executed long ago, but the call to `ext_codeplay_complete_fusion` still needs to return a valid event for synchronization.
To handle this case, the `KernelFusionCommand` for each queue remains in the fusion map until the next `ext_codeplay_start_fusion` on the same queue and does not undergo the cleanup process until its status has been set to `DELETE` by `ext_codeplay_start_fusion`.
This way, the lifetime of the event associated with the `KernelFusionCommand` is extended such that it is still valid for synchronization when returned from a later call to `ext_codeplay_complete_fusion`.

Note that even though the `KernelFusionCommand` is associated with a particular queue and context, the associated event does not have an underlying PI event, i.e., it is essentially a host event.


## Fusion Process

To fuse kernels, a small LLVM based JIT compiler (later just JIT for brevity) is responsible to perform the following tasks:

 - Load input modules in LLVM IR format (translate SPIR-V module to LLVM IR for now)
 - Fuse kernels in a new module
 - Perform fusion specific optimization
 - Finalize for the target (emit SPIR-V module for now)

The pipeline is triggered by the SYCL runtime by calling the `fuseKernels` function and its result is then injected into the runtime as a kernel bundle.

The fusion specific optimization is composed of new LLVM passes we wrote for this task.

The rest of the code is just plumbing code to build the pipeline. The SPIR-V loading and emission are done by the LLVM-SPIRV-Translator.

### Kernel fusion pipeline

The kernel fusion process is triggered by the `fuseKernels` function from the fusion JIT module.
It takes as parameters the information required for the fusion: a context, a list of information about kernels to be fused, an ordered list of kernels to fuse (the same kernel may appear more than once), the name of the fused kernel, a list of identical parameters, and a list of buffers to promote to private or local memory and constant from the runtime to inject into the module.

The function creates a new LLVM module with a stub function that will become the fused kernel and adds the kernels to fuse and their dependencies into this module.
In the case the original module is SPIR-V, the module is first translated to LLVM IR and then added to the module.
Information about the fusion is registered within the module by attaching metadata to the stub function (see section [Passing information to the fusion pipeline](#passing-information-to-the-fusion-pipeline)) and runs the fusion and optimization pipeline.

The pipeline currently consists of the following passes (in order):

  - `SYCLKernelFusion` performs the actual fusion process by inlining kernels to fuse inside the fused kernel
  - Generic optimization passes: `IndVarSimplifyPass`, `LoopUnrollPass`, `SROAPass`, `InferAddressSpacesPass` to remove pointers to the generic address-space
    - These optimizations are important to help the internalizer, see note below.
  - `SYCLInternalizer` promotes buffer to local or private memory
  - `SYCLCP` propagates runtime information as constant in the kernel
  - Generic optimization passes post-fusion: `SROAPass`, `SCCPPass`, `InstCombinePass`, `SimplifyCFGPass`, `SROAPass`, `InstCombinePass`, `SimplifyCFGPass`, `ADCEPass`, `EarlyCSEPass`

Note: ideally the `InferAddressSpacesPass` task should be done by the static compiler.
However, to help the inference and (in fine) the internalizer passes in general, we need to run optimizations more aggressively than the static currently does.
The main barrier that could prevent the internalization of buffers is the presence of generic pointers.
As we need to change a pointer to generic to a pointer to private / local, if it is casted to generic we need to ensure that any casts back
to global can be safely changed to the proper address space.
A more precise inference pass and memory analysis will be required to lift this workaround.

Once the pipeline is finished, the module is translated into a SPIR-V module and encapsulated into a `SYCLKernelInfo` object containing metadata required for its injection into the runtime.

### Passing information to the fusion pipeline

Most of the information passed from the runtime into the JIT is stored as metadata inside the LLVM module the JIT creates.
This eases lit testing as all the information is self-contained in the module.

The metadata is attached to a function that will become the fused kernel:

- `sycl.kernel.fused`: declare the kernels to fuse. Contains a list of kernel names to fuse.
- `sycl.kernel.param`: declare identical parameters. Contains a list of tuples, each tuple represents identical arguments and each element of that tuple contains a pair of indexes referencing the kernel index in `sycl.kernel.fused` and the parameter index of that kernel (0 indexed). For instance ((0,1),(2,3)) means the second argument of the first kernel is identical to the fourth argument of the third kernel.
- `sycl.kernel.promote`: declare identical parameters to be promoted. Contains a list of index (of the fused kernel, after identical arguments elision) and `private` if the argument is to be promoted to private memory or `local` if it is to local.
- `sycl.kernel.promote.size`: declare the address space size for the promoted memory. Contains a list of indexes (of the fused kernel, after identical arguments elision) and the number of elements.
- `sycl.kernel.constants`: declare the value of a scalar or aggregate to be used as constant values. Contains a list of indexes (of the fused kernel, after identical arguments elision) and the value as a string. Note: the string is used to store the value, the string is read as a buffer of char and reinterpreted into the value of the argument's type.


### Support for non SPIR-V targets

Non SPIR-V targets (NVPTX / AMDGCN) are not supported at the moment as they cannot ingest a SPIR-V module. However, we are looking into adding support for these targets once the initial SPIR-V based path is operational.

In this scenario, two options are possible to add JIT support:

 - During static compilation we store the LLVM module on top of the finalized binary. This behavior could be controlled by a flag to avoid a too important binary inflation. Then, during the fusion process, the JIT will load that LLVM IR and finalize the fused kernel to the final target as driven by the PI plugin.
 - SPIR-V ingestion support is added for these targets. The module to be loaded could then be the generic SPIR-V module. This path would however exclude target specific optimizations written in user's code. The current state of the SPIR-V translator does not allow this at the moment and significant work is needed to add this support.

In these cases, PI will need to be extended to allow to somehow drive the JIT process, so it is tailored to the plugin target needs.
