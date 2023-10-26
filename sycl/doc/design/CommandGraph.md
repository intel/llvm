# Command-Graph Extension

This document describes the implementation design of the
[SYCL Graph Extension](../extensions/proposed/sycl_ext_oneapi_graph.asciidoc).

A related presentation can be found
[here](https://www.youtube.com/watch?v=aOTAmyr04rM).

## Requirements

An efficient implementation of a lazy command-graph execution and its replay
requires extensions to the Unified Runtime (UR) layer. Such an extension is
the command-buffer experimental feature, where a command-buffer object
represents a series of operations to be enqueued to the backend device and
their dependencies. A single command-graph can be partitioned into more
than one command-buffer by the runtime. The SYCL Graph extension
distinguishes between backends that support the command-buffer extension
and those that do not. Currently command-buffer extensions are only supported
by Level Zero. All other backends would fall back to an emulation mode, or not
be reported as supported.

The emulation mode targets support of functionality only, without potentially
resulting performance improvements, i.e. execution of a closed Level Zero
command-list multiple times.

### UR Command-Buffer Experimental Feature

The command-buffer concept has been introduced to UR as an
[experimental feature](https://oneapi-src.github.io/unified-runtime/core/api.html#command-buffer-experimental)
with the following entry-points:

| Function                                     | Description |
| -------------------------------------------- | ----------- |
| `urCommandBufferCreateExp`                   | Create a command-buffer. |
| `urCommandBufferRetainExp`                   | Incrementing reference count of command-buffer. |
| `urCommandBufferReleaseExp`                  | Decrementing reference count of command-buffer. |
| `urCommandBufferFinalizeExp`                 | No more commands can be appended, makes command-buffer ready to enqueue on a command-queue. |
| `urCommandBufferAppendKernelLaunchExp`       | Append a kernel execution command to command-buffer. |
| `urCommandBufferAppendMemcpyUSMExp`          | Append a USM memcpy command to the command-buffer. |
| `urCommandBufferAppendMembufferCopyExp`      | Append a mem buffer copy command to the command-buffer. |
| `urCommandBufferAppendMembufferWriteExp`     | Append a memory write command to a command-buffer object. |
| `urCommandBufferAppendMembufferReadExp`      | Append a memory read command to a command-buffer object. |
| `urCommandBufferAppendMembufferCopyRectExp`  | Append a rectangular memory copy command to a command-buffer object. |
| `urCommandBufferAppendMembufferWriteRectExp` | Append a rectangular memory write command to a command-buffer object. |
| `urCommandBufferAppendMembufferReadRectExp`  | Append a rectangular memory read command to a command-buffer object. |
| `urCommandBufferEnqueueExp`                  | Submit command-buffer to a command-queue for execution. |

See the [UR EXP-COMMAND-BUFFER](https://oneapi-src.github.io/unified-runtime/core/EXP-COMMAND-BUFFER.html)
specification for more details.

## Design

![Basic architecture diagram.](images/SYCL-Graph-Architecture.svg)

There are two sets of user facing interfaces that can be used to create a
command-graph object: Explicit and Record & Replay API. Within the runtime they
share a common infrastructure.

## Nodes & Edges

A node in a graph is a SYCL [command-group](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#command-group)
(CG) that is defined by a [command-group function](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#command-group-function-object)
(CGF).

Internally, a node is represented by the `detail::node_impl` class, and a command-group
by the `sycl::detail::CG` class. An instance of `detail::node_impl` stores a
`sycl::detail::CG` object for the command-group that the node represents.

A [command-group handler](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#handler)
lets the user define the operations that are to be performed in the command-group,
e.g. kernel execution, memory copy, host-task. In DPC++ an internal "finalization"
operation is done inside the `sycl::handler` implementation, which constructs a
CG object of a specific type. During normal operation, `handler::finalize()`
then passes the CG object to the scheduler, and a `sycl::event` object
representing the command-group is returned.

However during graph construction, inside `hander::finalize()` the CG object is
not submitted for execution as normal, but stored in the graph as a new node
instead.

When a user adds a node to a graph using the explicit
`command_graph<modifiable>::add()` API passing a CGF, in our graph runtime
implementation a `sycl::handler` object is constructed with a graph parameter
telling it to not submit the CG object to the scheduler on finalization.
This handler finalizes the CGF, and after finalization the CG object from the
handler is moved to the node.

For creating a node in the graph using queue recording mode. When the
`sycl::handler` from a queue submission is finalized, if the queue the
handler was created from is in the recording mode, then the handler knows
not to submit the CG object to the scheduler. Instead, the CG object is
added to the graph associated with the queue as a new node.

Edges are stored in each node as lists of predecessor and successor nodes.

## Execution Order

The current way graph nodes are linearized into execution order is using a
reversed depth-first sorting algorithm. Alternative algorithms, such as
breadth-first, are possible and may give better performance on certain
workloads/hardware. In the future there might be options for allowing the
user to control this implementation detail.

## Scheduler Integration

When there are no requirements from accessors in a command-graph submission,
the scheduler is bypassed and the underlying UR command-buffer is directly
enqueued to a UR queue. If there are accessor requirements, the UR
command-buffer for the executable graph needs to be enqueued by the scheduler.

When individual graph nodes have requirements from SYCL accessors, the
underlying `sycl::detail::CG` object stored in the node is copied and passed to
the scheduler for adding to the UR command-buffer, otherwise the node can
be appended directly as a command in the UR command-buffer. This is in-keeping
with the existing behaviour of the handler with normal queue submissions.

## Memory handling: Buffer and Accessor

There is no extra support for graph-specific USM allocations in the current
proposal. Memory operations will be supported subsequently by the current
implementation starting with `memcpy`.

Buffers and accessors are supported in a command-graph. There are
[spec restrictions](../extensions/proposed/sycl_ext_oneapi_graph.asciidoc#storage-lifetimes)
on buffer usage in a graph so that their lifetime semantics are compatible with
a lazy work execution model. However these changes to storage lifetimes have not
yet been implemented.

## Backend Implementation

Implementation of [UR command-buffers](#UR-command-buffer-experimental-feature)
for each of the supported SYCL 2020 backends.

This is currently only Level Zero but more sub-sections will be added here as
other backends are implemented.

### Level Zero

The UR `urCommandBufferEnqueueExp` interface for submitting a command-buffer
takes a list of events to wait on, and returns an event representing the
completion of that specific submission of the command-buffer.

However, in the equivalent Level Zero function
[zeCommandQueueExecuteCommandLists](https://spec.oneapi.io/level-zero/latest/core/api.html#zecommandqueueexecutecommandlists)
there are no parameters to take a wait-list, and the only sync primitive
returned is blocking on host.

In order to achieve the expected UR command-buffer enqueue semantics with Level
Zero, the adapter implementation adds extra commands to the Level Zero
command-list representing a UR command-buffer.

* Prefix - Commands added to the start of the L0 command-list by L0 adapter.
* Suffix - Commands added to the end of the L0 command-list by L0 adapter.

These extra commands operate on L0 event synchronisation primitives, used by the
command-list to interact with the external UR wait-list and UR return event
required for the enqueue interface.

The `ur_exp_command_buffer_handle_t` class for this adapter contains a
*SignalEvent* which signals the completion of the command-list in the suffix,
and is reset in the prefix. This signal is detected by a new UR return event
created on UR command-buffer enqueue.

There is also a *WaitEvent* used by the `ur_exp_command_buffer_handle_t` class
in the prefix to wait on any dependencies passed in the enqueue wait-list.
This WaitEvent is reset at the end of the suffix, along with reset commands
to reset the L0 events used to implement the UR sync-points back to the
non-signaled state.

![L0 command-buffer diagram](images/L0_UR_command-buffer.svg)

For a call to `urCommandBufferEnqueueExp` with an `event_list` *EL*,
command-buffer *CB*, and return event *RE* our implementation has to submit two
new command-lists for the above approach to work. One before
the command-list with extra commands associated with *CB*, and the other
after *CB*. These two new command-lists are retrieved from the UR queue, which
will likely reuse existing command-lists and only create a new one in the worst
case.

The L0 command-list created on `urCommandBufferEnqueueExp` to execute **before**
*CB* contains a single command. This command is a barrier on *EL* that signals
*CB*'s *WaitEvent* when completed.

The L0 command-list created on `urCommandBufferEnqueueExp` to execute **after**
*CB* also contains a single command. This command is a barrier on *CB*'s
*SignalEvent* that signals *RE* when completed.

#### Drawbacks

There are two drawbacks of this approach to implementing UR command-buffers for
Level Zero:

1. 3x the command-list resources are used, if there are many UR command-buffers in
   flight, this may exhaust L0 driver resources. A trivial graph requires 3 L0
   command-lists and if we implement partitioning a graph into multiple UR
   command-buffers, then each partition will contain 3 L0 command-lists.

2. Each L0 command-list is submitted individually with a
   `ur_queue_handle_t_::executeCommandList` call which introduces serialization
   in the submission pipeline that is heavier than having a barrier or a
   `waitForEvents` on the same command-list. Resulting in additional latency when
   executing a UR command-buffer.

Future work will include exploring L0 API extensions to improve the mapping of
UR command-buffer to L0 command-list.
