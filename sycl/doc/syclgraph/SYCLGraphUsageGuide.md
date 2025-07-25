# SYCL Graph Usage Guide and Examples

This document describes recommended usage guidelines for using the
`sycl_ext_oneapi_graph` extension (referred to as SYCL Graph in this document)
as well as provides example code snippets for various features and usage
scenarios.

The specification for the `sycl_ext_oneapi_graph` extension can be found
[here](../extensions/experimental/sycl_ext_oneapi_graph.asciidoc).

## General Usage Guidelines

The following section provides some general usage guidelines when working
with SYCL Graph or adapting an existing SYCL application to use SYCL Graph.
Examples here will generally only use one graph creation API ("Explicit" or
"Record & Replay") for simplicity, but can generally be applied to either API
unless specifically noted.

### Use Host-Tasks For Host Work

SYCL Graph cannot capture work done on the host in application code which does
not go through the SYCL runtime, including direct operations (for example kernel
submissions) through a backend API (level-zero, cuda, etc). Any work to be done
on the host should be captured within a SYCL host-task as a node in the
graph. Workloads which require a lot of host-tasks may see reduced performance
due to host synchronization or preventing the runtime from submitted the entire
graph to the device at once.

Direct operations with backend APIs are supported in SYCL Graph through
host-tasks using an interop handle, but it is currently not possible to obtain
the native backend objects associated with SYCL Graph objects.

Some applications may also execute host code inside of the Command Group
Function (CGF). This is not explicitly disallowed in normal SYCL code but is
generally discouraged. Consider the following example:

```c++
Queue.submit([&](sycl::handler& CGH){
    // Do some host work here to prepare for the kernel to be executed
    do_some_host_work();
    CGH.parallel_for(...);
});
```

In normal SYCL usage `do_some_host_work()` will be evaluated during the call to
submit. With SYCL Graph this code will be evaluated once during the call to
`submit()` or `command_graph::add()` but will not be evaluated on subsequent
runs of the graph which may lead to incorrect behavior. This code should be
placed into a host-task node which the kernel (or other operation) depends on,
like so:

```c++
Graph.begin_recording(Queue);

sycl::event HostWorkEvent = Queue.submit([&](sycl::handler& CGH){
    CGH.host_task([=](){
        do_some_host_work();
    });
});

Queue.submit([&](sycl::handler& CGH){
    CGH.depends_on(HostWorkEvent);
    CGH.parallel_for(...);
});

Graph.end_recording(Queue);
```

### Graph Execution Concurrency

Normally the SYCL runtime will prevent multiple submissions of a given
`command_graph` from executing concurrently to prevent data races when accessing
the same memory resources. However, it is quite common to wish to execute the
same graph on different sets of data concurrently. This can be accomplished by
using graph update functionality, but this must also be serialized with a host
synchronization. Consider this example (it uses whole graph update for
simplicity):

```c++
namespace sycl_ext = sycl::ext::oneapi::experimental;

sycl_ext::command_graph<graph_state::executable> ExecutableGraph
   = ModifiableGraph.finalize();

Queue.ext_oneapi_graph(ExecutableGraph);
// Updating the graph here to use new memory, this forces a host synchronization
ExecutableGraph.update(ModifiableGraphWithNewParams);

// Re-execute the update graph
Queue.ext_oneapi_graph(ExecutableGraph);

```

If your sets of inputs are known (such as a double-buffering type scenario), you
can create multiple executable graphs up front, avoiding the need for expensive
host sync between executions (assuming there are no implicit dependencies from
accessing the same `sycl::buffer`) for updating, and potentially increasing
device occupancy (if the device supports it). Modifying the example from above:

```c++
namespace sycl_ext = sycl::ext::oneapi::experimental;

sycl_ext::command_graph<graph_state::executable> ExecutableGraph
   = ModifiableGraph.finalize();
sycl_ext::command_graph<graph_state::executable> ExecutableGraphOtherParams
   = ModifiableGraphWithNewParams.finalize();

Queue.ext_oneapi_graph(ExecutableGraph);
// We can now execute the second graph with no implicit dependency
Queue.ext_oneapi_graph(ExecutableGraphOtherParams);

// Repeatedly executing the graphs will create dependencies on their individual
// preceeding executions, but not on each other.
Queue.ext_oneapi_graph(ExecutableGraph);
Queue.ext_oneapi_graph(ExecutableGraphOtherParams);

```

If the amount of different inputs is not always known up front a similar
strategy could be employed but instead using a pool of graphs to hide the
potential host-synchronization caused when updating and increase device
occupancy.

### Recording Library Calls

#### A Note On Library Compatibility

Since the extension is still experimental and currently under active
development, compatibility with capturing calls to SYCL-based libraries is not
guaranteed. Typically these libraries were developed without knowledge of SYCL
Graph and thus may behave in ways that are incompatible with graph capture (such
as the ones detailed in this guide).

#### Warmups

Some libraries may perform one-time warmups on first execution of some
operations (for example to trigger kernel creation) that are not desirable to
capture in a graph. Consider the following simplified example:

```c++
// Hypothetical library function
void SomeLibrary::Operation(sycl::Queue Queue){
    static bool IsFirstTime = true;

    if(IsFirstTime){
        // Warmup by launching kernel once
        do_warmup(Queue);
        // Execute the actual operation
        execute_operation(Queue);
        IsFirstTime = false;
    } else{
        execute_operation(Queue);
    }
}

// SYCL Application Code

Graph.begin_recording(Queue);

// do_warmup() will be captured here and executed every time the graph is
// executed in future which is undesirable.
SomeLibrary::Operation(Queue);

Graph.end_recording(Queue);

```

In this case it may be necessary to first manually trigger the warmup by calling
`SomeLibrary::Operation()` before starting to record the queue with
`Graph.begin_recording(Queue)` to prevent the warmup from being captured in a
graph when recording.

#### ext_codeplay_enqueue_native_command

The SYCL-Graph extension is compatible with the
[ext_codeplay_enqueue_native_command](../extensions/experimental/sycl_ext_codeplay_enqueue_native_command.asciidoc)
extension that can be used to capture asynchronous library commands as graph
nodes. However, existing `ext_codeplay_enqueue_native_command` user code will
need modifications to work correctly for submission to a sycl queue that can be
in either the executable or recording state.

Using the CUDA backend as an example, existing code which uses a
native-command to invoke a library call:

```c++
q.submit([&](sycl::handler &CGH) {
    CGH.ext_codeplay_enqueue_native_command([=](sycl::interop_handle IH) {
       auto NativeStream = IH.get_native_queue<cuda>();
       myNativeLibraryCall(NativeStream);
    });
});
```

Can be ported as below to work with SYCL-Graph, where the queue may be in
a recording state. If the code is not ported but the queue is in a recording
state, then asynchronous work in `myNativeLibraryCall` will be scheduled
immediately as part of graph finalize, rather than being added to the graph as
a node, which is unlikely to be the desired user behavior.

```c++
q.submit([&](sycl::handler &CGH) {
    CGH.ext_codeplay_enqueue_native_command([=](sycl::interop_handle IH) {
        auto NativeStream = h.get_native_queue<cuda>();
        if (IH.ext_codeplay_has_graph())  {
            auto NativeGraph =
              IH.ext_codeplay_get_native_graph<sycl::backend::ext_oneapi_cuda>();

            // Start capture stream calls into graph
            cuStreamBeginCaptureToGraph(NativeStream, NativeGraph, nullptr,
                                        nullptr, 0,
                                        CU_STREAM_CAPTURE_MODE_GLOBAL);

            myNativeLibraryCall(NativeStream);

            // Stop capturing stream calls into graph
            cuStreamEndCapture(NativeStream, &NativeGraph);
        } else {
            myNativeLibraryCall(NativeStream);
        }
    });
});
```

### Guidance For Library Authors

In addition to the general SYCL-graph compatibility guidelines there are some
considerations that are more relevant to library authors to be compatible with
SYCL-Graph and allow seamless capturing of library calls in a graph.

#### Graph-owned Memory Allocations For Temporary Memory

A common pattern in libraries with specialized SYCL kernels can involve the
allocation and use of temporary memory for those kernels. One approach is custom
allocators which rely on SYCL events to control the lifetime and re-use of this
temporary memory, but these are not compatible with events returned from queue
submissions which are recorded to a graph. Instead the
[sycl_ext_oneapi_async_memory_alloc](../extensions/proposed/sycl_ext_oneapi_async_memory_alloc.asciidoc)
extension can be used which provides similar functionality for eager SYCL usage
as well as compatibility with graphs.

When captured in a graph calls to these extension functions create graph-owned
memory allocations which are tied to the lifetime of the graph. These
allocations can be created as needed for library kernels and the SYCL runtime
may be able to re-use memory where appropriate to minimize the memory footprint
of the graph. This can avoid the need for a library to manage the lifetime of
these allocations themselves, or be aware of the library calls being recorded to
a graph.

It is important to ensure correct dependencies between allocation commands,
kernels that use those allocations, and the calls to free the memory. This
allows the graph to determine when allocations are in-use at a given point in
the graph, and allow for re-using memory for subsequent allocation nodes if
those nodes are ordered after a free command which is no longer in use.

It is important to note that calling `async_free` will not deallocate memory
but simply mark it as free for re-use.

The following shows a simple example of how these allocations can be used in a
library function which is recorded to a graph:

```c++
using namespace sycl;

// Library code, this example is assuming an out of order SYCL queue
void launchLibraryKernel(queue& SyclQueue){
    size_t TempMemSize = 1024;
    void* Ptr = nullptr;

    // Get a pointer to some temporary memory for use in the kernel
    // This call creates an allocation node in the graph if this call is being
    // recorded.
    event AllocEvent = SyclQueue.submit([&](handler& CGH){
        Ptr = sycl_ext::async_malloc(CGH, usm::alloc::device, TempMemSize);
    });

    // Submit the actual library kernel
    event KernelEvent = SyclQueue.submit([&](handler& CGH){
        // Mark the allocation as a dependency so that the temporary memory
        // is available
        CGH.depends_on(AllocEvent);
        // Submit a kernel that uses the temp memory in Ptr
        CGH.parallel_for(...);
    });

    // Free the memory back to the pool or graph, indicating that it is free to
    // be re-used. Memory will not actually be released back to the OS.
    SyclQueue.submit([&](handler& CGH){
        // Mark the kernel as a dependency before freeing
        CGH.depends_on(KernelEvent);
        sycl_ext::async_free(CGH, Ptr);
    });
}

// Application code
void recordLibraryCall(queue& SyclQueue, sycl_ext::command_graph& Graph){
    Graph.begin_recording(SyclQueue);
    // Call into library to record queue commands to the graph
    launchLibraryKernel(SyclQueue);

    Graph.end_recording(SyclQueue);
}
```

Please see "graph-owned memory allocations" section of the
[sycl_ext_oneapi_graph
specification](../extensions/experimental/sycl_ext_oneapi_graph.asciidoc) for a
complete description of the feature.

## Code Examples

The examples below demonstrate intended usage of the extension, but may not be
compatible with the proof-of-concept implementation, as the proof-of-concept
implementation is currently under development.

These examples for demonstrative purposes only, and may leave out details such
as how input data is set.

### Dot Product

This example uses the explicit graph creation API to perform a dot product
operation.

```c++
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>

int main() {
    namespace sycl_ext = sycl::ext::oneapi::experimental;

    const size_t n = 10;
    float alpha = 1.0f;
    float beta = 2.0f;
    float gamma = 3.0f;

    sycl::queue q;
    sycl_ext::command_graph g(q.get_context(), q.get_device());

    float *dotp = sycl::malloc_shared<float>(1, q);
    float *x = sycl::malloc_device<float>(n, q);
    float *y = sycl::malloc_device<float>(n, q);
    float *z = sycl::malloc_device<float>(n, q);

    // Add commands to the graph to create the following topology.
    //
    //     i
    //    / \
    //   a   b
    //    \ /
    //     c

    // init data on the device
    auto node_i = g.add([&](sycl::handler& h) {
        h.parallel_for(n, [=](sycl::id<1> it){
            const size_t i = it[0];
            x[i] = 1.0f;
            y[i] = 2.0f;
            z[i] = 3.0f;
        });
    });

    auto node_a = g.add([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> it) {
            const size_t i = it[0];
            x[i] = alpha * x[i] + beta * y[i];
        });
    }, { sycl_ext::property::node::depends_on(node_i)});

    auto node_b = g.add([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> it) {
            const size_t i = it[0];
            z[i] = gamma * z[i] + beta * y[i];
        });
    }, { sycl_ext::property::node::depends_on(node_i)});

    auto node_c = g.add(
        [&](sycl::handler& h) {
            h.single_task([=]() {
                for(size_t i = 0; i < n; i++){
                  *dotp += x[i] * z[i];
                }
            });
        },
        { sycl_ext::property::node::depends_on(node_a, node_b)});

    auto exec = g.finalize();

    // use queue shortcut for graph submission
    q.ext_oneapi_graph(exec).wait();

    // memory can be freed inside or outside the graph
    sycl::free(x, q);
    sycl::free(y, q);
    sycl::free(z, q);
    sycl::free(dotp, q);

    return 0;
}
```

### Diamond Dependency

The following snippet of code shows how a SYCL `queue` can be put into a
recording state, which allows a `command_graph` object to be populated by the
command-groups submitted to the queue. Once the graph is complete, recording
finishes on the queue to put it back into the default executing state. The
graph is then finalized so that no more nodes can be added. Lastly, the graph is
submitted in its entirety for execution via
`handler::ext_oneapi_graph(command_graph<graph_state::executable>)`.

```c++
using namespace sycl;
namespace sycl_ext = sycl::ext::oneapi::experimental;

queue q{default_selector{}};

// Lifetime of buffers must exceed the lifetime of graphs they are used in.
buffer<T> bufferA{dataA.data(), range<1>{elements}};
bufferA.set_write_back(false);
buffer<T> bufferB{dataB.data(), range<1>{elements}};
bufferB.set_write_back(false);
buffer<T> bufferC{dataC.data(), range<1>{elements}};
bufferC.set_write_back(false);

{
    // New object representing graph of command-groups
    sycl_ext::command_graph graph(q.get_context(), q.get_device(),
          {sycl_ext::property::graph::assume_buffer_outlives_graph{}});


    // `q` will be put in the recording state where commands are recorded to
    // `graph` rather than submitted for execution immediately.
    graph.begin_recording(q);

    // Record commands to `graph` with the following topology.
    //
    //      increment_kernel
    //       /         \
    //   A->/        A->\
    //     /             \
    //   add_kernel  subtract_kernel
    //     \             /
    //   B->\        C->/
    //       \         /
    //     decrement_kernel

    q.submit([&](handler& cgh) {
        auto pData = bufferA.get_access<access::mode::read_write>(cgh);
        cgh.parallel_for<increment_kernel>(range<1>(elements),
                                          [=](item<1> id) { pData[id]++; });
    });

    q.submit([&](handler& cgh) {
        auto pData1 = bufferA.get_access<access::mode::read>(cgh);
        auto pData2 = bufferB.get_access<access::mode::read_write>(cgh);
        cgh.parallel_for<add_kernel>(range<1>(elements),
                                    [=](item<1> id) { pData2[id] += pData1[id]; });
    });

    q.submit([&](handler& cgh) {
        auto pData1 = bufferA.get_access<access::mode::read>(cgh);
        auto pData2 = bufferC.get_access<access::mode::read_write>(cgh);
        cgh.parallel_for<subtract_kernel>(
            range<1>(elements), [=](item<1> id) { pData2[id] -= pData1[id]; });
    });

    q.submit([&](handler& cgh) {
        auto pData1 = bufferB.get_access<access::mode::read_write>(cgh);
        auto pData2 = bufferC.get_access<access::mode::read_write>(cgh);
        cgh.parallel_for<decrement_kernel>(range<1>(elements), [=](item<1> id) {
            pData1[id]--;
            pData2[id]--;
        });
    });

    // queue `q` will be returned to the executing state where commands are
    // submitted immediately for extension.
    graph.end_recording();

    // Finalize the modifiable graph to create an executable graph that can be
    // submitted for execution.
    auto exec_graph = graph.finalize();

    // Execute graph
    q.submit([&](handler& cgh) {
        cgh.ext_oneapi_graph(exec_graph);
    }).wait();
}

// Check output using host accessors
host_accessor hostAccA(bufferA);
host_accessor hostAccB(bufferB);
host_accessor hostAccC(bufferC);

...
```

### Dynamic Parameter Update

Example showing a graph with a single kernel node that is created using a kernel
bundle with `handler::set_args()` and having its node arguments updated.

```c++
...

using namespace sycl;
namespace sycl_ext = sycl::ext::oneapi::experimental;

queue myQueue;
auto myContext = myQueue.get_context();
auto myDevice = myQueue.get_device();

// USM allocations for kernel input/output
const size_t n = 1024;
int *ptrX = malloc_shared<int>(n, myQueue);
int *ptrY = malloc_device<int>(n, myQueue);

int *ptrZ = malloc_shared<int>(n, myQueue);
int *ptrQ = malloc_device<int>(n, myQueue);

// Kernel loaded from kernel bundle
const std::vector<kernel_id> builtinKernelIds =
      myDevice.get_info<info::device::built_in_kernel_ids>();
kernel_bundle<bundle_state::executable> myBundle =
      get_kernel_bundle(myContext, { myDevice }, builtinKernelIds);
kernel builtinKernel = myBundle.get_kernel(builtinKernelIds[0]);

// Graph containing a kernel node
sycl_ext::command_graph myGraph(myContext, myDevice);

int myScalar = 42;
// Create graph dynamic parameters
sycl_ext::dynamic_parameter dynParamInput(myGraph, ptrX);
sycl_ext::dynamic_parameter dynParamScalar(myGraph, myScalar);

// The node uses ptrX as an input & output parameter, with operand
// mySclar as another argument.
sycl_ext::node kernelNode = myGraph.add([&](handler& cgh) {
    cgh.set_args(dynParamInput, ptrY, dynParamScalar);
    cgh.parallel_for(range {n}, builtinKernel);
});

// Create an executable graph with the updatable property.
auto execGraph = myGraph.finalize({sycl_ext::property::graph::updatable});

// Execute graph, then update without needing to wait for it to complete
myQueue.ext_oneapi_graph(execGraph);

// Change ptrX argument to ptrZ
dynParamInput.update(ptrZ);

// Change myScalar argument to newScalar
int newScalar = 12;
dynParamScalar.update(newScalar);

// Update kernelNode in the executable graph with the new parameters
execGraph.update(kernelNode);
// Execute graph again
myQueue.ext_oneapi_graph(execGraph);
myQueue.wait();

sycl::free(ptrX, myQueue);
sycl::free(ptrY, myQueue);
sycl::free(ptrZ, myQueue);
sycl::free(ptrQ, myQueue);

```

Example snippet showing how to use accessors with `dynamic_parameter` update:

```c++
sycl::buffer bufferA{...};
sycl::buffer bufferB{...};

// Create graph dynamic parameter using a placeholder accessor, since the
// sycl::handler is not available here outside of the command-group scope.
sycl_ext::dynamic_parameter dynParamAccessor(myGraph, bufferA.get_access());

sycl_ext::node kernelNode = myGraph.add([&](handler& cgh) {
    // Require the accessor contained in the dynamic paramter
    cgh.require(dynParamAccessor);
    // Set the arg on the kernel using the dynamic parameter directly
    cgh.set_args(dynParamAccessor);
    cgh.parallel_for(range {n}, builtinKernel);
});

...
// Update the dynamic parameter with a placeholder accessor from bufferB instead
dynParamAccessor.update(bufferB.get_access());
```

### Dynamic Command Groups

Example showing how a graph with a dynamic command group node can be updated.

```cpp
...
using namespace sycl;
namespace sycl_ext = sycl::ext::oneapi::experimental;

queue Queue{};
sycl_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

int *PtrA = malloc_device<int>(1024, Queue);
int *PtrB = malloc_device<int>(1024, Queue);

auto CgfA = [&](handler &cgh) {
  cgh.parallel_for(1024, [=](item<1> Item) {
    PtrA[Item.get_id()] = 1;
  });
};

auto CgfB = [&](handler &cgh) {
  cgh.parallel_for(512, [=](item<1> Item) {
    PtrB[Item.get_id()] = 2;
  });
};

// Construct a dynamic command-group with CgfA as the active cgf (index 0).
auto DynamicCG = sycl_ext::dynamic_command_group(Graph, {CgfA, CgfB});

// Create a dynamic command-group graph node.
auto DynamicCGNode = Graph.add(DynamicCG);

auto ExecGraph = Graph.finalize(sycl_ext::property::graph::updatable{});

// The graph will execute CgfA.
Queue.ext_oneapi_graph(ExecGraph).wait();

// Sets CgfB as active in the dynamic command-group (index 1).
DynamicCG.set_active_index(1);

// Calls update to update the executable graph node with the changes to DynamicCG.
ExecGraph.update(DynamicCGNode);

// The graph will execute CgfB.
Queue.ext_oneapi_graph(ExecGraph).wait();
```

### Dynamic Command Groups With Dynamic Parameters

Example showing how a graph with a dynamic command group that uses dynamic
parameters in a node can be updated.

```cpp
...
using namespace sycl;
namespace sycl_ext = sycl::ext::oneapi::experimental;

size_t N = 1024;
queue Queue{};
auto MyContext = Queue.get_context();
auto MyDevice = Queue.get_device();
sycl_ext::command_graph Graph{MyContext, MyDevice};

int *PtrA = malloc_device<int>(N, Queue);
int *PtrB = malloc_device<int>(N, Queue);

// Kernels loaded from kernel bundle
const std::vector<kernel_id> BuiltinKernelIds =
      MyDevice.get_info<info::device::built_in_kernel_ids>();
kernel_bundle<bundle_state::executable> MyBundle =
      get_kernel_bundle<sycl::bundle_state::executable>(MyContext, { MyDevice }, BuiltinKernelIds);

kernel BuiltinKernelA = MyBundle.get_kernel(BuiltinKernelIds[0]);
kernel BuiltinKernelB = MyBundle.get_kernel(BuiltinKernelIds[1]);

// Create a dynamic parameter with an initial value of PtrA
sycl_ext::dynamic_parameter DynamicPointerArg{Graph, PtrA};

// Create command groups for both kernels which use DynamicPointerArg
auto CgfA = [&](handler &cgh) {
  cgh.set_arg(0, DynamicPointerArg);
  cgh.parallel_for(range {N}, BuiltinKernelA);
};

auto CgfB = [&](handler &cgh) {
  cgh.set_arg(0, DynamicPointerArg);
  cgh.parallel_for(range {N / 2}, BuiltinKernelB);
};

// Construct a dynamic command-group with CgfA as the active cgf (index 0).
auto DynamicCG = sycl_ext::dynamic_command_group(Graph, {CgfA, CgfB});

// Create a dynamic command-group graph node.
auto DynamicCGNode = Graph.add(DynamicCG);

auto ExecGraph = Graph.finalize(sycl_ext::property::graph::updatable{});

// The graph will execute CgfA with PtrA.
Queue.ext_oneapi_graph(ExecGraph).wait();

//Update DynamicPointerArg with a new value
DynamicPointerArg.update(PtrB);

// Sets CgfB as active in the dynamic command-group (index 1).
DynamicCG.set_active_index(1);

// Calls update to update the executable graph node with the changes to
// DynamicCG and DynamicPointerArg.
ExecGraph.update(DynamicCGNode);

// The graph will execute CgfB with PtrB.
Queue.ext_oneapi_graph(ExecGraph).wait();
```

### Whole Graph Update

Example that shows recording and updating several nodes with different
parameters using whole-graph update.

```c++
...
using namespace sycl;
namespace sycl_ext = sycl::ext::oneapi::experimental;

// Enqueue several kernels which use inputPtr
void run_kernels(int* inputPtr, queue syclQueue){
    event eventA = syclQueue.submit([&](handler& CGH){
        CGH.parallel_for(...);
    });
    event eventB = syclQueue.submit([&](handler& CGH){
        CGH.depends_on(eventA);
        CGH.parallel_for(...);
    });
    syclQueue.submit([&](handler& CGH){
        CGH.depends_on(eventB);
        CGH.parallel_for(...);
    });
}

...

queue myQueue;

// USM allocations
const size_t n = 1024;
int *ptrA = malloc_device<int>(n, myQueue);
int *ptrB = malloc_device<int>(n, myQueue);

// Main graph which will be updated later
sycl_ext::command_graph mainGraph(myQueue);

// Record the kernels to mainGraph, using ptrA
mainGraph.begin_recording(myQueue);
run_kernels(ptrA, myQueue);
mainGraph.end_recording();

auto execMainGraph = mainGraph.finalize({sycl_ext::property::graph::updatable});

// Execute execMainGraph
myQueue.ext_oneapi_graph(execMainGraph);

// Record a second graph which records the same kernels, but using ptrB instead
sycl_ext::command_graph updateGraph(myQueue);
updateGraph.begin_recording(myQueue);
run_kernels(ptrB, myQueue);
updateGraph.end_recording();

// Update execMainGraph using updateGraph. We do not need to finalize
// updateGraph (this would be expensive)
execMainGraph.update(updateGraph);

// Execute execMainGraph again, which will now be operating on ptrB instead of
// ptrA
myQueue.ext_oneapi_graph(execMainGraph);
```

### Graph-Owned Memory Allocations

#### Explicit Graph Example

Using default memory pool.

```c++
using namespace sycl;
namespace sycl_ext = sycl::ext::oneapi::experimental;

void* Ptr = nullptr;
size_t AllocSize = 1024;
// Add an async_malloc node and capturing the returned pointer in Ptr
auto AllocNode = Graph.add([&](handler& CGH){
  Ptr = sycl_ext::async_malloc(CGH, usm::alloc::device, AllocSize);
});

// Use Ptr in another graph node which depends on AllocNode
auto OtherNodeA = Graph.add(..., {property::graph::depends_on{AllocNode}});
// Use Ptr in another node which has an indirect dependency on AllocNode
auto OtherNodeB = Graph.add(..., {property::graph::depends_on{OtherNodeA}});

// Free Ptr, indicating it is no longer in use at this point in the graph,
// with a dependency on any leaf nodes using Ptr
Graph.add([&](handler& CGH){
  sycl_ext::async_free(CGH, Ptr);
}, {property::graph::depends_on{OtherNodeB}});
```

#### Queue Recording Example

Using user-provided memory pool.

```c++
using namespace sycl;
namespace sycl_ext = sycl::ext::oneapi::experimental;

void* Ptr = nullptr;
size_t AllocSize = 1024;
queue Queue {syclContext, syclDevice};

// Device memory pool with zero init property
sycl_ext::memory_pool MemPool{syclContext, syclDevice, usm::alloc::device,
                             {sycl_ext::property::memory_pool::zero_init{}}};
Graph.begin_recording(Queue);
// Add an async_malloc node and capture the returned pointer in Ptr,
// zero_init property and usm::alloc kind of pool will be respected but pool
// is otherwise ignored
event AllocEvent = Queue.submit([&](handler& CGH){
  Ptr = sycl_ext::async_malloc_from_pool(CGH, AllocSize, MemPool);
});

// Use Ptr in another graph node which depends on AllocNode
event OtherEventA = Queue.submit([&](handler& CGH){
  CGH.depends_on(AllocEvent);
  // Do something with Ptr
  CGH.parallel_for(...);
});
// Use Ptr in another node which has an indirect dependency on AllocNode
event OtherEventB = Queue.submit([&](handler& CGH){
  CGH.depends_on(OtherEventA);
  // Do something with Ptr
  CGH.parallel_for(...);
});

// Free Ptr, indicating it is no longer in use at this point in the graph,
// with a dependency on any leaf nodes using Ptr
Queue.submit([&](handler& CGH){
  CGH.depends_on(OtherEventB);
  sycl_ext::async_free(CGH, Ptr);
});

Graph.end_recording(Queue);
```

#### In-Order Queue Recording Example

Using an in-order queue and the event-less async alloc functions.

```c++
using namespace sycl;
namespace sycl_ext = sycl::ext::oneapi::experimental;

void* Ptr = nullptr;
size_t AllocSize = 1024;
queue Queue {syclContext, syclDevice, {property::queue::in_order{}}};

Graph.begin_recording(Queue);
// Add an async_malloc node and capturing the returned pointer in Ptr
Ptr = sycl_ext::async_malloc(Queue, usm::alloc::device, AllocSize);

// Use Ptr in another graph node which has an in-order dependency on the
// allocation node
Queue.submit([&](handler& CGH){
  // Do something with Ptr
  CGH.parallel_for(...);
});
// Use Ptr in another node which has an in-order dependency on the
// previous kernel.
Queue.submit([&](handler& CGH){
  // Do something with Ptr
  CGH.parallel_for(...);
});

// Free Ptr, indicating it is no longer in use at this point in the graph,
// with an in-order dependency on the previous kernel.
sycl_ext::async_free(Queue, Ptr);

Graph.end_recording(Queue);
```
