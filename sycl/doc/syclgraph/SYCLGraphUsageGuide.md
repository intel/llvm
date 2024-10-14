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
dynamic_parameter dynParamInput(myGraph, ptrX);
dynamic_parameter dynParamScalar(myGraph, myScalar);

// The node uses ptrX as an input & output parameter, with operand
// mySclar as another argument.
node kernelNode = myGraph.add([&](handler& cgh) {
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
dynamic_parameter dynParamAccessor(myGraph, bufferA.get_access());

node kernelNode = myGraph.add([&](handler& cgh) {
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
