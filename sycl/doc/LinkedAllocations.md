# Linked allocations

## Brief overview of allocations for memory objects

A SYCL memory object (`buffer`/`image`) can be accessed in multiple contexts
throughout its lifetime. Since this is dynamic information that is unknown
during memory object construction, no allocation takes place at that point.
Instead, memory is allocated in each context whenever the SYCL memory object
is first accessed there:

```
  cl::sycl::buffer<int, 1> buf{cl::sycl::range<1>(1)}; // No allocation here

  cl::sycl::queue q;
  q.submit([&](cl::sycl::handler &cgh){
    // First access to buf in q's context: allocate memory
    auto acc = buf.get_access<cl::sycl::access::mode::read_write>(cgh);
	...
  });

  // First access to buf on host (assuming q is not host): allocate memory
  auto acc = buf.get_access<cl::sycl::access::mode::read_write>();
```

In the DPCPP execution graph these allocations are represented by allocation
command nodes (`cl::sycl::detail::AllocaCommand`). A finished allocation
command means that the associated memory object is ready for its first use in
that context, but for host allocation commands it might be the case that no
actual memory allocation takes place: either because it is possible to reuse the
data pointer provided by the user:

```
  int val;
  cl::sycl::buffer<int, 1> buf{&val, cl::sycl::range<1>(1)};

  // An alloca command is created, but it does not allocate new memory: &val
  // is reused instead.
  auto acc = buf.get_access<cl::sycl::access::mode::read_write>();
```

Or because a mapped host pointer obtained from a native device memory object
is used in its place (as is the case for linked commands, covered below).

## Linked allocation commands

Whenever an allocation command is created for a memory object, it can be created
as "linked" to another one if they satisfy these requirements:
- Both allocation commands are associated with the same memory object.
- Exactly one of the two commands is associated with a host context.
- Neither of the commands is already linked.

The idea behind linked commands is that the device allocation of the pair is
supposed to reuse the host allocation, i.e. the host memory is requested to be
shared between the two (the underlying backend is still free to ignore that
request and allocate additional memory if needed). The difference in handling
linked and unlinked allocations is summarized in the table below.

|   | Unlinked | Linked |
| - | -------- | ------ |
| Native memory object creation | Created without a host pointer, then initialized as a separate memory transfer operation if a host pointer is available and the first access mode does not discard the data. | Created with USE_HOST_PTR if a suitable host pointer is available, regardless of the first access mode. |
| Host allocation command behaviour | Skipped if a suitable user host pointer is available. | In addition to skipping the allocation if a suitable user pointer is provided, the allocation is also skipped if the host command is created after its linked counterpart (it's retrieved via map operation instead). |
| Memory transfer | Performed with read/write operations, device-to-device transfer is done with a host allocation as an intermediary (direct transfer is not supported by PI). | Only one allocation from the pair can be active at a time, the switch is done with map/unmap operations. Device-to-device transfer where one of the device allocations is linked is done with the host allocation from the pair as an intermediary (e.g. for transfer from unlinked device allocation A to linked device allocation B: map B -> read A to the host allocation -> unmap B). |

## Command linking approach

Whenever two allocation commands are considered for linking, the decision is
made based on the following criterion: the commands are linked if and only if
the non-host device of the pair supports host unified memory (i.e. the device
and host share the same physical memory). The motivation for this is two-fold:
- If the non-host device supports host unified memory, the USE_HOST_PTR flag
should not result in any additional device memory allocation or copying between
the two during map/unmap operations.
- Even if the point above makes no difference for a particular pair of
allocations (e.g. no host pointer is available for the device allocation),
it might be possible to exploit that later in the application for another device
that does support host unified memory.
