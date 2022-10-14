# Improved host task

In the SYCL 2020 specification the SYCL `host_task` allows for host work to be
scheduled within the SYCL DAG, in addition it can also be used to perform
interoperability work.

However the interoperability provided in the `host_task` doesn't include any
`sycl::event` interoperability, or any way to model native dependencies between
SYCL submitted work and work submitted with interoperability.

This means that synchronization between SYCL DAG tasks and interoperability
tasks enqueued using the `host_task` have to be done on host, even if the device
and SYCL implementation would support handling these dependencies on the device.
Which makes the SYCL 2020 `host_task` interface for interoperability inefficient
on some platforms.

This proposal aims to address these issues, and improve on the SYCL 2020
`host_task`, to provide better integrated interoperability solutions.

## SYCL specification changes

```diff
diff --git a/adoc/chapters/programming_interface.adoc b/adoc/chapters/programming_interface.adoc
index 45cb974..0fb092a 100644
--- a/adoc/chapters/programming_interface.adoc
+++ b/adoc/chapters/programming_interface.adoc
@@ -14252,65 +14252,62 @@ include::{code_dir}/usingSpecConstants.cpp[lines=4..-1]
 [[sec:interfaces.hosttasks.overview]]
 === Overview
 
-A <<host-task>> is a native {cpp} callable which is scheduled by the
-<<sycl-runtime>>. A <<host-task>> is submitted to a <<queue>> via a
-<<command-group>> by a <<host-task-command>>.
+A <<host-task>> is a native {cpp} callable which is scheduled and invoked on
+the host by the <<sycl-runtime>>. Therefore it does not have the same
+restrictions as a <<sycl-kernel-function>> and can contain arbitrary {cpp}
+code.
+
+A <<host-task>> is submitted to a <<queue>> via a <<command-group>> by a
+<<host-task-command>>. And all <<queue,queues>> support
+<<host-task-command,host task commands>>.
+
+When a <<host-task-command>> is submitted to a <<queue>> it is scheduled based
+on its explicit dependencies as well as its data dependencies with other
+<<command,commands>> including <<kernel-invocation-command,kernel invocation
+commands>> and asynchronous copies, and resolving any requisites created by
+<<accessor,accessors>> attached to the <<command-group>> as defined in
+<<sub.section.memmodel.app>>.
 
-When a <<host-task-command>> is submitted to a <<queue>> it is scheduled
-based on its data dependencies with other <<command,commands>> including
-<<kernel-invocation-command,kernel invocation commands>> and asynchronous copies, resolving any
-requisites created by <<accessor,accessors>> attached to the <<command-group>> as
-defined in <<sub.section.memmodel.app>>.
+Any uncaught exception thrown during the execution of a <<host-task>> will be
+turned into an <<async-error>> that can be handled as described in
+<<subsubsec:exception.async>>.
 
-Since a <<host-task>> is invoked directly by the <<sycl-runtime>> rather
-than being compiled as a <<sycl-kernel-function>>, it does not have the same
-restrictions as a <<sycl-kernel-function>>, and can therefore contain any
-arbitrary {cpp} code.
+A <<host-task,host task>> can be used as a task which can perform arbitrary
+{cpp} code within the scheduling of the <<sycl-runtime>> and additionally as a
+task which can perform interoperability at a point within the scheduling of the
+<<sycl-runtime>>.
 
 Capturing <<accessor,accessors>> in a <<host-task>> is allowed, however,
 capturing or using any other SYCL class that has reference semantics (see
 <<sec:reference-semantics>>) is undefined behavior.
 
-A <<host-task>> can be enqueued on any <<queue>> and the callable will be
-invoked directly by the SYCL runtime, regardless of which <<device>> the
-<<queue>> is associated with.
+A host task can use buffer or image accessors with [code]#target::host_task#,
+or [code]#image_target::host_task#. These accessors make the buffer or
+image available on the host when the callable is executed.
 
-A <<host-task>> is enqueued on a <<queue>> via the [code]#host_task#
-member function of the [code]#handler# class.
-The <<event>> returned by the submission of the associated <<command group>>
-enters the completed state (corresponding to a status of
-[code]#info::event_command_status::complete#) once the invocation of the
-provided {cpp} callable has returned.
-Any uncaught exception thrown during the execution of a <<host-task>> will be
-turned into an <<async-error>> that can be handled as described in
-<<subsubsec:exception.async>>.
-
-A <<host-task>> can optionally be used to interoperate with the
-<<native-backend-object,native backend objects>> associated with the <<queue>> executing the
-<<host-task>>, the <<context>> that the <<queue>> is associated with, the
-<<device>> that the <<queue>> is associated with  and the <<accessor,accessors>>
-that have been captured in the callable, via an optional
-[code]#interop_handle# parameter.
-
-This allows <<host-task,host tasks>> to be used for two purposes: either as a
-task which can perform arbitrary {cpp} code within the scheduling of the
-<<sycl-runtime>> or as a task which can perform interoperability at a point
-within the scheduling of the <<sycl-runtime>>.
-
-For the former use case, construct a buffer accessor with
-[code]#target::host_task# or an image accessor with
-[code]#image_target::host_task#.  This makes the buffer or image available
-on the host during execution of the <<host-task>>.
-
-For the latter case, construct a buffer accessor with
-[code]#target::device# or [code]#target::constant_buffer#, or construct
-an image accessor with [code]#image_target::device#.  This makes the buffer or
-image available on the device that is associated with the queue used to submit
-the <<host-task>>, so that it can be accessed via interoperability member
-functions provided by the [code]#interop_handle# class.
+A host task can use buffer or image accessors with [code]#target::device#,
+[code]#target::constant_buffer# or [code]#image_target::device#. These
+accessors can only be used for interoperability through the
+[code]#interop_handle#.
 
 Local <<accessor,accessors>> cannot be used within a <<host-task>>.
 
+The {cpp} callable can have either no parameters or one parameter, in which
+case it must be an [code]#interop_handle#. The {cpp} callabe can either return
+nothing, or a standard vector of native events.
+
+The <<event>> returned by the submission of the <<command group>> associated
+with a <<host-task>> enters the completed state (corresponding to a status of
+[code]#info::event_command_status::complete#) once the invocation of the
+provided {cpp} callable has returned, and if the callable has returned a vector
+of <<native-backend-object, native backend objects>> events, once all of these
+returned events are complete as well. Note that it means that at the SYCL level
+it is only possible to synchronize with the interoperability work enqueued from
+one <<host-task>> all at once. If more granularity is required in synchronizing
+the interoperability work, then multiple <<host-task,host tasks>> should be
+used instead, to obtain separate SYCL events for specific interoperability
+operations.
+
 // TODO: access mode/target resolution rules
 
 [source,,linenums]
@@ -14318,6 +14315,41 @@ Local <<accessor,accessors>> cannot be used within a <<host-task>>.
 include::{header_dir}/hostTask/hostTaskSynopsis.h[lines=4..-1]
 ----
 
+[[sec:interfaces.hosttasks.properties]]
+=== Properties
+
+[[table.properties.host_task]]
+.Properties supported by the SYCL [code]#host_task#
+[width="100%",options="header",separator="@",cols="65%,35%"]
+|====
+@ Property @ Description
+a@
+[source]
+----
+property::host_task::exec_on_submit
+----
+   a@ The [code]#exec_on_submit# property instructs the SYCL runtime to execute
+   the [code]#host_task# callable at submission. This means that with this
+   property the callable is guaranteed to have completed when the queue
+   submission returns.
+
+a@
+[source]
+----
+property::buffer::manual_interop_sync
+----
+   a@ The [code]#manual_interop_sync# property instructs the [code]#host_task#
+   that it may defer resolving dependencies to the user by providing
+   interoperability events through the [code]#interop_handle#'s
+   [code]#get_native_events# member function. This property can only be used
+   with a backend that supports event interoperability. When this property is
+   used, it is the user's responsibility to ensure that all the events returned
+   by [code]#get_native_events# are handled apropriately in the <<host-task>>
+   callable.  Accessing data from captured accessors before handling the
+   dependencies modeled by [code]#get_native_events# results undefined
+   behavior.
+
+|====
 
 [[subsec:interfaces.hosttasks.interophandle]]
 === Class [code]#interop_handle#
@@ -14474,6 +14506,40 @@ _Throws:_ Must throw an [code]#exception# with the
 [code]#errc::backend_mismatch# error code if [code]#Backend != get_backend()#.
 --
 
+[[subsec:interfaces.hosttask.interophandle.events]]
+==== Event handling member functions
+
+[source,,linenums]
+----
+include::{header_dir}/hostTask/classInteropHandle/events.h[lines=4..-1]
+----
+
+  . _Constraints:_ Available only if the optional interoperability
+    function [code]#get_native# taking an [code]#event# is
+    available.
++
+--
+_Returns:_ A list of <<native-backend-object, native backend objects>>
+associated with <<event, events>> that represent any unresolved dependencies at
+the time of the <<host-task>> callable execution. It is the responsibility of
+the user to resolve these dependencies using the backend APIs.
+
+_Throws:_ Must throw an [code]#exception# with the
+[code]#errc::backend_mismatch# error code if [code]#Backend != get_backend()#.
+--
+  . _Constraints:_ Available only if the optional interoperability
+    function [code]#get_native# taking an [code]#event# is
+    available.
++
+--
+_Parameters_: A list of <<native-backend-object, native backend objects>>
+events, that need to be completed before the <<host-task>> can be considered
+completed.
+_Returns:_ No returns.
+
+_Throws:_ Must throw an [code]#exception# with the
+[code]#errc::backend_mismatch# error code if [code]#Backend != get_backend()#.
+--
 
 [[subsec:interfaces.hosttask.handler]]
 === Additions to the [code]#handler# class
diff --git a/adoc/headers/hostTask/classHandler/hostTask.h b/adoc/headers/hostTask/classHandler/hostTask.h
index 374845e..ae4f616 100644
--- a/adoc/headers/hostTask/classHandler/hostTask.h
+++ b/adoc/headers/hostTask/classHandler/hostTask.h
@@ -7,7 +7,8 @@ class handler {
       public
       : template <typename T>
         void
-        host_task(T&& hostTaskCallable); // (1)
-
+        host_task(T&& hostTaskCallable,
+                  const property_list& propList = {}); // (1)
+                                                       //
   ...
 };
diff --git a/adoc/headers/hostTask/classInteropHandle/events.h b/adoc/headers/hostTask/classInteropHandle/events.h
new file mode 100644
index 0000000..009d576
--- /dev/null
+++ b/adoc/headers/hostTask/classInteropHandle/events.h
@@ -0,0 +1,6 @@
+// Copyright (c) 2011-2021 The Khronos Group, Inc.
+// SPDX-License-Identifier: MIT
+
+template <backend Backend>
+std::vector<backend_return_t<Backend, event>> get_native_events() const; // (1)
+                                                                         //
diff --git a/adoc/headers/hostTask/hostTaskSynopsis.h b/adoc/headers/hostTask/hostTaskSynopsis.h
index ace3430..82bb463 100644
--- a/adoc/headers/hostTask/hostTaskSynopsis.h
+++ b/adoc/headers/hostTask/hostTaskSynopsis.h
@@ -36,8 +36,26 @@ class interop_handle {
 
   template <backend Backend>
   backend_return_t<Backend, context> get_native_context() const;
+
+  template <backend Backend>
+  std::vector<backend_return_t<Backend, event>> get_native_events() const;
+
 };
 
+namespace property {
+namespace host_task {
+  class exec_on_submit {
+    public:
+      exec_on_submit() = default;
+  };
+
+  class manual_interop_sync {
+    public:
+      manual_interop_sync() = default;
+  };
+} // namespace host_task
+} // namespace property
+
 class handler {
   ...
 
@@ -46,7 +64,7 @@ class handler {
 
       template <typename T>
       void
-      host_task(T&& hostTaskCallable);
+      host_task(T&& hostTaskCallable, const property_list& propList = {});
 
   ...
 };
```

## Examples

This section collates examples illustrating the `host_task` interface.

For the purpose of demonstration OpenCL interoperability is used but the
principles are applicable to other backends.

### Regular `host_task`

The following examples are showcasing the original SYCL 2020 `host_task` and its
shortcomings.

Simple `host_task`:
```cpp
sycl::event e = queue.submit([&](sycl::handler &cgh) {
  cgh.host_task([&]() { printf("Hello World!\n"); });
});

// Waiting on the event waits for the lambda to complete
e.wait();
```

Interoperability `host_task`:
```cpp
sycl::event e = queue.submit([&](sycl::handler &cgh) {
  auto accb =
      B.get_access<sycl::access::mode::read_write, sycl::target::device>(cgh);

  cgh.host_task([&](sycl::interop_handle ih) {
    auto queue = ih.get_native_queue();
    auto mem = ih.get_native_mem(accb);

    int pattern = 42;
    cl_event ev;
    clEnqueueFillBuffer(queue, mem, &pattern, sizeof(int), 1 * sizeof(int),
                        1 * sizeof(int), 0, nullptr, &ev);

    // In this mode asynchronous work must be waited on inside of the host task
    clWaitForEvents(1, &ev);
  });
});

// Waiting on the event waits for the lambda to complete
e.wait();
```

Interoperability `host_task` depending on a kernel:
```cpp
sycl::event ek = queue.submit([=](sycl::handler &cgh) {
  auto accb =
      B.get_access<sycl::access::mode::read_write, sycl::target::device>(cgh);
  cgh.parallel_for<kernelA>(sycl::range<1>(1),
                            [=](sycl::id<1> idx) { accb[0] = 1; });
});

sycl::event eh = queue.submit([&](sycl::handler &cgh) {
  // Data dependency with kernelA, the SYCL runtime will wait for event ek to be
  // complete before executing the host_task lambda
  auto accb =
      B.get_access<sycl::access::mode::read_write, sycl::target::device>(cgh);

  cgh.host_task([&](sycl::interop_handle ih) {
    auto queue = ih.get_native_queue();
    auto mem = ih.get_native_mem(accb);

    int pattern = 42;
    cl_event ev;
    clEnqueueFillBuffer(queue, mem, &pattern, sizeof(int), 1 * sizeof(int),
                        1 * sizeof(int), 0, nullptr, &ev);

    // In this mode asynchronous work must be waited on inside of the host task
    clWaitForEvents(1, &ev);
  });
});

// Waiting on the event waits for the lambda of the `host_task` to complete
eh.wait();
```

### `host_task` returning native events

Interoperability `host_task` returning a native event:
```cpp
sycl::event e = queue.submit([&](sycl::handler &cgh) {
  auto accb =
      B.get_access<sycl::access::mode::read_write, sycl::target::device>(cgh);

  cgh.host_task([&](sycl::interop_handle ih) {
    auto queue = ih.get_native_queue();
    auto mem = ih.get_native_mem(accb);

    int pattern = 42;
    cl_event ev;
    clEnqueueFillBuffer(queue, mem, &pattern, sizeof(int), 1 * sizeof(int),
                        1 * sizeof(int), 0, nullptr, &ev);

    // return a vector of native events
    return {ev};
  });
});

// Waiting on the event waits for the lambda to complete, then waits on any
// returned events, here `ev` to complete.
e.wait();
```

### `host_task` with `manual_interop_sync` property

Interoperability `host_task` using `manual_interop_sync` and depending on a kernel:
```cpp
sycl::event ek = queue.submit([=](sycl::handler &cgh) {
  auto accb =
      B.get_access<sycl::access::mode::read_write, sycl::target::device>(cgh);
  cgh.parallel_for<kernelA>(sycl::range<1>(1),
                            [=](sycl::id<1> idx) { accb[0] = 1; });
});

sycl::event eh = queue.submit([&](sycl::handler &cgh) {
  // Data dependency with kernelA, because of `manual_interop_sync`, if the SYCL
  // runtime can model this data dependency using a native event, the native
  // event will be made available through the interop handle get_native_events
  // method and kernelA will not be waited one before executing the `host_task`
  // lambda.
  auto accb =
      B.get_access<sycl::access::mode::read_write, sycl::target::device>(cgh);

  cgh.host_task(
      [&](sycl::interop_handle ih) {
        auto queue = ih.get_native_queue();
        auto mem = ih.get_native_mem(accb);

        auto events = ih.get_native_events();

        int pattern = 42;
        cl_event ev;
        clEnqueueFillBuffer(queue, mem, &pattern, sizeof(int), 1 * sizeof(int),
                            1 * sizeof(int), events.size(), events.data(), &ev);

        return {ev};
      },
      sycl::property_list{sycl::property::host_task::manual_interop_sync{}});
});

// Waiting on the event waits for the `host_task` lambda to complete, then waits
// on the returned native event `ev`, which in turns wait for the fill command,
// which then waits on the native event from kernelA.
eh.wait();
```

### `host_task` with `exec_on_submit` property

Simple `host_task` with `exec_on_submit`:
```cpp
sycl::event e = queue.submit([&](sycl::handler &cgh) {
  cgh.host_task(
      [&]() { printf("Hello World!\n"); },
      sycl::property_list{sycl::property::host_task::exec_on_submit{}});
});

// With `exec_on_submit` the `host_task` lambda completes within the command
// submission so the event here is already marked complete and waiting does
// nothing. 
e.wait();
```

Interoperability `host_task` returning a native event and using
`exec_on_submit`:
```cpp
sycl::event e = queue.submit([&](sycl::handler &cgh) {
  auto accb =
      B.get_access<sycl::access::mode::read_write, sycl::target::device>(
          cgh);

  cgh.host_task(
      [&](sycl::interop_handle ih) {
        auto queue = ih.get_native_queue();
        auto mem = ih.get_native_mem(accb);

        int pattern = 42;
        cl_event ev;
        clEnqueueFillBuffer(queue, mem, &pattern, sizeof(int), 1 * sizeof(int),
                            1 * sizeof(int), 0, nullptr, &ev);

        // return a vector of native events
        return {ev};
      },
      sycl::property_list{sycl::property::host_task::exec_on_submit{}});
});

// With `exec_on_submit` the `host_task` lambda completes within the command
// submission so the event here only waits on the returned native events.
e.wait();
```

### Complete example using all the new interoperability features

```cpp
queue.submit([=](sycl::handler &cgh) {
  auto accb =
      B.get_access<sycl::access::mode::read_write, sycl::target::device>(cgh);
  cgh.parallel_for<kernelA>(sycl::range<1>(1),
                            [=](sycl::id<1> idx) { accb[0] = 1; });
});

queue.submit([&](sycl::handler &cgh) {
  // Data dependency with kernelA, because of `manual_interop_sync` a native
  // event for kernelA may be provided through `ih.get_native_events`.
  auto accb =
      B.get_access<sycl::access::mode::read_write, sycl::target::device>(cgh);

  // Because of `exec_on_submit`, the fill buffer here will be enqueued
  // immediately without the potential overhead of launching the `host_task` lambda
  // in a separate thread.
  cgh.host_task(
      [&](sycl::interop_handle ih) {
        auto queue = ih.get_native_queue();
        auto mem = ih.get_native_mem(accb);

        auto events = ih.get_native_events();

        int pattern = 42;
        cl_event ev;
        clEnqueueFillBuffer(queue, mem, &pattern, sizeof(int), 1 * sizeof(int),
                            1 * sizeof(int), events.size(), events.data(), &ev);

        // return a vector of native events
        return {ev};
      },
      sycl::property_list{sycl::property::host_task::manual_interop_sync{},
                          sycl::property::host_task::exec_on_submit{}});
});

sycl::event e = queue.submit([=](sycl::handler &cgh) {
  // Data dependency on the `host_task`, the native event `ev` is used
  // internally to synchronize with kernelB.
  auto accb =
      B.get_access<sycl::access::mode::read_write, sycl::target::device>(cgh);
  cgh.parallel_for<kernelB>(sycl::range<1>(1),
                            [=](sycl::id<1> idx) { accb[2] = 3; });
});

// Waiting here will wait on kernelB, which will then wait on the `host_task`
// which will then wait on kernelA with no unnecessary waits.
e.wait();
```
