<%
    OneApi=tags['$OneApi']
    x=tags['$x']
    X=x.upper()
%>

==========================
Level Zero UR Reference Document
==========================

This document gives general guidelines on differences in the UR L0 adapter for customer usecases.

Environment Variables
=====================

+---------------------------------------------+--------------------------------------------------------------+--------------------------------------------------------------+------------------+
| Environment Variable                        | Description                                                  | Possible Values                                              | Default Value    |
+=============================================+==============================================================+==============================================================+==================+
| UR_L0_USE_COPY_ENGINE                       | Controls the use of copy engines.                            | "0": Copy engines will not be used.                          | "1"              |
|                                             |                                                              | "1": All available copy engines can be used.                 |                  |
|                                             |                                                              | "lower_index:upper_index": Specifies a range of copy engines |                  |
|                                             |                                                              | to be used.                                                  |                  |
+---------------------------------------------+--------------------------------------------------------------+--------------------------------------------------------------+------------------+
| UR_L0_USE_IMMEDIATE_COMMANDLISTS            | Determines the mode of immediate command lists.              | "0": Immediate command lists are not used.                   | "0"              |
|                                             |                                                              | "1": Immediate command lists are used per queue.             |                  |
|                                             |                                                              | "2": Immediate command lists are used per thread per queue.  |                  |
+---------------------------------------------+--------------------------------------------------------------+--------------------------------------------------------------+------------------+
| UR_L0_USE_RELAXED_ALLOCATION_LIMITS         | Controls the use of relaxed allocation limits.               | "0": Relaxed allocation limits are not used.                 | "0"              |
|                                             |                                                              | "1": Relaxed allocation limits are used.                     |                  |
+---------------------------------------------+--------------------------------------------------------------+--------------------------------------------------------------+------------------+
| UR_L0_USE_DRIVER_INORDER_LISTS              | Controls the use of in-order lists from the driver.          | "0": In-order lists from the driver are not used.            | "0"              |
|                                             |                                                              | "1": In-order lists from the driver are used.                |                  |
+---------------------------------------------+--------------------------------------------------------------+--------------------------------------------------------------+------------------+
| UR_L0_USM_ALLOCATOR_TRACE                   | Enables tracing for the USM allocator.                       | "0": Tracing is disabled.                                    | "0"              |
|                                             |                                                              | "1": Tracing is enabled.                                     |                  |
+---------------------------------------------+--------------------------------------------------------------+--------------------------------------------------------------+------------------+
| UR_L0_USM_ALLOCATOR                         | Configures the USM allocator.                                | Specifies the configuration for the USM allocator.           | All Configs      |
+---------------------------------------------+--------------------------------------------------------------+--------------------------------------------------------------+------------------+
| UR_L0_DEBUG_BASIC                           | Enables basic debugging for Level Zero.                      | "0": Debugging is disabled.                                  | "0"              |
|                                             |                                                              | "1": Debugging is enabled.                                   |                  |
+---------------------------------------------+--------------------------------------------------------------+--------------------------------------------------------------+------------------+
| UR_L0_ENABLE_SYSMAN_ENV_DEFAULT             | Controls the default SysMan environment initialization.      | "1" or unset: Enables SysMan environment initialization.     | "1"              |
|                                             |                                                              | "0": Disables SysMan environment initialization.             |                  |
+---------------------------------------------+--------------------------------------------------------------+--------------------------------------------------------------+------------------+
| UR_L0_ENABLE_ZESINIT_DEFAULT                | Controls the default SysMan initialization with zesInit.     | "1": Enables SysMan initialization with zesInit.             | "0"              |
|                                             |                                                              | "0" or unset: Disables SysMan initialization with zesInit.   |                  |
+---------------------------------------------+--------------------------------------------------------------+--------------------------------------------------------------+------------------+
| SYCL_ENABLE_PCI                             | Deprecated and no longer needed.                             | Any value: Triggers a warning message.                       | None             |
+---------------------------------------------+--------------------------------------------------------------+--------------------------------------------------------------+------------------+
| UR_L0_USE_COPY_ENGINE_FOR_FILL              | Controls the use of copy engines for memory fill operations. | "0": Copy engines will not be used for fill operations.      | "0"              |
|                                             |                                                              | "1": Copy engines will be used for fill operations.          |                  |
+---------------------------------------------+--------------------------------------------------------------+--------------------------------------------------------------+------------------+
| UR_L0_DEBUG                                 | Controls the debug level for Level Zero.                     | "0": No debug information.                                   | "0"              |
|                                             |                                                              | "1": Basic debug information.                                |                  |
|                                             |                                                              | "2": Validation debug information.                           |                  |
|                                             |                                                              | "-1": All debug information.                                 |                  |
+---------------------------------------------+--------------------------------------------------------------+--------------------------------------------------------------+------------------+
| UR_L0_LEAKS_DEBUG                           | Enables debugging for memory leaks.                          | "0": Memory leaks debugging is disabled.                     | "0"              |
|                                             |                                                              | "1": Memory leaks debugging is enabled.                      |                  |
+---------------------------------------------+--------------------------------------------------------------+--------------------------------------------------------------+------------------+
| UR_L0_INIT_ALL_DRIVERS                      | Controls the initialization of all Level Zero drivers.       | "0": Only currently used drivers are initialized.            | "0"              |
|                                             |                                                              | "1": All drivers on the system are initialized.              |                  |
+---------------------------------------------+--------------------------------------------------------------+--------------------------------------------------------------+------------------+
| UR_L0_SERIALIZE                             | Controls serialization of Level Zero calls.                  | "0": No locking or blocking.                                 | "0"              |
|                                             |                                                              | "1": Locking around each UR_CALL.                            |                  |
|                                             |                                                              | "2": Blocking UR calls where supported.                      |                  |
+---------------------------------------------+--------------------------------------------------------------+--------------------------------------------------------------+------------------+
| UR_L0_QUEUE_SYNCHRONIZE_NON_BLOCKING        | Controls non-blocking synchronization of queues.             | "0": Non-blocking synchronization is disabled.               | "0"              |
|                                             |                                                              | "1": Non-blocking synchronization is enabled.                |                  |
+---------------------------------------------+--------------------------------------------------------------+--------------------------------------------------------------+------------------+
| UR_L0_OOQ_INTEGRATED_SIGNAL_EVENT           | Controls signal events for commands on integrated GPUs.      | "0": Signal events are not created.                          | "0"              |
|                                             |                                                              | "1": Signal events are created.                              |                  |
+---------------------------------------------+--------------------------------------------------------------+--------------------------------------------------------------+------------------+
| UR_L0_TRACK_INDIRECT_ACCESS_MEMORY          | Enables tracking of indirect access memory.                  | "0": Tracking is disabled.                                   | "0"              |
|                                             |                                                              | "1": Tracking is enabled.                                    |                  |
+---------------------------------------------+--------------------------------------------------------------+--------------------------------------------------------------+------------------+
| UR_L0_EXPOSE_CSLICE_IN_AFFINITY_PARTITIONING| Controls exposure of CSlice in affinity partitioning.        | "0": CSlice is not exposed.                                  | "0"              |
|                                             |                                                              | "1": CSlice is exposed.                                      |                  |
+---------------------------------------------+--------------------------------------------------------------+--------------------------------------------------------------+------------------+
| UR_L0_MAX_NUMBER_OF_EVENTS_PER_EVENT_POOL   | Sets the maximum number of events per event pool.            | Any positive integer: Specifies the maximum number of events | 256              |
|                                             |                                                              | per event pool.                                              |                  |
+---------------------------------------------+--------------------------------------------------------------+--------------------------------------------------------------+------------------+
| UR_L0_COMMANDLISTS_CLEANUP_THRESHOLD        | Sets the threshold for command lists cleanup.                | Any positive integer: Specifies the threshold for cleanup.   | 20               |
|                                             |                                                              | Negative value: Disables the threshold.                      |                  |
+---------------------------------------------+--------------------------------------------------------------+--------------------------------------------------------------+------------------+
| UR_L0_USE_NATIVE_USM_MEMCPY2D               | Controls the use of native USM memcpy2D operations.          | "0": Native USM memcpy2D operations are not used.            | "0"              |
|                                             |                                                              | "1": Native USM memcpy2D operations are used.                |                  |
+---------------------------------------------+--------------------------------------------------------------+--------------------------------------------------------------+------------------+
| UR_L0_ENABLE_USM_HOSTPTR_IMPORT             | Enables USM host pointer import.                             | "0": USM host pointer import is disabled.                    | "0"              |
|                                             |                                                              | "1": USM host pointer import is enabled.                     |                  |
+---------------------------------------------+--------------------------------------------------------------+--------------------------------------------------------------+------------------+
| UR_L0_ENABLE_USM_HOSTPTR_RELEASE            | Enables USM host pointer release.                            | "0": USM host pointer release is disabled.                   | "0"              |
|                                             |                                                              | "1": USM host pointer release is enabled.                    |                  |
+---------------------------------------------+--------------------------------------------------------------+--------------------------------------------------------------+------------------+
| UR_L0_ENABLE_USM_HOST_UNIFIED_MEMORY        | Enables USM host unified memory.                             | "0": USM host unified memory is disabled.                    | "0"              |
|                                             |                                                              | "1": USM host unified memory is enabled.                     |                  |
+---------------------------------------------+--------------------------------------------------------------+--------------------------------------------------------------+------------------+
| UR_L0_USE_MULTIPLE_COMMANDLIST_BARRIERS     | Controls the use of multiple command lists for barriers.     | "0": Multiple command lists are not used.                    | "0"              |
|                                             |                                                              | "1": Multiple command lists are used.                        |                  |
+---------------------------------------------+--------------------------------------------------------------+--------------------------------------------------------------+------------------+
| UR_L0_IN_ORDER_BARRIER_BY_SIGNAL            | Controls if in-order barriers are implemented by signal.     | "0": Barriers are implemented by true barrier command.       | "0"              |
|                                             |                                                              | "1": Barriers are implemented by signal.                     |                  |
+---------------------------------------------+--------------------------------------------------------------+--------------------------------------------------------------+------------------+
| UR_L0_DISABLE_EVENTS_CACHING                | Controls the caching of events in the context.               | "0" or unset: Event caching is enabled.                      | "0"              |
|                                             |                                                              | "1": Event caching is disabled.                              |                  |
+---------------------------------------------+--------------------------------------------------------------+--------------------------------------------------------------+------------------+
| UR_L0_REUSE_DISCARDED_EVENTS                | Controls the reuse of uncompleted events in in-order queues. | "0": Reuse of discarded events is disabled.                  | "1"              |
|                                             |                                                              | "1" or unset: Reuse of discarded events is enabled.          |                  |
+---------------------------------------------+--------------------------------------------------------------+--------------------------------------------------------------+------------------+
| SYCL_PI_LEVEL_ZERO_FILTER_EVENT_WAIT_LIST   | Controls filtering of event wait lists.                      | "0" or unset: Filtering is disabled.                         | "0"              |
|                                             |                                                              | "1": Filtering is enabled.                                   |                  |
+---------------------------------------------+--------------------------------------------------------------+--------------------------------------------------------------+------------------+
| UR_L0_DEVICE_SCOPE_EVENTS                   | Controls the scope of device events.                         | "0": All events are host-visible.                            | "0"              |
|                                             |                                                              | "1": On-demand host-visible proxy events.                    |                  |
|                                             |                                                              | "2": Last command in batch host-visible.                     |                  |
+---------------------------------------------+--------------------------------------------------------------+--------------------------------------------------------------+------------------+
| UR_L0_USE_COPY_ENGINE_FOR_D2D_COPY          | Controls the use of copy engines for device-to-device copy   | "0": Copy engines will not be used for D2D copy operations.  | "0"              |
|                                             | operations.                                                  | "1": Copy engines will be used for D2D copy operations.      |                  |
+---------------------------------------------+--------------------------------------------------------------+--------------------------------------------------------------+------------------+
| UR_L0_BATCH_SIZE                            | Controls the batch size for command lists.                   | "0": Dynamic batch size adjustment.                          | "0"              |
|                                             |                                                              | Any positive integer: Specifies the fixed batch size.        |                  |
+---------------------------------------------+--------------------------------------------------------------+--------------------------------------------------------------+------------------+
| UR_L0_COPY_BATCH_SIZE                       | Controls the batch size for copy command lists.              | "0": Dynamic batch size adjustment.                          | "0"              |
|                                             |                                                              | Any positive integer: Specifies the fixed batch size.        |                  |
+---------------------------------------------+--------------------------------------------------------------+--------------------------------------------------------------+------------------+
| UR_L0_IMMEDIATE_COMMANDLISTS_BATCH_MAX      | Sets the maximum number of immediate command lists batches.  | Any positive integer: Specifies the maximum number of batches| 10               |
+---------------------------------------------+--------------------------------------------------------------+--------------------------------------------------------------+------------------+
|UR_L0_IMMEDIATE_COMMANDLISTS_EVENTS_PER_BATCH| Sets the number of events per batch for immediate command    | Any positive integer: Specifies the number of events per     | 256              |
|                                             | lists.                                                       | batch.                                                       |                  |
+---------------------------------------------+--------------------------------------------------------------+--------------------------------------------------------------+------------------+
| UR_L0_USE_COMPUTE_ENGINE                    | Controls the use of compute engines.                         | "0": Only the first compute engine is used.                  | "0"              |
|                                             |                                                              | Any positive integer: Specifies the index of the compute     |                  |
|                                             |                                                              | engine to be used.                                           |                  |
|                                             |                                                              | Negative value: All available compute engines may be used.   |                  |
+---------------------------------------------+--------------------------------------------------------------+--------------------------------------------------------------+------------------+
| UR_L0_USM_RESIDENT                          | Controls memory residency for USM allocations.               | "0xHSD": Specifies residency for host, shared, and device    | 0x2              |
|                                             |                                                              | allocations.                                                 |                  |
+---------------------------------------------+--------------------------------------------------------------+--------------------------------------------------------------+------------------+
| UR_L0_DISABLE_USM_ALLOCATOR                 | Controls the use of the USM allocator.                       | "0": USM allocator is enabled.                               | "0"              |
|                                             |                                                              | Any other value: USM allocator is disabled.                  |                  |
+---------------------------------------------+--------------------------------------------------------------+--------------------------------------------------------------+------------------+

Contributors
------------

* Neil Spruit `neil.r.spruit@intel.com <neil.r.spruit@intel.com>`_