<%
    OneApi=tags['$OneApi']
    x=tags['$x']
    X=x.upper()
%>

.. _experimental-graph:

================================================================================
Graph
================================================================================

.. warning::

    Experimental features:

    *   May be replaced, updated, or removed at any time.
    *   Do not require maintaining API/ABI stability of their own additions over
        time.


Motivation
--------------------------------------------------------------------------------
This extension introduces functionality for recording enqueued operations into a
graph for later execution. Queues can enter graph capture mode, where operations
enqueued to them are recorded into a graph instead of being executed immediately.
This graph can then be instantiated as an executable graph and appended to a queue
multiple times for repeated execution.

API
--------------------------------------------------------------------------------

Enums
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ${x}_device_info_t
    * ${X}_DEVICE_INFO_GRAPH_RECORD_AND_REPLAY_SUPPORT_EXP

Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Graph
   * ${x}GraphCreateExp
   * ${x}GraphInstantiateGraphExp
   * ${x}GraphDestroyExp
   * ${x}GraphExecutableGraphDestroyExp
   * ${x}GraphIsEmptyExp
   * ${x}GraphDumpContentsExp
* Queue
   * ${x}QueueBeginGraphCaptureExp
   * ${x}QueueBeginCaptureIntoGraphExp
   * ${x}QueueEndGraphCaptureExp
   * ${x}QueueAppendGraphExp
   * ${x}QueueIsGraphCaptureEnabledExp

Changelog
--------------------------------------------------------------------------------

+-----------+---------------------------------------------+
| Revision  | Changes                                     |
+===========+=============================================+
| 1.0       | Initial Draft                               |
+-----------+---------------------------------------------+
| 1.1       | Extend ${x}_device_info_t enumerator with   |
|           | graph record and replay entry.              |
+-----------+---------------------------------------------+

Support
--------------------------------------------------------------------------------

Adapters which support this experimental feature *must* return true for the new
``${X}_DEVICE_INFO_GRAPH_RECORD_AND_REPLAY_SUPPORT_EXP`` device info query.

Contributors
--------------------------------------------------------------------------------

* Krzysztof, Filipek `krzysztof.filipek@intel.com <krzysztof.filipek@intel.com>`_
* Krzysztof, Swiecicki `krzysztof.swiecicki@intel.com <krzysztof.swiecicki@intel.com>`_
