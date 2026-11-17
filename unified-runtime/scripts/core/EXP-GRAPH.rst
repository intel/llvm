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
   * ${x}GraphGetIdExp
* Queue
   * ${x}QueueBeginGraphCaptureExp
   * ${x}QueueBeginCaptureIntoGraphExp
   * ${x}QueueEndGraphCaptureExp
   * ${x}QueueIsGraphCaptureEnabledExp
   * ${x}QueueGetGraphExp
   * ${x}EnqueueGraphExp

Error Codes
--------------------------------------------------------------------------------

The following result codes report misuse of graph capture. See
${x}QueueBeginGraphCaptureExp for the rules that apply while a queue is in graph
capture mode.

* ${X}_RESULT_ERROR_GRAPH_CAPTURE_UNSUPPORTED - an operation that is not
  supported during graph capture was called. This includes operations that
  synchronize the host with recorded work, such as ${x}QueueFinish,
  ${x}EventWait or a blocking enqueue.
* ${X}_RESULT_ERROR_GRAPH_CAPTURE_MERGE_ATTEMPT - a command enqueued to a queue
  that is already capturing waited on an event recorded by a different, still
  active, graph capture.
* ${X}_RESULT_ERROR_GRAPH_INTERNAL_EVENT - an event recorded into a graph was
  used outside of that graph after the capture ended, or a command recorded
  into a graph waited on an event signaled outside of it.
* ${X}_RESULT_ERROR_COMMAND_LIST_NOT_CAPTURING - ${x}QueueEndGraphCaptureExp was
  called on a queue that is not in graph capture mode.
* ${X}_RESULT_ERROR_GRAPH_UNJOINED_FORKS - ${x}QueueEndGraphCaptureExp was called
  while a forked queue was not joined back.
* ${X}_RESULT_ERROR_INVALID_GRAPH - an invalid graph, such as one whose capture
  ended with ${X}_RESULT_ERROR_GRAPH_UNJOINED_FORKS, was passed to
  ${x}GraphInstantiateGraphExp or ${x}GraphIsEmptyExp.

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
| 1.2       | Extend ${x}_command_t enumerator with       |
|           | enqueue graph event entry. Cleanup spec     |
|           | entry descriptions and return values.       |
|           | Rename QueueAppendGraphExp into             |
|           | EnqueueGraphExp.                            |
+-----------+---------------------------------------------+
| 1.3       | Add ${x}QueueGetGraphExp to retrieve graph  |
|           | handle from queue in capture mode.          |
+-----------+---------------------------------------------+
| 1.4       | Add ${x}GraphSetDestructionCallbackExp to   |
|           | register user callbacks invoked on graph    |
|           | destruction.                                |
+-----------+---------------------------------------------+
| 1.5       | Add ${x}GraphGetIdExp to retrieve a         |
|           | process-unique identifier for a graph.      |
+-----------+---------------------------------------------+
| 1.6       | Document when graph capture error codes are |
|           | returned.                                   |
+-----------+---------------------------------------------+

Support
--------------------------------------------------------------------------------

Adapters which support this experimental feature *must* return true for the new
``${X}_DEVICE_INFO_GRAPH_RECORD_AND_REPLAY_SUPPORT_EXP`` device info query.

Contributors
--------------------------------------------------------------------------------

* Krzysztof, Filipek `krzysztof.filipek@intel.com <krzysztof.filipek@intel.com>`_
* Krzysztof, Swiecicki `krzysztof.swiecicki@intel.com <krzysztof.swiecicki@intel.com>`_
* Matthew, Michel `matthew.michel@intel.com <matthew.michel@intel.com>`_
