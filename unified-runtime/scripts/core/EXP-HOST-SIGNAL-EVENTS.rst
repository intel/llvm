<%
    OneApi=tags['$OneApi']
    x=tags['$x']
    X=x.upper()
%>

.. _experimental-host-signal-events:

================================================================================
Host-Signalled Events
================================================================================

.. warning::

    Experimental features:

    *   May be replaced, updated, or removed at any time.
    *   Do not require maintaining API/ABI stability of their own additions over
        time.
    *   Do not require conformance testing of their own additions.


Motivation
--------------------------------------------------------------------------------

An adapter can express a dependency between two commands only when both belong
to the same context. A dependency that crosses a context boundary has to be
resolved on the host: the runtime waits for the producing event and only then
submits the consuming command.

Deferring submission that way is correct but it moves the consuming command out
of the adapter's reach until the dependency has retired, which prevents the
adapter from batching, recording or otherwise pre-processing it.

This extension provides the missing primitive. A host-signalled event is created
unsignalled, is placed in the wait list of the consuming command, and is
signalled by the runtime once the producing event has retired. The consuming
command can therefore be submitted immediately, and the adapter resolves the
dependency itself.

The event is a synchronization primitive only. It conveys no data visibility
between devices; a transfer between contexts remains a separate command.

API
--------------------------------------------------------------------------------

Enums
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ${x}_device_info_t
    * ${X}_DEVICE_INFO_HOST_SIGNAL_EVENT_SUPPORT_EXP

Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ${x}EventCreateHostSignalExp
* ${x}EventHostSignalExp

Limitations
--------------------------------------------------------------------------------

* An unsignalled event in the wait list of a command may block the device
  submission channel that command was placed on, and with it unrelated work
  submitted to the same channel. Callers should not create a host-signalled
  event for a dependency that is already known to have retired.

* Every created event must eventually be signalled. An event abandoned
  unsignalled can leave a device submission channel blocked for the lifetime of
  the context, so error and teardown paths must signal outstanding events.

* The event must be signalled only after any cross-context data transfer the
  consuming command depends on has completed. Signalling merely on completion of
  the producing command is not sufficient when data has to move between
  contexts.

Changelog
--------------------------------------------------------------------------------

+-----------+---------------------------------------------+
| Revision  | Changes                                     |
+===========+=============================================+
| 1.0       | Initial Draft                               |
+-----------+---------------------------------------------+

Support
--------------------------------------------------------------------------------

Adapters which support this experimental feature *must* return ``true`` when
queried for ${X}_DEVICE_INFO_HOST_SIGNAL_EVENT_SUPPORT_EXP via
${x}DeviceGetInfo. Conversely, before using any of the functionality defined in
this experimental feature the user *must* use the device query to determine if
the adapter supports this feature.

Contributors
--------------------------------------------------------------------------------

* Slawomir Ptak `slawomir.ptak@intel.com <slawomir.ptak@intel.com>`_
