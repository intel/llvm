//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
#pragma once

#include "xpti/xpti_data_types.h"

#if defined(XPTI_STATIC_LIBRARY)
// If we are building or using the proxy
// static library we don't export any symbols
//
#define XPTI_EXPORT_API
#else
#if defined(_WIN64) || defined(_WIN32) /* Windows */
#ifdef XPTI_API_EXPORTS
#define XPTI_EXPORT_API __declspec(dllexport)
#else
#define XPTI_EXPORT_API __declspec(dllimport)
#endif
#else /* Generic Unix/Linux */
#ifdef XPTI_API_EXPORTS
#define XPTI_EXPORT_API __attribute__((visibility("default")))
#else
#define XPTI_EXPORT_API
#endif
#endif
#endif

#define XPTI_EXTRACT_USER_DEFINED_ID(val) ((uint16_t)val & 0x007f)
#define XPTI_TOOL_ID(val) (((uint16_t)val >> 8) & 0x00ff)

extern "C" {

/// @brief Initializes XPTI framework.
/// @details Initialize XPTI framework resources. Each user of XPTI must call
/// this function prior to any other XPTI API call. It is framework's
/// responsibility to ensure that resources are initialized once. Each call to
/// this function must have corresponding call to xptiFrameworkFinalize() to
/// ensure resources are freed.
XPTI_EXPORT_API void xptiFrameworkInitialize();

/// @brief Deinitializes XPTI framework.
/// @details Call to this function decrements framework's internal reference
/// counter. Once its value is equal to zero, XPTI framework can release
/// resources and unload subscribers.
XPTI_EXPORT_API void xptiFrameworkFinalize();

/// @brief Initialization function that is called when a new stream is generated
/// @details When a runtime or application that uses XPTI instrumentation API
/// starts to generate a new stream, a call to xptiInitialize() must be made to
/// let all subscribers know that a new stream is being generated. If the
/// subscribers are interested in this stream, they can the choose to subscribe
/// to the stream.
/// @param stream Name of the stream, for example "sycl", "opencl" etc
/// @param maj Major version number
/// @param min Minor version number
/// @param version Full version as a string
/// @return None
XPTI_EXPORT_API xpti::result_t xptiInitialize(const char *stream, uint32_t maj,
                                              uint32_t min,
                                              const char *version);

/// @brief Finalization function that is called when a stream halted
/// @details When a runtime or application that uses XPTI instrumentation API
/// stops generating the stream, a call to xptiFinalize() must be made to let
/// all subscribers know that the stream identified by 'stream' has stopped
/// generating events. If the subscribers are registered to receive events from
/// this stream, they can choose to unsubscribe from the stream or handle the
/// situation when the stream stop sending events.
/// @param stream Name of the stream, for example "sycl", "opencl" etc
/// @return None
XPTI_EXPORT_API void xptiFinalize(const char *stream);

/// @brief Returns universal ID
/// @details Universal ID is a 64 bit value, that can be used to correlate
/// events from different software layers. It is generated once for top SW layer
/// and then re-used by subsequent layers to identify original source code
/// location. This value is stored in thread-local storage.
XPTI_EXPORT_API uint64_t xptiGetUniversalId();

/// @brief Update universal ID value
/// @detail Save new universal ID value to thread-local storage. This function
/// is typically called by xpti::framework::tracepoint_t constructor when
/// updating tracepoint information. See xptiGetUniversalId() for more info
/// about universal IDs.
/// @param uid Unique 64 bit identifier.
XPTI_EXPORT_API void xptiSetUniversalId(uint64_t uid);

/// @brief Generates a unique ID
/// @details When a tool is subscribing to the event stream and wants to
/// generate task IDs that do not collide with unique IDs currently being
/// generated for nodes, edges and graphs, this API can be used. Any time, a
/// task that represents the instance of a node executing on the device is being
/// traced, the event ID corresponds to the unique ID of the node it represents
/// and in order to disambiguate the instances, a task ID can be generated and
/// sent as the instance ID for that task.
XPTI_EXPORT_API uint64_t xptiGetUniqueId();

/// @brief Register a string to the string table
/// @details All strings in the XPTI framework are referred to by their string
/// IDs and this method allow you to register a string and get the string ID for
/// it. In addition to the string ID, a reference to the string in the string
/// table is also returned. This lifetime of this string reference is equal to
/// the lifetime of the XPTI framework.
/// @param string The string to be registered with the string table. If the
/// string already exists in the string table, the previous ID is returned along
/// with the reference to the string in the string table.
/// @param table_string A reference to the string in the string table. This
/// string reference is guaranteed to be valid for the lifetime of the XPTI
/// framework.
/// @return The string ID of the string being registered. If an error occurs
/// during registration, xpti::invalid_id is returned.
XPTI_EXPORT_API xpti::string_id_t xptiRegisterString(const char *string,
                                                     char **table_string);

/// @brief Lookup a string in the string tablewith its string ID
/// @details All strings in the XPTI framework are referred to by their string
/// IDs and this method allows you to lookup a string by its string ID. The
/// lifetime of the returned string reference is equal to the lifetime of the
/// XPTI framework.
/// @param id The string ID of the string to lookup.
/// @return A reference to the string identified by the string ID.
XPTI_EXPORT_API const char *xptiLookupString(xpti::string_id_t id);

/// @brief Register an object to the object table
///
/// @details All object in the XPTI framework are referred to by their object
/// IDs and this method allow you to register an object and get the object ID
/// for it. This lifetime of this object reference is equal to the lifetime of
/// the XPTI framework.
/// @param data Raw bytes of data to be registered with the object table. If the
/// object already exists in the table, the previous ID is returned.
/// @param size Size in bytes of the object.
/// @param type One of xpti::metadata_type_t values. These only serve as a hint
/// to the tools for processing unknown values.
/// @return The ID of the object being registered. If an error occurs
/// during registration, xpti::invalid_id is returned.
XPTI_EXPORT_API xpti::object_id_t xptiRegisterObject(const char *data,
                                                     size_t size, uint8_t type);

/// @brief Lookup an object in the object table with its ID
///
/// @details All object in the XPTI framework are referred to by their object
/// IDs and this method allows you to lookup an object by its object ID. The
/// lifetime of the returned object reference is equal to the lifetime of the
/// XPTI framework.
/// @param id The ID of the object to lookup.
/// @return A reference to the object identified by the object ID.
XPTI_EXPORT_API xpti::object_data_t xptiLookupObject(xpti::object_id_t id);

/// @brief Register a payload with the framework
/// @details Since a payload may contain multiple strings that may have been
/// defined on the stack, it is recommended the payload object is registered
/// with the system as soon as possible. The framework will register all the
/// strings in the payload in the string table and replace the pointers to
/// strings on the stack with the pointers from the string table that should be
/// valid for the lifetime of the application.
/// @param payload The payload object that is registered with the system.
/// @return The unique hash value for the payload.
XPTI_EXPORT_API uint64_t xptiRegisterPayload(xpti::payload_t *payload);

/// @brief Register a stream by its name and get a stream ID
/// @details When events in a given stream have to be notified to the
/// subscribers, the stream ID to which the events belong to is required. This
/// method will register a stream by its name and return an ID that can be used
/// for notifications.
/// @param stream_name The stream name that needs to be registered.
/// @return The stream ID. If the stream has already been registered, the
/// previously generated stream ID is returned.
XPTI_EXPORT_API uint8_t xptiRegisterStream(const char *stream_name);

/// @brief Unregister a stream by its name
/// @details Unregistering a stream will invalidate the stream ID associated
/// with it by calling xptiFinalize() on all subscribers registered to this
/// stream and disabling all registered callbacks for this stream.
/// @param stream_name The stream name that needs to be unregistered.
/// @return The result code is XPTI_RESULT_SUCCESS when successful and
/// XPTI_RESULT_NOTFOUND if the stream is not found.
XPTI_EXPORT_API xpti::result_t xptiUnregisterStream(const char *stream_name);

/// @brief Registers a user defined trace point
/// @details The framework allows applications or runtimes using the framework
/// to extend the pre-defined tracepoint types. In order to facilitate this, a
/// tool name must be provided. This allows multiple vendors to instrument and
/// extend different software modules and have them behave well when put
/// together. However, the tool_name must be unique for this to behave well.
///
/// @code
///   typedef enum {
///     my_tp_extn1_begin = XPTI_TRACE_POINT_BEGIN(0),
///     my_tp_extn1_end = XPTI_TRACE_POINT_END(0),
///     my_tp_extn2_begin = XPTI_TRACE_POINT_BEGIN(1),
///     my_tp_extn2_end = XPTI_TRACE_POINT_END(1)
///   }tp_extension_t;
///   ...
///   uint16_t tp1_start = xptiRegisterUserDefinedTracePoint("myTest",
///                       my_tp_extn1_begin);
///   uint16_t tp1_end = xptiRegisterUserDefinedTracePoint("myTest",
///                         my_tp_extn1_end);
///   uint16_t tp2_start = xptiRegisterUserDefinedTracePoint("myTest",
///                       my_tp_extn2_begin);
///   uint16_t tp2_end = xptiRegisterUserDefinedTracePoint("myTest",
///                         my_tp_extn2_end);
///   ...
///   xptiNotifySubscribers(stream_id, tp1_start, parent, event, instance,
///                         nullptr);
/// @endcode
///
/// @param tool_name The tool name that is extending tracepoint types for its
/// use.
/// @param user_defined_tp The user defined tracepoint is a value ranging from
/// 0-127, which would allow vendors to create 64 pairs of tracepoints.
/// @return The result code is XPTI_RESULT_SUCCESS when successful and
/// XPTI_RESULT_NOTFOUND if the stream is not found.
XPTI_EXPORT_API uint16_t xptiRegisterUserDefinedTracePoint(
    const char *tool_name, uint8_t user_defined_tp);

/// @brief Registers a user defined event type
/// @details The framework allows applications or runtimes using the framework
/// to extend the pre-defined event types. In order to facilitate this, a
/// tool name must be provided. This allows multiple vendors to instrument and
/// extend different software modules and have them behave well when put
/// together. However, the tool_name must be unique for this to behave well.
///
/// @code
///   typedef enum {
///     my_ev_extn1 = XPTI_EVENT(0),
///     my_ev_extn2 = XPTI_EVENT(1)
///   } event_extension_t;
///   ...
///   uint16_t my_ev1 = xptiRegisterUserDefinedEventType("myTest", my_ev_extn1);
///   uint16_t my_ev2 = xptiRegisterUserDefinedEventType("myTest", my_ev_extn2);
///   ...
///   uint64_t InstanceNo;
///   MyEvent = xptiMakeEvent("application_foo", &Payload,
///                           my_ev1, xpti::trace_activity_type_t::active,
///                           &InstanceNo);
/// @endcode
///
/// In order for an notification to be received for such an event, a callback
/// must be registered.
///
/// @param tool_name The tool name that is extending tracepoint types for its
/// use.
/// @param user_defined_event The user defined event is a value ranging
/// from 0-127, which would allow vendors to create 127 new events under
/// tool_name.
/// @return The result code is XPTI_RESULT_SUCCESS when successful and
/// XPTI_RESULT_NOTFOUND if the stream is not found.
XPTI_EXPORT_API uint16_t xptiRegisterUserDefinedEventType(
    const char *tool_name, uint8_t user_defined_event);

/// @brief Creates a trace point event
/// @details When the application or runtime wants to instrument interesting
/// sections of the code, they can create trace point events that represent
/// these sections and use the created event to notify subscribers that such an
/// event ocurred. Each created event will have a unique ID. If the same payload
/// is provided to the xptiMakeEvent() function, the same trace event is
/// returned after looking up the invariant information in the payload
/// parameter. If the unique ID or the event itself has been cached, there will
/// be no lookup costs. However, if they are not cached, the same payload is
/// provded each time the section is encountered and the event that has been
/// created previously will be returned. This will however incur a lookup cost
/// and it is recommended that this be avoided to keep the instrumentation
/// overheads minimal.
///
/// @code
///   uint64_t InstanceNo;
///   trace_event_data_t *MyEvent;
///   xpti::payload_t Payload("foo", "foo.cpp", 100, 0, (void *)this);
///   MyEvent = xptiMakeEvent("foo", &Payload,
///                           xpti::trace_event_type_t::algorithm,
///                           xpti::trace_activity_type_t::active,
///                           &InstanceNo);
///
///   // Cache MyEvent locally so it can be used the next time around by
///   // avoiding a lookup
/// @endcode
///
/// @param name The name of the event, typically the function name or kernel
/// name, etc
/// @param payload The payload that uniquely describes the trace point which can
/// be done by the function name, source file name and line number within the
/// source file and the address of the function, for example.
/// @param event The event type of the current trace event being created, as in
/// is it a graph event or an algorithm event, etc.
/// @param activity The activity type for the event - as in active, background,
/// overhead etc.
/// @param instance_no This value is returned by the framework and represents
/// the instance number of this event. If the same event is attempted to be
/// created again, the instance ID give you an indication of how many times this
/// section has been visited.
/// @return The trace event representing the section's payload is returned.
XPTI_EXPORT_API xpti::trace_event_data_t *
xptiMakeEvent(const char *name, xpti::payload_t *payload, uint16_t event,
              xpti::trace_activity_type_t activity, uint64_t *instance_no);

/// @brief Retrieves a trace event given the unique id of the event
/// @details If the unique ID of a trace event is cached, this function allows
/// you to query the framework for the trace event data structure.
///
/// @param uid The unique ID of the event for which the lookup needs to be
/// performed
/// @return The trace event with unique ID equal to uid. If the unique ID is not
/// present, then nullptr will be returned.
XPTI_EXPORT_API const xpti::trace_event_data_t *xptiFindEvent(uint64_t uid);

/// @brief Retrieves the payload information associated with an event
/// @details An event encapsulates the unique payload it represents and this
/// function allows you to query the payload with the trace event data pointer.
///
/// @param lookup_object The trace event object for which the payload
/// information must be retrieved.
/// @return The payload data structure pointer for the event.
XPTI_EXPORT_API const xpti::payload_t *
xptiQueryPayload(xpti::trace_event_data_t *lookup_object);

/// @brief Retrieves the payload information associated with an universal ID
/// @details An universal ID references the unique payload it represents and
/// this function allows you to query the payload with the universal ID.
///
/// @param uid The universal ID for which the payload is to be retrieved.
/// @return The payload data structure pointer for the event.
XPTI_EXPORT_API const xpti::payload_t *xptiQueryPayloadByUID(uint64_t uid);

/// @brief Registers a callback for a trace point type
/// @details Subscribers receive notifications to the trace point types they
/// register a callback with. This function allows subscribers to register the
/// same or different callback with all trace point types.
///
/// @param stream_id The stream for which the registration is requested
/// @param trace_type The trace point type for which the registration must be
/// made. For example, you can register a different call back for
/// xpti::trace_point_type_t::task_begin and xpti::trace_point_type_t::task_end.
/// @param cb The callback function who's signature is of the type
/// xpti::tracepoint_callback_api_t
/// @return The result code which can be one of:
///            1. XPTI_RESULT_SUCCESS when the registration is successful
///            2. XPTI_RESULT_DUPLICATE when the callback function has already
///               been registered for the stream and trace point type
///            3. XPTI_RESULT_UNDELETE when the registration is for a callback
///               that had been previously unregistered.
XPTI_EXPORT_API xpti::result_t
xptiRegisterCallback(uint8_t stream_id, uint16_t trace_type,
                     xpti::tracepoint_callback_api_t cb);

/// @brief Unregisters a previously registered callback for a trace point type
/// @details Subscribers receive notifications to the trace point types they
/// register a callback with. This function allows subscribers to unregister
/// any previously registered callback  functions with this function so they can
/// stop receiving notifications.
///
/// @param stream_id The stream for which the registration must be disabled
/// @param trace_type The trace point type for which the registration must be
/// disabled.
/// @param cb The callback function who's signature is of the type
/// xpti::tracepoint_callback_api_t and must be disabled.
/// @return The result code which can be one of:
///            1. XPTI_RESULT_SUCCESS when the unregistration is successful
///            2. XPTI_RESULT_DUPLICATE when the callback function has already
///               been disabled for the stream and trace point type
///            3. XPTI_RESULT_NOTFOUND if the callback has not been previously
///               registered.
XPTI_EXPORT_API xpti::result_t
xptiUnregisterCallback(uint8_t stream_id, uint16_t trace_type,
                       xpti::tracepoint_callback_api_t cb);

/// @brief Notifies all registered subscribers that an event has occurred
/// @details Subscribers receive notifications to the trace point types they
/// register a callback with. This function allows subscribers to unregister
/// any previously registered callback  functions with this function so they can
/// stop receiving notifications.
///
/// @param stream_id The stream for which the registration must be disabled
/// @param trace_type The trace point type for which the notification is being
/// sent out
/// @param parent The parent trace event type for the current event. If none
/// exist, this can be nullptr.
/// @param object The event object for which the notification must be sent out.
/// @param instance The instance number of the current event and this value is
/// guaranteed to be static for the duration of the callback handler.
/// @param per_instance_user_data This is the field where each tool can send in
/// some state information and the handshake of the type of this data type must
/// be handled by extending tracepoint types that handle diffent types of user
/// data. If the trace type is function_begin/function_end, then the parent and
/// object parameters can be null, but the per_instance_user_data must contain
/// information about the function being traced (preferably the function name).
/// @return The result code which can be one of:
///            1. XPTI_RESULT_SUCCESS when the notification is successful
///            2. XPTI_RESULT_FALSE when tracing is turned off
///            3. XPTI_RESULT_INVALIDARG when one or more input parameters are
///            invalid. For example, for all trace types except function_begin
///            and function_end, the event 'object' cannot be NULL. If a NULL
///            value is provided for this parameter, you will see an
///            XPTI_RESULT_INVALIDARG return value. Similarly, for
///            function_begin and function_end, the per_instance_user_data value
///            must be populated to not get this return value.
XPTI_EXPORT_API xpti::result_t
xptiNotifySubscribers(uint8_t stream_id, uint16_t trace_type,
                      xpti::trace_event_data_t *parent,
                      xpti::trace_event_data_t *object, uint64_t instance,
                      const void *per_instance_user_data);

/// @brief Associates <key-value> pairs with an event
/// @details If the instrumentation embedded in applications need to send
/// additional metadata to the framwork and eventually the subscribers, this
/// function can be used. The metadata is of the form of <key-value> pairs and
/// are only of string types. Internall, the data is represented as <key-value>
/// pairs of string IDs, so when one queries the metadata, they must look up the
/// value's string ID.
///
/// @param e The event for which the metadata is being added
/// @param key The key that identifies the metadata as a string
/// @param value_id The value for the key as an ID of a registered object.
/// @return The result code which can be one of:
///            1. XPTI_RESULT_SUCCESS when the add is successful
///            2. XPTI_RESULT_INVALIDARG when the inputs are invalid
///            3. XPTI_RESULT_DUPLICATE when the key-value pair already exists
XPTI_EXPORT_API xpti::result_t xptiAddMetadata(xpti::trace_event_data_t *e,
                                               const char *key,
                                               xpti::object_id_t value_id);

/// @brief Query the metadata table for a given event
/// @details In order to retrieve metadata information for a given event, you
/// must get the metadata tables and perform your queries on this table.
/// @code
///    auto m = xptiQueryMetadata(event);
///    // Example of printing all the metadata contents
///    for( auto &md : m ) {
///        printf("++ %20s:%s\n", xptiLookupString(md.first),
///                               xptiLookupString(md.second));
///    }
///    // Here's an example of a query on the table
///    char *table_string;
///    xpti::string_id_t key_id = xptiRegisterString("myKey", &table_string);
///    auto index = m.find(key_id);
///    if(index != m.end()) {
///       // Retrieve the value
///       const char *value = xptiLookupString((*index).second);
///    }
/// @endcode
///
/// @param e The event for which the metadata is being requested
/// @return The metadata table of type xpti::metadata_t *
XPTI_EXPORT_API xpti::metadata_t *
xptiQueryMetadata(xpti::trace_event_data_t *e);

/// @brief Returns a bool that indicates whether tracing is enabled or not
/// @details If the tracing is enabled by the XPTI_TRACE_ENABLE=1 environment
/// variable, a valid dispatcher for dispatching calls to the framework and if
/// there exists one or more valid subscribers, then this function will return
/// true, else false
/// @return bool that indicates whether it is enabled or not
XPTI_EXPORT_API bool xptiTraceEnabled();

/// @brief Resets internal state
/// @details This method is currently ONLY used by the tests and is NOT
/// recommended for use in the instrumentation of applications or runtimes.
/// The proxy/stub library does not implement this function.
XPTI_EXPORT_API void xptiReset();

/// @brief Force sets internal state to trace enabled
/// @details This method is currently ONLY used by the tests and is NOT
/// recommended for use in the instrumentation of applications or runtimes.
/// The proxy/stub library does not implement this function.
XPTI_EXPORT_API void xptiForceSetTraceEnabled(bool yesOrNo);

typedef xpti::result_t (*xpti_framework_initialize_t)();
typedef xpti::result_t (*xpti_framework_finalize_t)();
typedef xpti::result_t (*xpti_initialize_t)(const char *, uint32_t, uint32_t,
                                            const char *);
typedef void (*xpti_finalize_t)(const char *);
typedef uint64_t (*xpti_get_universal_id_t)();
typedef void (*xpti_set_universal_id_t)(uint64_t uid);
typedef uint64_t (*xpti_get_unique_id_t)();
typedef xpti::string_id_t (*xpti_register_string_t)(const char *, char **);
typedef const char *(*xpti_lookup_string_t)(xpti::string_id_t);
typedef xpti::string_id_t (*xpti_register_object_t)(const char *, size_t,
                                                    uint8_t);
typedef xpti::object_data_t (*xpti_lookup_object_t)(xpti::object_id_t);
typedef uint64_t (*xpti_register_payload_t)(xpti::payload_t *);
typedef uint8_t (*xpti_register_stream_t)(const char *);
typedef xpti::result_t (*xpti_unregister_stream_t)(const char *);
typedef uint16_t (*xpti_register_user_defined_tp_t)(const char *, uint8_t);
typedef uint16_t (*xpti_register_user_defined_et_t)(const char *, uint8_t);
typedef xpti::trace_event_data_t *(*xpti_make_event_t)(
    const char *, xpti::payload_t *, uint16_t, xpti::trace_activity_type_t,
    uint64_t *);
typedef const xpti::trace_event_data_t *(*xpti_find_event_t)(int64_t);
typedef const xpti::payload_t *(*xpti_query_payload_t)(
    xpti::trace_event_data_t *);
typedef const xpti::payload_t *(*xpti_query_payload_by_uid_t)(uint64_t uid);
typedef xpti::result_t (*xpti_register_cb_t)(uint8_t, uint16_t,
                                             xpti::tracepoint_callback_api_t);
typedef xpti::result_t (*xpti_unregister_cb_t)(uint8_t, uint16_t,
                                               xpti::tracepoint_callback_api_t);
typedef xpti::result_t (*xpti_notify_subscribers_t)(
    uint8_t, uint16_t, xpti::trace_event_data_t *, xpti::trace_event_data_t *,
    uint64_t instance, const void *temporal_user_data);
typedef xpti::result_t (*xpti_add_metadata_t)(xpti::trace_event_data_t *,
                                              const char *, xpti::object_id_t);
typedef xpti::metadata_t *(*xpti_query_metadata_t)(xpti::trace_event_data_t *);
typedef bool (*xpti_trace_enabled_t)();
}
