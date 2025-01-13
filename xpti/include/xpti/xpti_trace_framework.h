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

/// @brief Extracts the user-defined ID from a given value.
///
/// This macro is designed to extract the lowest 7 bits from the provided value,
/// which are reserved for a user-defined ID. The user-defined ID is expected to
/// be encoded in these bits for easy extraction.
///
/// @param val The input value from which the user-defined ID is to be
///            extracted. It is cast to `uint16_t` before extraction.
///
/// @return The extracted user-defined ID as a `uint16_t`.

#define XPTI_EXTRACT_USER_DEFINED_ID(val) ((uint16_t)val & 0x007f)

/// @brief Extracts the tool ID from a given value.
///
/// This macro extracts 8 bits starting from the 9th bit position of the
/// provided value, which are reserved for the tool ID. The tool ID is expected
/// to be encoded in these bits, allowing for identification of the tool or
/// component that generated the value.
///
/// @param val The input value from which the tool ID is to be extracted.
///            It is cast to `uint16_t` before extraction.
///
/// @return The extracted tool ID as a `uint16_t`.

#define XPTI_TOOL_ID(val) (((uint16_t)val >> 8) & 0x00ff)

extern "C" {

/// @struct xpti_payload_t
/// @brief Defines the interface for payload information in the tracing
/// framework.
///
/// This structure provides a virtual interface for accessing payload
/// information associated with trace points in the tracing framework. Payloads
/// typically contain metadata about the code region being traced, such as
/// function names, source file locations, and unique identifiers.
/// Implementations of this interface are expected to provide concrete details
/// for these metadata elements.
struct xpti_payload_t {
  /// @brief Destructor.
  virtual ~xpti_payload_t() {}

  /// @brief Gets the name associated with the payload.
  /// @return A pointer to a null-terminated string containing the name.
  virtual const char *name() const = 0;

  /// @brief Gets the source file path associated with the payload.
  /// @return A pointer to a null-terminated string containing the source file
  /// path.
  virtual const char *source_file() const = 0;

  /// @brief Gets the line number in the source file associated with the
  /// payload.
  /// @return The line number.
  virtual uint32_t line_no() const = 0;

  /// @brief Gets the column number in the source file associated with the
  /// payload.
  /// @return The column number.
  virtual uint32_t column_no() const = 0;

  /// @brief Gets the flags associated with the payload.
  /// @return A 64-bit integer representing the payload flags.
  virtual uint64_t payload_flags() const = 0;

  /// @brief Gets the string ID for the name.
  /// @return An integer representing the string ID for the name.
  virtual int32_t name_string_id() const = 0;

  /// @brief Gets the string ID for the source file path.
  /// @return An integer representing the string ID for the source file path.
  virtual int32_t file_string_id() const = 0;

  /// @brief Gets a unique identifier for the payload.
  /// @return A 64-bit integer representing the unique identifier.
  virtual uint64_t uid64() const = 0;

  /// @brief Checks if the payload is valid.
  /// @return True if the payload is valid, false otherwise.
  virtual bool is_valid() const = 0;

  /// @brief Gets a reference to the payload object.
  /// @return A pointer to the payload object.
  virtual xpti::payload_t *payload_ref() = 0;
};

/// @struct xpti_metadata_t
/// @brief Defines the interface for managing metadata in the tracing framework.
///
/// This structure provides a virtual interface for managing key-value pair
/// metadata associated with trace points or other objects in the tracing
/// framework. The metadata is used to store additional information that can be
/// queried and used by tools for analysis or visualization purposes.
/// Implementations of this interface are expected to provide concrete
/// mechanisms for storing, retrieving, and iterating over metadata items.

struct xpti_metadata_t {
  /// @brief Destructor.
  ///
  /// Virtual destructor to ensure derived classes are correctly cleaned up.
  virtual ~xpti_metadata_t() {}

  /// @brief Retrieves the first metadata item.
  ///
  /// Sets the provided key and value to the first metadata item's key and
  /// value. This function is typically used to start iterating over all
  /// metadata items.
  ///
  /// @param[out] key Pointer to a character pointer that will be set to the key
  ///                 of the first item.
  /// @param[out] value Reference to an object_id_t that will be set to the
  ///                 value of the first item.
  /// @return Result of the operation, indicating success or failure. Returns
  ///                 xpti::result_t::XPTI_RESULT_FALSE if the metadata is
  ///                 empty.
  virtual xpti::result_t first_item(char **key, xpti::object_id_t &value) = 0;

  /// @brief Retrieves the next metadata item.
  ///
  /// After calling first_item, this function can be called repeatedly to
  /// iterate through all metadata items. Each call sets the provided key and
  /// value to the next item's key and value.
  ///
  /// @param[out] key Pointer to a character pointer that will be set to the key
  /// of the next item.
  /// @param[out] value Reference to an object_id_t that will be set to the
  /// value of the next item.
  /// @return Result of the operation, indicating success or failure. Returns
  ///                 xpti::result_t::XPTI_RESULT_FALSE if the metadata is
  ///                 at the end. Returns xpti::result_t::XPTI_RESULT_SUCCESS
  ///                 if the next item is successfully retrieved.
  virtual xpti::result_t next_item(char **key, xpti::object_id_t &value) = 0;

  ///@brief Adds a metadata item.
  ///
  /// Adds a new key-value pair to the metadata. If the key already exists, its
  /// value is updated with the new value provided.
  ///
  ///@param[in] key Pointer to a null-terminated string representing the key.
  ///@param[in] value The value associated with the key.
  ///@return Result of the operation, indicating success or failure. Returns
  ///        xpti::result_t::XPTI_RESULT_SUCCESS if the item is successfully
  ///        added.
  virtual xpti::result_t add_item(const char *key, xpti::object_id_t value) = 0;

  /// @brief Counts the number of metadata items.
  ///
  /// Returns the total number of key-value pairs stored in the metadata.
  ///
  /// @return The number of metadata items.
  virtual size_t count() = 0;
};

/// @struct xpti_trace_event_t
/// @brief Represents a trace event in the instrumentation framework.
///
/// This structure defines the interface for a trace event, which is a key
/// component in the tracing system. Trace events are generated at various
/// points in the execution of a program to collect performance data or other
/// forms of diagnostic information.

struct xpti_trace_event_t {

  /// @brief Destructor.
  ///
  /// Virtual destructor to ensure proper cleanup of derived classes.
  virtual ~xpti_trace_event_t() {}

  /// @brief Gets the payload associated with the trace event.
  /// @return A pointer to the xpti_payload_t structure containing the payload.
  virtual xpti_payload_t *payload() = 0;

  /// @brief Gets the metadata associated with the trace event.
  /// @return A pointer to the xpti_metadata_t structure containing the
  /// metadata.
  virtual xpti_metadata_t *metadata() = 0;

  /// @brief Gets the unique identifier for the trace event.
  /// @return A 64-bit unsigned integer representing the unique identifier.
  virtual uint64_t uid64() const = 0;

  /// @brief Gets the instance number of the trace event.
  /// @return A 64-bit unsigned integer representing the instance number.
  virtual uint64_t instance() const = 0;

  /// @brief Gets the event type.
  /// @return A 16-bit unsigned integer representing the event type.
  virtual uint16_t event_type() const = 0;

  /// @brief Gets the activity type.
  /// @return A 16-bit unsigned integer representing the activity type.
  virtual uint16_t activity_type() const = 0;

  /// @brief Gets the unique identifier of the source of the event.
  /// @return A 64-bit unsigned integer representing the source's unique
  /// identifier. Used only for "edge" events that represent a relationship
  virtual uint64_t source_uid64() const = 0;

  /// @brief Gets the unique identifier of the target of the event.
  /// @return A 64-bit unsigned integer representing the target's unique
  /// identifier.
  virtual uint64_t target_uid64() const = 0;

  /// @brief Gets the flags associated with the event.
  /// @return A 64-bit unsigned integer representing the event flags.
  virtual uint64_t event_flags() const = 0;

  /// @brief Sets the activity type of the event.
  /// @param type The activity type to set.
  virtual void set_activity_type(xpti::trace_activity_type_t type) = 0;

  /// @brief Sets the event type.
  /// @param type The event type to set.
  virtual void set_event_type(uint16_t type) = 0;

  /// @brief Gets a reference to the payload object.
  /// @return A pointer to the payload object.
  virtual xpti::payload_t *payload_ref() = 0;

  /// @brief Gets a reference to the trace event data.
  /// @return A pointer to the xpti::trace_event_data_t structure containing the
  /// trace event data.
  virtual xpti::trace_event_data_t *event_ref() = 0;
};

/// @struct xpti_tracepoint_t
/// @brief Represents a trace point in the tracing framework.
///
/// A trace point is a specific location in the execution of a program where
/// information can be collected for tracing purposes. This structure provides
/// an interface for accessing the payload, metadata, and event associated with
/// a trace point, as well as unique identifiers for the trace point and its
/// instance.

struct xpti_tracepoint_t {

  /// @brief Destructor.
  ///
  /// Virtual destructor to ensure proper cleanup of derived classes.
  virtual ~xpti_tracepoint_t() {}

  /// @brief Gets the payload associated with the trace point.
  /// @return A pointer to the xpti_payload_t structure containing the payload.
  virtual xpti_payload_t *payload() = 0;

  /// @brief Gets the metadata associated with the trace point.
  /// @return A pointer to the xpti_metadata_t structure containing the
  /// metadata.
  virtual xpti_metadata_t *metadata() = 0;

  /// @brief Gets the event associated with the trace point.
  /// @return A pointer to the xpti_trace_event_t structure containing the
  /// event.
  virtual xpti_trace_event_t *event() = 0;

  /// @brief Gets the unique identifier for the trace point.
  /// @return A 64-bit unsigned integer representing the unique identifier.
  virtual uint64_t uid64() const = 0;

  /// @brief Gets the instance number of the trace point.
  /// @return A 64-bit unsigned integer representing the instance number.
  virtual uint64_t instance() const = 0;

  /// @brief Gets a reference to the trace event data.
  /// @return A pointer to the xpti::trace_event_data_t structure containing the
  /// trace event data.
  virtual xpti::trace_event_data_t *event_ref() = 0;

  /// @brief Gets a reference to the payload data.
  /// @return A pointer to the xpti::payload_t structure containing the payload
  /// data.
  virtual xpti::payload_t *payload_ref() = 0;
};

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

/// @brief Returns stashed tuple<std::string, uint64_t>
/// @details The XPTI Framework allows the notification mechanism to stash a
/// key-value tupe before a notification that can be accessed in the callback
/// handler fo the notification. This value is guranteed to be valid for the
/// duration of the notifiation.
/// @param key The Key of the stashed tuple is contained in this parameter after
/// the call
/// @param value The value that corresponds to key
/// @return The result code is XPTI_RESULT_SUCCESS when successful and
/// XPTI_RESULT_NOTFOUND if there is nothing stashed. Also returns error if
/// 'key' argument is invalid (XPTI_RESULT_INVALIDARG)
XPTI_EXPORT_API xpti::result_t xptiGetStashedTuple(char **key, uint64_t &value);

/// @brief Stash a key-value tuple
/// @details Certain notifications in XPTI may want to provide mutable values
/// associated with Universal IDs that can be captured in the notification
/// handler. The framework currently allows one such tuple to be provided and
/// stashed.
/// @param key The Key of the tuple that is being stashed and needs to be
/// available for the duration of the notification call.
/// @param value The value that corresponds to key
/// @return The result code is XPTI_RESULT_SUCCESS when successful and
/// XPTI_RESULT_FAIL if key is invalid
XPTI_EXPORT_API xpti::result_t xptiStashTuple(const char *key, uint64_t value);

/// @brief Un-Stash a key-value tuple or pop it from a stack, if one exists
/// @details Certain notifications in XPTI may want to provide mutable values
/// associated with Universal IDs that can be captured in the notification
/// handler. The framework currently allows such values to be provided and
/// stashed. This function pops the top of the stack tuple value when it is no
/// longer needed; Currently a stack depth of 1 is supported.
/// @return The result code is XPTI_RESULT_SUCCESS when successful and
/// XPTI_RESULT_FAIL if there are no tuples present
XPTI_EXPORT_API void xptiUnstashTuple();

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
XPTI_EXPORT_API
xpti::trace_event_data_t *
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

/// @brief Looks up the payload associated with a given 128-bit unique
/// identifier (UID).
///
/// This function searches for and retrieves the payload information associated
/// with a 64-bit universal ID. The payload contains metadata such as the
/// source file name, function name, and line number from which the UID was
/// generated. This function is typically used to retrieve contextual
/// information for tracing or profiling purposes, allowing for a more detailed
/// analysis of performance data.
///
/// @param uid A reference to the unique identifier (uid128_t) for which the
///            payload is being requested. The UID is expected to have been
///            previously generated and registered within the system.
/// @return   A pointer to the `xpti::payload_t` structure containing the
///           payload information. If the UID does not have an associated
///           payload, a nullptr is returned.
///
XPTI_EXPORT_API const xpti_payload_t *xptiLookupPayload(uint64_t uid);

/// @brief Retrieves the trace event data associated with a given unique
/// identifier (UID).
///
/// This function is designed to search for and return the trace event data
/// corresponding to a specific UID. Trace event data includes information
/// necessary for tracing and profiling, such as event names, types, and other
/// metadata. This function is crucial for correlating trace events with their
/// unique identifiers, enabling detailed analysis and debugging of performance
/// issues.
///
/// @param uid A reference to the 64-bit unique identifier for which the
///            trace event data is being requested. The UID should have been
///            previously generated and associated with a specific trace event.
/// @return    A pointer to the `xpti::trace_event_data_t` structure containing
///            the trace event data. If the UID does not have an associated
///            trace event, a nullptr is returned. This allows for easy checking
///            of whether a given UID corresponds to a valid trace event.
///
XPTI_EXPORT_API const xpti_trace_event_t *xptiLookupEvent(uint64_t uid);

/// @brief Creates a tracepoint in the XPTI framework.
///
/// This function is used to create a tracepoint in the XPTI framework. A
/// tracepoint is a specific location in the code, identified by function name,
/// file name, line number, and column number, where trace data can be generated
/// and collected. This function allocates and initializes a tracepoint object,
/// which can then be used to emit trace events at runtime. This tracepoint
/// interface object packages a payload, a trace event data structure and a
/// 128-bit universal ID and provides an interface to retrieve all of this
/// information. The interface pointer uniquely identifies a specific instance
/// of a tracepoint.
///
/// @param func_name A constant character pointer representing the name of the
/// function where the tracepoint is located. This provides context for the
/// tracepoint, aiding in identification and analysis of trace data.
/// @param file_name A constant character pointer representing the name of the
/// source file where the tracepoint is located. This helps in pinpointing the
/// exact location of the tracepoint in the codebase.
/// @param line_no A uint32_t value representing the line number in the source
/// file where the tracepoint is located. This further refines the location of
/// the tracepoint within the file.
/// @param column_no A uint32_t value representing the column number on the
/// specified line where the tracepoint is located. This provides the most
/// precise location of the tracepoint.
///
/// @return A pointer to the created `xpti_tracepoint_t` structure, which
/// represents the tracepoint. If the tracepoint cannot be created, the function
/// returns nullptr.
///
/// @note In order to preserve ABI compatibility, an interface pointer to
/// `xpti_tracepoint_t` is returned.

XPTI_EXPORT_API xpti_tracepoint_t *xptiCreateTracepoint(const char *func_name,
                                                        const char *file_name,
                                                        uint32_t line_no,
                                                        uint32_t column_no);

/// @brief Deletes a tracepoint object.
///
/// This function is responsible for safely deleting a tracepoint object created
/// by the XPTI framework. It ensures that any resources associated with the
/// tracepoint, such as memory allocations for payloads or metadata, are
/// properly released. This function is typically called when a tracepoint is no
/// longer needed, such as at the end of its scope or during application
/// shutdown, to prevent memory leaks.
///
/// @param tp A pointer to the `xpti_tracepoint_t` structure representing the
/// tracepoint to be deleted. If the pointer is null, the function has no
/// effect.
///
/// @return A result code of type `xpti::result_t` indicating the success or
/// failure of the deletion process. Possible return values include
/// `XPTI_RESULT_SUCCESS` if the tracepoint was successfully deleted, or an
/// error code indicating the reason for failure.
///
/// @note It is important to ensure that the tracepoint object is not accessed
/// after calling this function, as it will result in undefined behavior.

XPTI_EXPORT_API xpti::result_t xptiDeleteTracepoint(xpti_tracepoint_t *tp);

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

/// @brief Check if tracing is enabled for a stream or stream and trace type
/// @details If the tracing is enabled by the XPTI_TRACE_ENABLE=1
/// environment variable, a valid dispatcher is available for dispatching
/// calls to the framework and if there exists one or more valid subscribers
/// subscribing to a stream or a stream and trace type in that stream, then
/// this method will return TRUE
/// @param stream Stream ID
/// @param ttype The trace type within the stream
/// @return bool that indicates whether it is enabled or not
XPTI_EXPORT_API bool xptiCheckTraceEnabled(uint16_t stream, uint16_t ttype = 0);

/// @brief Resets internal state
/// @details This method is currently ONLY used by the tests and is NOT
/// recommended for use in the instrumentation of applications or runtimes.
/// The proxy/stub library does not implement this function.
XPTI_EXPORT_API void xptiReset();

/// @brief Force sets internal state to trace enabled
/// @details This method is currently ONLY used by the tests and is NOT
/// recommended for use in the instrumentation of applications or runtimes.
XPTI_EXPORT_API void xptiForceSetTraceEnabled(bool yesOrNo);

/// @brief Requery check of environment variables to set trace enabled in
/// runtime
/// @details This method is currently ONLY used by the tests and is NOT
/// recommended for use in the instrumentation of applications or runtimes.
/// The framework does not implement this function, only proxy library.
XPTI_EXPORT_API void xptiTraceTryToEnable();

/// @brief Retrieves the trace point scope data.
/// @details This function is used to get the trace point scope data that is
/// currently set in the tracing framework's thread-local storage.
///
/// @return The trace point interface pointer.
XPTI_EXPORT_API const xpti_tracepoint_t *xptiGetTracepointScopeData();

/// @brief Sets the trace point scope data.
/// @details This function is used to set the trace point scope data in the
/// tracing framework.
///
/// @param tp The trace point interface pointer.
/// @return Result of the operation, success or failure.
XPTI_EXPORT_API xpti::result_t
xptiSetTracepointScopeData(xpti_tracepoint_t *tp);

/// @brief Unsets the trace point scope data.
/// @details This function is used to unset the trace point scope data in the
/// tracing framework.
XPTI_EXPORT_API void xptiUnsetTracepointScopeData();

/// @brief Registers a trace point scope with the tracing framework.
///
/// This function is used to register a new trace point scope based on the
/// function name, file name, line number, and column number where the trace
/// point is defined. It is typically called at the entry of a function or a
/// specific scope within a function to mark the beginning of a traceable
/// region. The function returns a pointer to a `xpti_tracepoint_t` structure
/// that contains information about the registered trace point, including a
/// unique identifier that can be used to reference the trace point in
/// subsequent tracing calls.
///
/// @param funcName The name of the function or scope being registered. This
///                 parameter should not be NULL.
/// @param fileName The name of the source file where the trace point is
///                 defined. This parameter is optional and can be nullptr
/// @param lineNo   The line number in the source file where the trace point is
///                 defined. This parameter is optional and can be set to 0.
/// @param columnNo The column number in the source file where the trace point
///                 is defined. If column information is not available, this can
///                 be set to 0.
/// @return Returns a pointer to the registered `xpti_tracepoint_t` structure if
///                 the registration is successful; otherwise, returns NULL. The
///                 returned pointer should not be freed by the caller.
XPTI_EXPORT_API const xpti_tracepoint_t *
xptiRegisterTracepointScope(const char *funcName, const char *fileName,
                            uint32_t lineNo, uint32_t columnNo);

/// @brief Retrieves the default stream ID.
/// @details This function is used to get the default stream ID that is
/// currently set in the tracing framework.
/// @return The default stream ID.
XPTI_EXPORT_API uint8_t xptiGetDefaultStreamID();

/// @brief Sets the default stream ID.
/// @details This function is used to set the default stream ID in the tracing
/// framework. All MACROs and other scoped notification objects will use the
/// default stream to send the event data
///
/// @param defaultStreamID The stream ID to be set as default.
/// @return Result of the operation, success or failure.
XPTI_EXPORT_API xpti::result_t xptiSetDefaultStreamID(uint8_t defaultStreamID);

/// @brief Retrieves the default event type.
/// @details This function is used to get the default event type that is
/// currently set in the tracing framework. This is typically set to 'algorithm'
/// as most default events are algorithmic events trying to capture a task,
/// barrier/lock or function call.
///
/// @return The default event type.
XPTI_EXPORT_API xpti::trace_event_type_t xptiGetDefaultEventType();

/// @brief Sets the default event type.
/// @details This function is used to set the default event type in the tracing
/// framework.
///
/// @param defaultEventType The event type to be set as default.
/// @return Result of the operation, success or failure.
XPTI_EXPORT_API xpti::result_t
xptiSetDefaultEventType(xpti::trace_event_type_t defaultEventType);

/// @brief Retrieves the default trace point type.
/// @details This function is used to get the default trace point type that is
/// currently set in the tracing framework. This is typically set to
/// 'function_begin'
///
/// @return The default trace point type.
XPTI_EXPORT_API xpti::trace_point_type_t xptiGetDefaultTraceType();

/// @brief Sets the default trace point type.
/// @details This function is used to set the default trace point type in the
/// tracing framework.
///
/// @param defaultTraceType The trace point type to be set as default.
/// @return Result of the operation, success or failure.
XPTI_EXPORT_API xpti::result_t
xptiSetDefaultTraceType(xpti::trace_point_type_t defaultTraceType);

/// @brief Enables the trace point scope object to self notify.
/// @details This function is used to enable the tracepoint_scope_t object to
/// self notify at all tracepoints
/// @param yesOrNo The flag used to enable or disable a tracepoint scope for
/// notification.
XPTI_EXPORT_API void xptiEnableTracepointScopeNotification(bool yesOrNo);

/// @brief Checks if tracepoint scope notifications are enabled.
///
/// This function checks the global flag that controls whether tracepoints
/// should notify themselves when hit. If the flag is set to true, tracepoints
/// will self-notify, which can be useful for debugging or generating more
/// detailed trace information.
/// @return Returns true if tracepoint scope notifications are enabled, and
/// false otherwise.
XPTI_EXPORT_API bool xptiCheckTracepointScopeNotification();

/// @brief Removes cached event and associated metadata
/// @param e The event for which associated data will be removed
XPTI_EXPORT_API void xptiReleaseEvent(xpti::trace_event_data_t *e);

typedef xpti::result_t (*xpti_framework_initialize_t)();
typedef xpti::result_t (*xpti_framework_finalize_t)();
typedef xpti::result_t (*xpti_initialize_t)(const char *, uint32_t, uint32_t,
                                            const char *);
typedef void (*xpti_finalize_t)(const char *);
typedef uint64_t (*xpti_get_universal_id_t)();
typedef void (*xpti_set_universal_id_t)(uint64_t uid);
typedef uint64_t (*xpti_get_unique_id_t)();
typedef xpti::result_t (*xpti_stash_tuple_t)(const char *key, uint64_t value);
typedef xpti::result_t (*xpti_get_stashed_tuple_t)(char **key, uint64_t &value);
typedef void (*xpti_unstash_tuple_t)();
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
typedef bool (*xpti_check_trace_enabled_t)(uint16_t stream, uint16_t ttype);
typedef void (*xpti_force_set_trace_enabled_t)(bool);
typedef void (*xpti_release_event_t)(xpti::trace_event_data_t *);
typedef void (*xpti_enable_tracepoint_scope_notification_t)(bool);
typedef bool (*xpti_check_tracepoint_scope_notification_t)();
typedef xpti::result_t (*xpti_make_key_from_payload_t)(xpti::payload_t *,
                                                       xpti::uid128_t *);
typedef const xpti_tracepoint_t *(*xpti_get_trace_point_scope_data_t)();
typedef const xpti_tracepoint_t *(*xpti_register_tracepoint_scope_t)(
    const char *func, const char *file, uint32_t line, uint32_t col);
typedef xpti::result_t (*xpti_set_trace_point_scope_data_t)(
    xpti_tracepoint_t *);
typedef void (*xpti_unset_trace_point_scope_data_t)();
typedef uint8_t (*xpti_get_default_stream_id_t)();
typedef xpti::result_t (*xpti_set_default_stream_id_t)(uint8_t);
typedef xpti::trace_event_type_t (*xpti_get_default_event_type_t)();
typedef xpti::result_t (*xpti_set_default_event_type_t)(
    xpti::trace_event_type_t);
typedef xpti::trace_point_type_t (*xpti_get_default_trace_type_t)();
typedef xpti::result_t (*xpti_set_default_trace_type_t)(
    xpti::trace_point_type_t);

typedef const xpti_payload_t *(*xpti_lookup_payload_t)(uint64_t);
typedef const xpti_trace_event_t *(*xpti_lookup_event_t)(uint64_t);
typedef xpti_tracepoint_t *(*xpti_create_tracepoint_t)(const char *,
                                                       const char *, uint32_t,
                                                       uint32_t);
typedef xpti::result_t (*xpti_delete_tracepoint_t)(xpti_tracepoint_t *);
}
