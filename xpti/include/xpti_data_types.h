//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
#pragma once
#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>

namespace xpti {
/// @brief Universal ID data structure that is central to XPTI
/// @details A given trace point is referred to by it its universal ID and this
/// data structure has all the elements that are necessary to map to the code
/// location of the trace point. In the case the end-user opts out of embedding
/// the code location information in the trace point, other pieces of
/// information are leveraged to generate a unique 64-bit ID.
struct uid_t {
  /// Contains string ID for file name in upper 32-bits and the line number in
  /// lower 32-bits
  uint64_t p1 = 0;
  /// Contains the string ID for kernel name in lower 32-bits; in the case
  /// dynamic stack walk is performed, the upper 32-bits contain the string ID
  /// of the caller->callee combination string.
  uint64_t p2 = 0;
  /// Contains the address of the kernel object or SYCL object references and
  /// only the lower 32-bits will be used to generate the hash
  uint64_t p3 = 0;

  uid_t() = default;
  /// Computes a hash that is a bijection between N^3 and N
  /// (x,y,z) |-> (x) + (x+y+1)/2 + (x+y+z+2)/3
  uint64_t hash() const {
    /// Use lower 32-bits of the address
    uint32_t v3 = (uint32_t)(p3 & 0x00000000ffffffff);
    /// Use p1 and p2 as is; since p1 and p2 upper 32-bits is built from string
    /// IDs, they are more than likely to be less than 16 bits. Combining
    /// 48-bits of one value with ~16-bits/~32-bits will should not overflow a
    /// 64-bit accumulator.
    return (p1 + (p1 + p2 + 1) / 2 + (p1 + p2 + v3 + 2) / 3);
  }

  bool operator<(const uid_t &rhs) const {
    if (p1 < rhs.p1)
      return true;
    if (p1 == rhs.p1 && p2 < rhs.p2)
      return true;
    if (p1 == rhs.p1 && p2 == rhs.p2 && p3 < rhs.p3)
      return true;
    return false;
  }

  bool operator==(const uid_t &rhs) const {
    return p1 == rhs.p1 && p2 == rhs.p2 && p3 == rhs.p3;
  }
};
} // namespace xpti

/// Specialize std::hash to support xpti::uid_t
namespace std {
template <> struct hash<xpti::uid_t> {
  std::size_t operator()(const xpti::uid_t &key) const { return key.hash(); }
};
} // namespace std

namespace xpti {
constexpr int invalid_id = -1;
constexpr int invalid_uid = 0;
constexpr uint8_t default_vendor = 0;

/// @brief Flag values used by the payload_t structure to mark the information
/// present
/// @details When a payload is created, it is conceivable that only partial
/// information may be present and these flags are used to indicate the
/// available information. The hash generator will generate a hash based on
/// the flags set.
///
enum class payload_flag_t {
  /// The name for the tracepoint is available
  NameAvailable = 1,
  /// Source file information available
  SourceFileAvailable = 1 << 1,
  /// Code pointer VA is available
  CodePointerAvailable = 1 << 2,
  /// Line information available in the payload
  LineInfoAvailable = 1 << 3,
  /// Column information available in payload
  ColumnInfoAvailable = 1 << 4,
  /// Caller/Callee stack trace available when source/kernel info not available
  StackTraceAvailable = 1 << 5,
  // A 64-bit hash is already available for this payload
  HashAvailable = 2 << 16
};

//
//  Helper macros for creating new tracepoint and
//  event types
//
using trace_point_t = uint16_t;
using event_type_t = uint16_t;
using string_id_t = int32_t;

using safe_flag_t = std::atomic<bool>;
using safe_uint64_t = std::atomic<uint64_t>;
using safe_uint32_t = std::atomic<uint32_t>;
using safe_uint16_t = std::atomic<uint16_t>;
using safe_int64_t = std::atomic<int64_t>;
using safe_int32_t = std::atomic<int32_t>;
using safe_int16_t = std::atomic<int16_t>;
using metadata_t = std::unordered_map<string_id_t, string_id_t>;

#define XPTI_EVENT(val) xpti::event_type_t(val)
#define XPTI_TRACE_POINT_BEGIN(val) xpti::trace_point_t(val << 1 | 0)
#define XPTI_TRACE_POINT_END(val) xpti::trace_point_t(val << 1 | 1)

#define XPTI_PACK08_RET16(value1, value2) ((value1 << 8) | value2)
#define XPTI_PACK16_RET32(value1, value2) ((value1 << 16) | value2)
#define XPTI_PACK32_RET64(value1, value2) (((uint64_t)value1 << 32) | value2)

/// @brief Payload data structure that is optional for trace point callback
/// API
/// @details The payload structure, if determined at compile time, can deliver
/// the source association of various parallel constructs defined by the
/// language. In the case it is defined, a lookup table will provide the
/// association from a kernel/lambda (address) to a payload and the same
/// address to a unique ID created at runtime.
///
/// All instances of a kernel will be associated with the same unique ID
/// through the lifetime of an object. The hash maps that will be maintained
/// would be: # [unique_id]->[payload] # [kernel address]->[unique_id]
///
/// Unique_id MUST be propagated downstream to the OpenCL runtime to ensure
/// the associations back to the sources. This requires elp from the compiler
/// front-end.
///
struct payload_t {
  /// Name of the trace point; graph, algorithm, lock names, for example.
  const char *name = nullptr;
  /// Stack trace indicated by caller/callee as "caller->callee"
  const char *stack_trace = nullptr;
  /// Absolute path of the source file; may have to to be unicode string
  const char *source_file = nullptr;
  /// Line number information to correlate the trace point
  uint32_t line_no = invalid_id;
  /// For a complex statement, column number may be needed to resolve the
  /// trace point; currently none of the compiler builtins return a valid
  /// column no
  uint32_t column_no = invalid_id;
  /// Kernel/lambda/function address
  const void *code_ptr_va = nullptr;
  /// Internal bookkeeping slot - do not change.
  uint64_t internal;
  /// Flags indicating whether string name, codepointer, source file and hash
  /// values are available
  uint64_t flags = 0;
  /// Universal ID associated with this payload
  uid_t uid;

  payload_t() = default;

  //  If the address of the kernel/function name is provided, we mark it as
  //  valid since we can potentially reconstruct the name and the source file
  //  information during post-processing step of symbol resolution; this
  //  indicates a partial but valid payload.
  payload_t(void *codeptr) {
    code_ptr_va = codeptr;
    name = nullptr;         ///< Invalid name string pointer
    source_file = nullptr;  ///< Invalid source file string pointer
    line_no = invalid_id;   ///< Invalid line number
    column_no = invalid_id; ///< Invalid column number
    flags = (uint64_t)payload_flag_t::CodePointerAvailable;
  }

  //  If neither an address or the fully identifyable source file name and
  //  location are not available, we take in the name of the
  //  function/task/user-defined name as input and create a hash from it. We
  //  mark it as valid since we can display the name in a timeline view, but
  //  the payload is considered to be a partial but valid payload.
  payload_t(const char *func_name) {
    code_ptr_va = nullptr;
    name = func_name;      ///< Invalid name string pointer
    source_file = nullptr; ///< Invalid source file string pointer
    flags = (uint64_t)(payload_flag_t::NameAvailable);
  }

  payload_t(const char *func_name, void *codeptr) {
    code_ptr_va = codeptr;
    name = func_name;      ///< Invalid name string pointer
    source_file = nullptr; ///< Invalid source file string pointer
    flags = (uint64_t)payload_flag_t::NameAvailable |
            (uint64_t)payload_flag_t::CodePointerAvailable;
  }

  //  When the end user opts out of preserving the code location information and
  //  the KernelInfo is not available from the given entry point, we will rely
  //  on dynamic backtrace as a possibility. In this case, we send in the
  //  caller/callee information as a string in the form "caller->callee" that
  //  will be used to generate the unique ID.
  payload_t(const char *kname, const char *caller_callee, void *codeptr) {
    if (codeptr) {
      code_ptr_va = codeptr;
      flags |= (uint64_t)payload_flag_t::CodePointerAvailable;
    }
    /// Capture the rest of the parameters
    if (kname) {
      name = kname;
      flags |= (uint64_t)payload_flag_t::NameAvailable;
    }
    if (caller_callee) {
      stack_trace = caller_callee;
      flags |= (uint64_t)payload_flag_t::StackTraceAvailable;
    }
  }

  //  We need the payload to contain at the very least, the code pointer
  //  information of the kernel or function. In the full payload case, we will
  //  also have the function name and source file name along with the line and
  //  column number of the trace point that forms the payload.
  payload_t(const char *kname, const char *sf, int line, int col,
            void *codeptr) {
    code_ptr_va = codeptr;
    /// Capture the rest of the parameters
    name = kname;
    source_file = sf;
    line_no = line;
    column_no = col;
    flags = (uint64_t)payload_flag_t::NameAvailable |
            (uint64_t)payload_flag_t::SourceFileAvailable |
            (uint64_t)payload_flag_t::LineInfoAvailable |
            (uint64_t)payload_flag_t::ColumnInfoAvailable |
            (uint64_t)payload_flag_t::CodePointerAvailable;
  }

  int32_t name_sid() const { return (int32_t)(uid.p2 & 0x00000000ffffffff); }
  int32_t stacktrace_sid() const { return (int32_t)(uid.p2 >> 32); }
  int32_t source_file_sid() const { return (int32_t)(uid.p1 >> 32); }
};

/// A data structure that holds information about an API function call and its
/// arguments.
struct function_with_args_t {
  /// A stable API function ID. It is a contract between the profiled system
  /// and subscribers.
  uint32_t function_id;
  /// A null-terminated string, containing human-readable function name.
  const char *function_name;
  /// Pointer to packed function arguments.
  void *args_data;
  /// Pointer to the return value of the function.
  void *ret_data;
  /// [Provisional] Additional data, generated by the profiled system.
  void *user_data;
};

///  @brief Enumerator defining the global/basic trace point types
///  @details The frame work defines the global/basic trace point types
///  that are necessary for modeling parallel runtimes. A helper macro
///  provided to create the enum values as the LSB is reserved for
///  determining if the trace point is a 'begin' trace or an 'end'
///  trace. This reserved bit is used by the scoped_notify() class
///  to automatically send the closing enum trace type for a given
///  trace point type.
///
///  The provided macros TRACE_POINT_BEGIN(val) and TRACE_POINT_END(val)
///  must be used in all user defined enums that are defined to extend
///  the trace point types.
///
///  The trace_type data is of type uint8_t and the 7-LSB bits are used
///  to enumerate trace types. the MSB bit is reserved for user-defined
///  trace types and is set to 0 for predefined trace point types defined
///  by the framework.
///
///  When user-defined trace types are being declared, a new ID is added
///  to this value to create a uint16_t data type. The LSB 8-bits have
///  the 8th bit set indicating that it is user-defined and the remaining
///  7-bits will indicated the user defined trace point type. However,
///  since multiple tools or vendors could create their own trace point
///  types, we require the vendor_id to create a vendor namespace to avoid
///  collisions.
///
///                                  user-defined bit
///                                    |
///                                    |
///                                    |+-----+---- 127 possible values for
///                                    ||     |     defining trace types.
///                                    ||     |     Due to the scope bit,
///                                    ||     |     63 unique scope types
///                                    ||     |     can be defined.
///                                    vv     v
///   Field width (uint16_t) |........|........|
///                          15      8 7      0
///                           ^      ^
///                           |      |
///                           |      |
///                           +------+----- Reserved for vendor ID
///
enum class trace_point_type_t : uint16_t {
  unknown_type = 0,
  /// Indicates that a graph has been instantiated
  graph_create = XPTI_TRACE_POINT_BEGIN(1),
  /// Indicates that a new node object has been instantiated
  node_create = XPTI_TRACE_POINT_BEGIN(2),
  /// Indicates that a new edge object has been instantiated
  edge_create = XPTI_TRACE_POINT_BEGIN(3),
  /// Indicates the beginning of a parallel region
  region_begin = XPTI_TRACE_POINT_BEGIN(4),
  /// Indicates the end of a parallel region
  region_end = XPTI_TRACE_POINT_END(4),
  /// Indicates the begin of a task execution, the parent of which could be a
  /// graph or a parallel region
  task_begin = XPTI_TRACE_POINT_BEGIN(5),
  /// Indicates the end of an executing task
  task_end = XPTI_TRACE_POINT_END(5),
  /// Indicates the begin of a barrier call
  barrier_begin = XPTI_TRACE_POINT_BEGIN(6),
  /// Indicates the end of a barrier
  barrier_end = XPTI_TRACE_POINT_END(6),
  /// Similar to barrier begin, but captures the information for a lock
  lock_begin = XPTI_TRACE_POINT_BEGIN(7),
  /// Similar to barrier end, but captures the information for a lock
  lock_end = XPTI_TRACE_POINT_END(7),
  /// Use to model triggers (impulse) at various points in time - will not
  /// have an end equivalent
  signal = XPTI_TRACE_POINT_BEGIN(8),
  /// Used to model the data transfer initiation from device A to device B
  transfer_begin = XPTI_TRACE_POINT_BEGIN(9),
  /// Used to model the completion of a previously initiated data transfer
  /// event
  transfer_end = XPTI_TRACE_POINT_END(9),
  /// Is present for completeness to capture the spawning of new threads in a
  /// runtime
  thread_begin = XPTI_TRACE_POINT_BEGIN(10),
  /// Models the end of the lifetime of a thread
  thread_end = XPTI_TRACE_POINT_END(10),
  /// Models the explicit barrier begin in SYCL
  wait_begin = XPTI_TRACE_POINT_BEGIN(11),
  /// Models the explicit barrier end in SYCL
  wait_end = XPTI_TRACE_POINT_END(11),
  /// Used to trace function call begin, from libraries, for example. This
  /// trace point type does not require an event object for the parent or the
  /// event of interest, but information about the function being traced needs
  /// to be sent using the user_data parameter in the xptiNotifySubscribers()
  /// call.
  function_begin = XPTI_TRACE_POINT_BEGIN(12),
  /// Used to trace function call end
  function_end = XPTI_TRACE_POINT_END(12),
  /// Use to notify that a new metadata entry is available for a given event
  metadata = XPTI_TRACE_POINT_BEGIN(13),
  /// Used to trace function call begin and its arguments.
  function_with_args_begin = XPTI_TRACE_POINT_BEGIN(14),
  /// Used to trace function call end.
  function_with_args_end = XPTI_TRACE_POINT_END(15),
  /// Indicates that the trace point is user defined and only the tool defined
  /// for a stream will be able to handle it
  user_defined = 1 << 7
};

///  @brief Enumerator defining the global/basic trace event types
///  @details The frame work defines the global/basic trace event types that
///  are necessary for modeling parallel runtimes.
///
///  The event_type data is of type uint8_t and the 7-LSB bits are used to
///  enumerate event types. the MSB bit is reserved for user-defined event
///  types and is set to 0 for predefined event types defined by the
///  framework.
///
///  When user-defined event types are being declared, a new ID is added to
///  this value to create a uint16_t data type. The LSB 8-bits have the 8th
///  bit set indicating that it is user-defined and the remaining 7-bits will
///  indicated the user defined trace event type. However, since multiple
///  tools or vendors could create their own trace event types, we require the
///  vendor_id to create a vendor namespace to avoid collisions.
///
///                                  user-defined bit
///                                    |
///                                    |
///                                    |+-----+---- 127 possible values for
///                                    ||     |     defining event types.
///                                    ||     |
///                                    ||     |
///                                    ||     |
///                                    vv     v
///   Field width (uint16_t) |........|........|
///                          15      8 7      0
///                           ^      ^
///                           |      |
///                           |      |
///                           +------+----- Reserved for vendor ID
enum class trace_event_type_t : uint16_t {
  /// In this case, the callback can choose to map it to something called
  /// unknown or ignore it entirely
  unknown_event = 0,
  /// Event type is graph - usually reported for traces from  graph or for
  /// graph, node or edge object creation
  graph = XPTI_EVENT(1),
  /// Algorithm type describes a parallel algorithm such as a parallel_for
  algorithm = XPTI_EVENT(2),
  /// Barrier event is usually a synchronization type that causes threads to
  /// wait until something happens and found in parallel algorithms and
  /// explicit
  /// synchronization use cases in asynchronous programming
  barrier = XPTI_EVENT(3),
  /// Activity in the scheduler that is not useful work is reported as this
  /// event type
  scheduler = XPTI_EVENT(4),
  /// Asynchronous activity event
  async = XPTI_EVENT(5),
  /// Synchronization event - only the contention time is captured by this
  /// event and marked as overhead
  lock = XPTI_EVENT(6),
  /// Indicates that the current event is an offload read request
  offload_read = XPTI_EVENT(7),
  /// Indicates that the current event is an offload write request
  offload_write = XPTI_EVENT(8),
  /// User defined event for extensibility and will have to be registered by
  /// the tool/runtime
  user_defined = 1 << 7
};

enum class trace_activity_type_t {
  /// Activity type is unknown; it is upto the collector handling the callback
  /// to mark it as needed
  unknown_activity = 0,
  /// Any activity reported by the tracing that results in useful work, hence
  /// active time
  active = 1,
  /// Activity that was primarily due to overheads such as time spent in
  /// barriers and schedulers, acquiring locks, etc
  overhead = 1 << 1,
  /// Activities that may be considered as background tasks; for example,
  /// asynchronous activities or region callbacks that are placeholders for
  /// nested activities
  background = 1 << 2,
  /// Explicit sleeps could be a result of calling APIs that result in zero
  /// active time
  sleep_activity = 1 << 3
};

struct reserved_data_t {
  /// Has a reference to the associated payload field for an event
  payload_t *payload = nullptr;
  /// Has additional metadata that may be defined by the user as key-value
  /// pairs
  metadata_t metadata;
};

struct trace_event_data_t {
  /// Unique id that corresponds to an event type or event group type
  uint64_t unique_id = 0;
  /// Data ID: ID that tracks the data elements streaming through the
  /// algorithm (mostly graphs; will be the same as instance_id for
  /// algorithms)
  uint64_t data_id = 0;
  /// Instance id of an algorithm with id=unique_id
  uint64_t instance_id = 0;
  /// The type of event
  uint16_t event_type;
  /// How this event is classified: active, overhead, barrier etc
  uint16_t activity_type;
  /// Unused 32-bit slot that could be used for any ids that need to be
  /// propagated in the future
  uint32_t unused;
  /// If event_type is "graph" and trace_type is "edge_create", then the
  /// source ID is set
  int64_t source_id = invalid_id;
  /// If event_type is "graph" and trace_type is "edge_create", then the
  /// target ID is set
  int64_t target_id = invalid_id;
  /// A reserved slot for memory growth, if required by the framework
  reserved_data_t reserved;
  /// User defined data, if required; owned by the user shared object and will
  /// not be deleted when event data is destroyed
  void *global_user_data = nullptr;
};

///
///  The error code list is incomplete and still
///  being defined.
///
enum class result_t : int32_t {
  // Success codes here (values >=0)
  XPTI_RESULT_SUCCESS = int32_t(0),
  XPTI_RESULT_FALSE = int32_t(1),
  // Error codes here (values < 0)
  XPTI_RESULT_FAIL = int32_t(0x80004001),
  XPTI_RESULT_NOTIMPL = int32_t(0x80004002),
  XPTI_RESULT_DUPLICATE = int32_t(0x80004003),
  XPTI_RESULT_NOTFOUND = int32_t(0x80004004),
  XPTI_RESULT_UNDELETE = int32_t(0x80004005),
  XPTI_RESULT_INVALIDARG = int32_t(0x80004006)
};

// These defines are present to enable plugin developers
// who want to subscribe to the streams from the framework
//
#if defined(_WIN64) || defined(_WIN32) /* Windows */
#ifdef XPTI_CALLBACK_API_EXPORTS
#define XPTI_CALLBACK_API __declspec(dllexport)
#else
#define XPTI_CALLBACK_API __declspec(dllimport)
#endif
#else /* Generic Unix/Linux */
#ifdef XPTI_CALLBACK_API_EXPORTS
#define XPTI_CALLBACK_API __attribute__((visibility("default")))
#else
#define XPTI_CALLBACK_API
#endif
#endif
/// @brief Callback function prototype
/// @details All callback functions that are registered with
/// the tracing framework have this signature.
///
/// @param [in] trace_type The trace type for which this callback has been
/// invoked.
/// @param [in] parent  Parent object for which the current object/trace is a
/// child of. If the current trace is not nested, the parent object will be
/// NULL.
/// @param [in] child  Child object for this callback has been invoked.
/// @param [in] user_data Data sent by the caller which can be anything and
/// the tool trying to interpret it needs to know the type for the handshake
/// to be successful. Most of the time, this field is used to send in const
/// char * data.
typedef void (*tracepoint_callback_api_t)(uint16_t trace_type,
                                          xpti::trace_event_data_t *parent,
                                          xpti::trace_event_data_t *child,
                                          uint64_t instance,
                                          const void *user_data);
typedef void (*plugin_init_t)(unsigned int, unsigned int, const char *,
                              const char *);
typedef void (*plugin_fini_t)(const char *);

constexpr uint16_t trace_task_begin =
    static_cast<uint16_t>(xpti::trace_point_type_t::task_begin);
constexpr uint16_t trace_task_end =
    static_cast<uint16_t>(xpti::trace_point_type_t::task_end);
constexpr uint16_t trace_wait_begin =
    static_cast<uint16_t>(xpti::trace_point_type_t::wait_begin);
constexpr uint16_t trace_wait_end =
    static_cast<uint16_t>(xpti::trace_point_type_t::wait_end);
constexpr uint16_t trace_barrier_begin =
    static_cast<uint16_t>(xpti::trace_point_type_t::barrier_begin);
constexpr uint16_t trace_barrier_end =
    static_cast<uint16_t>(xpti::trace_point_type_t::barrier_end);
constexpr uint16_t trace_graph_create =
    static_cast<uint16_t>(xpti::trace_point_type_t::graph_create);
constexpr uint16_t trace_node_create =
    static_cast<uint16_t>(xpti::trace_point_type_t::node_create);
constexpr uint16_t trace_edge_create =
    static_cast<uint16_t>(xpti::trace_point_type_t::edge_create);
constexpr uint16_t trace_signal =
    static_cast<uint16_t>(xpti::trace_point_type_t::signal);

constexpr uint16_t trace_graph_event =
    static_cast<uint16_t>(xpti::trace_event_type_t::graph);
constexpr uint16_t trace_algorithm_event =
    static_cast<uint16_t>(xpti::trace_event_type_t::algorithm);
} // namespace xpti

using xpti_tp = xpti::trace_point_type_t;
using xpti_te = xpti::trace_event_type_t;
using xpti_at = xpti::trace_activity_type_t;
using xpti_td = xpti::trace_event_data_t;

extern "C" {
/// @brief The framework loads the tool which implements xptiTraceInit() and
/// calls it when the runtime is being initialized
/// @details When tools implement callbacks and want to register them with
/// the runtime, they must implement the xptiTraceInit() and xptiTraceFinish()
/// functions and the runtime will try to resolve these symbols on load.
/// xptiTraceInit() is then called by the runtime so that the tool knows when
/// the runtime is instantiated so it can register its callbacks in the
/// xptiTraceInit() function.
///
/// When the runtime calls the tool's implementation of the xptiTraceInit()
/// function, it also provides the version of the runtime that is invoking the
/// init call. This allows tools implementers to handle certain calls based on
/// the runtime version the tools supports.
///
/// @code
/// void XPTI_CALLBACK_API xptiTraceInit
///     (
///        unsigned int maj,
///        unsigned int min,
///        const char *version,
///        const char *stream_name
///     )
/// {
///     std::string v = version; // make a copy of the version string
///     if(maj < 3) {
///       // do something here like registering callbacks
///       g_stream_id = xptiRegisterStream(stream_name);
///       xptiRegisterCallback(g_stream_id, graph_create, trace_point_begin);
///       xptiRegisterCallback(g_stream_id, node_create, trace_point_begin);
///       xptiRegisterCallback(g_stream_id, edge_create, trace_point_begin);
///       xptiRegisterCallback(g_stream_id, region_begin, trace_point_begin);
///       xptiRegisterCallback(g_stream_id, region_end, trace_point_end);
///       xptiRegisterCallback(g_stream_id, task_begin, trace_point_begin);
///       xptiRegisterCallback(g_stream_id, task_end, trace_point_end);
///       xptiRegisterCallback(g_stream_id, barrier_begin, trace_point_begin);
///       xptiRegisterCallback(g_stream_id, barrier_end, trace_point_end);
///       xptiRegisterCallback(g_stream_id, lock_begin, trace_point_begin);
///       xptiRegisterCallback(g_stream_id, lock_end, trace_point_end);
///       xptiRegisterCallback(g_stream_id, transfer_begin, trace_point_begin);
///       xptiRegisterCallback(g_stream_id, transfer_end, trace_point_end);
///       xptiRegisterCallback(g_stream_id, thread_begin, trace_point_begin);
///       xptiRegisterCallback(g_stream_id, thread_end, trace_point_end);
///       xptiRegisterCallback(g_stream_id, wait_begin, trace_point_begin);
///       xptiRegisterCallback(g_stream_id, wait_end, trace_point_end);
///       xptiRegisterCallback(g_stream_id, signal, trace_point_begin);
///     } else {
///       // report incompatible tool error message
///     }
/// }
/// @endcode
///
/// @param [in]  major_version The major version of the runtime
/// @param [in]  minor_version The minor version of the runtime. if the version
/// consists a tertiary number, it will not be reported. For example, if we have
/// a version number 5.1.23776, the only 5 and 1 we be reported for major and
/// minor versions. The API assumes that semantic versioning is being used for
/// the runtime/application.
///
/// @see https://semver.org/ Major revision number change will break API
/// compatibility. Minor revision number change will always be backward
/// compatible, but may contain additional functionality.
///
/// @param [in]  version_str Null terminated version string. This value is
/// guaranteed to be valid for the duration of the xptiTraceInit() call.
/// @param [in]  stream_name Null terminated string indicating the stream name
/// that is invoking this xptiTraceInit() call. This value is guaranteed to be
/// valid for the duration of the xptiTraceInit() call.
/// @return none
XPTI_CALLBACK_API void xptiTraceInit(unsigned int major_version,
                                     unsigned int minor_version,
                                     const char *version_str,
                                     const char *stream_name);

/// @brief Function to handle unloading of the module or termination of
/// application
/// @details This function will get called when the application
/// or the runtime implementing the trace point is about to be
/// unloaded or terminated.
///
/// @param [in]  stream_name Null terminated string indicating the stream name
/// that is invoking this xptiTraceFinish() call. This value is guaranteed to be
/// valid for the duration of the xptiTraceFinish() call. The subscriber who has
/// subscribed to this stream can now free up all internal data structures and
/// memory that has been allocated to manage the stream data.
XPTI_CALLBACK_API void xptiTraceFinish(const char *stream_name);
}
