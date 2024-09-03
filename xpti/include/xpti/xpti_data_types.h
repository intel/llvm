//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
#pragma once
#include <atomic>
#include <cmath>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <unordered_map>

namespace xpti {
/// @typedef uid64_t
/// @brief Defines a 64-bit unique identifier type that maps to a uid128_t.
///
/// This type is used throughout the system to uniquely identify tracepoints.
/// The 64-bit size ensures a large enough space to generate unique IDs for the
/// lifetime of the application or system being traced or monitored and is
/// compatible with previous versions of the XPTI API.
///
/// The `uid64_t` is typically used to maintain backward compatibility, but adds
/// an additional lookup overhead as all framework implementations are for
/// 128-bit keys to avoid collisions.
///
using uid64_t = uint64_t;

/// @struct universal_id_t
/// @brief Represents a unique identifier for tracking entities.
///
/// This structure is designed to uniquely identify code location that includes
/// file name, function name, line and column numbers withing the file. It
/// combines multiple pieces of information into two 64-bit integers and an
/// instance count to form a comprehensive ID.
///
struct universal_id_t {
  /// @brief Holds the combined IDs for file and function names.
  ///
  /// The upper 32 bits contain the string ID for the file name, while the lower
  /// 32 bits contain the string ID for the function name. This allows for a
  /// compact representation of both file and function identifiers within a
  /// single 64-bit integer.
  ///
  uint64_t p1 = 0;

  /// @brief Contains the line and column numbers.
  ///
  /// The lower 32 bits are used to store the line number where the entity is
  /// located, and the upper 32 bits store the column number. This precise
  /// location information is useful for pinpointing the exact position of the
  /// tracepoint.
  ///
  uint64_t p2 = 0;

  /// @brief A mutable counter for instance information.
  ///
  /// This value is used to hold the count of the number of occurrences of a
  /// given UID, allowing for the tracking of multiple instances of an entity.
  /// It is mutable to permit modification even if the `universal_id_t` object
  /// is const.
  ///
  uint64_t instance = 0;

  /// @brief Unique 64-bit identifier that maps to the 128-bit key in p1,p2.
  ///
  /// This variable represents a 64-bit hash value used as a unique identifier
  /// within the tracing framework. The hash is a 64-bit mapping of the 128-bit
  /// represented by the attributes p1 and p2 and will be used by legacy API.
  /// This field is optional and only populated when using legacy API.
  ///
  uid64_t uid64 = 0;
};

/// @typedef uid128_t
/// @brief Alias for xpti::universal_id_t representing a unique 128-bit
/// identifier.
///
/// This type alias simplifies the usage of xpti::universal_id_t by providing a
/// shorter and more descriptive name. It is intended to be used in contexts
/// where a 128-bit unique identifier is required, encapsulating both the unique
/// identification and instance tracking capabilities of the universal_id_t
/// structure.
///
using uid128_t = xpti::universal_id_t;

/// @brief Creates a unique 128-bit identifier (UID) for tracking entities.
///
/// This function combines file and function identifiers with line and column
/// information to generate a unique identifier. The UID is composed of two
/// 64-bit parts: the first part (p1) combines the file and function IDs, and
/// the second part (p2) combines the column and line numbers. An initial
/// instance count of 1 is set, indicating the creation of a new UID.
///
/// @param FileID The unique identifier for the file.
/// @param FuncID The unique identifier for the function.
/// @param Line The line number where the entity is located.
/// @param Col The column number where the entity is located.
/// @return A `xpti::uid128_t` structure representing the unique identifier.
///
inline xpti::uid128_t make_uid128(uint64_t FileID, uint64_t FuncID, int Line,
                                  int Col) {
  xpti::uid128_t UID;
  UID.p1 = (FileID << 32) | FuncID;
  UID.p2 = ((uint64_t)Col << 32) | Line;
  UID.instance = 0;
  return UID;
}

/// @brief Checks if a given 128-bit UID is valid.
///
/// A 128-bit UID is considered valid if neither of its parts (p1, p2) are zero
/// and its instance number is greater than 0. This function evaluates these
/// conditions and returns true if all are met, indicating the UID is valid.
///
/// @param UID The 128-bit UID to be checked.
/// @return True if the UID is valid, false otherwise.
///
inline bool is_valid_uid(const xpti::uid128_t &UID) {
  return (UID.p1 != 0 || UID.p2 != 0) && UID.instance > 0;
}

/// @brief Hash generation helper
/// @details The Universal ID concept in XPTI requires a good hashing function
/// and this data type provides the necessary functionality for creating the
/// Universal ID. Earlier simplementation were derived from the paper:
/// "Strongly universal string hashing is fast" by Daniel Lemire and Owen Kaser
/// and the corresponding Java implementation available in the blog 2018/08/15:
/// https:// github.com/lemire/Code-used-on-Daniel-Lemire-s-blog/
///
/// However, the probability of collisions increase with not having a codeptr
/// field populated and relying on the other fields, where some of them may be
/// absent if NDEBUG is in use, to generate the hash. The current implementation
/// uses the file_id, func_id, line and column information to generate the hash.
/// The hash is generated by compacting the input values into a 64-bit value.
/// The hash can then be used to index into the hash table, if required
struct hash_t {
  /// @brief Calculates the number of bits required to represent a given value.
  ///
  /// This function uses the logarithm base 2 (log2) of the input value to
  /// calculate the number of bits required to represent it. It then adds 1 to
  /// the result to account for the fact that log2 of a value is one less than
  /// the number of bits required to represent it.
  ///
  /// @param value A 64-bit integer for which the bit count is to be calculated.
  /// @return The number of bits required to represent the input value.
  int bit_count(uint64_t value) { return (int)log2(value) + 1; }

  /// @brief Compacts the file ID, function ID, line number, and column number
  /// into a single 64-bit hash value.
  ///
  /// This function calculates the number of bits necessary to represent each of
  /// the input values using the bit_count function. It then creates a hash
  /// value by shifting and combining these values. The hash value may grow to
  /// more than 64-bits as the string tables grow and may overflow the
  /// accumulator.
  ///
  /// @param file_id A 64-bit value that represents the file ID.
  /// @param func_id A 64-bit value that represents the function ID.
  /// @param line An integer that represents the line number.
  /// @param col An integer that represents the column number.
  /// @return A 64-bit hash value that represents the compacted file ID,
  /// function ID, line number, and column number.
  uint64_t compact(uint64_t file_id, uint64_t func_id, int line, int col) {
    uint64_t funcB, lineB, colB;
    // Figure out the bit counts necessary to represent the input values
    funcB = bit_count(func_id);
    lineB = bit_count(line);
    colB = bit_count(col);
    // Prepare the hash value by compacting the input values; this hash may grow
    // to more than 64-bits as the string tables grow and may overflow the
    // accumulator
    uint64_t hash = file_id;
    hash <<= funcB;
    hash = hash | func_id;
    hash <<= lineB;
    hash = hash | (uint64_t)line;
    hash <<= colB;
    hash = hash | (uint64_t)col;
#ifdef DEBUG
    uint64_t fileB = bit_count(file_id);
    std::cout << "Total bits: " << (fileB + funcB + lineB + colB) << "\n";
    std::cout << "Hash = " << std::hex << hash << std::dec << std::endl;
#endif
    return hash;
  }

  /// @brief Compacts the file ID, function ID, and line number into a single
  /// 64-bit hash value.
  ///
  /// This function calculates the number of bits necessary to represent each of
  /// the input values using the bit_count function. It then creates a hash
  /// value by shifting and combining these values. The hash value may grow to
  /// more than 64-bits as the string tables grow and may overflow the
  /// accumulator. However, this function has a better chance of success than
  /// the compact function as the "column" information is not encoded.
  ///
  /// @param file_id A 64-bit value that represents the file ID.
  /// @param func_id A 64-bit value that represents the function ID.
  /// @param line An integer that represents the line number.
  /// @return A 64-bit hash value that represents the compacted file ID,
  /// function ID, and line number.
  uint64_t compact_short(uint64_t file_id, uint64_t func_id, int line) {
    uint64_t funcB, lineB;
    funcB = bit_count(func_id);
    lineB = bit_count(line);
    // Prepare the hash value by compacting the input values; this hash may
    // also grow to more than 64-bits as the string tables grow and may
    // overflow the accumulator, but we have a better chance of success as
    // the "column" information is not encoded
    uint64_t hash = file_id;
    hash <<= funcB;
    hash = hash | func_id;
    hash <<= lineB;
    hash = hash | (uint64_t)line;
#ifdef DEBUG
    uint64_t fileB = bit_count(file_id);
    std::cout << "Total bits: " << (fileB + funcB + lineB) << "\n";
    std::cout << "Short Hash = " << std::hex << hash << std::dec << std::endl;
#endif
    return hash;
  }

  /// @brief Combines the file ID, function ID, line number, and column number
  /// from a uid_t object into a single 64-bit value.
  ///
  /// This function extracts the file ID and function ID from the first 64-bit
  /// field of the uid_t object (p1), and the line number and column number from
  /// the second 64-bit field (p2). It then combines these four values into a
  /// single 64-bit value using the compact function.
  ///
  /// @param uid A uid_t object that contains the file ID, function ID, line
  /// number, and column number to be combined.
  /// @return A 64-bit value that represents the combined file ID, function ID,
  /// line number, and column number.
  ///
  uint64_t combine(const xpti::uid128_t &uid) {
    uint64_t FileID = uid.p1 >> 32;
    uint64_t FuncID = uid.p1 & 0x00000000ffffffff;
    uint32_t Line = (uint32_t)(uid.p2 & 0x00000000ffffffff);
    uint32_t Col = (uint32_t)(uid.p2 >> 32);
    return compact(FileID, FuncID, Line, Col);
  }

  /// @brief Combines the file ID, function ID, and line number from a uid_t
  /// object into a single 64-bit value.
  ///
  /// This function extracts the file ID and function ID from the first 64-bit
  /// field of the uid_t object (p1), and the line number from the second 64-bit
  /// field (p2). It then combines these three values into a single 64-bit value
  /// using the compact function.
  ///
  /// @param uid A uid_t object that contains the file ID, function ID, and line
  /// number to be combined.
  /// @return A 64-bit value that represents the combined file ID, function ID,
  /// line number, and column number.
  ///
  uint64_t combine_short(const xpti::uid128_t &uid) {
    uint64_t FileID = uid.p1 >> 32;
    uint64_t FuncID = uid.p1 & 0x00000000ffffffff;
    uint32_t Line = (uint32_t)(uid.p2 & 0x00000000ffffffff);
    return compact_short(FileID, FuncID, Line);
  }
};

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
    return false;
  }

  bool operator==(const uid_t &rhs) const {
    return p1 == rhs.p1 && p2 == rhs.p2;
  }
};
} // namespace xpti

namespace xpti {
constexpr int invalid_id = -1;
constexpr uint64_t invalid_uid = 0;
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
  /// Payload has been registered with the framework
  PayloadRegistered = 1 << 15,
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
using object_id_t = int32_t;

using safe_flag_t = std::atomic<bool>;
using safe_uint64_t = std::atomic<uint64_t>;
using safe_uint32_t = std::atomic<uint32_t>;
using safe_uint16_t = std::atomic<uint16_t>;
using safe_int64_t = std::atomic<int64_t>;
using safe_int32_t = std::atomic<int32_t>;
using safe_int16_t = std::atomic<int16_t>;
// We will always return the metadata as a std::unordered_map
using metadata_t = std::unordered_map<string_id_t, object_id_t>;

#define XPTI_EVENT(val) xpti::event_type_t(val)
#define XPTI_TRACE_POINT_BEGIN(val) xpti::trace_point_t(val << 1 | 0)
#define XPTI_TRACE_POINT_END(val) xpti::trace_point_t(val << 1 | 1)

#define XPTI_PACK08_RET16(value1, value2) ((value1 << 8) | value2)
#define XPTI_PACK16_RET32(value1, value2) ((value1 << 16) | value2)
#define XPTI_PACK32_RET64(value1, value2) (((uint64_t)value1 << 32) | value2)

struct object_data_t {
  size_t size;
  const char *data;
  uint8_t type;
};

/// @struct payload_t
/// @brief Represents the detailed information about a trace event.
///
/// This structure encapsulates all the necessary details about a trace event,
/// including its name, stack trace, source file location, and more. It is
/// designed to provide a comprehensive view of an event for tracing and
/// debugging purposes.
///
/// @var const char* payload_t::name
/// The name of the trace point, which could represent a graph, algorithm, lock
/// names, etc.
///
/// @var const char* payload_t::stack_trace
/// Stack trace information in the format "caller->callee", providing a snapshot
/// of the call stack.
///
/// @var const char* payload_t::source_file
/// The absolute path of the source file. This may need to support unicode
/// strings for full compatibility.
///
/// @var uint32_t payload_t::line_no
/// Line number information to correlate the trace point within its source file.
///
/// @var uint32_t payload_t::column_no
/// Column number information for a complex statement to precisely locate the
/// trace point.
///
/// @var const void* payload_t::code_ptr_va
/// The virtual address of the kernel/lambda/function, providing a direct
/// reference to the code.
///
/// @var uint64_t payload_t::internal
/// Reserved for internal bookkeeping; should not be modified externally and
/// contains the 64-bit Universal ID for use with legacy API.
///
/// @var uint64_t payload_t::flags
/// Flags indicating the availability of name, code pointer, source file, and
/// hash values.
///
/// @var uid_t payload_t::uid
/// Legacy universal ID associated with this payload that is used to generate a
/// 64-bit hash. This is deprecated and no longer used to create the 64-bit UID.
/// However, it may be used to generate a hash for std::unordered_map
/// containers.
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
  payload_t(const void *codeptr) {
    code_ptr_va = codeptr;
    name = nullptr;         ///< Invalid name string pointer
    source_file = nullptr;  ///< Invalid source file string pointer
    line_no = invalid_id;   ///< Invalid line number
    column_no = invalid_id; ///< Invalid column number
    if (codeptr) {
      flags = (uint64_t)payload_flag_t::CodePointerAvailable;
    }
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
    if (func_name) {
      flags = (uint64_t)(payload_flag_t::NameAvailable);
    }
  }

  payload_t(const char *func_name, const void *codeptr) {
    code_ptr_va = codeptr;
    name = func_name;      ///< Invalid name string pointer
    source_file = nullptr; ///< Invalid source file string pointer
    if (func_name) {
      flags = (uint64_t)(payload_flag_t::NameAvailable);
    }
    if (codeptr) {
      flags |= (uint64_t)payload_flag_t::CodePointerAvailable;
    }
  }

  //  When the end user opts out of preserving the code location information and
  //  the KernelInfo is not available from the given entry point, we will rely
  //  on dynamic backtrace as a possibility. In this case, we send in the
  //  caller/callee information as a string in the form "caller->callee" that
  //  will be used to generate the unique ID.
  payload_t(const char *kname, const char *caller_callee, const void *codeptr) {
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
            const void *codeptr = nullptr) {
    code_ptr_va = codeptr;
    /// Capture the rest of the parameters
    name = kname;
    source_file = sf;
    line_no = line;
    column_no = col;
    if (kname && kname[0] != '\0') {
      flags = (uint64_t)payload_flag_t::NameAvailable;
    }
    if (sf && sf[0] != '\0') {
      flags |= (uint64_t)payload_flag_t::SourceFileAvailable |
               (uint64_t)payload_flag_t::LineInfoAvailable |
               (uint64_t)payload_flag_t::ColumnInfoAvailable;
    }
    if (codeptr) {
      flags |= (uint64_t)payload_flag_t::CodePointerAvailable;
    }
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
///  trace. This reserved bit is used by the scoped notify classes
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
  function_with_args_end = XPTI_TRACE_POINT_END(14),
  /// Used to notify that a new memory allocation is about to start.
  mem_alloc_begin = XPTI_TRACE_POINT_BEGIN(16),
  /// Used to notify that a memory allocation took place.
  mem_alloc_end = XPTI_TRACE_POINT_END(16),
  /// Used to notify that memory chunk will be released.
  mem_release_begin = XPTI_TRACE_POINT_BEGIN(17),
  /// Used to notify that memory has been released.
  mem_release_end = XPTI_TRACE_POINT_END(17),
  /// Used to notify that offload memory object will be created
  offload_alloc_memory_object_construct = XPTI_TRACE_POINT_BEGIN(18),
  /// Used to notify that offload memory object will be destructed
  offload_alloc_memory_object_destruct = XPTI_TRACE_POINT_END(18),
  /// Used to notify about association between user and internal
  /// handle of the offload memory object
  offload_alloc_memory_object_associate = XPTI_TRACE_POINT_BEGIN(19),
  /// Used to notify about releasing internal handle for offload memory object
  offload_alloc_memory_object_release = XPTI_TRACE_POINT_END(19),
  /// Used to notify about creation accessor for offload memory object
  offload_alloc_accessor = XPTI_TRACE_POINT_BEGIN(20),
  /// User to notify when a queue has been created
  queue_create = XPTI_TRACE_POINT_BEGIN(21),
  /// User to notify when a queue has been destroyed
  queue_destroy = XPTI_TRACE_POINT_END(21),
  /// Used to notify error/informational messages and no action to take
  diagnostics = XPTI_TRACE_POINT_BEGIN(63),
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
  /// Indicates that the current event is an offload memory object related
  offload_memory_object = XPTI_EVENT(9),
  /// Indicates that the current event is an offload accessor related
  offload_accessor = XPTI_EVENT(10),
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

/// Provides hints to the tools on how to interpret unknown metadata values.
enum class metadata_type_t {
  binary = 0,
  string = 1,
  signed_integer = 2,
  unsigned_integer = 3,
  floating = 4,
  boolean = 5
};

/// @struct reserved_data_t
/// @brief Holds additional data associated with a trace event.
///
/// This structure is designed to extend a trace event with more detailed
/// information, allowing for user-defined metadata and a direct reference to
/// the event's payload.
///
/// @var payload_t* reserved_data_t::payload
/// A pointer to the associated payload for an event. This links the reserved
/// data directly to the detailed information about the trace event, such as its
/// name, source file, and stack trace.
///
/// @var metadata_t reserved_data_t::metadata
/// User-defined metadata for the event, stored as key-value pairs. This allows
/// for the attachment of additional contextual information to an event, beyond
/// what is captured in the standard payload structure.
///
struct reserved_data_t {
  /// Has a reference to the associated payload field for an event
  payload_t *payload = nullptr;
  /// Has additional metadata that may be defined by the user as key-value
  /// pairs
  metadata_t metadata;
};

/// @enum trace_event_flag_t
/// @brief Enumerates the flags used to indicate the availability of specific
/// types of information in a trace event.
///
/// This enumeration is used within the tracing system to specify what kinds of
/// information are available for a particular trace event. Each flag represents
/// a different type of information that can be attached to a trace event,
/// allowing for flexible and detailed event descriptions.
///
enum class trace_event_flag_t {
  /// @var trace_event_flag_t::UIDAvailable
  /// Indicates that a unique identifier (UID) for the trace event is available.
  /// This UID refers to the 128-bit key used for representing the Universal ID.
  UIDAvailable = 1,

  /// @var trace_event_flag_t::SourceUIDAvailable
  /// Signifies that the unique identifier (UID) for the source of an edge trace
  /// event is available. This could be used to identify the specific module,
  /// function, or the associated source node for the event. Edges usually
  /// represent dependencies.
  SourceUIDAvailable = 1 << 1,

  /// @var trace_event_flag_t::TargetUIDAvailable
  /// Denotes that the unique identifier (UID) for the target of the edge trace
  /// event is available. This is useful for events that represent dependencies
  /// and allow for precise identification of the target.
  TargetUIDAvailable = 1 << 2,

  /// @var trace_event_flag_t::EventTypeAvailable
  /// Indicates that information about the type of the event is available. This
  /// can be used to categorize the type of event.
  EventTypeAvailable = 1 << 3,

  /// @var trace_event_flag_t::ActivityTypeAvailable
  /// Signifies that information about the type of activity associated with the
  /// event is available. This provides additional context about the nature of
  /// the event, such as whether it is related to computation, data transfer, or
  /// other activities.
  ActivityTypeAvailable = 1 << 4,

  /// @var trace_event_flag_t::PayloadAvailable
  /// Indicates that a payload of additional data is available for the event.
  /// This payload can contain code location data related to the event or
  /// additional contextual information.
  PayloadAvailable = 1 << 5,

};

/// @struct trace_event_data_t
/// @brief Represents the data associated with a trace event.
///
/// This structure encapsulates all the necessary information for a trace event,
/// including unique identifiers, event types, and additional metadata. It
/// serves as a comprehensive data packet for tracing and profiling systems.
///
struct trace_event_data_t {
  /// @var trace_event_data_t::unique_id
  /// Unique identifier for the trace event. This is used to distinguish between
  /// different events in the tracing system and contains the legacy 64-bit
  /// universal ID
  uint64_t unique_id = 0;

  /// @var trace_event_data_t::data_id
  /// An identifier for the data associated with this event. This can be used to
  /// correlate this event with a data. This field is no longer used and will be
  /// deprecated in future versions.
  uint64_t data_id = 0;

  /// @var trace_event_data_t::instance_id
  /// An identifier for the instance of the event. This is useful for events
  /// that can occur multiple times in different contexts or locations. Will
  /// always be equal to `trace_event_data_t::universal_id.instance`.
  uint64_t instance_id = 0;

  /// @var trace_event_data_t::event_type
  /// A 16-bit code representing the type of event. This could be used to
  /// categorize events into groups such as start, stop, pause, etc. The default
  /// used is `algorithm`.
  uint16_t event_type;

  /// @var trace_event_data_t::activity_type
  /// A 16-bit code representing the type of activity associated with the event.
  /// This provides additional context about what the event is related to, such
  /// as computation, data transfer, wait, scheduler, etc. This is usually an
  /// optional field and the default is 'active' indicating useful compute time.
  uint16_t activity_type;

  /// @var trace_event_data_t::unused
  /// A 32-bit field reserved for padding to align the structure.
  uint32_t unused;

  /// @var trace_event_data_t::source_id
  /// An identifier for the source node of the current edge event. This is
  /// primarily used to represent relationships between entities in the trace
  /// data. Initialized to `invalid_uid` to indicate no source by default. Will
  /// be deprecated when the 128-bit UID is fully adopted.
  uint64_t source_id = invalid_uid;

  /// @var trace_event_data_t::target_id
  /// An identifier for the target node of the current edge or relationship
  /// event event. Similar to `source_id`, but represents the entity that is the
  /// recipient or focus of the event. Initialized to `invalid_uid` to indicate
  /// no target by default.Will be deprecated when the 128-bit UID is fully
  /// adopted.
  uint64_t target_id = invalid_uid;

  /// @var trace_event_data_t::reserved
  /// A `reserved_data_t` structure that holds a reference to an associated
  /// payload and additional user-defined metadata. This allows for
  /// extensibility and custom data to be attached to the event.
  reserved_data_t reserved;

  // @var trace_event_data_t::global_user_data
  /// A pointer to user-defined data that is globally relevant to the event.
  /// This could be used to attach arbitrary data that doesn't fit into the
  /// standard fields.
  void *global_user_data = nullptr;

  /// @var trace_event_data_t::flags
  /// A 64-bit field for flags or additional bitwise information related to the
  /// event. This is primarily used to determine if the event is valid and the
  /// pieces of information that are available.
  uint64_t flags = 0;
};

/// @struct tracepoint_data_t
/// @brief This struct represents a trace point's data in the tracing framework
/// and is populated with the current tracepoint's information before stashing
/// it in TLS.
///
/// It contains a unique identifiers (uid128, uid64), the payload used to
/// construct the UIDs and the corresponding trace event.
///
struct tracepoint_data_t {
  /// @brief This is a unique identifier for the trace point.
  ///
  /// It is a 128-bit unsigned integer, which represents the payload
  /// information.
  ///
  xpti::universal_id_t uid128;

  /// @brief This is a pointer to the payload associated with the trace point.
  ///
  /// It is of type payload_t that includes file name, function name, line
  /// number, and column number for the associated tracepoint. The pointer to
  /// the payload is valid through the lifetime of the program and points to the
  /// registered payload.
  ///
  payload_t *payload = nullptr;

  /// @brief This is a pointer to the event associated with the payload
  /// instance.
  ///
  /// When a payload is provided to tracepoint_data_t object, it will register
  /// the payload to get the the new UID which has an updated instance. Using
  /// this UID, a new event is also created and stashed here so it can be
  /// updated to TLS.
  ///
  trace_event_data_t *event = nullptr;

  /// @brief This is a 64-bit unique identifier representation for the `uid128`
  /// for the trace point.
  ///
  /// It is of type uint16_t, which is the key to lookup the payload information
  /// through `xptiQueryPayloadByUID()` call. This information is optional and
  /// has to be populated by the compatibility API for supporting the legacy
  /// APIs.
  ///
  uint64_t uid64 = xpti::invalid_uid;

  /// @brief This method checks if the trace point data is valid.
  ///
  /// It returns true if both the uids are valid, the payload and event pointers
  /// are not null. Otherwise, it returns false.
  ///
  /// @return True if data is valid, false otherwise.
  ///
  bool isValid() { return (xpti::is_valid_uid(uid128) && payload && event); }
};

/// Describes offload buffer
struct offload_buffer_data_t {
  /// A pointer to user level memory offload object.
  uintptr_t user_object_handle = 0;
  /// A pointer to host memory offload object.
  uintptr_t host_object_handle = 0;
  /// A string representing the type of buffer element.
  const char *element_type = nullptr;
  /// Buffer element size in bytes
  uint32_t element_size = 0;
  /// Buffer dimensions number.
  uint32_t dim = 0;
  /// Buffer size for each dimension.
  size_t range[3] = {0, 0, 0};
};

/// Describes offload sampled image
struct offload_image_data_t {
  /// A pointer to user level memory offload object.
  uintptr_t user_object_handle = 0;
  /// A pointer to host memory offload object.
  uintptr_t host_object_handle = 0;
  /// Buffer dimensions number.
  uint32_t dim = 0;
  /// Buffer size for each dimension.
  size_t range[3] = {0, 0, 0};
  /// Image format.
  uint32_t format = 0;
  /// Addressing mode of the associated sampler if the image is sampled.
  std::optional<uint32_t> addressing = std::nullopt;
  /// Coordinate normalization mode of the associated sampler if the image is
  /// sampled.
  std::optional<uint32_t> coordinate_normalization = std::nullopt;
  /// Filtering mode of the associated sampler if the image is sampled.
  std::optional<uint32_t> filtering = std::nullopt;
};

/// Describes offload accessor
struct offload_accessor_data_t {
  /// A pointer to user level buffer offload object.
  uintptr_t buffer_handle = 0;
  /// A pointer to user level accessor offload object.
  uintptr_t accessor_handle = 0;
  /// Access target
  uint32_t target = 0;
  /// Access mode
  uint32_t mode = 0;
};

/// Describes offload sampled image accessor
struct offload_image_accessor_data_t {
  /// A pointer to user level image offload object.
  uintptr_t image_handle = 0;
  /// A pointer to user level accessor offload object.
  uintptr_t accessor_handle = 0;
  /// Access target. Only present on non-host accessors.
  std::optional<uint32_t> target = std::nullopt;
  /// Access mode. Only present on unsampled image accessors.
  std::optional<uint32_t> mode = std::nullopt;
  /// A string representing the type of element.
  const char *element_type = nullptr;
  /// Element size in bytes
  uint32_t element_size = 0;
};

/// Describes association between user level and platform specific
/// offload memory object
struct offload_association_data_t {
  /// A pointer to user level memory offload object.
  uintptr_t user_object_handle = 0;
  /// A pointer to platform specific handler for the offload object
  uintptr_t mem_object_handle = 0;
};

/// Describes enqueued kernel object
struct offload_kernel_enqueue_data_t {
  /// Global size
  size_t global_size[3] = {0, 0, 0};
  /// Local size
  size_t local_size[3] = {0, 0, 0};
  /// Offset
  size_t offset[3] = {0, 0, 0};
  /// Number of kernel arguments
  size_t args_num = 0;
};

/// Describes enqueued kernel argument
struct offload_kernel_arg_data_t {
  /// Argument type as set in kernel_param_kind_t
  int type = -1;
  /// Pointer to the data
  void *pointer = nullptr;
  /// Size of the argument
  int size = 0;
  /// Index of the argument in the kernel
  int index = 0;
};

/// Describes memory allocation
struct mem_alloc_data_t {
  /// A platform-specific memory object handle. Some heterogeneous programming
  /// models (like OpenCL and SYCL) have notion of memory objects, that are
  /// universal across host and all devices. In such models, for each device a
  /// new device-specific allocation must take place. This handle can be used to
  /// tie different allocations across devices to their runtime-managed memory
  /// objects.
  uintptr_t mem_object_handle = 0;
  /// A pointer to allocated piece of memory.
  uintptr_t alloc_pointer = 0;
  /// Size of memory allocation in bytes.
  size_t alloc_size = 0;
  /// Size of guard zone in bytes. Some analysis tools can ask allocators to add
  /// some extra space in the end of memory allocation to catch out-of-bounds
  /// memory accesses. Allocators, however, must honor rules of the programming
  /// model when allocating memory. This value can be used to indicate the real
  /// guard zone size, that has been used to perform allocation.
  size_t guard_zone_size = 0;
  /// Reserved for future needs
  void *reserved = nullptr;
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
constexpr uint16_t trace_function_begin =
    static_cast<uint16_t>(xpti::trace_point_type_t::function_begin);
constexpr uint16_t trace_function_end =
    static_cast<uint16_t>(xpti::trace_point_type_t::function_end);
constexpr uint16_t trace_function_with_args_begin =
    static_cast<uint16_t>(xpti::trace_point_type_t::function_with_args_begin);
constexpr uint16_t trace_function_with_args_end =
    static_cast<uint16_t>(xpti::trace_point_type_t::function_with_args_end);
constexpr uint16_t trace_offload_alloc_memory_object_construct =
    static_cast<uint16_t>(
        xpti::trace_point_type_t::offload_alloc_memory_object_construct);
constexpr uint16_t trace_offload_alloc_memory_object_associate =
    static_cast<uint16_t>(
        xpti::trace_point_type_t::offload_alloc_memory_object_associate);
constexpr uint16_t trace_offload_alloc_memory_object_destruct =
    static_cast<uint16_t>(
        xpti::trace_point_type_t::offload_alloc_memory_object_destruct);
constexpr uint16_t trace_offload_alloc_memory_object_release =
    static_cast<uint16_t>(
        xpti::trace_point_type_t::offload_alloc_memory_object_release);
constexpr uint16_t trace_offload_alloc_accessor =
    static_cast<uint16_t>(xpti::trace_point_type_t::offload_alloc_accessor);

constexpr uint16_t trace_graph_event =
    static_cast<uint16_t>(xpti::trace_event_type_t::graph);
constexpr uint16_t trace_algorithm_event =
    static_cast<uint16_t>(xpti::trace_event_type_t::algorithm);
constexpr uint16_t trace_offload_memory_object_event =
    static_cast<uint16_t>(xpti::trace_event_type_t::offload_memory_object);
constexpr uint16_t trace_offload_accessor_event =
    static_cast<uint16_t>(xpti::trace_event_type_t::offload_accessor);

constexpr uint16_t trace_queue_create =
    static_cast<uint16_t>(xpti::trace_point_type_t::queue_create);
constexpr uint16_t trace_queue_destroy =
    static_cast<uint16_t>(xpti::trace_point_type_t::queue_destroy);

constexpr uint16_t trace_diagnostics =
    static_cast<uint16_t>(xpti::trace_point_type_t::diagnostics);
} // namespace xpti

namespace std {
// Specializations for std::unordered_map

// Specialization of std::hash for xpti::uid128_t
template <> struct hash<xpti::uid128_t> {
  // Overload of operator() to calculate hash of xpti::uid128_t
  size_t operator()(const xpti::uid128_t &UID) const {
    xpti::hash_t Hash;
    // The hash is calculated by combining the file ID, function ID, and line
    // number from the uid_t object into a single 64-bit value.
    return Hash.combine_short(UID);
  }
};

// Specialization of std::equal_to for xpti::uid128_t
template <> struct equal_to<xpti::uid128_t> {
  // Overload of operator() to compare two xpti::uid128_t objects
  bool operator()(const xpti::uid128_t &lhs, const xpti::uid128_t &rhs) const {
    // Two uid_t objects are considered equal if their p1 & p2 fields are equal.
    // p1 contains the combined file ID and function ID, p2 contains the
    // combined line number and column number.
    return lhs.p1 == rhs.p1 && lhs.p2 == rhs.p2;
  }
};

template <> struct less<xpti::uid128_t> {
  // Overload of operator() to compare two xpti::uid128_t objects
  bool operator()(const xpti::uid128_t &lhs, const xpti::uid128_t &rhs) const {
    // Two uid_t objects are considered equal if their p1 & p2 fields are equal.
    // p1 contains the combined file ID and function ID, p2 contains the
    // combined line number and column number. For one to be less than the
    // other, one.p1 should be less than two.p1 or one.p1 == two.p1 and one.p2
    // less than two.p2 for 'one' to be considered less than 'two'
    if (lhs.p1 < rhs.p1)
      return true;
    else if (lhs.p1 == rhs.p1 && lhs.p2 < rhs.p2)
      return true;
    return false;
  }
};
/// Specialize std::hash to support xpti::uid_t
template <> struct hash<xpti::uid_t> {
  std::size_t operator()(const xpti::uid_t &key) const { return key.hash(); }
};

} // namespace std

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
