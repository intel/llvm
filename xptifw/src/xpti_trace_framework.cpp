//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

/// @file xpti_trace_framework.cpp
/// @brief Implementation of the XPTI Trace Framework.
///
/// This file contains the implementation of the XPTI Trace Framework, which is
/// designed to provide a lightweight and flexible tracing mechanism for
/// performance analysis and debugging of parallel applications. The framework
/// allows for instrumentation of the host code to capture various performance
/// metrics and events.

#include "xpti/xpti_trace_framework.hpp"
#include "parallel_hashmap/phmap.h"
#include "xpti_int64_hash_table.hpp"
#include "xpti_object_table.hpp"
#include "xpti_string_table.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define XPTI_USER_DEFINED_TRACE_TYPE16(value)                                  \
  ((uint16_t)xpti::trace_point_type_t::user_defined | (uint16_t)value)
#define XPTI_USER_DEFINED_EVENT_TYPE16(value)                                  \
  ((uint16_t)xpti::trace_event_type_t::user_defined | (uint16_t)value)
#define XPTI_EXTRACT_MSB16(val) (val >> 16)
#define XPTI_EXTRACT_LSB16(val) (val & 0x0000ffff)

#define XPTI_VENDOR_DEFINED_TRACE_TYPE16(vendor_id, trace_type)                \
  ((uint16_t)vendor_id << 8 | XPTI_USER_DEFINED_TRACE_TYPE16(trace_type))
#define XPTI_VENDOR_DEFINED_EVENT_TYPE16(vendor_id, event_type)                \
  ((uint16_t)vendor_id << 8 | XPTI_USER_DEFINED_EVENT_TYPE16(event_type))

static_assert(std::is_trivially_destructible<xpti::utils::SpinLock>::value,
              "SpinLock is not trivial");
static_assert(
    std::is_trivially_destructible<xpti::utils::PlatformHelper>::value,
    "PlatformHelper is not trivial");

// TLS variables to support stashing tupples and universal IDs
/// @file xpti_trace_framework.cpp
/// @brief This file contains the implementation of the XPTI trace framework.

/// @brief Alias for a tuple that is used to stash a key-value pair.
using stash_tuple_t = std::tuple<const char *, uint64_t>;

/// @brief A thread-local variable of type stash_tuple_t, default initialized.
/// This variable is used to stash a key-value pair/thread for use within a
/// scope.
static thread_local stash_tuple_t g_tls_stash_tuple = stash_tuple_t(nullptr, 0);

/// @brief A thread-local 64-bit unsigned integer, default initialized.
/// This variable is used to store a unique identifier within a program scope
/// for each thread.
static thread_local uint64_t g_tls_uid = xpti::invalid_uid;

/// @brief A TLS of type xpti::trace_point_data_t, default initialized.
/// This variable is used to store trace point data generated from code location
/// information in a function scope for all top level entry points recorded for
/// each thread.
static thread_local xpti::tracepoint_data_t g_tls_tracepoint_scope_data;

/// Default stream for the trace framework, if none is provided by the user.
constexpr const char *g_default_stream = "xpti.framework";

static std::once_flag g_initialize_default_stream_flag;

namespace xpti {
/// @var env_subscribers
/// @brief A constant character pointer initialized with the string
/// "XPTI_SUBSCRIBERS". This variable represents the environment variable name
/// for subscribers in the XPTI trace framework.
constexpr const char *env_subscribers = "XPTI_SUBSCRIBERS";
/// @var g_helper
/// @brief An instance of the PlatformHelper class from the xpti::utils
/// namespace. This instance is used to perform platform-specific operations.
xpti::utils::PlatformHelper g_helper;

/// @var g_framework_mutex
/// @brief An instance of the SpinLock class from the xpti::utils namespace.
/// This instance is used to ensure thread-safety in the XPTI trace framework.
xpti::utils::SpinLock g_framework_mutex;

/// @var g_default_stream_id
/// @brief A uint8_t variable initialized with 0.
/// This variable represents the default stream ID in the XPTI trace framework.
uint8_t g_default_stream_id = 0;

/// @var g_default_event_type
/// @brief A variable of type xpti::trace_event_type_t, initialized with
/// xpti::trace_event_type_t::algorithm. This variable represents the default
/// event type in the XPTI trace framework.
xpti::trace_event_type_t g_default_event_type =
    xpti::trace_event_type_t::algorithm;

/// @var g_default_trace_type
/// @brief A variable of type xpti::trace_point_type_t, initialized with
/// xpti::trace_point_type_t::function_begin. This variable represents the
/// default trace point type in the XPTI trace framework.
xpti::trace_point_type_t g_default_trace_type =
    xpti::trace_point_type_t::function_begin;

/// @brief A global boolean flag to control self-notification of trace points.
///
/// If this flag is set to true, the trace point will notify itself when it is
/// hit. This can be useful for debugging or for generating more detailed trace
/// information. By default, this flag is set to false, meaning that trace
/// points do not notify themselves.
bool g_tracepoint_self_notify = false;

/// @var thread_local xpti::tracepoint_data_t g_tls_temp_scope_data
/// @brief Thread-local storage for temporary scope data in the tracing
/// framework.
///
/// This variable is used to store temporary data for a tracepoint within a
/// specific thread's context. Being thread-local ensures that each thread has
/// its own instance of this data, preventing data races and inconsistencies in
/// multi-threaded environments. It is primarily used to hold intermediate
/// information about a tracepoint's state.
static thread_local xpti::tracepoint_data_t g_tls_temp_scope_data;

struct TracepointInstance {
  xpti::uid128_t MUId;
  xpti::payload_t MPayload;
  xpti::trace_event_data_t MEvent;

  /// @brief Constructs a new TracepointInstance instance.
  ///
  /// This constructor initializes the TracepointInstance instance with the
  /// provided tracepoint data
  ///
  /// @param Data The tracepoint data for the new instance.
  /// @param StreamID The stream ID for the new instance.
  void Update(xpti::uid128_t &UID, xpti::payload_t *Payload) {
    MUId = UID;
    MPayload.name = Payload->name;
    MPayload.source_file = Payload->source_file;
    MPayload.line_no = Payload->line_no;
    MPayload.column_no = Payload->column_no;
    MPayload.flags = Payload->flags;

    xpti::framework::uid_object_t UidHelper(MUId);
    MPayload.uid.p1 = Payload->uid.p1 =
        XPTI_PACK32_RET64(UidHelper.fileId(), UidHelper.lineNo());
    MPayload.uid.p2 = Payload->uid.p2 =
        XPTI_PACK32_RET64(0, UidHelper.functionId());
    MEvent.universal_id = MUId;
    MEvent.reserved.payload = &MPayload;
    MEvent.instance_id = MEvent.data_id = MUId.instance;
    MEvent.event_type = (uint16_t)xpti::trace_event_type_t::algorithm;
    MEvent.activity_type = (uint16_t)xpti::trace_activity_type_t::active;
    MEvent.source_id = xpti::invalid_uid;
    MEvent.target_id = xpti::invalid_uid;
    MEvent.flags |=
        static_cast<uint64_t>(xpti::trace_event_flag_t::UIDAvailable);
    MEvent.flags |=
        static_cast<uint64_t>(xpti::trace_event_flag_t::EventTypeAvailable);
    MEvent.flags |=
        static_cast<uint64_t>(xpti::trace_event_flag_t::ActivityTypeAvailable);
    MEvent.flags |=
        static_cast<uint64_t>(xpti::trace_event_flag_t::PayloadAvailable);
  }

  /// @brief Destructs the TracepointInstance instance.
  ///
  /// This destructor is responsible for cleaning up the TracepointInstance
  /// instance and releasing any resources it holds. It does not perform any
  /// specific cleanup operations at present.
  ~TracepointInstance() {}

  xpti::uid128_t *UID() { return &MUId; }
  xpti::payload_t *Payload() { return &MPayload; }
  xpti::trace_event_data_t *Event() { return &MEvent; }
};

/// @class Subscribers
/// @brief Manages the lifecycle of trace framework subscriber plugins.
///
/// This class is responsible for loading, initializing, finalizing, and
/// unloading plugins that are used in a tracing framework. It supports
/// querying, loading, and unloading plugins by their names or handles. It also
/// provides mechanisms to initialize and finalize plugins for specific streams.
/// The subscribers are defined in the environment variable XPTI_SUBSCRIBERS

class Subscribers {
public:
  /// @struct plugin_data_t
  /// @brief Holds plugin related information including initialization and
  /// finalization functions.
  ///
  /// This structure contains the details necessary to manage a plugin,
  /// including its handle, initialization and finalization entry points, name,
  /// and a validity flag to indicate that a subscriber is a valid plugin.

  struct plugin_data_t {
    /// The handle of the loaded shared object
    xpti_plugin_handle_t handle = nullptr;
    /// The initialization entry point
    xpti::plugin_init_t init = nullptr;
    /// The finalization entry point
    xpti::plugin_fini_t fini = nullptr;
    /// The name of the shared object (in UTF8?))
    std::string name;
    /// indicates whether the data structure is valid
    bool valid = false;
  };

  // Data structures defined to hold the plugin data that can be looked up by
  // plugin name or the handle
  using plugin_handle_lut_t = std::map<xpti_plugin_handle_t, plugin_data_t>;
  using plugin_name_lut_t = std::map<std::string, plugin_data_t>;

  // We unload all loaded shared objects in the destructor; Must not be invoked
  // in the DLLMain() function and possibly the __fini() function in Linux

  ~Subscribers() { unloadAllPlugins(); }

  /// Queries the plugin data information using the handle.
  /// @param Handle The handle of the plugin to query.
  /// @return The plugin data associated with the handle. If no plugin is found,
  /// returns a structure with the valid attribute set to 'false'.

  plugin_data_t queryPlugin(xpti_plugin_handle_t Handle) {
    plugin_data_t PData;
    {
      std::lock_guard<std::mutex> Lock(MMutex);
      if (MHandleLUT.count(Handle))
        return MHandleLUT[Handle];
      else
        return PData; // return invalid plugin data
    }
  }

  /// Loads a plugin from the specified path.
  /// @details Load the provided shared object file name using the explicit load
  /// API. If the load is successful, a test is performed to see if the shared
  /// object has the required entry points for it to be considered a trace
  /// plugin subscriber. If so, the internal data structures are updated and a
  /// valid handle is returned.
  /// @param Path The file path of the plugin to load.
  /// @return The handle to the loaded plugin, or nullptr if the plugin could
  /// not be loaded successfully or was invalid.

  xpti_plugin_handle_t loadPlugin(const char *Path) {
    xpti_plugin_handle_t Handle = 0;
    std::string Error;
    // Check to see if the subscriber has already been loaded; if so, return the
    // handle from the previously loaded library
    if (MNameLUT.count(Path)) {
      {
        std::lock_guard<std::mutex> Lock(MMutex);
        // This plugin has already been loaded, so let's return previously
        // recorded handle
        plugin_data_t &Data = MNameLUT[Path];
        assert(Data.valid && "Lookup is invalid!");
        if (Data.valid)
          return Data.handle;
      }
    }

    Handle = g_helper.loadLibrary(Path, Error);
    if (Handle) {
      // The tracing framework requires the tool plugins to implement the
      // xptiTraceInit() and xptiTraceFinish() functions. If these are not
      // present, then the plugin will be ruled an invalid plugin and unloaded
      // from the process.
      xpti::plugin_init_t InitFunc = reinterpret_cast<xpti::plugin_init_t>(
          g_helper.findFunction(Handle, "xptiTraceInit"));
      xpti::plugin_fini_t FiniFunc = reinterpret_cast<xpti::plugin_fini_t>(
          g_helper.findFunction(Handle, "xptiTraceFinish"));
      if (InitFunc && FiniFunc) {
        //  We appear to have loaded a valid plugin, so we will insert the
        //  plugin information into the two maps guarded by a lock
        plugin_data_t Data;
        Data.valid = true;
        Data.handle = Handle;
        Data.name = Path;
        Data.init = InitFunc;
        Data.fini = FiniFunc;
        {
          std::lock_guard<std::mutex> Lock(MMutex);
          MNameLUT[Path] = Data;
          MHandleLUT[Handle] = Data;
        }
      } else {
        // We may have loaded another shared object that is not a tool plugin
        // for the tracing framework, so we'll unload it now
        unloadPlugin(Handle);
        Handle = nullptr;
      }
    } else {
      //  Get error from errno
      if (!Error.empty())
        std::cout << '[' << Path << "]: " << Error << '\n';
    }
    return Handle;
  }

  /// @brief Unloads the plugin identified by the given handle.
  /// @param PluginHandle The handle of the plugin to unload.
  /// @return A result code indicating success or failure of the operation.

  xpti::result_t unloadPlugin(xpti_plugin_handle_t PluginHandle) {
    xpti::result_t Res = g_helper.unloadLibrary(PluginHandle);
    if (xpti::result_t::XPTI_RESULT_SUCCESS == Res) {
      auto Loc = MHandleLUT.find(PluginHandle);
      if (Loc != MHandleLUT.end()) {
        MHandleLUT.erase(PluginHandle);
      }
    }
    return Res;
  }

  /// @brief Checks if there are any valid subscribers loaded.
  /// @return True if there are valid subscribers, false otherwise.

  bool hasValidSubscribers() { return (MHandleLUT.size() > 0); }

  /// Initializes plugins for a given stream.
  /// @param Stream The name of the stream.
  /// @param major_revision The major version of the stream.
  /// @param minor_revision The minor version of the stream.
  /// @param version_string The version string of the stream.

  void initializeForStream(const char *Stream, uint32_t major_revision,
                           uint32_t minor_revision,
                           const char *version_string) {
    //  If there are subscribers registered, then initialize the subscribers
    //  with the new stream information.
    if (MHandleLUT.size()) {
      for (auto &Handle : MHandleLUT) {
        Handle.second.init(major_revision, minor_revision, version_string,
                           Stream);
      }
    }
  }

  /// Finalizes plugins for a given stream.
  /// @param Stream The name of the stream to finalize.

  void finalizeForStream(const char *Stream) {
    //  If there are subscribers registered, then finalize the subscribers for
    //  the stream
    if (MHandleLUT.size()) {
      for (auto &Handle : MHandleLUT) {
        Handle.second.fini(Stream);
      }
    }
  }

  /// Loads plugins specified in the environment variable XPTI_SUBSCRIBERS.
  /// The environment variable should contain a comma-separated list of plugin
  /// paths.

  void loadFromEnvironmentVariable() {
    if (!g_helper.checkTraceEnv())
      return;
    //  Load all registered Listeners by scanning the environment variable in
    //  "Env"; The environment variable, if set, extract the comma separated
    //  tokens into a vector.
    std::string Token, Env = g_helper.getEnvironmentVariable(env_subscribers);
    std::vector<std::string> Listeners;
    std::stringstream Stream(Env);

    //  Split the environment variable value by ',' and build a vector of the
    //  tokens (subscribers)
    while (std::getline(Stream, Token, ',')) {
      Listeners.push_back(Token);
    }

    size_t ValidSubscribers = Listeners.size();
    if (ValidSubscribers) {
      //  Let's go through the subscribers and load these plugins;
      for (auto &Path : Listeners) {
        // Load the plugins listed in the environment variable
        {
          std::lock_guard<std::mutex> Lock(MLoader);
          auto SubscriberHandle = loadPlugin(Path.c_str());
          if (!SubscriberHandle) {
            ValidSubscribers--;
            std::cout << "Failed to load " << Path << " successfully\n";
          }
        }
      }
    }
  }

  /// Unloads all loaded plugins.
  void unloadAllPlugins() {
    for (auto &Item : MNameLUT) {
      unloadPlugin(Item.second.handle);
    }
    MHandleLUT.clear();
    MNameLUT.clear();
  }

private:
  /// Hash map that maps shared object name to the plugin data
  plugin_name_lut_t MNameLUT;
  /// Hash map that maps shared object handle to the plugin data
  plugin_handle_lut_t MHandleLUT;
  /// Lock to ensure the operation on these maps are safe
  std::mutex MMutex;
  /// Mutex to ensure that plugin loading is thread-safe.
  std::mutex MLoader;
};

/// @brief Helper class to create and  manage tracepoints
/// @details The class uses the global string table to register the strings it
/// encounters in various payloads and builds internal hash maps to manage them.
/// This is a single point for managing tracepoints.
class Tracepoints {
public:
  /// @typedef uid_entry_t
  /// @brief A type alias for a pair containing an xpti::payload_t and an
  /// xpti::safe_uint64_t. This type is used to store a payload and a
  /// thread-safe unsigned 64-bit integer that tracks the instance.
  using uid_entry_t = std::pair<xpti::payload_t, uint64_t>;

  /// @typedef uid_payload_lut
  /// @brief A type alias for an unordered map from uid_t to uid_entry_t.
  /// This type is used to create a lookup table (lut) that maps unique
  /// identifiers (uids) to their corresponding payload entries. emhash8 is a
  /// flat_hash_map implementation that is one of the better performing hash
  /// maps under many circumstances with erase/delete being a bit slower than
  /// other hash map implementations. Since we plan to keep the payload
  /// information until the end of the application lifetime, we should be able
  /// to use the references.
  using uid_payload_lut = std::unordered_map<xpti::uid128_t, uid_entry_t>;

  /// @typedef uid_instances_t
  /// @brief Defines a hash map for managing tracepoint instances by their
  /// unique identifiers.
  ///
  /// This type alias represents a hash map where the key is a 64-bit unsigned
  /// integer representing the unique instance of a tracepoint, and the value is
  /// an instance of `xpti::TracepointInstance`. The `phmap::node_hash_map` is
  /// chosen for its efficiency in managing hash collisions and overall
  /// performance in insertions and lookups. This map is used within the trace
  /// framework to efficiently manage and access tracepoint instances by their
  /// UIDs, allowing for quick updates and retrievals of tracepoint data.
  using uid_instances_t =
      phmap::node_hash_map<uint64_t, xpti::TracepointInstance>;

  /// @typedef uid_tracepoints_lut
  /// @brief Maps 128-bit unique identifiers to their corresponding tracepoint
  /// instances.
  ///
  /// This type alias represents a hash map designed to efficiently associate
  /// 128-bit unique identifiers (UIDs) with their corresponding tracepoint
  /// instances. The `phmap::node_hash_map` is utilized for its performance
  /// characteristics, including efficient handling of hash collisions and
  /// optimized insertions and lookups. This mapping is crucial within the trace
  /// framework for organizing and accessing tracepoint instances by their UIDs,
  /// facilitating quick updates, retrievals, and management of tracepoint data
  /// across the system.
  using uid_tracepoints_lut = phmap::node_hash_map<uid128_t, uid_instances_t>;

  /// @typedef uid64_validity_lut
  /// @brief Represents a set for tracking the validity of 64-bit unique
  /// identifiers (UIDs).
  ///
  /// This type alias defines a flat hash set optimized for fast lookup,
  /// insertion, and deletion operations. It is used within the trace framework
  /// to maintain a collection of 64-bit UIDs that have been validated or are
  /// known to be in use. The `phmap::flat_hash_set` is chosen for its
  /// performance efficiency, particularly in scenarios where the set size is
  /// large and performance is critical. This set acts as a registry to quickly
  /// verify the validity of UIDs encountered during trace operations.
  using uid64_validity_lut = phmap::flat_hash_set<uint64_t>;

  /// @struct PayloadInstance
  /// @brief Represents an instance of a payload associated with its unique
  /// identifier (UID).
  ///
  /// This structure is designed to link a unique identifier (UID), which is a
  /// 128-bit value, with a payload that the 128-bit key is generated from.
  ///
  /// @var PayloadInstance::UId
  /// The 128-bit unique identifier associated with the payload. This UID is
  /// used to uniquely identify the payload instance within a system or
  /// framework.
  ///
  /// @var PayloadInstance::Payload
  /// A pointer to the payload associated with the UID. The payload contains the
  /// actual data or information that is linked to the UID. This pointer can be
  /// null if the provided payload data is invalid. In this case, an invalid UID
  /// and a nullptr for the payload is returned.
  ///
  struct PayloadInstance {
    xpti::uid128_t UId;
    xpti::payload_t *Payload;
  };

  /// @brief Constructor for the Tracepoints class.
  ///
  /// Initializes a new instance of the Tracepoints class, setting up the
  /// initial state for managing tracepoints and their associated string data.
  /// The constructor initializes the unique ID generator for tracepoints to 1,
  /// which is primarily used to generate correlation and other monotonically
  /// increasing and unique IDs in the framework.
  ///
  /// @param st A reference to an existing StringTable instance. This string
  ///           table is used for storing string data associated with
  ///           tracepoints, such as names or descriptions and the same string
  ///           table reference is shared by all aspects of the framework.

  Tracepoints(xpti::StringTable &st)
      : MUId(1), MStringTableRef(st), MInsertions(0), MRetrievals(0) {
    // Nothing requires to be done at construction time
  }

  /// @brief Destructor for the Tracepoints class.
  ///
  /// Cleans up the resources used by the Tracepoints instance. This includes
  /// clearing any internal data structures that were used to manage tracepoints
  /// and their associated string data. The `clear` method is called to ensure
  /// that all allocated memory is properly released and that the Tracepoints
  /// instance is left in a clean state before destruction.

  ~Tracepoints() { clear(); }

  /// @brief Resets the Tracepoints instance to its initial state.
  ///
  /// This method clears all internal data structures of the Tracepoints
  /// instance, including the string table reference (MStringTableRef), payloads
  /// (MPayloads), and events (MEvents). It also resets the unique identifier
  /// (MUId) for tracepoints to 1. The counters for recording insertions
  /// (MInsertions) and retrievals (MRetrievals) are reset to 0. This method is
  /// typically called to prepare the Tracepoints instance for reuse or to
  /// ensure a clean state before destruction.

  void clear() {
    MStringTableRef.clear();
    // We will always start our ID
    // stream from 1. 0 is null_id
    // and -1 is invalid_id
    MUId = 1;
    MInsertions = MRetrievals = 0;
    MPayloads.clear();
    MTracepoints.clear();
  }

  /// @brief Generates a unique 64-bit identifier for use in the framwork.
  ///
  /// This function increments the internal counter (MUId) and returns its new
  /// value, providing a unique monotonically increasing 64-bit identifier for
  /// each call. This method is primarily used to generate uniue 64-bit IDs,
  /// especially needed for correlating scoped trace calls, such as
  /// `function_begin` and `function_end`.
  ///
  /// @return A unique 64-bit unsigned integer identifier.
  ///
  inline uint64_t makeUniqueID() { return MUId++; }

  /// @brief Creates a trace event based on the provided payload.
  ///
  /// It takes a payload and an
  /// optional instance value pointer as a return value as input. The payload
  /// contains the data associated with the event to be traced, such as the
  /// source file name, line number, and column number which is used to create a
  /// unique 128-bit Universal ID that will be associated with the trace event.
  /// If an instance number return value is provided, it will contain the
  /// specific instance of a given payload and is also associated with the trace
  /// event. The function returns a pointer to the newly created trace event
  /// data structure.
  ///
  ///  At the end of the function, the following tasks will be completed:
  ///  1. Create a UID for the payload (and cache it for legacy API)
  ///  2. Create a mapping from Universal ID <--> Payload
  ///  3. Create a mapping from Universal ID <--> Event
  ///
  /// @param Payload A pointer to the payload data for the event. The payload
  ///                includes information such as the source file name, line
  ///                number, and column number.
  /// @param InstanceNo A pointer to a uint64_t that holds the instance number
  ///                   of the created event.
  /// @return A pointer to the newly created trace event data structure.
  ///

  xpti::trace_event_data_t *create(const xpti::payload_t *Payload,
                                   uint64_t *InstanceNo) {
    return register_event(Payload, InstanceNo);
  }

  /// @brief Retrieves the payload data associated with a given trace event.
  ///
  /// This function attempts to retrieve the payload data for a specified trace
  /// event. The payload contains detailed information about the event, such as
  /// the source location where the event was triggered. If the event or its
  /// universal ID is invalid, the function returns `nullptr`.
  ///
  /// The function first checks if the payload is already cached in the event's
  /// reserved field. If not, it attempts to find the payload in the global
  /// payload map (`MPayloads`) using the event's universal ID. If found, the
  /// payload is cached in the event for future access. This caching mechanism
  /// avoids repeated lookups in the global map, improving performance.
  ///
  /// The access to the global payload map is protected by a shared mutex
  /// (`MPayloadMutex`), ensuring thread-safe read access. This is particularly
  /// important in a multi-threaded environment where trace events can be
  /// created and queried concurrently.
  ///
  /// @param Event A pointer to the trace event data structure whose payload is
  ///              to be retrieved.
  /// @return A pointer to the payload data associated with the event, or
  ///         `nullptr` if the event is invalid or the payload cannot be found.

  const xpti::payload_t *payloadData(xpti::trace_event_data_t *Event) {
    // Requires it to not be a null pointer and have the UID and payload
    // information set for the event
    if (!xpti::is_valid_event(Event))
      return nullptr;

    // If the payload is cahced already in the event, return it immedietely
    if (Event->reserved.payload)
      return Event->reserved.payload;
    else {
      // Scoped lock until the information is retrieved from the map
      std::shared_lock Lock(MPayloadMutex);
      auto &PayloadEntry = MPayloads[Event->universal_id];
      xpti::payload_t *Payload = &PayloadEntry.first;
      // Cache the payload information if it hasn't already been cached
      if (!Event->reserved.payload) {
        Event->reserved.payload = Payload;
        Event->flags |=
            static_cast<uint64_t>(xpti::trace_event_flag_t::PayloadAvailable);
      }
      return Payload;
    }
  }

  /// @brief Retrieves the payload data associated with a given universal ID.
  ///
  /// This function looks up the payload data corresponding to a specific
  /// universal ID (UId) in the global payload map. The universal ID is a unique
  /// identifier for trace events and their associated data. If the universal ID
  /// is valid and exists in the map, the function returns a pointer to the
  /// payload data. Otherwise, it returns `nullptr`.
  ///
  /// The function first checks if the provided universal ID pointer is
  /// `nullptr` or if the universal ID is invalid. If either condition is true,
  /// the function immediately returns `nullptr`.
  ///
  /// If the universal ID is valid, the function then acquires a shared lock on
  /// the global payload map (`MPayloadMutex`) to ensure thread-safe read
  /// access. This is crucial in a multi-threaded environment where concurrent
  /// access to the payload map can occur for read-access and insertions. If the
  /// payload is found, a pointer to the payload data is returned.
  ///
  /// Note: The shared lock is scoped and automatically released when the lock
  /// goes out of scope, ensuring that the lock is held only for the duration of
  /// the map access.
  ///
  /// @param UId A pointer to the universal ID for which the payload data is to
  ///            be retrieved.
  /// @return A pointer to the payload data associated with the given universal
  ///         ID, or `nullptr` if the universal ID is invalid or not associated
  ///         with a payload in the map.

  const xpti::payload_t *payloadDataByUniversalID(xpti::uid128_t *UId) {
    if (!UId || xpti::is_valid_uid(*UId) == false)
      return nullptr;
    // Scoped lock until the information is retrieved from the map
    {
      xpti::TracepointInstance *TP =
          reinterpret_cast<xpti::TracepointInstance *>(UId->uid64);
      if (xpti::is_valid_payload(&TP->MPayload))
        return &TP->MPayload;
      else
        return nullptr;
      // std::shared_lock Lock(MPayloadMutex);
      // auto &PayloadEntry = MPayloads[*UId];
      // return &PayloadEntry.first;
    }
  }

  /// @brief Looks up and retrieves the event data associated with a given
  /// universal ID.
  ///
  /// This function searches for the trace event data corresponding to a
  /// specific universal ID (UId) within the global event map. The universal ID
  /// uniquely identifies a trace event and consists of a high and low part,
  /// along with an instance number. If the event data for the given universal
  /// ID is found, a pointer to the event data structure is returned. Otherwise,
  /// the function returns `nullptr`.
  ///
  /// The search process involves several steps:
  /// 1. The function first checks if the provided universal ID pointer is
  ///    `nullptr` or if the universal ID is considered invalid according to the
  ///    `xpti::is_valid_uid` function. If either condition is true, the
  ///    function immediately returns `nullptr`.
  /// 2. A shared lock on the global event map (`MTracepointMutex`) is acquired
  /// to
  ///    ensure thread-safe read access.
  /// 3. The function then checks if the global event map contains an entry for
  ///    the high and low parts of the universal ID. If an entry exists, it
  ///    further checks if there is a sub-entry for the instance number of the
  ///    universal ID.
  /// 4. If both the main entry and the sub-entry for the instance number are
  ///    found, a pointer to the corresponding event data structure is returned.
  ///
  /// @param UId A pointer to the universal ID for which the event data is to be
  ///            retrieved.
  /// @return A pointer to the event data associated with the given universal
  ///         ID, or `nullptr` if the universal ID is invalid, not found in the
  ///         map, or there is no entry for the instance number.
  ///
  const xpti::trace_event_data_t *lookupEventData(xpti::uid128_t *UId) {
    if (!UId || !xpti::is_valid_uid(*UId))
      return nullptr;

    xpti::TracepointInstance *TP =
        reinterpret_cast<xpti::TracepointInstance *>(UId->uid64);
    if (xpti::is_valid_event(&TP->MEvent))
      return &TP->MEvent;
    else
      return nullptr;
  }

  /// @brief Releases a trace event and its associated resources.
  ///
  /// This function is responsible for releasing a trace event specified by the
  /// `Event` parameter. It ensures that the event and its resources are
  /// properly cleaned up from the internal data structures of the trace
  /// framework. The function performs several checks and operations:
  ///
  /// 1. It first checks if the `Event` pointer is `nullptr` or if the universal
  ///    ID (`universal_id`) associated with the event is not valid. If either
  ///    condition is true, the function returns immediately without performing
  ///    any cleanup.
  ///
  /// 2. It acquires a unique lock on the global event map mutex
  /// (`MTracepointMutex`)
  ///    to ensure thread-safe access to the `MEvents` map, which stores the
  ///    events indexed by their universal IDs.
  ///
  /// 3. The function then attempts to find the list of events associated with
  ///    the given universal ID in the `MEvents` map. If the event is found and
  ///    its instance exists within the list, the event is erased from the list.
  ///
  /// 4. If the event is successfully released (erased from the list), the
  ///    function also attempts to erase the 64-bit universal ID from the global
  ///    UID lookup table
  ///    (`MUidLut64x128`). This step is crucial for maintaining the integrity
  ///    of the UID lookup mechanism and ensuring that stale or released UIDs
  ///    are not accessible.
  ///
  /// Note: The function does not release the payload associated with the event,
  /// even if there are no more events associated with the UID. This decision is
  /// made to keep the payload available for potential future use, as the same
  /// payload may be revisited.
  ///
  /// @param Event A pointer to the trace event data structure that is to be
  ///              released.
  ///
  ///

  void releaseEvent(xpti::trace_event_data_t *Event) {
    if (!Event || !xpti::is_valid_uid(Event->universal_id))
      return;

    xpti::uid128_t UId = Event->universal_id;
    {
      std::unique_lock Lock(MTracepointMutex);
      // Find the event list for a given UID
      auto &Instances = MTracepoints[UId];
      MUID64Check.erase(UId.uid64);
      // Now release the event associated with the UID instance
      Instances.erase(UId.instance);
      // If there are no more events associated with the UID, we can release
      // the Payload as well, but we will not as the same payload may be
      // revisited and we need to keep the instance count going
    }
  }

  /// @brief Adds metadata to a trace event.
  ///
  /// This function associates a key-value pair as metadata with a given trace
  /// event. The key is a string, and the value is an object identifier
  /// (object_id_t). If the key already exists in the event's metadata, the
  /// value is overwritten, and XPTI_RESULT_DUPLICATE is returned. Otherwise,
  /// the key-value pair is added, and a success result is returned.
  ///
  /// The function first checks if the `Event` pointer or the `Key` pointer is
  /// `nullptr`. If either is `nullptr`, the function returns an invalid
  /// argument result.
  ///
  /// The key is then added to the global string table (`MStringTableRef`), and
  /// its string identifier
  /// (`string_id_t`) is retrieved. If the key is invalid and cannot be added to
  /// the string table, XPTI_RESULT_INVALIDARG is returned.
  ///
  /// The function uses a mutex (`MMetadataMutex`) to protect the metadata table
  /// from simultaneous insert operations, ensuring thread safety. It checks if
  /// the key already exists in the event's metadata. If it does, a duplicate
  /// result is returned, but the value is still updated. If the key does not
  /// exist, it is added along with its value, and XPTI_RESULT_SUCCESS is
  /// returned.
  ///
  /// @param Event A pointer to the trace event data structure to which the
  ///              metadata is to be added.
  /// @param Key A pointer to the string that represents the key in the
  ///              key-value pair.
  /// @param ValueID The object identifier that represents the value in the
  ///                key-value pair.
  /// @return A result code indicating the outcome of the operation. Possible
  ///         values are `xpti::result_t::XPTI_RESULT_INVALIDARG` if either
  ///         `Event` or `Key` is `nullptr`, or the key is invalid;
  ///         `xpti::result_t::XPTI_RESULT_DUPLICATE` if the key already exists
  ///         in the metadata; `xpti::result_t::XPTI_RESULT_SUCCESS` if the
  ///         key-value pair is successfully added.

  xpti::result_t addMetadata(xpti::trace_event_data_t *Event, const char *Key,
                             object_id_t ValueID) {
    if (!Event || !Key)
      return xpti::result_t::XPTI_RESULT_INVALIDARG;

    string_id_t KeyID = MStringTableRef.add(Key);
    if (KeyID == xpti::invalid_id) {
      return xpti::result_t::XPTI_RESULT_INVALIDARG;
    }

    // Protect simultaneous insert operations on the metadata tables
    {
      xpti::result_t res;
      std::lock_guard<std::mutex> HashLock(MMetadataMutex);
      if (Event->reserved.metadata.count(KeyID)) {
        // One already existed, but we overwrote it
        res = xpti::result_t::XPTI_RESULT_DUPLICATE;
      } else {
        res = xpti::result_t::XPTI_RESULT_SUCCESS;
      }
      Event->reserved.metadata[KeyID] = ValueID;
      return res;
    }
  }

  // Method to get the access statistics of the tracepoints.
  // It will print the number of insertions vs lookups that were
  // performed.
  //
  void printStatistics() {
#ifdef XPTI_STATISTICS
    std::cout << "Tracepoint inserts : " << MInsertions.load() << '\n';
    std::cout << "Tracepoint lookups : " << MRetrievals.load() << '\n';
    std::cout << "Tracepoint Hashmap :\n";
    MPayloadLUT.printStatistics();
#endif
  }

  /// @brief Generates a unique key from a given payload, updates  the incoming
  /// payload and returns the payload and UID associated with the payload.
  ///
  /// This function is designed to generate a unique identifier (UID) for a
  /// given payload and manage the association of this UID with the payload in a
  /// global payload map. It ensures that each unique payload is associated with
  /// a unique UID and manages instances of payloads that are not unique but
  /// share the same properties (e.g., name, source file, line number, and
  /// column number).
  ///
  /// The process involves several key steps:
  /// - First, it checks the validity of the payload. If the payload is invalid
  ///   (either being `nullptr` or failing the `xpti::is_valid_payload` check),
  ///   the function returns a `PayloadInstance` with an empty UID and `nullptr`
  ///   for the payload.
  /// - If the payload is valid, it uses the payload properties to generate a
  ///   UID through the `makeUniversalId` function.
  /// - With the generated UID, the function then locks the global payload map
  ///   (`MPayloads`) to ensure thread-safe access and attempts to find or
  ///   create an entry for the UID.
  /// - If the UID is new (indicated by a flags value of 0 in the payload
  ///   entry), the function updates the payload information in the map, sets
  ///   the instance number to 1, and marks the payload's 64-bit UID as
  ///   xpti::invalid_uid which is stored in the payload's `internal` attribute.
  /// - If the UID already exists in the map, it increments the instance number
  ///   for this UID, indicating another occurrence of the same payload.
  ///
  /// The function ultimately returns a `PayloadInstance` containing the UID
  /// (with the instance number set appropriately) and a pointer to the payload
  /// associated with this UID in the global map.
  ///
  /// @param Payload A pointer to the payload for which a unique key and
  ///                instance are to be generated.
  /// @return A `PayloadInstance` structure containing the generated UID and a
  ///         pointer to the payload associated with this UID in the global
  ///         payload map.

  PayloadInstance makeKeyFromPayload(xpti::payload_t *Payload) {
    xpti::uid128_t Key;
    // Validity of the payload structure is checked in the previous layer
    if (!Payload && !xpti::is_valid_payload(Payload))
      return PayloadInstance{Key, nullptr};

    // If not, we will use the name, source file, line number and column number
    // to create the key without the instance number
    Key = makeUniversalId(Payload);
    // Check is Key is valid; If the payload is fully populated, then we will
    // have both Key.p1 and Key.p2 set. However, if only a function name is
    // provided, then we will have Key.p1 populated.
    std::unique_lock Lock(MPayloadMutex);
    auto &PayloadEntry = MPayloads[Key];
    if (PayloadEntry.first.flags == 0) {
#ifdef XPTI_STATISTICS
      MInsertions++;
#endif
      // We are seeing this UID for the first time, so we can update the
      // Payload information and set the instance to 1
      PayloadEntry.first = *Payload;
      PayloadEntry.first.uid.p3 = 1;
      PayloadEntry.first.internal = xpti::invalid_uid;
      Key.instance = PayloadEntry.second = 1;
      PayloadEntry.first.flags |=
          static_cast<uint64_t>(xpti::payload_flag_t::PayloadRegistered);
      Payload->flags |=
          static_cast<uint64_t>(xpti::payload_flag_t::PayloadRegistered);
    } else {
      // Since we have seen this Payload before, let's increment the instance
      Key.instance = ++PayloadEntry.second;
    }
    // Now, we need to create the actually payload for this instance that we
    // will be passing back to the caller
    return PayloadInstance{Key, &PayloadEntry.first};
  }

  /// @brief Registers a new tracepoint with the given payload.
  ///
  /// This function is responsible for registering a new tracepoint based on the
  /// provided payload. It generates a unique identifier for the tracepoint,
  /// checks its validity, and then updates the tracepoint instance with the
  /// payload information. Additionally, it sets various flags and inserts the
  /// 64-bit ID for validation purposes when legacy APIs are used.
  ///
  /// @param Payload Pointer to the payload that needs to be registered as a
  ///                tracepoint instance.
  /// @return Pointer to the newly created TracepointInstance; returns nullptr
  ///         if the universal ID computed from the Payload is invalid.
  ///
  xpti::TracepointInstance *registerTracepoint(xpti::payload_t *Payload) {
    // Generate a unique key from the payload which includes a universal ID and
    // invariant payload; If the payload has been visited multiple times, the ID
    // will have the same information for id.p1 and id.p2, but the instance
    // value will be different
    auto [UniversalId, InvarPayload] = makeKeyFromPayload(Payload);
    // Check if the generated universal ID is valid
    if (!xpti::is_valid_uid(UniversalId))
      return nullptr;

    // Lock the mutex to ensure thread-safe access to the tracepoints map
    std::unique_lock Lock(MTracepointMutex);
    // Access or create the tracepoint instance associated with the universal ID
    auto &Tracepoint = MTracepoints[UniversalId];
    // Access or create the specific instance of the tracepoint based on the
    // universal ID instance
    auto &TP = Tracepoint[UniversalId.instance];
#ifdef XPTI_STATISTICS
    MInsertions++;
#endif
    // Update the tracepoint instance with the universal ID and payload
    TP.Update(UniversalId, InvarPayload);
    // Set various internal IDs and flags for the tracepoint and payload
    Payload->internal = TP.MEvent.unique_id = TP.MPayload.internal =
        TP.MUId.uid64 = TP.MEvent.universal_id.uid64 = (uint64_t)&TP;
    TP.MPayload.flags |=
        static_cast<uint64_t>(xpti::payload_flag_t::HashAvailable);
    TP.MEvent.flags |=
        static_cast<uint64_t>(xpti::trace_event_flag_t::HashAvailable);

    // Add the 64-bit ID for validation purposes when legacy API are used
    MUID64Check.insert(TP.MUId.uid64);
    // Return a pointer to the updated tracepoint instance which contains the
    // UID with the correct instance number, the payload with updated flags and
    // fields and the event that represents the tracepoint
    return &TP;
  }

  /// @brief Generates the universal unique identifier (UID) for a given
  /// payload.
  ///
  /// This function is responsible for generating a 128-bit universal unique
  /// identifier (UID) for a given payload. The UID is generated based on the
  /// payload's properties such as the function name, source file, line
  /// number, and column number. If the payload is valid and has not already
  /// been assigned a UID, this function computes and assigns a new UID to the
  /// payload.
  ///
  /// The process involves the following steps:
  /// - First, it checks if the payload pointer is `nullptr` or if the payload
  ///   is invalid by calling `xpti::is_valid_payload`. If the payload is
  ///   invalid, the function returns an empty UID.
  /// - It then clears the `HashAvailable` flag in the payload's flags to
  ///   indicate that the 64-bit UID needs to be generated.
  /// - If the payload's function name is available (indicated by the
  ///   `NameAvailable` flag), the function name is added to a global string
  ///   table, and its unique identifier is obtained.
  /// - Similarly, if the payload's source file is available (indicated by the
  ///   `SourceFileAvailable` flag), the source file information is added to
  ///   the global string table, and its unique identifier is obtained. The
  ///   line number and column number are also retrieved from the payload.
  /// - Using the identifiers for the function name and source file, along
  /// with
  ///   the line number and column number, a 128-bit UID is generated.
  /// - The sting ids and line number is then used to update the payload's UID
  ///   fields (`uid.p1` and `uid.p2`), which will remain invariant and is for
  ///   use by legacy API.
  /// - The instance number and `uid64` part of the UID remain unset and
  /// invalid
  ///   at this point.
  ///
  /// @param Payload A pointer to the payload for which the UID is to be
  ///                generated.
  /// @return A 128-bit UID uniquely identifying the given payload based on
  /// its
  ///         properties. If the payload is invalid, an empty UID is returned.
  ///

  xpti::uid128_t makeUniversalId(xpti::payload_t *Payload) {
    xpti::uid128_t UId;
    if (!Payload || !xpti::is_valid_payload(Payload))
      return UId;

    if (Payload->internal == xpti::invalid_uid) {
      // If the uid has not been generated and cached, update the flag to say
      // so
      Payload->flags &=
          (~static_cast<uint64_t>(xpti::payload_flag_t::HashAvailable));
    }
    uint64_t FileId = 0, FuncId = 0;
    int LineNo = 0, ColNo = 0;

    // If the payload's function name is available, add it to the string table
    // and get its id
    if ((Payload->flags &
         static_cast<uint64_t>(xpti::payload_flag_t::NameAvailable))) {
      // Add the kernel name/function name to the string table
      FuncId = MStringTableRef.add(Payload->name, &Payload->name);
    }

    // If the payload's source file is available, add it to the string table
    // and get its id Also, get the line number and column number from the
    // payload
    if ((Payload->flags &
         static_cast<uint64_t>(xpti::payload_flag_t::SourceFileAvailable))) {
      // Add source file information ot string table
      FileId = MStringTableRef.add(Payload->source_file, &Payload->source_file);
      LineNo = Payload->line_no;
      ColNo = Payload->column_no;
    }

    UId = xpti::make_uid128(FileId, FuncId, LineNo, ColNo);
    // Update the fields of Payload that will remain invariant and is for use
    // by legacy API that deals with 64-bit universal IDs
    Payload->uid.p1 = XPTI_PACK32_RET64(FileId, LineNo);
    Payload->uid.p2 = XPTI_PACK32_RET64(0, FuncId);
    // UId.instance and UId.uid64 are still set to 0 and invalid

    return UId;
  }

  /// @brief Checks the validity of a 64-bit unique identifier (UID).
  ///
  /// This function determines if the provided 64-bit UID exists within a
  /// maintained set of UIDs. The presence of the UID in the set indicates its
  /// validity. This is typically used to verify if a given UID has been
  /// previously registered or is known to the system.
  ///
  /// @param UId The 64-bit unique identifier to be checked for validity.
  /// @return Returns true if the UID exists in the set, indicating it is valid;
  ///         otherwise, returns false.
  ///
  bool isValidUID64(uint64_t UId) { return (MUID64Check.count(UId) > 0); }

private:
  /// @brief Registers an event with a given payload and returns the event
  /// data structure with its instance number.
  ///
  /// This function is responsible for registering an event based on the
  /// provided payload. It generates a universal unique identifier (UID) for
  /// the payload, creates a new trace event data structure, associates it
  /// with the generated UID and its instance, and then returns the event data
  /// structure. The function ensures thread safety by using a mutex lock
  /// during the event registration process to handle concurrent registrations
  /// of the same payload from multiple threads.
  ///
  /// The process involves the following steps:
  /// 1. A temporary copy of the payload is created to avoid modifying the
  ///    original payload and avoid overwriting fields in the incoming payload
  ///    in a parallel execution scenario.
  /// 2. A universal ID (UID) is generated based on the payload information
  ///    using `makeKeyFromPayload`.
  /// 3. If the generated UID is valid, the function proceeds to register the
  ///    event; otherwise, it returns `nullptr`.
  /// 4. The function locks the event map (`MEvents`) to ensure thread-safe
  ///    access.
  /// 5. A new trace event data structure is created and associated with the
  ///    generated UID and its instance number.
  /// 6. The event's `universal_id` is set to the generated UID, and the
  ///    `UIDAvailable` flag is set.
  /// 7. The instance number of the event is updated, and the
  /// `PayloadAvailable`
  ///    flag is set.
  /// 8. The payload associated with the UID is stored in the event's
  ///    `reserved.payload`.
  /// 9. The function returns a pointer to the newly created trace event data
  ///    structure and sets the InstanceNo pointer to the instance number
  ///
  /// @param Payload A pointer to the payload for which the event is to be
  ///                registered.
  /// @param InstanceNo A pointer to a uint64_t variable where the instance
  ///                   number of the event will be stored.
  /// @return A pointer to the registered trace event data structure, or
  ///         `nullptr` if the UID generated from the payload is not valid.
  ///

  xpti::trace_event_data_t *register_event(const xpti::payload_t *Payload,
                                           uint64_t *InstanceNo) {
    xpti::payload_t TempPayload = *Payload;

    /// Initialize to invalid
    /// We need an explicit lock for the rest of the operations as the same
    /// payload could be registered from multiple-threads.
    ///
    /// 1. Create a Universal ID based on the payload information
    /// 2. Using the generated UID, create a new trace event data structure
    /// and
    ///    associate it with the new UID and it's instance.
    /// 3. Return the event

    auto TracepointInstance = registerTracepoint(&TempPayload);
    *InstanceNo = TracepointInstance->MUId.instance;
    return TracepointInstance->Event();
  }

  /// @var xpti::safe_int64_t MUId
  /// @brief Monotonically increasing unique identifier for trace events.
  ///
  /// This variable is used to generate unique identifiers for trace events in
  /// a thread-safe manner. The `safe_int64_t` type ensures atomic operations
  /// on the identifier, preventing race conditions in a multi-threaded
  /// environment.
  xpti::safe_int64_t MUId;

  /// @var xpti::StringTable& MStringTableRef
  /// @brief Reference to a global string table used for string internment.
  ///
  /// This reference is used to access a global string table where all strings
  /// (e.g., function names, file names) are stored. String internment helps
  /// in reducing memory usage and improving comparison efficiency by ensuring
  /// that each unique string is stored only once.
  xpti::StringTable &MStringTableRef;

  /// @var xpti::safe_uint64_t MInsertions, MRetrievals
  /// @brief Counter for the number of payload insertions and retrievals from
  /// the payload lookup table. These are used only when the statictics are
  /// enabled with XPTI_STATISTICS=1 at compile time
  xpti::safe_uint64_t MInsertions, MRetrievals;

  /// @var uid_payload_lut MPayloads
  /// @brief Lookup table mapping unique identifiers to payload instances.
  ///
  /// This lookup table stores the association between unique identifiers
  /// (UIDs) and their corresponding payload instances. It enables efficient
  /// retrieval of payload information based on UIDs.
  uid_payload_lut MPayloads;

  /// @var uid_tracepoints_lut MTracepoints
  /// @brief Lookup table for managing tracepoints.
  ///
  /// This table maps unique identifiers (UIDs) to their corresponding
  /// tracepoint instances. It is used to efficiently find and manage
  /// tracepoints based on their UIDs throughout the trace framework. The exact
  /// structure of the UID to tracepoint mapping is defined by the
  /// `uid_tracepoints_lut` type.
  uid_tracepoints_lut MTracepoints;

  /// @var uid64_validity_lut MUID64Check
  /// @brief Set for validating 64-bit unique identifiers (UIDs).
  ///
  /// This set contains 64-bit UIDs that have been registered or are known to be
  /// valid within the trace framework. It is used to quickly check the validity
  /// of a 64-bit UID by seeing if it exists within this set. The
  /// `uid64_validity_lut` type defines the structure of this validation set.
  uid64_validity_lut MUID64Check;

  /// @var std::mutex MMetadataMutex
  /// @brief Mutex for protecting access to metadata.
  ///
  /// This mutex is used to synchronize access to metadata, ensuring
  /// thread-safe operations when modifying or accessing shared metadata
  /// information.
  std::mutex MMetadataMutex;

  /// @var mutable std::shared_mutex MTracepointMutex
  /// @brief Shared mutex for protecting access to the event lookup table.
  ///
  /// This shared mutex allows multiple readers or a single writer to access
  /// the `MEvents` lookup table, ensuring thread-safe operations. The
  /// `mutable` keyword allows the mutex to be locked even in const member
  /// functions.
  mutable std::shared_mutex MTracepointMutex;

  /// @var mutable std::shared_mutex MPayloadMutex
  /// @brief Reader/Writer mutex for protecting access to the payload lookup
  /// table.
  ///
  /// Similar to `MTracepointMutex`, this shared mutex ensures thread-safe
  /// access to the `MPayloads` lookup table, allowing multiple readers or a
  /// single writer.
  mutable std::shared_mutex MPayloadMutex;
};

/// @brief Helper class to manage subscriber callbacks for a given tracepoint
/// @details This class provides a thread-safe way to register and unregister
/// callbacks for a given stream. This will be used by tool plugins.
///
/// The class also provided a way to notify registered callbacks for a given
/// stream and trace point type. This will be used by framework to trigger
/// notifications are instrumentation points.
class Notifications {
public:
  /// @typedef cb_entry_t
  /// @brief Defines a callback entry as a pair consisting of a boolean and a
  /// tracepoint callback function.
  /// @details The boolean value indicates whether the callback is enabled
  /// (true) or disabled (false). The tracepoint callback function is defined
  /// by the xpti::tracepoint_callback_api_t type.
  using cb_entry_t = std::pair<bool, xpti::tracepoint_callback_api_t>;

  /// @typedef cb_entries_t
  /// @brief Represents a collection of callback entries.
  /// @details This is a vector of cb_entry_t, allowing for multiple callbacks
  /// to be associated with a single event or tracepoint.
  using cb_entries_t = std::vector<cb_entry_t>;

  /// @typedef cb_t
  /// @brief Maps a trace type to its associated callback entries.
  /// @details This unordered map uses a uint16_t as the key to represent the
  /// trace point type, and cb_entries_t to store the associated callbacks.
  using cb_t = std::unordered_map<uint16_t, cb_entries_t>;

  /// @typedef stream_cb_t
  /// @brief Maps a stream ID to its corresponding callbacks for different
  /// trace types
  /// @details This unordered map uses a uint16_t as the key for the stream
  /// ID, and cb_t to map the stream to registered callbacks for each trace
  /// type
  using stream_cb_t = std::unordered_map<uint16_t, cb_t>;

  /// @typedef statistics_t
  /// @brief Keeps track of statistics, typically counts, associated with
  /// different framework operations.
  /// @details This unordered map uses a uint16_t as the key for the tracking
  /// the type of statistical data and usually not defined by default. To
  /// enable it, XPTI_STATISTICS has to be defined while compiling the
  /// frmaework library.
  using statistics_t = std::unordered_map<uint16_t, uint64_t>;
  /// @typedef trace_flags_t
  /// @brief Maps an trace type to a boolean flag indicating its state.
  /// @details This unordered map uses a uint16_t as the key for the trace
  /// type, and a boolean value to indicate whether callbacks are registered
  /// for this trace type (e.g., registered or unregisterted/no callback).
  using trace_flags_t = std::unordered_map<uint16_t, bool>;

  /// @typedef stream_flags_t
  /// @brief Maps a stream ID to its corresponding trace flags for different
  /// trace point types.
  /// @details This unordered map uses a uint8_t as the key for trace type,
  /// and trace_flags_t to map the trace type to their boolean that indiciates
  /// whether a callback has been registered for this trace type in the given
  /// stream.
  using stream_flags_t = std::unordered_map<uint8_t, trace_flags_t>;

  /// @brief Registers a callback function for a specific trace type and stream
  /// ID.
  ///
  /// This function is responsible for registering a callback function that will
  /// be invoked for a specified trace type and stream ID. It ensures that the
  /// callback is not already registered to prevent duplicates. If the callback
  /// has been previously unregistered, it will be re-enabled.
  ///
  /// @param StreamID The unique identifier of the stream for which the callback
  ///                 is being registered.
  /// @param TraceType The trace type for which the callback is being
  ///                  registered. This corresponds to specific events or
  ///                  actions that the callback is interested in.
  /// @param cbFunc The callback function to be registered. This is a pointer to
  ///               the function that will be called for the given trace type
  ///               and stream ID.
  ///
  /// @return Returns `xpti::result_t::XPTI_RESULT_SUCCESS` if the callback is
  ///         successfully registered.
  ///         Returns `xpti::result_t::XPTI_RESULT_INVALIDARG` if the callback
  ///         function pointer is null.
  ///         Returns `xpti::result_t::XPTI_RESULT_DUPLICATE` if the callback is
  ///         already registered and active.
  ///         Returns `xpti::result_t::XPTI_RESULT_UNDELETE` if the callback was
  ///         previously unregistered and is now re-enabled.
  ///

  xpti::result_t registerCallback(uint8_t StreamID, uint16_t TraceType,
                                  xpti::tracepoint_callback_api_t cbFunc) {
    if (!cbFunc)
      return xpti::result_t::XPTI_RESULT_INVALIDARG;

#ifdef XPTI_STATISTICS
    //  Initialize first encountered trace
    //  type statistics counters
    {
      std::lock_guard<std::mutex> Lock(MStatsLock);
      auto InstanceNo = MStats.find(TraceType);
      if (InstanceNo == MStats.end()) {
        MStats[TraceType] = 0;
      }
    }
#endif
    // If reader-writer locks were emplyed, this is where the writer lock can
    // be used
    std::unique_lock Lock(MCBsLock);
    auto &TraceFlags = MStreamFlags[StreamID]; // Get the trace flags for the
                                               // stream ID
    TraceFlags[TraceType] = true; // Set the trace type flag to true

    auto &StreamCBs =
        MCallbacksByStream[StreamID]; // thread-safe
                                      // What we get is a concurrent_hash_map
                                      // of vectors holding the callbacks we
                                      // need access to;
    auto Acc = StreamCBs.find(TraceType);
    if (Acc == StreamCBs.end()) {
      // Create a new slot and return the accessor for the trace type
      auto Tmp = StreamCBs[TraceType];
      Acc = StreamCBs.find(TraceType);
    }
    // If the key does not exist, a new entry is created and an accessor to it
    // is returned. If it exists, we have access to the previous entry.
    //
    // Before we add this element, we scan all existing elements to see if it
    // has already been registered. If so, we return XPTI_RESULT_DUPLICATE.
    //
    // If not, we set the first element of new entry to 'true' indicating that
    // it is valid. Unregister will just set this flag to false, indicating
    // that it is no longer valid and is unregistered.
    for (auto &Ele : Acc->second) {
      if (Ele.second == cbFunc) {
        if (Ele.first) // Already here and active
          return xpti::result_t::XPTI_RESULT_DUPLICATE;
        else { // it has been unregistered before, re-enable
          Ele.first = true;
          return xpti::result_t::XPTI_RESULT_UNDELETE;
        }
      }
    }
    // If we come here, then we did not find the callback being registered
    // already in the framework. So, we insert it.
    Acc->second.push_back(std::make_pair(true, cbFunc));
    return xpti::result_t::XPTI_RESULT_SUCCESS;
  }

  /// @brief Unregisters a callback function for a specific trace type and
  /// stream ID.
  ///
  /// This function is designed to unregister (disable) a callback function that
  /// was previously registered for a specific trace type and stream ID. It does
  /// not physically remove the callback function from the internal storage but
  /// marks it as inactive. This approach is chosen to avoid deletion and
  /// simultaneous iterations by other threads, which could be unsafe.
  ///
  /// @param StreamID The unique identifier of the stream for which the callback
  ///                 is registered.
  /// @param TraceType The trace type for which the callback is registered. This
  ///                  corresponds to specific events or actions that the
  ///                  callback is interested in.
  /// @param cbFunc The callback function to be unregistered. This is a pointer
  ///               to the function that was previously registered for the given
  ///               trace type and stream ID.
  ///
  /// @return Returns `xpti::result_t::XPTI_RESULT_SUCCESS` if the callback is
  ///         successfully unregistered. Returns
  ///         `xpti::result_t::XPTI_RESULT_INVALIDARG` if the callback function
  ///         pointer is null. Returns `xpti::result_t::XPTI_RESULT_DUPLICATE`
  ///         if the callback is already marked as inactive. Returns
  ///         `xpti::result_t::XPTI_RESULT_NOTFOUND` if the callback is not
  ///         found for the specified trace type and stream ID.

  xpti::result_t unregisterCallback(uint8_t StreamID, uint16_t TraceType,
                                    xpti::tracepoint_callback_api_t cbFunc) {
    if (!cbFunc)
      return xpti::result_t::XPTI_RESULT_INVALIDARG;

    // Since we do not remove the callback function when they are unregistered
    // and only reset the flag, the writer lock is not held for very long; use
    // writer lock here.
    std::unique_lock Lock(MCBsLock);
    auto &TraceFlags = MStreamFlags[StreamID]; // Get the trace flags for the
                                               // stream ID
    TraceFlags[TraceType] = false; // Set the trace type flag to false

    auto &StreamCBs =
        MCallbacksByStream[StreamID]; // thread-safe
                                      //  What we get is a concurrent_hash_map
                                      //  of vectors holding the callbacks we
                                      //  need access to;
    auto Acc = StreamCBs.find(TraceType);
    bool Success = (Acc != StreamCBs.end());
    if (Success) {
      for (auto &Ele : Acc->second) {
        if (Ele.second == cbFunc) {
          if (Ele.first) { // Already here and active
                           // unregister, since delete and simultaneous
                           // iterations by other threads are unsafe
            Ele.first = false;
            // releases the accessor
            return xpti::result_t::XPTI_RESULT_SUCCESS;
          } else {
            // releases the accessor
            return xpti::result_t::XPTI_RESULT_DUPLICATE;
          }
        }
      }
    }
    //  Not here, so nothing to unregister
    return xpti::result_t::XPTI_RESULT_NOTFOUND;
  }

  /// @brief Unregisters all callbacks associated with a given stream ID.
  ///
  /// This function is responsible for disabling all callbacks that have been
  /// registered for a specific stream, identified by its StreamID. It first
  /// checks if there are any callbacks registered for the given StreamID. If
  /// none are found, it returns a 'not found' result. Otherwise, it proceeds to
  /// erase the stream's trace flags and disable all its callbacks.
  ///
  /// @param StreamID The unique identifier of the stream whose callbacks are to
  ///                 be unregistered.
  /// @return Returns `xpti::result_t::XPTI_RESULT_SUCCESS` if the operation is
  ///         successful. Returns `xpti::result_t::XPTI_RESULT_NOTFOUND` if no
  ///         callbacks are registered for the specified StreamID.
  ///
  /// @note This function uses a `std::unique_lock` to ensure thread safety when
  /// modifying the callbacks and stream flags. If the implementation evolves to
  /// use reader-writer locks, a reader lock should be used where appropriate.

  xpti::result_t unregisterStream(uint8_t StreamID) {
    // If there are no callbacks registered for the requested stream ID, we
    // return not found; use reader lock here if the implementation moves to
    // reader-writer locks.
    std::unique_lock Lock(MCBsLock);
    if (MCallbacksByStream.count(StreamID) == 0)
      return xpti::result_t::XPTI_RESULT_NOTFOUND;

    // Get the trace flags for the stream
    MStreamFlags.erase(StreamID);

    auto &StreamCBs = MCallbacksByStream[StreamID]; // thread-safe
    // Disable all callbacks registered for the stream represented by StreamID
    for (auto &Item : StreamCBs) {
      for (auto &Ele : Item.second) {
        Ele.first = false;
      }
    }
    //  Return success
    return xpti::result_t::XPTI_RESULT_SUCCESS;
  }

  /// @brief Checks if a trace type is subscribed in a specific stream.
  ///
  /// This function determines whether a given trace type (event or
  /// tracepoint) is currently subscribed to by a callback function in a
  /// specific stream. Clients receive the data when they subscribe and
  /// instrumentation is allowed only if there are subscribers registered for
  /// the trace type. This function is now lock free and uses a shadow data
  /// structure which is updated when a callback is registered or
  /// unregistered.
  ///
  /// @param StreamID The unique identifier for the stream being queried.
  /// @param TraceType The unique identifier for the trace type being checked
  /// for subscription.
  /// @return Returns true if the trace type ihas subscribers in the stream;
  /// otherwise, returns false.

  bool checkSubscribed(uint16_t StreamID, uint16_t TraceType) {
    if (StreamID == 0)
      return false;

    // Instead of checking the MCallbacksByStream to see if there are
    // registered callbacks for a given stream/trace type query, we check
    // this against a shadow data structure that sets a boolean flag equals
    // TRUE if a callback is registered for a Stream/Trace Type combination
    // and false if the callback has been unregistered or if one is not
    // present. This will be a lock-free operation and if trace type is not
    // set or is going to be set simultaneously, we may miss an event if we
    // access it earlier than the the write operation.
    auto &StreamFlags = MStreamFlags[StreamID];
    // When it is required that a particular stream has at least one active
    // subscriber, the TraceType will be set to 0. In this case we scan the
    // booleans of all set TraceType active subscribers and bail on the first
    // occurrence of TRUE.
    if (TraceType == 0) {
      for (auto &e : StreamFlags) {
        if (e.second)
          return true;
      }
      return false;
    } else {
      // If a specific TraceType has to be examined, we returns tis boolean
      // value
      if (StreamFlags.count(TraceType) == 0)
        return false;
      return StreamFlags[TraceType];
    }
  }

  /// @brief Notifies all subscribers about an event occurrence.
  ///
  /// This function is responsible for notifying all registered subscribers
  /// (callback functions) about an event occurrence. It does this by iterating
  /// through all callbacks registered for the specified stream ID and trace
  /// type, and invoking them with the provided event data. The function ensures
  /// thread safety by copying the callbacks to a local vector under a lock, and
  /// then invoking these callbacks without holding the lock to avoid deadlocks
  /// and reduce lock contention.
  ///
  /// @param StreamID The unique identifier of the stream for which the event
  ///                 occurred.
  /// @param TraceType The type of the trace event that occurred.
  /// @param Parent Pointer to the parent event data structure, if any; nullptr
  ///               otherwise.
  /// @param Object Pointer to the event data structure that describes the
  ///               event.
  /// @param InstanceNo An instance number that can be used to identify the
  ///                   specific occurrence of the event.
  /// @param UserData A pointer to user-defined data that can be passed to the
  ///                 callback.
  ///
  /// @return Always returns `xpti::result_t::XPTI_RESULT_SUCCESS`.
  ///
  /// @note This function uses a `std::shared_lock` (when compiled with C++14 or
  /// later) to allow
  ///       multiple readers for the callback registration data, improving
  ///       concurrency. The lock is released before invoking the callbacks to
  ///       prevent deadlocks and reduce lock contention. In environments where
  ///       statistics gathering is enabled (via `XPTI_STATISTICS`), this
  ///       function also updates the event occurrence count in a thread-safe
  ///       manner.

  xpti::result_t notifySubscribers(uint16_t StreamID, uint16_t TraceType,
                                   xpti::trace_event_data_t *Parent,
                                   xpti::trace_event_data_t *Object,
                                   uint64_t InstanceNo, const void *UserData) {
    bool Success = false;
    xpti::Notifications::cb_t::iterator Acc;
    std::vector<xpti::tracepoint_callback_api_t> LocalCBs;
    {
      // Addresses bug reported against XPTI where the lock was held for the
      // entire duration of the notification calls; now the logic will grab
      // the notification functions when the lock is held and then releases
      // the lock before calling the notification functions. When using
      // reader-writer locks, use reader lock here.
      std::shared_lock Lock(MCBsLock);
      cb_t &Stream = MCallbacksByStream[StreamID]; // Thread-safe
      Acc = Stream.find(TraceType);
      Success = (Acc != Stream.end());
      if (Success) {
        // Go through all registered callbacks and copy them
        for (auto &Ele : Acc->second) {
          if (Ele.first)
            LocalCBs.push_back(Ele.second);
        }
      }
    }

    // Go through all local copies of the callbacks and invoke them
    for (auto &CB : LocalCBs) {
      (CB)(TraceType, Parent, Object, InstanceNo, UserData);
    }
#ifdef XPTI_STATISTICS
    auto &Counter = MStats[TraceType];
    {
      std::lock_guard<std::mutex> Lock(MStatsLock);
      Counter++;
    }
#endif
    return xpti::result_t::XPTI_RESULT_SUCCESS;
  }

  void printStatistics() {
#ifdef XPTI_STATISTICS
    printf("Notification statistics:\n");
    for (auto &s : MStats) {
      printf("%19s: [%lu] \n",
             stringify_trace_type((xpti_trace_point_type_t)s.first).c_str(),
             s.second);
    }
#endif
  }

  void clear() { MCallbacksByStream.clear(); }

private:
#ifdef XPTI_STATISTICS
  std::string stringify_trace_type(xpti::trace_point_type_t TraceType) {
    switch (TraceType) {
    case xpti::trace_point_type_t::graph_create:
      return "graph_create";
    case xpti::trace_point_type_t::node_create:
      return "node_create";
    case xpti::trace_point_type_t::edge_create:
      return "edge_create";
    case xpti::trace_point_type_t::region_begin:
      return "region_begin";
    case xpti::trace_point_type_t::region_end:
      return "region_end";
    case xpti::trace_point_type_t::task_begin:
      return "task_begin";
    case xpti::trace_point_type_t::task_end:
      return "task_end";
    case xpti::trace_point_type_t::barrier_begin:
      return "barrier_begin";
    case xpti::trace_point_type_t::barrier_end:
      return "barrier_end";
    case xpti::trace_point_type_t::lock_begin:
      return "lock_begin";
    case xpti::trace_point_type_t::lock_end:
      return "lock_end";
    case xpti::trace_point_type_t::signal:
      return "signal";
    case xpti::trace_point_type_t::transfer_begin:
      return "transfer_begin";
    case xpti::trace_point_type_t::transfer_end:
      return "transfer_end";
    case xpti::trace_point_type_t::thread_begin:
      return "thread_begin";
    case xpti::trace_point_type_t::thread_end:
      return "thread_end";
    case xpti::trace_point_type_t::wait_begin:
      return "wait_begin";
    case xpti::trace_point_type_t::wait_end:
      return "wait_end";
      break;
    default: {
      std::string str =
          "unknown/user_defined[" +
          std::to_string(XPTI_EXTRACT_USER_DEFINED_ID(TraceType)) + "]";
      return str;
    }
    }
  }
#endif
  stream_cb_t MCallbacksByStream;
  mutable std::shared_mutex MCBsLock;
  std::mutex MStatsLock;
  statistics_t MStats;
  stream_flags_t MStreamFlags;
};

/// @class Framework
/// @brief Implements the XPTI tracing framwework for use in the applications.
///
/// The Framework class is designed to encapsulate the tracing and profiling
/// functionalities within the application. It acts as a central point for
/// managing trace events, collecting profiling data, and controlling the
/// overall tracing state.
///
/// @note This class is thread-safe and can be used from multiple threads
/// concurrently.

class Framework {
public:
  /// @brief Constructs a new Framework instance.
  ///
  /// Initializes the Framework instance with default values and then attempts
  /// to load trace subscribers provided through an environment variable. The
  /// tracing functionality is enabled if the environment variable for tracing
  /// is set and there are valid subscribers loaded.
  ///
  /// The constructor initializes the following member variables:
  /// - MUniversalIDs to 1, which is used as a starting point for generating
  ///   unique IDs for tracepoints.
  /// - MTracepoints with a reference to MStringTableRef, which manages a
  ///   collection of string entries used in tracepoints, payloads that define
  ///   a tracepoint and events associated with a tracepoint instance.
  /// - MTraceEnabled to false, indicating that tracing is disabled by
  /// default.
  ///   If the environment variable XPTI_TRACE_ENABLE=1,
  ///   XPTI_FRAMEWORK_DISPATCHER is set to the framework shared object and
  ///   XPTI_SUBSRIBERS is ste to at least one valid subscriber, MTraceEnabled
  ///   is set to true. will get set to true after initialization.
  ///
  /// After initialization, the constructor loads subscribers using the
  /// `loadFromEnvironmentVariable` method of the MSubscribers object. The
  /// tracing is then conditionally enabled based on two criteria:
  /// 1. The environment variable for tracing is set (checked by
  /// `g_helper.checkTraceEnv()`).
  /// 2. There are valid subscribers loaded (checked by
  /// `MSubscribers.hasValidSubscribers()`).
  ///
  /// @note The actual environment variable name and the mechanism for loading
  ///       subscribers are implemented in the `loadFromEnvironmentVariable`
  ///       method of the MSubscribers object and the `checkTraceEnv` method
  ///       of the global helper object `g_helper`.
  ///

  Framework()
      : MUniversalIDs(1), MTracepoints(MStringTableRef), MTraceEnabled(false) {
    //  Load all subscribers on construction
    MSubscribers.loadFromEnvironmentVariable();
    MTraceEnabled =
        (g_helper.checkTraceEnv() && MSubscribers.hasValidSubscribers());
    //  We create a default stream "xpti.framework" and save it in
    //  `g_default_stream
    g_default_stream_id = registerStream(g_default_stream);
  }

  /// @brief Resets the trace framework to its initial state.
  ///
  /// This method is responsible for resetting the internal state of the trace
  /// framework. It is typically called when the framework is being shut down
  /// or when a clear state is required without destroying the framework
  /// instance. The following actions are performed:
  ///
  /// After calling this method, the trace framework will be in a clean state,
  /// similar to its state immediately after initialization, but without
  /// requiring a reinitialization of the framework itself.

  void clear() {
    MUniversalIDs = 1;
    MTracepoints.clear();
    MStringTableRef.clear();
    MNotifier.clear();
  }

  inline void setTraceEnabled(bool yesOrNo = true) { MTraceEnabled = yesOrNo; }

  inline bool traceEnabled() { return MTraceEnabled; }

  inline uint64_t makeUniqueID() { return MTracepoints.makeUniqueID(); }

  uint64_t getUniversalID() const noexcept { return g_tls_uid; }

  void setUniversalID(uint64_t uid) noexcept { g_tls_uid = uid; }

  xpti::result_t stashTuple(const char *key, uint64_t value) {
    if (!key)
      return xpti::result_t::XPTI_RESULT_FAIL;

    std::get<0>(g_tls_stash_tuple) = key;
    std::get<1>(g_tls_stash_tuple) = value;
    return xpti::result_t::XPTI_RESULT_SUCCESS;
  }

  xpti::result_t getStashedTuple(char **key, uint64_t &value) {
    if (!key)
      return xpti::result_t::XPTI_RESULT_INVALIDARG;

    const char *tls_key = std::get<0>(g_tls_stash_tuple);
    if (!tls_key)
      return xpti::result_t::XPTI_RESULT_NOTFOUND;

    (*key) = const_cast<char *>(tls_key);
    value = std::get<1>(g_tls_stash_tuple);
    return xpti::result_t::XPTI_RESULT_SUCCESS;
  }

  void unstashTuple() {
    if (!std::get<0>(g_tls_stash_tuple))
      return;

    // std::get<0>(g_tls_stash_tuple) = nullptr;
    // std::get<1>(g_tls_stash_tuple) = 0;
    // We will use the actual unstash code when we implement a stack to allow
    // multiple stashes/thread
  }

  bool checkTraceEnabled(uint16_t stream, uint16_t type) {
    if (MTraceEnabled) {
      return MNotifier.checkSubscribed(stream, type);
    }
    return false;
  }

  xpti::result_t addMetadata(xpti::trace_event_data_t *Event, const char *Key,
                             object_id_t ValueID) {
    return MTracepoints.addMetadata(Event, Key, ValueID);
  }

  xpti::trace_event_data_t *newEvent(const xpti::payload_t *Payload,
                                     uint64_t *InstanceNo, uint16_t EventType,
                                     xpti::trace_activity_type_t ActivityType) {
    if (!Payload || !xpti::is_valid_payload(Payload) || !InstanceNo)
      return nullptr;

    xpti::trace_event_data_t *Event = MTracepoints.create(Payload, InstanceNo);
    if (!Event)
      return nullptr;

    Event->event_type = EventType;
    Event->flags |=
        static_cast<uint64_t>(xpti::trace_event_flag_t::EventTypeAvailable);
    Event->activity_type = (uint16_t)ActivityType;
    Event->flags |=
        static_cast<uint64_t>(xpti::trace_event_flag_t::ActivityTypeAvailable);
    return Event;
  }

  void releaseEvent(xpti::trace_event_data_t *event) {
    MTracepoints.releaseEvent(event);
  }

  inline const xpti::trace_event_data_t *findEvent(uint64_t UniversalID) {
    if (UniversalID == xpti::invalid_uid)
      return nullptr;

    if (MTracepoints.isValidUID64(UniversalID)) {
      xpti::TracepointInstance *TP =
          reinterpret_cast<xpti::TracepointInstance *>(UniversalID);
      if (TP && xpti::is_valid_event(&TP->MEvent))
        return &TP->MEvent;
      else
        return nullptr;
    }
    // // UId should be populated with the right instance information
    // xpti::uid128_t UId = findUID128(UniversalID);
    return nullptr;
  }

  inline const xpti::trace_event_data_t *lookupEvent(xpti::uid128_t *UId) {
    return MTracepoints.lookupEventData(UId);
  }

  xpti::result_t initializeStream(const char *Stream, uint32_t MajorRevision,
                                  uint32_t MinorRevision,
                                  const char *VersionString) {
    if (!Stream || !VersionString)
      return xpti::result_t::XPTI_RESULT_INVALIDARG;

    MSubscribers.initializeForStream(Stream, MajorRevision, MinorRevision,
                                     VersionString);
    return xpti::result_t::XPTI_RESULT_SUCCESS;
  }

  uint8_t registerStream(const char *StreamName) {
    return (uint8_t)MStreamStringTable.add(StreamName);
  }

  void closeAllStreams() {
    auto Table = MStreamStringTable.table();
    StringTable::st_reverse_t::iterator it;
    for (it = Table.begin(); it != Table.end(); ++it) {
      xptiFinalize(it->second);
    }
  }

  xpti::result_t unregisterStream(const char *StreamName) {
    return finalizeStream(StreamName);
  }

  uint8_t registerVendor(const char *StreamName) {
    return (uint8_t)MVendorStringTable.add(StreamName);
  }

  string_id_t registerString(const char *String, char **TableString) {
    if (!TableString || !String)
      return xpti::invalid_id;

    *TableString = 0;

    const char *RefStr;
    auto ID = MStringTableRef.add(String, &RefStr);
    *TableString = const_cast<char *>(RefStr);

    return ID;
  }

  const char *lookupString(string_id_t ID) {
    if (ID < 0)
      return nullptr;
    return MStringTableRef.query(ID);
  }

  object_id_t registerObject(const char *Object, size_t Size, uint8_t Type) {
    if (!Object)
      return xpti::invalid_id;

    return MObjectTable.insert(std::string_view(Object, Size), Type);
  }

  object_data_t lookupObject(object_id_t ID) {
    auto [Result, Type] = MObjectTable.lookup(ID);
    return {Result.size(), Result.data(), Type};
  }

  uint64_t registerPayload(xpti::payload_t *payload) {
    if (!payload || !xpti::is_valid_payload(payload))
      return xpti::invalid_uid;

    auto TP = MTracepoints.registerTracepoint(payload);
    if (xpti::is_valid_uid(TP->MUId) && xpti::is_valid_payload(&TP->MPayload) &&
        xpti::is_valid_event(&TP->MEvent)) {
      xpti::framework::uid_object_t UidHelper(TP->MUId);
      return TP->MUId.uid64;
    }

    return xpti::invalid_uid;

    // xpti::uid128_t UId;

    // if (makeKeyFromPayload(payload, &UId) !=
    //     xpti::result_t::XPTI_RESULT_SUCCESS)
    //   return xpti::invalid_uid;

    // // UId should be populated with the right instance information
    // xpti::uid64_t Id = findUID64(&UId);
    // payload->internal = Id;
    // payload->flags |= static_cast<uint64_t>(payload_flag_t::HashAvailable);

    // xpti::framework::uid_object_t UidHelper(UId);
    // payload->uid.p1 = XPTI_PACK32_RET64(UidHelper.fileId(),
    // UidHelper.lineNo()); payload->uid.p2 = XPTI_PACK32_RET64(0,
    // UidHelper.functionId());

    // return Id;
  }

  xpti::result_t makeKeyFromPayload(xpti::payload_t *payload,
                                    xpti::uid128_t *uid) {
    if (!payload || !uid || !xpti::is_valid_payload(payload))
      return xpti::result_t::XPTI_RESULT_INVALIDARG;

    auto TP = MTracepoints.registerTracepoint(payload);
    if (xpti::is_valid_uid(TP->MUId) && xpti::is_valid_payload(&TP->MPayload) &&
        xpti::is_valid_event(&TP->MEvent)) {
      *uid = TP->MUId;
      *payload = TP->MPayload;
      return xpti::result_t::XPTI_RESULT_SUCCESS;
    }
    return xpti::result_t::XPTI_RESULT_FAIL;
  }

  const xpti::tracepoint_data_t *
  registerTracepointScope(xpti::payload_t *payload) {
    g_tls_temp_scope_data = xpti::tracepoint_data_t();
    if (!payload || !is_valid_payload(payload))
      return &g_tls_temp_scope_data;

    uint64_t InstanceNo = 0;
    auto event = MTracepoints.create(payload, &InstanceNo);
    // Scope data is created by this function and needs to successfully
    // populate all attributes for it to be valid
    //
    // We will set the unique_id in the trace event as the UID64 for the
    // tracepoint, but this may not be set if we are using the new API. In
    // this case, it will be set to xpti::invalid_uid
    if (!event)
      return &g_tls_temp_scope_data;

    if (xpti::is_valid_event(event)) {
      if (event->flags &
          static_cast<uint64_t>(xpti::trace_event_flag_t::HashAvailable))
        g_tls_temp_scope_data.uid64 = event->unique_id;
      if (event->flags &
          static_cast<uint64_t>(xpti::trace_event_flag_t::UIDAvailable))
        g_tls_temp_scope_data.uid128 = event->universal_id;
      // Lifetime of a payload is the duration of the application
      if (event->flags &
          static_cast<uint64_t>(xpti::trace_event_flag_t::PayloadAvailable))
        g_tls_temp_scope_data.payload = event->reserved.payload;
      // The caller will manage the lifetime of the event, which is the scope
      g_tls_temp_scope_data.event = event;
    }
    return &g_tls_temp_scope_data;
  }

  xpti::result_t registerCallback(uint8_t StreamID, uint16_t TraceType,
                                  xpti::tracepoint_callback_api_t cbFunc) {
    return MNotifier.registerCallback(StreamID, TraceType, cbFunc);
  }

  xpti::result_t unregisterCallback(uint8_t StreamID, uint16_t TraceType,
                                    xpti::tracepoint_callback_api_t cbFunc) {
    return MNotifier.unregisterCallback(StreamID, TraceType, cbFunc);
  }

  xpti::result_t notifySubscribers(uint8_t StreamID, uint16_t TraceType,
                                   xpti::trace_event_data_t *Parent,
                                   xpti::trace_event_data_t *Object,
                                   uint64_t InstanceNo, const void *UserData) {
    if (!MTraceEnabled)
      return xpti::result_t::XPTI_RESULT_FALSE;
    if (!Object) {
      // We have relaxed the rules for notifications: Notifications can now
      // have 'nullptr' for both the Parent and Object only if UserData is
      // provided and the trace_point_type is function_begin/function_end.
      // This allows us to trace function calls without too much effort.
      std::array<trace_point_type_t, 26> AllowedTypes = {
          trace_point_type_t::function_begin,
          trace_point_type_t::function_end,
          trace_point_type_t::function_with_args_begin,
          trace_point_type_t::function_with_args_end,
          trace_point_type_t::mem_alloc_begin,
          trace_point_type_t::mem_alloc_end,
          trace_point_type_t::mem_release_begin,
          trace_point_type_t::mem_release_end,
          trace_point_type_t::offload_alloc_memory_object_construct,
          trace_point_type_t::offload_alloc_memory_object_associate,
          trace_point_type_t::offload_alloc_memory_object_release,
          trace_point_type_t::offload_alloc_memory_object_destruct,
          trace_point_type_t::offload_alloc_accessor,
          trace_point_type_t::diagnostics};
      const auto Predicate = [TraceType](trace_point_type_t RHS) {
        return TraceType == static_cast<uint16_t>(RHS);
      };
      if (!(UserData &&
            std::any_of(AllowedTypes.begin(), AllowedTypes.end(), Predicate))) {
        return xpti::result_t::XPTI_RESULT_INVALIDARG;
      }
    }
    //
    //  Notify all subscribers for the stream 'StreamID'
    //
    return MNotifier.notifySubscribers(StreamID, TraceType, Parent, Object,
                                       InstanceNo, UserData);
  }

  bool hasSubscribers() { return MSubscribers.hasValidSubscribers(); }

  xpti::result_t finalizeStream(const char *Stream) {
    if (!Stream)
      return xpti::result_t::XPTI_RESULT_INVALIDARG;
    MSubscribers.finalizeForStream(Stream);
    return MNotifier.unregisterStream(MStreamStringTable.add(Stream));
  }

  inline const xpti::payload_t *queryPayload(xpti::trace_event_data_t *Event) {
    return MTracepoints.payloadData(Event);
  }

  const xpti::payload_t *queryPayloadByUID(uint64_t uid) {
    if (uid == xpti::invalid_uid)
      return nullptr;

    xpti::TracepointInstance *TP =
        reinterpret_cast<xpti::TracepointInstance *>(uid);
    if (xpti::is_valid_payload(&TP->MPayload))
      return &TP->MPayload;

    return nullptr;
  }

  inline const xpti::payload_t *lookupPayload(xpti::uid128_t *uid) {
    return MTracepoints.payloadDataByUniversalID(uid);
  }

  void printStatistics() {
    MNotifier.printStatistics();
    MStringTableRef.printStatistics();
    MTracepoints.printStatistics();
  }

  static Framework &instance() {
    Framework *TmpFramework = MInstance.load(std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_acquire);
    if (TmpFramework == nullptr) {
      std::lock_guard<utils::SpinLock> Lock{MSingletonMutex};
      TmpFramework = MInstance.load(std::memory_order_relaxed);
      if (TmpFramework == nullptr) {
        TmpFramework = new Framework();
        std::atomic_thread_fence(std::memory_order_release);
        MInstance.store(TmpFramework, std::memory_order_relaxed);
      }
    }

    return *TmpFramework;
  }

private:
  friend void ::xptiFrameworkFinalize();

  static void release() {
    Framework *TmpFramework = MInstance.load(std::memory_order_relaxed);
    MInstance.store(nullptr, std::memory_order_relaxed);
    delete TmpFramework;
  }

  /// Stores singleton instance
  static std::atomic<Framework *> MInstance;
  /// Trivially destructible mutex for double-checked lock idiom
  static utils::SpinLock MSingletonMutex;
  /// Thread-safe counter used for generating universal IDs
  xpti::safe_uint64_t MUniversalIDs;
  /// Manages loading the subscribers and calling their init() functions
  xpti::Subscribers MSubscribers;
  /// Used to send event notification to subscribers
  xpti::Notifications MNotifier;
  /// Thread-safe string table
  xpti::StringTable MStringTableRef;
  /// Thread-safe object table
  xpti::ObjectTable<object_id_t> MObjectTable;
  /// Thread-safe string table, used for stream IDs
  xpti::StringTable MStreamStringTable;
  /// Thread-safe string table, used for vendor IDs
  xpti::StringTable MVendorStringTable;
  /// Manages the tracepoints - framework caching
  xpti::Tracepoints MTracepoints;
  /// Flag indicates whether tracing should be enabled
  bool MTraceEnabled;
};

/// @var static int GFrameworkReferenceCounter
/// @brief Global reference counter for the XPTI Trace Framework instances.
///
/// This static variable is used to keep track of the number of active
/// references to the XPTI Trace Framework instance. It is incremented when a
/// new instance is created and decremented when an instance is destroyed.
/// This mechanism ensures proper initialization and teardown of the
/// framework's resources.
static int GFrameworkReferenceCounter = 0;

/// @var std::atomic<Framework *> Framework::MInstance
/// @brief Atomic pointer to the singleton instance of the XPTI Trace
/// Framework.
///
/// This variable holds a pointer to the singleton instance of the XPTI Trace
/// Framework. The use of `std::atomic` ensures that operations on this
/// pointer are thread-safe, allowing for safe access and modification in a
/// concurrent environment. This is crucial for maintaining a single instance
/// of the framework across multiple threads.
std::atomic<Framework *> Framework::MInstance;

/// @var utils::SpinLock Framework::MSingletonMutex
/// @brief Spinlock mutex for controlling access to the singleton instance of
/// the framework.
///
/// This spinlock mutex is used to control access to the singleton instance of
/// the XPTI Trace Framework during its creation and destruction. The use of a
/// spinlock provides a lightweight synchronization mechanism that is
/// efficient in scenarios where lock contention is expected to be low and
/// lock hold times are short.
utils::SpinLock Framework::MSingletonMutex;

} // namespace xpti

extern "C" {

XPTI_EXPORT_API void xptiFrameworkInitialize() {
  std::lock_guard<xpti::utils::SpinLock> guard{xpti::g_framework_mutex};
  xpti::GFrameworkReferenceCounter++;
}

XPTI_EXPORT_API void xptiFrameworkFinalize() {
  std::lock_guard<xpti::utils::SpinLock> guard{xpti::g_framework_mutex};

  xpti::GFrameworkReferenceCounter--;
  if (xpti::GFrameworkReferenceCounter == 0) {
    xpti::Framework::release();
  }
}

XPTI_EXPORT_API uint64_t xptiGetUniversalId() {
  return xpti::Framework::instance().getUniversalID();
}

XPTI_EXPORT_API void xptiSetUniversalId(uint64_t uid) {
  xpti::Framework::instance().setUniversalID(uid);
}

XPTI_EXPORT_API xpti::result_t xptiStashTuple(const char *key, uint64_t value) {
  return xpti::Framework::instance().stashTuple(key, value);
}

XPTI_EXPORT_API xpti::result_t xptiGetStashedTuple(char **key,
                                                   uint64_t &value) {
  return xpti::Framework::instance().getStashedTuple(key, value);
}

XPTI_EXPORT_API void xptiUnstashTuple() {
  xpti::Framework::instance().unstashTuple();
}

XPTI_EXPORT_API uint16_t
xptiRegisterUserDefinedTracePoint(const char *ToolName, uint8_t UserDefinedTP) {
  uint8_t ToolID = xpti::Framework::instance().registerVendor(ToolName);
  UserDefinedTP |= (uint8_t)xpti::trace_point_type_t::user_defined;
  uint16_t UserDefTracepoint = XPTI_PACK08_RET16(ToolID, UserDefinedTP);

  return UserDefTracepoint;
}

XPTI_EXPORT_API uint16_t xptiRegisterUserDefinedEventType(
    const char *ToolName, uint8_t UserDefinedEvent) {
  uint8_t ToolID = xpti::Framework::instance().registerVendor(ToolName);
  UserDefinedEvent |= (uint8_t)xpti::trace_event_type_t::user_defined;
  uint16_t UserDefEventType = XPTI_PACK08_RET16(ToolID, UserDefinedEvent);
  return UserDefEventType;
}

XPTI_EXPORT_API xpti::result_t xptiInitialize(const char *Stream, uint32_t maj,
                                              uint32_t min,
                                              const char *version) {
  auto &FW = xpti::Framework::instance();
  // Before any stream gets initialized, we should initialize the framework
  // for the default stream
  std::call_once(g_initialize_default_stream_flag, [&]() {
    FW.initializeStream(g_default_stream, 1, 0, "1.0.0");
  });
  return FW.initializeStream(Stream, maj, min, version);
}

XPTI_EXPORT_API void xptiFinalize(const char *Stream) {
  xpti::Framework::instance().finalizeStream(Stream);
}

XPTI_EXPORT_API uint64_t xptiGetUniqueId() {
  return xpti::Framework::instance().makeUniqueID();
}

XPTI_EXPORT_API xpti::string_id_t xptiRegisterString(const char *String,
                                                     char **RefTableStr) {
  return xpti::Framework::instance().registerString(String, RefTableStr);
}

XPTI_EXPORT_API const char *xptiLookupString(xpti::string_id_t ID) {
  return xpti::Framework::instance().lookupString(ID);
}

XPTI_EXPORT_API xpti::object_id_t
xptiRegisterObject(const char *Data, size_t Size, uint8_t Type) {
  return xpti::Framework::instance().registerObject(Data, Size, Type);
}

XPTI_EXPORT_API xpti::object_data_t xptiLookupObject(xpti::object_id_t ID) {
  return xpti::Framework::instance().lookupObject(ID);
}

XPTI_EXPORT_API uint64_t xptiRegisterPayload(xpti::payload_t *payload) {
  return xpti::Framework::instance().registerPayload(payload);
}

XPTI_EXPORT_API xpti::result_t xptiMakeKeyFromPayload(xpti::payload_t *payload,
                                                      xpti::uid128_t *key) {
  if (!payload || !key)
    return xpti::result_t::XPTI_RESULT_INVALIDARG;

  auto &FW = xpti::Framework::instance();
  return FW.makeKeyFromPayload(payload, key);
}

XPTI_EXPORT_API uint8_t xptiRegisterStream(const char *StreamName) {
  return xpti::Framework::instance().registerStream(StreamName);
}

XPTI_EXPORT_API xpti::result_t xptiUnregisterStream(const char *StreamName) {
  return xpti::Framework::instance().unregisterStream(StreamName);
}

XPTI_EXPORT_API xpti::trace_event_data_t *
xptiMakeEvent(const char * /*Name*/, xpti::payload_t *Payload, uint16_t Event,
              xpti::trace_activity_type_t Activity, uint64_t *InstanceNo) {
  auto &FW = xpti::Framework::instance();
  auto RetEv = FW.newEvent(Payload, InstanceNo, Event, Activity);
  // if (RetEv && xpti::is_valid_event(RetEv)) {

  //   xpti::TracepointInstance *TP =
  //       reinterpret_cast<xpti::TracepointInstance *>(RetEv->unique_id);
  //   // RetEv->unique_id = FW.findUID64(&RetEv->universal_id);
  //   // RetEv->flags |=
  //   //     static_cast<uint64_t>(xpti::trace_event_flag_t::HashAvailable);
  //   // RetEv->universal_id.uid64 = RetEv->unique_id;
  // }
  return RetEv;
}

XPTI_EXPORT_API xpti::trace_event_data_t *
xptiCreateEvent(xpti::payload_t *Payload, uint64_t *InstanceNo, uint16_t Event,
                xpti::trace_activity_type_t Activity) {
  return xpti::Framework::instance().newEvent(Payload, InstanceNo, Event,
                                              Activity);
}

XPTI_EXPORT_API void xptiReset() { xpti::Framework::instance().clear(); }

XPTI_EXPORT_API const xpti::trace_event_data_t *xptiFindEvent(uint64_t UId) {
  return xpti::Framework::instance().findEvent(UId);
}

XPTI_EXPORT_API const xpti::trace_event_data_t *
xptiLookupEvent(xpti::uid128_t *UId) {
  return xpti::Framework::instance().lookupEvent(UId);
}

XPTI_EXPORT_API const xpti::payload_t *
xptiQueryPayload(xpti::trace_event_data_t *LookupObject) {
  return xpti::Framework::instance().queryPayload(LookupObject);
}

XPTI_EXPORT_API const xpti::payload_t *xptiQueryPayloadByUID(uint64_t uid) {
  return xpti::Framework::instance().queryPayloadByUID(uid);
}

XPTI_EXPORT_API const xpti::payload_t *xptiLookupPayload(xpti::uid128_t *uid) {
  return xpti::Framework::instance().lookupPayload(uid);
}

XPTI_EXPORT_API xpti::result_t
xptiRegisterCallback(uint8_t StreamID, uint16_t TraceType,
                     xpti::tracepoint_callback_api_t cbFunc) {
  return xpti::Framework::instance().registerCallback(StreamID, TraceType,
                                                      cbFunc);
}

XPTI_EXPORT_API xpti::result_t
xptiUnregisterCallback(uint8_t StreamID, uint16_t TraceType,
                       xpti::tracepoint_callback_api_t cbFunc) {
  return xpti::Framework::instance().unregisterCallback(StreamID, TraceType,
                                                        cbFunc);
}

XPTI_EXPORT_API xpti::result_t
xptiNotifySubscribers(uint8_t StreamID, uint16_t TraceType,
                      xpti::trace_event_data_t *Parent,
                      xpti::trace_event_data_t *Object, uint64_t InstanceNo,
                      const void *TemporalUserData) {
  return xpti::Framework::instance().notifySubscribers(
      StreamID, TraceType, Parent, Object, InstanceNo, TemporalUserData);
}

XPTI_EXPORT_API bool xptiTraceEnabled() {
  return xpti::Framework::instance().traceEnabled();
}

XPTI_EXPORT_API bool xptiCheckTraceEnabled(uint16_t stream, uint16_t ttype) {
  return xpti::Framework::instance().checkTraceEnabled(stream, ttype);
}

XPTI_EXPORT_API xpti::result_t xptiAddMetadata(xpti::trace_event_data_t *Event,
                                               const char *Key,
                                               xpti::object_id_t ID) {
  return xpti::Framework::instance().addMetadata(Event, Key, ID);
}

XPTI_EXPORT_API xpti::metadata_t *
xptiQueryMetadata(xpti::trace_event_data_t *Event) {
  return &Event->reserved.metadata;
}

XPTI_EXPORT_API void xptiForceSetTraceEnabled(bool YesOrNo) {
  xpti::Framework::instance().setTraceEnabled(YesOrNo);
}

XPTI_EXPORT_API const xpti::tracepoint_data_t *xptiGetTracepointScopeData() {
  return &g_tls_tracepoint_scope_data;
}

XPTI_EXPORT_API xpti::result_t
xptiSetTracepointScopeData(xpti::tracepoint_data_t *Data) {
  if (!Data->isValid())
    return xpti::result_t::XPTI_RESULT_INVALIDARG;
  // Copy to TLS so it is available for the remainder of the scope
  g_tls_tracepoint_scope_data = *Data;
  // Also set Universal ID separately as it may be in use by older
  // implementations of tools, but this field is set in the incoming data only
  // if the legacy API are in use
  xptiSetUniversalId(Data->uid64);
  return xpti::result_t::XPTI_RESULT_SUCCESS;
}

XPTI_EXPORT_API void xptiUnsetTracepointScopeData() {
  g_tls_tracepoint_scope_data = xpti::tracepoint_data_t();
}

XPTI_EXPORT_API const xpti::tracepoint_data_t *
xptiRegisterTracepointScope(xpti::payload_t *Payload) {
  return xpti::Framework::instance().registerTracepointScope(Payload);
}

XPTI_EXPORT_API void
xptiEnableTracepointScopeNotification(bool enableOrDisable) {
  xpti::g_tracepoint_self_notify = enableOrDisable;
}

XPTI_EXPORT_API bool xptiCheckTracepointScopeNotification() {
  return xpti::g_tracepoint_self_notify;
}

XPTI_EXPORT_API uint8_t xptiGetDefaultStreamID() {
  return xpti::g_default_stream_id;
}

XPTI_EXPORT_API xpti::result_t xptiSetDefaultStreamID(uint8_t DefaultStreamId) {
  if ((int8_t)DefaultStreamId < 0)
    return xpti::result_t::XPTI_RESULT_INVALIDARG;

  xpti::g_default_stream_id = DefaultStreamId;
  return xpti::result_t::XPTI_RESULT_SUCCESS;
}
XPTI_EXPORT_API xpti::trace_event_type_t xptiGetDefaultEventType() {
  return xpti::g_default_event_type;
}

XPTI_EXPORT_API xpti::result_t
xptiSetDefaultEventType(xpti::trace_event_type_t DefaultEventType) {
  if (DefaultEventType == xpti::trace_event_type_t::unknown_event)
    return xpti::result_t::XPTI_RESULT_INVALIDARG;

  xpti::g_default_event_type = DefaultEventType;
  return xpti::result_t::XPTI_RESULT_SUCCESS;
}

XPTI_EXPORT_API xpti::trace_point_type_t xptiGetDefaultTraceType() {
  return xpti::g_default_trace_type;
}

XPTI_EXPORT_API xpti::result_t
xptiSetDefaultTraceType(xpti::trace_point_type_t DefaultTraceType) {
  if (DefaultTraceType == xpti::trace_point_type_t::unknown_type)
    return xpti::result_t::XPTI_RESULT_INVALIDARG;

  xpti::g_default_trace_type = DefaultTraceType;
  return xpti::result_t::XPTI_RESULT_SUCCESS;
}

XPTI_EXPORT_API void xptiReleaseEvent(xpti::trace_event_data_t *Event) {
  return xpti::Framework::instance().releaseEvent(Event);
}

} // extern "C"

#if (defined(_WIN32) || defined(_WIN64))

#include <string>
#include <windows.h>

BOOL WINAPI DllMain(HINSTANCE hinstDLL, DWORD fwdReason, LPVOID lpvReserved) {
  switch (fwdReason) {
  case DLL_PROCESS_ATTACH:
    break;
  case DLL_PROCESS_DETACH:
    //
    //  We cannot unload all subscribers here...
    //
#ifdef XPTI_STATISTICS
    xpti::Framework::instance().printStatistics();
#endif
    break;
  }

  return TRUE;
}

#else // Linux (possibly macOS?)

__attribute__((constructor)) static void framework_init() {}

__attribute__((destructor)) static void framework_fini() {
#ifdef XPTI_STATISTICS
  xpti::Framework::instance().printStatistics();
#endif
}

#endif
