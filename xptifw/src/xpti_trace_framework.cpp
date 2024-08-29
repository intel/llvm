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
#include "hash_table7.hpp"
#include "parallel_hashmap/phmap.h"
#include "xpti/xpti_trace_framework.h"
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
static thread_local xpti_tracepoint_t *g_tls_tracepoint_scope_data;

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

/// @var thread_local xpti::tracepoint_t *g_tls_temp_scope_data
/// @brief Thread-local storage for temporary scope data in the tracing
/// framework.
///
/// This variable is used to store temporary data for a tracepoint interface for
/// the tracepoint created in the scoped class
static thread_local xpti_tracepoint_t *g_tls_temp_scope_data;

struct PayloadReferenceImpl : xpti_payload_t {
  xpti::payload_t MPayload;

  PayloadReferenceImpl(xpti::payload_t *Payload) {
    MPayload.name = Payload->name;
    MPayload.source_file = Payload->source_file;
    MPayload.line_no = Payload->line_no;
    MPayload.column_no = Payload->column_no;
    MPayload.internal = Payload->internal;
    MPayload.flags = Payload->flags;
    MPayload.uid.p1 = Payload->uid.p1;
    MPayload.uid.p2 = Payload->uid.p2;
    // Need to determine if setting uid.p3 to instance will help
  }

  const char *name() const override {
    return MPayload.name ? MPayload.name : "<unknown>";
  }
  const char *source_file() const override {
    return MPayload.source_file ? MPayload.source_file : "<unknown-file>";
  }
  uint32_t line_no() const override { return MPayload.line_no; }
  uint32_t column_no() const override { return MPayload.column_no; }
  uint64_t payload_flags() const override { return MPayload.flags; }
  uint64_t uid64() const override { return MPayload.internal; }
  int32_t name_string_id() const override {
    return (int32_t)(MPayload.uid.p2 & 0x00000000ffffffff);
  }
  int32_t file_string_id() const override {
    return (int32_t)(MPayload.uid.p1 >> 32);
  }
  bool is_valid() const override {
    return (MPayload.flags != 0) &&
           ((MPayload.flags &
                 static_cast<uint64_t>(payload_flag_t::SourceFileAvailable) ||
             (MPayload.flags &
              static_cast<uint64_t>(payload_flag_t::NameAvailable))));
  }
  xpti::payload_t *payload_ref() override { return &MPayload; }
};

/// @class TracePointImpl
/// @brief Implements a trace point in the tracing framework.
///
/// This class is designed to encapsulate all the necessary information about a
/// trace point, including its unique identifier (UID), payload, metadata, and
/// event data. It inherits from several structures to provide a comprehensive
/// view of a trace point, including its payload, event, metadata, and
/// tracepoint specifics.

struct TracePointImpl : xpti_payload_t,
                        xpti_trace_event_t,
                        xpti_metadata_t,
                        xpti_tracepoint_t {
  /// @brief  Universal identifier for the trace point.
  xpti::uid128_t MUId;
  /// @brief Payload for the trace point.
  xpti::payload_t MPayload;
  /// @brief Event data for the trace point.
  xpti::trace_event_data_t MEvent;
  /// @brief Cached Function string ID for the trace point.
  int32_t MFuncID = xpti::invalid_id;
  /// @brief Cached File string ID for the trace point.
  int32_t MFileID = xpti::invalid_id;
  /// @brief Iterator for the metadata associated with the trace point.
  xpti::metadata_t::iterator MCurr;

  ///  @brief Constructor for the TracePointImpl class.
  ///
  ///  Initializes a trace point with a unique identifier (UID) and associated
  ///  payload. The payload contains metadata about the trace point such as the
  ///  name, source file, line number, and additional flags. This constructor
  ///  also computes a packed UID for the payload, sets up the event structure
  ///  including event and activity types, and marks the event with flags
  ///  indicating the availability of various pieces of information.
  ///
  ///  @param UID     A reference to a unique identifier (uid128_t) for the
  ///                 trace point. This UID is used to uniquely identify the
  ///                 trace point across the tracing framework.
  ///  @param Payload A pointer to a payload_t structure containing metadata
  ///                 about the trace point. This includes information such as
  ///                 the trace point's name, source file, and location within
  ///                 the source file.
  ///  @note          The UID should be valid with the right information packed
  ///                 into p1 and p2 fields and the Payload should be a
  ///                 registered payload.

  TracePointImpl(xpti::uid128_t &UID, xpti::payload_t *Payload) {
    if (!Payload || !xpti::is_valid_payload(Payload) ||
        (UID.p1 == 0 && UID.p2 == 0))
      return;
    MUId = UID;
    MPayload.name = Payload->name;
    MPayload.source_file = Payload->source_file;
    MPayload.line_no = Payload->line_no;
    MPayload.column_no = Payload->column_no;
    MPayload.flags = Payload->flags;

    xpti::framework::uid_object_t UidHelper(MUId);
    MFuncID = UidHelper.functionId();
    MFileID = UidHelper.fileId();

    MPayload.uid.p1 = Payload->uid.p1 =
        XPTI_PACK32_RET64(MFileID, MPayload.line_no);
    MPayload.uid.p2 = Payload->uid.p2 = XPTI_PACK32_RET64(0, MFuncID);

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

  const char *name() const override {
    return MPayload.name ? MPayload.name : "<unknown>";
  }
  const char *source_file() const override {
    return MPayload.source_file ? MPayload.source_file : "<unknown-file>";
  }
  uint32_t line_no() const override { return MPayload.line_no; }
  uint32_t column_no() const override { return MPayload.column_no; }
  uint64_t payload_flags() const override { return MPayload.flags; }
  int32_t name_string_id() const override { return MFuncID; }
  int32_t file_string_id() const override { return MFileID; }
  uint64_t uid64() const override { return MPayload.internal; }
  bool is_valid() const override {
    return (xpti::is_valid_uid(MUId) && xpti::is_valid_payload(&MPayload) &&
            xpti::is_valid_event(&MEvent));
  }
  xpti::payload_t *payload_ref() override { return &MPayload; }

  uint64_t instance() const override { return MUId.instance; }

  // Methods for accessing base class interfaces.
  xpti_payload_t *payload() override {
    return static_cast<xpti_payload_t *>(this);
  }
  xpti_metadata_t *metadata() override {
    return static_cast<xpti_metadata_t *>(this);
  }
  xpti_trace_event_t *event() override {
    return static_cast<xpti_trace_event_t *>(this);
  }

  uint16_t event_type() const override { return MEvent.event_type; }
  uint16_t activity_type() const override { return MEvent.activity_type; }
  uint64_t source_uid64() const override { return MEvent.source_id; }
  uint64_t target_uid64() const override { return MEvent.target_id; }
  uint64_t event_flags() const override { return MEvent.flags; }
  void set_activity_type(xpti::trace_activity_type_t type) override {
    MEvent.activity_type = (uint16_t)type;
  }
  void set_event_type(uint16_t type) override { MEvent.event_type = type; }

  xpti::trace_event_data_t *event_ref() override { return &MEvent; }

  // Methods for iterating and manipulating metadata items.
  xpti::result_t first_item(char **key, xpti::object_id_t &value) override {
    MCurr = MEvent.reserved.metadata.begin();
    if (MCurr != MEvent.reserved.metadata.end()) {
      *key = const_cast<char *>(xptiLookupString(MCurr->first));
      value = MCurr->second;
      return xpti::result_t::XPTI_RESULT_SUCCESS;
    }

    return xpti::result_t::XPTI_RESULT_FALSE;
  }
  xpti::result_t next_item(char **key, xpti::object_id_t &value) override {
    MCurr++;
    if (MCurr != MEvent.reserved.metadata.end()) {
      *key = const_cast<char *>(xptiLookupString(MCurr->first));
      value = MCurr->second;
      return xpti::result_t::XPTI_RESULT_SUCCESS;
    }

    return xpti::result_t::XPTI_RESULT_FALSE;
  }

  xpti::result_t add_item(const char *key, xpti::object_id_t value) override {
    return xpti::addMetadata(&MEvent, key, value);
  }
  size_t count() override { return MEvent.reserved.metadata.size(); }
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
  /// identifiers (uids) to their corresponding payload entries. Since we plan
  /// to keep the payload information until the end of the application lifetime,
  /// we should be able to use the references.
  using uid_payload_lut = std::unordered_map<xpti::uid128_t, uid_entry_t>;

  /// @typedef uid_instances_t
  /// @brief Defines a hash map for managing tracepoint instances by their
  /// unique identifiers.
  ///
  /// This type alias represents a hash map where the key is a 64-bit unsigned
  /// integer representing the unique instance of a tracepoint, and the value is
  /// an instance of `xpti::TracePointImpl`. The `emhash7::HashMap` is
  /// chosen for its efficiency in managing hash collisions and overall
  /// performance in insertions and lookups. This map is used within the trace
  /// framework to efficiently manage and access tracepoint instances by their
  /// UIDs, allowing for quick updates and retrievals of tracepoint data.
  using uid_instances_t = emhash7::HashMap<uint64_t, xpti::TracePointImpl *>;

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
  using uid_tracepoints_lut = phmap::flat_hash_map<uid128_t, uid_instances_t>;

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

    for (auto &TP : MTracepoints) {
      for (auto &Instance : TP.second) {
        delete Instance.second;
      }
    }
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

    if (Event->flags &
        static_cast<uint64_t>(xpti::trace_event_flag_t::PayloadAvailable))
      return Event->reserved.payload;
    else {
      auto TP = reinterpret_cast<xpti::TracePointImpl *>(Event->unique_id);
      if (TP) {
        Event->reserved.payload = &TP->MPayload;
        Event->flags |=
            static_cast<uint64_t>(xpti::trace_event_flag_t::PayloadAvailable);
        return Event->reserved.payload;
      }
    }
    return nullptr;
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
  const xpti_trace_event_t *lookupEventData(uint64_t UId) {
    if (!UId)
      return nullptr;

    if (MUID64Check.count(UId) == 0)
      return nullptr;

    xpti::TracePointImpl *TP = reinterpret_cast<xpti::TracePointImpl *>(UId);
    if (xpti::is_valid_event(&TP->MEvent))
      return dynamic_cast<xpti_trace_event_t *>(TP);
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
    if (!Event || Event->unique_id == xpti::invalid_uid)
      return;

    if (MUID64Check.count(Event->unique_id) == 0)
      return;

    // We have a valid unique ID and we can proceed to release the event
    xpti::TracePointImpl *TP =
        reinterpret_cast<xpti::TracePointImpl *>(Event->unique_id);

    xpti::uid128_t UId = TP->MUId;
    {
      std::unique_lock<std::shared_mutex> Lock(MTracepointMutex);
      // Find the event list for a given UID
      auto &Instances = MTracepoints[UId];
      MUID64Check.erase(UId.uid64);
      // Now release the event associated with the UID instance
      delete Instances[UId.instance];
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
    std::unique_lock<std::shared_mutex> Lock(MPayloadMutex);
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
  /// @return Pointer to the newly created TracePointImpl; returns nullptr
  ///         if the universal ID computed from the Payload is invalid.
  ///
  xpti::TracePointImpl *registerTracepoint(xpti::payload_t *Payload) {
    // Generate a unique key from the payload which includes a universal ID and
    // invariant payload; If the payload has been visited multiple times, the ID
    // will have the same information for id.p1 and id.p2, but the instance
    // value will be different
    auto [UniversalId, InvarPayload] = makeKeyFromPayload(Payload);
    // Check if the generated universal ID is valid
    if (!xpti::is_valid_uid(UniversalId))
      return nullptr;

    // Lock the mutex to ensure thread-safe access to the tracepoints map
    std::unique_lock<std::shared_mutex> Lock(MTracepointMutex);
    // Access or create the tracepoint instance associated with the universal ID
    auto &Tracepoint = MTracepoints[UniversalId];
    // Access or create the specific instance of the tracepoint based on the
    // universal ID instance
    auto &TP = Tracepoint[UniversalId.instance];
#ifdef XPTI_STATISTICS
    MInsertions++;
#endif
    TP = new xpti::TracePointImpl(UniversalId, InvarPayload);
    // Set various internal IDs and flags for the tracepoint and payload
    Payload->internal = TP->MEvent.unique_id = TP->MPayload.internal =
        TP->MUId.uid64 = (uint64_t)TP;
    TP->MPayload.flags |=
        static_cast<uint64_t>(xpti::payload_flag_t::HashAvailable);

    // Add the 64-bit ID for validation purposes when legacy API are used
    MUID64Check.insert(TP->MUId.uid64);
    // Return a pointer to the updated tracepoint instance which contains the
    // UID with the correct instance number, the payload with updated flags and
    // fields and the event that represents the tracepoint
    return TP;
  }

  xpti::result_t deleteTracepoint(xpti_tracepoint_t *Tracepoint) {
    if (!Tracepoint)
      return xpti::result_t::XPTI_RESULT_INVALIDARG;

    xpti::TracePointImpl *TP = dynamic_cast<xpti::TracePointImpl *>(Tracepoint);
    if (!TP)
      return xpti::result_t::XPTI_RESULT_INVALIDARG;

    {
      xpti::uid128_t UId = TP->MUId;
      // Lock the mutex to ensure thread-safe access to the tracepoints map
      std::unique_lock<std::shared_mutex> Lock(MTracepointMutex);
      // Find the tracepoint for a given UID
      auto &Instances = MTracepoints[UId];
      // Now release the 64-bit UID associated with tracepoint instance
      MUID64Check.erase(UId.uid64);
      // Now release the tracepoint associated with the UID instance by deleting
      // the memory allocated first
      delete Instances[UId.instance];
      Instances.erase(UId.instance);
      // If there are no more events associated with the UID, we can release
      // the Payload as well, but we will not as the same payload may be
      // revisited and we need to keep the instance count going
    }
    return xpti::result_t::XPTI_RESULT_SUCCESS;
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

    auto TPInstance = registerTracepoint(&TempPayload);
    *InstanceNo = TPInstance->MUId.instance;
    return &TPInstance->MEvent;
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
    std::unique_lock<std::shared_mutex> Lock(MCBsLock);
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
    std::unique_lock<std::shared_mutex> Lock(MCBsLock);
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
    std::unique_lock<std::shared_mutex> Lock(MCBsLock);
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
      std::shared_lock<std::shared_mutex> Lock(MCBsLock);
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

  /// @brief Enables or disables tracing globally.
  ///
  /// This function sets the global tracing flag to the specified value. When
  /// tracing is enabled, the framework will start collecting trace data based
  /// on registered callbacks and trace points. Disabling tracing will halt data
  /// collection, potentially improving performance.
  ///
  /// @param yesOrNo A boolean value indicating whether tracing should be
  ///                enabled (true) or disabled (false). The default value is
  ///                true, enabling tracing.
  inline void setTraceEnabled(bool yesOrNo = true) { MTraceEnabled = yesOrNo; }

  /// @brief Checks if tracing is globally enabled.
  ///
  /// This function returns the current state of the global tracing flag. When
  /// tracing is enabled, the framework is in a state to collect and process
  /// trace data. This flag can be set or unset using the `setTraceEnabled`
  /// function, allowing for dynamic control over tracing activities.
  ///
  /// @return A boolean value indicating the current state of tracing. Returns
  ///         `true` if tracing is enabled, and `false` if it is disabled.
  inline bool traceEnabled() { return MTraceEnabled; }

  /// @brief Generates a unique identifier for use with tracepoints.
  ///
  /// It is used to ensure that each tracepoint's information in the system can
  /// be uniquely identified, facilitating the tracking and analysis of trace
  /// data.
  ///
  /// @return Returns a `uint64_t` value that represents a unique identifier for
  ///         a tracepoint.
  inline uint64_t makeUniqueID() { return MTracepoints.makeUniqueID(); }

  /// @brief Retrieves the universal identifier for the current thread.
  ///
  /// When the tracepoint is managed using scoped objects, the 64-bit universal
  /// ID is stashed in TLS to ensure downstream components have access to the
  /// TLS, if they are in the scope. This functions retieves the shashed value.
  ///
  /// @return A `uint64_t` value representing the universal identifier for the
  /// current scope on current thread. This identifier is unique across all
  /// threads within the process.
  uint64_t getUniversalID() const noexcept { return g_tls_uid; }

  /// @brief Sets the universal identifier for the current thread.
  ///
  /// When a scoped object is used for a tracepoint, the 64-bit universal ID is
  /// stashed in the TLS for use by downstream components. This ID represents
  /// the current event for the tracepoint.
  ///
  /// @param uid The `uint64_t` value to be set as the universal identifier for
  ///             the current thread.
  void setUniversalID(uint64_t uid) noexcept { g_tls_uid = uid; }

  /// @brief Stashes a key-value pair in a thread-local storage (TLS) tuple.
  ///
  /// This function is designed to temporarily store a key-value pair in a
  /// global, thread-local storage (TLS) tuple. The key is a string, and the
  /// value is a 64-bit unsigned integer. This mechanism is typically used to
  /// hold data that is specific to the current thread and needs to be accessed
  /// or modified during the thread's execution. This functionality maybe
  /// expanded to form a stack of stashed values using std::variant() for value
  /// field.
  ///
  /// @param key The C-style string that represents the key of the tuple. It
  /// must
  ///             not be a null pointer.
  /// @param value The 64-bit unsigned integer value to be associated with the
  ///              key.
  ///
  /// @return Returns `xpti::result_t::XPTI_RESULT_SUCCESS` if the operation is
  ///         successful. If the key is a null pointer, it returns
  ///         `xpti::result_t::XPTI_RESULT_FAIL`.
  xpti::result_t stashTuple(const char *key, uint64_t value) {
    if (!key)
      return xpti::result_t::XPTI_RESULT_FAIL;

    std::get<0>(g_tls_stash_tuple) = key;
    std::get<1>(g_tls_stash_tuple) = value;
    return xpti::result_t::XPTI_RESULT_SUCCESS;
  }

  /// @brief Retrieves a key-value pair stashed in thread-local storage (TLS).
  ///
  /// This function attempts to retrieve a previously stashed key-value pair
  /// from a global, thread-local storage (TLS) tuple. The key is a string, and
  /// the value is a 64-bit unsigned integer. This mechanism is used to access
  /// data that is specific to the current thread.
  ///
  /// @param key A pointer to a char pointer, which will be set to point to the
  ///            retrieved key. Must not be a null pointer.
  /// @param value A reference to a 64-bit unsigned integer, which will be set
  ///              to the retrieved value.
  ///
  /// @return Returns `xpti::result_t::XPTI_RESULT_SUCCESS` if the operation is
  ///         successful and the key-value pair is found. Returns
  ///         `xpti::result_t::XPTI_RESULT_INVALIDARG` if the input key pointer
  ///         is null, indicating an invalid argument. Returns
  ///         `xpti::result_t::XPTI_RESULT_NOTFOUND` if no key-value pair is
  ///         currently stashed in the TLS, indicating that the requested data
  ///         could not be found.
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

  /// @brief Clears the stashed key-value pair in the thread-local storage (TLS)
  /// tuple.
  ///
  /// This function checks if there is a stashed key-value pair in the global,
  /// thread-local storage (TLS) tuple and clears it if present. The key in the
  /// tuple is checked for its existence; if no key is stashed (i.e., the key is
  /// a null pointer), the function returns immediately without performing any
  /// action.
  void unstashTuple() {
    if (!std::get<0>(g_tls_stash_tuple))
      return;
  }

  /// @brief Wrapper for checking if there are subscribers to a specific stream
  /// and trace type
  /// @details This function implements a wrapper for the checking to see if
  /// there are any subscribers to a specific stream and trace type. The
  /// instruemtation can use this fine grain check mechanism to only allow parts
  /// of the code for which there are subscribers to take the instrumentation
  /// overhead.
  ///
  /// @param stream The identifier for the stream being queried. This could
  ///               represent different sources or categories of trace data.
  /// @param type The type of event being queried. This categorizes the events
  ///             within a stream, allowing for finer control over what is
  ///             traced.
  ///
  /// @return Returns `true` if tracing is enabled for the specified stream and
  ///         event type. Returns `false` otherwise.
  bool checkTraceEnabled(uint16_t stream, uint16_t type) {
    if (MTraceEnabled) {
      return MNotifier.checkSubscribed(stream, type);
    }
    return false;
  }

  /// @brief Wrapper for adding a metadata key-value pair to the metadata of a
  /// trace event
  xpti::result_t addMetadata(xpti::trace_event_data_t *Event, const char *Key,
                             object_id_t ValueID) {
    return MTracepoints.addMetadata(Event, Key, ValueID);
  }

  /// @brief Wrapper for creating a trace event
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

  /// @brief Wrapper for releasing a trace event
  void releaseEvent(xpti::trace_event_data_t *event) {
    MTracepoints.releaseEvent(event);
  }

  inline const xpti::trace_event_data_t *findEvent(uint64_t UniversalID) {
    if (UniversalID == xpti::invalid_uid)
      return nullptr;

    if (MTracepoints.isValidUID64(UniversalID)) {
      xpti::TracePointImpl *TP =
          reinterpret_cast<xpti::TracePointImpl *>(UniversalID);
      if (TP && xpti::is_valid_event(&TP->MEvent))
        return &TP->MEvent;
      else
        return nullptr;
    }

    return nullptr;
  }

  inline const xpti_trace_event_t *lookupEvent(uint64_t UId) {
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
      return TP->MUId.uid64;
    }

    return xpti::invalid_uid;
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

  xpti_tracepoint_t *registerTracepoint(const char *FuncName,
                                        const char *FileName, uint32_t LineNo,
                                        uint32_t ColumnNo) {
    xpti::payload_t Payload(FuncName, FileName, LineNo, ColumnNo, nullptr);
    if (!xpti::is_valid_payload(&Payload))
      Payload = xpti::unknown_payload();

    auto TP = MTracepoints.registerTracepoint(&Payload);
    if (xpti::is_valid_uid(TP->MUId) && xpti::is_valid_payload(&TP->MPayload) &&
        xpti::is_valid_event(&TP->MEvent)) {
      return dynamic_cast<xpti_tracepoint_t *>(TP);
    }

    return nullptr;
  }

  xpti::result_t deleteTracepoint(xpti_tracepoint_t *Tracepoint) {
    if (!Tracepoint)
      return xpti::result_t::XPTI_RESULT_INVALIDARG;

    return MTracepoints.deleteTracepoint(Tracepoint);
  }

  const xpti_tracepoint_t *registerTracepointScope(const char *FuncName,
                                                   const char *FileName,
                                                   uint32_t LineNo,
                                                   uint32_t ColumnNo) {
    g_tls_temp_scope_data =
        registerTracepoint(FuncName, FileName, LineNo, ColumnNo);
    return g_tls_temp_scope_data;
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
    if (!MTracepoints.isValidUID64(uid))
      return nullptr;

    xpti::TracePointImpl *TP = reinterpret_cast<xpti::TracePointImpl *>(uid);
    if (xpti::is_valid_payload(&TP->MPayload))
      return &TP->MPayload;

    return nullptr;
  }

  inline const xpti_payload_t *lookupPayload(uint64_t uid) {
    if (!MTracepoints.isValidUID64(uid))
      return nullptr;
    xpti::TracePointImpl *TP = reinterpret_cast<xpti::TracePointImpl *>(uid);
    if (TP && xpti::is_valid_payload(&TP->MPayload))
      return TP->payload();
    else
      return nullptr;
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

/// @brief Initializes the XPTI tracing framework.
///
/// This function is responsible for initializing the XPTI tracing framework. It
/// is designed to be thread-safe and can be called multiple times. Each call to
/// `xptiFrameworkInitialize` increments an internal reference counter, ensuring
/// that the framework is initialized only once but can be safely referenced
/// multiple times throughout the application.
///
/// @note This function uses a spin lock to ensure thread safety during the
/// initialization process.

XPTI_EXPORT_API void xptiFrameworkInitialize() {
  // Locks the framework mutex to ensure thread-safe access to the reference
  // counter.
  std::lock_guard<xpti::utils::SpinLock> guard{xpti::g_framework_mutex};
  // Increments the global framework reference counter.
  xpti::GFrameworkReferenceCounter++;
}

/// @brief Finalizes the XPTI tracing framework.
///
/// This function is responsible for finalizing the XPTI tracing framework. It
/// decrements an internal reference counter each time it is called, and when
/// the reference counter reaches zero, it triggers the release of resources
/// allocated by the framework. This design allows for multiple components to
/// safely use the framework and ensures that cleanup occurs only after all
/// components have finished using it.
///
/// The function is thread-safe, using a spin lock to protect access to the
/// global framework reference counter. This ensures that concurrent calls to
/// finalize the framework are serialized, preventing race conditions during
/// cleanup.
///
/// @note It is important to match each call to `xptiFrameworkInitialize` with a
/// call to `xptiFrameworkFinalize` to ensure proper cleanup of the framework's
/// resources.

XPTI_EXPORT_API void xptiFrameworkFinalize() {
  // Locks the framework mutex to ensure thread-safe access to the reference
  // counter.
  std::lock_guard<xpti::utils::SpinLock> guard{xpti::g_framework_mutex};

  // Decrements the global framework reference counter.
  xpti::GFrameworkReferenceCounter--;

  // If the reference counter reaches zero, release the framework resources.
  if (xpti::GFrameworkReferenceCounter == 0) {
    xpti::Framework::release();
  }
}

/// @brief Initializes an XPTI tracing framework data stream.
///
/// This function is responsible for initializing an XPTI tracing framework
/// stream identifies by its name, and a major version, minor version, and
/// version string for handling the stream differently, if the semantics of the
/// stream change. It must be called before any other XPTI functions are used to
/// send data over the stream.
///
/// @param Stream A null-terminated string representing the name of the stream
/// to be initialized. Streams are logical channels that can be used to
/// categorize trace data.
/// @param maj The major version number of the XPTI API that the application is
/// targeting. This is used to ensure compatibility between the application and
/// the XPTI framework.
/// @param min The minor version number of the XPTI API that the application is
/// targeting. This provides additional granularity for version compatibility.
/// @param version A null-terminated string representing the version of the XPTI
/// framework being initialized. This may include build or revision information
/// beyond the major and minor version numbers.
///
/// @return Returns a result code indicating the success or failure of the
/// initialization process. Possible return values are defined in the
/// `xpti::result_t` enumeration. A success code indicates that the framework
/// was initialized successfully, while a failure code indicates an error
/// occurred during initialization.

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

/// @brief Finalizes a specific stream in the XPTI tracing framework.
///
/// This function is responsible for finalizing or cleaning up a specific stream
/// identified by the Stream parameter within the XPTI tracing framework. It
/// should be called when the application no longer needs to capture trace data
/// for the specified stream, allowing the framework to release any resources
/// associated with that stream. This is an essential step in ensuring that the
/// application cleanly shuts down and does not leave any resources allocated.
///
/// @param Stream A null-terminated string representing the name of the stream
/// to be finalized. This should match the name of a stream that was previously
/// initialized and used for capturing trace data.
///
/// @note This function does not return a value and assumes that finalizing the
/// stream is always a safe operation. However, it is the caller's
/// responsibility to ensure that this function is called at an appropriate time
/// when the stream is no longer in use.

XPTI_EXPORT_API void xptiFinalize(const char *Stream) {
  xpti::Framework::instance().finalizeStream(Stream);
}

/// @brief Retrieves the 64-bit universal ID for the current scope, if published
///
/// In a given tracing scope, the framework publishes the currenbt universal ID
/// through newer scoped classes so that downstream API can leverage the ID for
/// generating their own metadata with the universal ID as the key.
///
/// @return Returns the stashed 64-bit universal ID

XPTI_EXPORT_API uint64_t xptiGetUniversalId() {
  return xpti::Framework::instance().getUniversalID();
}

/// @brief Publishes the 64-bit universal ID for the current scope
///
/// In a given tracing scope, the framework publishes the currenbt universal ID
/// through newer scoped classes so that downstream API can leverage the ID for
/// generating their own metadata with the universal ID as the key.
///
/// @param uid  The 64-bit universal ID to be published

XPTI_EXPORT_API void xptiSetUniversalId(uint64_t uid) {
  xpti::Framework::instance().setUniversalID(uid);
}

/// @brief Stashes a key-value pair for later retrieval.
///
/// This function is designed to store a key-value pair in a global TLS stash,
/// allowing for the association of a string key with a 64-bit unsigned integer
/// value. This can be used within the tracing framework to temporarily store
/// and later retrieve metadata or other information that needs to be associated
/// with specific trace event scope.
///
/// @param key A null-terminated string representing the key of the tuple. The
/// key should be unique to ensure correct retrieval of the associated value.
/// @param value A 64-bit unsigned integer representing the value to be
/// associated with the key.
///
/// @return Returns a result code indicating the success or failure of the
/// operation. Possible return values are defined in the `xpti::result_t`
/// enumeration.

XPTI_EXPORT_API xpti::result_t xptiStashTuple(const char *key, uint64_t value) {
  return xpti::Framework::instance().stashTuple(key, value);
}

/// @brief Retrieves a stashed key-value pair by its key.
///
/// This function is designed to retrieve the value associated with a given key
/// that was previously stashed using `xptiStashTuple`. It is intended for use
/// within the tracing framework to access metadata or other information that
/// has been associated with a specific trace event scope.
///
/// @param[out] key A pointer referes so the key can be returned back to the
/// caller
/// @param[out] value A pointer to a 64-bit unsigned integer where the function
/// will store the value associated with the key, if it has been set.
///
/// @return Returns a result code indicating the success or failure of the
/// operation. Possible return values are defined in the `xpti::result_t`
/// enumeration. A success code indicates that the key -value pair was stashed
/// and was found. A failure code indicates that the key-value pair was not
/// stashed.

XPTI_EXPORT_API xpti::result_t xptiGetStashedTuple(char **key,
                                                   uint64_t &value) {
  return xpti::Framework::instance().getStashedTuple(key, value);
}

/// @brief Clears stashed key-value pair from the global stash.
///
/// This function is designed to unset the key-value pair that has been
/// previously stashed using `xptiStashTuple`. It is intended for use within the
/// tracing framework to manage the lifecycle of metadata or other information
/// associated with a trace event scope.
///
/// @note This function does not return a value and does not provide error
/// handling. It is assumed that clearing the stash is always a safe operation
/// and usually called by a scoped class destructor.

XPTI_EXPORT_API void xptiUnstashTuple() {
  xpti::Framework::instance().unstashTuple();
}

/// @brief Registers a user-defined trace point with the XPTI framework.
///
/// This function allows tools or libraries to register their own custom trace
/// point types with the XPTI tracing framework. A trace point type is a
/// specific type od a trace event meaningful for the tool or the user.
///
/// @param ToolName A null-terminated string representing the name of the tool
/// or library that is registering the trace point. This name is used by the
/// XPTI framework to categorize and manage trace point types across different
/// tools.
/// @param UserDefinedTP An 8-bit unsigned integer representing the identifier
/// of the user-defined trace point type. This identifier should be unique
/// within the context of the tool or library to avoid conflicts with other
/// trace points.
///
/// @return Returns a 16-bit unsigned integer that represents a globally unique
/// identifier for the registered trace point within the XPTI framework. This
/// identifier can be used in subsequent calls to the XPTI API to refer to the
/// registered trace point.
///
/// @note It is the responsibility of the tool or library to ensure that the
/// UserDefinedTP values are unique within their domain to prevent registration
/// conflicts. Also, the function name should be changed to
/// `xptiRegisterUserDefinedTracePointType` to convey its true intent.

XPTI_EXPORT_API uint16_t
xptiRegisterUserDefinedTracePoint(const char *ToolName, uint8_t UserDefinedTP) {
  uint8_t ToolID = xpti::Framework::instance().registerVendor(ToolName);
  UserDefinedTP |= (uint8_t)xpti::trace_point_type_t::user_defined;
  uint16_t UserDefTracepoint = XPTI_PACK08_RET16(ToolID, UserDefinedTP);

  return UserDefTracepoint;
}

/// @brief Registers a user-defined event type with the XPTI framework.
///
/// This function allows tools or libraries to register their own custom event
/// types with the XPTI tracing framework. An event type is a categorization of
/// events that share common characteristics or purposes. By registering a
/// user-defined event type, tools can integrate more closely with the XPTI
/// framework, enabling detailed performance analysis and debugging capabilities
/// through the categorization of events.
///
/// @param ToolName A null-terminated string representing the name of the tool
/// or library that is registering the event type. This name is used by the XPTI
/// framework to categorize and manage event types across different tools.
/// @param UserDefinedEvent An 8-bit unsigned integer representing the
/// identifier of the user-defined event type. This identifier should be unique
/// within the context of the tool or library to avoid conflicts with other
/// event types.
///
/// @return Returns a 16-bit unsigned integer that represents a globally unique
/// identifier for the registered event type within the XPTI framework. This
/// identifier can be used in subsequent calls to the XPTI API to refer to the
/// registered event type.
///
/// @note It is the responsibility of the tool or library to ensure that the
/// UserDefinedEvent values are unique within their domain to prevent
/// registration conflicts.

XPTI_EXPORT_API uint16_t xptiRegisterUserDefinedEventType(
    const char *ToolName, uint8_t UserDefinedEvent) {
  uint8_t ToolID = xpti::Framework::instance().registerVendor(ToolName);
  UserDefinedEvent |= (uint8_t)xpti::trace_event_type_t::user_defined;
  uint16_t UserDefEventType = XPTI_PACK08_RET16(ToolID, UserDefinedEvent);
  return UserDefEventType;
}

/// @brief Generates a unique 64-bit value.
///
/// This function is designed to generate a unique 64-bit value that can be used
/// within the XPTI tracing framework for managing correlation ID for scoped
/// function calls (function_begin/function_end).
///
/// @return Returns a 64-bit unsigned integer representing the unique
/// identifier. The value starts from 1 and increments with each call to this
/// function.

XPTI_EXPORT_API uint64_t xptiGetUniqueId() {
  return xpti::Framework::instance().makeUniqueID();
}

/// @brief Registers a string with the XPTI framework and returns its string ID.
///
/// This function is designed to register a string with the XPTI framework. It
/// ensures that each unique string is stored only once in the framework's
/// internal database, reducing memory usage and improving lookup efficiency for
/// string-based operations within the tracing infrastructure. The function
/// returns a unique identifier for the registered string, which can be used in
/// subsequent API calls to refer to the string without passing the string
/// itself.
///
/// @param String A null-terminated C string that is to be registered with the
/// XPTI framework. This string could represent a variable name, a function
/// name, or any other entity that needs to be identified in the tracing data.
/// @param RefTableStr A pointer to a character pointer. After the function
/// call, this pointer will point to the internal copy of the string stored by
/// the XPTI framework. This allows for efficient reuse of the string's internal
/// representation without additional allocations.
///
/// @return Returns a unique identifier (of type xpti::string_id_t) for the
/// registered string. This identifier is a non-negative integer that uniquely
/// represents the string within the XPTI framework's internal structures.
///
/// @note If the string has already been registered, this function will not
/// create a duplicate entry but will return the existing identifier for the
/// string. This makes the function safe to call multiple times with the same
/// string.

XPTI_EXPORT_API xpti::string_id_t xptiRegisterString(const char *String,
                                                     char **RefTableStr) {
  return xpti::Framework::instance().registerString(String, RefTableStr);
}

/// @brief Looks up and returns the string associated with a given string ID.
///
/// This function is part of the XPTI framework and is used to retrieve a
/// previously registered string by its unique identifier. The unique identifier
/// is typically obtained through calls to functions like `xptiRegisterString`,
/// which register strings with the XPTI framework and return a unique
/// identifier for each. This mechanism allows for efficient string management
/// and retrieval within the framework, facilitating the tracking of performance
/// metrics and other tracing information.
///
/// @param ID The unique identifier of the string to be retrieved. This
/// identifier must have been previously obtained by registering the string with
/// the XPTI framework using `xptiRegisterString` or similar functions.
///
/// @return Returns a pointer to the null-terminated C string associated with
/// the given identifier. If the identifier does not correspond to a registered
/// string, the function returns `nullptr`.
///
/// @note The returned string pointer points to internal storage managed by the
/// XPTI framework and must not be modified or freed by the caller. The lifetime
/// of the returned string is managed by the framework, and it remains valid
/// until the string is explicitly deregistered or the framework is shut down.

XPTI_EXPORT_API const char *xptiLookupString(xpti::string_id_t ID) {
  return xpti::Framework::instance().lookupString(ID);
}

/// @brief Registers an object with the XPTI framework and returns its unique
/// identifier.
///
/// This function is designed to register various types of objects (e.g., data
/// structures, strings, kernels) with the XPTI framework. It allows for the
/// tracking and identification of these objects within the performance tracing
/// infrastructure. Each object is registered with a type and associated data,
/// enabling detailed performance analysis and debugging capabilities. This is
/// primarily used to store metadata associated with a tracepoint.
///
/// @param Data A pointer to the data representing the object to be registered.
/// This could be a pointer to a data structure, function, or any other entity
/// that needs to be identified and tracked by the XPTI framework.
/// @param Size The size of the data pointed to by `Data`. This parameter is
/// necessary to accurately manage the memory and identification of the object
/// within the framework.
/// @param Type An 8-bit unsigned integer representing the type of the object
/// being registered. The type categorizes the object for tracking and analysis
/// purposes.
///
/// @return Returns a unique identifier (of type xpti::object_id_t) for the
/// registered object. This identifier is a non-negative integer that uniquely
/// represents the object within the XPTI framework's internal structures.

XPTI_EXPORT_API xpti::object_id_t
xptiRegisterObject(const char *Data, size_t Size, uint8_t Type) {
  return xpti::Framework::instance().registerObject(Data, Size, Type);
}

/// @brief Retrieves the data associated with a registered object by its unique
/// ID.
///
/// This function is a part of the XPTI  framework and is used to retrieve the
/// data of an object that has been previously registered with the framework
/// using `xptiRegisterObject`. The function facilitates the retrieval of object
/// data, allowing for introspection and analysis of the object within the
/// performance tracing infrastructure. It is primarily used to pack user
/// defined data types into the metadata associated with a tracepoint.
///
/// @param ID The unique identifier of the object whose data is to be retrieved.
/// This identifier must have been obtained through a previous call to
/// `xptiRegisterObject` when the object was registered with the XPTI framework.
///
/// @return Returns an `xpti::object_data_t` structure that contains the data
/// associated with the object identified by `ID`. If the identifier does not
/// correspond to a registered object, the function returns an empty
/// `xpti::object_data_t` structure with its members set to default values.
///

XPTI_EXPORT_API xpti::object_data_t xptiLookupObject(xpti::object_id_t ID) {
  return xpti::Framework::instance().lookupObject(ID);
}

/// @brief Registers a tracing stream with the XPTI framework and returns its
/// unique identifier.
///
/// This function is part of the XPTI framework, designed to facilitate the
/// creation and management of tracing streams. Tracing streams are logical
/// channels through which performance data and tracing information are
/// collected and organized. By registering a stream, users can categorize and
/// separate data for different components or aspects of their application,
/// enhancing the analysis and debugging process.
///
/// @param StreamName A null-terminated C string representing the name of the
/// stream to be registered. This name is used to uniquely identify the stream
/// within the XPTI framework and can be used in subsequent API calls to refer
/// to this stream.
///
/// @return Returns an 8-bit unsigned integer that serves as a unique identifier
/// for the registered stream. This identifier is used in subsequent API calls
/// to refer to the stream. If the stream cannot be registered (e.g., due to a
/// name collision or memory constraints), a predefined constant (such as
/// XPTI_STREAM_ID_INVALID) is returned to indicate failure.
///
/// @note The function performs a check to ensure that the stream name is unique
/// within the framework. If a stream with the given name already exists, the
/// function will return the identifier of the existing stream instead of
/// creating a new one.

XPTI_EXPORT_API uint8_t xptiRegisterStream(const char *StreamName) {
  return xpti::Framework::instance().registerStream(StreamName);
}

/// @brief Unregisters a previously registered tracing stream from the XPTI
/// framework.
///
/// This function is a part of the XPTI framework. It is used to unregister a
/// tracing stream that was previously registered using `xptiRegisterStream`.
/// Unregistering a stream removes it from the XPTI framework's internal
/// management, effectively disabling any further tracing or performance data
/// collection through that stream. This is useful for cleanup purposes or when
/// the tracing needs for a particular component or aspect of the application
/// have ended.
///
/// @param StreamName A null-terminated C string representing the name of the
/// stream to be unregistered. This name must match the name used during the
/// stream's registration with `xptiRegisterStream`.
///
/// @return Returns a result code of type `xpti::result_t`. A successful
/// operation returns `xpti::result_t::XPTI_RESULT_SUCCESS`. If the stream
/// cannot be found or if an error occurs during the unregistration process, an
/// appropriate error code is returned.
///
/// @note It is important to ensure that no ongoing tracing activities are using
/// the stream being unregistered. Attempting to unregister a stream while it is
/// still in use may lead to loss of performance data.

XPTI_EXPORT_API xpti::result_t xptiUnregisterStream(const char *StreamName) {
  return xpti::Framework::instance().unregisterStream(StreamName);
}

/// @brief Registers a payload with the XPTI framework and returns a 64-bit
/// unique identifier.
///
/// This function is part of the XPTI framework. It is designed to register a
/// payload, which typically contains metadata about a specific code region
/// (such as a loop or function) or a task, with the framework. Once registered,
/// the payload can be associated with various performance events or traces,
/// enabling detailed performance analysis and debugging. This will however be
/// deprecated in favor od xptiCreateTracepoint() which combines
/// xptiRegisterPayload and xptiMakeEvent to ensure data consistency in the new
/// architecture.
///
/// @param payload A pointer to an `xpti::payload_t` structure that contains the
/// metadata to be registered. This structure includes information such as the
/// source file name, function name, and line number, among other details. The
/// payload must be properly initialized before calling this function.
///
/// @return Returns a 64-bit unsigned integer that serves as a unique identifier
/// for the registered payload. This identifier can be used in subsequent API
/// calls to refer to the payload.
///
/// @note TThe lifetime of a registered paload is the duration of the
/// application since a give payload or code location can be revisted at any
/// time in the application run. The unique 128-bit ID for a given payload is
/// invariant, but the 64-bit ID generated from this will change with each
/// instance so it can be used to identify the instance.

XPTI_EXPORT_API uint64_t xptiRegisterPayload(xpti::payload_t *payload) {
  // Will be deprectaed and replaced by registerTracepoint()
  return xpti::Framework::instance().registerPayload(payload);
}

/// @brief Creates a tracepoint and returns a pointer to an interface
/// representing the tracepoint.
///
/// The function call is designed to create a tracepoint, which is a
/// specific point in the code (such as the entry or exit of a function) that is
/// instrumented for collecting performance data or for tracing program
/// execution. Tracepoints are fundamental to performance analysis and
/// debugging, allowing developers to pinpoint areas of interest within their
/// code. Using the tracepoint interface, one can have access to the payload and
/// trace event interfaces along with the 64-bit universal ID.
///
/// @param FuncName A null-terminated C string representing the name of the
/// function where the tracepoint is created. This information is used to
/// identify the tracepoint in performance reports and analysis tools.
///
/// @param FileName A null-terminated C string representing the name of the
/// source file where the tracepoint is created. This helps in locating the
/// tracepoint within the codebase.
///
/// @param LineNo A 32-bit unsigned integer representing the line number in the
/// source file where the tracepoint is created. This provides precise location
/// information for the tracepoint.
///
/// @param ColumnNo A 32-bit unsigned integer representing the column number on
/// the line specified by LineNo where the tracepoint is created. This offers
/// even more precise location information within the code.
///
/// @return Returns a pointer to the created `xpti_tracepoint_t` structure,
/// which contains the metadata for the tracepoint. If the tracepoint cannot be
/// created (e.g., due to memory constraints), a null pointer is returned.
///
/// @note It is important to manage the memory for the created tracepoint
/// appropriately. Depending on the implementation, you may need to ensure that
/// the tracepoint is destroyed or unregistered to manage memory growth. The
/// scoped classes such as tracepoint_scope_t will automatically create a trace
/// point and delete it when it goes out of scope.
///
XPTI_EXPORT_API xpti_tracepoint_t *xptiCreateTracepoint(const char *FuncName,
                                                        const char *FileName,
                                                        uint32_t LineNo,
                                                        uint32_t ColumnNo) {
  auto &FW = xpti::Framework::instance();
  return FW.registerTracepoint(FuncName, FileName, LineNo, ColumnNo);
}

/// @brief Deletes a tracepoint that was previously created.
///
/// This function is responsible for deleting a tracepoint that was previously
/// created using `xptiCreateTracepoint`. Deleting a tracepoint is necessary to
/// free up resources and to determine the end scope of a tracepoint.
///
/// @param TP A pointer to the `xpti_tracepoint_t` interface representing the
/// tracepoint to be deleted.
///
/// @return Returns a result code of type `xpti::result_t`. A successful
/// operation returns `xpti::result_t::XPTI_RESULT_SUCCESS`. If the tracepoint
/// cannot be found or if an error occurs during the deletion process, an
/// appropriate error code is returned.

XPTI_EXPORT_API xpti::result_t xptiDeleteTracepoint(xpti_tracepoint_t *TP) {
  auto &FW = xpti::Framework::instance();
  return FW.deleteTracepoint(TP);
}

/// @brief Creates or retrieves a trace event based on the provided parameters.
///
/// This function is designed to either create a new trace event or retrieve an
/// existing one based on the provided name, payload, event type, and activity
/// type. If the event already exists, it increments the instance number to
/// differentiate between instances of the same event. This usage mode will be
/// deprecated and replaced by xptiCreateTracepoint() which will manage the
/// instances automatically and return a new trace event for each instance. The
/// API is provided for backward compatibility.
///
/// @param Name A null-terminated string representing the name of the event.
/// This parameter is currently unused in this implementation but reserved for
/// future use. It can be used to identify the event in a human-readable form.
/// @param Payload A pointer to a `xpti::payload_t` structure that contains
/// metadata about the event, such as the source file name, function name, and
/// line number. This information is crucial for identifying the source of the
/// event.
/// @param Event A 16-bit unsigned integer representing the event type. This
/// type is user-defined and can be used to categorize events into different
/// types for easier management and analysis.
/// @param Activity A value of type `xpti::trace_activity_type_t` representing
/// the activity associated with the event. This parameter specifies the kind of
/// activity that the event is tracking, such as a function call, loop
/// execution, etc.
/// @param InstanceNo A pointer to a 64-bit unsigned integer that will be
/// incremented to provide a unique instance number for the event. This
/// parameter is optional and can be null if instance numbering is not required.
/// Instance numbers are useful for distinguishing between multiple occurrences
/// of the same event.
///
/// @return Returns a pointer to a `xpti::trace_event_data_t` structure
/// representing the created or retrieved event. If the event cannot be created
/// or retrieved, returns nullptr.

XPTI_EXPORT_API xpti::trace_event_data_t *
xptiMakeEvent(const char * /*Name*/, xpti::payload_t *Payload, uint16_t Event,
              xpti::trace_activity_type_t Activity, uint64_t *InstanceNo) {
  auto &FW = xpti::Framework::instance();
  auto RetEv = FW.newEvent(Payload, InstanceNo, Event, Activity);
  return RetEv;
}

/// @brief Resets the XPTI framework by clearing all stored data.
///
/// This function is responsible for resetting the state of the XPTI framework.
/// It is primarily provided for use during shutdown or for testing frameworks.
///
/// @note This function is particularly useful for scenarios where the framework
/// needs to be reinitialized without restarting the application, such as
/// between test runs in a suite. Care should be taken when calling this
/// function to ensure that no other part of the application is currently using
/// the XPTI framework, as it will remove all existing data and state.

XPTI_EXPORT_API void xptiReset() { xpti::Framework::instance().clear(); }

/// @brief Retrieves a pointer to a trace event data structure based on a unique
/// identifier.
///
/// This function searches for a trace event within the XPTI framework using a
/// unique identifier (UId) provided as input. If the event is found, a pointer
/// to the `xpti::trace_event_data_t` structure representing the event is
/// returned. This structure contains detailed information about the event,
/// including its name, type, and associated payload. If no event with the
/// specified UId exists, the function returns nullptr, indicating that the
/// search was unsuccessful. This API will be deprecated and replaced with
/// xptiLookupEvent() which returns an interface for accessing the trace event
/// data structure and will provide better ABI compatibility with changes to the
/// trace evenbt data structure.
///
/// @param UId A 64-bit unsigned integer representing the unique identifier of
/// the trace event to be retrieved. This identifier is typically assigned when
/// the event is created and is used to uniquely identify the event within the
/// XPTI framework.
///
/// @return A pointer to the `xpti::trace_event_data_t` structure representing
/// the found event, or nullptr if no event with the specified UId exists.

XPTI_EXPORT_API const xpti::trace_event_data_t *xptiFindEvent(uint64_t UId) {
  return xpti::Framework::instance().findEvent(UId);
}

/// @brief Queries the payload information for a trace event by its unique
/// identifier (UID).
///
/// This function searches the XPTI framework's internal data structures for a
/// trace event that matches the given unique identifier (UID). If such an event
/// is found, the function returns a pointer to the `xpti::payload_t` structure
/// associated with the event. This structure contains metadata about the event,
/// such as the source file name, function name, line number, and other relevant
/// information that was provided when the event was registered. If no event
/// with the specified UID is found, the function returns nullptr.
///
/// @param uid A 64-bit unsigned integer representing the unique identifier of
/// the trace event whose payload is being queried. This UID is typically
/// assigned by the XPTI framework when the event is registered.
///
/// @return A constant pointer to the `xpti::payload_t` structure containing the
/// metadata of the event associated with the given UID, or nullptr if no
/// matching event is found.

XPTI_EXPORT_API const xpti::payload_t *xptiQueryPayloadByUID(uint64_t uid) {
  return xpti::Framework::instance().queryPayloadByUID(uid);
}

/// @brief Retrieves the payload associated with a given trace event.
///
/// This function is designed to extract the payload information from a trace
/// event represented by the `xpti::trace_event_data_t` structure. The payload
/// contains metadata about the event, such as the source file name, function
/// name, line number, and any additional information that was provided when the
/// event was registered.
///
/// @param LookupObject A pointer to the `xpti::trace_event_data_t` structure
/// representing the trace event for which the payload information is being
/// queried.
///
/// @return A constant pointer to the `xpti::payload_t` structure containing the
/// metadata of the event. If the `LookupObject` is null or the payload cannot
/// be found, the function returns nullptr.

XPTI_EXPORT_API const xpti::payload_t *
xptiQueryPayload(xpti::trace_event_data_t *LookupObject) {
  return xpti::Framework::instance().queryPayload(LookupObject);
}

/// @brief Registers a callback function for a specific trace type on a given
/// stream.
///
/// This function allows the user to register a callback function that will be
/// invoked whenever a trace event of the specified type occurs on the specified
/// stream.
///
/// @param StreamID A uint8_t value representing the identifier of the stream on
/// which the callback function is to be registered.
/// @param TraceType A uint16_t value representing the type of trace event that
/// the callback function is interested in.
/// @param cbFunc A function pointer to the callback function that conforms to
/// the `xpti::tracepoint_callback_api_t` signature. This callback function will
/// be called whenever a trace event of the specified type occurs on the
/// specified stream.
///
/// @return A result code of type `xpti::result_t` indicating the success or
/// failure of the callback registration. Possible return values include
/// `XPTI_RESULT_SUCCESS` if the callback was successfully registered, or
/// `XPTI_RESULT_DUPLICATE` if a callback is being registered for the same
/// stream and trace type.
///
/// @note It is possible to register multiple callbacks for the same trace type
/// on the same stream, in which case they will be invoked in the order they
/// were registered. Care should be taken to ensure that callback functions are
/// efficient and do not introduce significant performance overhead.

XPTI_EXPORT_API xpti::result_t
xptiRegisterCallback(uint8_t StreamID, uint16_t TraceType,
                     xpti::tracepoint_callback_api_t cbFunc) {
  return xpti::Framework::instance().registerCallback(StreamID, TraceType,
                                                      cbFunc);
}

/// @brief Unregisters a previously registered callback function for a specific
/// trace type on a given stream.
///
/// This function is used to remove a callback function that was previously
/// registered with `xptiRegisterCallback` for a specific trace event type on a
/// specified stream.
///
/// @param StreamID A uint8_t value representing the identifier of the stream
/// from which the callback function is to be unregistered.
/// @param TraceType A uint16_t value representing the type of trace event for
/// which the callback function was registered.
/// @param cbFunc A function pointer to the callback function that conforms to
/// the `xpti::tracepoint_callback_api_t` signature. This is the callback
/// function that will be unregistered and will no longer be called when the
/// specified trace event occurs on the specified stream.
///
/// @return A result code of type `xpti::result_t` indicating the success or
/// failure of the callback unregistration. Possible return values include
/// `XPTI_RESULT_SUCCESS` if the callback was successfully unregistered, or an
/// error code indicating the reason for failure.

XPTI_EXPORT_API xpti::result_t
xptiUnregisterCallback(uint8_t StreamID, uint16_t TraceType,
                       xpti::tracepoint_callback_api_t cbFunc) {
  return xpti::Framework::instance().unregisterCallback(StreamID, TraceType,
                                                        cbFunc);
}

/// @brief Notifies subscribers about a trace event occurrence.
///
/// This function is used to notify all registered subscribers about the
/// occurrence of a trace event. Subscribers are notified based on the stream ID
/// and trace type they have registered interest in. This mechanism allows for
/// efficient filtering and handling of trace events for performance monitoring
/// and analysis.
///
/// @param StreamID A uint8_t value representing the identifier of the stream on
/// which the event occurred.
/// @param TraceType A uint16_t value representing the type of the trace event.
///
/// @param Parent A pointer to a `xpti::trace_event_data_t` structure
/// representing the parent event, if any, of the current event. This allows for
/// the construction of a hierarchy or a chain of related events.
/// @param Object A pointer to a `xpti::trace_event_data_t` structure
/// representing the current event that is being notified to the subscribers.
/// @param InstanceNo A uint64_t value representing the instance number of the
/// event or used as correclation ID for scoped calls such as
/// function_begin/function_end. This can be used to differentiate between
/// instances of events that occur multiple times.
/// @param TemporalUserData A pointer to a constant void that represents
/// user-defined data associated with the event. This can be used to pass
/// additional information to the subscribers.
///
/// @return A result code of type `xpti::result_t` indicating the success or
/// failure of the notification process. Possible return values include
/// `XPTI_RESULT_SUCCESS` if the notification was successfully delivered to all
/// relevant subscribers, or an error code indicating the reason for failure.

XPTI_EXPORT_API xpti::result_t
xptiNotifySubscribers(uint8_t StreamID, uint16_t TraceType,
                      xpti::trace_event_data_t *Parent,
                      xpti::trace_event_data_t *Object, uint64_t InstanceNo,
                      const void *TemporalUserData) {
  return xpti::Framework::instance().notifySubscribers(
      StreamID, TraceType, Parent, Object, InstanceNo, TemporalUserData);
}

/// @brief Checks if tracing is enabled in the XPTI framework.
///
/// This function queries the current state of the XPTI framework to determine
/// if tracing is enabled. This is a higher order check basedon whether
/// XPTI_TRACE_ENABLE=1, XPTI_FRAMEWORK_DISPATCHER has been set to a valid
/// dispatcher and XPTI_SUBSCIBERS have a valid subscriber.
///
/// @return A boolean value indicating the current tracing state. Returns `true`
/// if tracing is enabled, and `false` otherwise.

XPTI_EXPORT_API bool xptiTraceEnabled() {
  return xpti::Framework::instance().traceEnabled();
}

/// @brief Checks if tracing for a specific stream and trace type is enabled in
/// the XPTI framework.
///
/// This function queries the XPTI framework to determine
/// if tracing is enabled for a specific stream and trace type.  This allows for
/// finer grain control over managing instrumentation and the resulting
/// performance overheads. Tracing is selectively enabled or disabled for
/// different streams and types of trace events, based on whether there are
/// actual subscribers consuming data from the stream and trace type.
///
/// @param stream A uint16_t value representing the identifier of the stream for
/// which the tracing status is being queried.
/// @param ttype A uint16_t value representing the type of the trace event for
/// which the tracing status is being queried.
///
/// @return A boolean value indicating the current tracing status for the
/// specified stream and trace type. Returns `true` if tracing is enabled, and
/// `false` otherwise.

XPTI_EXPORT_API bool xptiCheckTraceEnabled(uint16_t stream, uint16_t ttype) {
  return xpti::Framework::instance().checkTraceEnabled(stream, ttype);
}

/// @brief Adds metadata to a trace event.
///
/// This function allows for the association of additional metadata with a trace
/// event in the XPTI framework. Metadata is added in the form of key-value
/// pairs, where the key is a string and the value is an object identifier (ID).
/// This can be used to attach supplementary information to events, enhancing
/// their descriptiveness and utility for performance analysis.
///
/// @param Event A pointer to the `xpti::trace_event_data_t` structure
/// representing the trace event to which metadata is to be added.
/// @param Key A constant character pointer representing the key of the metadata
/// to be added. The key is a string that describes the nature of the metadata.
/// @param ID An `xpti::object_id_t` value representing the object identifier
/// that serves as the value of the metadata.
///
/// @return A result code of type `xpti::result_t` indicating the success or
/// failure of the operation. Possible return values include
/// `XPTI_RESULT_SUCCESS` if the metadata was successfully added to the event,
/// or an error code indicating the reason for failure.

XPTI_EXPORT_API xpti::result_t xptiAddMetadata(xpti::trace_event_data_t *Event,
                                               const char *Key,
                                               xpti::object_id_t ID) {
  return xpti::Framework::instance().addMetadata(Event, Key, ID);
}

/// @brief Queries the metadata associated with a trace event.
///
/// This function is part of the XPTI framework and allows querying the metadata
/// that has been associated with a specific trace event. Metadata is stored as
/// key-value pairs and can include additional information about the event such
/// as parameters, results, or any user-defined data that enhances the event's
/// descriptiveness for performance analysis or debugging purposes.
///
/// @param Event A pointer to the `xpti::trace_event_data_t` structure
/// representing the trace event for which the metadata is being queried.
///
/// @return A pointer to the `xpti::metadata_t` structure that contains the
/// metadata associated with the event. If no metadata is associated with the
/// event or if the event pointer is null, the function returns nullptr.

XPTI_EXPORT_API xpti::metadata_t *
xptiQueryMetadata(xpti::trace_event_data_t *Event) {
  return &Event->reserved.metadata;
}

/// @brief Forces the XPTI tracing state to be enabled or disabled.
///
/// This function directly sets the global tracing state within the
/// XPTIframework. It overrides any previous settings or default behavior
/// regarding whether tracing is enabled or disabled. This can be useful for
/// dynamically controlling the collection of trace data based on specific
/// conditions or phases in an application's execution. It is primarily provided
/// for use in testing frameworks.
///
/// @param YesOrNo A boolean value indicating the desired state of tracing. If
/// `true`, tracing will be enabled; if `false`, tracing will be disabled.

XPTI_EXPORT_API void xptiForceSetTraceEnabled(bool YesOrNo) {
  xpti::Framework::instance().setTraceEnabled(YesOrNo);
}

XPTI_EXPORT_API const xpti_trace_event_t *xptiLookupEvent(uint64_t UId) {
  auto &FW = xpti::Framework::instance();
  return FW.lookupEvent(UId);
}

XPTI_EXPORT_API const xpti_payload_t *xptiLookupPayload(uint64_t uid) {
  auto &FW = xpti::Framework::instance();
  return FW.lookupPayload(uid);
}

XPTI_EXPORT_API const xpti_tracepoint_t *xptiGetTracepointScopeData() {
  return g_tls_tracepoint_scope_data;
}

XPTI_EXPORT_API xpti::result_t
xptiSetTracepointScopeData(xpti_tracepoint_t *Data) {
  if (!Data)
    return xpti::result_t::XPTI_RESULT_INVALIDARG;
  // Copy to TLS so it is available for the remainder of the scope
  g_tls_tracepoint_scope_data = Data;
  // Also set Universal ID separately as it may be in use by older
  // implementations of tools, but this field is set in the incoming data only
  // if the legacy API are in use
  xptiSetUniversalId(Data->uid64());
  return xpti::result_t::XPTI_RESULT_SUCCESS;
}

XPTI_EXPORT_API void xptiUnsetTracepointScopeData() {
  g_tls_uid = xpti::invalid_uid;
  g_tls_tracepoint_scope_data = nullptr;
}

XPTI_EXPORT_API const xpti_tracepoint_t *
xptiRegisterTracepointScope(const char *FuncName, const char *FileName,
                            uint32_t LineNo, uint32_t ColumnNo) {
  return xpti::Framework::instance().registerTracepointScope(FuncName, FileName,
                                                             LineNo, ColumnNo);
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
