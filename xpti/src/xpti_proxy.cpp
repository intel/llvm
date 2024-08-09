//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
#include "xpti/xpti_trace_framework.hpp"

#include <atomic>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>

enum functions_t {
  XPTI_FRAMEWORK_INITIALIZE,
  XPTI_FRAMEWORK_FINALIZE,
  XPTI_INITIALIZE,
  XPTI_FINALIZE,
  XPTI_GET_UNIVERSAL_ID,
  XPTI_SET_UNIVERSAL_ID,
  XPTI_GET_UNIQUE_ID,
  XPTI_REGISTER_STRING,
  XPTI_LOOKUP_STRING,
  XPTI_REGISTER_OBJECT,
  XPTI_LOOKUP_OBJECT,
  XPTI_REGISTER_STREAM,
  XPTI_UNREGISTER_STREAM,
  XPTI_REGISTER_USER_DEFINED_TP,
  XPTI_REGISTER_USER_DEFINED_ET,
  XPTI_MAKE_EVENT,
  XPTI_FIND_EVENT,
  XPTI_QUERY_PAYLOAD,
  XPTI_REGISTER_CALLBACK,
  XPTI_UNREGISTER_CALLBACK,
  XPTI_NOTIFY_SUBSCRIBERS,
  XPTI_ADD_METADATA,
  XPTI_QUERY_METADATA,
  XPTI_TRACE_ENABLED,
  XPTI_REGISTER_PAYLOAD,
  XPTI_QUERY_PAYLOAD_BY_UID,
  XPTI_FORCE_SET_TRACE_ENABLED,
  XPTI_CHECK_TRACE_ENABLED,
  XPTI_RELEASE_EVENT,
  XPTI_STASH_TUPLE,
  XPTI_GET_STASHED_TUPLE,
  XPTI_UNSTASH_TUPLE,
  XPTI_ENABLE_TRACEPOINT_SCOPE_NOTIFICATION,
  XPTI_CHECK_TRACEPOINT_SCOPE_NOTIFICATION,
  XPTI_LOOKUP_PAYLOAD,
  XPTI_LOOKUP_EVENT,
  XPTI_CREATE_TRACEPOINT,
  XPTI_DELETE_TRACEPOINT,
  XPTI_GET_TRACEPOINT_SCOPE_DATA,
  XPTI_SET_TRACEPOINT_SCOPE_DATA,
  XPTI_UNSET_TRACEPOINT_SCOPE_DATA,
  XPTI_REGISTER_TRACEPOINT_SCOPE,
  XPTI_GET_DEFAULT_STREAM_ID,
  XPTI_SET_DEFAULT_STREAM_ID,
  XPTI_GET_DEFAULT_EVENT_TYPE,
  XPTI_SET_DEFAULT_EVENT_TYPE,
  XPTI_GET_DEFAULT_TRACE_TYPE,
  XPTI_SET_DEFAULT_TRACE_TYPE,
  // All additional functions need to appear before
  // the XPTI_FW_API_COUNT enum
  XPTI_FW_API_COUNT ///< This enum must always be the last one in the list
};

namespace xpti {
class ProxyLoader {
  std::unordered_map<int, const char *> m_function_names = {
      {XPTI_FRAMEWORK_INITIALIZE, "xptiFrameworkInitialize"},
      {XPTI_FRAMEWORK_FINALIZE, "xptiFrameworkFinalize"},
      {XPTI_INITIALIZE, "xptiInitialize"},
      {XPTI_FINALIZE, "xptiFinalize"},
      {XPTI_GET_UNIVERSAL_ID, "xptiGetUniversalId"},
      {XPTI_SET_UNIVERSAL_ID, "xptiSetUniversalId"},
      {XPTI_GET_UNIQUE_ID, "xptiGetUniqueId"},
      {XPTI_REGISTER_STRING, "xptiRegisterString"},
      {XPTI_LOOKUP_STRING, "xptiLookupString"},
      {XPTI_REGISTER_OBJECT, "xptiRegisterObject"},
      {XPTI_LOOKUP_OBJECT, "xptiLookupObject"},
      {XPTI_REGISTER_PAYLOAD, "xptiRegisterPayload"},
      {XPTI_REGISTER_STREAM, "xptiRegisterStream"},
      {XPTI_UNREGISTER_STREAM, "xptiUnregisterStream"},
      {XPTI_REGISTER_USER_DEFINED_TP, "xptiRegisterUserDefinedTracePoint"},
      {XPTI_REGISTER_USER_DEFINED_ET, "xptiRegisterUserDefinedEventType"},
      {XPTI_MAKE_EVENT, "xptiMakeEvent"},
      {XPTI_FIND_EVENT, "xptiFindEvent"},
      {XPTI_QUERY_PAYLOAD, "xptiQueryPayload"},
      {XPTI_QUERY_PAYLOAD_BY_UID, "xptiQueryPayloadByUID"},
      {XPTI_REGISTER_CALLBACK, "xptiRegisterCallback"},
      {XPTI_UNREGISTER_CALLBACK, "xptiUnregisterCallback"},
      {XPTI_NOTIFY_SUBSCRIBERS, "xptiNotifySubscribers"},
      {XPTI_ADD_METADATA, "xptiAddMetadata"},
      {XPTI_QUERY_METADATA, "xptiQueryMetadata"},
      {XPTI_TRACE_ENABLED, "xptiTraceEnabled"},
      {XPTI_CHECK_TRACE_ENABLED, "xptiCheckTraceEnabled"},
      {XPTI_FORCE_SET_TRACE_ENABLED, "xptiForceSetTraceEnabled"},
      {XPTI_STASH_TUPLE, "xptiStashTuple"},
      {XPTI_GET_STASHED_TUPLE, "xptiGetStashedTuple"},
      {XPTI_UNSTASH_TUPLE, "xptiUnstashTuple"},
      {XPTI_ENABLE_TRACEPOINT_SCOPE_NOTIFICATION,
       "xptiEnableTracepointScopeNotification"},
      {XPTI_CHECK_TRACEPOINT_SCOPE_NOTIFICATION,
       "xptiCheckTracepointScopeNotification"},
      {XPTI_LOOKUP_PAYLOAD, "xptiLookupPayload"},
      {XPTI_LOOKUP_EVENT, "xptiLookupEvent"},
      {XPTI_CREATE_TRACEPOINT, "xptiCreateTracepoint"},
      {XPTI_DELETE_TRACEPOINT, "xptiDeleteTracepoint"},
      {XPTI_GET_TRACEPOINT_SCOPE_DATA, "xptiGetTracepointScopeData"},
      {XPTI_SET_TRACEPOINT_SCOPE_DATA, "xptiSetTracepointScopeData"},
      {XPTI_UNSET_TRACEPOINT_SCOPE_DATA, "xptiUnsetTracepointScopeData"},
      {XPTI_REGISTER_TRACEPOINT_SCOPE, "xptiRegisterTracepointScope"},
      {XPTI_GET_DEFAULT_STREAM_ID, "xptiGetDefaultStreamID"},
      {XPTI_SET_DEFAULT_STREAM_ID, "xptiSetDefaultStreamID"},
      {XPTI_GET_DEFAULT_EVENT_TYPE, "xptiGetDefaultEventType"},
      {XPTI_SET_DEFAULT_EVENT_TYPE, "xptiSetDefaultEventType"},
      {XPTI_GET_DEFAULT_TRACE_TYPE, "xptiGetDefaultTraceType"},
      {XPTI_SET_DEFAULT_TRACE_TYPE, "xptiSetDefaultTraceType"},
      {XPTI_RELEASE_EVENT, "xptiReleaseEvent"}};

public:
  typedef std::vector<xpti_plugin_function_t> dispatch_table_t;

  ProxyLoader() : m_loaded(false), m_fw_plugin_handle(nullptr) {
    // When this object is created, we attempt to load
    // the share object implementation. We look for the
    // environment variable XPTI_FRAMEWORK_DISPATCHER to
    // see if it has been set. If not, all methods in
    // the proxy should end up being close to no-ops
    //
    tryToEnable();
  }

  ProxyLoader(const ProxyLoader &) = delete;
  ProxyLoader &operator=(const ProxyLoader &) = delete;

  ~ProxyLoader() {
    // If the loading of the framework library was
    // successful, we should close the handle in the
    // destructor to decrement the reference count
    // maintained by the loader.
    //
    if (m_fw_plugin_handle) {
      m_loader.unloadLibrary(m_fw_plugin_handle);
    }
  }

  inline bool noErrors() { return m_loaded; }

  void *functionByIndex(int index) {
    if (index >= XPTI_FRAMEWORK_INITIALIZE && index < XPTI_FW_API_COUNT) {
      return reinterpret_cast<void *>(m_dispatch_table[index]);
    }
    return nullptr;
  }

  static ProxyLoader &instance() {
    static ProxyLoader *loader = new ProxyLoader();
    return *loader;
  }
  // needed for testing
  void tryToEnable() {
    if (m_loaded)
      return;
    std::string env =
        m_loader.getEnvironmentVariable("XPTI_FRAMEWORK_DISPATCHER");
    if (!env.empty()) {
      std::string error;
      m_fw_plugin_handle = m_loader.loadLibrary(env.c_str(), error);
      if (m_fw_plugin_handle) {
        // We will defer changing m_loaded = true until the
        // end of this block after we are able to resolve
        // all of the entry points
        //
        m_dispatch_table.resize(XPTI_FW_API_COUNT);
        for (auto &func_name : m_function_names) {
          xpti_plugin_function_t func =
              m_loader.findFunction(m_fw_plugin_handle, func_name.second);
          if (!func) { // Return if we fail on even one function
            m_loader.unloadLibrary(m_fw_plugin_handle);
            m_fw_plugin_handle = nullptr;
            return;
          }
          m_dispatch_table[func_name.first] = func;
        }
        // Only if all the functions are found and loaded,
        // do we set the m_loaded = true
        //
        m_loaded = true;
      }
    }
  }

private:
  bool m_loaded;
  xpti_plugin_handle_t m_fw_plugin_handle;
  dispatch_table_t m_dispatch_table;
  xpti::utils::PlatformHelper m_loader;
};
} // namespace xpti

XPTI_EXPORT_API void xptiFrameworkInitialize() {
  if (xpti::ProxyLoader::instance().noErrors()) {
    void *f = xpti::ProxyLoader::instance().functionByIndex(
        XPTI_FRAMEWORK_INITIALIZE);
    if (f) {
      (*reinterpret_cast<xpti_framework_initialize_t>(f))();
    }
  }
}

XPTI_EXPORT_API void xptiFrameworkFinalize() {
  if (xpti::ProxyLoader::instance().noErrors()) {
    void *f =
        xpti::ProxyLoader::instance().functionByIndex(XPTI_FRAMEWORK_FINALIZE);
    if (f) {
      (*reinterpret_cast<xpti_framework_finalize_t>(f))();
    }
  }

  delete &xpti::ProxyLoader::instance();
}

XPTI_EXPORT_API uint16_t xptiRegisterUserDefinedTracePoint(
    const char *tool_name, uint8_t user_defined_tp) {
  if (xpti::ProxyLoader::instance().noErrors()) {
    auto f = xpti::ProxyLoader::instance().functionByIndex(
        XPTI_REGISTER_USER_DEFINED_TP);
    if (f) {
      return (*(xpti_register_user_defined_tp_t)f)(tool_name, user_defined_tp);
    }
  }
  return xpti::invalid_id;
}

XPTI_EXPORT_API uint16_t xptiRegisterUserDefinedEventType(
    const char *tool_name, uint8_t user_defined_event) {
  if (xpti::ProxyLoader::instance().noErrors()) {
    auto f = xpti::ProxyLoader::instance().functionByIndex(
        XPTI_REGISTER_USER_DEFINED_ET);
    if (f) {
      return (*(xpti_register_user_defined_et_t)f)(tool_name,
                                                   user_defined_event);
    }
  }
  return xpti::invalid_id;
}

XPTI_EXPORT_API xpti::result_t xptiInitialize(const char *stream, uint32_t maj,
                                              uint32_t min,
                                              const char *version) {
  if (xpti::ProxyLoader::instance().noErrors()) {
    auto f = xpti::ProxyLoader::instance().functionByIndex(XPTI_INITIALIZE);
    if (f) {
      return (*(xpti_initialize_t)f)(stream, maj, min, version);
    }
  }
  return xpti::result_t::XPTI_RESULT_FAIL;
}

XPTI_EXPORT_API void xptiFinalize(const char *stream) {
  if (xpti::ProxyLoader::instance().noErrors()) {
    auto f = xpti::ProxyLoader::instance().functionByIndex(XPTI_FINALIZE);
    if (f) {
      (*(xpti_finalize_t)f)(stream);
    }
  }
}

XPTI_EXPORT_API uint64_t xptiGetUniversalId() {
  if (xpti::ProxyLoader::instance().noErrors()) {
    auto f =
        xpti::ProxyLoader::instance().functionByIndex(XPTI_GET_UNIVERSAL_ID);
    if (f) {
      return (*reinterpret_cast<xpti_get_universal_id_t>(f))();
    }
  }
  return xpti::invalid_id;
}

XPTI_EXPORT_API void xptiSetUniversalId(uint64_t uid) {
  if (xpti::ProxyLoader::instance().noErrors()) {
    auto f =
        xpti::ProxyLoader::instance().functionByIndex(XPTI_SET_UNIVERSAL_ID);
    if (f) {
      return (*reinterpret_cast<xpti_set_universal_id_t>(f))(uid);
    }
  }
}

XPTI_EXPORT_API xpti::result_t xptiStashTuple(const char *key, uint64_t value) {
  if (xpti::ProxyLoader::instance().noErrors()) {
    auto f = xpti::ProxyLoader::instance().functionByIndex(XPTI_STASH_TUPLE);
    if (f) {
      return (*reinterpret_cast<xpti_stash_tuple_t>(f))(key, value);
    }
  }
  return xpti::result_t::XPTI_RESULT_FAIL;
}

XPTI_EXPORT_API xpti::result_t xptiSetGetStashedTuple(char **key,
                                                      uint64_t &value) {
  if (xpti::ProxyLoader::instance().noErrors()) {
    auto f =
        xpti::ProxyLoader::instance().functionByIndex(XPTI_GET_STASHED_TUPLE);
    if (f) {
      return (*reinterpret_cast<xpti_get_stashed_tuple_t>(f))(key, value);
    }
  }
  return xpti::result_t::XPTI_RESULT_FAIL;
}

XPTI_EXPORT_API void xptiUnstashTuple() {
  if (xpti::ProxyLoader::instance().noErrors()) {
    auto f = xpti::ProxyLoader::instance().functionByIndex(XPTI_UNSTASH_TUPLE);
    if (f) {
      return (*reinterpret_cast<xpti_unstash_tuple_t>(f))();
    }
  }
}

XPTI_EXPORT_API uint64_t xptiGetUniqueId() {
  if (xpti::ProxyLoader::instance().noErrors()) {
    auto f = xpti::ProxyLoader::instance().functionByIndex(XPTI_GET_UNIQUE_ID);
    if (f) {
      return (*(xpti_get_unique_id_t)f)();
    }
  }
  return xpti::invalid_id;
}

XPTI_EXPORT_API xpti::string_id_t xptiRegisterString(const char *string,
                                                     char **table_string) {
  if (xpti::ProxyLoader::instance().noErrors()) {
    auto f =
        xpti::ProxyLoader::instance().functionByIndex(XPTI_REGISTER_STRING);
    if (f) {
      return (*(xpti_register_string_t)f)(string, table_string);
    }
  }
  return xpti::invalid_id;
}

XPTI_EXPORT_API const char *xptiLookupString(xpti::string_id_t id) {
  if (xpti::ProxyLoader::instance().noErrors()) {
    auto f = xpti::ProxyLoader::instance().functionByIndex(XPTI_LOOKUP_STRING);
    if (f) {
      return (*(xpti_lookup_string_t)f)(id);
    }
  }
  return nullptr;
}

XPTI_EXPORT_API xpti::object_id_t
xptiRegisterObject(const char *data, size_t size, uint8_t type) {
  if (xpti::ProxyLoader::instance().noErrors()) {
    auto f =
        xpti::ProxyLoader::instance().functionByIndex(XPTI_REGISTER_OBJECT);
    if (f) {
      return (*(xpti_register_object_t)f)(data, size, type);
    }
  }
  return xpti::invalid_id;
}

XPTI_EXPORT_API xpti::object_data_t xptiLookupObject(xpti::object_id_t id) {
  if (xpti::ProxyLoader::instance().noErrors()) {
    auto f = xpti::ProxyLoader::instance().functionByIndex(XPTI_LOOKUP_OBJECT);
    if (f) {
      return (*(xpti_lookup_object_t)f)(id);
    }
  }
  return xpti::object_data_t{0, nullptr, 0};
}

XPTI_EXPORT_API uint64_t xptiRegisterPayload(xpti::payload_t *payload) {
  if (xpti::ProxyLoader::instance().noErrors()) {
    auto f =
        xpti::ProxyLoader::instance().functionByIndex(XPTI_REGISTER_PAYLOAD);
    if (f) {
      return (*(xpti_register_payload_t)f)(payload);
    }
  }
  return xpti::invalid_uid;
}

XPTI_EXPORT_API uint8_t xptiRegisterStream(const char *stream_name) {
  if (xpti::ProxyLoader::instance().noErrors()) {
    auto f =
        xpti::ProxyLoader::instance().functionByIndex(XPTI_REGISTER_STREAM);
    if (f) {
      return (*(xpti_register_stream_t)f)(stream_name);
    }
  }
  return xpti::invalid_id;
}

XPTI_EXPORT_API xpti::result_t xptiUnregisterStream(const char *stream_name) {
  if (xpti::ProxyLoader::instance().noErrors()) {
    auto f =
        xpti::ProxyLoader::instance().functionByIndex(XPTI_UNREGISTER_STREAM);
    if (f) {
      return (*(xpti_unregister_stream_t)f)(stream_name);
    }
  }
  return xpti::result_t::XPTI_RESULT_FAIL;
}
XPTI_EXPORT_API xpti::trace_event_data_t *
xptiMakeEvent(const char *name, xpti::payload_t *payload, uint16_t event,
              xpti::trace_activity_type_t activity, uint64_t *instance_no) {
  if (xpti::ProxyLoader::instance().noErrors()) {
    auto f = xpti::ProxyLoader::instance().functionByIndex(XPTI_MAKE_EVENT);
    if (f) {
      return (*(xpti_make_event_t)f)(name, payload, event, activity,
                                     instance_no);
    }
  }
  return nullptr;
}

XPTI_EXPORT_API const xpti::trace_event_data_t *xptiFindEvent(uint64_t uid) {
  if (xpti::ProxyLoader::instance().noErrors()) {
    auto f = xpti::ProxyLoader::instance().functionByIndex(XPTI_FIND_EVENT);
    if (f) {
      return (*(xpti_find_event_t)f)(uid);
    }
  }
  return nullptr;
}

XPTI_EXPORT_API const xpti::payload_t *
xptiQueryPayload(xpti::trace_event_data_t *lookup_object) {
  if (xpti::ProxyLoader::instance().noErrors()) {
    auto f = xpti::ProxyLoader::instance().functionByIndex(XPTI_QUERY_PAYLOAD);
    if (f) {
      return (*(xpti_query_payload_t)f)(lookup_object);
    }
  }
  return nullptr;
}

XPTI_EXPORT_API const xpti::payload_t *xptiQueryPayloadByUID(uint64_t uid) {
  if (xpti::ProxyLoader::instance().noErrors()) {
    auto f = xpti::ProxyLoader::instance().functionByIndex(
        XPTI_QUERY_PAYLOAD_BY_UID);
    if (f) {
      return (*(xpti_query_payload_by_uid_t)f)(uid);
    }
  }
  return nullptr;
}

XPTI_EXPORT_API xpti::result_t
xptiRegisterCallback(uint8_t stream_id, uint16_t trace_type,
                     xpti::tracepoint_callback_api_t cb) {
  if (xpti::ProxyLoader::instance().noErrors()) {
    auto f =
        xpti::ProxyLoader::instance().functionByIndex(XPTI_REGISTER_CALLBACK);
    if (f) {
      return (*(xpti_register_cb_t)f)(stream_id, trace_type, cb);
    }
  }
  return xpti::result_t::XPTI_RESULT_FAIL;
}

XPTI_EXPORT_API xpti::result_t
xptiUnregisterCallback(uint8_t stream_id, uint16_t trace_type,
                       xpti::tracepoint_callback_api_t cb) {
  if (xpti::ProxyLoader::instance().noErrors()) {
    auto f =
        xpti::ProxyLoader::instance().functionByIndex(XPTI_UNREGISTER_CALLBACK);
    if (f) {
      return (*(xpti_unregister_cb_t)f)(stream_id, trace_type, cb);
    }
  }
  return xpti::result_t::XPTI_RESULT_FAIL;
}

XPTI_EXPORT_API xpti::result_t
xptiNotifySubscribers(uint8_t stream_id, uint16_t trace_type,
                      xpti::trace_event_data_t *parent,
                      xpti::trace_event_data_t *object, uint64_t instance,
                      const void *user_data) {
  if (xpti::ProxyLoader::instance().noErrors()) {
    auto f =
        xpti::ProxyLoader::instance().functionByIndex(XPTI_NOTIFY_SUBSCRIBERS);
    if (f) {
      return (*(xpti_notify_subscribers_t)f)(stream_id, trace_type, parent,
                                             object, instance, user_data);
    }
  }
  return xpti::result_t::XPTI_RESULT_FAIL;
}

XPTI_EXPORT_API bool xptiTraceEnabled() {
  if (xpti::ProxyLoader::instance().noErrors()) {
    auto f = xpti::ProxyLoader::instance().functionByIndex(XPTI_TRACE_ENABLED);
    if (f) {
      return (*(xpti_trace_enabled_t)f)();
    }
  }
  return false;
}

XPTI_EXPORT_API bool xptiCheckTraceEnabled(uint16_t stream, uint16_t ttype) {
  if (xpti::ProxyLoader::instance().noErrors()) {
    auto f =
        xpti::ProxyLoader::instance().functionByIndex(XPTI_CHECK_TRACE_ENABLED);
    if (f) {
      return (*(xpti_check_trace_enabled_t)f)(stream, ttype);
    }
  }
  return false;
}

XPTI_EXPORT_API void xptiTraceTryToEnable() {
  xpti::ProxyLoader::instance().tryToEnable();
}

XPTI_EXPORT_API void xptiForceSetTraceEnabled(bool YesOrNo) {
  if (xpti::ProxyLoader::instance().noErrors()) {
    auto f = xpti::ProxyLoader::instance().functionByIndex(
        XPTI_FORCE_SET_TRACE_ENABLED);
    if (f) {
      (*(xpti_force_set_trace_enabled_t)f)(YesOrNo);
    }
  }
}

XPTI_EXPORT_API xpti::result_t xptiAddMetadata(xpti::trace_event_data_t *e,
                                               const char *key,
                                               xpti::object_id_t value_id) {
  if (xpti::ProxyLoader::instance().noErrors()) {
    auto f = xpti::ProxyLoader::instance().functionByIndex(XPTI_ADD_METADATA);
    if (f) {
      return (*(xpti_add_metadata_t)f)(e, key, value_id);
    }
  }
  return xpti::result_t::XPTI_RESULT_FAIL;
}

XPTI_EXPORT_API xpti::metadata_t *
xptiQueryMetadata(xpti::trace_event_data_t *lookup_object) {
  if (xpti::ProxyLoader::instance().noErrors()) {
    auto f = xpti::ProxyLoader::instance().functionByIndex(XPTI_QUERY_METADATA);
    if (f) {
      return (*(xpti_query_metadata_t)f)(lookup_object);
    }
  }
  return nullptr;
}

XPTI_EXPORT_API void xptiReleaseEvent(xpti::trace_event_data_t *lookup_object) {
  if (xpti::ProxyLoader::instance().noErrors()) {
    auto f = xpti::ProxyLoader::instance().functionByIndex(XPTI_RELEASE_EVENT);
    if (f) {
      (*(xpti_release_event_t)f)(lookup_object);
    }
  }
}

// New version of XPTI has these new APIs as XPTI is moving to 128-bit keys to
// avoid hash collisions and improve performance. The older APIs -
// xptiQueryPayloadByUID(), xptiFindEvent(), xptiMakeEvent(),
// xptiRegisterPayload() will all be deprecated in the future.

XPTI_EXPORT_API void xptiEnableTracepointScopeNotification(bool enable) {
  if (xpti::ProxyLoader::instance().noErrors()) {
    auto f = xpti::ProxyLoader::instance().functionByIndex(
        XPTI_ENABLE_TRACEPOINT_SCOPE_NOTIFICATION);
    if (f) {
      (*(xpti_enable_tracepoint_scope_notification_t)f)(enable);
    }
  }
}

XPTI_EXPORT_API bool xptiCheckTracepointScopeNotification() {
  if (xpti::ProxyLoader::instance().noErrors()) {
    auto f = xpti::ProxyLoader::instance().functionByIndex(
        XPTI_CHECK_TRACEPOINT_SCOPE_NOTIFICATION);
    if (f) {
      return (*(xpti_check_tracepoint_scope_notification_t)f)();
    }
  }
  return false;
}

XPTI_EXPORT_API xpti_tracepoint_t *xptiCreateTracepoint(const char *name,
                                                        const char *source_file,
                                                        uint32_t line_no,
                                                        uint32_t column_no) {
  if (xpti::ProxyLoader::instance().noErrors()) {
    auto f =
        xpti::ProxyLoader::instance().functionByIndex(XPTI_CREATE_TRACEPOINT);
    if (f) {
      return (*(xpti_create_tracepoint_t)f)(name, source_file, line_no,
                                            column_no);
    }
  }
  return nullptr;
}

XPTI_EXPORT_API xpti::result_t xptiDeleteTracepoint(xpti_tracepoint_t *tp) {
  if (xpti::ProxyLoader::instance().noErrors()) {
    auto f =
        xpti::ProxyLoader::instance().functionByIndex(XPTI_DELETE_TRACEPOINT);
    if (f) {
      return (*(xpti_delete_tracepoint_t)f)(tp);
    }
  }
  return xpti::result_t::XPTI_RESULT_FAIL;
}

XPTI_EXPORT_API const xpti_payload_t *xptiLookupPayload(uint64_t uid) {
  if (xpti::ProxyLoader::instance().noErrors()) {
    auto f = xpti::ProxyLoader::instance().functionByIndex(XPTI_LOOKUP_PAYLOAD);
    if (f) {
      return (*(xpti_lookup_payload_t)f)(uid);
    }
  }
  return nullptr;
}

XPTI_EXPORT_API const xpti_trace_event_t *xptiLookupEvent(uint64_t uid) {
  if (xpti::ProxyLoader::instance().noErrors()) {
    auto f = xpti::ProxyLoader::instance().functionByIndex(XPTI_LOOKUP_EVENT);
    if (f) {
      return (*(xpti_lookup_event_t)f)(uid);
    }
  }
  return nullptr;
}

XPTI_EXPORT_API const xpti_tracepoint_t *xptiGetTracepointScopeData() {
  if (xpti::ProxyLoader::instance().noErrors()) {
    auto f = xpti::ProxyLoader::instance().functionByIndex(
        XPTI_GET_TRACEPOINT_SCOPE_DATA);
    if (f) {
      return (*(xpti_get_trace_point_scope_data_t)f)();
    }
  }
  return nullptr;
}

XPTI_EXPORT_API xpti::result_t
xptiSetTracepointScopeData(xpti_tracepoint_t *data) {
  if (xpti::ProxyLoader::instance().noErrors()) {
    auto f = xpti::ProxyLoader::instance().functionByIndex(
        XPTI_SET_TRACEPOINT_SCOPE_DATA);
    if (f) {
      return (*(xpti_set_trace_point_scope_data_t)f)(data);
    }
  }
  return xpti::result_t::XPTI_RESULT_FAIL;
}

XPTI_EXPORT_API void xptiUnsetTracepointScopeData() {
  if (xpti::ProxyLoader::instance().noErrors()) {
    auto f = xpti::ProxyLoader::instance().functionByIndex(
        XPTI_UNSET_TRACEPOINT_SCOPE_DATA);
    if (f) {
      (*(xpti_unset_trace_point_scope_data_t)f)();
    }
  }
}

XPTI_EXPORT_API const xpti_tracepoint_t *
xptiRegisterTracepointScope(const char *FuncName, const char *FileName,
                            uint32_t LineNo, uint32_t ColumnNo) {
  if (xpti::ProxyLoader::instance().noErrors()) {
    auto f = xpti::ProxyLoader::instance().functionByIndex(
        XPTI_REGISTER_TRACEPOINT_SCOPE);
    if (f) {
      return (*(xpti_register_tracepoint_scope_t)f)(FuncName, FileName, LineNo,
                                                    ColumnNo);
    }
  }
  return nullptr;
}

XPTI_EXPORT_API uint8_t xptiGetDefaultStreamID() {
  if (xpti::ProxyLoader::instance().noErrors()) {
    auto f = xpti::ProxyLoader::instance().functionByIndex(
        XPTI_GET_DEFAULT_STREAM_ID);
    if (f) {
      return (*(xpti_get_default_stream_id_t)f)();
    }
  }
  return xpti::invalid_id;
}

XPTI_EXPORT_API xpti::result_t xptiSetDefaultStreamID(uint8_t stream_id) {
  if (xpti::ProxyLoader::instance().noErrors()) {
    auto f = xpti::ProxyLoader::instance().functionByIndex(
        XPTI_SET_DEFAULT_STREAM_ID);
    if (f) {
      return (*(xpti_set_default_stream_id_t)f)(stream_id);
    }
  }
  return xpti::result_t::XPTI_RESULT_FAIL;
}

XPTI_EXPORT_API xpti::trace_event_type_t xptiGetDefaultEventType() {
  if (xpti::ProxyLoader::instance().noErrors()) {
    auto f = xpti::ProxyLoader::instance().functionByIndex(
        XPTI_GET_DEFAULT_EVENT_TYPE);
    if (f) {
      return (*(xpti_get_default_event_type_t)f)();
    }
  }
  return xpti::trace_event_type_t::algorithm;
}

XPTI_EXPORT_API xpti::result_t
xptiSetDefaultEventType(xpti::trace_event_type_t event_type) {
  if (xpti::ProxyLoader::instance().noErrors()) {
    auto f = xpti::ProxyLoader::instance().functionByIndex(
        XPTI_SET_DEFAULT_EVENT_TYPE);
    if (f) {
      return (*(xpti_set_default_event_type_t)f)(event_type);
    }
  }
  return xpti::result_t::XPTI_RESULT_FAIL;
}

XPTI_EXPORT_API xpti::trace_point_type_t xptiGetDefaultTraceType() {
  if (xpti::ProxyLoader::instance().noErrors()) {
    auto f = xpti::ProxyLoader::instance().functionByIndex(
        XPTI_GET_DEFAULT_TRACE_TYPE);
    if (f) {
      return (*(xpti_get_default_trace_type_t)f)();
    }
  }
  return xpti::trace_point_type_t::function_begin;
}

XPTI_EXPORT_API xpti::result_t
xptiSetDefaultTraceType(xpti::trace_point_type_t trace_type) {
  if (xpti::ProxyLoader::instance().noErrors()) {
    auto f = xpti::ProxyLoader::instance().functionByIndex(
        XPTI_SET_DEFAULT_TRACE_TYPE);
    if (f) {
      return (*(xpti_set_default_trace_type_t)f)(trace_type);
    }
  }
  return xpti::result_t::XPTI_RESULT_FAIL;
}