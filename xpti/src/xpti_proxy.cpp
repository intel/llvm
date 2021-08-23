//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
#include "xpti_trace_framework.hpp"

#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>

enum functions_t {
  XPTI_INITIALIZE,
  XPTI_FINALIZE,
  XPTI_GET_UNIQUE_ID,
  XPTI_REGISTER_STRING,
  XPTI_LOOKUP_STRING,
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

  // All additional functions need to appear before
  // the XPTI_FW_API_COUNT enum
  XPTI_FW_API_COUNT ///< This enum must always be the last one in the list
};

namespace xpti {
class ProxyLoader {
  std::unordered_map<int, const char *> m_function_names = {
      {XPTI_INITIALIZE, "xptiInitialize"},
      {XPTI_FINALIZE, "xptiFinalize"},
      {XPTI_GET_UNIQUE_ID, "xptiGetUniqueId"},
      {XPTI_REGISTER_STRING, "xptiRegisterString"},
      {XPTI_LOOKUP_STRING, "xptiLookupString"},
      {XPTI_REGISTER_STREAM, "xptiRegisterStream"},
      {XPTI_UNREGISTER_STREAM, "xptiUnregisterStream"},
      {XPTI_REGISTER_USER_DEFINED_TP, "xptiRegisterUserDefinedTracePoint"},
      {XPTI_REGISTER_USER_DEFINED_ET, "xptiRegisterUserDefinedEventType"},
      {XPTI_MAKE_EVENT, "xptiMakeEvent"},
      {XPTI_FIND_EVENT, "xptiFindEvent"},
      {XPTI_QUERY_PAYLOAD, "xptiQueryPayload"},
      {XPTI_REGISTER_CALLBACK, "xptiRegisterCallback"},
      {XPTI_UNREGISTER_CALLBACK, "xptiUnregisterCallback"},
      {XPTI_NOTIFY_SUBSCRIBERS, "xptiNotifySubscribers"},
      {XPTI_ADD_METADATA, "xptiAddMetadata"},
      {XPTI_QUERY_METADATA, "xptiQueryMetadata"},
      {XPTI_TRACE_ENABLED, "xptiTraceEnabled"}};

public:
  typedef std::vector<xpti_plugin_function_t> dispatch_table_t;

  ProxyLoader() : m_loaded(false), m_fw_plugin_handle(nullptr) {
    // When this object is created, we attempt to load
    // the share object implementation. We look for the
    // environment variable XPTI_FRAMEWORK_DISPATCHER to
    // see if it has been set. If not, all methods in
    // the proxy should end up being close to no-ops
    //
    std::string env =
        m_loader.getEnvironmentVariable("XPTI_FRAMEWORK_DISPATCHER");
    if (env.empty())
      return;
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
    if (index >= XPTI_INITIALIZE && index < XPTI_FW_API_COUNT) {
      return reinterpret_cast<void *>(m_dispatch_table[index]);
    }
    return nullptr;
  }

private:
  bool m_loaded;
  xpti_plugin_handle_t m_fw_plugin_handle;
  dispatch_table_t m_dispatch_table;
  xpti::utils::PlatformHelper m_loader;
};

static ProxyLoader g_loader;
} // namespace xpti

XPTI_EXPORT_API uint16_t xptiRegisterUserDefinedTracePoint(
    const char *tool_name, uint8_t user_defined_tp) {
  if (xpti::g_loader.noErrors()) {
    auto f = xpti::g_loader.functionByIndex(XPTI_REGISTER_USER_DEFINED_TP);
    if (f) {
      return (*(xpti_register_user_defined_tp_t)f)(tool_name, user_defined_tp);
    }
  }
  return xpti::invalid_id;
}

XPTI_EXPORT_API uint16_t xptiRegisterUserDefinedEventType(
    const char *tool_name, uint8_t user_defined_event) {
  if (xpti::g_loader.noErrors()) {
    auto f = xpti::g_loader.functionByIndex(XPTI_REGISTER_USER_DEFINED_ET);
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
  if (xpti::g_loader.noErrors()) {
    auto f = xpti::g_loader.functionByIndex(XPTI_INITIALIZE);
    if (f) {
      return (*(xpti_initialize_t)f)(stream, maj, min, version);
    }
  }
  return xpti::result_t::XPTI_RESULT_FAIL;
}

XPTI_EXPORT_API void xptiFinalize(const char *stream) {
  if (xpti::g_loader.noErrors()) {
    auto f = xpti::g_loader.functionByIndex(XPTI_FINALIZE);
    if (f) {
      (*(xpti_finalize_t)f)(stream);
    }
  }
}

XPTI_EXPORT_API uint64_t xptiGetUniqueId() {
  if (xpti::g_loader.noErrors()) {
    auto f = xpti::g_loader.functionByIndex(XPTI_GET_UNIQUE_ID);
    if (f) {
      return (*(xpti_get_unique_id_t)f)();
    }
  }
  return xpti::invalid_id;
}

XPTI_EXPORT_API xpti::string_id_t xptiRegisterString(const char *string,
                                                     char **table_string) {
  if (xpti::g_loader.noErrors()) {
    auto f = xpti::g_loader.functionByIndex(XPTI_REGISTER_STRING);
    if (f) {
      return (*(xpti_register_string_t)f)(string, table_string);
    }
  }
  return xpti::invalid_id;
}

XPTI_EXPORT_API const char *xptiLookupString(xpti::string_id_t id) {
  if (xpti::g_loader.noErrors()) {
    auto f = xpti::g_loader.functionByIndex(XPTI_LOOKUP_STRING);
    if (f) {
      return (*(xpti_lookup_string_t)f)(id);
    }
  }
  return nullptr;
}

XPTI_EXPORT_API uint8_t xptiRegisterStream(const char *stream_name) {
  if (xpti::g_loader.noErrors()) {
    auto f = xpti::g_loader.functionByIndex(XPTI_REGISTER_STREAM);
    if (f) {
      return (*(xpti_register_stream_t)f)(stream_name);
    }
  }
  return xpti::invalid_id;
}

XPTI_EXPORT_API xpti::result_t xptiUnregisterStream(const char *stream_name) {
  if (xpti::g_loader.noErrors()) {
    auto f = xpti::g_loader.functionByIndex(XPTI_UNREGISTER_STREAM);
    if (f) {
      return (*(xpti_unregister_stream_t)f)(stream_name);
    }
  }
  return xpti::result_t::XPTI_RESULT_FAIL;
}
XPTI_EXPORT_API xpti::trace_event_data_t *
xptiMakeEvent(const char *name, xpti::payload_t *payload, uint16_t event,
              xpti::trace_activity_type_t activity, uint64_t *instance_no) {
  if (xpti::g_loader.noErrors()) {
    auto f = xpti::g_loader.functionByIndex(XPTI_MAKE_EVENT);
    if (f) {
      return (*(xpti_make_event_t)f)(name, payload, event, activity,
                                     instance_no);
    }
  }
  return nullptr;
}

XPTI_EXPORT_API const xpti::trace_event_data_t *xptiFindEvent(uint64_t uid) {
  if (xpti::g_loader.noErrors()) {
    auto f = xpti::g_loader.functionByIndex(XPTI_FIND_EVENT);
    if (f) {
      return (*(xpti_find_event_t)f)(uid);
    }
  }
  return nullptr;
}

XPTI_EXPORT_API const xpti::payload_t *
xptiQueryPayload(xpti::trace_event_data_t *lookup_object) {
  if (xpti::g_loader.noErrors()) {
    auto f = xpti::g_loader.functionByIndex(XPTI_QUERY_PAYLOAD);
    if (f) {
      return (*(xpti_query_payload_t)f)(lookup_object);
    }
  }
  return nullptr;
}

XPTI_EXPORT_API xpti::result_t
xptiRegisterCallback(uint8_t stream_id, uint16_t trace_type,
                     xpti::tracepoint_callback_api_t cb) {
  if (xpti::g_loader.noErrors()) {
    auto f = xpti::g_loader.functionByIndex(XPTI_REGISTER_CALLBACK);
    if (f) {
      return (*(xpti_register_cb_t)f)(stream_id, trace_type, cb);
    }
  }
  return xpti::result_t::XPTI_RESULT_FAIL;
}

XPTI_EXPORT_API xpti::result_t
xptiUnregisterCallback(uint8_t stream_id, uint16_t trace_type,
                       xpti::tracepoint_callback_api_t cb) {
  if (xpti::g_loader.noErrors()) {
    auto f = xpti::g_loader.functionByIndex(XPTI_UNREGISTER_CALLBACK);
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
  if (xpti::g_loader.noErrors()) {
    auto f = xpti::g_loader.functionByIndex(XPTI_NOTIFY_SUBSCRIBERS);
    if (f) {
      return (*(xpti_notify_subscribers_t)f)(stream_id, trace_type, parent,
                                             object, instance, user_data);
    }
  }
  return xpti::result_t::XPTI_RESULT_FAIL;
}

XPTI_EXPORT_API bool xptiTraceEnabled() {
  if (xpti::g_loader.noErrors()) {
    auto f = xpti::g_loader.functionByIndex(XPTI_TRACE_ENABLED);
    if (f) {
      return (*(xpti_trace_enabled_t)f)();
    }
  }
  return false;
}

XPTI_EXPORT_API xpti::result_t xptiAddMetadata(xpti::trace_event_data_t *e,
                                               const char *key,
                                               const char *value) {
  if (xpti::g_loader.noErrors()) {
    auto f = xpti::g_loader.functionByIndex(XPTI_ADD_METADATA);
    if (f) {
      return (*(xpti_add_metadata_t)f)(e, key, value);
    }
  }
  return xpti::result_t::XPTI_RESULT_FAIL;
}

XPTI_EXPORT_API xpti::metadata_t *
xptiQueryMetadata(xpti::trace_event_data_t *lookup_object) {
  if (xpti::g_loader.noErrors()) {
    auto f = xpti::g_loader.functionByIndex(XPTI_QUERY_METADATA);
    if (f) {
      return (*(xpti_query_metadata_t)f)(lookup_object);
    }
  }
  return nullptr;
}
