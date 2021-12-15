//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
#include "xpti_itt_helper.h"
#include <unordered_map>
#include <vector>

constexpr int graph_task = 1;

namespace std {
template <>
struct less<__itt_id> : public binary_function<__itt_id, __itt_id,
                                               bool> { // functor for operator<
  bool operator()(const __itt_id &_Left,
                  const __itt_id &_Right) const { // apply operator< to operands
    if (_Left.d1 < _Right.d1)
      return true;
    if (_Left.d1 == _Right.d1 && _Left.d2 < _Right.d2)
      return true;
    if (_Left.d1 == _Right.d1 && _Left.d2 == _Right.d2 && _Left.d3 < _Right.d3)
      return true;
    return false;
  }
};
} // namespace std
namespace xpti {
namespace itt {

const __itt_id null_id = {0, 0, 0};
static domains_t g_domains;
static strings_t g_strings;
typedef std::unordered_map<uint64_t, __itt_id> itt_id_map_t;
std::unordered_map<uint64_t, itt_id_map_t> g_itt_ids;

/*
struct resource_string {
  const char *str;
  __itt_string_handle *handle;
};

/*
#define TBB_STRING_RESOURCE(index_name, str) {str, NULL},
// Populate the resource string like it is done in TBB
//
static resource_string strings_for_itt[] = {
    #include "_tbb_strings.h"
    {"num_resource_strings", NULL}
};
#undef TBB_STRING_RESOURCE

__itt_string_handle *get_string_handle(int index) {
    return (index >= 0 && index < NUM_STRINGS) ?
           strings_for_itt[index].handle : nullptr;
}

void init_tbb_strings() {
    for(int i = 0; i < NUM_STRINGS; ++i )  {
        strings_for_itt[i].handle
=__itt_string_handle_create(strings_for_itt[i].str);
        g_strings[strings_for_itt[i].str] = strings_for_itt[i].handle;
    }
}
*/

void make_id(__itt_domain *dom, __itt_id *id, void *addr,
             unsigned long long extra) {

  if (g_itt_ids.count((uint64_t)addr)) {
    auto &sub_map = g_itt_ids[(uint64_t)addr];
    if (sub_map.count(extra)) {
      *id = sub_map[extra];
      return;
    } else {
      __itt_id &rid = sub_map[extra];
      rid.d1 = (unsigned long long)((uintptr_t)addr);
      rid.d2 = (unsigned long long)extra;
      rid.d3 = (unsigned long long)0; /*Reserved. Must be zero */
      *id = rid;
      return;
    }
  }

  // Not present in the map; so create the
  // ID, add it to the map and also emit
  // an _itt_create call to register it.
  auto &sub_map = g_itt_ids[(uint64_t)addr];
  __itt_id &rid = sub_map[extra];
  rid.d1 = (unsigned long long)((uintptr_t)addr);
  rid.d2 = (unsigned long long)extra;
  rid.d3 = (unsigned long long)0; /*Reserved. Must be zero */
  *id = rid;
  __itt_id_create(dom, rid);
}

__itt_domain *create_domain(const char *domain) {
  if (g_domains.count(domain))
    return g_domains[domain];
  else {
    auto d = __itt_domain_create(domain);
    d->flags = 1;
    g_domains[domain] = d;
    return d;
  }
}

__itt_string_handle *create_string(const char *string) {
  if (!string)
    return nullptr;

  if (g_strings.count(string)) {
    return g_strings[string];
  } else {
    auto s = __itt_string_handle_create(string);
    g_strings[string] = s;
    return s;
  }
}

__itt_id create_task_group(__itt_domain *domain, void *parent,
                           unsigned long long pextra, void *group,
                           unsigned long long gextra, const char *name) {
  __itt_id group_id = null_id;
  __itt_id parent_id = null_id;

  make_id(domain, &group_id, group, gextra);

  if (parent) {
    make_id(domain, &parent_id, parent, pextra);
  }
  __itt_task_group(domain, group_id, parent_id, create_string(name));

  return group_id;
}

__itt_id create_graph(__itt_domain *domain, xpti::trace_event_data_t *parent,
                      xpti::trace_event_data_t *graph,
                      const xpti::payload_t *payload) {
  // Using TBB example of capturing a graph, but this data may be ignored by the
  // Analyzers
  /*
  __itt_id id =
      create_task_group(domain, nullptr, FLOW_NULL, (void *)graph->unique_id,
                        FLOW_GRAPH, strings_for_itt[FLOW_GRAPH].str);
  std::string v = payload->name;
  __itt_metadata_str_add(domain, id, get_string_handle(FLOW_OBJECT_NAME),
                         payload->name, v.size());
  return id;
  */
  return null_id;
}

void create_edge(__itt_domain *domain, xpti::trace_event_data_t *parent,
                 xpti::trace_event_data_t *edge) {
  // Using TBB example of capturing an edge, but this data may be ignored by the
  // Analyzers
  /*
   __itt_id sid = xpti::itt::null_id;
   __itt_id tid = xpti::itt::null_id;

   make_id(domain, &sid, (void *)edge->source_id, (unsigned long
   long)FLOW_OUTPUT_PORT); make_id(domain, &tid, (void *)edge->target_id,
   (unsigned long long)FLOW_INPUT_PORT);

   __itt_relation_add(domain, sid, __itt_relation_is_predecessor_to, tid);
  */
}

__itt_id create_node(__itt_domain *domain, xpti::trace_event_data_t *parent,
                     xpti::trace_event_data_t *node,
                     const xpti::payload_t *payload,
                     const char *optional_node_type) {
  // Using TBB example of capturing a node, but this data may be ignored by the
  // Analyzers
  /*
  __itt_id id = create_task_group(
      domain, (void *)parent->unique_id, FLOW_GRAPH, (void *)node->unique_id,
      FLOW_NODE, (optional_node_type ? optional_node_type : "fga_node"));
  uint64_t node_id = node->unique_id;
  // We need to add any metadata + relations here
  // based on what is done in TBB
  create_task_group(domain, (void *)node_id, FLOW_NODE, (void *)node_id,
                    FLOW_OUTPUT_PORT, strings_for_itt[FLOW_OUTPUT_PORT_0].str);
  create_task_group(domain, (void *)node_id, FLOW_NODE, (void *)node_id,
                    FLOW_INPUT_PORT, strings_for_itt[FLOW_INPUT_PORT_0].str);
  __itt_id pay_id = xpti::itt::null_id;
  __itt_id n_id = xpti::itt::null_id;
  make_id(domain, &n_id, (void *)node_id, (unsigned long long)FLOW_OUTPUT_PORT);
  make_id(domain, &pay_id, (void *)node_id, (unsigned long long)FLOW_BODY);
  //  The body() is the child of output_port in TBB as that is what is used
  //  to identify the node. Here, we make body() a child of the node itself
  __itt_relation_add(domain, pay_id, __itt_relation_is_child_of, id);

  std::string v = payload->name;
  __itt_metadata_str_add(domain, id, get_string_handle(FLOW_OBJECT_NAME),
                         v.c_str(), v.size());
  return id;
  */
  return null_id;
}

void task_begin(__itt_domain *domain, xpti::trace_event_data_t *task,
                xpti::trace_event_data_t *parent,
                const xpti::payload_t *payload) {
  __itt_id task_id = null_id;
  __itt_id parent_id = null_id;
  uint64_t gextra = 0;

  if (task->event_type == (int)xpti::trace_event_type_t::graph)
    gextra = graph_task;

  make_id(domain, &task_id, (void *)task->unique_id, gextra);
  if (parent) {
    make_id(domain, &parent_id, (void *)parent->unique_id, 0);
  }
  __itt_task_begin(domain, task_id, parent_id, create_string(payload->name));
}

void function_begin(__itt_domain *domain, void *task, void *parent,
                    const char *type) {
  __itt_id task_id = null_id;
  __itt_id parent_id = null_id;

  make_id(domain, &task_id, task, 0);
  if (parent) {
    make_id(domain, &parent_id, parent, 0);
  }
  __itt_task_begin(domain, task_id, parent_id, create_string(type));
}

void task_end(__itt_domain *domain) { __itt_task_end(domain); }

} // namespace itt
} // namespace xpti
