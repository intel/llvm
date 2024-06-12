// RUN: %{build} -o %t.out
// RUN: %if level_zero || cuda %{ %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %} %else %{ %{run} %t.out %}
//
// CHECK: complete

#define GRAPH_E2E_EXPLICIT

#include "../Inputs/event_status_querying.cpp"
