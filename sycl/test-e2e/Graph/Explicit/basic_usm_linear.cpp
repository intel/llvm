// RUN: %{build} -o %t.out
// RUN: env UR_L0_USE_DRIVER_COUNTER_BASED_EVENTS=1 %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 env UR_L0_USE_DRIVER_COUNTER_BASED_EVENTS=1 %{l0_leak_check} %{run} %t.out  %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero && linux %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 env UR_L0_USE_DRIVER_COUNTER_BASED_EVENTS=1 %{l0_leak_check} %{run} %t.out  %}
//

#define GRAPH_E2E_EXPLICIT

#include "../Inputs/basic_usm_linear.cpp"
