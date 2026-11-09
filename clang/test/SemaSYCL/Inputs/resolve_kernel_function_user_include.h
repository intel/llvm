// An application header passed with -include, which is processed before the
// integration header.
#pragma once

[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 1)]]
void from_include(int *p, float v);
