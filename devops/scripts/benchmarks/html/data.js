benchmarkRuns = [
  {
    "results": [],
    "name": "OneDNN_V1",
    "hostname": "gkdse-pre-dnp-02",
    "git_hash": "86f36f78ef5c",
    "github_repo": "mateuszpn/llvm",
    "date": "2025-08-26T15:37:28.240097+00:00",
    "compute_runtime": "unknown",
    "platform": {
      "timestamp": "2025-08-26T15:37:28.240080",
      "os": "Linux 5.15.0-142-generic #152-Ubuntu SMP Mon May 19 10:54:31 UTC 2025",
      "python": "CPython 3.10.12",
      "cpu_count": 224,
      "cpu_info": "Intel(R) Xeon(R) Platinum 8480L",
      "gpu_count": 3,
      "gpu_info": [
        "ASPEED Technology, Inc. ASPEED Graphics Family (rev 52)",
        "Intel Corporation Device 0bda (rev 2f)",
        "Intel Corporation Device 0bda (rev 2f)"
      ],
      "gpu_driver_version": "i915 backported to 5.15.0-142 from (905bc8bed62f2) using backports I915_25.2.7_PSB_250224.10",
      "gcc_version": "gcc (Ubuntu 11.4.0-1ubuntu1~22.04.2) 11.4.0",
      "clang_version": "clang version 21.0.0git (git@github.com:mateuszpn/llvm.git fb18321705f6d11ebd2f15c369d49826c18fe329)",
      "level_zero_version": "L0 v1 adapter | level-zero (version unknown)",
      "compute_runtime_version": ""
    }
  }
];

benchmarkMetadata = {
  "SubmitKernel": {
    "type": "group",
    "description": "Measures CPU time overhead of submitting kernels through different APIs.",
    "notes": "Each layer builds on top of the previous layer, adding functionality and overhead.\nThe first layer is the Level Zero API, the second is the Unified Runtime API, and the third is the SYCL API.\nThe UR v2 adapter noticeably reduces UR layer overhead, also improving SYCL performance.\nWork is ongoing to reduce the overhead of the SYCL API\n",
    "unstable": null,
    "tags": [
      "submit",
      "micro",
      "SYCL",
      "UR",
      "L0"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": null,
    "explicit_group": null
  },
  "SinKernelGraph": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": "This benchmark combines both eager and graph execution, and may not be representative of real use cases.",
    "tags": [
      "submit",
      "memory",
      "proxy",
      "SYCL",
      "UR",
      "L0",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": null
  },
  "SubmitGraph": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "micro",
      "SYCL",
      "UR",
      "L0",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": null
  },
  "FinalizeGraph": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "finalize",
      "micro",
      "SYCL",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": null
  },
  "SubmitKernel in order long kernel": {
    "type": "group",
    "description": "Measures CPU time overhead of submitting in order kernels with longer execution times through different APIs.",
    "notes": "Each layer builds on top of the previous layer, adding functionality and overhead.\nThe first layer is the Level Zero API, the second is the Unified Runtime API, and the third is the SYCL API.\nThe UR v2 adapter noticeably reduces UR layer overhead, also improving SYCL performance.\nWork is ongoing to reduce the overhead of the SYCL API\n",
    "unstable": null,
    "tags": [
      "submit",
      "micro",
      "SYCL",
      "UR",
      "L0"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": null,
    "explicit_group": null
  },
  "SubmitKernel in order long kernel, CPU count": {
    "type": "group",
    "description": "Measures CPU time overhead of submitting in order kernels with longer execution times through different APIs.",
    "notes": "Each layer builds on top of the previous layer, adding functionality and overhead.\nThe first layer is the Level Zero API, the second is the Unified Runtime API, and the third is the SYCL API.\nThe UR v2 adapter noticeably reduces UR layer overhead, also improving SYCL performance.\nWork is ongoing to reduce the overhead of the SYCL API\n",
    "unstable": null,
    "tags": [
      "submit",
      "micro",
      "SYCL",
      "UR",
      "L0"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": null,
    "explicit_group": null
  },
  "SubmitKernel in order using events long kernel": {
    "type": "group",
    "description": "Measures CPU time overhead of submitting in order kernels with longer execution times through different APIs.",
    "notes": "Each layer builds on top of the previous layer, adding functionality and overhead.\nThe first layer is the Level Zero API, the second is the Unified Runtime API, and the third is the SYCL API.\nThe UR v2 adapter noticeably reduces UR layer overhead, also improving SYCL performance.\nWork is ongoing to reduce the overhead of the SYCL API\n",
    "unstable": null,
    "tags": [
      "submit",
      "micro",
      "SYCL",
      "UR",
      "L0"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": null,
    "explicit_group": null
  },
  "SubmitKernel in order using events long kernel, CPU count": {
    "type": "group",
    "description": "Measures CPU time overhead of submitting in order kernels with longer execution times through different APIs.",
    "notes": "Each layer builds on top of the previous layer, adding functionality and overhead.\nThe first layer is the Level Zero API, the second is the Unified Runtime API, and the third is the SYCL API.\nThe UR v2 adapter noticeably reduces UR layer overhead, also improving SYCL performance.\nWork is ongoing to reduce the overhead of the SYCL API\n",
    "unstable": null,
    "tags": [
      "submit",
      "micro",
      "SYCL",
      "UR",
      "L0"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": null,
    "explicit_group": null
  },
  "SubmitKernel in order with completion long kernel": {
    "type": "group",
    "description": "Measures CPU time overhead of submitting in order kernels with longer execution times through different APIs.",
    "notes": "Each layer builds on top of the previous layer, adding functionality and overhead.\nThe first layer is the Level Zero API, the second is the Unified Runtime API, and the third is the SYCL API.\nThe UR v2 adapter noticeably reduces UR layer overhead, also improving SYCL performance.\nWork is ongoing to reduce the overhead of the SYCL API\n",
    "unstable": null,
    "tags": [
      "submit",
      "micro",
      "SYCL",
      "UR",
      "L0"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": null,
    "explicit_group": null
  },
  "SubmitKernel in order with completion long kernel, CPU count": {
    "type": "group",
    "description": "Measures CPU time overhead of submitting in order kernels with longer execution times through different APIs.",
    "notes": "Each layer builds on top of the previous layer, adding functionality and overhead.\nThe first layer is the Level Zero API, the second is the Unified Runtime API, and the third is the SYCL API.\nThe UR v2 adapter noticeably reduces UR layer overhead, also improving SYCL performance.\nWork is ongoing to reduce the overhead of the SYCL API\n",
    "unstable": null,
    "tags": [
      "submit",
      "micro",
      "SYCL",
      "UR",
      "L0"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": null,
    "explicit_group": null
  },
  "SubmitKernel in order with completion using events long kernel": {
    "type": "group",
    "description": "Measures CPU time overhead of submitting in order kernels with longer execution times through different APIs.",
    "notes": "Each layer builds on top of the previous layer, adding functionality and overhead.\nThe first layer is the Level Zero API, the second is the Unified Runtime API, and the third is the SYCL API.\nThe UR v2 adapter noticeably reduces UR layer overhead, also improving SYCL performance.\nWork is ongoing to reduce the overhead of the SYCL API\n",
    "unstable": null,
    "tags": [
      "submit",
      "micro",
      "SYCL",
      "UR",
      "L0"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": null,
    "explicit_group": null
  },
  "SubmitKernel in order with completion using events long kernel, CPU count": {
    "type": "group",
    "description": "Measures CPU time overhead of submitting in order kernels with longer execution times through different APIs.",
    "notes": "Each layer builds on top of the previous layer, adding functionality and overhead.\nThe first layer is the Level Zero API, the second is the Unified Runtime API, and the third is the SYCL API.\nThe UR v2 adapter noticeably reduces UR layer overhead, also improving SYCL performance.\nWork is ongoing to reduce the overhead of the SYCL API\n",
    "unstable": null,
    "tags": [
      "submit",
      "micro",
      "SYCL",
      "UR",
      "L0"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": null,
    "explicit_group": null
  },
  "SubmitKernel out of order long kernel": {
    "type": "group",
    "description": "Measures CPU time overhead of submitting out of order kernels with longer execution times through different APIs.",
    "notes": "Each layer builds on top of the previous layer, adding functionality and overhead.\nThe first layer is the Level Zero API, the second is the Unified Runtime API, and the third is the SYCL API.\nThe UR v2 adapter noticeably reduces UR layer overhead, also improving SYCL performance.\nWork is ongoing to reduce the overhead of the SYCL API\n",
    "unstable": null,
    "tags": [
      "submit",
      "micro",
      "SYCL",
      "UR",
      "L0"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": null,
    "explicit_group": null
  },
  "SubmitKernel out of order long kernel, CPU count": {
    "type": "group",
    "description": "Measures CPU time overhead of submitting out of order kernels with longer execution times through different APIs.",
    "notes": "Each layer builds on top of the previous layer, adding functionality and overhead.\nThe first layer is the Level Zero API, the second is the Unified Runtime API, and the third is the SYCL API.\nThe UR v2 adapter noticeably reduces UR layer overhead, also improving SYCL performance.\nWork is ongoing to reduce the overhead of the SYCL API\n",
    "unstable": null,
    "tags": [
      "submit",
      "micro",
      "SYCL",
      "UR",
      "L0"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": null,
    "explicit_group": null
  },
  "SubmitKernel out of order using events long kernel": {
    "type": "group",
    "description": "Measures CPU time overhead of submitting out of order kernels with longer execution times through different APIs.",
    "notes": "Each layer builds on top of the previous layer, adding functionality and overhead.\nThe first layer is the Level Zero API, the second is the Unified Runtime API, and the third is the SYCL API.\nThe UR v2 adapter noticeably reduces UR layer overhead, also improving SYCL performance.\nWork is ongoing to reduce the overhead of the SYCL API\n",
    "unstable": null,
    "tags": [
      "submit",
      "micro",
      "SYCL",
      "UR",
      "L0"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": null,
    "explicit_group": null
  },
  "SubmitKernel out of order using events long kernel, CPU count": {
    "type": "group",
    "description": "Measures CPU time overhead of submitting out of order kernels with longer execution times through different APIs.",
    "notes": "Each layer builds on top of the previous layer, adding functionality and overhead.\nThe first layer is the Level Zero API, the second is the Unified Runtime API, and the third is the SYCL API.\nThe UR v2 adapter noticeably reduces UR layer overhead, also improving SYCL performance.\nWork is ongoing to reduce the overhead of the SYCL API\n",
    "unstable": null,
    "tags": [
      "submit",
      "micro",
      "SYCL",
      "UR",
      "L0"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": null,
    "explicit_group": null
  },
  "SubmitKernel out of order with completion long kernel": {
    "type": "group",
    "description": "Measures CPU time overhead of submitting out of order kernels with longer execution times through different APIs.",
    "notes": "Each layer builds on top of the previous layer, adding functionality and overhead.\nThe first layer is the Level Zero API, the second is the Unified Runtime API, and the third is the SYCL API.\nThe UR v2 adapter noticeably reduces UR layer overhead, also improving SYCL performance.\nWork is ongoing to reduce the overhead of the SYCL API\n",
    "unstable": null,
    "tags": [
      "submit",
      "micro",
      "SYCL",
      "UR",
      "L0"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": null,
    "explicit_group": null
  },
  "SubmitKernel out of order with completion long kernel, CPU count": {
    "type": "group",
    "description": "Measures CPU time overhead of submitting out of order kernels with longer execution times through different APIs.",
    "notes": "Each layer builds on top of the previous layer, adding functionality and overhead.\nThe first layer is the Level Zero API, the second is the Unified Runtime API, and the third is the SYCL API.\nThe UR v2 adapter noticeably reduces UR layer overhead, also improving SYCL performance.\nWork is ongoing to reduce the overhead of the SYCL API\n",
    "unstable": null,
    "tags": [
      "submit",
      "micro",
      "SYCL",
      "UR",
      "L0"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": null,
    "explicit_group": null
  },
  "SubmitKernel out of order with completion using events long kernel": {
    "type": "group",
    "description": "Measures CPU time overhead of submitting out of order kernels with longer execution times through different APIs.",
    "notes": "Each layer builds on top of the previous layer, adding functionality and overhead.\nThe first layer is the Level Zero API, the second is the Unified Runtime API, and the third is the SYCL API.\nThe UR v2 adapter noticeably reduces UR layer overhead, also improving SYCL performance.\nWork is ongoing to reduce the overhead of the SYCL API\n",
    "unstable": null,
    "tags": [
      "submit",
      "micro",
      "SYCL",
      "UR",
      "L0"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": null,
    "explicit_group": null
  },
  "SubmitKernel out of order with completion using events long kernel, CPU count": {
    "type": "group",
    "description": "Measures CPU time overhead of submitting out of order kernels with longer execution times through different APIs.",
    "notes": "Each layer builds on top of the previous layer, adding functionality and overhead.\nThe first layer is the Level Zero API, the second is the Unified Runtime API, and the third is the SYCL API.\nThe UR v2 adapter noticeably reduces UR layer overhead, also improving SYCL performance.\nWork is ongoing to reduce the overhead of the SYCL API\n",
    "unstable": null,
    "tags": [
      "submit",
      "micro",
      "SYCL",
      "UR",
      "L0"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": null,
    "explicit_group": null
  },
  "SubmitGraph in order, 4 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "micro",
      "SYCL",
      "UR",
      "L0",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": null
  },
  "SubmitGraph in order, 10 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "micro",
      "SYCL",
      "UR",
      "L0",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": null
  },
  "SubmitGraph in order, 32 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "micro",
      "SYCL",
      "UR",
      "L0",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": null
  },
  "SubmitGraph in order using events, 4 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "micro",
      "SYCL",
      "UR",
      "L0",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": null
  },
  "SubmitGraph in order using events, 10 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "micro",
      "SYCL",
      "UR",
      "L0",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": null
  },
  "SubmitGraph in order using events, 32 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "micro",
      "SYCL",
      "UR",
      "L0",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": null
  },
  "SubmitGraph in order with completion, 4 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "micro",
      "SYCL",
      "UR",
      "L0",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": null
  },
  "SubmitGraph in order with completion, 10 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "micro",
      "SYCL",
      "UR",
      "L0",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": null
  },
  "SubmitGraph in order with completion, 32 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "micro",
      "SYCL",
      "UR",
      "L0",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": null
  },
  "SubmitGraph in order with completion using events, 4 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "micro",
      "SYCL",
      "UR",
      "L0",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": null
  },
  "SubmitGraph in order with completion using events, 10 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "micro",
      "SYCL",
      "UR",
      "L0",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": null
  },
  "SubmitGraph in order with completion using events, 32 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "micro",
      "SYCL",
      "UR",
      "L0",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": null
  },
  "SubmitGraph out of order, 4 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "micro",
      "SYCL",
      "UR",
      "L0",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": null
  },
  "SubmitGraph out of order, 10 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "micro",
      "SYCL",
      "UR",
      "L0",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": null
  },
  "SubmitGraph out of order, 32 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "micro",
      "SYCL",
      "UR",
      "L0",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": null
  },
  "SubmitGraph out of order using events, 4 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "micro",
      "SYCL",
      "UR",
      "L0",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": null
  },
  "SubmitGraph out of order using events, 10 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "micro",
      "SYCL",
      "UR",
      "L0",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": null
  },
  "SubmitGraph out of order using events, 32 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "micro",
      "SYCL",
      "UR",
      "L0",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": null
  },
  "SubmitGraph out of order with completion, 4 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "micro",
      "SYCL",
      "UR",
      "L0",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": null
  },
  "SubmitGraph out of order with completion, 10 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "micro",
      "SYCL",
      "UR",
      "L0",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": null
  },
  "SubmitGraph out of order with completion, 32 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "micro",
      "SYCL",
      "UR",
      "L0",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": null
  },
  "SubmitGraph out of order with completion using events, 4 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "micro",
      "SYCL",
      "UR",
      "L0",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": null
  },
  "SubmitGraph out of order with completion using events, 10 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "micro",
      "SYCL",
      "UR",
      "L0",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": null
  },
  "SubmitGraph out of order with completion using events, 32 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "micro",
      "SYCL",
      "UR",
      "L0",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": null
  },
  "api_overhead_benchmark_syclpreview SubmitKernel out of order not using events": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL Preview API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel out of order, NumKernels 10",
    "explicit_group": "SubmitKernel out of order"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel out of order not using events CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL Preview API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel out of order, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order, CPU count"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel out of order not using events KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL Preview API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel out of order KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel out of order long kernel"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel out of order not using events KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL Preview API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel out of order KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order long kernel, CPU count"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel out of order not using events KernelExecTime=200": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL Preview API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 200 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel out of order KernelExecTime=200, NumKernels 10",
    "explicit_group": "SubmitKernel out of order long kernel"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel out of order not using events KernelExecTime=200 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL Preview API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 200 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel out of order KernelExecTime=200, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order long kernel, CPU count"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel out of order": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL Preview API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel out of order using events, NumKernels 10",
    "explicit_group": "SubmitKernel out of order using events"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel out of order CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL Preview API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel out of order using events, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order using events, CPU count"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel out of order KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL Preview API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel out of order using events KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel out of order using events long kernel"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel out of order KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL Preview API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel out of order using events KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order using events long kernel, CPU count"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel out of order KernelExecTime=200": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL Preview API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 200 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel out of order using events KernelExecTime=200, NumKernels 10",
    "explicit_group": "SubmitKernel out of order using events long kernel"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel out of order KernelExecTime=200 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL Preview API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 200 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel out of order using events KernelExecTime=200, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order using events long kernel, CPU count"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel out of order with measure completion not using events": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL Preview API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel out of order with measure completion, NumKernels 10",
    "explicit_group": "SubmitKernel out of order with completion"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel out of order with measure completion not using events CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL Preview API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel out of order with measure completion, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order with completion, CPU count"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel out of order with measure completion not using events KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL Preview API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel out of order with measure completion KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel out of order with completion long kernel"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel out of order with measure completion not using events KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL Preview API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel out of order with measure completion KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order with completion long kernel, CPU count"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel out of order with measure completion not using events KernelExecTime=200": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL Preview API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 200 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel out of order with measure completion KernelExecTime=200, NumKernels 10",
    "explicit_group": "SubmitKernel out of order with completion long kernel"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel out of order with measure completion not using events KernelExecTime=200 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL Preview API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 200 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel out of order with measure completion KernelExecTime=200, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order with completion long kernel, CPU count"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel out of order with measure completion": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL Preview API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel out of order with measure completion using events, NumKernels 10",
    "explicit_group": "SubmitKernel out of order with completion using events"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel out of order with measure completion CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL Preview API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel out of order with measure completion using events, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order with completion using events, CPU count"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel out of order with measure completion KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL Preview API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel out of order with measure completion using events KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel out of order with completion using events long kernel"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel out of order with measure completion KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL Preview API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel out of order with measure completion using events KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order with completion using events long kernel, CPU count"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel out of order with measure completion KernelExecTime=200": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL Preview API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 200 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel out of order with measure completion using events KernelExecTime=200, NumKernels 10",
    "explicit_group": "SubmitKernel out of order with completion using events long kernel"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel out of order with measure completion KernelExecTime=200 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL Preview API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 200 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel out of order with measure completion using events KernelExecTime=200, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order with completion using events long kernel, CPU count"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel in order not using events": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL Preview API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel in order, NumKernels 10",
    "explicit_group": "SubmitKernel in order"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel in order not using events CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL Preview API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel in order, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order, CPU count"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel in order not using events KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL Preview API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel in order KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel in order long kernel"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel in order not using events KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL Preview API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel in order KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order long kernel, CPU count"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel in order": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL Preview API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel in order using events, NumKernels 10",
    "explicit_group": "SubmitKernel in order using events"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel in order CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL Preview API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel in order using events, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order using events, CPU count"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel in order KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL Preview API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel in order using events KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel in order using events long kernel"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel in order KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL Preview API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel in order using events KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order using events long kernel, CPU count"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel in order with measure completion not using events": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL Preview API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel in order with measure completion, NumKernels 10",
    "explicit_group": "SubmitKernel in order with completion"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel in order with measure completion not using events CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL Preview API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel in order with measure completion, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order with completion, CPU count"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel in order with measure completion not using events KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL Preview API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel in order with measure completion KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel in order with completion long kernel"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel in order with measure completion not using events KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL Preview API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel in order with measure completion KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order with completion long kernel, CPU count"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel in order with measure completion": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL Preview API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel in order with measure completion using events, NumKernels 10",
    "explicit_group": "SubmitKernel in order with completion using events"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel in order with measure completion CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL Preview API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel in order with measure completion using events, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order with completion using events, CPU count"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel in order with measure completion KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL Preview API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel in order with measure completion using events KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel in order with completion using events long kernel"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel in order with measure completion KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL Preview API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel in order with measure completion using events KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order with completion using events long kernel, CPU count"
  },
  "graph_api_benchmark_syclpreview SinKernelGraph graphs:0, numKernels:5": {
    "type": "benchmark",
    "description": "Measures SYCLPREVIEW performance when executing 5 sin kernels without graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": "This benchmark combines both eager and graph execution, and may not be representative of real use cases.",
    "tags": [
      "graph",
      "SYCL",
      "proxy",
      "submit",
      "memory",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW SinKernelGraph, graphs 0, numKernels 5",
    "explicit_group": "SinKernelGraph, numKernels: 5"
  },
  "graph_api_benchmark_syclpreview SinKernelGraph graphs:0, numKernels:100": {
    "type": "benchmark",
    "description": "Measures SYCLPREVIEW performance when executing 100 sin kernels without graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": "This benchmark combines both eager and graph execution, and may not be representative of real use cases.",
    "tags": [
      "graph",
      "SYCL",
      "proxy",
      "submit",
      "memory",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW SinKernelGraph, graphs 0, numKernels 100",
    "explicit_group": "SinKernelGraph, numKernels: 100"
  },
  "graph_api_benchmark_syclpreview SinKernelGraph graphs:1, numKernels:5": {
    "type": "benchmark",
    "description": "Measures SYCLPREVIEW performance when executing 5 sin kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": "This benchmark combines both eager and graph execution, and may not be representative of real use cases.",
    "tags": [
      "graph",
      "SYCL",
      "proxy",
      "submit",
      "memory",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW SinKernelGraph, graphs 1, numKernels 5",
    "explicit_group": "SinKernelGraph, numKernels: 5"
  },
  "graph_api_benchmark_syclpreview SinKernelGraph graphs:1, numKernels:100": {
    "type": "benchmark",
    "description": "Measures SYCLPREVIEW performance when executing 100 sin kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": "This benchmark combines both eager and graph execution, and may not be representative of real use cases.",
    "tags": [
      "graph",
      "SYCL",
      "proxy",
      "submit",
      "memory",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW SinKernelGraph, graphs 1, numKernels 100",
    "explicit_group": "SinKernelGraph, numKernels: 100"
  },
  "ulls_benchmark_syclpreview EmptyKernel wgc:1000, wgs:256": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "SYCL",
      "micro",
      "latency",
      "submit"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW EmptyKernel, wgc 1000, wgs 256",
    "explicit_group": "EmptyKernel, wgc: 1000, wgs: 256"
  },
  "ulls_benchmark_syclpreview KernelSwitch count 8 kernelTime 200": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "SYCL",
      "micro",
      "latency",
      "submit"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSwitch, count 8, kernelTime 200",
    "explicit_group": "KernelSwitch, count: 8, kernelTime: 200"
  },
  "graph_api_benchmark_syclpreview SubmitGraph numKernels:4 ioq 0 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures SYCLPREVIEW performance when executing 4 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitGraph out of order, 4 kernels",
    "explicit_group": "SubmitGraph out of order, 4 kernels"
  },
  "graph_api_benchmark_syclpreview SubmitGraph with events numKernels:4 ioq 0 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures SYCLPREVIEW performance when executing 4 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitGraph out of order with events, 4 kernels",
    "explicit_group": "SubmitGraph out of order with events, 4 kernels"
  },
  "graph_api_benchmark_syclpreview SubmitGraph numKernels:4 ioq 0 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures SYCLPREVIEW performance when executing 4 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitGraph out of order with measure completion, 4 kernels",
    "explicit_group": "SubmitGraph out of order with measure completion, 4 kernels"
  },
  "graph_api_benchmark_syclpreview SubmitGraph with events numKernels:4 ioq 0 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures SYCLPREVIEW performance when executing 4 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitGraph out of order with measure completion with events, 4 kernels",
    "explicit_group": "SubmitGraph out of order with measure completion with events, 4 kernels"
  },
  "graph_api_benchmark_syclpreview SubmitGraph numKernels:10 ioq 0 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures SYCLPREVIEW performance when executing 10 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitGraph out of order, 10 kernels",
    "explicit_group": "SubmitGraph out of order, 10 kernels"
  },
  "graph_api_benchmark_syclpreview SubmitGraph with events numKernels:10 ioq 0 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures SYCLPREVIEW performance when executing 10 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitGraph out of order with events, 10 kernels",
    "explicit_group": "SubmitGraph out of order with events, 10 kernels"
  },
  "graph_api_benchmark_syclpreview SubmitGraph numKernels:10 ioq 0 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures SYCLPREVIEW performance when executing 10 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitGraph out of order with measure completion, 10 kernels",
    "explicit_group": "SubmitGraph out of order with measure completion, 10 kernels"
  },
  "graph_api_benchmark_syclpreview SubmitGraph with events numKernels:10 ioq 0 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures SYCLPREVIEW performance when executing 10 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitGraph out of order with measure completion with events, 10 kernels",
    "explicit_group": "SubmitGraph out of order with measure completion with events, 10 kernels"
  },
  "graph_api_benchmark_syclpreview SubmitGraph numKernels:32 ioq 0 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures SYCLPREVIEW performance when executing 32 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitGraph out of order, 32 kernels",
    "explicit_group": "SubmitGraph out of order, 32 kernels"
  },
  "graph_api_benchmark_syclpreview SubmitGraph with events numKernels:32 ioq 0 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures SYCLPREVIEW performance when executing 32 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitGraph out of order with events, 32 kernels",
    "explicit_group": "SubmitGraph out of order with events, 32 kernels"
  },
  "graph_api_benchmark_syclpreview SubmitGraph numKernels:32 ioq 0 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures SYCLPREVIEW performance when executing 32 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitGraph out of order with measure completion, 32 kernels",
    "explicit_group": "SubmitGraph out of order with measure completion, 32 kernels"
  },
  "graph_api_benchmark_syclpreview SubmitGraph with events numKernels:32 ioq 0 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures SYCLPREVIEW performance when executing 32 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitGraph out of order with measure completion with events, 32 kernels",
    "explicit_group": "SubmitGraph out of order with measure completion with events, 32 kernels"
  },
  "graph_api_benchmark_syclpreview SubmitGraph numKernels:4 ioq 1 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures SYCLPREVIEW performance when executing 4 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitGraph in order, 4 kernels",
    "explicit_group": "SubmitGraph in order, 4 kernels"
  },
  "graph_api_benchmark_syclpreview SubmitGraph with events numKernels:4 ioq 1 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures SYCLPREVIEW performance when executing 4 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitGraph in order with events, 4 kernels",
    "explicit_group": "SubmitGraph in order with events, 4 kernels"
  },
  "graph_api_benchmark_syclpreview SubmitGraph numKernels:4 ioq 1 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures SYCLPREVIEW performance when executing 4 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitGraph in order with measure completion, 4 kernels",
    "explicit_group": "SubmitGraph in order with measure completion, 4 kernels"
  },
  "graph_api_benchmark_syclpreview SubmitGraph with events numKernels:4 ioq 1 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures SYCLPREVIEW performance when executing 4 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitGraph in order with measure completion with events, 4 kernels",
    "explicit_group": "SubmitGraph in order with measure completion with events, 4 kernels"
  },
  "graph_api_benchmark_syclpreview SubmitGraph numKernels:10 ioq 1 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures SYCLPREVIEW performance when executing 10 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitGraph in order, 10 kernels",
    "explicit_group": "SubmitGraph in order, 10 kernels"
  },
  "graph_api_benchmark_syclpreview SubmitGraph with events numKernels:10 ioq 1 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures SYCLPREVIEW performance when executing 10 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitGraph in order with events, 10 kernels",
    "explicit_group": "SubmitGraph in order with events, 10 kernels"
  },
  "graph_api_benchmark_syclpreview SubmitGraph numKernels:10 ioq 1 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures SYCLPREVIEW performance when executing 10 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitGraph in order with measure completion, 10 kernels",
    "explicit_group": "SubmitGraph in order with measure completion, 10 kernels"
  },
  "graph_api_benchmark_syclpreview SubmitGraph with events numKernels:10 ioq 1 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures SYCLPREVIEW performance when executing 10 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitGraph in order with measure completion with events, 10 kernels",
    "explicit_group": "SubmitGraph in order with measure completion with events, 10 kernels"
  },
  "graph_api_benchmark_syclpreview SubmitGraph numKernels:32 ioq 1 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures SYCLPREVIEW performance when executing 32 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitGraph in order, 32 kernels",
    "explicit_group": "SubmitGraph in order, 32 kernels"
  },
  "graph_api_benchmark_syclpreview SubmitGraph with events numKernels:32 ioq 1 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures SYCLPREVIEW performance when executing 32 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitGraph in order with events, 32 kernels",
    "explicit_group": "SubmitGraph in order with events, 32 kernels"
  },
  "graph_api_benchmark_syclpreview SubmitGraph numKernels:32 ioq 1 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures SYCLPREVIEW performance when executing 32 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitGraph in order with measure completion, 32 kernels",
    "explicit_group": "SubmitGraph in order with measure completion, 32 kernels"
  },
  "graph_api_benchmark_syclpreview SubmitGraph with events numKernels:32 ioq 1 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures SYCLPREVIEW performance when executing 32 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitGraph in order with measure completion with events, 32 kernels",
    "explicit_group": "SubmitGraph in order with measure completion with events, 32 kernels"
  },
  "api_overhead_benchmark_sycl SubmitKernel out of order not using events": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel out of order, NumKernels 10",
    "explicit_group": "SubmitKernel out of order"
  },
  "api_overhead_benchmark_sycl SubmitKernel out of order not using events CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel out of order, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order, CPU count"
  },
  "api_overhead_benchmark_sycl SubmitKernel out of order not using events KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel out of order KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel out of order long kernel"
  },
  "api_overhead_benchmark_sycl SubmitKernel out of order not using events KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel out of order KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order long kernel, CPU count"
  },
  "api_overhead_benchmark_sycl SubmitKernel out of order not using events KernelExecTime=200": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 200 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel out of order KernelExecTime=200, NumKernels 10",
    "explicit_group": "SubmitKernel out of order long kernel"
  },
  "api_overhead_benchmark_sycl SubmitKernel out of order not using events KernelExecTime=200 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 200 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel out of order KernelExecTime=200, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order long kernel, CPU count"
  },
  "api_overhead_benchmark_sycl SubmitKernel out of order": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel out of order using events, NumKernels 10",
    "explicit_group": "SubmitKernel out of order using events"
  },
  "api_overhead_benchmark_sycl SubmitKernel out of order CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel out of order using events, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order using events, CPU count"
  },
  "api_overhead_benchmark_sycl SubmitKernel out of order KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel out of order using events KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel out of order using events long kernel"
  },
  "api_overhead_benchmark_sycl SubmitKernel out of order KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel out of order using events KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order using events long kernel, CPU count"
  },
  "api_overhead_benchmark_sycl SubmitKernel out of order KernelExecTime=200": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 200 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel out of order using events KernelExecTime=200, NumKernels 10",
    "explicit_group": "SubmitKernel out of order using events long kernel"
  },
  "api_overhead_benchmark_sycl SubmitKernel out of order KernelExecTime=200 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 200 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel out of order using events KernelExecTime=200, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order using events long kernel, CPU count"
  },
  "api_overhead_benchmark_sycl SubmitKernel out of order with measure completion not using events": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel out of order with measure completion, NumKernels 10",
    "explicit_group": "SubmitKernel out of order with completion"
  },
  "api_overhead_benchmark_sycl SubmitKernel out of order with measure completion not using events CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel out of order with measure completion, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order with completion, CPU count"
  },
  "api_overhead_benchmark_sycl SubmitKernel out of order with measure completion not using events KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel out of order with measure completion KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel out of order with completion long kernel"
  },
  "api_overhead_benchmark_sycl SubmitKernel out of order with measure completion not using events KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel out of order with measure completion KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order with completion long kernel, CPU count"
  },
  "api_overhead_benchmark_sycl SubmitKernel out of order with measure completion not using events KernelExecTime=200": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 200 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel out of order with measure completion KernelExecTime=200, NumKernels 10",
    "explicit_group": "SubmitKernel out of order with completion long kernel"
  },
  "api_overhead_benchmark_sycl SubmitKernel out of order with measure completion not using events KernelExecTime=200 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 200 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel out of order with measure completion KernelExecTime=200, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order with completion long kernel, CPU count"
  },
  "api_overhead_benchmark_sycl SubmitKernel out of order with measure completion": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel out of order with measure completion using events, NumKernels 10",
    "explicit_group": "SubmitKernel out of order with completion using events"
  },
  "api_overhead_benchmark_sycl SubmitKernel out of order with measure completion CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel out of order with measure completion using events, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order with completion using events, CPU count"
  },
  "api_overhead_benchmark_sycl SubmitKernel out of order with measure completion KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel out of order with measure completion using events KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel out of order with completion using events long kernel"
  },
  "api_overhead_benchmark_sycl SubmitKernel out of order with measure completion KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel out of order with measure completion using events KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order with completion using events long kernel, CPU count"
  },
  "api_overhead_benchmark_sycl SubmitKernel out of order with measure completion KernelExecTime=200": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 200 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel out of order with measure completion using events KernelExecTime=200, NumKernels 10",
    "explicit_group": "SubmitKernel out of order with completion using events long kernel"
  },
  "api_overhead_benchmark_sycl SubmitKernel out of order with measure completion KernelExecTime=200 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 200 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel out of order with measure completion using events KernelExecTime=200, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order with completion using events long kernel, CPU count"
  },
  "api_overhead_benchmark_sycl SubmitKernel in order not using events": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel in order, NumKernels 10",
    "explicit_group": "SubmitKernel in order"
  },
  "api_overhead_benchmark_sycl SubmitKernel in order not using events CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel in order, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order, CPU count"
  },
  "api_overhead_benchmark_sycl SubmitKernel in order not using events KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel in order KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel in order long kernel"
  },
  "api_overhead_benchmark_sycl SubmitKernel in order not using events KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel in order KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order long kernel, CPU count"
  },
  "api_overhead_benchmark_sycl SubmitKernel in order": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel in order using events, NumKernels 10",
    "explicit_group": "SubmitKernel in order using events"
  },
  "api_overhead_benchmark_sycl SubmitKernel in order CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel in order using events, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order using events, CPU count"
  },
  "api_overhead_benchmark_sycl SubmitKernel in order KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel in order using events KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel in order using events long kernel"
  },
  "api_overhead_benchmark_sycl SubmitKernel in order KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel in order using events KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order using events long kernel, CPU count"
  },
  "api_overhead_benchmark_sycl SubmitKernel in order with measure completion not using events": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel in order with measure completion, NumKernels 10",
    "explicit_group": "SubmitKernel in order with completion"
  },
  "api_overhead_benchmark_sycl SubmitKernel in order with measure completion not using events CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel in order with measure completion, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order with completion, CPU count"
  },
  "api_overhead_benchmark_sycl SubmitKernel in order with measure completion not using events KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel in order with measure completion KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel in order with completion long kernel"
  },
  "api_overhead_benchmark_sycl SubmitKernel in order with measure completion not using events KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel in order with measure completion KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order with completion long kernel, CPU count"
  },
  "api_overhead_benchmark_sycl SubmitKernel in order with measure completion": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel in order with measure completion using events, NumKernels 10",
    "explicit_group": "SubmitKernel in order with completion using events"
  },
  "api_overhead_benchmark_sycl SubmitKernel in order with measure completion CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel in order with measure completion using events, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order with completion using events, CPU count"
  },
  "api_overhead_benchmark_sycl SubmitKernel in order with measure completion KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel in order with measure completion using events KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel in order with completion using events long kernel"
  },
  "api_overhead_benchmark_sycl SubmitKernel in order with measure completion KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel in order with measure completion using events KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order with completion using events long kernel, CPU count"
  },
  "graph_api_benchmark_sycl SinKernelGraph graphs:0, numKernels:5": {
    "type": "benchmark",
    "description": "Measures SYCL performance when executing 5 sin kernels without graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": "This benchmark combines both eager and graph execution, and may not be representative of real use cases.",
    "tags": [
      "graph",
      "SYCL",
      "proxy",
      "submit",
      "memory",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL SinKernelGraph, graphs 0, numKernels 5",
    "explicit_group": "SinKernelGraph, numKernels: 5"
  },
  "graph_api_benchmark_sycl SinKernelGraph graphs:0, numKernels:100": {
    "type": "benchmark",
    "description": "Measures SYCL performance when executing 100 sin kernels without graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": "This benchmark combines both eager and graph execution, and may not be representative of real use cases.",
    "tags": [
      "graph",
      "SYCL",
      "proxy",
      "submit",
      "memory",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL SinKernelGraph, graphs 0, numKernels 100",
    "explicit_group": "SinKernelGraph, numKernels: 100"
  },
  "graph_api_benchmark_sycl SinKernelGraph graphs:1, numKernels:5": {
    "type": "benchmark",
    "description": "Measures SYCL performance when executing 5 sin kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": "This benchmark combines both eager and graph execution, and may not be representative of real use cases.",
    "tags": [
      "graph",
      "SYCL",
      "proxy",
      "submit",
      "memory",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL SinKernelGraph, graphs 1, numKernels 5",
    "explicit_group": "SinKernelGraph, numKernels: 5"
  },
  "graph_api_benchmark_sycl SinKernelGraph graphs:1, numKernels:100": {
    "type": "benchmark",
    "description": "Measures SYCL performance when executing 100 sin kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": "This benchmark combines both eager and graph execution, and may not be representative of real use cases.",
    "tags": [
      "graph",
      "SYCL",
      "proxy",
      "submit",
      "memory",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL SinKernelGraph, graphs 1, numKernels 100",
    "explicit_group": "SinKernelGraph, numKernels: 100"
  },
  "ulls_benchmark_sycl EmptyKernel wgc:1000, wgs:256": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "SYCL",
      "micro",
      "latency",
      "submit"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL EmptyKernel, wgc 1000, wgs 256",
    "explicit_group": "EmptyKernel, wgc: 1000, wgs: 256"
  },
  "ulls_benchmark_sycl KernelSwitch count 8 kernelTime 200": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "SYCL",
      "micro",
      "latency",
      "submit"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSwitch, count 8, kernelTime 200",
    "explicit_group": "KernelSwitch, count: 8, kernelTime: 200"
  },
  "graph_api_benchmark_sycl SubmitGraph numKernels:4 ioq 0 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures SYCL performance when executing 4 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL SubmitGraph out of order, 4 kernels",
    "explicit_group": "SubmitGraph out of order, 4 kernels"
  },
  "graph_api_benchmark_sycl SubmitGraph with events numKernels:4 ioq 0 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures SYCL performance when executing 4 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL SubmitGraph out of order with events, 4 kernels",
    "explicit_group": "SubmitGraph out of order with events, 4 kernels"
  },
  "graph_api_benchmark_sycl SubmitGraph numKernels:4 ioq 0 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures SYCL performance when executing 4 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL SubmitGraph out of order with measure completion, 4 kernels",
    "explicit_group": "SubmitGraph out of order with measure completion, 4 kernels"
  },
  "graph_api_benchmark_sycl SubmitGraph with events numKernels:4 ioq 0 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures SYCL performance when executing 4 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL SubmitGraph out of order with measure completion with events, 4 kernels",
    "explicit_group": "SubmitGraph out of order with measure completion with events, 4 kernels"
  },
  "graph_api_benchmark_sycl SubmitGraph numKernels:10 ioq 0 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures SYCL performance when executing 10 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL SubmitGraph out of order, 10 kernels",
    "explicit_group": "SubmitGraph out of order, 10 kernels"
  },
  "graph_api_benchmark_sycl SubmitGraph with events numKernels:10 ioq 0 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures SYCL performance when executing 10 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL SubmitGraph out of order with events, 10 kernels",
    "explicit_group": "SubmitGraph out of order with events, 10 kernels"
  },
  "graph_api_benchmark_sycl SubmitGraph numKernels:10 ioq 0 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures SYCL performance when executing 10 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL SubmitGraph out of order with measure completion, 10 kernels",
    "explicit_group": "SubmitGraph out of order with measure completion, 10 kernels"
  },
  "graph_api_benchmark_sycl SubmitGraph with events numKernels:10 ioq 0 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures SYCL performance when executing 10 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL SubmitGraph out of order with measure completion with events, 10 kernels",
    "explicit_group": "SubmitGraph out of order with measure completion with events, 10 kernels"
  },
  "graph_api_benchmark_sycl SubmitGraph numKernels:32 ioq 0 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures SYCL performance when executing 32 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL SubmitGraph out of order, 32 kernels",
    "explicit_group": "SubmitGraph out of order, 32 kernels"
  },
  "graph_api_benchmark_sycl SubmitGraph with events numKernels:32 ioq 0 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures SYCL performance when executing 32 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL SubmitGraph out of order with events, 32 kernels",
    "explicit_group": "SubmitGraph out of order with events, 32 kernels"
  },
  "graph_api_benchmark_sycl SubmitGraph numKernels:32 ioq 0 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures SYCL performance when executing 32 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL SubmitGraph out of order with measure completion, 32 kernels",
    "explicit_group": "SubmitGraph out of order with measure completion, 32 kernels"
  },
  "graph_api_benchmark_sycl SubmitGraph with events numKernels:32 ioq 0 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures SYCL performance when executing 32 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL SubmitGraph out of order with measure completion with events, 32 kernels",
    "explicit_group": "SubmitGraph out of order with measure completion with events, 32 kernels"
  },
  "graph_api_benchmark_sycl SubmitGraph numKernels:4 ioq 1 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures SYCL performance when executing 4 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL SubmitGraph in order, 4 kernels",
    "explicit_group": "SubmitGraph in order, 4 kernels"
  },
  "graph_api_benchmark_sycl SubmitGraph with events numKernels:4 ioq 1 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures SYCL performance when executing 4 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL SubmitGraph in order with events, 4 kernels",
    "explicit_group": "SubmitGraph in order with events, 4 kernels"
  },
  "graph_api_benchmark_sycl SubmitGraph numKernels:4 ioq 1 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures SYCL performance when executing 4 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL SubmitGraph in order with measure completion, 4 kernels",
    "explicit_group": "SubmitGraph in order with measure completion, 4 kernels"
  },
  "graph_api_benchmark_sycl SubmitGraph with events numKernels:4 ioq 1 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures SYCL performance when executing 4 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL SubmitGraph in order with measure completion with events, 4 kernels",
    "explicit_group": "SubmitGraph in order with measure completion with events, 4 kernels"
  },
  "graph_api_benchmark_sycl SubmitGraph numKernels:10 ioq 1 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures SYCL performance when executing 10 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL SubmitGraph in order, 10 kernels",
    "explicit_group": "SubmitGraph in order, 10 kernels"
  },
  "graph_api_benchmark_sycl SubmitGraph with events numKernels:10 ioq 1 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures SYCL performance when executing 10 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL SubmitGraph in order with events, 10 kernels",
    "explicit_group": "SubmitGraph in order with events, 10 kernels"
  },
  "graph_api_benchmark_sycl SubmitGraph numKernels:10 ioq 1 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures SYCL performance when executing 10 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL SubmitGraph in order with measure completion, 10 kernels",
    "explicit_group": "SubmitGraph in order with measure completion, 10 kernels"
  },
  "graph_api_benchmark_sycl SubmitGraph with events numKernels:10 ioq 1 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures SYCL performance when executing 10 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL SubmitGraph in order with measure completion with events, 10 kernels",
    "explicit_group": "SubmitGraph in order with measure completion with events, 10 kernels"
  },
  "graph_api_benchmark_sycl SubmitGraph numKernels:32 ioq 1 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures SYCL performance when executing 32 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL SubmitGraph in order, 32 kernels",
    "explicit_group": "SubmitGraph in order, 32 kernels"
  },
  "graph_api_benchmark_sycl SubmitGraph with events numKernels:32 ioq 1 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures SYCL performance when executing 32 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL SubmitGraph in order with events, 32 kernels",
    "explicit_group": "SubmitGraph in order with events, 32 kernels"
  },
  "graph_api_benchmark_sycl SubmitGraph numKernels:32 ioq 1 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures SYCL performance when executing 32 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL SubmitGraph in order with measure completion, 32 kernels",
    "explicit_group": "SubmitGraph in order with measure completion, 32 kernels"
  },
  "graph_api_benchmark_sycl SubmitGraph with events numKernels:32 ioq 1 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures SYCL performance when executing 32 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL SubmitGraph in order with measure completion with events, 32 kernels",
    "explicit_group": "SubmitGraph in order with measure completion with events, 32 kernels"
  },
  "api_overhead_benchmark_l0 SubmitKernel out of order not using events": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Level Zero API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel out of order, NumKernels 10",
    "explicit_group": "SubmitKernel out of order"
  },
  "api_overhead_benchmark_l0 SubmitKernel out of order not using events CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Level Zero API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel out of order, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order, CPU count"
  },
  "api_overhead_benchmark_l0 SubmitKernel out of order not using events KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Level Zero API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel out of order KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel out of order long kernel"
  },
  "api_overhead_benchmark_l0 SubmitKernel out of order not using events KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Level Zero API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel out of order KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order long kernel, CPU count"
  },
  "api_overhead_benchmark_l0 SubmitKernel out of order not using events KernelExecTime=200": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Level Zero API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 200 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel out of order KernelExecTime=200, NumKernels 10",
    "explicit_group": "SubmitKernel out of order long kernel"
  },
  "api_overhead_benchmark_l0 SubmitKernel out of order not using events KernelExecTime=200 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Level Zero API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 200 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel out of order KernelExecTime=200, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order long kernel, CPU count"
  },
  "api_overhead_benchmark_l0 SubmitKernel out of order": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Level Zero API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel out of order using events, NumKernels 10",
    "explicit_group": "SubmitKernel out of order using events"
  },
  "api_overhead_benchmark_l0 SubmitKernel out of order CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Level Zero API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel out of order using events, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order using events, CPU count"
  },
  "api_overhead_benchmark_l0 SubmitKernel out of order KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Level Zero API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel out of order using events KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel out of order using events long kernel"
  },
  "api_overhead_benchmark_l0 SubmitKernel out of order KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Level Zero API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel out of order using events KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order using events long kernel, CPU count"
  },
  "api_overhead_benchmark_l0 SubmitKernel out of order KernelExecTime=200": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Level Zero API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 200 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel out of order using events KernelExecTime=200, NumKernels 10",
    "explicit_group": "SubmitKernel out of order using events long kernel"
  },
  "api_overhead_benchmark_l0 SubmitKernel out of order KernelExecTime=200 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Level Zero API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 200 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel out of order using events KernelExecTime=200, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order using events long kernel, CPU count"
  },
  "api_overhead_benchmark_l0 SubmitKernel out of order with measure completion not using events": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Level Zero API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel out of order with measure completion, NumKernels 10",
    "explicit_group": "SubmitKernel out of order with completion"
  },
  "api_overhead_benchmark_l0 SubmitKernel out of order with measure completion not using events CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Level Zero API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel out of order with measure completion, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order with completion, CPU count"
  },
  "api_overhead_benchmark_l0 SubmitKernel out of order with measure completion not using events KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Level Zero API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel out of order with measure completion KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel out of order with completion long kernel"
  },
  "api_overhead_benchmark_l0 SubmitKernel out of order with measure completion not using events KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Level Zero API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel out of order with measure completion KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order with completion long kernel, CPU count"
  },
  "api_overhead_benchmark_l0 SubmitKernel out of order with measure completion not using events KernelExecTime=200": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Level Zero API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 200 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel out of order with measure completion KernelExecTime=200, NumKernels 10",
    "explicit_group": "SubmitKernel out of order with completion long kernel"
  },
  "api_overhead_benchmark_l0 SubmitKernel out of order with measure completion not using events KernelExecTime=200 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Level Zero API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 200 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel out of order with measure completion KernelExecTime=200, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order with completion long kernel, CPU count"
  },
  "api_overhead_benchmark_l0 SubmitKernel out of order with measure completion": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Level Zero API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel out of order with measure completion using events, NumKernels 10",
    "explicit_group": "SubmitKernel out of order with completion using events"
  },
  "api_overhead_benchmark_l0 SubmitKernel out of order with measure completion CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Level Zero API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel out of order with measure completion using events, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order with completion using events, CPU count"
  },
  "api_overhead_benchmark_l0 SubmitKernel out of order with measure completion KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Level Zero API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel out of order with measure completion using events KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel out of order with completion using events long kernel"
  },
  "api_overhead_benchmark_l0 SubmitKernel out of order with measure completion KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Level Zero API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel out of order with measure completion using events KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order with completion using events long kernel, CPU count"
  },
  "api_overhead_benchmark_l0 SubmitKernel out of order with measure completion KernelExecTime=200": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Level Zero API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 200 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel out of order with measure completion using events KernelExecTime=200, NumKernels 10",
    "explicit_group": "SubmitKernel out of order with completion using events long kernel"
  },
  "api_overhead_benchmark_l0 SubmitKernel out of order with measure completion KernelExecTime=200 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Level Zero API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 200 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel out of order with measure completion using events KernelExecTime=200, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order with completion using events long kernel, CPU count"
  },
  "api_overhead_benchmark_l0 SubmitKernel in order not using events": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Level Zero API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel in order, NumKernels 10",
    "explicit_group": "SubmitKernel in order"
  },
  "api_overhead_benchmark_l0 SubmitKernel in order not using events CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Level Zero API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel in order, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order, CPU count"
  },
  "api_overhead_benchmark_l0 SubmitKernel in order not using events KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Level Zero API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel in order KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel in order long kernel"
  },
  "api_overhead_benchmark_l0 SubmitKernel in order not using events KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Level Zero API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel in order KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order long kernel, CPU count"
  },
  "api_overhead_benchmark_l0 SubmitKernel in order": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Level Zero API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel in order using events, NumKernels 10",
    "explicit_group": "SubmitKernel in order using events"
  },
  "api_overhead_benchmark_l0 SubmitKernel in order CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Level Zero API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel in order using events, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order using events, CPU count"
  },
  "api_overhead_benchmark_l0 SubmitKernel in order KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Level Zero API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel in order using events KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel in order using events long kernel"
  },
  "api_overhead_benchmark_l0 SubmitKernel in order KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Level Zero API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel in order using events KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order using events long kernel, CPU count"
  },
  "api_overhead_benchmark_l0 SubmitKernel in order with measure completion not using events": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Level Zero API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel in order with measure completion, NumKernels 10",
    "explicit_group": "SubmitKernel in order with completion"
  },
  "api_overhead_benchmark_l0 SubmitKernel in order with measure completion not using events CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Level Zero API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel in order with measure completion, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order with completion, CPU count"
  },
  "api_overhead_benchmark_l0 SubmitKernel in order with measure completion not using events KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Level Zero API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel in order with measure completion KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel in order with completion long kernel"
  },
  "api_overhead_benchmark_l0 SubmitKernel in order with measure completion not using events KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Level Zero API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel in order with measure completion KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order with completion long kernel, CPU count"
  },
  "api_overhead_benchmark_l0 SubmitKernel in order with measure completion": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Level Zero API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel in order with measure completion using events, NumKernels 10",
    "explicit_group": "SubmitKernel in order with completion using events"
  },
  "api_overhead_benchmark_l0 SubmitKernel in order with measure completion CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Level Zero API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel in order with measure completion using events, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order with completion using events, CPU count"
  },
  "api_overhead_benchmark_l0 SubmitKernel in order with measure completion KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Level Zero API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel in order with measure completion using events KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel in order with completion using events long kernel"
  },
  "api_overhead_benchmark_l0 SubmitKernel in order with measure completion KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Level Zero API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel in order with measure completion using events KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order with completion using events long kernel, CPU count"
  },
  "graph_api_benchmark_l0 SinKernelGraph graphs:0, numKernels:5": {
    "type": "benchmark",
    "description": "Measures L0 performance when executing 5 sin kernels without graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": "This benchmark combines both eager and graph execution, and may not be representative of real use cases.",
    "tags": [
      "graph",
      "L0",
      "proxy",
      "submit",
      "memory",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 SinKernelGraph, graphs 0, numKernels 5",
    "explicit_group": "SinKernelGraph, numKernels: 5"
  },
  "graph_api_benchmark_l0 SinKernelGraph graphs:0, numKernels:100": {
    "type": "benchmark",
    "description": "Measures L0 performance when executing 100 sin kernels without graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": "This benchmark combines both eager and graph execution, and may not be representative of real use cases.",
    "tags": [
      "graph",
      "L0",
      "proxy",
      "submit",
      "memory",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 SinKernelGraph, graphs 0, numKernels 100",
    "explicit_group": "SinKernelGraph, numKernels: 100"
  },
  "graph_api_benchmark_l0 SinKernelGraph graphs:1, numKernels:5": {
    "type": "benchmark",
    "description": "Measures L0 performance when executing 5 sin kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": "This benchmark combines both eager and graph execution, and may not be representative of real use cases.",
    "tags": [
      "graph",
      "L0",
      "proxy",
      "submit",
      "memory",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 SinKernelGraph, graphs 1, numKernels 5",
    "explicit_group": "SinKernelGraph, numKernels: 5"
  },
  "graph_api_benchmark_l0 SinKernelGraph graphs:1, numKernels:100": {
    "type": "benchmark",
    "description": "Measures L0 performance when executing 100 sin kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": "This benchmark combines both eager and graph execution, and may not be representative of real use cases.",
    "tags": [
      "graph",
      "L0",
      "proxy",
      "submit",
      "memory",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 SinKernelGraph, graphs 1, numKernels 100",
    "explicit_group": "SinKernelGraph, numKernels: 100"
  },
  "ulls_benchmark_l0 EmptyKernel wgc:1000, wgs:256": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "micro",
      "latency",
      "submit"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 EmptyKernel, wgc 1000, wgs 256",
    "explicit_group": "EmptyKernel, wgc: 1000, wgs: 256"
  },
  "ulls_benchmark_l0 KernelSwitch count 8 kernelTime 200": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "micro",
      "latency",
      "submit"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSwitch, count 8, kernelTime 200",
    "explicit_group": "KernelSwitch, count: 8, kernelTime: 200"
  },
  "graph_api_benchmark_l0 SubmitGraph numKernels:4 ioq 0 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures L0 performance when executing 4 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "L0",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 SubmitGraph out of order, 4 kernels",
    "explicit_group": "SubmitGraph out of order, 4 kernels"
  },
  "graph_api_benchmark_l0 SubmitGraph with events numKernels:4 ioq 0 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures L0 performance when executing 4 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "L0",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 SubmitGraph out of order with events, 4 kernels",
    "explicit_group": "SubmitGraph out of order with events, 4 kernels"
  },
  "graph_api_benchmark_l0 SubmitGraph numKernels:4 ioq 0 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures L0 performance when executing 4 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "L0",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 SubmitGraph out of order with measure completion, 4 kernels",
    "explicit_group": "SubmitGraph out of order with measure completion, 4 kernels"
  },
  "graph_api_benchmark_l0 SubmitGraph with events numKernels:4 ioq 0 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures L0 performance when executing 4 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "L0",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 SubmitGraph out of order with measure completion with events, 4 kernels",
    "explicit_group": "SubmitGraph out of order with measure completion with events, 4 kernels"
  },
  "graph_api_benchmark_l0 SubmitGraph numKernels:10 ioq 0 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures L0 performance when executing 10 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "L0",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 SubmitGraph out of order, 10 kernels",
    "explicit_group": "SubmitGraph out of order, 10 kernels"
  },
  "graph_api_benchmark_l0 SubmitGraph with events numKernels:10 ioq 0 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures L0 performance when executing 10 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "L0",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 SubmitGraph out of order with events, 10 kernels",
    "explicit_group": "SubmitGraph out of order with events, 10 kernels"
  },
  "graph_api_benchmark_l0 SubmitGraph numKernels:10 ioq 0 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures L0 performance when executing 10 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "L0",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 SubmitGraph out of order with measure completion, 10 kernels",
    "explicit_group": "SubmitGraph out of order with measure completion, 10 kernels"
  },
  "graph_api_benchmark_l0 SubmitGraph with events numKernels:10 ioq 0 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures L0 performance when executing 10 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "L0",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 SubmitGraph out of order with measure completion with events, 10 kernels",
    "explicit_group": "SubmitGraph out of order with measure completion with events, 10 kernels"
  },
  "graph_api_benchmark_l0 SubmitGraph numKernels:32 ioq 0 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures L0 performance when executing 32 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "L0",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 SubmitGraph out of order, 32 kernels",
    "explicit_group": "SubmitGraph out of order, 32 kernels"
  },
  "graph_api_benchmark_l0 SubmitGraph with events numKernels:32 ioq 0 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures L0 performance when executing 32 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "L0",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 SubmitGraph out of order with events, 32 kernels",
    "explicit_group": "SubmitGraph out of order with events, 32 kernels"
  },
  "graph_api_benchmark_l0 SubmitGraph numKernels:32 ioq 0 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures L0 performance when executing 32 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "L0",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 SubmitGraph out of order with measure completion, 32 kernels",
    "explicit_group": "SubmitGraph out of order with measure completion, 32 kernels"
  },
  "graph_api_benchmark_l0 SubmitGraph with events numKernels:32 ioq 0 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures L0 performance when executing 32 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "L0",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 SubmitGraph out of order with measure completion with events, 32 kernels",
    "explicit_group": "SubmitGraph out of order with measure completion with events, 32 kernels"
  },
  "graph_api_benchmark_l0 SubmitGraph numKernels:4 ioq 1 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures L0 performance when executing 4 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "L0",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 SubmitGraph in order, 4 kernels",
    "explicit_group": "SubmitGraph in order, 4 kernels"
  },
  "graph_api_benchmark_l0 SubmitGraph with events numKernels:4 ioq 1 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures L0 performance when executing 4 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "L0",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 SubmitGraph in order with events, 4 kernels",
    "explicit_group": "SubmitGraph in order with events, 4 kernels"
  },
  "graph_api_benchmark_l0 SubmitGraph numKernels:4 ioq 1 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures L0 performance when executing 4 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "L0",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 SubmitGraph in order with measure completion, 4 kernels",
    "explicit_group": "SubmitGraph in order with measure completion, 4 kernels"
  },
  "graph_api_benchmark_l0 SubmitGraph with events numKernels:4 ioq 1 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures L0 performance when executing 4 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "L0",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 SubmitGraph in order with measure completion with events, 4 kernels",
    "explicit_group": "SubmitGraph in order with measure completion with events, 4 kernels"
  },
  "graph_api_benchmark_l0 SubmitGraph numKernels:10 ioq 1 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures L0 performance when executing 10 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "L0",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 SubmitGraph in order, 10 kernels",
    "explicit_group": "SubmitGraph in order, 10 kernels"
  },
  "graph_api_benchmark_l0 SubmitGraph with events numKernels:10 ioq 1 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures L0 performance when executing 10 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "L0",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 SubmitGraph in order with events, 10 kernels",
    "explicit_group": "SubmitGraph in order with events, 10 kernels"
  },
  "graph_api_benchmark_l0 SubmitGraph numKernels:10 ioq 1 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures L0 performance when executing 10 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "L0",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 SubmitGraph in order with measure completion, 10 kernels",
    "explicit_group": "SubmitGraph in order with measure completion, 10 kernels"
  },
  "graph_api_benchmark_l0 SubmitGraph with events numKernels:10 ioq 1 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures L0 performance when executing 10 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "L0",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 SubmitGraph in order with measure completion with events, 10 kernels",
    "explicit_group": "SubmitGraph in order with measure completion with events, 10 kernels"
  },
  "graph_api_benchmark_l0 SubmitGraph numKernels:32 ioq 1 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures L0 performance when executing 32 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "L0",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 SubmitGraph in order, 32 kernels",
    "explicit_group": "SubmitGraph in order, 32 kernels"
  },
  "graph_api_benchmark_l0 SubmitGraph with events numKernels:32 ioq 1 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures L0 performance when executing 32 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "L0",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 SubmitGraph in order with events, 32 kernels",
    "explicit_group": "SubmitGraph in order with events, 32 kernels"
  },
  "graph_api_benchmark_l0 SubmitGraph numKernels:32 ioq 1 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures L0 performance when executing 32 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "L0",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 SubmitGraph in order with measure completion, 32 kernels",
    "explicit_group": "SubmitGraph in order with measure completion, 32 kernels"
  },
  "graph_api_benchmark_l0 SubmitGraph with events numKernels:32 ioq 1 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures L0 performance when executing 32 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "L0",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 SubmitGraph in order with measure completion with events, 32 kernels",
    "explicit_group": "SubmitGraph in order with measure completion with events, 32 kernels"
  },
  "api_overhead_benchmark_ur SubmitKernel out of order not using events": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Unified Runtime API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel out of order, NumKernels 10",
    "explicit_group": "SubmitKernel out of order"
  },
  "api_overhead_benchmark_ur SubmitKernel out of order not using events CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Unified Runtime API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel out of order, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order, CPU count"
  },
  "api_overhead_benchmark_ur SubmitKernel out of order not using events KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Unified Runtime API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel out of order KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel out of order long kernel"
  },
  "api_overhead_benchmark_ur SubmitKernel out of order not using events KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Unified Runtime API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel out of order KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order long kernel, CPU count"
  },
  "api_overhead_benchmark_ur SubmitKernel out of order not using events KernelExecTime=200": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Unified Runtime API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 200 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel out of order KernelExecTime=200, NumKernels 10",
    "explicit_group": "SubmitKernel out of order long kernel"
  },
  "api_overhead_benchmark_ur SubmitKernel out of order not using events KernelExecTime=200 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Unified Runtime API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 200 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel out of order KernelExecTime=200, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order long kernel, CPU count"
  },
  "api_overhead_benchmark_ur SubmitKernel out of order": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Unified Runtime API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel out of order using events, NumKernels 10",
    "explicit_group": "SubmitKernel out of order using events"
  },
  "api_overhead_benchmark_ur SubmitKernel out of order CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Unified Runtime API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel out of order using events, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order using events, CPU count"
  },
  "api_overhead_benchmark_ur SubmitKernel out of order KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Unified Runtime API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel out of order using events KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel out of order using events long kernel"
  },
  "api_overhead_benchmark_ur SubmitKernel out of order KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Unified Runtime API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel out of order using events KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order using events long kernel, CPU count"
  },
  "api_overhead_benchmark_ur SubmitKernel out of order KernelExecTime=200": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Unified Runtime API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 200 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel out of order using events KernelExecTime=200, NumKernels 10",
    "explicit_group": "SubmitKernel out of order using events long kernel"
  },
  "api_overhead_benchmark_ur SubmitKernel out of order KernelExecTime=200 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Unified Runtime API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 200 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel out of order using events KernelExecTime=200, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order using events long kernel, CPU count"
  },
  "api_overhead_benchmark_ur SubmitKernel out of order with measure completion not using events": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Unified Runtime API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel out of order with measure completion, NumKernels 10",
    "explicit_group": "SubmitKernel out of order with completion"
  },
  "api_overhead_benchmark_ur SubmitKernel out of order with measure completion not using events CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Unified Runtime API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel out of order with measure completion, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order with completion, CPU count"
  },
  "api_overhead_benchmark_ur SubmitKernel out of order with measure completion not using events KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Unified Runtime API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel out of order with measure completion KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel out of order with completion long kernel"
  },
  "api_overhead_benchmark_ur SubmitKernel out of order with measure completion not using events KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Unified Runtime API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel out of order with measure completion KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order with completion long kernel, CPU count"
  },
  "api_overhead_benchmark_ur SubmitKernel out of order with measure completion not using events KernelExecTime=200": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Unified Runtime API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 200 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel out of order with measure completion KernelExecTime=200, NumKernels 10",
    "explicit_group": "SubmitKernel out of order with completion long kernel"
  },
  "api_overhead_benchmark_ur SubmitKernel out of order with measure completion not using events KernelExecTime=200 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Unified Runtime API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 200 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel out of order with measure completion KernelExecTime=200, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order with completion long kernel, CPU count"
  },
  "api_overhead_benchmark_ur SubmitKernel out of order with measure completion": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Unified Runtime API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel out of order with measure completion using events, NumKernels 10",
    "explicit_group": "SubmitKernel out of order with completion using events"
  },
  "api_overhead_benchmark_ur SubmitKernel out of order with measure completion CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Unified Runtime API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel out of order with measure completion using events, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order with completion using events, CPU count"
  },
  "api_overhead_benchmark_ur SubmitKernel out of order with measure completion KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Unified Runtime API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel out of order with measure completion using events KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel out of order with completion using events long kernel"
  },
  "api_overhead_benchmark_ur SubmitKernel out of order with measure completion KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Unified Runtime API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel out of order with measure completion using events KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order with completion using events long kernel, CPU count"
  },
  "api_overhead_benchmark_ur SubmitKernel out of order with measure completion KernelExecTime=200": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Unified Runtime API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 200 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel out of order with measure completion using events KernelExecTime=200, NumKernels 10",
    "explicit_group": "SubmitKernel out of order with completion using events long kernel"
  },
  "api_overhead_benchmark_ur SubmitKernel out of order with measure completion KernelExecTime=200 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Unified Runtime API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 200 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel out of order with measure completion using events KernelExecTime=200, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order with completion using events long kernel, CPU count"
  },
  "api_overhead_benchmark_ur SubmitKernel in order not using events": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Unified Runtime API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel in order, NumKernels 10",
    "explicit_group": "SubmitKernel in order"
  },
  "api_overhead_benchmark_ur SubmitKernel in order not using events CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Unified Runtime API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel in order, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order, CPU count"
  },
  "api_overhead_benchmark_ur SubmitKernel in order not using events KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Unified Runtime API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel in order KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel in order long kernel"
  },
  "api_overhead_benchmark_ur SubmitKernel in order not using events KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Unified Runtime API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel in order KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order long kernel, CPU count"
  },
  "api_overhead_benchmark_ur SubmitKernel in order": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Unified Runtime API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel in order using events, NumKernels 10",
    "explicit_group": "SubmitKernel in order using events"
  },
  "api_overhead_benchmark_ur SubmitKernel in order CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Unified Runtime API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel in order using events, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order using events, CPU count"
  },
  "api_overhead_benchmark_ur SubmitKernel in order KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Unified Runtime API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel in order using events KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel in order using events long kernel"
  },
  "api_overhead_benchmark_ur SubmitKernel in order KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Unified Runtime API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel in order using events KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order using events long kernel, CPU count"
  },
  "api_overhead_benchmark_ur SubmitKernel in order with measure completion not using events": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Unified Runtime API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel in order with measure completion, NumKernels 10",
    "explicit_group": "SubmitKernel in order with completion"
  },
  "api_overhead_benchmark_ur SubmitKernel in order with measure completion not using events CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Unified Runtime API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel in order with measure completion, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order with completion, CPU count"
  },
  "api_overhead_benchmark_ur SubmitKernel in order with measure completion not using events KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Unified Runtime API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel in order with measure completion KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel in order with completion long kernel"
  },
  "api_overhead_benchmark_ur SubmitKernel in order with measure completion not using events KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Unified Runtime API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel in order with measure completion KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order with completion long kernel, CPU count"
  },
  "api_overhead_benchmark_ur SubmitKernel in order with measure completion": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Unified Runtime API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel in order with measure completion using events, NumKernels 10",
    "explicit_group": "SubmitKernel in order with completion using events"
  },
  "api_overhead_benchmark_ur SubmitKernel in order with measure completion CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Unified Runtime API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel in order with measure completion using events, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order with completion using events, CPU count"
  },
  "api_overhead_benchmark_ur SubmitKernel in order with measure completion KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Unified Runtime API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel in order with measure completion using events KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel in order with completion using events long kernel"
  },
  "api_overhead_benchmark_ur SubmitKernel in order with measure completion KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Unified Runtime API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel in order with measure completion using events KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order with completion using events long kernel, CPU count"
  },
  "graph_api_benchmark_ur SinKernelGraph graphs:0, numKernels:5": {
    "type": "benchmark",
    "description": "Measures UR performance when executing 5 sin kernels without graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": "This benchmark combines both eager and graph execution, and may not be representative of real use cases.",
    "tags": [
      "graph",
      "UR",
      "proxy",
      "submit",
      "memory",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR SinKernelGraph, graphs 0, numKernels 5",
    "explicit_group": "SinKernelGraph, numKernels: 5"
  },
  "graph_api_benchmark_ur SinKernelGraph graphs:0, numKernels:100": {
    "type": "benchmark",
    "description": "Measures UR performance when executing 100 sin kernels without graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": "This benchmark combines both eager and graph execution, and may not be representative of real use cases.",
    "tags": [
      "graph",
      "UR",
      "proxy",
      "submit",
      "memory",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR SinKernelGraph, graphs 0, numKernels 100",
    "explicit_group": "SinKernelGraph, numKernels: 100"
  },
  "graph_api_benchmark_ur SinKernelGraph graphs:1, numKernels:5": {
    "type": "benchmark",
    "description": "Measures UR performance when executing 5 sin kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": "This benchmark combines both eager and graph execution, and may not be representative of real use cases.",
    "tags": [
      "graph",
      "UR",
      "proxy",
      "submit",
      "memory",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR SinKernelGraph, graphs 1, numKernels 5",
    "explicit_group": "SinKernelGraph, numKernels: 5"
  },
  "graph_api_benchmark_ur SinKernelGraph graphs:1, numKernels:100": {
    "type": "benchmark",
    "description": "Measures UR performance when executing 100 sin kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": "This benchmark combines both eager and graph execution, and may not be representative of real use cases.",
    "tags": [
      "graph",
      "UR",
      "proxy",
      "submit",
      "memory",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR SinKernelGraph, graphs 1, numKernels 100",
    "explicit_group": "SinKernelGraph, numKernels: 100"
  },
  "ulls_benchmark_ur EmptyKernel wgc:1000, wgs:256": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "UR",
      "micro",
      "latency",
      "submit"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR EmptyKernel, wgc 1000, wgs 256",
    "explicit_group": "EmptyKernel, wgc: 1000, wgs: 256"
  },
  "ulls_benchmark_ur KernelSwitch count 8 kernelTime 200": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "UR",
      "micro",
      "latency",
      "submit"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR KernelSwitch, count 8, kernelTime 200",
    "explicit_group": "KernelSwitch, count: 8, kernelTime: 200"
  },
  "graph_api_benchmark_ur SubmitGraph numKernels:4 ioq 0 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures UR performance when executing 4 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "UR",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR SubmitGraph out of order, 4 kernels",
    "explicit_group": "SubmitGraph out of order, 4 kernels"
  },
  "graph_api_benchmark_ur SubmitGraph with events numKernels:4 ioq 0 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures UR performance when executing 4 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "UR",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR SubmitGraph out of order with events, 4 kernels",
    "explicit_group": "SubmitGraph out of order with events, 4 kernels"
  },
  "graph_api_benchmark_ur SubmitGraph numKernels:4 ioq 0 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures UR performance when executing 4 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "UR",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR SubmitGraph out of order with measure completion, 4 kernels",
    "explicit_group": "SubmitGraph out of order with measure completion, 4 kernels"
  },
  "graph_api_benchmark_ur SubmitGraph with events numKernels:4 ioq 0 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures UR performance when executing 4 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "UR",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR SubmitGraph out of order with measure completion with events, 4 kernels",
    "explicit_group": "SubmitGraph out of order with measure completion with events, 4 kernels"
  },
  "graph_api_benchmark_ur SubmitGraph numKernels:10 ioq 0 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures UR performance when executing 10 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "UR",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR SubmitGraph out of order, 10 kernels",
    "explicit_group": "SubmitGraph out of order, 10 kernels"
  },
  "graph_api_benchmark_ur SubmitGraph with events numKernels:10 ioq 0 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures UR performance when executing 10 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "UR",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR SubmitGraph out of order with events, 10 kernels",
    "explicit_group": "SubmitGraph out of order with events, 10 kernels"
  },
  "graph_api_benchmark_ur SubmitGraph numKernels:10 ioq 0 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures UR performance when executing 10 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "UR",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR SubmitGraph out of order with measure completion, 10 kernels",
    "explicit_group": "SubmitGraph out of order with measure completion, 10 kernels"
  },
  "graph_api_benchmark_ur SubmitGraph with events numKernels:10 ioq 0 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures UR performance when executing 10 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "UR",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR SubmitGraph out of order with measure completion with events, 10 kernels",
    "explicit_group": "SubmitGraph out of order with measure completion with events, 10 kernels"
  },
  "graph_api_benchmark_ur SubmitGraph numKernels:32 ioq 0 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures UR performance when executing 32 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "UR",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR SubmitGraph out of order, 32 kernels",
    "explicit_group": "SubmitGraph out of order, 32 kernels"
  },
  "graph_api_benchmark_ur SubmitGraph with events numKernels:32 ioq 0 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures UR performance when executing 32 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "UR",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR SubmitGraph out of order with events, 32 kernels",
    "explicit_group": "SubmitGraph out of order with events, 32 kernels"
  },
  "graph_api_benchmark_ur SubmitGraph numKernels:32 ioq 0 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures UR performance when executing 32 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "UR",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR SubmitGraph out of order with measure completion, 32 kernels",
    "explicit_group": "SubmitGraph out of order with measure completion, 32 kernels"
  },
  "graph_api_benchmark_ur SubmitGraph with events numKernels:32 ioq 0 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures UR performance when executing 32 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "UR",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR SubmitGraph out of order with measure completion with events, 32 kernels",
    "explicit_group": "SubmitGraph out of order with measure completion with events, 32 kernels"
  },
  "graph_api_benchmark_ur SubmitGraph numKernels:4 ioq 1 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures UR performance when executing 4 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "UR",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR SubmitGraph in order, 4 kernels",
    "explicit_group": "SubmitGraph in order, 4 kernels"
  },
  "graph_api_benchmark_ur SubmitGraph with events numKernels:4 ioq 1 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures UR performance when executing 4 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "UR",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR SubmitGraph in order with events, 4 kernels",
    "explicit_group": "SubmitGraph in order with events, 4 kernels"
  },
  "graph_api_benchmark_ur SubmitGraph numKernels:4 ioq 1 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures UR performance when executing 4 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "UR",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR SubmitGraph in order with measure completion, 4 kernels",
    "explicit_group": "SubmitGraph in order with measure completion, 4 kernels"
  },
  "graph_api_benchmark_ur SubmitGraph with events numKernels:4 ioq 1 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures UR performance when executing 4 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "UR",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR SubmitGraph in order with measure completion with events, 4 kernels",
    "explicit_group": "SubmitGraph in order with measure completion with events, 4 kernels"
  },
  "graph_api_benchmark_ur SubmitGraph numKernels:10 ioq 1 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures UR performance when executing 10 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "UR",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR SubmitGraph in order, 10 kernels",
    "explicit_group": "SubmitGraph in order, 10 kernels"
  },
  "graph_api_benchmark_ur SubmitGraph with events numKernels:10 ioq 1 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures UR performance when executing 10 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "UR",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR SubmitGraph in order with events, 10 kernels",
    "explicit_group": "SubmitGraph in order with events, 10 kernels"
  },
  "graph_api_benchmark_ur SubmitGraph numKernels:10 ioq 1 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures UR performance when executing 10 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "UR",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR SubmitGraph in order with measure completion, 10 kernels",
    "explicit_group": "SubmitGraph in order with measure completion, 10 kernels"
  },
  "graph_api_benchmark_ur SubmitGraph with events numKernels:10 ioq 1 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures UR performance when executing 10 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "UR",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR SubmitGraph in order with measure completion with events, 10 kernels",
    "explicit_group": "SubmitGraph in order with measure completion with events, 10 kernels"
  },
  "graph_api_benchmark_ur SubmitGraph numKernels:32 ioq 1 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures UR performance when executing 32 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "UR",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR SubmitGraph in order, 32 kernels",
    "explicit_group": "SubmitGraph in order, 32 kernels"
  },
  "graph_api_benchmark_ur SubmitGraph with events numKernels:32 ioq 1 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures UR performance when executing 32 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "UR",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR SubmitGraph in order with events, 32 kernels",
    "explicit_group": "SubmitGraph in order with events, 32 kernels"
  },
  "graph_api_benchmark_ur SubmitGraph numKernels:32 ioq 1 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures UR performance when executing 32 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "UR",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR SubmitGraph in order with measure completion, 32 kernels",
    "explicit_group": "SubmitGraph in order with measure completion, 32 kernels"
  },
  "graph_api_benchmark_ur SubmitGraph with events numKernels:32 ioq 1 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures UR performance when executing 32 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "UR",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR SubmitGraph in order with measure completion with events, 32 kernels",
    "explicit_group": "SubmitGraph in order with measure completion with events, 32 kernels"
  },
  "memory_benchmark_sycl QueueInOrderMemcpy from Device to Device, size 1024": {
    "type": "benchmark",
    "description": "Measures SYCL in-order queue memory copy performance for copy and command submission from Device to Device with 1024 bytes, executed 100 times per iteration.",
    "notes": null,
    "unstable": null,
    "tags": [
      "memory",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL QueueInOrderMemcpy from Device to Device, size 1024",
    "explicit_group": ""
  },
  "memory_benchmark_sycl QueueInOrderMemcpy from Host to Device, size 1024": {
    "type": "benchmark",
    "description": "Measures SYCL in-order queue memory copy performance for copy and command submission from Host to Device with 1024 bytes, executed 100 times per iteration.",
    "notes": null,
    "unstable": null,
    "tags": [
      "memory",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL QueueInOrderMemcpy from Host to Device, size 1024",
    "explicit_group": ""
  },
  "memory_benchmark_sycl QueueMemcpy from Device to Device, size 1024": {
    "type": "benchmark",
    "description": "Measures general SYCL queue memory copy performance from Device to Device with 1024 bytes per operation.",
    "notes": null,
    "unstable": null,
    "tags": [
      "memory",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL QueueMemcpy from Device to Device, size 1024",
    "explicit_group": ""
  },
  "memory_benchmark_sycl StreamMemory, placement Device, type Triad, size 10240": {
    "type": "benchmark",
    "description": "Measures Device memory bandwidth using Triad pattern with 10240 bytes. Higher values (GB/s) indicate better performance.",
    "notes": null,
    "unstable": null,
    "tags": [
      "memory",
      "throughput",
      "SYCL",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL StreamMemory, placement Device, type Triad, size 10240",
    "explicit_group": ""
  },
  "api_overhead_benchmark_sycl ExecImmediateCopyQueue out of order from Device to Device, size 1024": {
    "type": "benchmark",
    "description": "Measures SYCL out-of-order queue overhead for copy-only from Device to Device memory with 1024 bytes. Tests immediate execution overheads.",
    "notes": null,
    "unstable": null,
    "tags": [
      "memory",
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL ExecImmediateCopyQueue out of order from Device to Device, size 1024",
    "explicit_group": ""
  },
  "api_overhead_benchmark_sycl ExecImmediateCopyQueue in order from Device to Host, size 1024": {
    "type": "benchmark",
    "description": "Measures SYCL in-order queue overhead for copy-only from Device to Host memory with 1024 bytes. Tests immediate execution overheads.",
    "notes": null,
    "unstable": null,
    "tags": [
      "memory",
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL ExecImmediateCopyQueue in order from Device to Host, size 1024",
    "explicit_group": ""
  },
  "miscellaneous_benchmark_sycl VectorSum": {
    "type": "benchmark",
    "description": "Measures performance of vector addition across 3D grid (512x256x256 elements) using SYCL.",
    "notes": null,
    "unstable": null,
    "tags": [
      "math",
      "throughput",
      "SYCL",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL VectorSum",
    "explicit_group": ""
  },
  "graph_api_benchmark_sycl FinalizeGraph rebuildGraphEveryIter:0 graphStructure:Gromacs": {
    "type": "benchmark",
    "description": "Measures the time taken to finalize a SYCL graph, using a graph structure based on the usage of graphs in Gromacs. It measures finalizing the same modifiable graph repeatedly over multiple iterations.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "finalize",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL FinalizeGraph, rebuildGraphEveryIter 0, graphStructure Gromacs",
    "explicit_group": "FinalizeGraph, GraphStructure: Gromacs"
  },
  "graph_api_benchmark_sycl FinalizeGraph rebuildGraphEveryIter:1 graphStructure:Gromacs": {
    "type": "benchmark",
    "description": "Measures the time taken to finalize a SYCL graph, using a graph structure based on the usage of graphs in Gromacs. It measures finalizing a unique modifiable graph per iteration.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "finalize",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL FinalizeGraph, rebuildGraphEveryIter 1, graphStructure Gromacs",
    "explicit_group": "FinalizeGraph, GraphStructure: Gromacs"
  },
  "graph_api_benchmark_sycl FinalizeGraph rebuildGraphEveryIter:0 graphStructure:Llama": {
    "type": "benchmark",
    "description": "Measures the time taken to finalize a SYCL graph, using a graph structure based on the usage of graphs in Llama. It measures finalizing the same modifiable graph repeatedly over multiple iterations.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "finalize",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL FinalizeGraph, rebuildGraphEveryIter 0, graphStructure Llama",
    "explicit_group": "FinalizeGraph, GraphStructure: Llama"
  },
  "graph_api_benchmark_sycl FinalizeGraph rebuildGraphEveryIter:1 graphStructure:Llama": {
    "type": "benchmark",
    "description": "Measures the time taken to finalize a SYCL graph, using a graph structure based on the usage of graphs in Llama. It measures finalizing a unique modifiable graph per iteration.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "finalize",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL FinalizeGraph, rebuildGraphEveryIter 1, graphStructure Llama",
    "explicit_group": "FinalizeGraph, GraphStructure: Llama"
  },
  "multithread_benchmark_ur MemcpyExecute opsPerThread:400, numThreads:1, allocSize:102400 srcUSM:1 dstUSM:1": {
    "type": "benchmark",
    "description": "Measures multithreaded memory copy performance with 1 threads each performing 400 operations on 102400 bytes from device to device memory with events with driver copy offload without barrier. ",
    "notes": null,
    "unstable": null,
    "tags": [
      "memory",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR MemcpyExecute, opsPerThread 400, numThreads 1, allocSize 102400, srcUSM 1, dstUSM 1",
    "explicit_group": "MemcpyExecute, opsPerThread: 400, numThreads: 1, allocSize: 102400"
  },
  "multithread_benchmark_ur MemcpyExecute opsPerThread:400, numThreads:1, allocSize:102400 srcUSM:0 dstUSM:1": {
    "type": "benchmark",
    "description": "Measures multithreaded memory copy performance with 1 threads each performing 400 operations on 102400 bytes from host to device memory with events with driver copy offload without barrier. ",
    "notes": null,
    "unstable": null,
    "tags": [
      "memory",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR MemcpyExecute, opsPerThread 400, numThreads 1, allocSize 102400, srcUSM 0, dstUSM 1",
    "explicit_group": "MemcpyExecute, opsPerThread: 400, numThreads: 1, allocSize: 102400"
  },
  "multithread_benchmark_ur MemcpyExecute opsPerThread:100, numThreads:4, allocSize:102400 srcUSM:1 dstUSM:1 without events": {
    "type": "benchmark",
    "description": "Measures multithreaded memory copy performance with 4 threads each performing 100 operations on 102400 bytes from device to device memory without events with driver copy offload without barrier. ",
    "notes": null,
    "unstable": null,
    "tags": [
      "memory",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR MemcpyExecute, opsPerThread 100, numThreads 4, allocSize 102400, srcUSM 1, dstUSM 1, without events",
    "explicit_group": "MemcpyExecute, opsPerThread: 100, numThreads: 4, allocSize: 102400"
  },
  "multithread_benchmark_ur MemcpyExecute opsPerThread:100, numThreads:4, allocSize:102400 srcUSM:1 dstUSM:1 without events without copy offload": {
    "type": "benchmark",
    "description": "Measures multithreaded memory copy performance with 4 threads each performing 100 operations on 102400 bytes from device to device memory without events without driver copy offload without barrier. ",
    "notes": null,
    "unstable": null,
    "tags": [
      "memory",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR MemcpyExecute, opsPerThread 100, numThreads 4, allocSize 102400, srcUSM 1, dstUSM 1, without events without copy offload",
    "explicit_group": "MemcpyExecute, opsPerThread: 100, numThreads: 4, allocSize: 102400"
  },
  "multithread_benchmark_ur MemcpyExecute opsPerThread:4096, numThreads:4, allocSize:1024 srcUSM:0 dstUSM:1 without events": {
    "type": "benchmark",
    "description": "Measures multithreaded memory copy performance with 4 threads each performing 4096 operations on 1024 bytes from host to device memory without events with driver copy offload without barrier. ",
    "notes": null,
    "unstable": null,
    "tags": [
      "memory",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR MemcpyExecute, opsPerThread 4096, numThreads 4, allocSize 1024, srcUSM 0, dstUSM 1, without events",
    "explicit_group": "MemcpyExecute, opsPerThread: 4096, numThreads: 4, allocSize: 1024"
  },
  "multithread_benchmark_ur MemcpyExecute opsPerThread:4096, numThreads:4, allocSize:1024 srcUSM:0 dstUSM:1 without events with barrier": {
    "type": "benchmark",
    "description": "Measures multithreaded memory copy performance with 4 threads each performing 4096 operations on 1024 bytes from host to device memory without events with driver copy offload with barrier. ",
    "notes": null,
    "unstable": null,
    "tags": [
      "memory",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR MemcpyExecute, opsPerThread 4096, numThreads 4, allocSize 1024, srcUSM 0, dstUSM 1, without events",
    "explicit_group": "MemcpyExecute, opsPerThread: 4096, numThreads: 4, allocSize: 1024"
  },
  "api_overhead_benchmark_ur UsmMemoryAllocation usmMemoryPlacement:Device size:256 measureMode:Both": {
    "type": "benchmark",
    "description": "Measures memory allocation overhead by allocating 256 bytes of usm Device memory and free'ing it immediately. Both memory allocation and memory free are timed. ",
    "notes": null,
    "unstable": null,
    "tags": [
      "UR",
      "micro",
      "latency",
      "memory"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR UsmMemoryAllocation, usmMemoryPlacement Device, size 256, measureMode Both",
    "explicit_group": "UsmMemoryAllocation"
  },
  "api_overhead_benchmark_ur UsmMemoryAllocation usmMemoryPlacement:Device size:262144 measureMode:Both": {
    "type": "benchmark",
    "description": "Measures memory allocation overhead by allocating 262144 bytes of usm Device memory and free'ing it immediately. Both memory allocation and memory free are timed. ",
    "notes": null,
    "unstable": null,
    "tags": [
      "UR",
      "micro",
      "latency",
      "memory"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR UsmMemoryAllocation, usmMemoryPlacement Device, size 262144, measureMode Both",
    "explicit_group": "UsmMemoryAllocation"
  },
  "api_overhead_benchmark_ur UsmBatchMemoryAllocation usmMemoryPlacement:Device allocationCount:128 size:256 measureMode:Both": {
    "type": "benchmark",
    "description": "Measures memory allocation overhead by allocating 256 bytes of usm Device memory 128 times, then free'ing it all at once. Both memory allocation and memory free are timed. ",
    "notes": null,
    "unstable": null,
    "tags": [
      "UR",
      "micro",
      "latency",
      "memory"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR UsmBatchMemoryAllocation, usmMemoryPlacement Device, allocationCount 128, size 256, measureMode Both",
    "explicit_group": "UsmBatchMemoryAllocation"
  },
  "api_overhead_benchmark_ur UsmBatchMemoryAllocation usmMemoryPlacement:Device allocationCount:128 size:16384 measureMode:Both": {
    "type": "benchmark",
    "description": "Measures memory allocation overhead by allocating 16384 bytes of usm Device memory 128 times, then free'ing it all at once. Both memory allocation and memory free are timed. ",
    "notes": null,
    "unstable": null,
    "tags": [
      "UR",
      "micro",
      "latency",
      "memory"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR UsmBatchMemoryAllocation, usmMemoryPlacement Device, allocationCount 128, size 16384, measureMode Both",
    "explicit_group": "UsmBatchMemoryAllocation"
  },
  "api_overhead_benchmark_ur UsmBatchMemoryAllocation usmMemoryPlacement:Device allocationCount:128 size:131072 measureMode:Both": {
    "type": "benchmark",
    "description": "Measures memory allocation overhead by allocating 131072 bytes of usm Device memory 128 times, then free'ing it all at once. Both memory allocation and memory free are timed. ",
    "notes": null,
    "unstable": null,
    "tags": [
      "UR",
      "micro",
      "latency",
      "memory"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR UsmBatchMemoryAllocation, usmMemoryPlacement Device, allocationCount 128, size 131072, measureMode Both",
    "explicit_group": "UsmBatchMemoryAllocation"
  },
  "multithread_benchmark_syclpreview MemcpyExecute opsPerThread:4096, numThreads:1, allocSize:1024 srcUSM:1 dstUSM:1 without events": {
    "type": "benchmark",
    "description": "Measures multithreaded memory copy performance with 1 threads each performing 4096 operations on 1024 bytes from device to device memory without events with driver copy offload without barrier. ",
    "notes": null,
    "unstable": null,
    "tags": [
      "memory",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR MemcpyExecute, opsPerThread 4096, numThreads 1, allocSize 1024, srcUSM 1, dstUSM 1, without events",
    "explicit_group": "MemcpyExecute, opsPerThread: 4096, numThreads: 1, allocSize: 1024"
  },
  "multithread_benchmark_syclpreview MemcpyExecute opsPerThread:4096, numThreads:1, allocSize:1024 srcUSM:1 dstUSM:1 without events with barrier": {
    "type": "benchmark",
    "description": "Measures multithreaded memory copy performance with 1 threads each performing 4096 operations on 1024 bytes from device to device memory without events with driver copy offload with barrier. ",
    "notes": null,
    "unstable": null,
    "tags": [
      "memory",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR MemcpyExecute, opsPerThread 4096, numThreads 1, allocSize 1024, srcUSM 1, dstUSM 1, without events",
    "explicit_group": "MemcpyExecute, opsPerThread: 4096, numThreads: 1, allocSize: 1024"
  },
  "multithread_benchmark_syclpreview MemcpyExecute opsPerThread:4096, numThreads:4, allocSize:1024 srcUSM:1 dstUSM:1 without events": {
    "type": "benchmark",
    "description": "Measures multithreaded memory copy performance with 4 threads each performing 4096 operations on 1024 bytes from device to device memory without events with driver copy offload without barrier. ",
    "notes": null,
    "unstable": null,
    "tags": [
      "memory",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR MemcpyExecute, opsPerThread 4096, numThreads 4, allocSize 1024, srcUSM 1, dstUSM 1, without events",
    "explicit_group": "MemcpyExecute, opsPerThread: 4096, numThreads: 4, allocSize: 1024"
  },
  "multithread_benchmark_syclpreview MemcpyExecute opsPerThread:4096, numThreads:4, allocSize:1024 srcUSM:1 dstUSM:1 without events with barrier": {
    "type": "benchmark",
    "description": "Measures multithreaded memory copy performance with 4 threads each performing 4096 operations on 1024 bytes from device to device memory without events with driver copy offload with barrier. ",
    "notes": null,
    "unstable": null,
    "tags": [
      "memory",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR MemcpyExecute, opsPerThread 4096, numThreads 4, allocSize 1024, srcUSM 1, dstUSM 1, without events",
    "explicit_group": "MemcpyExecute, opsPerThread: 4096, numThreads: 4, allocSize: 1024"
  },
  "Velocity-Bench Hashtable": {
    "type": "benchmark",
    "description": "Measures hash table search performance using an efficient lock-free algorithm with linear probing. Reports throughput in millions of keys processed per second. Higher values indicate better performance.",
    "notes": null,
    "unstable": null,
    "tags": [
      "SYCL",
      "application",
      "throughput"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "Velocity-Bench Hashtable",
    "explicit_group": ""
  },
  "Velocity-Bench Bitcracker": {
    "type": "benchmark",
    "description": "Password-cracking application for BitLocker-encrypted memory units. Uses dictionary attack to find user or recovery passwords. Measures total time required to process 60000 passwords.",
    "notes": null,
    "unstable": null,
    "tags": [
      "SYCL",
      "application",
      "throughput"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "Velocity-Bench Bitcracker",
    "explicit_group": ""
  },
  "Velocity-Bench CudaSift": {
    "type": "benchmark",
    "description": "Implementation of the SIFT (Scale Invariant Feature Transform) algorithm for detecting, describing, and matching local features in images. Measures average processing time in milliseconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "SYCL",
      "application",
      "image"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "Velocity-Bench CudaSift",
    "explicit_group": ""
  },
  "Velocity-Bench Easywave": {
    "type": "benchmark",
    "description": "A tsunami wave simulator used for researching tsunami generation and wave propagation. Measures the elapsed time in milliseconds to simulate a specified tsunami event based on real-world data.",
    "notes": null,
    "unstable": null,
    "tags": [
      "SYCL",
      "application",
      "simulation"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "Velocity-Bench Easywave",
    "explicit_group": ""
  },
  "Velocity-Bench QuickSilver": {
    "type": "benchmark",
    "description": "Solves a simplified dynamic Monte Carlo particle-transport problem used in HPC. Replicates memory access patterns, communication patterns, and branching of Mercury workloads. Reports a figure of merit in MMS/CTT where higher values indicate better performance.",
    "notes": null,
    "unstable": null,
    "tags": [
      "SYCL",
      "application",
      "simulation",
      "throughput"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "Velocity-Bench QuickSilver",
    "explicit_group": ""
  },
  "Velocity-Bench Sobel Filter": {
    "type": "benchmark",
    "description": "Popular RGB-to-grayscale image conversion technique that applies a gaussian filter to reduce edge artifacts. Processes a large 32K x 32K image and measures the time required to apply the filter.",
    "notes": null,
    "unstable": null,
    "tags": [
      "SYCL",
      "application",
      "image",
      "throughput"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "Velocity-Bench Sobel Filter",
    "explicit_group": ""
  },
  "Velocity-Bench dl-cifar": {
    "type": "benchmark",
    "description": "Deep learning image classification workload based on the CIFAR-10 dataset of 60,000 32x32 color images in 10 classes. Uses neural networks to classify input images and measures total calculation time.",
    "notes": null,
    "unstable": null,
    "tags": [
      "SYCL",
      "application",
      "inference",
      "image"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "Velocity-Bench dl-cifar",
    "explicit_group": ""
  },
  "Velocity-Bench dl-mnist": {
    "type": "benchmark",
    "description": "Digit recognition based on the MNIST database, one of the oldest and most popular databases of handwritten digits. Uses neural networks to identify digits and measures total calculation time.",
    "notes": null,
    "unstable": null,
    "tags": [
      "SYCL",
      "application",
      "inference",
      "image"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "Velocity-Bench dl-mnist",
    "explicit_group": ""
  },
  "Velocity-Bench svm": {
    "type": "benchmark",
    "description": "Implementation of Support Vector Machine, a popular classical machine learning technique. Uses supervised learning models with associated algorithms to analyze data for classification and regression analysis. Measures total elapsed time.",
    "notes": null,
    "unstable": null,
    "tags": [
      "SYCL",
      "application",
      "inference"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "Velocity-Bench svm",
    "explicit_group": ""
  },
  "SYCL-Bench IndependentDAGTaskThroughput_multi": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "SYCL",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL-Bench IndependentDAGTaskThroughput_multi",
    "explicit_group": ""
  },
  "SYCL-Bench DAGTaskThroughput_multi": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "SYCL",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL-Bench DAGTaskThroughput_multi",
    "explicit_group": ""
  },
  "SYCL-Bench HostDeviceBandwidth_multi": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "SYCL",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL-Bench HostDeviceBandwidth_multi",
    "explicit_group": ""
  },
  "SYCL-Bench LocalMem_multi": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "SYCL",
      "micro",
      "memory"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL-Bench LocalMem_multi",
    "explicit_group": ""
  },
  "SYCL-Bench ScalarProduct_multi": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "SYCL",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL-Bench ScalarProduct_multi",
    "explicit_group": ""
  },
  "SYCL-Bench Pattern_SegmentedReduction_multi": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "SYCL",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL-Bench Pattern_SegmentedReduction_multi",
    "explicit_group": ""
  },
  "SYCL-Bench USM_Allocation_latency_multi": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "SYCL",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL-Bench USM_Allocation_latency_multi",
    "explicit_group": ""
  },
  "SYCL-Bench VectorAddition_multi": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "SYCL",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL-Bench VectorAddition_multi",
    "explicit_group": ""
  },
  "SYCL-Bench 2mm": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "SYCL",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL-Bench 2mm",
    "explicit_group": ""
  },
  "SYCL-Bench 3mm": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "SYCL",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL-Bench 3mm",
    "explicit_group": ""
  },
  "SYCL-Bench Atax": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "SYCL",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL-Bench Atax",
    "explicit_group": ""
  },
  "SYCL-Bench Bicg": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "SYCL",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL-Bench Bicg",
    "explicit_group": ""
  },
  "SYCL-Bench Kmeans": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "SYCL",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL-Bench Kmeans",
    "explicit_group": ""
  },
  "SYCL-Bench LinearRegressionCoeff": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "SYCL",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL-Bench LinearRegressionCoeff",
    "explicit_group": ""
  },
  "SYCL-Bench MolecularDynamics": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "SYCL",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL-Bench MolecularDynamics",
    "explicit_group": ""
  },
  "SYCL-Bench sf_16": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "SYCL",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL-Bench sf_16",
    "explicit_group": ""
  },
  "llama.cpp DeepSeek-R1-Distill-Qwen-1.5B-Q4_0.gguf": {
    "type": "benchmark",
    "description": "Performance testing tool for llama.cpp that measures LLM inference speed in tokens per second. Runs both prompt processing (initial context processing) and text generation benchmarks with different batch sizes. Higher values indicate better performance. Uses the DeepSeek-R1-Distill-Qwen-1.5B-Q4_0.gguf quantized model and leverages SYCL with oneDNN for acceleration.",
    "notes": null,
    "unstable": null,
    "tags": [
      "SYCL",
      "application",
      "inference",
      "throughput"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "llama.cpp DeepSeek-R1-Distill-Qwen-1.5B-Q4_0.gguf",
    "explicit_group": ""
  },
  "umf-benchmark": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "umf-benchmark",
    "explicit_group": ""
  },
  "gromacs-0006-pme-graphs": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "gromacs-0006-pme-graphs",
    "explicit_group": ""
  },
  "gromacs-0006-pme-eager": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "gromacs-0006-pme-eager",
    "explicit_group": ""
  },
  "gromacs-0006-rf-graphs": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "gromacs-0006-rf-graphs",
    "explicit_group": ""
  },
  "gromacs-0006-rf-eager": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "gromacs-0006-rf-eager",
    "explicit_group": ""
  },
  "onednn-sum-f32-2-eager": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "onednn-sum-f32-2-eager",
    "explicit_group": "sum-f32-2"
  },
  "onednn-graph-sdpa-plain-f16-eager": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "onednn-graph-sdpa-plain-f16-eager",
    "explicit_group": "graph-sdpa-plain-f16"
  },
  "onednn-graph-sdpa-plain-f32-eager": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "onednn-graph-sdpa-plain-f32-eager",
    "explicit_group": "graph-sdpa-plain-f32"
  },
  "onednn-graph-sdpa-plain-f32-graph": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "onednn-graph-sdpa-plain-f32-graph",
    "explicit_group": "graph-sdpa-plain-f32"
  },
  "Foo Group": {
    "type": "group",
    "description": "This is a test benchmark for Foo Group.",
    "notes": "This is a test note for Foo Group.\nLook, multiple lines!",
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": null
  },
  "Bar Group": {
    "type": "group",
    "description": "This is a test benchmark for Bar Group.",
    "notes": null,
    "unstable": "This is an unstable note for Bar Group.",
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": null
  },
  "Memory Bandwidth 1": {
    "type": "benchmark",
    "description": "This is a test benchmark for Memory Bandwidth 1.",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "Memory Bandwidth 1",
    "explicit_group": ""
  },
  "Memory Bandwidth 2": {
    "type": "benchmark",
    "description": "This is a test benchmark for Memory Bandwidth 2.",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "Memory Bandwidth 2",
    "explicit_group": ""
  },
  "Memory Bandwidth 3": {
    "type": "benchmark",
    "description": "This is a test benchmark for Memory Bandwidth 3.",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "Memory Bandwidth 3",
    "explicit_group": ""
  },
  "Memory Bandwidth 4": {
    "type": "benchmark",
    "description": "This is a test benchmark for Memory Bandwidth 4.",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "Memory Bandwidth 4",
    "explicit_group": ""
  },
  "Memory Bandwidth 5": {
    "type": "benchmark",
    "description": "This is a test benchmark for Memory Bandwidth 5.",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "Memory Bandwidth 5",
    "explicit_group": ""
  },
  "Memory Bandwidth 6": {
    "type": "benchmark",
    "description": "This is a test benchmark for Memory Bandwidth 6.",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "Memory Bandwidth 6",
    "explicit_group": ""
  },
  "Latency 1": {
    "type": "benchmark",
    "description": "This is a test benchmark for Latency 1.",
    "notes": "A Latency test note!",
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "Latency 1",
    "explicit_group": ""
  },
  "Latency 2": {
    "type": "benchmark",
    "description": "This is a test benchmark for Latency 2.",
    "notes": "A Latency test note!",
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "Latency 2",
    "explicit_group": ""
  },
  "Latency 3": {
    "type": "benchmark",
    "description": "This is a test benchmark for Latency 3.",
    "notes": "A Latency test note!",
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "Latency 3",
    "explicit_group": ""
  },
  "Latency 4": {
    "type": "benchmark",
    "description": "This is a test benchmark for Latency 4.",
    "notes": "A Latency test note!",
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "Latency 4",
    "explicit_group": ""
  },
  "Latency 5": {
    "type": "benchmark",
    "description": "This is a test benchmark for Latency 5.",
    "notes": "A Latency test note!",
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "Latency 5",
    "explicit_group": ""
  },
  "Latency 6": {
    "type": "benchmark",
    "description": "This is a test benchmark for Latency 6.",
    "notes": "A Latency test note!",
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "Latency 6",
    "explicit_group": ""
  },
  "Throughput 1": {
    "type": "benchmark",
    "description": "This is a test benchmark for Throughput 1.",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "Throughput 1",
    "explicit_group": ""
  },
  "Throughput 2": {
    "type": "benchmark",
    "description": "This is a test benchmark for Throughput 2.",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "Throughput 2",
    "explicit_group": ""
  },
  "Throughput 3": {
    "type": "benchmark",
    "description": "This is a test benchmark for Throughput 3.",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "Throughput 3",
    "explicit_group": ""
  },
  "Throughput 4": {
    "type": "benchmark",
    "description": "This is a test benchmark for Throughput 4.",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "Throughput 4",
    "explicit_group": ""
  },
  "Throughput 5": {
    "type": "benchmark",
    "description": "This is a test benchmark for Throughput 5.",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "Throughput 5",
    "explicit_group": ""
  },
  "Throughput 6": {
    "type": "benchmark",
    "description": "This is a test benchmark for Throughput 6.",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "Throughput 6",
    "explicit_group": ""
  },
  "FLOPS 1": {
    "type": "benchmark",
    "description": "This is a test benchmark for FLOPS 1.",
    "notes": null,
    "unstable": "Unstable FLOPS test!",
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "FLOPS 1",
    "explicit_group": ""
  },
  "FLOPS 2": {
    "type": "benchmark",
    "description": "This is a test benchmark for FLOPS 2.",
    "notes": null,
    "unstable": "Unstable FLOPS test!",
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "FLOPS 2",
    "explicit_group": ""
  },
  "FLOPS 3": {
    "type": "benchmark",
    "description": "This is a test benchmark for FLOPS 3.",
    "notes": null,
    "unstable": "Unstable FLOPS test!",
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "FLOPS 3",
    "explicit_group": ""
  },
  "FLOPS 4": {
    "type": "benchmark",
    "description": "This is a test benchmark for FLOPS 4.",
    "notes": null,
    "unstable": "Unstable FLOPS test!",
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "FLOPS 4",
    "explicit_group": ""
  },
  "FLOPS 5": {
    "type": "benchmark",
    "description": "This is a test benchmark for FLOPS 5.",
    "notes": null,
    "unstable": "Unstable FLOPS test!",
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "FLOPS 5",
    "explicit_group": ""
  },
  "FLOPS 6": {
    "type": "benchmark",
    "description": "This is a test benchmark for FLOPS 6.",
    "notes": null,
    "unstable": "Unstable FLOPS test!",
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "FLOPS 6",
    "explicit_group": ""
  },
  "Cache Miss Rate 1": {
    "type": "benchmark",
    "description": "This is a test benchmark for Cache Miss Rate 1.",
    "notes": "Test Note",
    "unstable": "And another note!",
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "Cache Miss Rate 1",
    "explicit_group": ""
  },
  "Cache Miss Rate 2": {
    "type": "benchmark",
    "description": "This is a test benchmark for Cache Miss Rate 2.",
    "notes": "Test Note",
    "unstable": "And another note!",
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "Cache Miss Rate 2",
    "explicit_group": ""
  },
  "Cache Miss Rate 3": {
    "type": "benchmark",
    "description": "This is a test benchmark for Cache Miss Rate 3.",
    "notes": "Test Note",
    "unstable": "And another note!",
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "Cache Miss Rate 3",
    "explicit_group": ""
  },
  "Cache Miss Rate 4": {
    "type": "benchmark",
    "description": "This is a test benchmark for Cache Miss Rate 4.",
    "notes": "Test Note",
    "unstable": "And another note!",
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "Cache Miss Rate 4",
    "explicit_group": ""
  },
  "Cache Miss Rate 5": {
    "type": "benchmark",
    "description": "This is a test benchmark for Cache Miss Rate 5.",
    "notes": "Test Note",
    "unstable": "And another note!",
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "Cache Miss Rate 5",
    "explicit_group": ""
  },
  "Cache Miss Rate 6": {
    "type": "benchmark",
    "description": "This is a test benchmark for Cache Miss Rate 6.",
    "notes": "Test Note",
    "unstable": "And another note!",
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "Cache Miss Rate 6",
    "explicit_group": ""
  }
};

benchmarkTags = {
  "SYCL": {
    "name": "SYCL",
    "description": "Benchmark uses SYCL runtime"
  },
  "UR": {
    "name": "UR",
    "description": "Benchmark uses Unified Runtime API"
  },
  "L0": {
    "name": "L0",
    "description": "Benchmark uses Level Zero API directly"
  },
  "UMF": {
    "name": "UMF",
    "description": "Benchmark uses Unified Memory Framework directly"
  },
  "micro": {
    "name": "micro",
    "description": "Microbenchmark focusing on a specific functionality"
  },
  "application": {
    "name": "application",
    "description": "Real application-based performance test"
  },
  "proxy": {
    "name": "proxy",
    "description": "Benchmark that simulates real application use-cases"
  },
  "submit": {
    "name": "submit",
    "description": "Tests kernel submission performance"
  },
  "math": {
    "name": "math",
    "description": "Tests math computation performance"
  },
  "memory": {
    "name": "memory",
    "description": "Tests memory transfer or bandwidth performance"
  },
  "allocation": {
    "name": "allocation",
    "description": "Tests memory allocation performance"
  },
  "graph": {
    "name": "graph",
    "description": "Tests graph-based execution performance"
  },
  "latency": {
    "name": "latency",
    "description": "Measures operation latency"
  },
  "throughput": {
    "name": "throughput",
    "description": "Measures operation throughput"
  },
  "inference": {
    "name": "inference",
    "description": "Tests ML/AI inference performance"
  },
  "image": {
    "name": "image",
    "description": "Image processing benchmark"
  },
  "simulation": {
    "name": "simulation",
    "description": "Physics or scientific simulation benchmark"
  }
};

defaultCompareNames = [
  "OneDNN_V1"
];

flamegraphData = {
  "runs": {
    "OneDNN_V1": {
      "suites": {
        "onednn-graph-sdpa-plain-f16-eager": "BenchDNN",
        "onednn-graph-sdpa-plain-f32-graph": "BenchDNN",
        "onednn-graph-sdpa-plain-f32-eager": "BenchDNN",
        "onednn-sum-f32-2-eager": "BenchDNN"
      },
      "timestamp": "20250826_153500"
    }
  },
  "last_updated": "2025-08-26T15:37:28.306683"
};

