benchmarkRuns = [
  {
    "results": [
      {
        "label": "api_overhead_benchmark_l0 SubmitKernel in order not using events",
        "value": 11.027,
        "command": [
          "/home/jan_trusillo/workspace/benchmarks_workdir/compute-benchmarks-build/bin/api_overhead_benchmark_l0",
          "--test=SubmitKernel",
          "--csv",
          "--noHeaders",
          "--iterations=100000",
          "--Ioq=1",
          "--MeasureCompletion=0",
          "--Profiling=0",
          "--NumKernels=10",
          "--KernelExecTime=1",
          "--UseEvents=0",
          "--profilerType=timer"
        ],
        "env": {},
        "unit": "\u03bcs",
        "stddev": 1.1461001999999998,
        "git_url": "https://github.com/intel/compute-benchmarks.git",
        "git_hash": "2f1c59bd731477de9b99b95a37bad5ebc9dae922",
        "lower_is_better": true,
        "suite": "Compute Benchmarks"
      },
      {
        "label": "api_overhead_benchmark_l0 SubmitKernel in order",
        "value": 12.204,
        "command": [
          "/home/jan_trusillo/workspace/benchmarks_workdir/compute-benchmarks-build/bin/api_overhead_benchmark_l0",
          "--test=SubmitKernel",
          "--csv",
          "--noHeaders",
          "--iterations=100000",
          "--Ioq=1",
          "--MeasureCompletion=0",
          "--Profiling=0",
          "--NumKernels=10",
          "--KernelExecTime=1",
          "--UseEvents=1",
          "--profilerType=timer"
        ],
        "env": {},
        "unit": "\u03bcs",
        "stddev": 0.9900187,
        "git_url": "https://github.com/intel/compute-benchmarks.git",
        "git_hash": "2f1c59bd731477de9b99b95a37bad5ebc9dae922",
        "lower_is_better": true,
        "suite": "Compute Benchmarks"
      },
      {
        "label": "api_overhead_benchmark_l0 SubmitKernel in order with measure completion not using events",
        "value": 13.725,
        "command": [
          "/home/jan_trusillo/workspace/benchmarks_workdir/compute-benchmarks-build/bin/api_overhead_benchmark_l0",
          "--test=SubmitKernel",
          "--csv",
          "--noHeaders",
          "--iterations=100000",
          "--Ioq=1",
          "--MeasureCompletion=1",
          "--Profiling=0",
          "--NumKernels=10",
          "--KernelExecTime=1",
          "--UseEvents=0",
          "--profilerType=timer"
        ],
        "env": {},
        "unit": "\u03bcs",
        "stddev": 2.6116675000000003,
        "git_url": "https://github.com/intel/compute-benchmarks.git",
        "git_hash": "2f1c59bd731477de9b99b95a37bad5ebc9dae922",
        "lower_is_better": true,
        "suite": "Compute Benchmarks"
      },
      {
        "label": "api_overhead_benchmark_l0 SubmitKernel in order with measure completion",
        "value": 14.847,
        "command": [
          "/home/jan_trusillo/workspace/benchmarks_workdir/compute-benchmarks-build/bin/api_overhead_benchmark_l0",
          "--test=SubmitKernel",
          "--csv",
          "--noHeaders",
          "--iterations=100000",
          "--Ioq=1",
          "--MeasureCompletion=1",
          "--Profiling=0",
          "--NumKernels=10",
          "--KernelExecTime=1",
          "--UseEvents=1",
          "--profilerType=timer"
        ],
        "env": {},
        "unit": "\u03bcs",
        "stddev": 1.1346502,
        "git_url": "https://github.com/intel/compute-benchmarks.git",
        "git_hash": "2f1c59bd731477de9b99b95a37bad5ebc9dae922",
        "lower_is_better": true,
        "suite": "Compute Benchmarks"
      }
    ],
    "name": "offload_ApiOverhead_l0",
    "hostname": "gklabdsemom53.dss.lab",
    "git_hash": "5ab357387291",
    "github_repo": "311Volt/llvm",
    "date": "2026-08-03T10:34:56.377714+00:00",
    "compute_runtime": "unknown",
    "platform": {
      "os": "Linux 6.19.0-rc6-v6.19-rc6 #1 SMP PREEMPT_DYNAMIC Thu Jan 22 12:50:16 UTC 2026",
      "python": "CPython 3.13.7",
      "cpu_count": 128,
      "cpu_info": "Intel(R) Xeon(R) 696X",
      "gpu_info": [
        "Intel Corporation Battlemage G21 [Arc B580]",
        "Intel Corporation Device 7f2f (rev 10)",
        "Intel Corporation Battlemage G21 [Arc B580]",
        "Intel Corporation DG2 [Arc A380] (rev 05)"
      ],
      "gpu_driver_version": "xe (kernel 6.19.0-rc6-v6.19-rc6)",
      "gcc_version": "gcc (Ubuntu 15.2.0-4ubuntu4) 15.2.0",
      "clang_version": "DPC++ compiler 7.1.0 (pre-release) build based on: clang version 23.0.0git (git@github.com:311Volt/llvm.git 38239b406b7af44e4d8093d9f95b763c5b10a34e)",
      "level_zero_version": "L0 v2 adapter | level-zero (version unknown)",
      "compute_runtime_version": ""
    }
  },
  {
    "results": [
      {
        "label": "api_overhead_benchmark_ol SubmitKernel in order not using events",
        "value": 13.145,
        "command": [
          "/home/jan_trusillo/workspace/benchmarks_workdir/compute-benchmarks-build/bin/api_overhead_benchmark_ol",
          "--test=SubmitKernel",
          "--csv",
          "--noHeaders",
          "--iterations=100000",
          "--Ioq=1",
          "--MeasureCompletion=0",
          "--Profiling=0",
          "--NumKernels=10",
          "--KernelExecTime=1",
          "--UseEvents=0",
          "--profilerType=timer"
        ],
        "env": {
          "FORCE_OFFLOAD_PLUGIN": "level_zero"
        },
        "unit": "\u03bcs",
        "stddev": 2.6142413,
        "git_url": "https://github.com/intel/compute-benchmarks.git",
        "git_hash": "2f1c59bd731477de9b99b95a37bad5ebc9dae922",
        "lower_is_better": true,
        "suite": "Compute Benchmarks"
      },
      {
        "label": "api_overhead_benchmark_ol SubmitKernel in order",
        "value": 13.789,
        "command": [
          "/home/jan_trusillo/workspace/benchmarks_workdir/compute-benchmarks-build/bin/api_overhead_benchmark_ol",
          "--test=SubmitKernel",
          "--csv",
          "--noHeaders",
          "--iterations=100000",
          "--Ioq=1",
          "--MeasureCompletion=0",
          "--Profiling=0",
          "--NumKernels=10",
          "--KernelExecTime=1",
          "--UseEvents=1",
          "--profilerType=timer"
        ],
        "env": {
          "FORCE_OFFLOAD_PLUGIN": "level_zero"
        },
        "unit": "\u03bcs",
        "stddev": 0.8172611999999999,
        "git_url": "https://github.com/intel/compute-benchmarks.git",
        "git_hash": "2f1c59bd731477de9b99b95a37bad5ebc9dae922",
        "lower_is_better": true,
        "suite": "Compute Benchmarks"
      },
      {
        "label": "api_overhead_benchmark_ol SubmitKernel in order with measure completion not using events",
        "value": 15.824,
        "command": [
          "/home/jan_trusillo/workspace/benchmarks_workdir/compute-benchmarks-build/bin/api_overhead_benchmark_ol",
          "--test=SubmitKernel",
          "--csv",
          "--noHeaders",
          "--iterations=100000",
          "--Ioq=1",
          "--MeasureCompletion=1",
          "--Profiling=0",
          "--NumKernels=10",
          "--KernelExecTime=1",
          "--UseEvents=0",
          "--profilerType=timer"
        ],
        "env": {
          "FORCE_OFFLOAD_PLUGIN": "level_zero"
        },
        "unit": "\u03bcs",
        "stddev": 0.9299032,
        "git_url": "https://github.com/intel/compute-benchmarks.git",
        "git_hash": "2f1c59bd731477de9b99b95a37bad5ebc9dae922",
        "lower_is_better": true,
        "suite": "Compute Benchmarks"
      },
      {
        "label": "api_overhead_benchmark_ol SubmitKernel in order with measure completion",
        "value": 16.467,
        "command": [
          "/home/jan_trusillo/workspace/benchmarks_workdir/compute-benchmarks-build/bin/api_overhead_benchmark_ol",
          "--test=SubmitKernel",
          "--csv",
          "--noHeaders",
          "--iterations=100000",
          "--Ioq=1",
          "--MeasureCompletion=1",
          "--Profiling=0",
          "--NumKernels=10",
          "--KernelExecTime=1",
          "--UseEvents=1",
          "--profilerType=timer"
        ],
        "env": {
          "FORCE_OFFLOAD_PLUGIN": "level_zero"
        },
        "unit": "\u03bcs",
        "stddev": 2.8444989999999994,
        "git_url": "https://github.com/intel/compute-benchmarks.git",
        "git_hash": "2f1c59bd731477de9b99b95a37bad5ebc9dae922",
        "lower_is_better": true,
        "suite": "Compute Benchmarks"
      }
    ],
    "name": "offload_ApiOverhead_ol",
    "hostname": "gklabdsemom53.dss.lab",
    "git_hash": "5ab357387291",
    "github_repo": "311Volt/llvm",
    "date": "2026-08-03T10:34:49.108423+00:00",
    "compute_runtime": "unknown",
    "platform": {
      "os": "Linux 6.19.0-rc6-v6.19-rc6 #1 SMP PREEMPT_DYNAMIC Thu Jan 22 12:50:16 UTC 2026",
      "python": "CPython 3.13.7",
      "cpu_count": 128,
      "cpu_info": "Intel(R) Xeon(R) 696X",
      "gpu_info": [
        "Intel Corporation Battlemage G21 [Arc B580]",
        "Intel Corporation Device 7f2f (rev 10)",
        "Intel Corporation Battlemage G21 [Arc B580]",
        "Intel Corporation DG2 [Arc A380] (rev 05)"
      ],
      "gpu_driver_version": "xe (kernel 6.19.0-rc6-v6.19-rc6)",
      "gcc_version": "gcc (Ubuntu 15.2.0-4ubuntu4) 15.2.0",
      "clang_version": "DPC++ compiler 7.1.0 (pre-release) build based on: clang version 23.0.0git (git@github.com:311Volt/llvm.git 38239b406b7af44e4d8093d9f95b763c5b10a34e)",
      "level_zero_version": "L0 v2 adapter | level-zero (version unknown)",
      "compute_runtime_version": ""
    }
  },
  {
    "results": [
      {
        "label": "api_overhead_benchmark_ur SubmitKernel in order not using events",
        "value": 22.819,
        "command": [
          "/home/jan_trusillo/workspace/benchmarks_workdir/compute-benchmarks-build/bin/api_overhead_benchmark_ur",
          "--test=SubmitKernel",
          "--csv",
          "--noHeaders",
          "--iterations=100000",
          "--Ioq=1",
          "--MeasureCompletion=0",
          "--Profiling=0",
          "--NumKernels=10",
          "--KernelExecTime=1",
          "--UseEvents=0",
          "--profilerType=timer"
        ],
        "env": {},
        "unit": "\u03bcs",
        "stddev": 5.427402,
        "git_url": "https://github.com/intel/compute-benchmarks.git",
        "git_hash": "2f1c59bd731477de9b99b95a37bad5ebc9dae922",
        "lower_is_better": true,
        "suite": "Compute Benchmarks"
      },
      {
        "label": "api_overhead_benchmark_ur SubmitKernel in order",
        "value": 23.445,
        "command": [
          "/home/jan_trusillo/workspace/benchmarks_workdir/compute-benchmarks-build/bin/api_overhead_benchmark_ur",
          "--test=SubmitKernel",
          "--csv",
          "--noHeaders",
          "--iterations=100000",
          "--Ioq=1",
          "--MeasureCompletion=0",
          "--Profiling=0",
          "--NumKernels=10",
          "--KernelExecTime=1",
          "--UseEvents=1",
          "--profilerType=timer"
        ],
        "env": {},
        "unit": "\u03bcs",
        "stddev": 2.1549663,
        "git_url": "https://github.com/intel/compute-benchmarks.git",
        "git_hash": "2f1c59bd731477de9b99b95a37bad5ebc9dae922",
        "lower_is_better": true,
        "suite": "Compute Benchmarks"
      },
      {
        "label": "api_overhead_benchmark_ur SubmitKernel in order with measure completion not using events",
        "value": 25.882,
        "command": [
          "/home/jan_trusillo/workspace/benchmarks_workdir/compute-benchmarks-build/bin/api_overhead_benchmark_ur",
          "--test=SubmitKernel",
          "--csv",
          "--noHeaders",
          "--iterations=100000",
          "--Ioq=1",
          "--MeasureCompletion=1",
          "--Profiling=0",
          "--NumKernels=10",
          "--KernelExecTime=1",
          "--UseEvents=0",
          "--profilerType=timer"
        ],
        "env": {},
        "unit": "\u03bcs",
        "stddev": 3.737296,
        "git_url": "https://github.com/intel/compute-benchmarks.git",
        "git_hash": "2f1c59bd731477de9b99b95a37bad5ebc9dae922",
        "lower_is_better": true,
        "suite": "Compute Benchmarks"
      },
      {
        "label": "api_overhead_benchmark_ur SubmitKernel in order with measure completion",
        "value": 16.236,
        "command": [
          "/home/jan_trusillo/workspace/benchmarks_workdir/compute-benchmarks-build/bin/api_overhead_benchmark_ur",
          "--test=SubmitKernel",
          "--csv",
          "--noHeaders",
          "--iterations=100000",
          "--Ioq=1",
          "--MeasureCompletion=1",
          "--Profiling=0",
          "--NumKernels=10",
          "--KernelExecTime=1",
          "--UseEvents=1",
          "--profilerType=timer"
        ],
        "env": {},
        "unit": "\u03bcs",
        "stddev": 1.12832,
        "git_url": "https://github.com/intel/compute-benchmarks.git",
        "git_hash": "2f1c59bd731477de9b99b95a37bad5ebc9dae922",
        "lower_is_better": true,
        "suite": "Compute Benchmarks"
      },
      {
        "label": "api_overhead_benchmark_ur UsmMemoryAllocation usmMemoryPlacement:Device size:256 measureMode:Both",
        "value": 0.264,
        "command": [
          "/home/jan_trusillo/workspace/benchmarks_workdir/compute-benchmarks-build/bin/api_overhead_benchmark_ur",
          "--test=UsmMemoryAllocation",
          "--csv",
          "--noHeaders",
          "--iterations=10000",
          "--type=Device",
          "--size=256",
          "--measureMode=Both"
        ],
        "env": {},
        "unit": "\u03bcs",
        "stddev": 0.117406,
        "git_url": "https://github.com/intel/compute-benchmarks.git",
        "git_hash": "2f1c59bd731477de9b99b95a37bad5ebc9dae922",
        "lower_is_better": true,
        "suite": "Compute Benchmarks"
      },
      {
        "label": "api_overhead_benchmark_ur UsmMemoryAllocation usmMemoryPlacement:Device size:262144 measureMode:Both",
        "value": 0.258,
        "command": [
          "/home/jan_trusillo/workspace/benchmarks_workdir/compute-benchmarks-build/bin/api_overhead_benchmark_ur",
          "--test=UsmMemoryAllocation",
          "--csv",
          "--noHeaders",
          "--iterations=10000",
          "--type=Device",
          "--size=262144",
          "--measureMode=Both"
        ],
        "env": {},
        "unit": "\u03bcs",
        "stddev": 0.16313440000000004,
        "git_url": "https://github.com/intel/compute-benchmarks.git",
        "git_hash": "2f1c59bd731477de9b99b95a37bad5ebc9dae922",
        "lower_is_better": true,
        "suite": "Compute Benchmarks"
      },
      {
        "label": "api_overhead_benchmark_ur UsmBatchMemoryAllocation usmMemoryPlacement:Device allocationCount:128 size:256 measureMode:Both",
        "value": 16.176,
        "command": [
          "/home/jan_trusillo/workspace/benchmarks_workdir/compute-benchmarks-build/bin/api_overhead_benchmark_ur",
          "--test=UsmBatchMemoryAllocation",
          "--csv",
          "--noHeaders",
          "--iterations=1000",
          "--type=Device",
          "--allocationCount=128",
          "--size=256",
          "--measureMode=Both"
        ],
        "env": {},
        "unit": "\u03bcs",
        "stddev": 13.243428,
        "git_url": "https://github.com/intel/compute-benchmarks.git",
        "git_hash": "2f1c59bd731477de9b99b95a37bad5ebc9dae922",
        "lower_is_better": true,
        "suite": "Compute Benchmarks"
      },
      {
        "label": "api_overhead_benchmark_ur UsmBatchMemoryAllocation usmMemoryPlacement:Device allocationCount:128 size:16384 measureMode:Both",
        "value": 10396.854,
        "command": [
          "/home/jan_trusillo/workspace/benchmarks_workdir/compute-benchmarks-build/bin/api_overhead_benchmark_ur",
          "--test=UsmBatchMemoryAllocation",
          "--csv",
          "--noHeaders",
          "--iterations=1000",
          "--type=Device",
          "--allocationCount=128",
          "--size=16384",
          "--measureMode=Both"
        ],
        "env": {},
        "unit": "\u03bcs",
        "stddev": 2375.723259,
        "git_url": "https://github.com/intel/compute-benchmarks.git",
        "git_hash": "2f1c59bd731477de9b99b95a37bad5ebc9dae922",
        "lower_is_better": true,
        "suite": "Compute Benchmarks"
      },
      {
        "label": "api_overhead_benchmark_ur UsmBatchMemoryAllocation usmMemoryPlacement:Device allocationCount:128 size:131072 measureMode:Both",
        "value": 47744.542,
        "command": [
          "/home/jan_trusillo/workspace/benchmarks_workdir/compute-benchmarks-build/bin/api_overhead_benchmark_ur",
          "--test=UsmBatchMemoryAllocation",
          "--csv",
          "--noHeaders",
          "--iterations=1000",
          "--type=Device",
          "--allocationCount=128",
          "--size=131072",
          "--measureMode=Both"
        ],
        "env": {},
        "unit": "\u03bcs",
        "stddev": 9355.5424656,
        "git_url": "https://github.com/intel/compute-benchmarks.git",
        "git_hash": "2f1c59bd731477de9b99b95a37bad5ebc9dae922",
        "lower_is_better": true,
        "suite": "Compute Benchmarks"
      }
    ],
    "name": "offload_ApiOverhead_ur",
    "hostname": "gklabdsemom53.dss.lab",
    "git_hash": "5ab357387291",
    "github_repo": "311Volt/llvm",
    "date": "2026-08-03T10:34:41.136372+00:00",
    "compute_runtime": "unknown",
    "platform": {
      "os": "Linux 6.19.0-rc6-v6.19-rc6 #1 SMP PREEMPT_DYNAMIC Thu Jan 22 12:50:16 UTC 2026",
      "python": "CPython 3.13.7",
      "cpu_count": 128,
      "cpu_info": "Intel(R) Xeon(R) 696X",
      "gpu_info": [
        "Intel Corporation Battlemage G21 [Arc B580]",
        "Intel Corporation Device 7f2f (rev 10)",
        "Intel Corporation Battlemage G21 [Arc B580]",
        "Intel Corporation DG2 [Arc A380] (rev 05)"
      ],
      "gpu_driver_version": "xe (kernel 6.19.0-rc6-v6.19-rc6)",
      "gcc_version": "gcc (Ubuntu 15.2.0-4ubuntu4) 15.2.0",
      "clang_version": "DPC++ compiler 7.1.0 (pre-release) build based on: clang version 23.0.0git (git@github.com:311Volt/llvm.git 38239b406b7af44e4d8093d9f95b763c5b10a34e)",
      "level_zero_version": "L0 v2 adapter | level-zero (version unknown)",
      "compute_runtime_version": ""
    }
  }
];
flamegraphData = { runs: {} };
benchmarkMetadata = {
  "SubmitKernel out of order": {
    "type": "group",
    "description": "Measures CPU time overhead of submitting kernels through different APIs.",
    "notes": "Each layer builds on top of the previous layer, adding functionality and overhead.\nThe first layer is the Level Zero API, the second is the Unified Runtime API, and the third is the SYCL API.\nThe UR v2 adapter noticeably reduces UR layer overhead, also improving SYCL performance.\nWork is ongoing to reduce the overhead of the SYCL API\n",
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "OFFLOAD",
      "submit",
      "latency"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitKernel out of order"
  },
  "SubmitKernel out of order, CPU count": {
    "type": "group",
    "description": "Measures CPU time overhead of submitting kernels through different APIs.",
    "notes": "Each layer builds on top of the previous layer, adding functionality and overhead.\nThe first layer is the Level Zero API, the second is the Unified Runtime API, and the third is the SYCL API.\nThe UR v2 adapter noticeably reduces UR layer overhead, also improving SYCL performance.\nWork is ongoing to reduce the overhead of the SYCL API\n",
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "OFFLOAD",
      "submit",
      "latency"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitKernel out of order, CPU count"
  },
  "SubmitKernel out of order long kernel": {
    "type": "group",
    "description": "Measures CPU time overhead of submitting kernels through different APIs.",
    "notes": "Each layer builds on top of the previous layer, adding functionality and overhead.\nThe first layer is the Level Zero API, the second is the Unified Runtime API, and the third is the SYCL API.\nThe UR v2 adapter noticeably reduces UR layer overhead, also improving SYCL performance.\nWork is ongoing to reduce the overhead of the SYCL API\n",
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "OFFLOAD",
      "submit",
      "latency"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitKernel out of order long kernel"
  },
  "SubmitKernel out of order long kernel, CPU count": {
    "type": "group",
    "description": "Measures CPU time overhead of submitting kernels through different APIs.",
    "notes": "Each layer builds on top of the previous layer, adding functionality and overhead.\nThe first layer is the Level Zero API, the second is the Unified Runtime API, and the third is the SYCL API.\nThe UR v2 adapter noticeably reduces UR layer overhead, also improving SYCL performance.\nWork is ongoing to reduce the overhead of the SYCL API\n",
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "OFFLOAD",
      "submit",
      "latency"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitKernel out of order long kernel, CPU count"
  },
  "SubmitKernel out of order using events": {
    "type": "group",
    "description": "Measures CPU time overhead of submitting kernels through different APIs.",
    "notes": "Each layer builds on top of the previous layer, adding functionality and overhead.\nThe first layer is the Level Zero API, the second is the Unified Runtime API, and the third is the SYCL API.\nThe UR v2 adapter noticeably reduces UR layer overhead, also improving SYCL performance.\nWork is ongoing to reduce the overhead of the SYCL API\n",
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "OFFLOAD",
      "submit",
      "latency"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitKernel out of order using events"
  },
  "SubmitKernel out of order using events, CPU count": {
    "type": "group",
    "description": "Measures CPU time overhead of submitting kernels through different APIs.",
    "notes": "Each layer builds on top of the previous layer, adding functionality and overhead.\nThe first layer is the Level Zero API, the second is the Unified Runtime API, and the third is the SYCL API.\nThe UR v2 adapter noticeably reduces UR layer overhead, also improving SYCL performance.\nWork is ongoing to reduce the overhead of the SYCL API\n",
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "OFFLOAD",
      "submit",
      "latency"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitKernel out of order using events, CPU count"
  },
  "SubmitKernel out of order using events long kernel": {
    "type": "group",
    "description": "Measures CPU time overhead of submitting kernels through different APIs.",
    "notes": "Each layer builds on top of the previous layer, adding functionality and overhead.\nThe first layer is the Level Zero API, the second is the Unified Runtime API, and the third is the SYCL API.\nThe UR v2 adapter noticeably reduces UR layer overhead, also improving SYCL performance.\nWork is ongoing to reduce the overhead of the SYCL API\n",
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "OFFLOAD",
      "submit",
      "latency"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitKernel out of order using events long kernel"
  },
  "SubmitKernel out of order using events long kernel, CPU count": {
    "type": "group",
    "description": "Measures CPU time overhead of submitting kernels through different APIs.",
    "notes": "Each layer builds on top of the previous layer, adding functionality and overhead.\nThe first layer is the Level Zero API, the second is the Unified Runtime API, and the third is the SYCL API.\nThe UR v2 adapter noticeably reduces UR layer overhead, also improving SYCL performance.\nWork is ongoing to reduce the overhead of the SYCL API\n",
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "OFFLOAD",
      "submit",
      "latency"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitKernel out of order using events long kernel, CPU count"
  },
  "SubmitKernel out of order with completion": {
    "type": "group",
    "description": "Measures CPU time overhead of submitting kernels through different APIs.",
    "notes": "Each layer builds on top of the previous layer, adding functionality and overhead.\nThe first layer is the Level Zero API, the second is the Unified Runtime API, and the third is the SYCL API.\nThe UR v2 adapter noticeably reduces UR layer overhead, also improving SYCL performance.\nWork is ongoing to reduce the overhead of the SYCL API\n",
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "OFFLOAD",
      "submit",
      "latency"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitKernel out of order with completion"
  },
  "SubmitKernel out of order with completion, CPU count": {
    "type": "group",
    "description": "Measures CPU time overhead of submitting kernels through different APIs.",
    "notes": "Each layer builds on top of the previous layer, adding functionality and overhead.\nThe first layer is the Level Zero API, the second is the Unified Runtime API, and the third is the SYCL API.\nThe UR v2 adapter noticeably reduces UR layer overhead, also improving SYCL performance.\nWork is ongoing to reduce the overhead of the SYCL API\n",
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "OFFLOAD",
      "submit",
      "latency"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitKernel out of order with completion, CPU count"
  },
  "SubmitKernel out of order with completion long kernel": {
    "type": "group",
    "description": "Measures CPU time overhead of submitting kernels through different APIs.",
    "notes": "Each layer builds on top of the previous layer, adding functionality and overhead.\nThe first layer is the Level Zero API, the second is the Unified Runtime API, and the third is the SYCL API.\nThe UR v2 adapter noticeably reduces UR layer overhead, also improving SYCL performance.\nWork is ongoing to reduce the overhead of the SYCL API\n",
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "OFFLOAD",
      "submit",
      "latency"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitKernel out of order with completion long kernel"
  },
  "SubmitKernel out of order with completion long kernel, CPU count": {
    "type": "group",
    "description": "Measures CPU time overhead of submitting kernels through different APIs.",
    "notes": "Each layer builds on top of the previous layer, adding functionality and overhead.\nThe first layer is the Level Zero API, the second is the Unified Runtime API, and the third is the SYCL API.\nThe UR v2 adapter noticeably reduces UR layer overhead, also improving SYCL performance.\nWork is ongoing to reduce the overhead of the SYCL API\n",
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "OFFLOAD",
      "submit",
      "latency"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitKernel out of order with completion long kernel, CPU count"
  },
  "SubmitKernel out of order with completion using events": {
    "type": "group",
    "description": "Measures CPU time overhead of submitting kernels through different APIs.",
    "notes": "Each layer builds on top of the previous layer, adding functionality and overhead.\nThe first layer is the Level Zero API, the second is the Unified Runtime API, and the third is the SYCL API.\nThe UR v2 adapter noticeably reduces UR layer overhead, also improving SYCL performance.\nWork is ongoing to reduce the overhead of the SYCL API\n",
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "OFFLOAD",
      "submit",
      "latency"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitKernel out of order with completion using events"
  },
  "SubmitKernel out of order with completion using events, CPU count": {
    "type": "group",
    "description": "Measures CPU time overhead of submitting kernels through different APIs.",
    "notes": "Each layer builds on top of the previous layer, adding functionality and overhead.\nThe first layer is the Level Zero API, the second is the Unified Runtime API, and the third is the SYCL API.\nThe UR v2 adapter noticeably reduces UR layer overhead, also improving SYCL performance.\nWork is ongoing to reduce the overhead of the SYCL API\n",
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "OFFLOAD",
      "submit",
      "latency"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitKernel out of order with completion using events, CPU count"
  },
  "SubmitKernel out of order with completion using events long kernel": {
    "type": "group",
    "description": "Measures CPU time overhead of submitting kernels through different APIs.",
    "notes": "Each layer builds on top of the previous layer, adding functionality and overhead.\nThe first layer is the Level Zero API, the second is the Unified Runtime API, and the third is the SYCL API.\nThe UR v2 adapter noticeably reduces UR layer overhead, also improving SYCL performance.\nWork is ongoing to reduce the overhead of the SYCL API\n",
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "OFFLOAD",
      "submit",
      "latency"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitKernel out of order with completion using events long kernel"
  },
  "SubmitKernel out of order with completion using events long kernel, CPU count": {
    "type": "group",
    "description": "Measures CPU time overhead of submitting kernels through different APIs.",
    "notes": "Each layer builds on top of the previous layer, adding functionality and overhead.\nThe first layer is the Level Zero API, the second is the Unified Runtime API, and the third is the SYCL API.\nThe UR v2 adapter noticeably reduces UR layer overhead, also improving SYCL performance.\nWork is ongoing to reduce the overhead of the SYCL API\n",
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "OFFLOAD",
      "submit",
      "latency"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitKernel out of order with completion using events long kernel, CPU count"
  },
  "SubmitKernel in order": {
    "type": "group",
    "description": "Measures CPU time overhead of submitting kernels through different APIs.",
    "notes": "Each layer builds on top of the previous layer, adding functionality and overhead.\nThe first layer is the Level Zero API, the second is the Unified Runtime API, and the third is the SYCL API.\nThe UR v2 adapter noticeably reduces UR layer overhead, also improving SYCL performance.\nWork is ongoing to reduce the overhead of the SYCL API\n",
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "OFFLOAD",
      "submit",
      "latency"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitKernel in order"
  },
  "SubmitKernel in order, CPU count": {
    "type": "group",
    "description": "Measures CPU time overhead of submitting kernels through different APIs.",
    "notes": "Each layer builds on top of the previous layer, adding functionality and overhead.\nThe first layer is the Level Zero API, the second is the Unified Runtime API, and the third is the SYCL API.\nThe UR v2 adapter noticeably reduces UR layer overhead, also improving SYCL performance.\nWork is ongoing to reduce the overhead of the SYCL API\n",
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "OFFLOAD",
      "submit",
      "latency"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitKernel in order, CPU count"
  },
  "SubmitKernel in order long kernel": {
    "type": "group",
    "description": "Measures CPU time overhead of submitting kernels through different APIs.",
    "notes": "Each layer builds on top of the previous layer, adding functionality and overhead.\nThe first layer is the Level Zero API, the second is the Unified Runtime API, and the third is the SYCL API.\nThe UR v2 adapter noticeably reduces UR layer overhead, also improving SYCL performance.\nWork is ongoing to reduce the overhead of the SYCL API\n",
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "OFFLOAD",
      "submit",
      "latency"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitKernel in order long kernel"
  },
  "SubmitKernel in order long kernel, CPU count": {
    "type": "group",
    "description": "Measures CPU time overhead of submitting kernels through different APIs.",
    "notes": "Each layer builds on top of the previous layer, adding functionality and overhead.\nThe first layer is the Level Zero API, the second is the Unified Runtime API, and the third is the SYCL API.\nThe UR v2 adapter noticeably reduces UR layer overhead, also improving SYCL performance.\nWork is ongoing to reduce the overhead of the SYCL API\n",
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "OFFLOAD",
      "submit",
      "latency"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitKernel in order long kernel, CPU count"
  },
  "SubmitKernel in order using events": {
    "type": "group",
    "description": "Measures CPU time overhead of submitting kernels through different APIs.",
    "notes": "Each layer builds on top of the previous layer, adding functionality and overhead.\nThe first layer is the Level Zero API, the second is the Unified Runtime API, and the third is the SYCL API.\nThe UR v2 adapter noticeably reduces UR layer overhead, also improving SYCL performance.\nWork is ongoing to reduce the overhead of the SYCL API\n",
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "OFFLOAD",
      "submit",
      "latency"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitKernel in order using events"
  },
  "SubmitKernel in order using events, CPU count": {
    "type": "group",
    "description": "Measures CPU time overhead of submitting kernels through different APIs.",
    "notes": "Each layer builds on top of the previous layer, adding functionality and overhead.\nThe first layer is the Level Zero API, the second is the Unified Runtime API, and the third is the SYCL API.\nThe UR v2 adapter noticeably reduces UR layer overhead, also improving SYCL performance.\nWork is ongoing to reduce the overhead of the SYCL API\n",
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "OFFLOAD",
      "submit",
      "latency"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitKernel in order using events, CPU count"
  },
  "SubmitKernel in order using events long kernel": {
    "type": "group",
    "description": "Measures CPU time overhead of submitting kernels through different APIs.",
    "notes": "Each layer builds on top of the previous layer, adding functionality and overhead.\nThe first layer is the Level Zero API, the second is the Unified Runtime API, and the third is the SYCL API.\nThe UR v2 adapter noticeably reduces UR layer overhead, also improving SYCL performance.\nWork is ongoing to reduce the overhead of the SYCL API\n",
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "OFFLOAD",
      "submit",
      "latency"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitKernel in order using events long kernel"
  },
  "SubmitKernel in order using events long kernel, CPU count": {
    "type": "group",
    "description": "Measures CPU time overhead of submitting kernels through different APIs.",
    "notes": "Each layer builds on top of the previous layer, adding functionality and overhead.\nThe first layer is the Level Zero API, the second is the Unified Runtime API, and the third is the SYCL API.\nThe UR v2 adapter noticeably reduces UR layer overhead, also improving SYCL performance.\nWork is ongoing to reduce the overhead of the SYCL API\n",
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "OFFLOAD",
      "submit",
      "latency"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitKernel in order using events long kernel, CPU count"
  },
  "SubmitKernel in order with completion": {
    "type": "group",
    "description": "Measures CPU time overhead of submitting kernels through different APIs.",
    "notes": "Each layer builds on top of the previous layer, adding functionality and overhead.\nThe first layer is the Level Zero API, the second is the Unified Runtime API, and the third is the SYCL API.\nThe UR v2 adapter noticeably reduces UR layer overhead, also improving SYCL performance.\nWork is ongoing to reduce the overhead of the SYCL API\n",
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "OFFLOAD",
      "submit",
      "latency"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitKernel in order with completion"
  },
  "SubmitKernel in order with completion, CPU count": {
    "type": "group",
    "description": "Measures CPU time overhead of submitting kernels through different APIs.",
    "notes": "Each layer builds on top of the previous layer, adding functionality and overhead.\nThe first layer is the Level Zero API, the second is the Unified Runtime API, and the third is the SYCL API.\nThe UR v2 adapter noticeably reduces UR layer overhead, also improving SYCL performance.\nWork is ongoing to reduce the overhead of the SYCL API\n",
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "OFFLOAD",
      "submit",
      "latency"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitKernel in order with completion, CPU count"
  },
  "SubmitKernel in order with completion long kernel": {
    "type": "group",
    "description": "Measures CPU time overhead of submitting kernels through different APIs.",
    "notes": "Each layer builds on top of the previous layer, adding functionality and overhead.\nThe first layer is the Level Zero API, the second is the Unified Runtime API, and the third is the SYCL API.\nThe UR v2 adapter noticeably reduces UR layer overhead, also improving SYCL performance.\nWork is ongoing to reduce the overhead of the SYCL API\n",
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "OFFLOAD",
      "submit",
      "latency"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitKernel in order with completion long kernel"
  },
  "SubmitKernel in order with completion long kernel, CPU count": {
    "type": "group",
    "description": "Measures CPU time overhead of submitting kernels through different APIs.",
    "notes": "Each layer builds on top of the previous layer, adding functionality and overhead.\nThe first layer is the Level Zero API, the second is the Unified Runtime API, and the third is the SYCL API.\nThe UR v2 adapter noticeably reduces UR layer overhead, also improving SYCL performance.\nWork is ongoing to reduce the overhead of the SYCL API\n",
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "OFFLOAD",
      "submit",
      "latency"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitKernel in order with completion long kernel, CPU count"
  },
  "SubmitKernel in order with completion using events": {
    "type": "group",
    "description": "Measures CPU time overhead of submitting kernels through different APIs.",
    "notes": "Each layer builds on top of the previous layer, adding functionality and overhead.\nThe first layer is the Level Zero API, the second is the Unified Runtime API, and the third is the SYCL API.\nThe UR v2 adapter noticeably reduces UR layer overhead, also improving SYCL performance.\nWork is ongoing to reduce the overhead of the SYCL API\n",
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "OFFLOAD",
      "submit",
      "latency"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitKernel in order with completion using events"
  },
  "SubmitKernel in order with completion using events, CPU count": {
    "type": "group",
    "description": "Measures CPU time overhead of submitting kernels through different APIs.",
    "notes": "Each layer builds on top of the previous layer, adding functionality and overhead.\nThe first layer is the Level Zero API, the second is the Unified Runtime API, and the third is the SYCL API.\nThe UR v2 adapter noticeably reduces UR layer overhead, also improving SYCL performance.\nWork is ongoing to reduce the overhead of the SYCL API\n",
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "OFFLOAD",
      "submit",
      "latency"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitKernel in order with completion using events, CPU count"
  },
  "SubmitKernel in order with completion using events long kernel": {
    "type": "group",
    "description": "Measures CPU time overhead of submitting kernels through different APIs.",
    "notes": "Each layer builds on top of the previous layer, adding functionality and overhead.\nThe first layer is the Level Zero API, the second is the Unified Runtime API, and the third is the SYCL API.\nThe UR v2 adapter noticeably reduces UR layer overhead, also improving SYCL performance.\nWork is ongoing to reduce the overhead of the SYCL API\n",
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "OFFLOAD",
      "submit",
      "latency"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitKernel in order with completion using events long kernel"
  },
  "SubmitKernel in order with completion using events long kernel, CPU count": {
    "type": "group",
    "description": "Measures CPU time overhead of submitting kernels through different APIs.",
    "notes": "Each layer builds on top of the previous layer, adding functionality and overhead.\nThe first layer is the Level Zero API, the second is the Unified Runtime API, and the third is the SYCL API.\nThe UR v2 adapter noticeably reduces UR layer overhead, also improving SYCL performance.\nWork is ongoing to reduce the overhead of the SYCL API\n",
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "OFFLOAD",
      "submit",
      "latency"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitKernel in order with completion using events long kernel, CPU count"
  },
  "SinKernelGraph, numKernels: 5": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "proxy",
      "SYCL",
      "memory",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SinKernelGraph, numKernels: 5"
  },
  "SinKernelGraph, numKernels: 100": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "proxy",
      "SYCL",
      "memory",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SinKernelGraph, numKernels: 100"
  },
  "EmptyKernel, wgc: 1000, wgs: 256": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "SYCL",
      "micro",
      "L0",
      "latency",
      "UR"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "EmptyKernel, wgc: 1000, wgs: 256"
  },
  "KernelSwitch, count: 8, kernelTime: 200": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "SYCL",
      "micro",
      "L0",
      "latency",
      "UR"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSwitch, count: 8, kernelTime: 200"
  },
  "EmptyKernel, wgc: 1000, wgs: 256, CPU count": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "EmptyKernel, wgc: 1000, wgs: 256, CPU count"
  },
  "SubmitGraph out of order, 4 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph out of order, 4 kernels"
  },
  "SubmitGraph out of order with events, 4 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph out of order with events, 4 kernels"
  },
  "SubmitGraph out of order with measure completion, 4 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph out of order with measure completion, 4 kernels"
  },
  "SubmitGraph out of order with measure completion with events, 4 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph out of order with measure completion with events, 4 kernels"
  },
  "SubmitGraph out of order, 10 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph out of order, 10 kernels"
  },
  "SubmitGraph out of order with events, 10 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph out of order with events, 10 kernels"
  },
  "SubmitGraph out of order with measure completion, 10 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph out of order with measure completion, 10 kernels"
  },
  "SubmitGraph out of order with measure completion with events, 10 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph out of order with measure completion with events, 10 kernels"
  },
  "SubmitGraph out of order, 32 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph out of order, 32 kernels"
  },
  "SubmitGraph out of order with events, 32 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph out of order with events, 32 kernels"
  },
  "SubmitGraph out of order with measure completion, 32 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph out of order with measure completion, 32 kernels"
  },
  "SubmitGraph out of order with measure completion with events, 32 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph out of order with measure completion with events, 32 kernels"
  },
  "SubmitGraph in order, 4 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph in order, 4 kernels"
  },
  "SubmitGraph in order with events, 4 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph in order with events, 4 kernels"
  },
  "SubmitGraph in order with measure completion, 4 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph in order with measure completion, 4 kernels"
  },
  "SubmitGraph in order with measure completion with events, 4 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph in order with measure completion with events, 4 kernels"
  },
  "SubmitGraph in order, 10 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph in order, 10 kernels"
  },
  "SubmitGraph in order with events, 10 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph in order with events, 10 kernels"
  },
  "SubmitGraph in order with measure completion, 10 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph in order with measure completion, 10 kernels"
  },
  "SubmitGraph in order with measure completion with events, 10 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph in order with measure completion with events, 10 kernels"
  },
  "SubmitGraph in order, 32 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph in order, 32 kernels"
  },
  "SubmitGraph in order with events, 32 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph in order with events, 32 kernels"
  },
  "SubmitGraph in order with measure completion, 32 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph in order with measure completion, 32 kernels"
  },
  "SubmitGraph in order with measure completion with events, 32 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph in order with measure completion with events, 32 kernels"
  },
  "SubmitGraph out of order, 4 kernels, CPU count": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph out of order, 4 kernels, CPU count"
  },
  "SubmitGraph out of order with events, 4 kernels, CPU count": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph out of order with events, 4 kernels, CPU count"
  },
  "SubmitGraph out of order with measure completion, 4 kernels, CPU count": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph out of order with measure completion, 4 kernels, CPU count"
  },
  "SubmitGraph out of order with measure completion with events, 4 kernels, CPU count": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph out of order with measure completion with events, 4 kernels, CPU count"
  },
  "SubmitGraph out of order, 10 kernels, CPU count": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph out of order, 10 kernels, CPU count"
  },
  "SubmitGraph out of order with events, 10 kernels, CPU count": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph out of order with events, 10 kernels, CPU count"
  },
  "SubmitGraph out of order with measure completion, 10 kernels, CPU count": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph out of order with measure completion, 10 kernels, CPU count"
  },
  "SubmitGraph out of order with measure completion with events, 10 kernels, CPU count": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph out of order with measure completion with events, 10 kernels, CPU count"
  },
  "SubmitGraph out of order, 32 kernels, CPU count": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph out of order, 32 kernels, CPU count"
  },
  "SubmitGraph out of order with events, 32 kernels, CPU count": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph out of order with events, 32 kernels, CPU count"
  },
  "SubmitGraph out of order with measure completion, 32 kernels, CPU count": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph out of order with measure completion, 32 kernels, CPU count"
  },
  "SubmitGraph out of order with measure completion with events, 32 kernels, CPU count": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph out of order with measure completion with events, 32 kernels, CPU count"
  },
  "SubmitGraph in order, 4 kernels, CPU count": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph in order, 4 kernels, CPU count"
  },
  "SubmitGraph in order with events, 4 kernels, CPU count": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph in order with events, 4 kernels, CPU count"
  },
  "SubmitGraph in order with measure completion, 4 kernels, CPU count": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph in order with measure completion, 4 kernels, CPU count"
  },
  "SubmitGraph in order with measure completion with events, 4 kernels, CPU count": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph in order with measure completion with events, 4 kernels, CPU count"
  },
  "SubmitGraph in order, 10 kernels, CPU count": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph in order, 10 kernels, CPU count"
  },
  "SubmitGraph in order with events, 10 kernels, CPU count": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph in order with events, 10 kernels, CPU count"
  },
  "SubmitGraph in order with measure completion, 10 kernels, CPU count": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph in order with measure completion, 10 kernels, CPU count"
  },
  "SubmitGraph in order with measure completion with events, 10 kernels, CPU count": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph in order with measure completion with events, 10 kernels, CPU count"
  },
  "SubmitGraph in order, 32 kernels, CPU count": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph in order, 32 kernels, CPU count"
  },
  "SubmitGraph in order with events, 32 kernels, CPU count": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph in order with events, 32 kernels, CPU count"
  },
  "SubmitGraph in order with measure completion, 32 kernels, CPU count": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph in order with measure completion, 32 kernels, CPU count"
  },
  "SubmitGraph in order with measure completion with events, 32 kernels, CPU count": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph in order with measure completion with events, 32 kernels, CPU count"
  },
  "SubmitGraph native recording out of order, 4 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph native recording out of order, 4 kernels"
  },
  "SubmitGraph native recording out of order with events, 4 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph native recording out of order with events, 4 kernels"
  },
  "SubmitGraph native recording out of order with measure completion, 4 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph native recording out of order with measure completion, 4 kernels"
  },
  "SubmitGraph native recording out of order with measure completion with events, 4 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph native recording out of order with measure completion with events, 4 kernels"
  },
  "SubmitGraph native recording out of order, 10 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph native recording out of order, 10 kernels"
  },
  "SubmitGraph native recording out of order with events, 10 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph native recording out of order with events, 10 kernels"
  },
  "SubmitGraph native recording out of order with measure completion, 10 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph native recording out of order with measure completion, 10 kernels"
  },
  "SubmitGraph native recording out of order with measure completion with events, 10 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph native recording out of order with measure completion with events, 10 kernels"
  },
  "SubmitGraph native recording out of order, 32 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph native recording out of order, 32 kernels"
  },
  "SubmitGraph native recording out of order with events, 32 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph native recording out of order with events, 32 kernels"
  },
  "SubmitGraph native recording out of order with measure completion, 32 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph native recording out of order with measure completion, 32 kernels"
  },
  "SubmitGraph native recording out of order with measure completion with events, 32 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph native recording out of order with measure completion with events, 32 kernels"
  },
  "SubmitGraph native recording in order, 4 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph native recording in order, 4 kernels"
  },
  "SubmitGraph native recording in order with events, 4 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph native recording in order with events, 4 kernels"
  },
  "SubmitGraph native recording in order with measure completion, 4 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph native recording in order with measure completion, 4 kernels"
  },
  "SubmitGraph native recording in order with measure completion with events, 4 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph native recording in order with measure completion with events, 4 kernels"
  },
  "SubmitGraph native recording in order, 10 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph native recording in order, 10 kernels"
  },
  "SubmitGraph native recording in order with events, 10 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph native recording in order with events, 10 kernels"
  },
  "SubmitGraph native recording in order with measure completion, 10 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph native recording in order with measure completion, 10 kernels"
  },
  "SubmitGraph native recording in order with measure completion with events, 10 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph native recording in order with measure completion with events, 10 kernels"
  },
  "SubmitGraph native recording in order, 32 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph native recording in order, 32 kernels"
  },
  "SubmitGraph native recording in order with events, 32 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph native recording in order with events, 32 kernels"
  },
  "SubmitGraph native recording in order with measure completion, 32 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph native recording in order with measure completion, 32 kernels"
  },
  "SubmitGraph native recording in order with measure completion with events, 32 kernels": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "micro",
      "SYCL",
      "L0",
      "UR",
      "submit",
      "latency",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "SubmitGraph native recording in order with measure completion with events, 32 kernels"
  },
  "FinalizeGraph, GraphStructure: Gromacs": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "latency",
      "micro",
      "SYCL",
      "finalize",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "FinalizeGraph, GraphStructure: Gromacs"
  },
  "FinalizeGraph, GraphStructure: Llama": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "latency",
      "micro",
      "SYCL",
      "finalize",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "FinalizeGraph, GraphStructure: Llama"
  },
  "RecordGraph large": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "RecordGraph large"
  },
  "RecordGraph medium": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "RecordGraph medium"
  },
  "RecordGraph short": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "RecordGraph short"
  },
  "KernelSubmitSingleQueue Int32Large": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitSingleQueue Int32Large"
  },
  "KernelSubmitSingleQueue Int32Medium": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitSingleQueue Int32Medium"
  },
  "KernelSubmitSingleQueue Int32Small": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitSingleQueue Int32Small"
  },
  "KernelSubmitSingleQueue MixedLarge": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitSingleQueue MixedLarge"
  },
  "KernelSubmitSingleQueue MixedMedium": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitSingleQueue MixedMedium"
  },
  "KernelSubmitSingleQueue MixedSmall": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitSingleQueue MixedSmall"
  },
  "KernelSubmitSingleQueue Int32Large, CPU count": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitSingleQueue Int32Large, CPU count"
  },
  "KernelSubmitSingleQueue Int32Medium, CPU count": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitSingleQueue Int32Medium, CPU count"
  },
  "KernelSubmitSingleQueue Int32Small, CPU count": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitSingleQueue Int32Small, CPU count"
  },
  "KernelSubmitSingleQueue MixedLarge, CPU count": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitSingleQueue MixedLarge, CPU count"
  },
  "KernelSubmitSingleQueue MixedMedium, CPU count": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitSingleQueue MixedMedium, CPU count"
  },
  "KernelSubmitSingleQueue MixedSmall, CPU count": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitSingleQueue MixedSmall, CPU count"
  },
  "KernelSubmitMultiQueue large": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitMultiQueue large"
  },
  "KernelSubmitMultiQueue medium": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitMultiQueue medium"
  },
  "KernelSubmitMultiQueue small": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitMultiQueue small"
  },
  "KernelSubmitMultiQueue large with measure completion": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitMultiQueue large with measure completion"
  },
  "KernelSubmitMultiQueue medium with measure completion": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitMultiQueue medium with measure completion"
  },
  "KernelSubmitMultiQueue small with measure completion": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitMultiQueue small with measure completion"
  },
  "KernelSubmitMultiQueue large, CPU count": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitMultiQueue large, CPU count"
  },
  "KernelSubmitMultiQueue medium, CPU count": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitMultiQueue medium, CPU count"
  },
  "KernelSubmitMultiQueue small, CPU count": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitMultiQueue small, CPU count"
  },
  "KernelSubmitMultiQueue large with measure completion, CPU count": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitMultiQueue large with measure completion, CPU count"
  },
  "KernelSubmitMultiQueue medium with measure completion, CPU count": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitMultiQueue medium with measure completion, CPU count"
  },
  "KernelSubmitMultiQueue small with measure completion, CPU count": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitMultiQueue small with measure completion, CPU count"
  },
  "KernelSubmitSlmSize small": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitSlmSize small"
  },
  "KernelSubmitSlmSize medium": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitSlmSize medium"
  },
  "KernelSubmitSlmSize large": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitSlmSize large"
  },
  "KernelSubmitSlmSize small with measure completion": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitSlmSize small with measure completion"
  },
  "KernelSubmitSlmSize medium with measure completion": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitSlmSize medium with measure completion"
  },
  "KernelSubmitSlmSize large with measure completion": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitSlmSize large with measure completion"
  },
  "KernelSubmitSlmSize small, CPU count": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitSlmSize small, CPU count"
  },
  "KernelSubmitSlmSize medium, CPU count": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitSlmSize medium, CPU count"
  },
  "KernelSubmitSlmSize large, CPU count": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitSlmSize large, CPU count"
  },
  "KernelSubmitSlmSize small with measure completion, CPU count": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitSlmSize small with measure completion, CPU count"
  },
  "KernelSubmitSlmSize medium with measure completion, CPU count": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitSlmSize medium with measure completion, CPU count"
  },
  "KernelSubmitSlmSize large with measure completion, CPU count": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitSlmSize large with measure completion, CPU count"
  },
  "KernelSubmitMemoryReuse Int32Large": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitMemoryReuse Int32Large"
  },
  "KernelSubmitMemoryReuse Int32Medium": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitMemoryReuse Int32Medium"
  },
  "KernelSubmitMemoryReuse FloatLarge": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitMemoryReuse FloatLarge"
  },
  "KernelSubmitMemoryReuse FloatMedium": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitMemoryReuse FloatMedium"
  },
  "KernelSubmitMemoryReuse Int32Large, CPU count": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitMemoryReuse Int32Large, CPU count"
  },
  "KernelSubmitMemoryReuse Int32Medium, CPU count": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitMemoryReuse Int32Medium, CPU count"
  },
  "KernelSubmitMemoryReuse FloatLarge, CPU count": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitMemoryReuse FloatLarge, CPU count"
  },
  "KernelSubmitMemoryReuse FloatMedium, CPU count": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitMemoryReuse FloatMedium, CPU count"
  },
  "KernelSubmitLinearKernelSize array32": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitLinearKernelSize array32"
  },
  "KernelSubmitLinearKernelSize array128": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitLinearKernelSize array128"
  },
  "KernelSubmitLinearKernelSize array512": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitLinearKernelSize array512"
  },
  "KernelSubmitLinearKernelSize array1024": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitLinearKernelSize array1024"
  },
  "KernelSubmitLinearKernelSize array5120": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitLinearKernelSize array5120"
  },
  "KernelSubmitLinearKernelSize array32, CPU count": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitLinearKernelSize array32, CPU count"
  },
  "KernelSubmitLinearKernelSize array128, CPU count": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitLinearKernelSize array128, CPU count"
  },
  "KernelSubmitLinearKernelSize array512, CPU count": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitLinearKernelSize array512, CPU count"
  },
  "KernelSubmitLinearKernelSize array1024, CPU count": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitLinearKernelSize array1024, CPU count"
  },
  "KernelSubmitLinearKernelSize array5120, CPU count": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitLinearKernelSize array5120, CPU count"
  },
  "KernelSubmitEventRecordWait medium": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitEventRecordWait medium"
  },
  "KernelSubmitEventRecordWait medium, CPU count": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitEventRecordWait medium, CPU count"
  },
  "KernelSubmitEventRecordQuery medium": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitEventRecordQuery medium"
  },
  "KernelSubmitEventRecordQuery medium, CPU count": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitEventRecordQuery medium, CPU count"
  },
  "KernelSubmitGraphSingleQueue small": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitGraphSingleQueue small"
  },
  "KernelSubmitGraphSingleQueue medium": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitGraphSingleQueue medium"
  },
  "KernelSubmitGraphSingleQueue large": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitGraphSingleQueue large"
  },
  "KernelSubmitGraphSingleQueue small, CPU count": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitGraphSingleQueue small, CPU count"
  },
  "KernelSubmitGraphSingleQueue medium, CPU count": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitGraphSingleQueue medium, CPU count"
  },
  "KernelSubmitGraphSingleQueue large, CPU count": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitGraphSingleQueue large, CPU count"
  },
  "KernelSubmitGraphMultiQueue small": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitGraphMultiQueue small"
  },
  "KernelSubmitGraphMultiQueue medium": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitGraphMultiQueue medium"
  },
  "KernelSubmitGraphMultiQueue large": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitGraphMultiQueue large"
  },
  "KernelSubmitGraphMultiQueue small, CPU count": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitGraphMultiQueue small, CPU count"
  },
  "KernelSubmitGraphMultiQueue medium, CPU count": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitGraphMultiQueue medium, CPU count"
  },
  "KernelSubmitGraphMultiQueue large, CPU count": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitGraphMultiQueue large, CPU count"
  },
  "KernelSubmitGraphVllmMock small": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitGraphVllmMock small"
  },
  "KernelSubmitGraphVllmMock large": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitGraphVllmMock large"
  },
  "KernelSubmitGraphVllmMock small, CPU count": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitGraphVllmMock small, CPU count"
  },
  "KernelSubmitGraphVllmMock large, CPU count": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "SYCL",
      "pytorch"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "KernelSubmitGraphVllmMock large, CPU count"
  },
  "UsmMemoryAllocation": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "latency",
      "micro",
      "memory",
      "UR"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "UsmMemoryAllocation"
  },
  "UsmBatchMemoryAllocation": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "latency",
      "micro",
      "memory",
      "UR"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "UsmBatchMemoryAllocation"
  },
  "MemcpyExecute, opsPerThread: 4096, numThreads: 1, allocSize: 1024": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "latency",
      "memory",
      "micro",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "MemcpyExecute, opsPerThread: 4096, numThreads: 1, allocSize: 1024"
  },
  "MemcpyExecute, opsPerThread: 4096, numThreads: 4, allocSize: 1024": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "latency",
      "memory",
      "micro",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": "MemcpyExecute, opsPerThread: 4096, numThreads: 4, allocSize: 1024"
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
  "api_overhead_benchmark_ol SubmitKernel out of order not using events": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Offload API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "OFFLOAD",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "OL SubmitKernel out of order, NumKernels 10",
    "explicit_group": "SubmitKernel out of order"
  },
  "api_overhead_benchmark_ol SubmitKernel out of order not using events CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Offload API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "OFFLOAD",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "OL SubmitKernel out of order, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order, CPU count"
  },
  "api_overhead_benchmark_ol SubmitKernel out of order not using events KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Offload API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "OFFLOAD",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "OL SubmitKernel out of order KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel out of order long kernel"
  },
  "api_overhead_benchmark_ol SubmitKernel out of order not using events KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Offload API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "OFFLOAD",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "OL SubmitKernel out of order KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order long kernel, CPU count"
  },
  "api_overhead_benchmark_ol SubmitKernel out of order not using events KernelExecTime=200": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Offload API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 200 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "OFFLOAD",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "OL SubmitKernel out of order KernelExecTime=200, NumKernels 10",
    "explicit_group": "SubmitKernel out of order long kernel"
  },
  "api_overhead_benchmark_ol SubmitKernel out of order not using events KernelExecTime=200 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Offload API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 200 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "OFFLOAD",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "OL SubmitKernel out of order KernelExecTime=200, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order long kernel, CPU count"
  },
  "api_overhead_benchmark_ol SubmitKernel out of order": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Offload API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "OFFLOAD",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "OL SubmitKernel out of order using events, NumKernels 10",
    "explicit_group": "SubmitKernel out of order using events"
  },
  "api_overhead_benchmark_ol SubmitKernel out of order CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Offload API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "OFFLOAD",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "OL SubmitKernel out of order using events, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order using events, CPU count"
  },
  "api_overhead_benchmark_ol SubmitKernel out of order KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Offload API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "OFFLOAD",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "OL SubmitKernel out of order using events KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel out of order using events long kernel"
  },
  "api_overhead_benchmark_ol SubmitKernel out of order KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Offload API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "OFFLOAD",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "OL SubmitKernel out of order using events KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order using events long kernel, CPU count"
  },
  "api_overhead_benchmark_ol SubmitKernel out of order KernelExecTime=200": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Offload API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 200 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "OFFLOAD",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "OL SubmitKernel out of order using events KernelExecTime=200, NumKernels 10",
    "explicit_group": "SubmitKernel out of order using events long kernel"
  },
  "api_overhead_benchmark_ol SubmitKernel out of order KernelExecTime=200 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Offload API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 200 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "OFFLOAD",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "OL SubmitKernel out of order using events KernelExecTime=200, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order using events long kernel, CPU count"
  },
  "api_overhead_benchmark_ol SubmitKernel out of order with measure completion not using events": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Offload API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "OFFLOAD",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "OL SubmitKernel out of order with measure completion, NumKernels 10",
    "explicit_group": "SubmitKernel out of order with completion"
  },
  "api_overhead_benchmark_ol SubmitKernel out of order with measure completion not using events CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Offload API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "OFFLOAD",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "OL SubmitKernel out of order with measure completion, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order with completion, CPU count"
  },
  "api_overhead_benchmark_ol SubmitKernel out of order with measure completion not using events KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Offload API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "OFFLOAD",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "OL SubmitKernel out of order with measure completion KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel out of order with completion long kernel"
  },
  "api_overhead_benchmark_ol SubmitKernel out of order with measure completion not using events KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Offload API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "OFFLOAD",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "OL SubmitKernel out of order with measure completion KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order with completion long kernel, CPU count"
  },
  "api_overhead_benchmark_ol SubmitKernel out of order with measure completion not using events KernelExecTime=200": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Offload API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 200 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "OFFLOAD",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "OL SubmitKernel out of order with measure completion KernelExecTime=200, NumKernels 10",
    "explicit_group": "SubmitKernel out of order with completion long kernel"
  },
  "api_overhead_benchmark_ol SubmitKernel out of order with measure completion not using events KernelExecTime=200 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Offload API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 200 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "OFFLOAD",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "OL SubmitKernel out of order with measure completion KernelExecTime=200, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order with completion long kernel, CPU count"
  },
  "api_overhead_benchmark_ol SubmitKernel out of order with measure completion": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Offload API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "OFFLOAD",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "OL SubmitKernel out of order with measure completion using events, NumKernels 10",
    "explicit_group": "SubmitKernel out of order with completion using events"
  },
  "api_overhead_benchmark_ol SubmitKernel out of order with measure completion CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Offload API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "OFFLOAD",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "OL SubmitKernel out of order with measure completion using events, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order with completion using events, CPU count"
  },
  "api_overhead_benchmark_ol SubmitKernel out of order with measure completion KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Offload API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "OFFLOAD",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "OL SubmitKernel out of order with measure completion using events KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel out of order with completion using events long kernel"
  },
  "api_overhead_benchmark_ol SubmitKernel out of order with measure completion KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Offload API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "OFFLOAD",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "OL SubmitKernel out of order with measure completion using events KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order with completion using events long kernel, CPU count"
  },
  "api_overhead_benchmark_ol SubmitKernel out of order with measure completion KernelExecTime=200": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Offload API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 200 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "OFFLOAD",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "OL SubmitKernel out of order with measure completion using events KernelExecTime=200, NumKernels 10",
    "explicit_group": "SubmitKernel out of order with completion using events long kernel"
  },
  "api_overhead_benchmark_ol SubmitKernel out of order with measure completion KernelExecTime=200 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Offload API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 200 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "OFFLOAD",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "OL SubmitKernel out of order with measure completion using events KernelExecTime=200, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order with completion using events long kernel, CPU count"
  },
  "api_overhead_benchmark_ol SubmitKernel in order not using events": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Offload API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "OFFLOAD",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "OL SubmitKernel in order, NumKernels 10",
    "explicit_group": "SubmitKernel in order"
  },
  "api_overhead_benchmark_ol SubmitKernel in order not using events CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Offload API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "OFFLOAD",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "OL SubmitKernel in order, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order, CPU count"
  },
  "api_overhead_benchmark_ol SubmitKernel in order not using events KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Offload API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "OFFLOAD",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "OL SubmitKernel in order KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel in order long kernel"
  },
  "api_overhead_benchmark_ol SubmitKernel in order not using events KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Offload API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "OFFLOAD",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "OL SubmitKernel in order KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order long kernel, CPU count"
  },
  "api_overhead_benchmark_ol SubmitKernel in order": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Offload API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "OFFLOAD",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "OL SubmitKernel in order using events, NumKernels 10",
    "explicit_group": "SubmitKernel in order using events"
  },
  "api_overhead_benchmark_ol SubmitKernel in order CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Offload API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "OFFLOAD",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "OL SubmitKernel in order using events, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order using events, CPU count"
  },
  "api_overhead_benchmark_ol SubmitKernel in order KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Offload API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "OFFLOAD",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "OL SubmitKernel in order using events KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel in order using events long kernel"
  },
  "api_overhead_benchmark_ol SubmitKernel in order KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Offload API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "OFFLOAD",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "OL SubmitKernel in order using events KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order using events long kernel, CPU count"
  },
  "api_overhead_benchmark_ol SubmitKernel in order with measure completion not using events": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Offload API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "OFFLOAD",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "OL SubmitKernel in order with measure completion, NumKernels 10",
    "explicit_group": "SubmitKernel in order with completion"
  },
  "api_overhead_benchmark_ol SubmitKernel in order with measure completion not using events CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Offload API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "OFFLOAD",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "OL SubmitKernel in order with measure completion, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order with completion, CPU count"
  },
  "api_overhead_benchmark_ol SubmitKernel in order with measure completion not using events KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Offload API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "OFFLOAD",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "OL SubmitKernel in order with measure completion KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel in order with completion long kernel"
  },
  "api_overhead_benchmark_ol SubmitKernel in order with measure completion not using events KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Offload API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "OFFLOAD",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "OL SubmitKernel in order with measure completion KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order with completion long kernel, CPU count"
  },
  "api_overhead_benchmark_ol SubmitKernel in order with measure completion": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Offload API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "OFFLOAD",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "OL SubmitKernel in order with measure completion using events, NumKernels 10",
    "explicit_group": "SubmitKernel in order with completion using events"
  },
  "api_overhead_benchmark_ol SubmitKernel in order with measure completion CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Offload API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "OFFLOAD",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "OL SubmitKernel in order with measure completion using events, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order with completion using events, CPU count"
  },
  "api_overhead_benchmark_ol SubmitKernel in order with measure completion KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Offload API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "OFFLOAD",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "OL SubmitKernel in order with measure completion using events KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel in order with completion using events long kernel"
  },
  "api_overhead_benchmark_ol SubmitKernel in order with measure completion KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Offload API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "OFFLOAD",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "OL SubmitKernel in order with measure completion using events KernelExecTime=20, NumKernels 10, CPU count",
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
  "ulls_benchmark_sycl EmptyKernel wgc:1000, wgs:256 CPU count": {
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
    "display_name": "SYCL EmptyKernel, wgc 1000, wgs 256, CPU count",
    "explicit_group": "EmptyKernel, wgc: 1000, wgs: 256, CPU count"
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
  "graph_api_benchmark_syclpreview SubmitGraph numKernels:4 ioq 0 MeasureCompletionTime 0": {
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
  "graph_api_benchmark_syclpreview SubmitGraph with events numKernels:4 ioq 0 MeasureCompletionTime 0": {
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
  "graph_api_benchmark_syclpreview SubmitGraph numKernels:4 ioq 0 MeasureCompletionTime 1": {
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
  "graph_api_benchmark_syclpreview SubmitGraph with events numKernels:4 ioq 0 MeasureCompletionTime 1": {
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
  "graph_api_benchmark_syclpreview SubmitGraph numKernels:10 ioq 0 MeasureCompletionTime 0": {
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
  "graph_api_benchmark_syclpreview SubmitGraph with events numKernels:10 ioq 0 MeasureCompletionTime 0": {
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
  "graph_api_benchmark_syclpreview SubmitGraph numKernels:10 ioq 0 MeasureCompletionTime 1": {
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
  "graph_api_benchmark_syclpreview SubmitGraph with events numKernels:10 ioq 0 MeasureCompletionTime 1": {
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
  "graph_api_benchmark_syclpreview SubmitGraph numKernels:32 ioq 0 MeasureCompletionTime 0": {
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
  "graph_api_benchmark_syclpreview SubmitGraph with events numKernels:32 ioq 0 MeasureCompletionTime 0": {
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
  "graph_api_benchmark_syclpreview SubmitGraph numKernels:32 ioq 0 MeasureCompletionTime 1": {
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
  "graph_api_benchmark_syclpreview SubmitGraph with events numKernels:32 ioq 0 MeasureCompletionTime 1": {
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
  "graph_api_benchmark_syclpreview SubmitGraph numKernels:4 ioq 1 MeasureCompletionTime 0": {
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
  "graph_api_benchmark_syclpreview SubmitGraph with events numKernels:4 ioq 1 MeasureCompletionTime 0": {
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
  "graph_api_benchmark_syclpreview SubmitGraph numKernels:4 ioq 1 MeasureCompletionTime 1": {
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
  "graph_api_benchmark_syclpreview SubmitGraph with events numKernels:4 ioq 1 MeasureCompletionTime 1": {
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
  "graph_api_benchmark_syclpreview SubmitGraph numKernels:10 ioq 1 MeasureCompletionTime 0": {
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
  "graph_api_benchmark_syclpreview SubmitGraph with events numKernels:10 ioq 1 MeasureCompletionTime 0": {
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
  "graph_api_benchmark_syclpreview SubmitGraph numKernels:10 ioq 1 MeasureCompletionTime 1": {
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
  "graph_api_benchmark_syclpreview SubmitGraph with events numKernels:10 ioq 1 MeasureCompletionTime 1": {
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
  "graph_api_benchmark_syclpreview SubmitGraph numKernels:32 ioq 1 MeasureCompletionTime 0": {
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
  "graph_api_benchmark_syclpreview SubmitGraph with events numKernels:32 ioq 1 MeasureCompletionTime 0": {
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
  "graph_api_benchmark_syclpreview SubmitGraph numKernels:32 ioq 1 MeasureCompletionTime 1": {
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
  "graph_api_benchmark_syclpreview SubmitGraph with events numKernels:32 ioq 1 MeasureCompletionTime 1": {
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
  "graph_api_benchmark_sycl SubmitGraph numKernels:4 ioq 0 MeasureCompletionTime 0": {
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
  "graph_api_benchmark_sycl SubmitGraph numKernels:4 ioq 0 MeasureCompletionTime 0 CPU count": {
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
    "display_name": "SYCL SubmitGraph out of order, 4 kernels, CPU count",
    "explicit_group": "SubmitGraph out of order, 4 kernels, CPU count"
  },
  "graph_api_benchmark_sycl SubmitGraph with events numKernels:4 ioq 0 MeasureCompletionTime 0": {
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
  "graph_api_benchmark_sycl SubmitGraph with events numKernels:4 ioq 0 MeasureCompletionTime 0 CPU count": {
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
    "display_name": "SYCL SubmitGraph out of order with events, 4 kernels, CPU count",
    "explicit_group": "SubmitGraph out of order with events, 4 kernels, CPU count"
  },
  "graph_api_benchmark_sycl SubmitGraph numKernels:4 ioq 0 MeasureCompletionTime 1": {
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
  "graph_api_benchmark_sycl SubmitGraph numKernels:4 ioq 0 MeasureCompletionTime 1 CPU count": {
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
    "display_name": "SYCL SubmitGraph out of order with measure completion, 4 kernels, CPU count",
    "explicit_group": "SubmitGraph out of order with measure completion, 4 kernels, CPU count"
  },
  "graph_api_benchmark_sycl SubmitGraph with events numKernels:4 ioq 0 MeasureCompletionTime 1": {
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
  "graph_api_benchmark_sycl SubmitGraph with events numKernels:4 ioq 0 MeasureCompletionTime 1 CPU count": {
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
    "display_name": "SYCL SubmitGraph out of order with measure completion with events, 4 kernels, CPU count",
    "explicit_group": "SubmitGraph out of order with measure completion with events, 4 kernels, CPU count"
  },
  "graph_api_benchmark_sycl SubmitGraph numKernels:10 ioq 0 MeasureCompletionTime 0": {
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
  "graph_api_benchmark_sycl SubmitGraph numKernels:10 ioq 0 MeasureCompletionTime 0 CPU count": {
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
    "display_name": "SYCL SubmitGraph out of order, 10 kernels, CPU count",
    "explicit_group": "SubmitGraph out of order, 10 kernels, CPU count"
  },
  "graph_api_benchmark_sycl SubmitGraph with events numKernels:10 ioq 0 MeasureCompletionTime 0": {
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
  "graph_api_benchmark_sycl SubmitGraph with events numKernels:10 ioq 0 MeasureCompletionTime 0 CPU count": {
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
    "display_name": "SYCL SubmitGraph out of order with events, 10 kernels, CPU count",
    "explicit_group": "SubmitGraph out of order with events, 10 kernels, CPU count"
  },
  "graph_api_benchmark_sycl SubmitGraph numKernels:10 ioq 0 MeasureCompletionTime 1": {
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
  "graph_api_benchmark_sycl SubmitGraph numKernels:10 ioq 0 MeasureCompletionTime 1 CPU count": {
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
    "display_name": "SYCL SubmitGraph out of order with measure completion, 10 kernels, CPU count",
    "explicit_group": "SubmitGraph out of order with measure completion, 10 kernels, CPU count"
  },
  "graph_api_benchmark_sycl SubmitGraph with events numKernels:10 ioq 0 MeasureCompletionTime 1": {
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
  "graph_api_benchmark_sycl SubmitGraph with events numKernels:10 ioq 0 MeasureCompletionTime 1 CPU count": {
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
    "display_name": "SYCL SubmitGraph out of order with measure completion with events, 10 kernels, CPU count",
    "explicit_group": "SubmitGraph out of order with measure completion with events, 10 kernels, CPU count"
  },
  "graph_api_benchmark_sycl SubmitGraph numKernels:32 ioq 0 MeasureCompletionTime 0": {
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
  "graph_api_benchmark_sycl SubmitGraph numKernels:32 ioq 0 MeasureCompletionTime 0 CPU count": {
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
    "display_name": "SYCL SubmitGraph out of order, 32 kernels, CPU count",
    "explicit_group": "SubmitGraph out of order, 32 kernels, CPU count"
  },
  "graph_api_benchmark_sycl SubmitGraph with events numKernels:32 ioq 0 MeasureCompletionTime 0": {
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
  "graph_api_benchmark_sycl SubmitGraph with events numKernels:32 ioq 0 MeasureCompletionTime 0 CPU count": {
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
    "display_name": "SYCL SubmitGraph out of order with events, 32 kernels, CPU count",
    "explicit_group": "SubmitGraph out of order with events, 32 kernels, CPU count"
  },
  "graph_api_benchmark_sycl SubmitGraph numKernels:32 ioq 0 MeasureCompletionTime 1": {
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
  "graph_api_benchmark_sycl SubmitGraph numKernels:32 ioq 0 MeasureCompletionTime 1 CPU count": {
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
    "display_name": "SYCL SubmitGraph out of order with measure completion, 32 kernels, CPU count",
    "explicit_group": "SubmitGraph out of order with measure completion, 32 kernels, CPU count"
  },
  "graph_api_benchmark_sycl SubmitGraph with events numKernels:32 ioq 0 MeasureCompletionTime 1": {
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
  "graph_api_benchmark_sycl SubmitGraph with events numKernels:32 ioq 0 MeasureCompletionTime 1 CPU count": {
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
    "display_name": "SYCL SubmitGraph out of order with measure completion with events, 32 kernels, CPU count",
    "explicit_group": "SubmitGraph out of order with measure completion with events, 32 kernels, CPU count"
  },
  "graph_api_benchmark_sycl SubmitGraph numKernels:4 ioq 1 MeasureCompletionTime 0": {
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
  "graph_api_benchmark_sycl SubmitGraph numKernels:4 ioq 1 MeasureCompletionTime 0 CPU count": {
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
    "display_name": "SYCL SubmitGraph in order, 4 kernels, CPU count",
    "explicit_group": "SubmitGraph in order, 4 kernels, CPU count"
  },
  "graph_api_benchmark_sycl SubmitGraph with events numKernels:4 ioq 1 MeasureCompletionTime 0": {
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
  "graph_api_benchmark_sycl SubmitGraph with events numKernels:4 ioq 1 MeasureCompletionTime 0 CPU count": {
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
    "display_name": "SYCL SubmitGraph in order with events, 4 kernels, CPU count",
    "explicit_group": "SubmitGraph in order with events, 4 kernels, CPU count"
  },
  "graph_api_benchmark_sycl SubmitGraph numKernels:4 ioq 1 MeasureCompletionTime 1": {
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
  "graph_api_benchmark_sycl SubmitGraph numKernels:4 ioq 1 MeasureCompletionTime 1 CPU count": {
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
    "display_name": "SYCL SubmitGraph in order with measure completion, 4 kernels, CPU count",
    "explicit_group": "SubmitGraph in order with measure completion, 4 kernels, CPU count"
  },
  "graph_api_benchmark_sycl SubmitGraph with events numKernels:4 ioq 1 MeasureCompletionTime 1": {
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
  "graph_api_benchmark_sycl SubmitGraph with events numKernels:4 ioq 1 MeasureCompletionTime 1 CPU count": {
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
    "display_name": "SYCL SubmitGraph in order with measure completion with events, 4 kernels, CPU count",
    "explicit_group": "SubmitGraph in order with measure completion with events, 4 kernels, CPU count"
  },
  "graph_api_benchmark_sycl SubmitGraph numKernels:10 ioq 1 MeasureCompletionTime 0": {
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
  "graph_api_benchmark_sycl SubmitGraph numKernels:10 ioq 1 MeasureCompletionTime 0 CPU count": {
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
    "display_name": "SYCL SubmitGraph in order, 10 kernels, CPU count",
    "explicit_group": "SubmitGraph in order, 10 kernels, CPU count"
  },
  "graph_api_benchmark_sycl SubmitGraph with events numKernels:10 ioq 1 MeasureCompletionTime 0": {
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
  "graph_api_benchmark_sycl SubmitGraph with events numKernels:10 ioq 1 MeasureCompletionTime 0 CPU count": {
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
    "display_name": "SYCL SubmitGraph in order with events, 10 kernels, CPU count",
    "explicit_group": "SubmitGraph in order with events, 10 kernels, CPU count"
  },
  "graph_api_benchmark_sycl SubmitGraph numKernels:10 ioq 1 MeasureCompletionTime 1": {
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
  "graph_api_benchmark_sycl SubmitGraph numKernels:10 ioq 1 MeasureCompletionTime 1 CPU count": {
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
    "display_name": "SYCL SubmitGraph in order with measure completion, 10 kernels, CPU count",
    "explicit_group": "SubmitGraph in order with measure completion, 10 kernels, CPU count"
  },
  "graph_api_benchmark_sycl SubmitGraph with events numKernels:10 ioq 1 MeasureCompletionTime 1": {
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
  "graph_api_benchmark_sycl SubmitGraph with events numKernels:10 ioq 1 MeasureCompletionTime 1 CPU count": {
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
    "display_name": "SYCL SubmitGraph in order with measure completion with events, 10 kernels, CPU count",
    "explicit_group": "SubmitGraph in order with measure completion with events, 10 kernels, CPU count"
  },
  "graph_api_benchmark_sycl SubmitGraph numKernels:32 ioq 1 MeasureCompletionTime 0": {
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
  "graph_api_benchmark_sycl SubmitGraph numKernels:32 ioq 1 MeasureCompletionTime 0 CPU count": {
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
    "display_name": "SYCL SubmitGraph in order, 32 kernels, CPU count",
    "explicit_group": "SubmitGraph in order, 32 kernels, CPU count"
  },
  "graph_api_benchmark_sycl SubmitGraph with events numKernels:32 ioq 1 MeasureCompletionTime 0": {
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
  "graph_api_benchmark_sycl SubmitGraph with events numKernels:32 ioq 1 MeasureCompletionTime 0 CPU count": {
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
    "display_name": "SYCL SubmitGraph in order with events, 32 kernels, CPU count",
    "explicit_group": "SubmitGraph in order with events, 32 kernels, CPU count"
  },
  "graph_api_benchmark_sycl SubmitGraph numKernels:32 ioq 1 MeasureCompletionTime 1": {
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
  "graph_api_benchmark_sycl SubmitGraph numKernels:32 ioq 1 MeasureCompletionTime 1 CPU count": {
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
    "display_name": "SYCL SubmitGraph in order with measure completion, 32 kernels, CPU count",
    "explicit_group": "SubmitGraph in order with measure completion, 32 kernels, CPU count"
  },
  "graph_api_benchmark_sycl SubmitGraph with events numKernels:32 ioq 1 MeasureCompletionTime 1": {
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
  "graph_api_benchmark_sycl SubmitGraph with events numKernels:32 ioq 1 MeasureCompletionTime 1 CPU count": {
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
    "display_name": "SYCL SubmitGraph in order with measure completion with events, 32 kernels, CPU count",
    "explicit_group": "SubmitGraph in order with measure completion with events, 32 kernels, CPU count"
  },
  "graph_api_benchmark_l0 SubmitGraph native recording numKernels:4 ioq 0 MeasureCompletionTime 0": {
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
    "display_name": "L0 SubmitGraph native recording out of order, 4 kernels",
    "explicit_group": "SubmitGraph native recording out of order, 4 kernels"
  },
  "graph_api_benchmark_l0 SubmitGraph numKernels:4 ioq 0 MeasureCompletionTime 0": {
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
  "graph_api_benchmark_l0 SubmitGraph native recording with events numKernels:4 ioq 0 MeasureCompletionTime 0": {
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
    "display_name": "L0 SubmitGraph native recording out of order with events, 4 kernels",
    "explicit_group": "SubmitGraph native recording out of order with events, 4 kernels"
  },
  "graph_api_benchmark_l0 SubmitGraph with events numKernels:4 ioq 0 MeasureCompletionTime 0": {
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
  "graph_api_benchmark_l0 SubmitGraph native recording numKernels:4 ioq 0 MeasureCompletionTime 1": {
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
    "display_name": "L0 SubmitGraph native recording out of order with measure completion, 4 kernels",
    "explicit_group": "SubmitGraph native recording out of order with measure completion, 4 kernels"
  },
  "graph_api_benchmark_l0 SubmitGraph numKernels:4 ioq 0 MeasureCompletionTime 1": {
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
  "graph_api_benchmark_l0 SubmitGraph native recording with events numKernels:4 ioq 0 MeasureCompletionTime 1": {
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
    "display_name": "L0 SubmitGraph native recording out of order with measure completion with events, 4 kernels",
    "explicit_group": "SubmitGraph native recording out of order with measure completion with events, 4 kernels"
  },
  "graph_api_benchmark_l0 SubmitGraph with events numKernels:4 ioq 0 MeasureCompletionTime 1": {
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
  "graph_api_benchmark_l0 SubmitGraph native recording numKernels:10 ioq 0 MeasureCompletionTime 0": {
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
    "display_name": "L0 SubmitGraph native recording out of order, 10 kernels",
    "explicit_group": "SubmitGraph native recording out of order, 10 kernels"
  },
  "graph_api_benchmark_l0 SubmitGraph numKernels:10 ioq 0 MeasureCompletionTime 0": {
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
  "graph_api_benchmark_l0 SubmitGraph native recording with events numKernels:10 ioq 0 MeasureCompletionTime 0": {
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
    "display_name": "L0 SubmitGraph native recording out of order with events, 10 kernels",
    "explicit_group": "SubmitGraph native recording out of order with events, 10 kernels"
  },
  "graph_api_benchmark_l0 SubmitGraph with events numKernels:10 ioq 0 MeasureCompletionTime 0": {
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
  "graph_api_benchmark_l0 SubmitGraph native recording numKernels:10 ioq 0 MeasureCompletionTime 1": {
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
    "display_name": "L0 SubmitGraph native recording out of order with measure completion, 10 kernels",
    "explicit_group": "SubmitGraph native recording out of order with measure completion, 10 kernels"
  },
  "graph_api_benchmark_l0 SubmitGraph numKernels:10 ioq 0 MeasureCompletionTime 1": {
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
  "graph_api_benchmark_l0 SubmitGraph native recording with events numKernels:10 ioq 0 MeasureCompletionTime 1": {
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
    "display_name": "L0 SubmitGraph native recording out of order with measure completion with events, 10 kernels",
    "explicit_group": "SubmitGraph native recording out of order with measure completion with events, 10 kernels"
  },
  "graph_api_benchmark_l0 SubmitGraph with events numKernels:10 ioq 0 MeasureCompletionTime 1": {
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
  "graph_api_benchmark_l0 SubmitGraph native recording numKernels:32 ioq 0 MeasureCompletionTime 0": {
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
    "display_name": "L0 SubmitGraph native recording out of order, 32 kernels",
    "explicit_group": "SubmitGraph native recording out of order, 32 kernels"
  },
  "graph_api_benchmark_l0 SubmitGraph numKernels:32 ioq 0 MeasureCompletionTime 0": {
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
  "graph_api_benchmark_l0 SubmitGraph native recording with events numKernels:32 ioq 0 MeasureCompletionTime 0": {
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
    "display_name": "L0 SubmitGraph native recording out of order with events, 32 kernels",
    "explicit_group": "SubmitGraph native recording out of order with events, 32 kernels"
  },
  "graph_api_benchmark_l0 SubmitGraph with events numKernels:32 ioq 0 MeasureCompletionTime 0": {
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
  "graph_api_benchmark_l0 SubmitGraph native recording numKernels:32 ioq 0 MeasureCompletionTime 1": {
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
    "display_name": "L0 SubmitGraph native recording out of order with measure completion, 32 kernels",
    "explicit_group": "SubmitGraph native recording out of order with measure completion, 32 kernels"
  },
  "graph_api_benchmark_l0 SubmitGraph numKernels:32 ioq 0 MeasureCompletionTime 1": {
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
  "graph_api_benchmark_l0 SubmitGraph native recording with events numKernels:32 ioq 0 MeasureCompletionTime 1": {
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
    "display_name": "L0 SubmitGraph native recording out of order with measure completion with events, 32 kernels",
    "explicit_group": "SubmitGraph native recording out of order with measure completion with events, 32 kernels"
  },
  "graph_api_benchmark_l0 SubmitGraph with events numKernels:32 ioq 0 MeasureCompletionTime 1": {
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
  "graph_api_benchmark_l0 SubmitGraph native recording numKernels:4 ioq 1 MeasureCompletionTime 0": {
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
    "display_name": "L0 SubmitGraph native recording in order, 4 kernels",
    "explicit_group": "SubmitGraph native recording in order, 4 kernels"
  },
  "graph_api_benchmark_l0 SubmitGraph numKernels:4 ioq 1 MeasureCompletionTime 0": {
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
  "graph_api_benchmark_l0 SubmitGraph native recording with events numKernels:4 ioq 1 MeasureCompletionTime 0": {
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
    "display_name": "L0 SubmitGraph native recording in order with events, 4 kernels",
    "explicit_group": "SubmitGraph native recording in order with events, 4 kernels"
  },
  "graph_api_benchmark_l0 SubmitGraph with events numKernels:4 ioq 1 MeasureCompletionTime 0": {
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
  "graph_api_benchmark_l0 SubmitGraph native recording numKernels:4 ioq 1 MeasureCompletionTime 1": {
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
    "display_name": "L0 SubmitGraph native recording in order with measure completion, 4 kernels",
    "explicit_group": "SubmitGraph native recording in order with measure completion, 4 kernels"
  },
  "graph_api_benchmark_l0 SubmitGraph numKernels:4 ioq 1 MeasureCompletionTime 1": {
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
  "graph_api_benchmark_l0 SubmitGraph native recording with events numKernels:4 ioq 1 MeasureCompletionTime 1": {
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
    "display_name": "L0 SubmitGraph native recording in order with measure completion with events, 4 kernels",
    "explicit_group": "SubmitGraph native recording in order with measure completion with events, 4 kernels"
  },
  "graph_api_benchmark_l0 SubmitGraph with events numKernels:4 ioq 1 MeasureCompletionTime 1": {
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
  "graph_api_benchmark_l0 SubmitGraph native recording numKernels:10 ioq 1 MeasureCompletionTime 0": {
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
    "display_name": "L0 SubmitGraph native recording in order, 10 kernels",
    "explicit_group": "SubmitGraph native recording in order, 10 kernels"
  },
  "graph_api_benchmark_l0 SubmitGraph numKernels:10 ioq 1 MeasureCompletionTime 0": {
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
  "graph_api_benchmark_l0 SubmitGraph native recording with events numKernels:10 ioq 1 MeasureCompletionTime 0": {
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
    "display_name": "L0 SubmitGraph native recording in order with events, 10 kernels",
    "explicit_group": "SubmitGraph native recording in order with events, 10 kernels"
  },
  "graph_api_benchmark_l0 SubmitGraph with events numKernels:10 ioq 1 MeasureCompletionTime 0": {
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
  "graph_api_benchmark_l0 SubmitGraph native recording numKernels:10 ioq 1 MeasureCompletionTime 1": {
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
    "display_name": "L0 SubmitGraph native recording in order with measure completion, 10 kernels",
    "explicit_group": "SubmitGraph native recording in order with measure completion, 10 kernels"
  },
  "graph_api_benchmark_l0 SubmitGraph numKernels:10 ioq 1 MeasureCompletionTime 1": {
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
  "graph_api_benchmark_l0 SubmitGraph native recording with events numKernels:10 ioq 1 MeasureCompletionTime 1": {
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
    "display_name": "L0 SubmitGraph native recording in order with measure completion with events, 10 kernels",
    "explicit_group": "SubmitGraph native recording in order with measure completion with events, 10 kernels"
  },
  "graph_api_benchmark_l0 SubmitGraph with events numKernels:10 ioq 1 MeasureCompletionTime 1": {
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
  "graph_api_benchmark_l0 SubmitGraph native recording numKernels:32 ioq 1 MeasureCompletionTime 0": {
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
    "display_name": "L0 SubmitGraph native recording in order, 32 kernels",
    "explicit_group": "SubmitGraph native recording in order, 32 kernels"
  },
  "graph_api_benchmark_l0 SubmitGraph numKernels:32 ioq 1 MeasureCompletionTime 0": {
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
  "graph_api_benchmark_l0 SubmitGraph native recording with events numKernels:32 ioq 1 MeasureCompletionTime 0": {
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
    "display_name": "L0 SubmitGraph native recording in order with events, 32 kernels",
    "explicit_group": "SubmitGraph native recording in order with events, 32 kernels"
  },
  "graph_api_benchmark_l0 SubmitGraph with events numKernels:32 ioq 1 MeasureCompletionTime 0": {
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
  "graph_api_benchmark_l0 SubmitGraph native recording numKernels:32 ioq 1 MeasureCompletionTime 1": {
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
    "display_name": "L0 SubmitGraph native recording in order with measure completion, 32 kernels",
    "explicit_group": "SubmitGraph native recording in order with measure completion, 32 kernels"
  },
  "graph_api_benchmark_l0 SubmitGraph numKernels:32 ioq 1 MeasureCompletionTime 1": {
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
  "graph_api_benchmark_l0 SubmitGraph native recording with events numKernels:32 ioq 1 MeasureCompletionTime 1": {
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
    "display_name": "L0 SubmitGraph native recording in order with measure completion with events, 32 kernels",
    "explicit_group": "SubmitGraph native recording in order with measure completion with events, 32 kernels"
  },
  "graph_api_benchmark_l0 SubmitGraph with events numKernels:32 ioq 1 MeasureCompletionTime 1": {
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
  "graph_api_benchmark_ur SubmitGraph numKernels:4 ioq 0 MeasureCompletionTime 0": {
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
  "graph_api_benchmark_ur SubmitGraph with events numKernels:4 ioq 0 MeasureCompletionTime 0": {
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
  "graph_api_benchmark_ur SubmitGraph numKernels:4 ioq 0 MeasureCompletionTime 1": {
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
  "graph_api_benchmark_ur SubmitGraph with events numKernels:4 ioq 0 MeasureCompletionTime 1": {
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
  "graph_api_benchmark_ur SubmitGraph numKernels:10 ioq 0 MeasureCompletionTime 0": {
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
  "graph_api_benchmark_ur SubmitGraph with events numKernels:10 ioq 0 MeasureCompletionTime 0": {
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
  "graph_api_benchmark_ur SubmitGraph numKernels:10 ioq 0 MeasureCompletionTime 1": {
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
  "graph_api_benchmark_ur SubmitGraph with events numKernels:10 ioq 0 MeasureCompletionTime 1": {
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
  "graph_api_benchmark_ur SubmitGraph numKernels:32 ioq 0 MeasureCompletionTime 0": {
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
  "graph_api_benchmark_ur SubmitGraph with events numKernels:32 ioq 0 MeasureCompletionTime 0": {
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
  "graph_api_benchmark_ur SubmitGraph numKernels:32 ioq 0 MeasureCompletionTime 1": {
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
  "graph_api_benchmark_ur SubmitGraph with events numKernels:32 ioq 0 MeasureCompletionTime 1": {
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
  "graph_api_benchmark_ur SubmitGraph native recording numKernels:4 ioq 1 MeasureCompletionTime 0": {
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
    "display_name": "UR SubmitGraph native recording in order, 4 kernels",
    "explicit_group": "SubmitGraph native recording in order, 4 kernels"
  },
  "graph_api_benchmark_ur SubmitGraph numKernels:4 ioq 1 MeasureCompletionTime 0": {
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
  "graph_api_benchmark_ur SubmitGraph native recording with events numKernels:4 ioq 1 MeasureCompletionTime 0": {
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
    "display_name": "UR SubmitGraph native recording in order with events, 4 kernels",
    "explicit_group": "SubmitGraph native recording in order with events, 4 kernels"
  },
  "graph_api_benchmark_ur SubmitGraph with events numKernels:4 ioq 1 MeasureCompletionTime 0": {
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
  "graph_api_benchmark_ur SubmitGraph native recording numKernels:4 ioq 1 MeasureCompletionTime 1": {
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
    "display_name": "UR SubmitGraph native recording in order with measure completion, 4 kernels",
    "explicit_group": "SubmitGraph native recording in order with measure completion, 4 kernels"
  },
  "graph_api_benchmark_ur SubmitGraph numKernels:4 ioq 1 MeasureCompletionTime 1": {
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
  "graph_api_benchmark_ur SubmitGraph native recording with events numKernels:4 ioq 1 MeasureCompletionTime 1": {
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
    "display_name": "UR SubmitGraph native recording in order with measure completion with events, 4 kernels",
    "explicit_group": "SubmitGraph native recording in order with measure completion with events, 4 kernels"
  },
  "graph_api_benchmark_ur SubmitGraph with events numKernels:4 ioq 1 MeasureCompletionTime 1": {
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
  "graph_api_benchmark_ur SubmitGraph native recording numKernels:10 ioq 1 MeasureCompletionTime 0": {
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
    "display_name": "UR SubmitGraph native recording in order, 10 kernels",
    "explicit_group": "SubmitGraph native recording in order, 10 kernels"
  },
  "graph_api_benchmark_ur SubmitGraph numKernels:10 ioq 1 MeasureCompletionTime 0": {
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
  "graph_api_benchmark_ur SubmitGraph native recording with events numKernels:10 ioq 1 MeasureCompletionTime 0": {
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
    "display_name": "UR SubmitGraph native recording in order with events, 10 kernels",
    "explicit_group": "SubmitGraph native recording in order with events, 10 kernels"
  },
  "graph_api_benchmark_ur SubmitGraph with events numKernels:10 ioq 1 MeasureCompletionTime 0": {
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
  "graph_api_benchmark_ur SubmitGraph native recording numKernels:10 ioq 1 MeasureCompletionTime 1": {
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
    "display_name": "UR SubmitGraph native recording in order with measure completion, 10 kernels",
    "explicit_group": "SubmitGraph native recording in order with measure completion, 10 kernels"
  },
  "graph_api_benchmark_ur SubmitGraph numKernels:10 ioq 1 MeasureCompletionTime 1": {
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
  "graph_api_benchmark_ur SubmitGraph native recording with events numKernels:10 ioq 1 MeasureCompletionTime 1": {
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
    "display_name": "UR SubmitGraph native recording in order with measure completion with events, 10 kernels",
    "explicit_group": "SubmitGraph native recording in order with measure completion with events, 10 kernels"
  },
  "graph_api_benchmark_ur SubmitGraph with events numKernels:10 ioq 1 MeasureCompletionTime 1": {
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
  "graph_api_benchmark_ur SubmitGraph native recording numKernels:32 ioq 1 MeasureCompletionTime 0": {
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
    "display_name": "UR SubmitGraph native recording in order, 32 kernels",
    "explicit_group": "SubmitGraph native recording in order, 32 kernels"
  },
  "graph_api_benchmark_ur SubmitGraph numKernels:32 ioq 1 MeasureCompletionTime 0": {
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
  "graph_api_benchmark_ur SubmitGraph native recording with events numKernels:32 ioq 1 MeasureCompletionTime 0": {
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
    "display_name": "UR SubmitGraph native recording in order with events, 32 kernels",
    "explicit_group": "SubmitGraph native recording in order with events, 32 kernels"
  },
  "graph_api_benchmark_ur SubmitGraph with events numKernels:32 ioq 1 MeasureCompletionTime 0": {
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
  "graph_api_benchmark_ur SubmitGraph native recording numKernels:32 ioq 1 MeasureCompletionTime 1": {
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
    "display_name": "UR SubmitGraph native recording in order with measure completion, 32 kernels",
    "explicit_group": "SubmitGraph native recording in order with measure completion, 32 kernels"
  },
  "graph_api_benchmark_ur SubmitGraph numKernels:32 ioq 1 MeasureCompletionTime 1": {
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
  "graph_api_benchmark_ur SubmitGraph native recording with events numKernels:32 ioq 1 MeasureCompletionTime 1": {
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
    "display_name": "UR SubmitGraph native recording in order with measure completion with events, 32 kernels",
    "explicit_group": "SubmitGraph native recording in order with measure completion with events, 32 kernels"
  },
  "graph_api_benchmark_ur SubmitGraph with events numKernels:32 ioq 1 MeasureCompletionTime 1": {
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
  "api_overhead_benchmark_sycl ExecImmCopy out of order from Device to Device, size 1024": {
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
    "display_name": "SYCL ExecImmCopy out of order from Device to Device, size 1024",
    "explicit_group": ""
  },
  "api_overhead_benchmark_sycl ExecImmCopy in order from Device to Host, size 1024": {
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
    "display_name": "SYCL ExecImmCopy in order from Device to Host, size 1024",
    "explicit_group": ""
  },
  "memory_benchmark_sycl QueueInOrderMemcpy from Device to Device, size 1024 CPU count": {
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
    "display_name": "SYCL QueueInOrderMemcpy from Device to Device, size 1024, CPU count",
    "explicit_group": ""
  },
  "memory_benchmark_sycl QueueInOrderMemcpy from Host to Device, size 1024 CPU count": {
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
    "display_name": "SYCL QueueInOrderMemcpy from Host to Device, size 1024, CPU count",
    "explicit_group": ""
  },
  "memory_benchmark_sycl QueueMemcpy from Device to Device, size 1024 CPU count": {
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
    "display_name": "SYCL QueueMemcpy from Device to Device, size 1024, CPU count",
    "explicit_group": ""
  },
  "api_overhead_benchmark_sycl ExecImmCopy out of order from Device to Device, size 1024 CPU count": {
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
    "display_name": "SYCL ExecImmCopy out of order from Device to Device, size 1024, CPU count",
    "explicit_group": ""
  },
  "api_overhead_benchmark_sycl ExecImmCopy in order from Device to Host, size 1024 CPU count": {
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
    "display_name": "SYCL ExecImmCopy in order from Device to Host, size 1024, CPU count",
    "explicit_group": ""
  },
  "record_and_replay_benchmark_l0 AppendCopy 1, AppendKern 10, CmdSetsInLvl 10, ForksInLvl 2, Instantiations 10, Lvls 4, Rec": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "RecordGraph large_l0",
    "explicit_group": "RecordGraph large"
  },
  "record_and_replay_benchmark_l0 AppendCopy 10, AppendKern 10, CmdSetsInLvl 10, ForksInLvl 1, Instantiations 10, Lvls 1, Rec": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "RecordGraph medium_l0",
    "explicit_group": "RecordGraph medium"
  },
  "record_and_replay_benchmark_l0 AppendCopy 0, AppendKern 1, CmdSetsInLvl 1, ForksInLvl 1, Instantiations 0, Lvls 4, Rec": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "RecordGraph short_l0",
    "explicit_group": "RecordGraph short"
  },
  "record_and_replay_benchmark_l0 AppendCopy 1, AppendKern 10, CmdSetsInLvl 10, ForksInLvl 2, Inst, Instantiations 10, Lvls 4, Rec": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "RecordGraph large_l0",
    "explicit_group": "RecordGraph large"
  },
  "record_and_replay_benchmark_l0 AppendCopy 10, AppendKern 10, CmdSetsInLvl 10, ForksInLvl 1, Inst, Instantiations 10, Lvls 1, Rec": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "RecordGraph medium_l0",
    "explicit_group": "RecordGraph medium"
  },
  "record_and_replay_benchmark_l0 AppendCopy 0, AppendKern 1, CmdSetsInLvl 1, ForksInLvl 1, Inst, Instantiations 0, Lvls 4, Rec": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "RecordGraph short_l0",
    "explicit_group": "RecordGraph short"
  },
  "record_and_replay_benchmark_l0 AppendCopy 1, AppendKern 10, CmdSetsInLvl 10, ForksInLvl 2, Instantiations 10, Lvls 4, Rec, emulate": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "RecordGraph large_l0",
    "explicit_group": "RecordGraph large"
  },
  "record_and_replay_benchmark_l0 AppendCopy 10, AppendKern 10, CmdSetsInLvl 10, ForksInLvl 1, Instantiations 10, Lvls 1, Rec, emulate": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "RecordGraph medium_l0",
    "explicit_group": "RecordGraph medium"
  },
  "record_and_replay_benchmark_l0 AppendCopy 0, AppendKern 1, CmdSetsInLvl 1, ForksInLvl 1, Instantiations 0, Lvls 4, Rec, emulate": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "RecordGraph short_l0",
    "explicit_group": "RecordGraph short"
  },
  "record_and_replay_benchmark_l0 AppendCopy 1, AppendKern 10, CmdSetsInLvl 10, ForksInLvl 2, Inst, Instantiations 10, Lvls 4, Rec, emulate": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "RecordGraph large_l0",
    "explicit_group": "RecordGraph large"
  },
  "record_and_replay_benchmark_l0 AppendCopy 10, AppendKern 10, CmdSetsInLvl 10, ForksInLvl 1, Inst, Instantiations 10, Lvls 1, Rec, emulate": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "RecordGraph medium_l0",
    "explicit_group": "RecordGraph medium"
  },
  "record_and_replay_benchmark_l0 AppendCopy 0, AppendKern 1, CmdSetsInLvl 1, ForksInLvl 1, Inst, Instantiations 0, Lvls 4, Rec, emulate": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "RecordGraph short_l0",
    "explicit_group": "RecordGraph short"
  },
  "torch_benchmark_syclpreview KernelSubmitSingleQueue KernelDataType Int32, KernelWGCount 4096, KernelWGSize 512": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitSingleQueue Int32Large",
    "explicit_group": "KernelSubmitSingleQueue Int32Large"
  },
  "torch_benchmark_syclpreview KernelSubmitSingleQueue KernelDataType Int32, KernelWGCount 512, KernelWGSize 256": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitSingleQueue Int32Medium",
    "explicit_group": "KernelSubmitSingleQueue Int32Medium"
  },
  "torch_benchmark_syclpreview KernelSubmitSingleQueue KernelDataType Int32, KernelWGCount 256, KernelWGSize 128": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitSingleQueue Int32Small",
    "explicit_group": "KernelSubmitSingleQueue Int32Small"
  },
  "torch_benchmark_syclpreview KernelSubmitSingleQueue KernelDataType Mixed, KernelWGCount 4096, KernelWGSize 512": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitSingleQueue MixedLarge",
    "explicit_group": "KernelSubmitSingleQueue MixedLarge"
  },
  "torch_benchmark_syclpreview KernelSubmitSingleQueue KernelDataType Mixed, KernelWGCount 512, KernelWGSize 256": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitSingleQueue MixedMedium",
    "explicit_group": "KernelSubmitSingleQueue MixedMedium"
  },
  "torch_benchmark_syclpreview KernelSubmitSingleQueue KernelDataType Mixed, KernelWGCount 256, KernelWGSize 128": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitSingleQueue MixedSmall",
    "explicit_group": "KernelSubmitSingleQueue MixedSmall"
  },
  "torch_benchmark_syclpreview KernelSubmitSingleQueue KernelDataType Int32, KernelWGCount 4096, KernelWGSize 512 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitSingleQueue Int32Large, CPU count",
    "explicit_group": "KernelSubmitSingleQueue Int32Large, CPU count"
  },
  "torch_benchmark_syclpreview KernelSubmitSingleQueue KernelDataType Int32, KernelWGCount 512, KernelWGSize 256 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitSingleQueue Int32Medium, CPU count",
    "explicit_group": "KernelSubmitSingleQueue Int32Medium, CPU count"
  },
  "torch_benchmark_syclpreview KernelSubmitSingleQueue KernelDataType Int32, KernelWGCount 256, KernelWGSize 128 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitSingleQueue Int32Small, CPU count",
    "explicit_group": "KernelSubmitSingleQueue Int32Small, CPU count"
  },
  "torch_benchmark_syclpreview KernelSubmitSingleQueue KernelDataType Mixed, KernelWGCount 4096, KernelWGSize 512 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitSingleQueue MixedLarge, CPU count",
    "explicit_group": "KernelSubmitSingleQueue MixedLarge, CPU count"
  },
  "torch_benchmark_syclpreview KernelSubmitSingleQueue KernelDataType Mixed, KernelWGCount 512, KernelWGSize 256 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitSingleQueue MixedMedium, CPU count",
    "explicit_group": "KernelSubmitSingleQueue MixedMedium, CPU count"
  },
  "torch_benchmark_syclpreview KernelSubmitSingleQueue KernelDataType Mixed, KernelWGCount 256, KernelWGSize 128 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitSingleQueue MixedSmall, CPU count",
    "explicit_group": "KernelSubmitSingleQueue MixedSmall, CPU count"
  },
  "torch_benchmark_sycl KernelSubmitSingleQueue KernelDataType Int32, KernelWGCount 4096, KernelWGSize 512": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitSingleQueue Int32Large",
    "explicit_group": "KernelSubmitSingleQueue Int32Large"
  },
  "torch_benchmark_sycl KernelSubmitSingleQueue KernelDataType Int32, KernelWGCount 512, KernelWGSize 256": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitSingleQueue Int32Medium",
    "explicit_group": "KernelSubmitSingleQueue Int32Medium"
  },
  "torch_benchmark_sycl KernelSubmitSingleQueue KernelDataType Int32, KernelWGCount 256, KernelWGSize 128": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitSingleQueue Int32Small",
    "explicit_group": "KernelSubmitSingleQueue Int32Small"
  },
  "torch_benchmark_sycl KernelSubmitSingleQueue KernelDataType Mixed, KernelWGCount 4096, KernelWGSize 512": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitSingleQueue MixedLarge",
    "explicit_group": "KernelSubmitSingleQueue MixedLarge"
  },
  "torch_benchmark_sycl KernelSubmitSingleQueue KernelDataType Mixed, KernelWGCount 512, KernelWGSize 256": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitSingleQueue MixedMedium",
    "explicit_group": "KernelSubmitSingleQueue MixedMedium"
  },
  "torch_benchmark_sycl KernelSubmitSingleQueue KernelDataType Mixed, KernelWGCount 256, KernelWGSize 128": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitSingleQueue MixedSmall",
    "explicit_group": "KernelSubmitSingleQueue MixedSmall"
  },
  "torch_benchmark_sycl KernelSubmitSingleQueue KernelDataType Int32, KernelWGCount 4096, KernelWGSize 512 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitSingleQueue Int32Large, CPU count",
    "explicit_group": "KernelSubmitSingleQueue Int32Large, CPU count"
  },
  "torch_benchmark_sycl KernelSubmitSingleQueue KernelDataType Int32, KernelWGCount 512, KernelWGSize 256 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitSingleQueue Int32Medium, CPU count",
    "explicit_group": "KernelSubmitSingleQueue Int32Medium, CPU count"
  },
  "torch_benchmark_sycl KernelSubmitSingleQueue KernelDataType Int32, KernelWGCount 256, KernelWGSize 128 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitSingleQueue Int32Small, CPU count",
    "explicit_group": "KernelSubmitSingleQueue Int32Small, CPU count"
  },
  "torch_benchmark_sycl KernelSubmitSingleQueue KernelDataType Mixed, KernelWGCount 4096, KernelWGSize 512 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitSingleQueue MixedLarge, CPU count",
    "explicit_group": "KernelSubmitSingleQueue MixedLarge, CPU count"
  },
  "torch_benchmark_sycl KernelSubmitSingleQueue KernelDataType Mixed, KernelWGCount 512, KernelWGSize 256 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitSingleQueue MixedMedium, CPU count",
    "explicit_group": "KernelSubmitSingleQueue MixedMedium, CPU count"
  },
  "torch_benchmark_sycl KernelSubmitSingleQueue KernelDataType Mixed, KernelWGCount 256, KernelWGSize 128 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitSingleQueue MixedSmall, CPU count",
    "explicit_group": "KernelSubmitSingleQueue MixedSmall, CPU count"
  },
  "torch_benchmark_l0 KernelSubmitSingleQueue KernelDataType Int32, KernelWGCount 4096, KernelWGSize 512": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitSingleQueue Int32Large",
    "explicit_group": "KernelSubmitSingleQueue Int32Large"
  },
  "torch_benchmark_l0 KernelSubmitSingleQueue KernelDataType Int32, KernelWGCount 512, KernelWGSize 256": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitSingleQueue Int32Medium",
    "explicit_group": "KernelSubmitSingleQueue Int32Medium"
  },
  "torch_benchmark_l0 KernelSubmitSingleQueue KernelDataType Int32, KernelWGCount 256, KernelWGSize 128": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitSingleQueue Int32Small",
    "explicit_group": "KernelSubmitSingleQueue Int32Small"
  },
  "torch_benchmark_l0 KernelSubmitSingleQueue KernelDataType Mixed, KernelWGCount 4096, KernelWGSize 512": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitSingleQueue MixedLarge",
    "explicit_group": "KernelSubmitSingleQueue MixedLarge"
  },
  "torch_benchmark_l0 KernelSubmitSingleQueue KernelDataType Mixed, KernelWGCount 512, KernelWGSize 256": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitSingleQueue MixedMedium",
    "explicit_group": "KernelSubmitSingleQueue MixedMedium"
  },
  "torch_benchmark_l0 KernelSubmitSingleQueue KernelDataType Mixed, KernelWGCount 256, KernelWGSize 128": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitSingleQueue MixedSmall",
    "explicit_group": "KernelSubmitSingleQueue MixedSmall"
  },
  "torch_benchmark_l0 KernelSubmitSingleQueue KernelDataType Int32, KernelWGCount 4096, KernelWGSize 512 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitSingleQueue Int32Large, CPU count",
    "explicit_group": "KernelSubmitSingleQueue Int32Large, CPU count"
  },
  "torch_benchmark_l0 KernelSubmitSingleQueue KernelDataType Int32, KernelWGCount 512, KernelWGSize 256 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitSingleQueue Int32Medium, CPU count",
    "explicit_group": "KernelSubmitSingleQueue Int32Medium, CPU count"
  },
  "torch_benchmark_l0 KernelSubmitSingleQueue KernelDataType Int32, KernelWGCount 256, KernelWGSize 128 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitSingleQueue Int32Small, CPU count",
    "explicit_group": "KernelSubmitSingleQueue Int32Small, CPU count"
  },
  "torch_benchmark_l0 KernelSubmitSingleQueue KernelDataType Mixed, KernelWGCount 4096, KernelWGSize 512 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitSingleQueue MixedLarge, CPU count",
    "explicit_group": "KernelSubmitSingleQueue MixedLarge, CPU count"
  },
  "torch_benchmark_l0 KernelSubmitSingleQueue KernelDataType Mixed, KernelWGCount 512, KernelWGSize 256 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitSingleQueue MixedMedium, CPU count",
    "explicit_group": "KernelSubmitSingleQueue MixedMedium, CPU count"
  },
  "torch_benchmark_l0 KernelSubmitSingleQueue KernelDataType Mixed, KernelWGCount 256, KernelWGSize 128 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitSingleQueue MixedSmall, CPU count",
    "explicit_group": "KernelSubmitSingleQueue MixedSmall, CPU count"
  },
  "torch_benchmark_syclpreview KernelSubmitMultiQueue KernelsPerQueue 20, MeasureCompletionTime 0": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitMultiQueue large",
    "explicit_group": "KernelSubmitMultiQueue large"
  },
  "torch_benchmark_syclpreview KernelSubmitMultiQueue KernelsPerQueue 10, MeasureCompletionTime 0": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitMultiQueue medium",
    "explicit_group": "KernelSubmitMultiQueue medium"
  },
  "torch_benchmark_syclpreview KernelSubmitMultiQueue KernelsPerQueue 4, MeasureCompletionTime 0": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitMultiQueue small",
    "explicit_group": "KernelSubmitMultiQueue small"
  },
  "torch_benchmark_syclpreview KernelSubmitMultiQueue KernelsPerQueue 20, MeasureCompletionTime 1": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitMultiQueue large with measure completion",
    "explicit_group": "KernelSubmitMultiQueue large with measure completion"
  },
  "torch_benchmark_syclpreview KernelSubmitMultiQueue KernelsPerQueue 10, MeasureCompletionTime 1": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitMultiQueue medium with measure completion",
    "explicit_group": "KernelSubmitMultiQueue medium with measure completion"
  },
  "torch_benchmark_syclpreview KernelSubmitMultiQueue KernelsPerQueue 4, MeasureCompletionTime 1": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitMultiQueue small with measure completion",
    "explicit_group": "KernelSubmitMultiQueue small with measure completion"
  },
  "torch_benchmark_syclpreview KernelSubmitMultiQueue KernelsPerQueue 20, MeasureCompletionTime 0 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitMultiQueue large, CPU count",
    "explicit_group": "KernelSubmitMultiQueue large, CPU count"
  },
  "torch_benchmark_syclpreview KernelSubmitMultiQueue KernelsPerQueue 10, MeasureCompletionTime 0 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitMultiQueue medium, CPU count",
    "explicit_group": "KernelSubmitMultiQueue medium, CPU count"
  },
  "torch_benchmark_syclpreview KernelSubmitMultiQueue KernelsPerQueue 4, MeasureCompletionTime 0 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitMultiQueue small, CPU count",
    "explicit_group": "KernelSubmitMultiQueue small, CPU count"
  },
  "torch_benchmark_syclpreview KernelSubmitMultiQueue KernelsPerQueue 20, MeasureCompletionTime 1 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitMultiQueue large with measure completion, CPU count",
    "explicit_group": "KernelSubmitMultiQueue large with measure completion, CPU count"
  },
  "torch_benchmark_syclpreview KernelSubmitMultiQueue KernelsPerQueue 10, MeasureCompletionTime 1 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitMultiQueue medium with measure completion, CPU count",
    "explicit_group": "KernelSubmitMultiQueue medium with measure completion, CPU count"
  },
  "torch_benchmark_syclpreview KernelSubmitMultiQueue KernelsPerQueue 4, MeasureCompletionTime 1 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitMultiQueue small with measure completion, CPU count",
    "explicit_group": "KernelSubmitMultiQueue small with measure completion, CPU count"
  },
  "torch_benchmark_sycl KernelSubmitMultiQueue KernelsPerQueue 20, MeasureCompletionTime 0": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitMultiQueue large",
    "explicit_group": "KernelSubmitMultiQueue large"
  },
  "torch_benchmark_sycl KernelSubmitMultiQueue KernelsPerQueue 10, MeasureCompletionTime 0": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitMultiQueue medium",
    "explicit_group": "KernelSubmitMultiQueue medium"
  },
  "torch_benchmark_sycl KernelSubmitMultiQueue KernelsPerQueue 4, MeasureCompletionTime 0": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitMultiQueue small",
    "explicit_group": "KernelSubmitMultiQueue small"
  },
  "torch_benchmark_sycl KernelSubmitMultiQueue KernelsPerQueue 20, MeasureCompletionTime 1": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitMultiQueue large with measure completion",
    "explicit_group": "KernelSubmitMultiQueue large with measure completion"
  },
  "torch_benchmark_sycl KernelSubmitMultiQueue KernelsPerQueue 10, MeasureCompletionTime 1": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitMultiQueue medium with measure completion",
    "explicit_group": "KernelSubmitMultiQueue medium with measure completion"
  },
  "torch_benchmark_sycl KernelSubmitMultiQueue KernelsPerQueue 4, MeasureCompletionTime 1": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitMultiQueue small with measure completion",
    "explicit_group": "KernelSubmitMultiQueue small with measure completion"
  },
  "torch_benchmark_sycl KernelSubmitMultiQueue KernelsPerQueue 20, MeasureCompletionTime 0 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitMultiQueue large, CPU count",
    "explicit_group": "KernelSubmitMultiQueue large, CPU count"
  },
  "torch_benchmark_sycl KernelSubmitMultiQueue KernelsPerQueue 10, MeasureCompletionTime 0 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitMultiQueue medium, CPU count",
    "explicit_group": "KernelSubmitMultiQueue medium, CPU count"
  },
  "torch_benchmark_sycl KernelSubmitMultiQueue KernelsPerQueue 4, MeasureCompletionTime 0 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitMultiQueue small, CPU count",
    "explicit_group": "KernelSubmitMultiQueue small, CPU count"
  },
  "torch_benchmark_sycl KernelSubmitMultiQueue KernelsPerQueue 20, MeasureCompletionTime 1 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitMultiQueue large with measure completion, CPU count",
    "explicit_group": "KernelSubmitMultiQueue large with measure completion, CPU count"
  },
  "torch_benchmark_sycl KernelSubmitMultiQueue KernelsPerQueue 10, MeasureCompletionTime 1 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitMultiQueue medium with measure completion, CPU count",
    "explicit_group": "KernelSubmitMultiQueue medium with measure completion, CPU count"
  },
  "torch_benchmark_sycl KernelSubmitMultiQueue KernelsPerQueue 4, MeasureCompletionTime 1 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitMultiQueue small with measure completion, CPU count",
    "explicit_group": "KernelSubmitMultiQueue small with measure completion, CPU count"
  },
  "torch_benchmark_l0 KernelSubmitMultiQueue KernelsPerQueue 20, MeasureCompletionTime 0": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitMultiQueue large",
    "explicit_group": "KernelSubmitMultiQueue large"
  },
  "torch_benchmark_l0 KernelSubmitMultiQueue KernelsPerQueue 10, MeasureCompletionTime 0": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitMultiQueue medium",
    "explicit_group": "KernelSubmitMultiQueue medium"
  },
  "torch_benchmark_l0 KernelSubmitMultiQueue KernelsPerQueue 4, MeasureCompletionTime 0": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitMultiQueue small",
    "explicit_group": "KernelSubmitMultiQueue small"
  },
  "torch_benchmark_l0 KernelSubmitMultiQueue KernelsPerQueue 20, MeasureCompletionTime 1": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitMultiQueue large with measure completion",
    "explicit_group": "KernelSubmitMultiQueue large with measure completion"
  },
  "torch_benchmark_l0 KernelSubmitMultiQueue KernelsPerQueue 10, MeasureCompletionTime 1": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitMultiQueue medium with measure completion",
    "explicit_group": "KernelSubmitMultiQueue medium with measure completion"
  },
  "torch_benchmark_l0 KernelSubmitMultiQueue KernelsPerQueue 4, MeasureCompletionTime 1": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitMultiQueue small with measure completion",
    "explicit_group": "KernelSubmitMultiQueue small with measure completion"
  },
  "torch_benchmark_l0 KernelSubmitMultiQueue KernelsPerQueue 20, MeasureCompletionTime 0 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitMultiQueue large, CPU count",
    "explicit_group": "KernelSubmitMultiQueue large, CPU count"
  },
  "torch_benchmark_l0 KernelSubmitMultiQueue KernelsPerQueue 10, MeasureCompletionTime 0 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitMultiQueue medium, CPU count",
    "explicit_group": "KernelSubmitMultiQueue medium, CPU count"
  },
  "torch_benchmark_l0 KernelSubmitMultiQueue KernelsPerQueue 4, MeasureCompletionTime 0 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitMultiQueue small, CPU count",
    "explicit_group": "KernelSubmitMultiQueue small, CPU count"
  },
  "torch_benchmark_l0 KernelSubmitMultiQueue KernelsPerQueue 20, MeasureCompletionTime 1 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitMultiQueue large with measure completion, CPU count",
    "explicit_group": "KernelSubmitMultiQueue large with measure completion, CPU count"
  },
  "torch_benchmark_l0 KernelSubmitMultiQueue KernelsPerQueue 10, MeasureCompletionTime 1 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitMultiQueue medium with measure completion, CPU count",
    "explicit_group": "KernelSubmitMultiQueue medium with measure completion, CPU count"
  },
  "torch_benchmark_l0 KernelSubmitMultiQueue KernelsPerQueue 4, MeasureCompletionTime 1 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitMultiQueue small with measure completion, CPU count",
    "explicit_group": "KernelSubmitMultiQueue small with measure completion, CPU count"
  },
  "torch_benchmark_syclpreview KernelSubmitSlmSize MeasureCompletionTime 0, SlmNum 1": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitSlmSize small",
    "explicit_group": "KernelSubmitSlmSize small"
  },
  "torch_benchmark_syclpreview KernelSubmitSlmSize MeasureCompletionTime 0, SlmNum 1024": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitSlmSize medium",
    "explicit_group": "KernelSubmitSlmSize medium"
  },
  "torch_benchmark_syclpreview KernelSubmitSlmSize MeasureCompletionTime 0, SlmNum 16384": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitSlmSize large",
    "explicit_group": "KernelSubmitSlmSize large"
  },
  "torch_benchmark_syclpreview KernelSubmitSlmSize MeasureCompletionTime 1, SlmNum 1": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitSlmSize small with measure completion",
    "explicit_group": "KernelSubmitSlmSize small with measure completion"
  },
  "torch_benchmark_syclpreview KernelSubmitSlmSize MeasureCompletionTime 1, SlmNum 1024": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitSlmSize medium with measure completion",
    "explicit_group": "KernelSubmitSlmSize medium with measure completion"
  },
  "torch_benchmark_syclpreview KernelSubmitSlmSize MeasureCompletionTime 1, SlmNum 16384": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitSlmSize large with measure completion",
    "explicit_group": "KernelSubmitSlmSize large with measure completion"
  },
  "torch_benchmark_syclpreview KernelSubmitSlmSize MeasureCompletionTime 0, SlmNum 1 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitSlmSize small, CPU count",
    "explicit_group": "KernelSubmitSlmSize small, CPU count"
  },
  "torch_benchmark_syclpreview KernelSubmitSlmSize MeasureCompletionTime 0, SlmNum 1024 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitSlmSize medium, CPU count",
    "explicit_group": "KernelSubmitSlmSize medium, CPU count"
  },
  "torch_benchmark_syclpreview KernelSubmitSlmSize MeasureCompletionTime 0, SlmNum 16384 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitSlmSize large, CPU count",
    "explicit_group": "KernelSubmitSlmSize large, CPU count"
  },
  "torch_benchmark_syclpreview KernelSubmitSlmSize MeasureCompletionTime 1, SlmNum 1 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitSlmSize small with measure completion, CPU count",
    "explicit_group": "KernelSubmitSlmSize small with measure completion, CPU count"
  },
  "torch_benchmark_syclpreview KernelSubmitSlmSize MeasureCompletionTime 1, SlmNum 1024 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitSlmSize medium with measure completion, CPU count",
    "explicit_group": "KernelSubmitSlmSize medium with measure completion, CPU count"
  },
  "torch_benchmark_syclpreview KernelSubmitSlmSize MeasureCompletionTime 1, SlmNum 16384 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitSlmSize large with measure completion, CPU count",
    "explicit_group": "KernelSubmitSlmSize large with measure completion, CPU count"
  },
  "torch_benchmark_sycl KernelSubmitSlmSize MeasureCompletionTime 0, SlmNum 1": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitSlmSize small",
    "explicit_group": "KernelSubmitSlmSize small"
  },
  "torch_benchmark_sycl KernelSubmitSlmSize MeasureCompletionTime 0, SlmNum 1024": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitSlmSize medium",
    "explicit_group": "KernelSubmitSlmSize medium"
  },
  "torch_benchmark_sycl KernelSubmitSlmSize MeasureCompletionTime 0, SlmNum 16384": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitSlmSize large",
    "explicit_group": "KernelSubmitSlmSize large"
  },
  "torch_benchmark_sycl KernelSubmitSlmSize MeasureCompletionTime 1, SlmNum 1": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitSlmSize small with measure completion",
    "explicit_group": "KernelSubmitSlmSize small with measure completion"
  },
  "torch_benchmark_sycl KernelSubmitSlmSize MeasureCompletionTime 1, SlmNum 1024": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitSlmSize medium with measure completion",
    "explicit_group": "KernelSubmitSlmSize medium with measure completion"
  },
  "torch_benchmark_sycl KernelSubmitSlmSize MeasureCompletionTime 1, SlmNum 16384": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitSlmSize large with measure completion",
    "explicit_group": "KernelSubmitSlmSize large with measure completion"
  },
  "torch_benchmark_sycl KernelSubmitSlmSize MeasureCompletionTime 0, SlmNum 1 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitSlmSize small, CPU count",
    "explicit_group": "KernelSubmitSlmSize small, CPU count"
  },
  "torch_benchmark_sycl KernelSubmitSlmSize MeasureCompletionTime 0, SlmNum 1024 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitSlmSize medium, CPU count",
    "explicit_group": "KernelSubmitSlmSize medium, CPU count"
  },
  "torch_benchmark_sycl KernelSubmitSlmSize MeasureCompletionTime 0, SlmNum 16384 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitSlmSize large, CPU count",
    "explicit_group": "KernelSubmitSlmSize large, CPU count"
  },
  "torch_benchmark_sycl KernelSubmitSlmSize MeasureCompletionTime 1, SlmNum 1 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitSlmSize small with measure completion, CPU count",
    "explicit_group": "KernelSubmitSlmSize small with measure completion, CPU count"
  },
  "torch_benchmark_sycl KernelSubmitSlmSize MeasureCompletionTime 1, SlmNum 1024 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitSlmSize medium with measure completion, CPU count",
    "explicit_group": "KernelSubmitSlmSize medium with measure completion, CPU count"
  },
  "torch_benchmark_sycl KernelSubmitSlmSize MeasureCompletionTime 1, SlmNum 16384 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitSlmSize large with measure completion, CPU count",
    "explicit_group": "KernelSubmitSlmSize large with measure completion, CPU count"
  },
  "torch_benchmark_l0 KernelSubmitSlmSize MeasureCompletionTime 0, SlmNum 1": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitSlmSize small",
    "explicit_group": "KernelSubmitSlmSize small"
  },
  "torch_benchmark_l0 KernelSubmitSlmSize MeasureCompletionTime 0, SlmNum 1024": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitSlmSize medium",
    "explicit_group": "KernelSubmitSlmSize medium"
  },
  "torch_benchmark_l0 KernelSubmitSlmSize MeasureCompletionTime 0, SlmNum 16384": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitSlmSize large",
    "explicit_group": "KernelSubmitSlmSize large"
  },
  "torch_benchmark_l0 KernelSubmitSlmSize MeasureCompletionTime 1, SlmNum 1": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitSlmSize small with measure completion",
    "explicit_group": "KernelSubmitSlmSize small with measure completion"
  },
  "torch_benchmark_l0 KernelSubmitSlmSize MeasureCompletionTime 1, SlmNum 1024": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitSlmSize medium with measure completion",
    "explicit_group": "KernelSubmitSlmSize medium with measure completion"
  },
  "torch_benchmark_l0 KernelSubmitSlmSize MeasureCompletionTime 1, SlmNum 16384": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitSlmSize large with measure completion",
    "explicit_group": "KernelSubmitSlmSize large with measure completion"
  },
  "torch_benchmark_l0 KernelSubmitSlmSize MeasureCompletionTime 0, SlmNum 1 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitSlmSize small, CPU count",
    "explicit_group": "KernelSubmitSlmSize small, CPU count"
  },
  "torch_benchmark_l0 KernelSubmitSlmSize MeasureCompletionTime 0, SlmNum 1024 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitSlmSize medium, CPU count",
    "explicit_group": "KernelSubmitSlmSize medium, CPU count"
  },
  "torch_benchmark_l0 KernelSubmitSlmSize MeasureCompletionTime 0, SlmNum 16384 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitSlmSize large, CPU count",
    "explicit_group": "KernelSubmitSlmSize large, CPU count"
  },
  "torch_benchmark_l0 KernelSubmitSlmSize MeasureCompletionTime 1, SlmNum 1 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitSlmSize small with measure completion, CPU count",
    "explicit_group": "KernelSubmitSlmSize small with measure completion, CPU count"
  },
  "torch_benchmark_l0 KernelSubmitSlmSize MeasureCompletionTime 1, SlmNum 1024 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitSlmSize medium with measure completion, CPU count",
    "explicit_group": "KernelSubmitSlmSize medium with measure completion, CPU count"
  },
  "torch_benchmark_l0 KernelSubmitSlmSize MeasureCompletionTime 1, SlmNum 16384 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitSlmSize large with measure completion, CPU count",
    "explicit_group": "KernelSubmitSlmSize large with measure completion, CPU count"
  },
  "torch_benchmark_syclpreview KernelSubmitMemoryReuse KernelBatchSize 4096, KernelDataType Int32, Profiling 0, UseEvents 0": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitMemoryReuse Int32Large",
    "explicit_group": "KernelSubmitMemoryReuse Int32Large"
  },
  "torch_benchmark_syclpreview KernelSubmitMemoryReuse KernelBatchSize 512, KernelDataType Int32, Profiling 0, UseEvents 0": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitMemoryReuse Int32Medium",
    "explicit_group": "KernelSubmitMemoryReuse Int32Medium"
  },
  "torch_benchmark_syclpreview KernelSubmitMemoryReuse KernelBatchSize 4096, KernelDataType Float, Profiling 0, UseEvents 0": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitMemoryReuse FloatLarge",
    "explicit_group": "KernelSubmitMemoryReuse FloatLarge"
  },
  "torch_benchmark_syclpreview KernelSubmitMemoryReuse KernelBatchSize 512, KernelDataType Float, Profiling 0, UseEvents 0": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitMemoryReuse FloatMedium",
    "explicit_group": "KernelSubmitMemoryReuse FloatMedium"
  },
  "torch_benchmark_syclpreview KernelSubmitMemoryReuse KernelBatchSize 4096, KernelDataType Int32, Profiling 0, UseEvents 0 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitMemoryReuse Int32Large, CPU count",
    "explicit_group": "KernelSubmitMemoryReuse Int32Large, CPU count"
  },
  "torch_benchmark_syclpreview KernelSubmitMemoryReuse KernelBatchSize 512, KernelDataType Int32, Profiling 0, UseEvents 0 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitMemoryReuse Int32Medium, CPU count",
    "explicit_group": "KernelSubmitMemoryReuse Int32Medium, CPU count"
  },
  "torch_benchmark_syclpreview KernelSubmitMemoryReuse KernelBatchSize 4096, KernelDataType Float, Profiling 0, UseEvents 0 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitMemoryReuse FloatLarge, CPU count",
    "explicit_group": "KernelSubmitMemoryReuse FloatLarge, CPU count"
  },
  "torch_benchmark_syclpreview KernelSubmitMemoryReuse KernelBatchSize 512, KernelDataType Float, Profiling 0, UseEvents 0 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitMemoryReuse FloatMedium, CPU count",
    "explicit_group": "KernelSubmitMemoryReuse FloatMedium, CPU count"
  },
  "torch_benchmark_sycl KernelSubmitMemoryReuse KernelBatchSize 4096, KernelDataType Int32, Profiling 0, UseEvents 0": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitMemoryReuse Int32Large",
    "explicit_group": "KernelSubmitMemoryReuse Int32Large"
  },
  "torch_benchmark_sycl KernelSubmitMemoryReuse KernelBatchSize 512, KernelDataType Int32, Profiling 0, UseEvents 0": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitMemoryReuse Int32Medium",
    "explicit_group": "KernelSubmitMemoryReuse Int32Medium"
  },
  "torch_benchmark_sycl KernelSubmitMemoryReuse KernelBatchSize 4096, KernelDataType Float, Profiling 0, UseEvents 0": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitMemoryReuse FloatLarge",
    "explicit_group": "KernelSubmitMemoryReuse FloatLarge"
  },
  "torch_benchmark_sycl KernelSubmitMemoryReuse KernelBatchSize 512, KernelDataType Float, Profiling 0, UseEvents 0": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitMemoryReuse FloatMedium",
    "explicit_group": "KernelSubmitMemoryReuse FloatMedium"
  },
  "torch_benchmark_sycl KernelSubmitMemoryReuse KernelBatchSize 4096, KernelDataType Int32, Profiling 0, UseEvents 0 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitMemoryReuse Int32Large, CPU count",
    "explicit_group": "KernelSubmitMemoryReuse Int32Large, CPU count"
  },
  "torch_benchmark_sycl KernelSubmitMemoryReuse KernelBatchSize 512, KernelDataType Int32, Profiling 0, UseEvents 0 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitMemoryReuse Int32Medium, CPU count",
    "explicit_group": "KernelSubmitMemoryReuse Int32Medium, CPU count"
  },
  "torch_benchmark_sycl KernelSubmitMemoryReuse KernelBatchSize 4096, KernelDataType Float, Profiling 0, UseEvents 0 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitMemoryReuse FloatLarge, CPU count",
    "explicit_group": "KernelSubmitMemoryReuse FloatLarge, CPU count"
  },
  "torch_benchmark_sycl KernelSubmitMemoryReuse KernelBatchSize 512, KernelDataType Float, Profiling 0, UseEvents 0 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitMemoryReuse FloatMedium, CPU count",
    "explicit_group": "KernelSubmitMemoryReuse FloatMedium, CPU count"
  },
  "torch_benchmark_l0 KernelSubmitMemoryReuse KernelBatchSize 4096, KernelDataType Int32, Profiling 0, UseEvents 0": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitMemoryReuse Int32Large",
    "explicit_group": "KernelSubmitMemoryReuse Int32Large"
  },
  "torch_benchmark_l0 KernelSubmitMemoryReuse KernelBatchSize 512, KernelDataType Int32, Profiling 0, UseEvents 0": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitMemoryReuse Int32Medium",
    "explicit_group": "KernelSubmitMemoryReuse Int32Medium"
  },
  "torch_benchmark_l0 KernelSubmitMemoryReuse KernelBatchSize 4096, KernelDataType Float, Profiling 0, UseEvents 0": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitMemoryReuse FloatLarge",
    "explicit_group": "KernelSubmitMemoryReuse FloatLarge"
  },
  "torch_benchmark_l0 KernelSubmitMemoryReuse KernelBatchSize 512, KernelDataType Float, Profiling 0, UseEvents 0": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitMemoryReuse FloatMedium",
    "explicit_group": "KernelSubmitMemoryReuse FloatMedium"
  },
  "torch_benchmark_l0 KernelSubmitMemoryReuse KernelBatchSize 4096, KernelDataType Int32, Profiling 0, UseEvents 0 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitMemoryReuse Int32Large, CPU count",
    "explicit_group": "KernelSubmitMemoryReuse Int32Large, CPU count"
  },
  "torch_benchmark_l0 KernelSubmitMemoryReuse KernelBatchSize 512, KernelDataType Int32, Profiling 0, UseEvents 0 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitMemoryReuse Int32Medium, CPU count",
    "explicit_group": "KernelSubmitMemoryReuse Int32Medium, CPU count"
  },
  "torch_benchmark_l0 KernelSubmitMemoryReuse KernelBatchSize 4096, KernelDataType Float, Profiling 0, UseEvents 0 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitMemoryReuse FloatLarge, CPU count",
    "explicit_group": "KernelSubmitMemoryReuse FloatLarge, CPU count"
  },
  "torch_benchmark_l0 KernelSubmitMemoryReuse KernelBatchSize 512, KernelDataType Float, Profiling 0, UseEvents 0 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitMemoryReuse FloatMedium, CPU count",
    "explicit_group": "KernelSubmitMemoryReuse FloatMedium, CPU count"
  },
  "torch_benchmark_syclpreview KernelSubmitLinearKernelSize KernelSize 32": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitLinearKernelSize array32",
    "explicit_group": "KernelSubmitLinearKernelSize array32"
  },
  "torch_benchmark_syclpreview KernelSubmitLinearKernelSize KernelSize 128": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitLinearKernelSize array128",
    "explicit_group": "KernelSubmitLinearKernelSize array128"
  },
  "torch_benchmark_syclpreview KernelSubmitLinearKernelSize KernelSize 512": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitLinearKernelSize array512",
    "explicit_group": "KernelSubmitLinearKernelSize array512"
  },
  "torch_benchmark_syclpreview KernelSubmitLinearKernelSize KernelSize 1024": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitLinearKernelSize array1024",
    "explicit_group": "KernelSubmitLinearKernelSize array1024"
  },
  "torch_benchmark_syclpreview KernelSubmitLinearKernelSize KernelSize 5120": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitLinearKernelSize array5120",
    "explicit_group": "KernelSubmitLinearKernelSize array5120"
  },
  "torch_benchmark_syclpreview KernelSubmitLinearKernelSize KernelSize 32 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitLinearKernelSize array32, CPU count",
    "explicit_group": "KernelSubmitLinearKernelSize array32, CPU count"
  },
  "torch_benchmark_syclpreview KernelSubmitLinearKernelSize KernelSize 128 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitLinearKernelSize array128, CPU count",
    "explicit_group": "KernelSubmitLinearKernelSize array128, CPU count"
  },
  "torch_benchmark_syclpreview KernelSubmitLinearKernelSize KernelSize 512 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitLinearKernelSize array512, CPU count",
    "explicit_group": "KernelSubmitLinearKernelSize array512, CPU count"
  },
  "torch_benchmark_syclpreview KernelSubmitLinearKernelSize KernelSize 1024 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitLinearKernelSize array1024, CPU count",
    "explicit_group": "KernelSubmitLinearKernelSize array1024, CPU count"
  },
  "torch_benchmark_syclpreview KernelSubmitLinearKernelSize KernelSize 5120 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitLinearKernelSize array5120, CPU count",
    "explicit_group": "KernelSubmitLinearKernelSize array5120, CPU count"
  },
  "torch_benchmark_sycl KernelSubmitLinearKernelSize KernelSize 32": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitLinearKernelSize array32",
    "explicit_group": "KernelSubmitLinearKernelSize array32"
  },
  "torch_benchmark_sycl KernelSubmitLinearKernelSize KernelSize 128": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitLinearKernelSize array128",
    "explicit_group": "KernelSubmitLinearKernelSize array128"
  },
  "torch_benchmark_sycl KernelSubmitLinearKernelSize KernelSize 512": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitLinearKernelSize array512",
    "explicit_group": "KernelSubmitLinearKernelSize array512"
  },
  "torch_benchmark_sycl KernelSubmitLinearKernelSize KernelSize 1024": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitLinearKernelSize array1024",
    "explicit_group": "KernelSubmitLinearKernelSize array1024"
  },
  "torch_benchmark_sycl KernelSubmitLinearKernelSize KernelSize 5120": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitLinearKernelSize array5120",
    "explicit_group": "KernelSubmitLinearKernelSize array5120"
  },
  "torch_benchmark_sycl KernelSubmitLinearKernelSize KernelSize 32 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitLinearKernelSize array32, CPU count",
    "explicit_group": "KernelSubmitLinearKernelSize array32, CPU count"
  },
  "torch_benchmark_sycl KernelSubmitLinearKernelSize KernelSize 128 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitLinearKernelSize array128, CPU count",
    "explicit_group": "KernelSubmitLinearKernelSize array128, CPU count"
  },
  "torch_benchmark_sycl KernelSubmitLinearKernelSize KernelSize 512 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitLinearKernelSize array512, CPU count",
    "explicit_group": "KernelSubmitLinearKernelSize array512, CPU count"
  },
  "torch_benchmark_sycl KernelSubmitLinearKernelSize KernelSize 1024 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitLinearKernelSize array1024, CPU count",
    "explicit_group": "KernelSubmitLinearKernelSize array1024, CPU count"
  },
  "torch_benchmark_sycl KernelSubmitLinearKernelSize KernelSize 5120 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitLinearKernelSize array5120, CPU count",
    "explicit_group": "KernelSubmitLinearKernelSize array5120, CPU count"
  },
  "torch_benchmark_l0 KernelSubmitLinearKernelSize KernelSize 32": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitLinearKernelSize array32",
    "explicit_group": "KernelSubmitLinearKernelSize array32"
  },
  "torch_benchmark_l0 KernelSubmitLinearKernelSize KernelSize 128": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitLinearKernelSize array128",
    "explicit_group": "KernelSubmitLinearKernelSize array128"
  },
  "torch_benchmark_l0 KernelSubmitLinearKernelSize KernelSize 512": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitLinearKernelSize array512",
    "explicit_group": "KernelSubmitLinearKernelSize array512"
  },
  "torch_benchmark_l0 KernelSubmitLinearKernelSize KernelSize 1024": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitLinearKernelSize array1024",
    "explicit_group": "KernelSubmitLinearKernelSize array1024"
  },
  "torch_benchmark_l0 KernelSubmitLinearKernelSize KernelSize 5120": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitLinearKernelSize array5120",
    "explicit_group": "KernelSubmitLinearKernelSize array5120"
  },
  "torch_benchmark_l0 KernelSubmitLinearKernelSize KernelSize 32 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitLinearKernelSize array32, CPU count",
    "explicit_group": "KernelSubmitLinearKernelSize array32, CPU count"
  },
  "torch_benchmark_l0 KernelSubmitLinearKernelSize KernelSize 128 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitLinearKernelSize array128, CPU count",
    "explicit_group": "KernelSubmitLinearKernelSize array128, CPU count"
  },
  "torch_benchmark_l0 KernelSubmitLinearKernelSize KernelSize 512 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitLinearKernelSize array512, CPU count",
    "explicit_group": "KernelSubmitLinearKernelSize array512, CPU count"
  },
  "torch_benchmark_l0 KernelSubmitLinearKernelSize KernelSize 1024 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitLinearKernelSize array1024, CPU count",
    "explicit_group": "KernelSubmitLinearKernelSize array1024, CPU count"
  },
  "torch_benchmark_l0 KernelSubmitLinearKernelSize KernelSize 5120 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitLinearKernelSize array5120, CPU count",
    "explicit_group": "KernelSubmitLinearKernelSize array5120, CPU count"
  },
  "torch_benchmark_syclpreview KernelSubmitEventRecordWait KernelWGCount 256, KernelWGSize 512, Profiling 0": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitEventRecordWait medium",
    "explicit_group": "KernelSubmitEventRecordWait medium"
  },
  "torch_benchmark_syclpreview KernelSubmitEventRecordWait KernelWGCount 256, KernelWGSize 512, Profiling 0 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitEventRecordWait medium, CPU count",
    "explicit_group": "KernelSubmitEventRecordWait medium, CPU count"
  },
  "torch_benchmark_sycl KernelSubmitEventRecordWait KernelWGCount 256, KernelWGSize 512, Profiling 0": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitEventRecordWait medium",
    "explicit_group": "KernelSubmitEventRecordWait medium"
  },
  "torch_benchmark_sycl KernelSubmitEventRecordWait KernelWGCount 256, KernelWGSize 512, Profiling 0 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitEventRecordWait medium, CPU count",
    "explicit_group": "KernelSubmitEventRecordWait medium, CPU count"
  },
  "torch_benchmark_l0 KernelSubmitEventRecordWait KernelWGCount 256, KernelWGSize 512, Profiling 0": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitEventRecordWait medium",
    "explicit_group": "KernelSubmitEventRecordWait medium"
  },
  "torch_benchmark_l0 KernelSubmitEventRecordWait KernelWGCount 256, KernelWGSize 512, Profiling 0 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitEventRecordWait medium, CPU count",
    "explicit_group": "KernelSubmitEventRecordWait medium, CPU count"
  },
  "torch_benchmark_syclpreview KernelSubmitEventRecordQuery EventQueryIterations 1000, KernelWGCount 256, KernelWGSize 512, Profiling 0": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitEventRecordQuery medium",
    "explicit_group": "KernelSubmitEventRecordQuery medium"
  },
  "torch_benchmark_syclpreview KernelSubmitEventRecordQuery EventQueryIterations 1000, KernelWGCount 256, KernelWGSize 512, Profiling 0 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitEventRecordQuery medium, CPU count",
    "explicit_group": "KernelSubmitEventRecordQuery medium, CPU count"
  },
  "torch_benchmark_sycl KernelSubmitEventRecordQuery EventQueryIterations 1000, KernelWGCount 256, KernelWGSize 512, Profiling 0": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitEventRecordQuery medium",
    "explicit_group": "KernelSubmitEventRecordQuery medium"
  },
  "torch_benchmark_sycl KernelSubmitEventRecordQuery EventQueryIterations 1000, KernelWGCount 256, KernelWGSize 512, Profiling 0 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitEventRecordQuery medium, CPU count",
    "explicit_group": "KernelSubmitEventRecordQuery medium, CPU count"
  },
  "torch_benchmark_l0 KernelSubmitEventRecordQuery EventQueryIterations 1000, KernelWGCount 256, KernelWGSize 512, Profiling 0": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitEventRecordQuery medium",
    "explicit_group": "KernelSubmitEventRecordQuery medium"
  },
  "torch_benchmark_l0 KernelSubmitEventRecordQuery EventQueryIterations 1000, KernelWGCount 256, KernelWGSize 512, Profiling 0 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitEventRecordQuery medium, CPU count",
    "explicit_group": "KernelSubmitEventRecordQuery medium, CPU count"
  },
  "torch_benchmark_syclpreview KernelSubmitGraphSingleQueue KernelBatchSize 10, KernelName Add, KernelsPerQueue 10": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitGraphSingleQueue small",
    "explicit_group": "KernelSubmitGraphSingleQueue small"
  },
  "torch_benchmark_syclpreview KernelSubmitGraphSingleQueue KernelBatchSize 32, KernelName Add, KernelsPerQueue 32": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitGraphSingleQueue medium",
    "explicit_group": "KernelSubmitGraphSingleQueue medium"
  },
  "torch_benchmark_syclpreview KernelSubmitGraphSingleQueue KernelBatchSize 64, KernelName Add, KernelsPerQueue 64": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitGraphSingleQueue large",
    "explicit_group": "KernelSubmitGraphSingleQueue large"
  },
  "torch_benchmark_syclpreview KernelSubmitGraphSingleQueue KernelBatchSize 10, KernelName AddSequence, KernelsPerQueue 10": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitGraphSingleQueue small",
    "explicit_group": "KernelSubmitGraphSingleQueue small"
  },
  "torch_benchmark_syclpreview KernelSubmitGraphSingleQueue KernelBatchSize 32, KernelName AddSequence, KernelsPerQueue 32": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitGraphSingleQueue medium",
    "explicit_group": "KernelSubmitGraphSingleQueue medium"
  },
  "torch_benchmark_syclpreview KernelSubmitGraphSingleQueue KernelBatchSize 64, KernelName AddSequence, KernelsPerQueue 64": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitGraphSingleQueue large",
    "explicit_group": "KernelSubmitGraphSingleQueue large"
  },
  "torch_benchmark_syclpreview KernelSubmitGraphSingleQueue KernelBatchSize 10, KernelName Empty, KernelsPerQueue 10": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitGraphSingleQueue small",
    "explicit_group": "KernelSubmitGraphSingleQueue small"
  },
  "torch_benchmark_syclpreview KernelSubmitGraphSingleQueue KernelBatchSize 32, KernelName Empty, KernelsPerQueue 32": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitGraphSingleQueue medium",
    "explicit_group": "KernelSubmitGraphSingleQueue medium"
  },
  "torch_benchmark_syclpreview KernelSubmitGraphSingleQueue KernelBatchSize 64, KernelName Empty, KernelsPerQueue 64": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitGraphSingleQueue large",
    "explicit_group": "KernelSubmitGraphSingleQueue large"
  },
  "torch_benchmark_syclpreview KernelSubmitGraphSingleQueue KernelBatchSize 10, KernelName Add, KernelsPerQueue 10 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitGraphSingleQueue small, CPU count",
    "explicit_group": "KernelSubmitGraphSingleQueue small, CPU count"
  },
  "torch_benchmark_syclpreview KernelSubmitGraphSingleQueue KernelBatchSize 32, KernelName Add, KernelsPerQueue 32 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitGraphSingleQueue medium, CPU count",
    "explicit_group": "KernelSubmitGraphSingleQueue medium, CPU count"
  },
  "torch_benchmark_syclpreview KernelSubmitGraphSingleQueue KernelBatchSize 64, KernelName Add, KernelsPerQueue 64 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitGraphSingleQueue large, CPU count",
    "explicit_group": "KernelSubmitGraphSingleQueue large, CPU count"
  },
  "torch_benchmark_syclpreview KernelSubmitGraphSingleQueue KernelBatchSize 10, KernelName AddSequence, KernelsPerQueue 10 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitGraphSingleQueue small, CPU count",
    "explicit_group": "KernelSubmitGraphSingleQueue small, CPU count"
  },
  "torch_benchmark_syclpreview KernelSubmitGraphSingleQueue KernelBatchSize 32, KernelName AddSequence, KernelsPerQueue 32 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitGraphSingleQueue medium, CPU count",
    "explicit_group": "KernelSubmitGraphSingleQueue medium, CPU count"
  },
  "torch_benchmark_syclpreview KernelSubmitGraphSingleQueue KernelBatchSize 64, KernelName AddSequence, KernelsPerQueue 64 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitGraphSingleQueue large, CPU count",
    "explicit_group": "KernelSubmitGraphSingleQueue large, CPU count"
  },
  "torch_benchmark_syclpreview KernelSubmitGraphSingleQueue KernelBatchSize 10, KernelName Empty, KernelsPerQueue 10 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitGraphSingleQueue small, CPU count",
    "explicit_group": "KernelSubmitGraphSingleQueue small, CPU count"
  },
  "torch_benchmark_syclpreview KernelSubmitGraphSingleQueue KernelBatchSize 32, KernelName Empty, KernelsPerQueue 32 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitGraphSingleQueue medium, CPU count",
    "explicit_group": "KernelSubmitGraphSingleQueue medium, CPU count"
  },
  "torch_benchmark_syclpreview KernelSubmitGraphSingleQueue KernelBatchSize 64, KernelName Empty, KernelsPerQueue 64 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitGraphSingleQueue large, CPU count",
    "explicit_group": "KernelSubmitGraphSingleQueue large, CPU count"
  },
  "torch_benchmark_sycl KernelSubmitGraphSingleQueue KernelBatchSize 10, KernelName Add, KernelsPerQueue 10": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitGraphSingleQueue small",
    "explicit_group": "KernelSubmitGraphSingleQueue small"
  },
  "torch_benchmark_sycl KernelSubmitGraphSingleQueue KernelBatchSize 32, KernelName Add, KernelsPerQueue 32": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitGraphSingleQueue medium",
    "explicit_group": "KernelSubmitGraphSingleQueue medium"
  },
  "torch_benchmark_sycl KernelSubmitGraphSingleQueue KernelBatchSize 64, KernelName Add, KernelsPerQueue 64": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitGraphSingleQueue large",
    "explicit_group": "KernelSubmitGraphSingleQueue large"
  },
  "torch_benchmark_sycl KernelSubmitGraphSingleQueue KernelBatchSize 10, KernelName AddSequence, KernelsPerQueue 10": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitGraphSingleQueue small",
    "explicit_group": "KernelSubmitGraphSingleQueue small"
  },
  "torch_benchmark_sycl KernelSubmitGraphSingleQueue KernelBatchSize 32, KernelName AddSequence, KernelsPerQueue 32": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitGraphSingleQueue medium",
    "explicit_group": "KernelSubmitGraphSingleQueue medium"
  },
  "torch_benchmark_sycl KernelSubmitGraphSingleQueue KernelBatchSize 64, KernelName AddSequence, KernelsPerQueue 64": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitGraphSingleQueue large",
    "explicit_group": "KernelSubmitGraphSingleQueue large"
  },
  "torch_benchmark_sycl KernelSubmitGraphSingleQueue KernelBatchSize 10, KernelName Empty, KernelsPerQueue 10": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitGraphSingleQueue small",
    "explicit_group": "KernelSubmitGraphSingleQueue small"
  },
  "torch_benchmark_sycl KernelSubmitGraphSingleQueue KernelBatchSize 32, KernelName Empty, KernelsPerQueue 32": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitGraphSingleQueue medium",
    "explicit_group": "KernelSubmitGraphSingleQueue medium"
  },
  "torch_benchmark_sycl KernelSubmitGraphSingleQueue KernelBatchSize 64, KernelName Empty, KernelsPerQueue 64": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitGraphSingleQueue large",
    "explicit_group": "KernelSubmitGraphSingleQueue large"
  },
  "torch_benchmark_sycl KernelSubmitGraphSingleQueue KernelBatchSize 10, KernelName Add, KernelsPerQueue 10 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitGraphSingleQueue small, CPU count",
    "explicit_group": "KernelSubmitGraphSingleQueue small, CPU count"
  },
  "torch_benchmark_sycl KernelSubmitGraphSingleQueue KernelBatchSize 32, KernelName Add, KernelsPerQueue 32 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitGraphSingleQueue medium, CPU count",
    "explicit_group": "KernelSubmitGraphSingleQueue medium, CPU count"
  },
  "torch_benchmark_sycl KernelSubmitGraphSingleQueue KernelBatchSize 64, KernelName Add, KernelsPerQueue 64 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitGraphSingleQueue large, CPU count",
    "explicit_group": "KernelSubmitGraphSingleQueue large, CPU count"
  },
  "torch_benchmark_sycl KernelSubmitGraphSingleQueue KernelBatchSize 10, KernelName AddSequence, KernelsPerQueue 10 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitGraphSingleQueue small, CPU count",
    "explicit_group": "KernelSubmitGraphSingleQueue small, CPU count"
  },
  "torch_benchmark_sycl KernelSubmitGraphSingleQueue KernelBatchSize 32, KernelName AddSequence, KernelsPerQueue 32 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitGraphSingleQueue medium, CPU count",
    "explicit_group": "KernelSubmitGraphSingleQueue medium, CPU count"
  },
  "torch_benchmark_sycl KernelSubmitGraphSingleQueue KernelBatchSize 64, KernelName AddSequence, KernelsPerQueue 64 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitGraphSingleQueue large, CPU count",
    "explicit_group": "KernelSubmitGraphSingleQueue large, CPU count"
  },
  "torch_benchmark_sycl KernelSubmitGraphSingleQueue KernelBatchSize 10, KernelName Empty, KernelsPerQueue 10 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitGraphSingleQueue small, CPU count",
    "explicit_group": "KernelSubmitGraphSingleQueue small, CPU count"
  },
  "torch_benchmark_sycl KernelSubmitGraphSingleQueue KernelBatchSize 32, KernelName Empty, KernelsPerQueue 32 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitGraphSingleQueue medium, CPU count",
    "explicit_group": "KernelSubmitGraphSingleQueue medium, CPU count"
  },
  "torch_benchmark_sycl KernelSubmitGraphSingleQueue KernelBatchSize 64, KernelName Empty, KernelsPerQueue 64 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitGraphSingleQueue large, CPU count",
    "explicit_group": "KernelSubmitGraphSingleQueue large, CPU count"
  },
  "torch_benchmark_l0 KernelSubmitGraphSingleQueue KernelBatchSize 10, KernelName Add, KernelsPerQueue 10": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitGraphSingleQueue small",
    "explicit_group": "KernelSubmitGraphSingleQueue small"
  },
  "torch_benchmark_l0 KernelSubmitGraphSingleQueue KernelBatchSize 32, KernelName Add, KernelsPerQueue 32": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitGraphSingleQueue medium",
    "explicit_group": "KernelSubmitGraphSingleQueue medium"
  },
  "torch_benchmark_l0 KernelSubmitGraphSingleQueue KernelBatchSize 64, KernelName Add, KernelsPerQueue 64": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitGraphSingleQueue large",
    "explicit_group": "KernelSubmitGraphSingleQueue large"
  },
  "torch_benchmark_l0 KernelSubmitGraphSingleQueue KernelBatchSize 10, KernelName AddSequence, KernelsPerQueue 10": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitGraphSingleQueue small",
    "explicit_group": "KernelSubmitGraphSingleQueue small"
  },
  "torch_benchmark_l0 KernelSubmitGraphSingleQueue KernelBatchSize 32, KernelName AddSequence, KernelsPerQueue 32": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitGraphSingleQueue medium",
    "explicit_group": "KernelSubmitGraphSingleQueue medium"
  },
  "torch_benchmark_l0 KernelSubmitGraphSingleQueue KernelBatchSize 64, KernelName AddSequence, KernelsPerQueue 64": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitGraphSingleQueue large",
    "explicit_group": "KernelSubmitGraphSingleQueue large"
  },
  "torch_benchmark_l0 KernelSubmitGraphSingleQueue KernelBatchSize 10, KernelName Empty, KernelsPerQueue 10": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitGraphSingleQueue small",
    "explicit_group": "KernelSubmitGraphSingleQueue small"
  },
  "torch_benchmark_l0 KernelSubmitGraphSingleQueue KernelBatchSize 32, KernelName Empty, KernelsPerQueue 32": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitGraphSingleQueue medium",
    "explicit_group": "KernelSubmitGraphSingleQueue medium"
  },
  "torch_benchmark_l0 KernelSubmitGraphSingleQueue KernelBatchSize 64, KernelName Empty, KernelsPerQueue 64": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitGraphSingleQueue large",
    "explicit_group": "KernelSubmitGraphSingleQueue large"
  },
  "torch_benchmark_l0 KernelSubmitGraphSingleQueue KernelBatchSize 10, KernelName Add, KernelsPerQueue 10 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitGraphSingleQueue small, CPU count",
    "explicit_group": "KernelSubmitGraphSingleQueue small, CPU count"
  },
  "torch_benchmark_l0 KernelSubmitGraphSingleQueue KernelBatchSize 32, KernelName Add, KernelsPerQueue 32 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitGraphSingleQueue medium, CPU count",
    "explicit_group": "KernelSubmitGraphSingleQueue medium, CPU count"
  },
  "torch_benchmark_l0 KernelSubmitGraphSingleQueue KernelBatchSize 64, KernelName Add, KernelsPerQueue 64 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitGraphSingleQueue large, CPU count",
    "explicit_group": "KernelSubmitGraphSingleQueue large, CPU count"
  },
  "torch_benchmark_l0 KernelSubmitGraphSingleQueue KernelBatchSize 10, KernelName AddSequence, KernelsPerQueue 10 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitGraphSingleQueue small, CPU count",
    "explicit_group": "KernelSubmitGraphSingleQueue small, CPU count"
  },
  "torch_benchmark_l0 KernelSubmitGraphSingleQueue KernelBatchSize 32, KernelName AddSequence, KernelsPerQueue 32 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitGraphSingleQueue medium, CPU count",
    "explicit_group": "KernelSubmitGraphSingleQueue medium, CPU count"
  },
  "torch_benchmark_l0 KernelSubmitGraphSingleQueue KernelBatchSize 64, KernelName AddSequence, KernelsPerQueue 64 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitGraphSingleQueue large, CPU count",
    "explicit_group": "KernelSubmitGraphSingleQueue large, CPU count"
  },
  "torch_benchmark_l0 KernelSubmitGraphSingleQueue KernelBatchSize 10, KernelName Empty, KernelsPerQueue 10 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitGraphSingleQueue small, CPU count",
    "explicit_group": "KernelSubmitGraphSingleQueue small, CPU count"
  },
  "torch_benchmark_l0 KernelSubmitGraphSingleQueue KernelBatchSize 32, KernelName Empty, KernelsPerQueue 32 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitGraphSingleQueue medium, CPU count",
    "explicit_group": "KernelSubmitGraphSingleQueue medium, CPU count"
  },
  "torch_benchmark_l0 KernelSubmitGraphSingleQueue KernelBatchSize 64, KernelName Empty, KernelsPerQueue 64 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitGraphSingleQueue large, CPU count",
    "explicit_group": "KernelSubmitGraphSingleQueue large, CPU count"
  },
  "torch_benchmark_syclpreview KernelSubmitGraphMultiQueue KernelsPerQueue 10": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitGraphMultiQueue small",
    "explicit_group": "KernelSubmitGraphMultiQueue small"
  },
  "torch_benchmark_syclpreview KernelSubmitGraphMultiQueue KernelsPerQueue 32": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitGraphMultiQueue medium",
    "explicit_group": "KernelSubmitGraphMultiQueue medium"
  },
  "torch_benchmark_syclpreview KernelSubmitGraphMultiQueue KernelsPerQueue 64": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitGraphMultiQueue large",
    "explicit_group": "KernelSubmitGraphMultiQueue large"
  },
  "torch_benchmark_syclpreview KernelSubmitGraphMultiQueue KernelsPerQueue 10 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitGraphMultiQueue small, CPU count",
    "explicit_group": "KernelSubmitGraphMultiQueue small, CPU count"
  },
  "torch_benchmark_syclpreview KernelSubmitGraphMultiQueue KernelsPerQueue 32 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitGraphMultiQueue medium, CPU count",
    "explicit_group": "KernelSubmitGraphMultiQueue medium, CPU count"
  },
  "torch_benchmark_syclpreview KernelSubmitGraphMultiQueue KernelsPerQueue 64 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitGraphMultiQueue large, CPU count",
    "explicit_group": "KernelSubmitGraphMultiQueue large, CPU count"
  },
  "torch_benchmark_sycl KernelSubmitGraphMultiQueue KernelsPerQueue 10": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitGraphMultiQueue small",
    "explicit_group": "KernelSubmitGraphMultiQueue small"
  },
  "torch_benchmark_sycl KernelSubmitGraphMultiQueue KernelsPerQueue 32": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitGraphMultiQueue medium",
    "explicit_group": "KernelSubmitGraphMultiQueue medium"
  },
  "torch_benchmark_sycl KernelSubmitGraphMultiQueue KernelsPerQueue 64": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitGraphMultiQueue large",
    "explicit_group": "KernelSubmitGraphMultiQueue large"
  },
  "torch_benchmark_sycl KernelSubmitGraphMultiQueue KernelsPerQueue 10 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitGraphMultiQueue small, CPU count",
    "explicit_group": "KernelSubmitGraphMultiQueue small, CPU count"
  },
  "torch_benchmark_sycl KernelSubmitGraphMultiQueue KernelsPerQueue 32 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitGraphMultiQueue medium, CPU count",
    "explicit_group": "KernelSubmitGraphMultiQueue medium, CPU count"
  },
  "torch_benchmark_sycl KernelSubmitGraphMultiQueue KernelsPerQueue 64 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitGraphMultiQueue large, CPU count",
    "explicit_group": "KernelSubmitGraphMultiQueue large, CPU count"
  },
  "torch_benchmark_l0 KernelSubmitGraphMultiQueue KernelsPerQueue 10": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitGraphMultiQueue small",
    "explicit_group": "KernelSubmitGraphMultiQueue small"
  },
  "torch_benchmark_l0 KernelSubmitGraphMultiQueue KernelsPerQueue 32": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitGraphMultiQueue medium",
    "explicit_group": "KernelSubmitGraphMultiQueue medium"
  },
  "torch_benchmark_l0 KernelSubmitGraphMultiQueue KernelsPerQueue 64": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitGraphMultiQueue large",
    "explicit_group": "KernelSubmitGraphMultiQueue large"
  },
  "torch_benchmark_l0 KernelSubmitGraphMultiQueue KernelsPerQueue 10 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitGraphMultiQueue small, CPU count",
    "explicit_group": "KernelSubmitGraphMultiQueue small, CPU count"
  },
  "torch_benchmark_l0 KernelSubmitGraphMultiQueue KernelsPerQueue 32 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitGraphMultiQueue medium, CPU count",
    "explicit_group": "KernelSubmitGraphMultiQueue medium, CPU count"
  },
  "torch_benchmark_l0 KernelSubmitGraphMultiQueue KernelsPerQueue 64 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitGraphMultiQueue large, CPU count",
    "explicit_group": "KernelSubmitGraphMultiQueue large, CPU count"
  },
  "torch_benchmark_syclpreview KernelSubmitGraphVllmMock AllocCount 32, GraphScenario 0": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitGraphVllmMock small",
    "explicit_group": "KernelSubmitGraphVllmMock small"
  },
  "torch_benchmark_syclpreview KernelSubmitGraphVllmMock AllocCount 128, GraphScenario 0": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitGraphVllmMock large",
    "explicit_group": "KernelSubmitGraphVllmMock large"
  },
  "torch_benchmark_syclpreview KernelSubmitGraphVllmMock AllocCount 128, GraphScenario 1": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitGraphVllmMock large",
    "explicit_group": "KernelSubmitGraphVllmMock large"
  },
  "torch_benchmark_syclpreview KernelSubmitGraphVllmMock AllocCount 128, GraphScenario 2": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitGraphVllmMock large",
    "explicit_group": "KernelSubmitGraphVllmMock large"
  },
  "torch_benchmark_syclpreview KernelSubmitGraphVllmMock AllocCount 128, GraphScenario 3": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitGraphVllmMock large",
    "explicit_group": "KernelSubmitGraphVllmMock large"
  },
  "torch_benchmark_syclpreview KernelSubmitGraphVllmMock AllocCount 32, GraphScenario 0 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitGraphVllmMock small, CPU count",
    "explicit_group": "KernelSubmitGraphVllmMock small, CPU count"
  },
  "torch_benchmark_syclpreview KernelSubmitGraphVllmMock AllocCount 128, GraphScenario 0 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitGraphVllmMock large, CPU count",
    "explicit_group": "KernelSubmitGraphVllmMock large, CPU count"
  },
  "torch_benchmark_syclpreview KernelSubmitGraphVllmMock AllocCount 128, GraphScenario 1 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitGraphVllmMock large, CPU count",
    "explicit_group": "KernelSubmitGraphVllmMock large, CPU count"
  },
  "torch_benchmark_syclpreview KernelSubmitGraphVllmMock AllocCount 128, GraphScenario 2 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitGraphVllmMock large, CPU count",
    "explicit_group": "KernelSubmitGraphVllmMock large, CPU count"
  },
  "torch_benchmark_syclpreview KernelSubmitGraphVllmMock AllocCount 128, GraphScenario 3 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSubmitGraphVllmMock large, CPU count",
    "explicit_group": "KernelSubmitGraphVllmMock large, CPU count"
  },
  "torch_benchmark_sycl KernelSubmitGraphVllmMock AllocCount 32, GraphScenario 0": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitGraphVllmMock small",
    "explicit_group": "KernelSubmitGraphVllmMock small"
  },
  "torch_benchmark_sycl KernelSubmitGraphVllmMock AllocCount 128, GraphScenario 0": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitGraphVllmMock large",
    "explicit_group": "KernelSubmitGraphVllmMock large"
  },
  "torch_benchmark_sycl KernelSubmitGraphVllmMock AllocCount 128, GraphScenario 1": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitGraphVllmMock large",
    "explicit_group": "KernelSubmitGraphVllmMock large"
  },
  "torch_benchmark_sycl KernelSubmitGraphVllmMock AllocCount 128, GraphScenario 2": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitGraphVllmMock large",
    "explicit_group": "KernelSubmitGraphVllmMock large"
  },
  "torch_benchmark_sycl KernelSubmitGraphVllmMock AllocCount 128, GraphScenario 3": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitGraphVllmMock large",
    "explicit_group": "KernelSubmitGraphVllmMock large"
  },
  "torch_benchmark_sycl KernelSubmitGraphVllmMock AllocCount 32, GraphScenario 0 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitGraphVllmMock small, CPU count",
    "explicit_group": "KernelSubmitGraphVllmMock small, CPU count"
  },
  "torch_benchmark_sycl KernelSubmitGraphVllmMock AllocCount 128, GraphScenario 0 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitGraphVllmMock large, CPU count",
    "explicit_group": "KernelSubmitGraphVllmMock large, CPU count"
  },
  "torch_benchmark_sycl KernelSubmitGraphVllmMock AllocCount 128, GraphScenario 1 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitGraphVllmMock large, CPU count",
    "explicit_group": "KernelSubmitGraphVllmMock large, CPU count"
  },
  "torch_benchmark_sycl KernelSubmitGraphVllmMock AllocCount 128, GraphScenario 2 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitGraphVllmMock large, CPU count",
    "explicit_group": "KernelSubmitGraphVllmMock large, CPU count"
  },
  "torch_benchmark_sycl KernelSubmitGraphVllmMock AllocCount 128, GraphScenario 3 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "SYCL"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSubmitGraphVllmMock large, CPU count",
    "explicit_group": "KernelSubmitGraphVllmMock large, CPU count"
  },
  "torch_benchmark_l0 KernelSubmitGraphVllmMock AllocCount 32, GraphScenario 0": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitGraphVllmMock small",
    "explicit_group": "KernelSubmitGraphVllmMock small"
  },
  "torch_benchmark_l0 KernelSubmitGraphVllmMock AllocCount 128, GraphScenario 0": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitGraphVllmMock large",
    "explicit_group": "KernelSubmitGraphVllmMock large"
  },
  "torch_benchmark_l0 KernelSubmitGraphVllmMock AllocCount 128, GraphScenario 1": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitGraphVllmMock large",
    "explicit_group": "KernelSubmitGraphVllmMock large"
  },
  "torch_benchmark_l0 KernelSubmitGraphVllmMock AllocCount 128, GraphScenario 2": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitGraphVllmMock large",
    "explicit_group": "KernelSubmitGraphVllmMock large"
  },
  "torch_benchmark_l0 KernelSubmitGraphVllmMock AllocCount 128, GraphScenario 3": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitGraphVllmMock large",
    "explicit_group": "KernelSubmitGraphVllmMock large"
  },
  "torch_benchmark_l0 KernelSubmitGraphVllmMock AllocCount 32, GraphScenario 0 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitGraphVllmMock small, CPU count",
    "explicit_group": "KernelSubmitGraphVllmMock small, CPU count"
  },
  "torch_benchmark_l0 KernelSubmitGraphVllmMock AllocCount 128, GraphScenario 0 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitGraphVllmMock large, CPU count",
    "explicit_group": "KernelSubmitGraphVllmMock large, CPU count"
  },
  "torch_benchmark_l0 KernelSubmitGraphVllmMock AllocCount 128, GraphScenario 1 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitGraphVllmMock large, CPU count",
    "explicit_group": "KernelSubmitGraphVllmMock large, CPU count"
  },
  "torch_benchmark_l0 KernelSubmitGraphVllmMock AllocCount 128, GraphScenario 2 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitGraphVllmMock large, CPU count",
    "explicit_group": "KernelSubmitGraphVllmMock large, CPU count"
  },
  "torch_benchmark_l0 KernelSubmitGraphVllmMock AllocCount 128, GraphScenario 3 CPU count": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "L0"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSubmitGraphVllmMock large, CPU count",
    "explicit_group": "KernelSubmitGraphVllmMock large, CPU count"
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
  "real-world-app": {
    "type": "benchmark",
    "description": "Measures real-world XPU application performance via PyTorch dynamo microbenchmark.",
    "notes": null,
    "unstable": null,
    "tags": [
      "pytorch",
      "micro",
      "inference",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "PyTorch Real-World App Microbenchmark",
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
  "onednn-sum-f16-1-eager": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "onednn-sum-f16-1-eager",
    "explicit_group": "sum-f16-1"
  },
  "onednn-sum-f16-2-eager": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "onednn-sum-f16-2-eager",
    "explicit_group": "sum-f16-2"
  },
  "onednn-sum-f32-1-eager": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "onednn-sum-f32-1-eager",
    "explicit_group": "sum-f32-1"
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
  "onednn-sum-padding-1-eager": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "onednn-sum-padding-1-eager",
    "explicit_group": "sum-padding-1"
  },
  "onednn-sum-padding-1-graph": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "onednn-sum-padding-1-graph",
    "explicit_group": "sum-padding-1"
  },
  "onednn-sum-padding-2-eager": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "onednn-sum-padding-2-eager",
    "explicit_group": "sum-padding-2"
  },
  "onednn-sum-padding-2-graph": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "onednn-sum-padding-2-graph",
    "explicit_group": "sum-padding-2"
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
  "OFFLOAD": {
    "name": "OFFLOAD",
    "description": "Benchmark uses the LLVM Offload API directly"
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
  },
  "pytorch": {
    "name": "pytorch",
    "description": "Tests workloads close to Pytorch ones"
  }
};
defaultCompareNames = ["offload_ApiOverhead_l0"];
