benchmarkRuns = [
  {
    "results": [
      {
        "label": "gromacs-0006-rf-rf",
        "value": 2.764,
        "command": "/home/mateuszpn/workdir/gromacs-build/bin/gmx mdrun -s /home/mateuszpn/workdir/grappa-1.5k-6.1M_rc0.9/0006/rf.tpr -nb gpu -update gpu -bonded gpu -ntmpi 1 -ntomp 1 -nobackup -noconfout -nstlist 100 -pin on",
        "env": {
          "SYCL_CACHE_PERSISTENT": "1",
          "GMX_CUDA_GRAPH": "1"
        },
        "stdout": "          :-) GROMACS - gmx mdrun, 2025.1-dev-20250311-9b544f7b15 (-:\n\nExecutable:   /home/mateuszpn/workdir/gromacs-build/bin/gmx\nData prefix:  /home/mateuszpn/workdir/gromacs-repo (source tree)\nWorking dir:  /home/mateuszpn/workdir/bcwd/bcwd\nCommand line:\n  gmx mdrun -s /home/mateuszpn/workdir/grappa-1.5k-6.1M_rc0.9/0006/rf.tpr -nb gpu -update gpu -bonded gpu -ntmpi 1 -ntomp 1 -nobackup -noconfout -nstlist 100 -pin on\n\nReading file /home/mateuszpn/workdir/grappa-1.5k-6.1M_rc0.9/0006/rf.tpr, VERSION 2025.1-dev-20250311-9b544f7b15 (single precision)\n\nWARNING: Can not increase nstlist because the box is too small\n\nGMX_CUDA_GRAPH environment variable is detected. The experimental CUDA Graphs feature will be used if run conditions allow.\n\n1 GPU selected for this run.\nMapping of GPU IDs to the 1 GPU task in the 1 rank on this node:\n  PP:0\nPP tasks will do (non-perturbed) short-ranged and most bonded interactions on the GPU\nPP task will update and constrain coordinates on the GPU\nCUDA Graphs will be used, provided there are no CPU force computations.\nUsing 1 MPI thread\nstarting mdrun 'ethanol in water'\n10000 steps,     20.0 ps.\n\nNOTE: 50 % of the run time was spent in pair search,\n      you might want to increase nstlist (this has no effect on accuracy)\n\n               Core t (s)   Wall t (s)        (%)\n       Time:        2.764        2.764      100.0\n                 (ns/day)    (hour/ns)\nPerformance:      625.268        0.038\n\nGROMACS reminds you: \"I Used To Care, But Things Have Changed\" (Bob Dylan)\n\n",
        "passed": true,
        "unit": "s",
        "explicit_group": "",
        "stddev": 0.011532562594670854,
        "git_url": "https://gitlab.com/gromacs/gromacs.git",
        "git_hash": "v2025.1",
        "name": "gromacs-0006-rf-rf",
        "lower_is_better": true,
        "suite": "Gromacs Bench"
      },
      {
        "label": "gromacs-0192-pme-pme",
        "value": 15.782,
        "command": "/home/mateuszpn/workdir/gromacs-build/bin/gmx mdrun -s /home/mateuszpn/workdir/grappa-1.5k-6.1M_rc0.9/0192/pme.tpr -nb gpu -update gpu -bonded gpu -ntmpi 1 -ntomp 1 -nobackup -noconfout -nstlist 100 -pin on -pme gpu -pmefft gpu -notunepme",
        "env": {
          "SYCL_CACHE_PERSISTENT": "1",
          "GMX_CUDA_GRAPH": "1"
        },
        "stdout": "          :-) GROMACS - gmx mdrun, 2025.1-dev-20250311-9b544f7b15 (-:\n\nExecutable:   /home/mateuszpn/workdir/gromacs-build/bin/gmx\nData prefix:  /home/mateuszpn/workdir/gromacs-repo (source tree)\nWorking dir:  /home/mateuszpn/workdir/bcwd/bcwd\nCommand line:\n  gmx mdrun -s /home/mateuszpn/workdir/grappa-1.5k-6.1M_rc0.9/0192/pme.tpr -nb gpu -update gpu -bonded gpu -ntmpi 1 -ntomp 1 -nobackup -noconfout -nstlist 100 -pin on -pme gpu -pmefft gpu -notunepme\n\nReading file /home/mateuszpn/workdir/grappa-1.5k-6.1M_rc0.9/0192/pme.tpr, VERSION 2025.1-dev-20250311-9b544f7b15 (single precision)\nChanging nstlist from 10 to 100, rlist from 0.903 to 1.118\n\nGMX_CUDA_GRAPH environment variable is detected. The experimental CUDA Graphs feature will be used if run conditions allow.\n\n1 GPU selected for this run.\nMapping of GPU IDs to the 2 GPU tasks in the 1 rank on this node:\n  PP:0,PME:0\nPP tasks will do (non-perturbed) short-ranged and most bonded interactions on the GPU\nPP task will update and constrain coordinates on the GPU\nPME tasks will do all aspects on the GPU\nCUDA Graphs will be used, provided there are no CPU force computations.\nUsing 1 MPI thread\nstarting mdrun 'ethanol in water'\n10000 steps,     20.0 ps.\n\nNOTE: 31 % of the run time was spent in pair search,\n      you might want to increase nstlist (this has no effect on accuracy)\n\n               Core t (s)   Wall t (s)        (%)\n       Time:       15.782       15.782      100.0\n                 (ns/day)    (hour/ns)\nPerformance:      109.503        0.219\n\nGROMACS reminds you: \"XML is not a language in the sense of a programming language any more than sketches on a napkin are a language.\" (Charles Simonyi)\n\n",
        "passed": true,
        "unit": "s",
        "explicit_group": "",
        "stddev": 0.013527749258469196,
        "git_url": "https://gitlab.com/gromacs/gromacs.git",
        "git_hash": "v2025.1",
        "name": "gromacs-0192-pme-pme",
        "lower_is_better": true,
        "suite": "Gromacs Bench"
      }
    ],
    "name": "results",
    "hostname": "gkdse-pre-dnp-02",
    "git_hash": "acd567e61065",
    "github_repo": "mateuszpn/llvm",
    "date": "2025-04-15T17:20:54.891634+00:00",
    "compute_runtime": "Unknown"
  },
  {
    "results": [],
    "name": "results",
    "hostname": "gkdse-pre-dnp-02",
    "git_hash": "acd567e61065",
    "github_repo": "mateuszpn/llvm",
    "date": "2025-04-15T17:16:17.078366+00:00",
    "compute_runtime": "Unknown"
  },
  {
    "results": [],
    "name": "results",
    "hostname": "gkdse-pre-dnp-02",
    "git_hash": "acd567e61065",
    "github_repo": "mateuszpn/llvm",
    "date": "2025-04-15T17:06:45.976403+00:00",
    "compute_runtime": "Unknown"
  },
  {
    "results": [],
    "name": "results",
    "hostname": "gkdse-pre-dnp-02",
    "git_hash": "acd567e61065",
    "github_repo": "mateuszpn/llvm",
    "date": "2025-04-15T16:48:00.803042+00:00",
    "compute_runtime": "Unknown"
  },
  {
    "results": [],
    "name": "results",
    "hostname": "gkdse-pre-dnp-02",
    "git_hash": "acd567e61065",
    "github_repo": "mateuszpn/llvm",
    "date": "2025-04-15T16:43:10.019669+00:00",
    "compute_runtime": "Unknown"
  },
  {
    "results": [],
    "name": "results",
    "hostname": "gkdse-pre-dnp-02",
    "git_hash": "acd567e61065",
    "github_repo": "mateuszpn/llvm",
    "date": "2025-04-15T16:16:04.818326+00:00",
    "compute_runtime": "Unknown"
  },
  {
    "results": [],
    "name": "results",
    "hostname": "gkdse-pre-dnp-02",
    "git_hash": "acd567e61065",
    "github_repo": "mateuszpn/llvm",
    "date": "2025-04-15T16:10:40.043779+00:00",
    "compute_runtime": "Unknown"
  },
  {
    "results": [],
    "name": "results",
    "hostname": "gkdse-pre-dnp-02",
    "git_hash": "acd567e61065",
    "github_repo": "mateuszpn/llvm",
    "date": "2025-04-15T15:46:34.962739+00:00",
    "compute_runtime": "Unknown"
  },
  {
    "results": [],
    "name": "results",
    "hostname": "gkdse-pre-dnp-02",
    "git_hash": "acd567e61065",
    "github_repo": "mateuszpn/llvm",
    "date": "2025-04-15T15:42:18.372784+00:00",
    "compute_runtime": "Unknown"
  },
  {
    "results": [],
    "name": "results",
    "hostname": "gkdse-pre-dnp-02",
    "git_hash": "acd567e61065",
    "github_repo": "mateuszpn/llvm",
    "date": "2025-04-15T15:34:04.594174+00:00",
    "compute_runtime": "Unknown"
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
  },
  "api_overhead_benchmark_sycl SubmitKernel out of order": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL API. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null
  },
  "api_overhead_benchmark_sycl SubmitKernel out of order using eventless SYCL enqueue": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL API. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null
  },
  "api_overhead_benchmark_sycl SubmitKernel out of order with measure completion": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL API. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null
  },
  "api_overhead_benchmark_sycl SubmitKernel out of order with measure completion using eventless SYCL enqueue": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL API. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null
  },
  "api_overhead_benchmark_sycl SubmitKernel in order": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL API. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null
  },
  "api_overhead_benchmark_sycl SubmitKernel in order using eventless SYCL enqueue": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL API. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null
  },
  "api_overhead_benchmark_sycl SubmitKernel in order with measure completion": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL API. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null
  },
  "api_overhead_benchmark_sycl SubmitKernel in order with measure completion using eventless SYCL enqueue": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL API. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null
  },
  "api_overhead_benchmark_l0 SubmitKernel out of order": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Level Zero API. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null
  },
  "api_overhead_benchmark_l0 SubmitKernel out of order with measure completion": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Level Zero API. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null
  },
  "api_overhead_benchmark_l0 SubmitKernel in order": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Level Zero API. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null
  },
  "api_overhead_benchmark_l0 SubmitKernel in order with measure completion": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Level Zero API. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null
  },
  "api_overhead_benchmark_ur SubmitKernel out of order": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Unified Runtime API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null
  },
  "api_overhead_benchmark_ur SubmitKernel out of order with measure completion": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Unified Runtime API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null
  },
  "api_overhead_benchmark_ur SubmitKernel in order": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Unified Runtime API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null
  },
  "api_overhead_benchmark_ur SubmitKernel in order with measure completion": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Unified Runtime API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
  },
  "multithread_benchmark_ur MemcpyExecute opsPerThread:400, numThreads:1, allocSize:102400 srcUSM:1 dstUSM:1": {
    "type": "benchmark",
    "description": "Measures multithreaded memory copy performance with 1 threads each performing 400 operations on 102400 bytes from device to device memory with events.",
    "notes": null,
    "unstable": null,
    "tags": [
      "memory",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": null,
    "range_max": null
  },
  "multithread_benchmark_ur MemcpyExecute opsPerThread:400, numThreads:1, allocSize:102400 srcUSM:0 dstUSM:1": {
    "type": "benchmark",
    "description": "Measures multithreaded memory copy performance with 1 threads each performing 400 operations on 102400 bytes from host to device memory with events.",
    "notes": null,
    "unstable": null,
    "tags": [
      "memory",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": null,
    "range_max": null
  },
  "multithread_benchmark_ur MemcpyExecute opsPerThread:4096, numThreads:4, allocSize:1024 srcUSM:0 dstUSM:1 without events": {
    "type": "benchmark",
    "description": "Measures multithreaded memory copy performance with 4 threads each performing 4096 operations on 1024 bytes from host to device memory without events.",
    "notes": null,
    "unstable": null,
    "tags": [
      "memory",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": null,
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
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
    "range_max": null
  },
  "gromacs-0006-rf": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null
  },
  "gromacs-0192-pme": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null
  },
  "Foo Group": {
    "type": "group",
    "description": "This is a test benchmark for Foo Group.",
    "notes": "This is a test note for Foo Group.\nLook, multiple lines!",
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null
  },
  "Bar Group": {
    "type": "group",
    "description": "This is a test benchmark for Bar Group.",
    "notes": null,
    "unstable": "This is an unstable note for Bar Group.",
    "tags": [],
    "range_min": null,
    "range_max": null
  },
  "Memory Bandwidth 1": {
    "type": "benchmark",
    "description": "This is a test benchmark for Memory Bandwidth 1.",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null
  },
  "Memory Bandwidth 2": {
    "type": "benchmark",
    "description": "This is a test benchmark for Memory Bandwidth 2.",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null
  },
  "Memory Bandwidth 3": {
    "type": "benchmark",
    "description": "This is a test benchmark for Memory Bandwidth 3.",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null
  },
  "Memory Bandwidth 4": {
    "type": "benchmark",
    "description": "This is a test benchmark for Memory Bandwidth 4.",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null
  },
  "Memory Bandwidth 5": {
    "type": "benchmark",
    "description": "This is a test benchmark for Memory Bandwidth 5.",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null
  },
  "Memory Bandwidth 6": {
    "type": "benchmark",
    "description": "This is a test benchmark for Memory Bandwidth 6.",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null
  },
  "Latency 1": {
    "type": "benchmark",
    "description": "This is a test benchmark for Latency 1.",
    "notes": "A Latency test note!",
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null
  },
  "Latency 2": {
    "type": "benchmark",
    "description": "This is a test benchmark for Latency 2.",
    "notes": "A Latency test note!",
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null
  },
  "Latency 3": {
    "type": "benchmark",
    "description": "This is a test benchmark for Latency 3.",
    "notes": "A Latency test note!",
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null
  },
  "Latency 4": {
    "type": "benchmark",
    "description": "This is a test benchmark for Latency 4.",
    "notes": "A Latency test note!",
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null
  },
  "Latency 5": {
    "type": "benchmark",
    "description": "This is a test benchmark for Latency 5.",
    "notes": "A Latency test note!",
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null
  },
  "Latency 6": {
    "type": "benchmark",
    "description": "This is a test benchmark for Latency 6.",
    "notes": "A Latency test note!",
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null
  },
  "Throughput 1": {
    "type": "benchmark",
    "description": "This is a test benchmark for Throughput 1.",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null
  },
  "Throughput 2": {
    "type": "benchmark",
    "description": "This is a test benchmark for Throughput 2.",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null
  },
  "Throughput 3": {
    "type": "benchmark",
    "description": "This is a test benchmark for Throughput 3.",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null
  },
  "Throughput 4": {
    "type": "benchmark",
    "description": "This is a test benchmark for Throughput 4.",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null
  },
  "Throughput 5": {
    "type": "benchmark",
    "description": "This is a test benchmark for Throughput 5.",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null
  },
  "Throughput 6": {
    "type": "benchmark",
    "description": "This is a test benchmark for Throughput 6.",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null
  },
  "FLOPS 1": {
    "type": "benchmark",
    "description": "This is a test benchmark for FLOPS 1.",
    "notes": null,
    "unstable": "Unstable FLOPS test!",
    "tags": [],
    "range_min": null,
    "range_max": null
  },
  "FLOPS 2": {
    "type": "benchmark",
    "description": "This is a test benchmark for FLOPS 2.",
    "notes": null,
    "unstable": "Unstable FLOPS test!",
    "tags": [],
    "range_min": null,
    "range_max": null
  },
  "FLOPS 3": {
    "type": "benchmark",
    "description": "This is a test benchmark for FLOPS 3.",
    "notes": null,
    "unstable": "Unstable FLOPS test!",
    "tags": [],
    "range_min": null,
    "range_max": null
  },
  "FLOPS 4": {
    "type": "benchmark",
    "description": "This is a test benchmark for FLOPS 4.",
    "notes": null,
    "unstable": "Unstable FLOPS test!",
    "tags": [],
    "range_min": null,
    "range_max": null
  },
  "FLOPS 5": {
    "type": "benchmark",
    "description": "This is a test benchmark for FLOPS 5.",
    "notes": null,
    "unstable": "Unstable FLOPS test!",
    "tags": [],
    "range_min": null,
    "range_max": null
  },
  "FLOPS 6": {
    "type": "benchmark",
    "description": "This is a test benchmark for FLOPS 6.",
    "notes": null,
    "unstable": "Unstable FLOPS test!",
    "tags": [],
    "range_min": null,
    "range_max": null
  },
  "Cache Miss Rate 1": {
    "type": "benchmark",
    "description": "This is a test benchmark for Cache Miss Rate 1.",
    "notes": "Test Note",
    "unstable": "And another note!",
    "tags": [],
    "range_min": null,
    "range_max": null
  },
  "Cache Miss Rate 2": {
    "type": "benchmark",
    "description": "This is a test benchmark for Cache Miss Rate 2.",
    "notes": "Test Note",
    "unstable": "And another note!",
    "tags": [],
    "range_min": null,
    "range_max": null
  },
  "Cache Miss Rate 3": {
    "type": "benchmark",
    "description": "This is a test benchmark for Cache Miss Rate 3.",
    "notes": "Test Note",
    "unstable": "And another note!",
    "tags": [],
    "range_min": null,
    "range_max": null
  },
  "Cache Miss Rate 4": {
    "type": "benchmark",
    "description": "This is a test benchmark for Cache Miss Rate 4.",
    "notes": "Test Note",
    "unstable": "And another note!",
    "tags": [],
    "range_min": null,
    "range_max": null
  },
  "Cache Miss Rate 5": {
    "type": "benchmark",
    "description": "This is a test benchmark for Cache Miss Rate 5.",
    "notes": "Test Note",
    "unstable": "And another note!",
    "tags": [],
    "range_min": null,
    "range_max": null
  },
  "Cache Miss Rate 6": {
    "type": "benchmark",
    "description": "This is a test benchmark for Cache Miss Rate 6.",
    "notes": "Test Note",
    "unstable": "And another note!",
    "tags": [],
    "range_min": null,
    "range_max": null
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
  "results"
];
