# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Benchmarks built from the SYCL E2E PerformanceTests sources.

The tests in sycl/test-e2e/PerformanceTests are regular lit tests. This suite
reads a strict subset of their RUN lines (see RunLineReader), builds the
binaries itself and collects the PERF_RESULT lines they print. Lit is not
involved. Anything outside the supported subset is reported with its
file:line and fails the suite setup.
"""

import json
import os
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path

from .base import Benchmark, Suite
from options import options
from utils.logger import log
from utils.result import BenchmarkMetadata, Result
from utils.utils import run

PERF_RESULT_PREFIX = "PERF_RESULT "
PERF_RESULT_KEYS = (
    "name",
    "variant",
    "unit",
    "value",
    "stddev",
    "min",
    "reps",
    "lower_is_better",
)
# Characters allowed in case names, variants and compile names.
NAME_RE = re.compile(r"[A-Za-z0-9 _.,:;<>()=+/*&#-]*")
# Characters allowed in paths substituted into RUN lines. Paths are inserted
# before shlex splitting, so they must not need quoting.
SAFE_PATH_RE = re.compile(r"[A-Za-z0-9_./+,=@:-]+")
SFX = r"[A-Za-z0-9_.-]*"

BUILD_RE = re.compile(
    r"%\{build\}(?P<flags>(?:\s.*?)?)\s+-o\s+%t(?P<sfx>" + SFX + r")\.out"
)
RUN_RE = re.compile(r"%\{run\}\s+%t(?P<sfx>" + SFX + r")\.out(?P<args>(?:\s.*)?)")
COMPILE_RE = re.compile(
    r"python3\s+%S/(?P<script>(?:[A-Za-z0-9_.-]+/)*perf_compile\.py)"
    r"(?P<opts>(?:\s.*?)?)\s+--\s+(?P<cmd>%\{build\}(?:\s.*)?)"
)
DIRECTIVE_RE = re.compile(r"^\s*//\s*(?P<key>[A-Z][A-Z_-]*):(?P<val>.*)$")
# Directive names (and prefixes) that lit or the E2E format interpret.
LIT_DIRECTIVES = (
    "RUN",
    "REQUIRES",
    "UNSUPPORTED",
    "XFAIL",
    "DEFINE",
    "REDEFINE",
    "ALLOW_RETRIES",
    "END",
)
SUBST_RE = re.compile(r"%(\{[A-Za-z0-9_-]+\}|/t|[tsS])")
FEATURE_TOKEN_RE = re.compile(r"\s*(&&|\|\||!|\(|\)|[-+=._a-zA-Z0-9]+)")

# Variables that must not leak into compiles: they change include/library
# search paths, wrap the compiler, or tune the SYCL runtime used by the
# compiler (e.g. for kernel_compiler tests).
SCRUBBED_ENV_PREFIXES = ("CCACHE", "SYCL_")
SCRUBBED_ENV_VARS = ("CPATH", "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH", "LIBRARY_PATH")

# lit.cfg.py device_family_arch_map plus PVC, keyed by sycl-ls architecture.
DEVICE_FAMILIES = {
    "gpu-intel-dg2": {
        "intel_gpu_acm_g12",
        "intel_gpu_dg2_g12",
        "intel_gpu_acm_g11",
        "intel_gpu_dg2_g11",
        "intel_gpu_acm_g10",
        "intel_gpu_dg2_g10",
    },
    "gpu-intel-gen12": {
        "intel_gpu_tgllp",
        "intel_gpu_tgl",
        "intel_gpu_rkl",
        "intel_gpu_adl_s",
        "intel_gpu_rpl_s",
        "intel_gpu_adl_p",
        "intel_gpu_adl_n",
        "intel_gpu_mtl_h",
    },
    "gpu-intel-gen11": {"intel_gpu_icllp", "intel_gpu_icl"},
    "gpu-intel-pvc": {"intel_gpu_pvc"},
    "gpu-intel-pvc-vg": {"intel_gpu_pvc_vg"},
}


class RunLineError(ValueError):
    """An unsupported or malformed directive in a test source."""

    def __init__(self, path: Path | str, line: int, msg: str):
        super().__init__(f"{path}:{line}: {msg}")


# ---------------------------------------------------------------------------
# PERF_RESULT parsing
# ---------------------------------------------------------------------------


@dataclass
class PerfResult:
    name: str
    variant: str
    unit: str
    value: float
    stddev: float
    min: float
    reps: int
    lower_is_better: bool


def parse_perf_results(output: str) -> list[PerfResult]:
    """Return the PERF_RESULT lines in output; other lines are ignored.

    Raises ValueError for a malformed PERF_RESULT line or if there are none.
    """
    results = []
    for line in output.splitlines():
        if not line.startswith(PERF_RESULT_PREFIX):
            continue
        payload = line[len(PERF_RESULT_PREFIX) :]
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as e:
            raise ValueError(f"malformed PERF_RESULT line ({e}): {line}")
        if not isinstance(data, dict):
            raise ValueError(f"PERF_RESULT is not a JSON object: {line}")
        missing = [k for k in PERF_RESULT_KEYS if k not in data]
        if missing:
            raise ValueError(f"PERF_RESULT misses keys {missing}: {line}")
        for key in ("name", "variant", "unit"):
            if not isinstance(data[key], str):
                raise ValueError(f"PERF_RESULT '{key}' is not a string: {line}")
        for key in ("name", "variant"):
            if not NAME_RE.fullmatch(data[key]):
                raise ValueError(f"PERF_RESULT '{key}' has invalid characters: {line}")
        if not data["name"]:
            raise ValueError(f"PERF_RESULT has an empty name: {line}")
        for key in ("value", "stddev", "min"):
            if isinstance(data[key], bool) or not isinstance(data[key], (int, float)):
                raise ValueError(f"PERF_RESULT '{key}' is not a number: {line}")
        if isinstance(data["reps"], bool) or not isinstance(data["reps"], int):
            raise ValueError(f"PERF_RESULT 'reps' is not an integer: {line}")
        if not isinstance(data["lower_is_better"], bool):
            raise ValueError(f"PERF_RESULT 'lower_is_better' is not a bool: {line}")
        results.append(PerfResult(**{k: data[k] for k in PERF_RESULT_KEYS}))
    if not results:
        raise ValueError("no PERF_RESULT lines in the output")
    return results


# ---------------------------------------------------------------------------
# Feature expressions (REQUIRES / UNSUPPORTED)
# ---------------------------------------------------------------------------


def available_features(device_architecture: str = "") -> set[str]:
    """Features the suite's runs provide, mirroring the lit feature names."""
    features = {"linux", "gpu", "level_zero", "enable-perf-tests"}
    if device_architecture:
        features.add(f"arch-{device_architecture}")
        for family, archs in DEVICE_FAMILIES.items():
            if device_architecture in archs:
                features.add(family)
    return features


def eval_feature_expr(expr: str, features: set[str]) -> bool:
    """Evaluate a lit boolean feature expression (&&, ||, !, parentheses).

    Unknown features evaluate to false. Raises ValueError on syntax errors.
    """
    tokens = []
    pos = 0
    text = expr.rstrip()
    while pos < len(text):
        m = FEATURE_TOKEN_RE.match(text, pos)
        if not m:
            raise ValueError(f"unsupported feature expression: '{expr}'")
        tokens.append(m.group(1))
        pos = m.end()
    if not tokens:
        raise ValueError("empty feature expression")

    def peek():
        return tokens[0] if tokens else None

    def take(expected=None):
        if not tokens or (expected and tokens[0] != expected):
            raise ValueError(f"malformed feature expression: '{expr}'")
        return tokens.pop(0)

    def parse_or():
        value = parse_and()
        while peek() == "||":
            take()
            rhs = parse_and()
            value = value or rhs
        return value

    def parse_and():
        value = parse_not()
        while peek() == "&&":
            take()
            rhs = parse_not()
            value = value and rhs
        return value

    def parse_not():
        tok = take()
        if tok == "!":
            return not parse_not()
        if tok == "(":
            value = parse_or()
            take(")")
            return value
        if tok in ("&&", "||", ")"):
            raise ValueError(f"malformed feature expression: '{expr}'")
        if tok == "true":
            return True
        if tok == "false":
            return False
        return tok in features

    value = parse_or()
    if tokens:
        raise ValueError(f"malformed feature expression: '{expr}'")
    return value


def eval_feature_list(text: str, features: set[str]) -> list[bool]:
    """Evaluate a comma-separated list of feature expressions."""
    return [eval_feature_expr(item, features) for item in text.split(",")]


# ---------------------------------------------------------------------------
# RUN-line reader
# ---------------------------------------------------------------------------


@dataclass
class BuildStep:
    line: int
    command: str  # starts with %{build}
    sfx: str


@dataclass
class RunStep:
    line: int
    sfx: str
    args: str
    build: BuildStep
    variant: str = ""


@dataclass
class CompileStep:
    line: int
    name: str
    script: str  # relative to %S
    command: str  # starts with %{build}


@dataclass
class PerfTest:
    path: Path
    test_id: str  # path relative to the tests dir, without .cpp
    description: str = ""
    requires: list[tuple[int, str]] = field(default_factory=list)
    unsupported: list[tuple[int, str]] = field(default_factory=list)
    xfail: list[str] = field(default_factory=list)
    builds: list[BuildStep] = field(default_factory=list)
    runs: list[RunStep] = field(default_factory=list)
    compiles: list[CompileStep] = field(default_factory=list)

    def skip_reason(self, features: set[str]) -> str | None:
        """Why the test does not apply to features, or None if it does."""
        for line, text in self.requires:
            try:
                ok = all(eval_feature_list(text, features))
            except ValueError as e:
                raise RunLineError(self.path, line, str(e))
            if not ok:
                return f"REQUIRES: {text.strip()} not satisfied"
        for line, text in self.unsupported:
            try:
                hit = any(eval_feature_list(text, features))
            except ValueError as e:
                raise RunLineError(self.path, line, str(e))
            if hit:
                return f"UNSUPPORTED: {text.strip()}"
        return None


def drop_if_segments(text: str) -> str:
    """Remove every `%if <cond> %{ ... %}` segment from a RUN command."""
    out = []
    pos = 0
    while True:
        start = text.find("%if", pos)
        if start < 0:
            out.append(text[pos:])
            break
        out.append(text[pos:start])
        body = text.find("%{", start)
        if body < 0 or not re.fullmatch(r"%if\s+\S.*?\s*", text[start:body]):
            raise ValueError("malformed %if segment")
        depth = 0
        i = body
        while i < len(text):
            if re.match(r"%\{[A-Za-z0-9_-]+\}", text[i:]):
                i = text.index("}", i) + 1  # a substitution like %{amd_arch}
                continue
            if text.startswith("%{", i):
                depth += 1
                i += 2
                continue
            if text.startswith("%}", i):
                depth -= 1
                i += 2
                if depth == 0:
                    break
                continue
            i += 1
        if depth != 0:
            raise ValueError("unterminated %if segment")
        if re.match(r"\s*%else\b", text[i:]):
            raise ValueError("%else is not supported")
        pos = i
    return "".join(out).strip()


def read_perf_test(path: Path, test_id: str) -> PerfTest:
    """Parse the directives of one test source. Raises RunLineError."""
    test = PerfTest(path=path, test_id=test_id)
    commands: list[tuple[int, str]] = []
    pending: tuple[int, str] | None = None

    lines = path.read_text(errors="replace").splitlines()
    for lineno, raw in enumerate(lines, start=1):
        m = DIRECTIVE_RE.match(raw)
        key = m.group("key") if m else None
        if key != "RUN" and pending is not None:
            raise RunLineError(
                path, pending[0], "RUN line continuation is not followed by a RUN line"
            )
        if m is None:
            if re.search(r"\b(RUN|REQUIRES|UNSUPPORTED|XFAIL):", raw):
                raise RunLineError(path, lineno, "directive must start with '// '")
            continue
        val = m.group("val")
        if key == "RUN":
            content = val.rstrip()
            if pending is not None:
                content = pending[1][:-1] + content
                start = pending[0]
            else:
                start = lineno
            if content.endswith("\\"):
                pending = (start, content)
            else:
                pending = None
                commands.append((start, content.strip()))
        elif key == "REQUIRES":
            test.requires.append((lineno, val))
        elif key == "UNSUPPORTED":
            test.unsupported.append((lineno, val))
        elif key == "XFAIL":
            test.xfail.append(val.strip())
        elif key == "PERF-DESCRIPTION":
            if not test.description:
                test.description = val.strip()
        elif key.endswith("-TRACKER") or key == "UNSUPPORTED-INTENDED":
            pass
        elif key in LIT_DIRECTIVES or key.startswith(
            tuple(d + "-" for d in LIT_DIRECTIVES)
        ):
            # e.g. DEFINE:, REDEFINE:, REQUIRES-INTEL-DRIVER:
            raise RunLineError(path, lineno, f"'{key}:' is not supported")
        # Other upper-case comment prefixes (e.g. "NOTE:", "TODO:") are text.
    if pending is not None:
        raise RunLineError(path, pending[0], "RUN line ends with a continuation")

    last_build: dict[str, BuildStep] = {}
    for lineno, text in commands:
        try:
            text = drop_if_segments(text)
        except ValueError as e:
            raise RunLineError(path, lineno, f"{e}: {text}")
        if not text:
            continue  # the line was only %if segments
        if m := BUILD_RE.fullmatch(text):
            sfx = m.group("sfx")
            prev = last_build.get(sfx)
            if prev is not None and prev.command != text:
                raise RunLineError(
                    path,
                    lineno,
                    f"%t{sfx}.out is already built differently at line {prev.line}",
                )
            step = BuildStep(lineno, text, sfx)
            test.builds.append(step)
            last_build[sfx] = step
        elif m := RUN_RE.fullmatch(text):
            sfx = m.group("sfx")
            if sfx not in last_build:
                raise RunLineError(
                    path,
                    lineno,
                    f"%t{sfx}.out is not built by a preceding %{{build}} line",
                )
            args = m.group("args").strip()
            step = RunStep(lineno, sfx, args, last_build[sfx])
            step.variant = _variant_of(path, lineno, args)
            test.runs.append(step)
        elif m := COMPILE_RE.fullmatch(text):
            name = _compile_name(path, lineno, m.group("opts"))
            test.compiles.append(
                CompileStep(lineno, name, m.group("script"), m.group("cmd"))
            )
        else:
            raise RunLineError(path, lineno, f"unsupported RUN line: {text}")
    return test


def _variant_of(path: Path, lineno: int, args: str) -> str:
    """The --variant value in the run args ('' if none)."""
    try:
        tokens = shlex.split(args)
    except ValueError as e:
        raise RunLineError(path, lineno, f"cannot split arguments ({e}): {args}")
    variants = [t[len("--variant=") :] for t in tokens if t.startswith("--variant=")]
    if len(variants) > 1:
        raise RunLineError(path, lineno, "--variant is given more than once")
    variant = variants[0] if variants else ""
    if not NAME_RE.fullmatch(variant):
        raise RunLineError(path, lineno, f"invalid characters in variant '{variant}'")
    return variant


def _compile_name(path: Path, lineno: int, opts: str) -> str:
    """The --name of a perf_compile.py line, validating its other options."""
    if "%" in opts:
        raise RunLineError(
            path,
            lineno,
            f"substitutions are not supported in perf_compile.py options: {opts}",
        )
    try:
        tokens = shlex.split(opts)
    except ValueError as e:
        raise RunLineError(path, lineno, f"cannot split perf_compile.py options ({e})")
    name = None
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        opt, eq, val = tok.partition("=")
        if opt not in ("--name", "--reps", "--warmup"):
            raise RunLineError(
                path, lineno, f"unsupported perf_compile.py option '{tok}'"
            )
        if not eq:
            if i + 1 >= len(tokens):
                raise RunLineError(path, lineno, f"missing value for {opt}")
            i += 1
            val = tokens[i]
        if opt == "--name":
            name = val
        elif not val.isdigit():
            raise RunLineError(path, lineno, f"{opt} needs an integer, got '{val}'")
        i += 1
    if not name:
        raise RunLineError(path, lineno, "perf_compile.py line has no --name")
    if not NAME_RE.fullmatch(name):
        raise RunLineError(path, lineno, f"invalid characters in name '{name}'")
    return name


class Substitutions:
    """Expands the supported lit substitutions for one test."""

    def __init__(self, test: PerfTest, sycl: str, tmp_prefix: Path):
        src = str(test.path.resolve())
        self.path = test.path
        self.values = {
            "{build}": f"{sycl}/bin/clang++ -Werror -fsycl -fsycl-targets=spir64 {src}",
            "s": src,
            "S": str(Path(src).parent),
            "t": str(tmp_prefix),
            "/t": str(tmp_prefix),
        }
        for value in (sycl, src, str(tmp_prefix)):
            if not SAFE_PATH_RE.fullmatch(value):
                raise ValueError(
                    f"{test.path}: path '{value}' contains characters that "
                    "would need quoting in a RUN line"
                )

    def expand(self, text: str, lineno: int) -> list[str]:
        """Substitute and split text into an argv list."""

        def repl(m: re.Match) -> str:
            key = m.group(1)
            if key not in self.values:
                raise RunLineError(
                    self.path, lineno, f"unsupported substitution '%{key}'"
                )
            return self.values[key]

        expanded = SUBST_RE.sub(repl, text)
        if "%" in expanded:
            raise RunLineError(
                self.path, lineno, f"unsupported substitution in: {text}"
            )
        try:
            return shlex.split(expanded)
        except ValueError as e:
            raise RunLineError(self.path, lineno, f"cannot split command ({e}): {text}")


# ---------------------------------------------------------------------------
# Suite and benchmarks
# ---------------------------------------------------------------------------


def default_tests_dir() -> Path:
    # <repo>/devops/scripts/benchmarks/benches/sycl_perf.py
    return (
        Path(__file__).resolve().parents[4] / "sycl" / "test-e2e" / "PerformanceTests"
    )


def scrubbed_env_names(env_vars: dict | None = None) -> list[str]:
    """Names of inherited/passed variables that must not reach a compile."""
    names = set(os.environ) | set(env_vars or {})
    return sorted(
        n
        for n in names
        if n in SCRUBBED_ENV_VARS or n.startswith(SCRUBBED_ENV_PREFIXES)
    )


def scrub_env_cmd(env_vars: dict | None = None) -> list[str]:
    """An `env -u ...` prefix that drops the scrubbed variables."""
    cmd = ["env"]
    for name in scrubbed_env_names(env_vars):
        cmd += ["-u", name]
    return cmd


class SyclPerfSuite(Suite):
    def __init__(self, tests_dir: str | None = None):
        self._tests_dir = tests_dir
        self._benchmarks: list[Benchmark] | None = None
        self._errors: list[str] = []
        self._built: set[str] = set()

    def name(self) -> str:
        return "SYCL Performance Tests"

    def git_url(self) -> str:
        return "https://github.com/intel/llvm.git"

    def git_hash(self) -> str:
        return options.git_commit_override or ""

    @property
    def tests_dir(self) -> Path:
        d = self._tests_dir or getattr(options, "sycl_perf_tests_dir", None)
        return Path(d).resolve() if d else default_tests_dir()

    @property
    def harness_header(self) -> Path:
        return self.tests_dir / "perf_harness.hpp"

    def work_dir(self) -> Path:
        return Path(options.workdir).resolve() / "sycl-perf"

    def tmp_prefix(self, test: PerfTest) -> Path:
        """%t for test: <workdir>/sycl-perf/<test-id>/t."""
        return self.work_dir() / test.test_id / "t"

    def substitutions(self, test: PerfTest) -> Substitutions:
        return Substitutions(
            test, options.sycl or "SYCL_ROOT_UNSET", self.tmp_prefix(test)
        )

    def setup(self) -> None:
        if options.sycl is None:
            raise RuntimeError(f"{self.name()} requires --sycl")
        self.benchmarks()
        if self._errors:
            raise RuntimeError("; ".join(self._errors))
        self.work_dir().mkdir(parents=True, exist_ok=True)

    def benchmarks(self) -> list[Benchmark]:
        if self._benchmarks is None:
            self._benchmarks = self._discover()
        return self._benchmarks

    def _discover(self) -> list[Benchmark]:
        tests_dir = self.tests_dir
        if not tests_dir.is_dir():
            log.warning(f"{self.name()}: tests directory {tests_dir} not found")
            return []
        features = available_features(getattr(options, "device_architecture", ""))
        benches: list[Benchmark] = []
        names: dict[str, str] = {}
        for path in sorted(tests_dir.rglob("*.cpp")):
            test_id = path.relative_to(tests_dir).with_suffix("").as_posix()
            try:
                test = read_perf_test(path, test_id)
                reason = test.skip_reason(features)
                if reason:
                    log.info(f"{self.name()}: skipping {test_id}: {reason}")
                    continue
                for xfail in test.xfail:
                    log.info(f"{self.name()}: {test_id} has XFAIL: {xfail} (ignored)")
                found = [SyclPerfRuntimeBench(self, test, r) for r in test.runs]
                found += [SyclCompileBench(self, test, c) for c in test.compiles]
                for b in found:
                    where = f"{path}:{b.step.line}"
                    if b.name() in names:
                        raise RunLineError(
                            path,
                            b.step.line,
                            f"duplicate benchmark name '{b.name()}' (also {names[b.name()]})",
                        )
                    names[b.name()] = where
            except RunLineError as e:
                log.error(f"{self.name()}: {e}")
                self._errors.append(str(e))
                continue
            benches += found
        return benches

    def additional_metadata(self) -> dict[str, BenchmarkMetadata]:
        groups: dict[str, BenchmarkMetadata] = {}
        for b in self.benchmarks():
            group = groups.setdefault(
                b.explicit_group(),
                BenchmarkMetadata(type="group", description=b.description(), tags=[]),
            )
            group.tags = sorted(set(group.tags) | set(b.get_tags()))
        return groups

    def build(self, test: PerfTest, step: BuildStep) -> Path:
        """Run a build step once per suite; return the produced binary."""
        subst = self.substitutions(test)
        binary = Path(f"{self.tmp_prefix(test)}{step.sfx}.out")
        command = subst.expand(step.command, step.line)
        key = f"{binary}\0{command}"
        if key in self._built:
            return binary
        stamp = binary.with_name(binary.name + ".cmd")
        if self._up_to_date(test, binary, stamp, command):
            log.info(f"{self.name()}: {binary} is up to date")
        else:
            binary.parent.mkdir(parents=True, exist_ok=True)
            stamp.unlink(missing_ok=True)
            run(
                scrub_env_cmd() + command,
                add_sycl=True,
                cwd=str(binary.parent),
                ld_library=options.extra_ld_libraries,
            )
            stamp.write_text(json.dumps(command))
        self._built.add(key)
        return binary

    def _up_to_date(
        self, test: PerfTest, binary: Path, stamp: Path, command: list[str]
    ) -> bool:
        if not binary.is_file():
            return False
        if options.offline:
            return True
        try:
            if json.loads(stamp.read_text()) != command:
                return False
        except (OSError, ValueError):
            return False
        deps = [
            test.path,
            self.harness_header,
            Path(options.sycl) / "lib" / "libsycl.so",
        ]
        newest = max((d.stat().st_mtime for d in deps if d.exists()), default=0)
        return binary.stat().st_mtime > newest


class SyclPerfBenchmark(Benchmark):
    """Common metadata for benchmarks derived from one RUN line."""

    def __init__(self, suite: SyclPerfSuite, test: PerfTest, step):
        super().__init__(suite)
        self.suite: SyclPerfSuite = suite
        self.test = test
        self.step = step

    def enabled(self) -> bool:
        return options.sycl is not None

    def explicit_group(self) -> str:
        return self.test.test_id

    def description(self) -> str:
        return self.test.description

    def _result(
        self, label: str, r: PerfResult, command: list[str], env: dict
    ) -> Result:
        return Result(
            label=label,
            value=r.value,
            unit=r.unit,
            stddev=r.stddev,
            command=command,
            env=env,
            git_url=self.suite.git_url(),
            git_hash=self.suite.git_hash(),
        )


class SyclPerfRuntimeBench(SyclPerfBenchmark):
    """One `%{run} %t<sfx>.out [args]` line; one Result per printed case."""

    step: RunStep

    def __init__(self, suite: SyclPerfSuite, test: PerfTest, step: RunStep):
        super().__init__(suite, test, step)
        self.binary: Path | None = None

    def name(self) -> str:
        name = f"sycl_perf {self.test.test_id}"
        return f"{name} {self.step.variant}" if self.step.variant else name

    def get_tags(self) -> list[str]:
        return ["SYCL", "micro", "latency"]

    def stddev_threshold(self):
        return 0.05 if self.test.test_id.startswith("KernelCompiler/") else None

    def args(self) -> list[str]:
        return self.suite.substitutions(self.test).expand(
            self.step.args, self.step.line
        )

    def setup(self):
        self.binary = self.suite.build(self.test, self.step.build)
        listing = run([str(self.binary), "--list", *self.args()], add_sycl=True)
        cases = [l for l in listing.stdout.decode().splitlines() if l.strip()]
        log.info(f"{self.name()}: {len(cases)} cases")
        log.debug(f"{self.name()} cases: {', '.join(cases)}")

    def run(
        self,
        env_vars,
        flamegraph_enabled: bool = False,
        force_trace: bool = False,
    ) -> list[Result]:
        if self.binary is None:
            self.binary = Path(
                f"{self.suite.tmp_prefix(self.test)}{self.step.build.sfx}.out"
            )
        command = [str(self.binary), *self.args()]
        env_vars = dict(env_vars) if env_vars else {}
        output = self.run_bench(
            command,
            env_vars,
            flamegraph_enabled=flamegraph_enabled,
            force_trace=force_trace,
        )
        results = []
        seen = set()
        for r in parse_perf_results(output):
            if r.variant != self.step.variant:
                log.warning(
                    f"{self.name()}: case '{r.name}' reports variant '{r.variant}', "
                    f"expected '{self.step.variant}'"
                )
            if r.name in seen:
                raise ValueError(f"{self.name()}: case '{r.name}' reported twice")
            seen.add(r.name)
            results.append(
                self._result(f"{self.name()} {r.name}", r, command, env_vars)
            )
        return results


class SyclCompileBench(SyclPerfBenchmark):
    """One `python3 %S/.../perf_compile.py --name <n> -- %{build} ...` line."""

    step: CompileStep

    def name(self) -> str:
        return f"sycl_compile {self.step.name}"

    def get_tags(self) -> list[str]:
        return ["SYCL", "compile"]

    def stddev_threshold(self):
        return 0.05

    def traceable(self) -> bool:
        return False

    def compiler_command(self) -> list[str]:
        return self.suite.substitutions(self.test).expand(
            self.step.command, self.step.line
        )

    def script(self) -> Path:
        return (self.test.path.parent / self.step.script).resolve()

    def setup(self):
        self.suite.tmp_prefix(self.test).parent.mkdir(parents=True, exist_ok=True)
        # Untimed warm-up: fills the page cache with headers and the toolchain.
        run(
            scrub_env_cmd() + self.compiler_command(),
            add_sycl=True,
            cwd=str(self.suite.tmp_prefix(self.test).parent),
            ld_library=options.extra_ld_libraries,
        )

    def run(
        self,
        env_vars,
        flamegraph_enabled: bool = False,
        force_trace: bool = False,
    ) -> list[Result]:
        env_vars = dict(env_vars) if env_vars else {}
        command = [
            "python3",
            str(self.script()),
            "--name",
            self.step.name,
            "--warmup",
            "0",
            "--reps",
            "1",
            "--",
            *self.compiler_command(),
        ]
        # Not run_bench(): UR_ADAPTERS_FORCE_LOAD is irrelevant for a compile,
        # but pin to the same cores as the runtime benchmarks.
        result = run(
            self.taskset_cmd() + scrub_env_cmd(env_vars) + command,
            env_vars=env_vars,
            add_sycl=True,
            cwd=str(self.suite.tmp_prefix(self.test).parent),
            ld_library=options.extra_ld_libraries,
        )
        by_name = {r.name: r for r in parse_perf_results(result.stdout.decode())}
        env = {
            k: v for k, v in env_vars.items() if k not in scrubbed_env_names(env_vars)
        }
        results = []
        for suffix in ("", ", CPU time"):
            r = by_name.get(self.step.name + suffix)
            if r is None:
                raise ValueError(
                    f"{self.name()}: perf_compile.py did not report '{self.step.name}{suffix}'"
                )
            results.append(self._result(self.name() + suffix, r, command, env))
        return results
