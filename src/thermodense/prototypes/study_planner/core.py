"""Pure planning and simulated dispatch logic for the throwaway study planner."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import product
from pathlib import Path
import re
import tomllib
from typing import Any


RESOURCE_CLASSES = {
    "dependence": "small-cpu-prep",
    "direct_trend": "small-cpu-prep",
}
ESTIMATED_TICKS = {
    "small-cpu-prep": 1,
    "cpu-parcorr": 3, "heavy-nonlinear-cpu": 5, "gpu-method": 2,
}
PCMCI_TEST_SPECS = {
    "parcorr": ("cpu-parcorr", "parcorr", 1, False),
    "cmiknn": ("heavy-nonlinear-cpu", "cmiknn", 1, False),
    "gpdc": ("heavy-nonlinear-cpu", "gpdc", 1, False),
    "gpdctorch": ("gpu-method", "gpdctorch", 1, True),
}


@dataclass(frozen=True)
class Artifact:
    id: str
    fingerprint: str
    kind: str
    label: str
    inputs: tuple[str, ...]


@dataclass(frozen=True)
class Case:
    id: str
    target_id: str
    target_kind: str
    target_value: str
    analysis: str
    independence_test: str | None
    cadence: str
    solar_proxy: str
    solar_variant_rows: str
    geomagnetic_driver: str
    window: str
    case_type: str
    extension: str | None
    altitude_group: tuple[int, ...] | None
    preprocessing_profile: str | None
    physical_lags: tuple[str, str] | None
    lag_steps: tuple[int, int] | None
    artifacts: tuple[str, ...]


@dataclass
class Job:
    id: str
    case_id: str
    resource_class: str
    data_locality: str
    estimated_ticks: int
    cpu_slots: int
    gpu_required: bool
    required_capabilities: tuple[str, ...]
    status: str = "queued"
    attempt: int = 0
    executor_id: str | None = None
    remaining_ticks: float | None = None
    blocked_reason: str | None = None


@dataclass(frozen=True)
class ExecutorPool:
    id: str
    adapter_kind: str
    enabled: bool
    slots: int
    resource_classes: tuple[str, ...]
    data_localities: tuple[str, ...]
    capabilities: tuple[str, ...]
    tick_multipliers: dict[str, float]
    quota_group: str
    quota_limit: int
    available: bool
    minimum_job_ticks: int = 0
    startup_ticks: float = 0.0
    staging_ticks: float = 0.0


@dataclass
class Controller:
    jobs: list[Job]
    pools: list[ExecutorPool]
    tick_number: int = 0

    def candidates(self, job: Job) -> list[ExecutorPool]:
        return [
            pool
            for pool in self.pools
            if pool.enabled and pool.available
            and job.estimated_ticks >= pool.minimum_job_ticks
            and job.resource_class in pool.resource_classes
            and job.data_locality in pool.data_localities
            and set(job.required_capabilities).issubset(pool.capabilities)
            and self._used_cpu_slots(pool.id) + job.cpu_slots <= pool.slots
            and self._used_quota_slots(pool.quota_group) + job.cpu_slots <= pool.quota_limit
        ]

    def tick(self) -> None:
        """Advance the simulated controller; no job is executed or submitted."""
        self.tick_number += 1
        for job in self.jobs:
            if job.status == "running" and job.remaining_ticks is not None:
                job.remaining_ticks -= 1
                if job.remaining_ticks <= 0:
                    job.status = "completed"
                    job.executor_id = None
        for job in sorted(
            (job for job in self.jobs if job.status == "queued"),
            key=lambda job: (-job.estimated_ticks, job.id),
        ):
            candidates = self.candidates(job)
            if not candidates:
                continue
            pool = min(
                candidates,
                key=lambda item: (self.projected_completion(job, item), item.id),
            )
            job.status = "running"
            job.attempt += 1
            job.executor_id = pool.id
            job.remaining_ticks = self._adjusted_duration(job, pool)

    def fail_first_running(self) -> None:
        running = next((job for job in self.jobs if job.status == "running"), None)
        if running is not None:
            running.status = "queued"
            running.executor_id = None
            running.remaining_ticks = None

    def state(self) -> dict[str, Any]:
        """Return a JSON-serializable durable-controller representation."""
        return {"tick_number": self.tick_number, "jobs": [asdict(job) for job in self.jobs]}

    @classmethod
    def reconcile(
        cls, state: dict[str, Any], pools: list[ExecutorPool]) -> Controller:
        """Rebuild controller state and requeue allocations to missing pools."""
        pool_ids = {pool.id for pool in pools}
        jobs = [Job(**item) for item in state["jobs"]]
        for job in jobs:
            if job.status == "running" and job.executor_id not in pool_ids:
                job.status, job.executor_id, job.remaining_ticks = "queued", None, None
        return cls(jobs=jobs, pools=pools, tick_number=state["tick_number"])

    def projected_completion(self, job: Job, pool: ExecutorPool) -> float:
        """Estimate finish time from normalized active load plus pool-relative duration."""
        return (self._load(pool.id) / pool.slots + self._adjusted_duration(job, pool)
                + pool.startup_ticks + pool.staging_ticks)

    def _adjusted_duration(self, job: Job, pool: ExecutorPool) -> float:
        return job.estimated_ticks * pool.tick_multipliers.get(job.resource_class, 1.0)

    def _used_cpu_slots(self, pool_id: str) -> int:
        return sum(
            job.cpu_slots
            for job in self.jobs
            if job.status == "running" and job.executor_id == pool_id
        )

    def _used_quota_slots(self, quota_group: str) -> int:
        pool_groups = {pool.id: pool.quota_group for pool in self.pools}
        return sum(
            job.cpu_slots
            for job in self.jobs
            if job.status == "running" and pool_groups.get(job.executor_id) == quota_group
        )

    def _load(self, pool_id: str) -> float:
        return sum(
            (job.remaining_ticks or 0) * job.cpu_slots
            for job in self.jobs
            if job.status == "running" and job.executor_id == pool_id
        )


@dataclass(frozen=True)
class Plan:
    study_id: str
    solar_variant_rows: str
    artifacts: tuple[Artifact, ...]
    cases: tuple[Case, ...]
    jobs: tuple[Job, ...]


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        return tomllib.load(file)


def load_pools(path: Path) -> list[ExecutorPool]:
    data = load_toml(path)
    return [
        ExecutorPool(
            id=item["id"],
            adapter_kind=item["adapter_kind"],
            enabled=item["enabled"],
            slots=item["slots"],
            resource_classes=tuple(item["resource_classes"]),
            data_localities=tuple(item["data_localities"]),
            capabilities=tuple(item.get("capabilities", [])),
            tick_multipliers=item.get("tick_multipliers", {}),
            quota_group=item["quota_group"],
            quota_limit=item["quota_limit"],
            available=item.get("available", True),
            minimum_job_ticks=item.get("minimum_job_ticks", 0),
            startup_ticks=item.get("startup_ticks", 0.0),
            staging_ticks=item.get("staging_ticks", 0.0),
        )
        for item in data["pools"]
    ]


def plan_study(data: dict[str, Any], pools: list[ExecutorPool]) -> Plan:
    defaults = data["defaults"]
    artifacts: dict[str, Artifact] = {}
    cases: list[Case] = []
    jobs: list[Job] = []
    for target in data["targets"]:
        analyses = _analyses(target, data.get("presets", {}))
        values = target["products"]
        for value, cadence, solar, geomagnetic, analysis in product(
            values, defaults["cadences"], defaults["solar_proxies"], defaults["geomagnetic_drivers"], analyses
        ):
            groups = target.get("pcmci_altitude_groups", [[]]) if analysis == "pcmci" else [[]]
            profiles = defaults["pcmci_preprocessing_profiles"] if analysis == "pcmci" else [None]
            tests = target.get("independence_tests", []) if analysis == "pcmci" else [None]
            for group, profile, test in product(groups, profiles, tests):
                case_window = target.get("window", defaults["window"])
                case_type = "full-core"
                if target.get("extension"):
                    _append_case(
                        artifacts, cases, jobs, target, value, cadence, solar, geomagnetic, analysis,
                        case_window, group, profile, test, "full-core", defaults["solar_variant_rows"], pools,
                    )
                    _append_case(
                        artifacts, cases, jobs, target, value, cadence, solar, geomagnetic, analysis,
                        target["extension"]["overlap_window"], group, profile, test, "overlap-core", defaults["solar_variant_rows"], pools,
                    )
                    case_window = target["extension"]["overlap_window"]
                    case_type = "extension"
                _append_case(
                    artifacts, cases, jobs, target, value, cadence, solar, geomagnetic, analysis,
                    case_window, group, profile, test, case_type, defaults["solar_variant_rows"], pools,
                )
    return Plan(
        study_id=data["study"]["id"],
        solar_variant_rows=defaults["solar_variant_rows"],
        artifacts=tuple(artifacts.values()),
        cases=tuple(cases),
        jobs=tuple(jobs),
    )


def _analyses(target: dict[str, Any], presets: dict[str, list[str]]) -> list[str]:
    if "preset" in target and "analyses" in target:
        raise ValueError("A target must specify either preset or analyses, not both.")
    if "preset" in target:
        return presets[target["preset"]]
    return target.get("analyses", [])


def _append_case(
    artifacts: dict[str, Artifact],
    cases: list[Case],
    jobs: list[Job],
    target: dict[str, Any],
    value: str,
    cadence: str,
    solar: str,
    geomagnetic: str,
    analysis: str,
    window: str,
    group: list[int],
    profile: str | None,
    independence_test: str | None,
    case_type: str,
    solar_variant_rows: str,
    pools: list[ExecutorPool],
) -> None:
    source = _artifact(artifacts, "acquire", target["source_id"], ())
    prepared = _artifact(
        artifacts, "prepare", f"{target['source_id']}:{cadence}:{window}", (source.id,)
    )
    drivers = _artifact(
        artifacts,
        "drivers",
        f"{solar}:{geomagnetic}:{cadence}:{window}:{solar_variant_rows}",
        (),
    )
    group_id = "-".join(map(str, group)) if group else "all-altitudes"
    physical_lags = _physical_lags(target) if analysis == "pcmci" else None
    lag_steps = _lag_steps(physical_lags, cadence) if physical_lags else None
    case_id = _readable_id(
        target["id"], value, analysis, cadence, solar, geomagnetic, case_type, group_id, profile or "no-profile"
        , independence_test or "no-test"
    )
    extension = target.get("extension", {}).get("name") if case_type == "extension" else None
    target_artifacts = _target_artifacts(artifacts, target, value, cadence, window, prepared.id)
    inputs = [target_artifacts[-1].id, drivers.id]
    extension_artifacts: tuple[str, ...] = ()
    if extension:
        extension_source_id = target["extension"]["source_id"]
        extension_source = _artifact(artifacts, "acquire", extension_source_id, ())
        extension_prepared = _artifact(
            artifacts, "prepare", f"{extension_source_id}:{cadence}:{window}", (extension_source.id,)
        )
        inputs.append(extension_prepared.id)
        extension_artifacts = (extension_source.id, extension_prepared.id)
    result = _artifact(
        artifacts, "analysis-result", case_id, tuple(inputs)
    )
    case = Case(
        id=case_id,
        target_id=target["id"],
        target_kind=target["kind"],
        target_value=value,
        analysis=analysis,
        independence_test=independence_test,
        cadence=cadence,
        solar_proxy=solar,
        solar_variant_rows=solar_variant_rows,
        geomagnetic_driver=geomagnetic,
        window=window,
        case_type=case_type,
        extension=extension,
        altitude_group=tuple(group) or None,
        preprocessing_profile=profile,
        physical_lags=physical_lags,
        lag_steps=lag_steps,
        artifacts=(source.id, prepared.id, *(item.id for item in target_artifacts), drivers.id, *extension_artifacts, result.id),
    )
    resource_class, capability, cpu_slots, gpu_required = _resource_spec(analysis, independence_test)
    job = Job(
        id=f"job-{len(jobs) + 1:03d}", case_id=case.id, resource_class=resource_class,
        data_locality=target.get("data_locality", "shared"),
        estimated_ticks=ESTIMATED_TICKS[resource_class],
        cpu_slots=cpu_slots,
        gpu_required=gpu_required,
        required_capabilities=(capability,) if capability else (),
    )
    if not Controller([job], pools).candidates(job):
        job.status, job.blocked_reason = "gated", "no compatible available executor"
    cases.append(case)
    jobs.append(job)


def _target_artifacts(
    artifacts: dict[str, Artifact],
    target: dict[str, Any],
    value: str,
    cadence: str,
    window: str,
    prepared_id: str,
) -> tuple[Artifact, ...]:
    specification = f"{value}:{cadence}:{window}"
    if target["kind"] == "density":
        return (_artifact(artifacts, "diagnostic", specification, (prepared_id,)),)
    evaluation = _artifact(artifacts, "model-evaluation", specification, (prepared_id,))
    if target["kind"] == "model_density":
        return (evaluation,)
    if target["kind"] == "model_error":
        error = _artifact(
            artifacts,
            "log-density-ratio-error",
            f"{value}:ln-model-over-reference:{cadence}:{window}",
            (evaluation.id, prepared_id),
        )
        return evaluation, error
    raise ValueError(f"Unknown target kind {target['kind']!r}.")


def _resource_spec(analysis: str, independence_test: str | None) -> tuple[str, str | None, int, bool]:
    if analysis == "pcmci":
        if independence_test not in PCMCI_TEST_SPECS:
            raise ValueError(f"Unknown PCMCI independence test {independence_test!r}.")
        return PCMCI_TEST_SPECS[independence_test]
    resource_class = RESOURCE_CLASSES[analysis]
    return resource_class, None, 1, False


def _artifact(
    artifacts: dict[str, Artifact], kind: str, specification: str, inputs: tuple[str, ...]
) -> Artifact:
    fingerprint = sha256(f"{kind}|{specification}|{'|'.join(inputs)}|prototype-v1".encode()).hexdigest()[:12]
    artifact = Artifact(f"{kind}-{fingerprint}", fingerprint, kind, specification, inputs)
    return artifacts.setdefault(artifact.id, artifact)


def _physical_lags(target: dict[str, Any]) -> tuple[str, str]:
    return target["min_lag"], target["max_lag"]


def _lag_steps(lags: tuple[str, str], cadence: str) -> tuple[int, int]:
    hours = tuple(_duration_hours(value) for value in lags)
    cadence_hours = {"daily": 24, "3-hour": 3}[cadence]
    steps = tuple(value // cadence_hours for value in hours)
    if any(value % cadence_hours for value in hours):
        raise ValueError("PCMCI lag duration must fall on a cadence boundary.")
    return steps


def _duration_hours(value: str) -> int:
    match = re.fullmatch(r"(\d+)([hd])", value)
    if not match:
        raise ValueError(f"Unsupported duration {value!r}; use whole h or d values.")
    amount, unit = match.groups()
    return int(amount) * (24 if unit == "d" else 1)


def _readable_id(*parts: str) -> str:
    return "-".join(str(part).replace("_", "-").lower() for part in parts)
