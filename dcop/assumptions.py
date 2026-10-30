"""Assumptions dump, source flags, and cost-line tracing (spec §8 output 7, §9, §11).

``build_audit`` collects every input a run used with its source and date, flags the
ones that are placeholders, stale, or untagged, and maps each ledger cost line to the
inputs that drive it. ``render_markdown`` writes the result as ``assumptions.md``.

Conventions
- A source that contains "placeholder" or "synthetic" marks the input as a placeholder.
- An input is stale when its ``as_of`` is more than ``stale_days`` before the run date.
- An input with no source or no ``as_of`` is untagged.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import dcop
from dcop.io import Inputs, resolve_ref
from dcop.scenarios import RunResult, apply_overrides, get_path, inputs_tree
from dcop.schemas import Distributions, Site

PLACEHOLDER_WORDS = ("placeholder", "synthetic")
STRICT_FLAGS = {"placeholder", "stale", "no source", "no as_of", "unresolved", "missing file"}


def is_placeholder(source: str | None) -> bool:
    return bool(source) and any(w in source.lower() for w in PLACEHOLDER_WORDS)


def source_flags(source: str | None, as_of: date | None, today: date, stale_days: int) -> tuple[str, ...]:
    flags: list[str] = []
    if not source:
        flags.append("no source")
    if as_of is None:
        flags.append("no as_of")
    if is_placeholder(source):
        flags.append("placeholder")
    if as_of is not None and (today - as_of).days > stale_days:
        flags.append("stale")
    return tuple(flags)


# ---- cost line to inputs (the trace) ----
_LOAD = ["sites.{site}.load_mw_peak", "sites.{site}.load_mw_ramp", "sites.{site}.load_factor",
         "sites.{site}.load_profile_ref"]
_UNSERVED = ["global.value_of_unserved_usd_mwh", "global.clock_start", *_LOAD]
_PLANT_SIZE = ["plants.{plant}.unit_mw_iso", "plants.{plant}.n_units", "plants.{plant}.redundancy"]
_HEAT_RATE = ["plants.{plant}.heat_rate_mmbtu_mwh_hhv_iso", "plants.{plant}.heat_rate_part_load_curve_ref",
              "plants.{plant}.degradation_pct_yr"]
_AVAILABILITY = ["plants.{plant}.forced_outage_rate", "plants.{plant}.planned_outage_days_yr"]
_BTM_TIMING = ["plants.{plant}.permit_lead_months", "plants.{plant}.cod_lead_months",
               "gas.{hub}.lateral_lead_months", "global.clock_start"]
_DERATES = ["plants.{plant}.ambient_derate_pct_per_10c", "plants.{plant}.elevation_derate_pct_per_1000m",
            "sites.{site}.ambient_c_design", "sites.{site}.elevation_m"]

COST_LINE_INPUTS: dict[str, dict[str, list[str]]] = {
    "grid": {
        "energy": ["tariffs.{tariff}.energy_charges", "tariffs.{tariff}.energy_tiers", "tariffs.{tariff}.riders",
                   "upgrades.{site}.items", "global.clock_start", *_LOAD],
        "demand": ["tariffs.{tariff}.demand_charges", "tariffs.{tariff}.power_factor_min", *_LOAD],
        "riders_fees": ["tariffs.{tariff}.riders", "tariffs.{tariff}.customer_charge_usd_mo", *_LOAD],
        "minimum_bill": ["tariffs.{tariff}.min_bill_pct", "tariffs.{tariff}.min_bill_trailing_months",
                         "tariffs.{tariff}.demand_charges", "tariffs.{tariff}.riders"],
        "upgrade_capex": ["upgrades.{site}.items", "global.clock_start"],
        "upgrade_refund": ["upgrades.{site}.items", "global.clock_start", "global.horizon_years"],
        "delay": ["upgrades.{site}.items", *_UNSERVED],
        "unserved_energy": _UNSERVED,
    },
    "btm": {
        "fuel": ["gas.{hub}.henry_hub_curve_ref", "gas.{hub}.basis_curve_ref", "gas.{hub}.henry_hub_multiplier",
                 "gas.{hub}.basis_adder_usd_mmbtu", "gas.{hub}.transport_commodity_usd_mmbtu",
                 "gas.{hub}.fuel_retention_pct", *_HEAT_RATE, *_PLANT_SIZE, *_AVAILABILITY, *_DERATES, *_LOAD],
        "transport_fixed": ["gas.{hub}.transport_firm_usd_mmbtu", *_HEAT_RATE, *_PLANT_SIZE, *_LOAD,
                            *_BTM_TIMING],
        "lateral": ["gas.{hub}.lateral_capex_usd", "gas.{hub}.lateral_lead_months", "global.discount_rate",
                    "global.clock_start", *_BTM_TIMING],
        "plant_capex": ["plants.{plant}.capex_usd_kw", *_PLANT_SIZE, "global.discount_rate", *_BTM_TIMING],
        "emissions_controls": ["plants.{plant}.emissions_control_capex_usd_kw", *_PLANT_SIZE,
                               "global.discount_rate", *_BTM_TIMING],
        "fixed_om": ["plants.{plant}.fixed_om_usd_kw_yr", *_PLANT_SIZE, *_BTM_TIMING],
        "variable_om": ["plants.{plant}.variable_om_usd_mwh", "plants.{plant}.major_maintenance_usd_mwh",
                        *_AVAILABILITY, *_PLANT_SIZE, *_LOAD],
        "carbon": ["plants.{plant}.co2_price_usd_t", "plants.{plant}.emission_factor_t_co2_mmbtu", *_HEAT_RATE],
        "residual": ["global.btm_salvage_pct_capex", "plants.{plant}.capex_usd_kw"],
        "delay": [*_BTM_TIMING, *_UNSERVED],
        "unserved_energy": [*_AVAILABILITY, *_PLANT_SIZE, *_DERATES, *_UNSERVED],
    },
}


def _merge_traces() -> dict[str, list[str]]:
    """The hybrid's cost lines draw on both paths' inputs, plus its own settings."""
    merged: dict[str, list[str]] = {}
    for path in ("grid", "btm"):
        for category, templates in COST_LINE_INPUTS[path].items():
            merged.setdefault(category, [])
            merged[category] += [t for t in templates if t not in merged[category]]
    return {c: [*t, "sites.{site}.hybrid"] for c, t in merged.items()}


COST_LINE_INPUTS["hybrid"] = _merge_traces()


def site_names(site: Site) -> dict[str, str]:
    names = {"site": site.site_id}
    if site.grid_paths:
        names["tariff"] = site.grid_paths[0]
    if site.gas_hub:
        names["hub"] = site.gas_hub
    if site.btm_configs:
        names["plant"] = site.btm_configs[0]
    return names


class _Missing(dict):
    def __missing__(self, key: str) -> str:
        raise KeyError(key)


def expand_for_sites(inputs: Inputs, template: str) -> list[tuple[str | None, str]]:
    """(site, concrete path) for each site the template applies to; a template with no
    placeholders gives one row with no site."""
    if "{" not in template:
        return [(None, template)]
    out = []
    for site in inputs.sites.values():
        try:
            out.append((site.site_id, template.format_map(_Missing(site_names(site)))))
        except KeyError:
            continue  # this site has no such input
    return out


# ---- audit records ----
@dataclass(frozen=True)
class AuditObject:
    kind: str
    id: str
    prefix: str  # override-style path to this object, e.g. tariffs.AE-TRANS
    source: str | None
    as_of: date | None
    flags: tuple[str, ...]
    rows: tuple[tuple[str, str], ...]  # (path relative to prefix, value)


@dataclass(frozen=True)
class RegisterRow:
    id: str
    name: str
    site: str | None
    path: str | None
    value: str
    source: str | None
    as_of: date | None
    owner: str
    confidence: str | None
    flags: tuple[str, ...]


@dataclass(frozen=True)
class DataFileRow:
    path: str
    sha256: str | None
    source: str | None
    as_of: date | None
    flags: tuple[str, ...]


@dataclass(frozen=True)
class OverrideRow:
    scenario: str
    path: str
    base: str
    override: str
    description: str


@dataclass(frozen=True)
class TraceRow:
    scenario: str
    site_id: str
    path_id: str
    category: str
    usd_per_mwh: float
    inputs: tuple[str, ...]


@dataclass(frozen=True)
class Audit:
    today: date
    stale_days: int
    version: str
    commit: str
    scenarios: tuple[str, ...]
    config_hashes: tuple[tuple[str, str], ...]
    objects: tuple[AuditObject, ...]
    register: tuple[RegisterRow, ...]
    data_files: tuple[DataFileRow, ...]
    overrides: tuple[OverrideRow, ...]
    trace: tuple[TraceRow, ...]
    unused: tuple[tuple[str, str], ...] = field(default=())  # (kind, id) loaded but not used

    def flagged(self) -> list[tuple[str, tuple[str, ...]]]:
        """Every flagged item as (where, flags), including soft flags such as a TBD owner."""
        out = [(f"{o.kind} {o.id}", o.flags) for o in self.objects if o.flags]
        out += [(f"{r.id} {r.name}" + (f" ({r.site})" if r.site else ""), r.flags) for r in self.register if r.flags]
        out += [(f"data file {d.path}", d.flags) for d in self.data_files if d.flags]
        return out

    def counts(self) -> dict[str, int]:
        """How many items carry each flag, plus the total checked."""
        items = [o.flags for o in self.objects] + [r.flags for r in self.register] + [d.flags for d in self.data_files]
        counts = {"items": len(items)}
        for flag in ("placeholder", "stale", "no source", "no as_of", "unresolved", "missing file",
                     "owner TBD", "low confidence"):
            counts[flag] = sum(flag in f for f in items)
        return counts

    def strict_problems(self) -> list[tuple[str, tuple[str, ...]]]:
        """Items a run should not be presented with: placeholders, stale, untagged, broken."""
        return [(w, tuple(f for f in flags if f in STRICT_FLAGS)) for w, flags in self.flagged()
                if STRICT_FLAGS & set(flags)]


# ---- helpers ----
def fmt(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(fmt(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}: {fmt(v)}" for k, v in value.items()) + "}"
    if isinstance(value, date):
        return value.isoformat()
    return repr(value) if isinstance(value, float) else str(value)


def flatten(node: Any, prefix: str = "", skip: frozenset[str] = frozenset({"source", "as_of"})) -> list[tuple[str, str]]:
    """Leaf values as (dotted path, text). Lists of named dicts use the override selector
    ``name=...`` so a path can be pasted into a scenario."""
    rows: list[tuple[str, str]] = []
    if isinstance(node, dict):
        for k, v in node.items():
            if not prefix and k in skip:
                continue
            rows += flatten(v, f"{prefix}.{k}" if prefix else str(k), skip)
    elif isinstance(node, list) and node and all(isinstance(x, dict) for x in node):
        names = [x.get("name") for x in node]
        by_name = all(isinstance(n, str) and n and "." not in n for n in names) and len(set(names)) == len(names)
        for i, item in enumerate(node):
            key = f"name={item['name']}" if by_name else str(i)
            body = {k: v for k, v in item.items() if not (by_name and k == "name")}
            rows += flatten(body, f"{prefix}.{key}", skip)
    else:
        rows.append((prefix, fmt(node)))
    return rows


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit(cwd: Path | None = None) -> str:
    try:
        def run(*args: str) -> str:
            return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=5, check=True).stdout.strip()

        return run("rev-parse", "--short", "HEAD") + (" (uncommitted changes)" if run("status", "--porcelain") else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _object(inputs: Inputs, kind: str, id_: str, prefix: str, model: Any, today: date) -> AuditObject:
    stale_days = inputs.global_cfg.stale_days
    return AuditObject(
        kind=kind, id=id_, prefix=prefix, source=model.source, as_of=model.as_of,
        flags=source_flags(model.source, model.as_of, today, stale_days),
        rows=tuple(flatten(model.model_dump())),
    )


def _used_objects(inputs: Inputs, today: date) -> tuple[list[AuditObject], list[tuple[str, str]]]:
    objs = [_object(inputs, "global settings", "global", "global", inputs.global_cfg, today)]
    used = {"tariffs": set(), "plants": set(), "gas": set(), "upgrades": set()}
    for site in inputs.sites.values():
        objs.append(_object(inputs, "site", site.site_id, f"sites.{site.site_id}", site, today))
    for site in inputs.sites.values():
        if site.grid_paths and site.grid_paths[0] not in used["tariffs"]:
            used["tariffs"].add(site.grid_paths[0])
            objs.append(_object(inputs, "tariff", site.grid_paths[0], f"tariffs.{site.grid_paths[0]}",
                                inputs.tariffs[site.grid_paths[0]], today))
        if site.site_id in inputs.upgrades and site.grid_paths:
            used["upgrades"].add(site.site_id)
            objs.append(_object(inputs, "network upgrades", site.site_id, f"upgrades.{site.site_id}",
                                inputs.upgrades[site.site_id], today))
        if site.btm_configs and site.btm_configs[0] not in used["plants"]:
            used["plants"].add(site.btm_configs[0])
            objs.append(_object(inputs, "plant", site.btm_configs[0], f"plants.{site.btm_configs[0]}",
                                inputs.plants[site.btm_configs[0]], today))
        if site.btm_configs and site.gas_hub and site.gas_hub not in used["gas"]:
            used["gas"].add(site.gas_hub)
            objs.append(_object(inputs, "gas supply", site.gas_hub, f"gas.{site.gas_hub}", inputs.gas[site.gas_hub], today))
    unused = [(kind, id_) for kind, store, key in
              (("tariff", inputs.tariffs, "tariffs"), ("plant", inputs.plants, "plants"),
               ("gas supply", inputs.gas, "gas"), ("network upgrades", inputs.upgrades, "upgrades"))
              for id_ in store if id_ not in used[key]]
    return objs, unused


def _register_rows(inputs: Inputs, tree: dict[str, Any], today: date) -> list[RegisterRow]:
    stale_days = inputs.global_cfg.stale_days
    rows = []
    for a in inputs.assumptions:
        base_flags = list(source_flags(a.source, a.as_of, today, stale_days))
        if a.owner.strip().upper() in ("", "TBD"):
            base_flags.append("owner TBD")
        if a.confidence == "low":
            base_flags.append("low confidence")
        if a.path is None:
            rows.append(RegisterRow(a.id, a.name, None, None, a.description, a.source, a.as_of, a.owner,
                                    a.confidence, tuple(base_flags)))
            continue
        targets = expand_for_sites(inputs, a.path)
        if not targets:
            rows.append(RegisterRow(a.id, a.name, None, a.path, "(no site uses this input)", a.source, a.as_of,
                                    a.owner, a.confidence, tuple(base_flags)))
        for site_id, path in targets:
            try:
                value, flags = fmt(get_path(tree, path)), list(base_flags)
            except KeyError:
                value, flags = "(path not found)", [*base_flags, "unresolved"]
            rows.append(RegisterRow(a.id, a.name, site_id, path, value, a.source, a.as_of, a.owner,
                                    a.confidence, tuple(flags)))
    return rows


def _data_files(inputs: Inputs, effective: list[Inputs], today: date) -> list[DataFileRow]:
    refs: set[str] = set()
    for eff in effective:
        for site in eff.sites.values():
            if site.load_profile_ref:
                refs.add(site.load_profile_ref)
            if site.btm_configs:
                refs.add(eff.plants[site.btm_configs[0]].heat_rate_part_load_curve_ref)
                if site.gas_hub:
                    gas = eff.gas[site.gas_hub]
                    refs |= {gas.henry_hub_curve_ref, gas.basis_curve_ref}
    rows = []
    for ref in sorted(refs):
        meta = inputs.data_sources.get(ref)
        source, as_of = (meta.source, meta.as_of) if meta else (None, None)
        flags = list(source_flags(source, as_of, today, inputs.global_cfg.stale_days))
        try:
            target = resolve_ref(inputs.data_dir, ref)
            digest = sha256_file(target) if target.exists() else None
        except ValueError:
            digest = None
        if digest is None:
            flags.append("missing file")
        rows.append(DataFileRow(ref, digest, source, as_of, tuple(flags)))
    return rows


def build_audit(
    inputs: Inputs,
    scenario_ids: list[str],
    *,
    today: date | None = None,
    config_dir: Path | None = None,
    dist: Distributions | None = None,
    result: RunResult | None = None,
) -> Audit:
    from dcop.reports import levelized_categories  # local import: reports imports scenarios

    today = today or date.today()
    tree = inputs_tree(inputs)
    effective = [apply_overrides(inputs, inputs.scenarios[s].overrides) for s in scenario_ids]

    objects, unused = _used_objects(inputs, today)
    if dist is not None:
        objects.append(AuditObject(
            "monte carlo settings", "distributions", "distributions", dist.source, dist.as_of,
            source_flags(dist.source, dist.as_of, today, inputs.global_cfg.stale_days),
            tuple(flatten(dist.model_dump()))))

    overrides = []
    for sid in scenario_ids:
        scen = inputs.scenarios[sid]
        for path, value in scen.overrides.items():
            try:
                base = fmt(get_path(tree, path))
            except KeyError:
                base = "(path not found)"
            overrides.append(OverrideRow(sid, path, base, fmt(value), scen.description))

    trace: list[TraceRow] = []
    if result is not None and scenario_ids:
        first = scenario_ids[0]
        cats = levelized_categories(result)
        cats = cats[cats["scenario_id"] == first]
        for row in cats.itertuples(index=False):
            site = inputs.sites[row.site_id]
            templates = COST_LINE_INPUTS.get(row.path_id, {}).get(row.category, [])
            paths = tuple(dict.fromkeys(p for t in templates for s_, p in expand_for_sites(inputs, t)
                                        if s_ in (None, site.site_id)))
            trace.append(TraceRow(first, row.site_id, row.path_id, row.category, float(row.usd_per_mwh), paths))

    hashes = ()
    if config_dir is not None:
        hashes = tuple((f.name, sha256_file(f)[:12]) for f in sorted(Path(config_dir).glob("*.yaml")))
    return Audit(
        today=today, stale_days=inputs.global_cfg.stale_days, version=dcop.__version__,
        commit=git_commit(Path(config_dir).parent if config_dir else None),
        scenarios=tuple(scenario_ids), config_hashes=hashes, objects=tuple(objects),
        register=tuple(_register_rows(inputs, tree, today)), data_files=tuple(_data_files(inputs, effective, today)),
        overrides=tuple(overrides), trace=tuple(trace), unused=tuple(unused),
    )


# ---- markdown ----
def _cell(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _table(header: list[str], rows: list[list[Any]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return out + [""]


def _flags(flags: tuple[str, ...]) -> str:
    return ", ".join(flags) if flags else "ok"


def render_markdown(audit: Audit) -> str:
    c = audit.counts()
    lines = [
        "# Assumptions and sources", "",
        f"Run date {audit.today.isoformat()}. Model version {audit.version}, commit {audit.commit}. "
        f"Scenarios: {', '.join(audit.scenarios)}. Inputs older than {audit.stale_days} days are flagged stale.", "",
    ]
    if audit.config_hashes:
        lines += ["Config files (first 12 characters of the SHA-256):", ""]
        lines += _table(["File", "SHA-256"], [[f"`{n}`", h] for n, h in audit.config_hashes])

    lines += ["## Audit summary", "",
              f"{c['items']} items checked: {c['placeholder']} placeholder, {c['stale']} stale, "
              f"{c['no source'] + c['no as_of']} missing a source or date, {c['unresolved']} register entries "
              f"whose path was not found, {c['missing file']} missing data files. "
              f"{c['owner TBD']} register entries have no owner and {c['low confidence']} are low confidence.", ""]
    problems = audit.strict_problems()
    if problems:
        lines += _table(["Item", "Flags"], [[w, ", ".join(f)] for w, f in problems])
    else:
        lines += ["No placeholders, stale, untagged or broken inputs.", ""]

    lines += ["## Assumptions register", ""]
    lines += _table(["ID", "Assumption", "Site", "Value", "Source", "As of", "Owner", "Confidence", "Flags"],
                    [[r.id, r.name, r.site or "", r.value, r.source or "", r.as_of or "", r.owner,
                      r.confidence or "", _flags(r.flags)] for r in audit.register])

    lines += ["## Inputs used", "",
              "Paths can be pasted into a scenario's overrides. Source and date apply to the whole object.", ""]
    for o in audit.objects:
        lines += [f"### {o.kind.capitalize()}: {o.id}", "",
                  f"Path `{o.prefix}`. Source: {o.source or '(none)'}. As of: {o.as_of or '(none)'}. "
                  f"Flags: {_flags(o.flags)}.", ""]
        lines += _table(["Input", "Value"], [[f"`{p}`", v] for p, v in o.rows])
    if audit.unused:
        lines += ["Loaded but not used by this run: " + ", ".join(f"{k} {i}" for k, i in audit.unused) + ".", ""]

    lines += ["## Data files", ""]
    lines += _table(["File", "SHA-256", "Source", "As of", "Flags"],
                    [[f"`{d.path}`", (d.sha256 or "")[:12], d.source or "", d.as_of or "", _flags(d.flags)]
                     for d in audit.data_files])

    lines += ["## Scenario overrides", ""]
    if audit.overrides:
        lines += _table(["Scenario", "Path", "Base value", "Override", "Why"],
                        [[o.scenario, f"`{o.path}`", o.base, o.override, o.description] for o in audit.overrides])
    else:
        lines += ["None of the scenarios in this run override an input.", ""]

    if audit.trace:
        lines += [f"## Where each cost line comes from (scenario {audit.trace[0].scenario})", "",
                  "Levelized $/MWh by ledger category. The paths are the inputs that drive the line; "
                  "find their values and sources above. All lines also depend on `global.discount_rate` "
                  "and `global.base_year`.", ""]
        lines += _table(["Site", "Path", "Category", "$/MWh", "Inputs"],
                        [[t.site_id, t.path_id, t.category, f"{t.usd_per_mwh:,.2f}",
                          ", ".join(f"`{p}`" for p in t.inputs)] for t in audit.trace])
    return "\n".join(lines)
