import os
import json

import pyarrow as pa
import pyarrow.parquet as pq


ENTRIES_CHUNK_SIZE = 10_000


def _flatten_fault(fault):
    if isinstance(fault, list):
        inner = fault[0] if fault else None
        if isinstance(inner, list):
            return inner[0] if inner else "other"
        return inner if inner else "other"
    return str(fault) if fault else "other"


class LogStore:
    """
    Stores the structured Drain3 output so log lines can be revisited without
    re-parsing raw files.

    Two Parquet files are written under output_dir/store/:
      templates.parquet  — one row per unique log pattern with golden signal,
                           fault category, and occurrence count.
      log_entries.parquet — one row per log line: template_id, original_text,
                            variable tokens, timestamp, file source.

    JSON equivalents (entries include original_text + golden_signal) are written
    to developer_debug_files/ for explorer.html to load without a server.
    Entries in both Parquet and JSON are sorted by timestamp DESC so the
    explorer's default view (most recent first) needs no client-side sort.
    """

    TEMPLATES_FILE = "templates.parquet"
    ENTRIES_FILE = "log_entries.parquet"
    TEMPLATES_JSON = "store_templates.json"
    ENTRIES_JSON_PREFIX = "store_entries"

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.store_dir = os.path.join(output_dir, "store")
        self.debug_dir = os.path.join(output_dir, "developer_debug_files")
        os.makedirs(self.store_dir, exist_ok=True)

        self._templates: dict[int, dict] = {}
        self._entries: list[dict] = []

    # ------------------------------------------------------------------
    # Variable extraction
    # ------------------------------------------------------------------

    @staticmethod
    def extract_variables(log_line: str, template_str: str) -> list[str]:
        """
        Return the variable tokens from log_line given its Drain3 template.

        Drain3 tokenises by whitespace and replaces variable positions with <*>.
        Static tokens are skipped; wildcard positions are collected.  The last
        wildcard absorbs any trailing extra tokens.
        """
        log_tokens = log_line.split()
        tmpl_tokens = template_str.split()
        variables: list[str] = []

        li = 0
        for ti, tmpl_tok in enumerate(tmpl_tokens):
            if li >= len(log_tokens):
                break
            if tmpl_tok == "<*>":
                is_final_token = (ti == len(tmpl_tokens) - 1)
                if is_final_token and li < len(log_tokens):
                    variables.append(" ".join(log_tokens[li:]))
                    li = len(log_tokens)
                else:
                    variables.append(log_tokens[li])
                    li += 1
            else:
                li += 1

        return variables

    # ------------------------------------------------------------------
    # Building the store
    # ------------------------------------------------------------------

    def build_from_df(self, df, temp_id_to_signal_map: dict):
        """
        Populate templates and log_entries from the post-classification DataFrame.

        Required columns: test_ids, template_str, variables, epoch, file_names.
        original_text is used when present (added by run_drain.py from df['text']
        before Drain3 masking strips it).

        temp_id_to_signal_map: (template_id, file_name) -> (golden_signal, fault_list)
        """
        required = {"test_ids", "template_str", "variables", "epoch", "file_names"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing columns needed by LogStore: {missing}")

        has_original = "original_text" in df.columns
        has_component = "component" in df.columns

        # --- templates ---
        agg_kwargs = {
            "template": ("template_str", "first"),
            "occurrence_count": ("test_ids", "count"),
        }
        if has_component:
            agg_kwargs["component"] = ("component", "first")
        grp = (
            df.groupby("test_ids")
            .agg(**agg_kwargs)
            .reset_index()
        )
        for _, row in grp.iterrows():
            try:
                tid = int(row["test_ids"])
            except (ValueError, TypeError):
                continue
            tmpl = str(row["template"]) if row["template"] else ""
            self._templates[tid] = {
                "template_id": tid,
                "template": tmpl,
                "var_count": tmpl.count("<*>"),
                "golden_signal": None,
                "fault_category": None,
                "occurrence_count": int(row["occurrence_count"]),
                "component": str(row["component"]) if has_component else None,
            }

        self._update_signals(temp_id_to_signal_map)

        # --- log entries (vectorised) ---
        cols = ["test_ids", "epoch", "variables", "file_names"]
        if has_original:
            cols.append("original_text")

        subset = df[cols].copy().rename(
            columns={
                "test_ids": "template_id",
                "epoch": "timestamp",
                "file_names": "file_source",
            }
        )
        subset["template_id"] = subset["template_id"].apply(
            lambda v: int(v) if str(v).lstrip("-").isdigit() else -1
        )
        subset["timestamp"] = subset["timestamp"].fillna(0).astype("int64")
        subset["variables"] = subset["variables"].apply(
            lambda v: json.dumps([str(x) for x in v]) if isinstance(v, list)
            else (v if isinstance(v, str) and v else "[]")
        )
        subset["file_source"] = subset["file_source"].fillna("").astype(str)
        if has_original:
            subset["original_text"] = subset["original_text"].fillna("").astype(str)
        else:
            subset["original_text"] = ""

        # Capture original position before any reordering.
        # Within a single file this is the source line order.  Across files it
        # reflects the I/O merge order (non-deterministic), so file_source is
        # used as a secondary sort key in queries to keep cross-file ordering
        # deterministic.
        subset = subset.reset_index(drop=True)
        subset["entry_id"] = subset.index

        self._entries = subset.to_dict("records")

        # Annotate templates with a representative log line and per-file counts.
        # Both are derived from entries so we compute them here while subset is still live.
        rep_log_map = (
            subset[subset["original_text"] != ""]
            .groupby("template_id")["original_text"]
            .first()
            .to_dict()
        )
        file_sources_map: dict[int, dict] = {}
        for _, row in (
            subset.groupby(["template_id", "file_source"])
            .size()
            .reset_index(name="cnt")
            .iterrows()
        ):
            tid = int(row["template_id"])
            if tid not in file_sources_map:
                file_sources_map[tid] = {}
            file_sources_map[tid][row["file_source"]] = int(row["cnt"])
        for tid in self._templates:
            self._templates[tid]["rep_log"] = rep_log_map.get(tid, "")
            self._templates[tid]["file_sources"] = json.dumps(file_sources_map.get(tid, {}))

    def _update_signals(self, temp_id_to_signal_map: dict):
        signal_votes: dict[int, dict] = {}
        fault_by_tid: dict[int, str] = {}

        for (tid_raw, _), (gs, fault) in temp_id_to_signal_map.items():
            try:
                tid = int(tid_raw)
            except (ValueError, TypeError):
                continue
            votes = signal_votes.setdefault(tid, {})
            votes[gs] = votes.get(gs, 0) + 1
            if tid not in fault_by_tid:
                fault_by_tid[tid] = _flatten_fault(fault)

        for tid, votes in signal_votes.items():
            if tid in self._templates:
                self._templates[tid]["golden_signal"] = max(votes, key=votes.get)
                self._templates[tid]["fault_category"] = fault_by_tid.get(tid, "other")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_parquet(self):
        """
        Write templates.parquet and log_entries.parquet.

        log_entries.parquet includes original_text, variables, and entry_id.
        entry_id is the row's position in the pre-sort DataFrame — it preserves
        within-file line order and is used as a sort tiebreaker in DuckDB queries.
        """
        if self._templates:
            tbl = pa.Table.from_pylist(list(self._templates.values()))
            pq.write_table(tbl, os.path.join(self.store_dir, self.TEMPLATES_FILE))

        if self._entries:
            tbl = pa.Table.from_pylist(self._entries)
            pq.write_table(tbl, os.path.join(self.store_dir, self.ENTRIES_FILE))

    def save_json_for_explorer(self) -> dict:
        """
        Write the templates JSON consumed by explorer.html's sidebar and return
        metadata the Jinja2 template embeds.

        Log entries are no longer written as JSON chunks — explorer.html queries
        log_entries.parquet directly via DuckDB WASM.
        """
        templates_out = list(self._templates.values())
        with open(os.path.join(self.debug_dir, self.TEMPLATES_JSON), "w") as fh:
            json.dump(templates_out, fh)

        return {
            "total_templates": len(templates_out),
            "total_entries": len(self._entries),
            "templates_file": "../developer_debug_files/" + self.TEMPLATES_JSON,
            "parquet_entries": "../store/" + self.ENTRIES_FILE,
            "parquet_templates": "../store/" + self.TEMPLATES_FILE,
            "timeline_file": "../developer_debug_files/golden_signal_timeline.json",
            "metrics_file": "../metrics/preprocessing.json",
        }
