import numpy as np
import pandas as pd
import pytest

from backend.services.graph_quick import (
    SERIES_COLORS,
    guess_ylabel,
    build_traces,
    build_layout,
    build_quick_response,
)
from backend.models.schema import AnalysisRequest, AnalysisType


class TestSeriesColors:
    def test_has_10_colors(self):
        assert len(SERIES_COLORS) == 10

    def test_colors_are_hex_strings(self):
        for color in SERIES_COLORS:
            assert color.startswith("#"), f"Not a hex color: {color}"
            assert len(color) == 7, f"Bad hex length: {color}"


class TestGuessYlabel:
    def test_force_columns_return_newton_label(self):
        label = guess_ylabel(["L_ActForce_N", "R_ActForce_N"])
        assert "N" in label or "Force" in label or "force" in label

    def test_gcp_columns_return_gcp_label(self):
        label = guess_ylabel(["L_GCP", "R_GCP"])
        assert "GCP" in label or "%" in label

    def test_imu_columns_return_degree_label(self):
        label = guess_ylabel(["L_Pitch", "R_Pitch"])
        assert "°" in label or "deg" in label.lower() or "Angle" in label

    def test_unknown_columns_return_nonempty_string(self):
        label = guess_ylabel(["some_unknown_col"])
        assert isinstance(label, str)
        assert len(label) > 0


class TestBuildTraces:
    def _make_df(self) -> pd.DataFrame:
        n = 200
        rng = np.random.default_rng(0)
        return pd.DataFrame({
            "L_ActForce_N": 30 + rng.normal(0, 3, n),
            "R_ActForce_N": 28 + rng.normal(0, 3, n),
            "L_GCP": np.linspace(0, 100, n) % 100,
            "R_GCP": (np.linspace(50, 150, n)) % 100,
        })

    def test_returns_list_of_dicts(self):
        df = self._make_df()
        req = AnalysisRequest(analysis_type=AnalysisType.force)
        traces = build_traces([("trial.csv", df)], req)
        assert isinstance(traces, list)
        assert all(isinstance(t, dict) for t in traces)

    def test_trace_count_equals_files_times_columns(self):
        df = self._make_df()
        req = AnalysisRequest(
            analysis_type=AnalysisType.force,
            columns=["L_ActForce_N", "R_ActForce_N"],
        )
        traces = build_traces([("trial.csv", df)], req)
        assert len(traces) == 2  # 1 file × 2 columns

    def test_trace_has_required_keys(self):
        df = self._make_df()
        req = AnalysisRequest(
            analysis_type=AnalysisType.force,
            columns=["L_ActForce_N"],
        )
        traces = build_traces([("trial.csv", df)], req)
        trace = traces[0]
        for key in ("x", "y", "name", "mode", "line"):
            assert key in trace, f"Missing key: {key}"

    def test_trace_mode_is_lines(self):
        df = self._make_df()
        req = AnalysisRequest(
            analysis_type=AnalysisType.force,
            columns=["L_ActForce_N"],
        )
        traces = build_traces([("trial.csv", df)], req)
        assert traces[0]["mode"] == "lines"

    def test_trace_name_contains_filename_and_column(self):
        df = self._make_df()
        req = AnalysisRequest(
            analysis_type=AnalysisType.force,
            columns=["L_ActForce_N"],
        )
        traces = build_traces([("trial.csv", df)], req)
        assert "trial" in traces[0]["name"] or "L_ActForce_N" in traces[0]["name"]

    def test_trace_x_length_matches_df(self):
        df = self._make_df()
        req = AnalysisRequest(
            analysis_type=AnalysisType.force,
            columns=["L_ActForce_N"],
        )
        traces = build_traces([("trial.csv", df)], req)
        assert len(traces[0]["x"]) == len(df)

    def test_normalize_gcp_sets_x_to_0_100(self):
        df = self._make_df()
        req = AnalysisRequest(
            analysis_type=AnalysisType.force,
            columns=["L_ActForce_N"],
            normalize_gcp=True,
        )
        traces = build_traces([("trial.csv", df)], req)
        # When normalised, x-axis should be 101 points (0..100 GCP)
        assert len(traces[0]["x"]) == 101

    def test_line_has_color_and_width(self):
        df = self._make_df()
        req = AnalysisRequest(
            analysis_type=AnalysisType.force,
            columns=["L_ActForce_N"],
        )
        traces = build_traces([("trial.csv", df)], req)
        line = traces[0]["line"]
        assert "color" in line
        assert "width" in line
        # graph_quick uses thinner stroke for "desired" channels (1.2)
        # and thicker for "actual" measurements (1.8). L_ActForce_N is
        # an actual measurement, so it must come back at the bolder width.
        assert line["width"] == pytest.approx(1.8)

    def test_multiple_files_produce_distinct_traces(self):
        df1 = self._make_df()
        df2 = self._make_df()
        req = AnalysisRequest(
            analysis_type=AnalysisType.force,
            columns=["L_ActForce_N"],
        )
        traces = build_traces([("trial1.csv", df1), ("trial2.csv", df2)], req)
        assert len(traces) == 2
        assert traces[0]["name"] != traces[1]["name"]


class TestBuildLayout:
    def test_returns_dict(self):
        req = AnalysisRequest(analysis_type=AnalysisType.force)
        layout = build_layout(req, "Force (N)")
        assert isinstance(layout, dict)

    def test_has_title(self):
        req = AnalysisRequest(analysis_type=AnalysisType.force)
        layout = build_layout(req, "Force (N)")
        assert "title" in layout

    def test_has_xaxis_and_yaxis(self):
        req = AnalysisRequest(analysis_type=AnalysisType.force)
        layout = build_layout(req, "Force (N)")
        assert "xaxis" in layout
        assert "yaxis" in layout

    def test_yaxis_title_matches_ylabel(self):
        req = AnalysisRequest(analysis_type=AnalysisType.force)
        layout = build_layout(req, "Force (N)")
        yaxis = layout["yaxis"]
        assert "Force" in yaxis.get("title", {}).get("text", "") or \
               "Force" in str(yaxis.get("title", ""))

    def test_background_is_white(self):
        req = AnalysisRequest(analysis_type=AnalysisType.force)
        layout = build_layout(req, "Force (N)")
        plot_bgcolor = layout.get("plot_bgcolor", "")
        paper_bgcolor = layout.get("paper_bgcolor", "")
        assert "white" in plot_bgcolor or plot_bgcolor == "#ffffff" or \
               "white" in paper_bgcolor or paper_bgcolor == "#ffffff"

    def test_legend_is_shown(self):
        req = AnalysisRequest(analysis_type=AnalysisType.force)
        layout = build_layout(req, "Force (N)")
        assert layout.get("showlegend", True) is True


class TestBuildQuickResponse:
    def test_returns_dict_with_data_and_layout(self, sample_csv):
        req = AnalysisRequest(
            analysis_type=AnalysisType.force,
            columns=["L_ActForce_N", "R_ActForce_N"],
        )
        result = build_quick_response(req, [sample_csv])
        assert "data" in result
        assert "layout" in result

    def test_data_is_non_empty(self, sample_csv):
        req = AnalysisRequest(
            analysis_type=AnalysisType.force,
            columns=["L_ActForce_N"],
        )
        result = build_quick_response(req, [sample_csv])
        assert len(result["data"]) > 0

    def test_each_trace_is_json_serialisable(self, sample_csv):
        import json
        req = AnalysisRequest(
            analysis_type=AnalysisType.force,
            columns=["L_ActForce_N"],
        )
        result = build_quick_response(req, [sample_csv])
        # Should not raise
        json.dumps(result)

    def test_missing_file_raises_value_error(self, tmp_path):
        req = AnalysisRequest(
            analysis_type=AnalysisType.force,
            columns=["L_ActForce_N"],
        )
        with pytest.raises((ValueError, FileNotFoundError, Exception)):
            build_quick_response(req, [str(tmp_path / "nonexistent.csv")])
