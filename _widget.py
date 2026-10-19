import colorsys
from itertools import combinations

import numpy as np
import polars as pl
from napari import Viewer
from napari.layers import Image, Labels, Points
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)
from skimage.draw import disk
import pyclesperanto as cle
from skimage.measure import regionprops_table, regionprops
from skimage.filters import gaussian, threshold_otsu
from scipy import ndimage as ndi

ZYX = list("zyx")


def create_table(
    labels: Labels, nuc_gfp: Image, tf_stains: list[Image]
) -> tuple[pl.DataFrame, dict[str, float]]:
    scale = labels.scale
    half_widths_voxels = [round(0.5 / spacing) for spacing in scale]
    box_footprint = np.ones(
        [2 * half_width + 1 for half_width in half_widths_voxels], dtype=bool
    )
    eroaded_labels = np.zeros_like(labels.data)
    for region in regionprops(labels.data):
        eroded_mask = ndi.binary_erosion(region.image, structure=box_footprint)
        eroaded_labels[region.slice][eroded_mask] = region.label
    shells = (eroaded_labels == 0) * labels.data
    blur_sigma = 0.2 / scale
    blur_nuc = cle.pull(
        cle.gaussian_blur(
            nuc_gfp.data,
            sigma_x=blur_sigma[0],
            sigma_y=blur_sigma[1],
            sigma_z=blur_sigma[2],
        )
    )
    core_df = pl.DataFrame(
        regionprops_table(
            eroaded_labels,
            intensity_image=nuc_gfp.data,
            properties=(
                "label",
                "area",
                "solidity",
                "intensity_mean",
                "centroid",
            ),
            spacing=scale,
        )
    )
    shell_df = pl.DataFrame(
        regionprops_table(
            shells,
            intensity_image=blur_nuc,
            properties=(
                "label",
                "intensity_max",
            ),
        )
    )
    df = core_df.join(shell_df, on="label")
    defaults: dict[str, float] = {}
    for stain in tf_stains:
        stain_df = pl.DataFrame(
            regionprops_table(
                eroaded_labels,
                intensity_image=cle.pull(
                    cle.gaussian_blur(
                        stain.data,
                        sigma_x=blur_sigma[0] * 2,
                        sigma_y=blur_sigma[1] * 2,
                        sigma_z=blur_sigma[2] * 2,
                    )
                ),
                properties=("label", "intensity_mean"),
            )
        ).rename({"intensity_mean": stain.name})
        defaults[stain.name] = threshold_otsu(np.asarray(stain_df[stain.name]))
        df = df.join(
            stain_df,
            on="label",
        )
    return (
        df.filter(df["solidity"].is_finite())
        .rename({"centroid-0": "z", "centroid-1": "y", "centroid-2": "x"})
        .with_columns(
            (pl.col("intensity_mean") / pl.col("intensity_max")).alias(
                "intensity_ratio"
            ),
        )
    ), defaults


class CellFilterWidget(QWidget):
    def __init__(
        self,
        viewer: Viewer,
        labels: Labels,
        nuc_gfp: Image,
        tf_stains: list[Image],
        radius: float = 5.0,
    ) -> None:
        super().__init__()
        df, stain_defaults = create_table(labels, nuc_gfp, tf_stains)
        self.viewer, self.radius = viewer, radius
        self.tf_stains = dict(zip(stain_defaults, tf_stains))
        self.criteria = [
            c for c in df.columns if c not in stain_defaults and c not in ZYX + ["label"]
        ]
        self.df = df.with_columns(
            pl.struct(ZYX).alias("pos"),
            pl.lit(False).alias("manual"),
            pl.lit(None).cast(str).alias("type"),
        )
        self.local_df = self.df

        form = QFormLayout(self)
        self.lo: dict[str, QDoubleSpinBox] = {}
        self.hi: dict[str, QDoubleSpinBox] = {}
        self.thr: dict[str, QDoubleSpinBox] = {}
        # set up spin boxes
        for c in self.criteria:
            vmin, vmax = df[c].min(), df[c].max()
            self.lo[c] = self.create_spin_box(vmin, vmin, vmax)
            self.hi[c] = self.create_spin_box(vmax, vmin, vmax)
            row = QHBoxLayout()
            row.addWidget(self.lo[c])
            row.addWidget(self.hi[c])
            form.addRow(f"{c} (min / max)", row)
        for s, default in stain_defaults.items():
            self.thr[s] = self.create_spin_box(default, df[s].min(), df[s].max())
            form.addRow(f"{s} (min)", self.thr[s])
        # stains
        self.names = ["none"] + [
            "+".join(k)
            for r in range(1, len(self.tf_stains) + 1)
            for k in combinations(self.tf_stains, r)
        ]
        self.cycle = {
            n: (*colorsys.hsv_to_rgb(i / len(self.names), 0.8, 1.0), 1.0)
            for i, n in enumerate(self.names)
        }
        self.table = QTableWidget(len(self.names), 3)
        self.table.setHorizontalHeaderLabels(["color", "type", "count"])
        self.table.verticalHeader().hide()
        for i, n in enumerate(self.names):
            r, g, b, _ = self.cycle[n]
            swatch = QTableWidgetItem()
            swatch.setBackground(QColor.fromRgbF(r, g, b))
            self.table.setItem(i, 0, swatch)
            self.table.setItem(i, 1, QTableWidgetItem(n))
            self.table.setItem(i, 2, QTableWidgetItem("0"))
        self.table.cellClicked.connect(self.on_cell_clicked)
        form.addRow(self.table)

        btn = QPushButton("Export CSV")
        btn.clicked.connect(self.export_csv)
        form.addRow(btn)
        viewer.add_points(
            self.df.select(ZYX),
            features=self.df.select(list(self.criteria)).to_pandas(),
            name="pre_filter",
            size=2,
        )
        self.layer: Points = viewer.add_points(
            np.empty((0, 3)), name="cells", size=2, ndim=3
        )
        self.layer.events.data.connect(self.on_data)
        self.refresh()

    def create_spin_box(self, value: float, lo: float, hi: float) -> QDoubleSpinBox:
        value, lo, hi = float(value), float(lo), float(hi)
        sb = QDoubleSpinBox()
        sb.setDecimals(4)
        sb.setRange(lo, hi)
        sb.setSingleStep((hi - lo) / 100 or 1.0)
        sb.setValue(value)
        sb.valueChanged.connect(self.refresh)
        return sb

    def measure(self, coords: np.ndarray) -> dict[str, float]:
        """Mean intensity of each tf_stain within a circle of self.radius in plane z."""
        out = {}
        img = next(iter(self.tf_stains.values()))
        coords_pix = coords / img.scale
        z, y, x = coords_pix
        rr, cc = disk((y, x), self.radius, shape=img.data[0].shape)
        for s, img in self.tf_stains.items():
            out[s] = float(img.data[round(z), rr, cc].mean())
        return out

    def refresh(self, *_) -> None:
        auto_type = (
            pl.when("manual")
            .then("type")
            .otherwise(
                pl.concat_str(
                    [
                        pl.when(pl.col(s) >= self.thr[s].value()).then(pl.lit(s))
                        for s in self.tf_stains
                    ],
                    separator="+",
                    ignore_nulls=True,
                ).replace("", "none")
            )
        )
        keep = pl.col("manual") | pl.all_horizontal(
            [
                pl.col(c).is_between(self.lo[c].value(), self.hi[c].value())
                for c in self.criteria
            ]
        )
        self.local_df = self.df.filter(keep).with_columns(auto_type.alias("type"))
        with self.layer.events.data.blocker():
            self.layer.data = self.local_df.select(ZYX).to_numpy()
            self.layer.features = self.local_df.select(
                self.criteria + list(self.tf_stains) + ["type"]
            ).to_pandas()
            self.layer.face_color_cycle = self.cycle
            self.layer.face_color = "type"

        counts = dict(self.local_df["type"].value_counts().iter_rows())
        for i, n in enumerate(self.names):
            self.table.item(i, 2).setText(str(counts.get(n, 0)))

    def on_data(self, event) -> None:
        """update self.df to reflect data"""
        if event.action == "added":
            new = self.layer.data[np.array(event.data_indices)]
            if new.shape[0] == 1:
                new = new[0]
            mean = self.measure(new)
            to_add = {c: None for c in self.df.columns}
            to_add["z"], to_add["y"], to_add["x"] = new
            to_add["type"] = (
                "+".join(s for s, v in mean.items() if v >= self.thr[s].value())
                or "none"
            )
            to_add["manual"] = True
            ta_df = pl.DataFrame([to_add]).with_columns(pl.struct(ZYX).alias("pos"))
            self.df = pl.concat([self.df, ta_df])
            self.refresh()
        if event.action == "removing":
            to_delete = (
                pl.DataFrame(self.layer.data[np.array(event.data_indices)], schema=ZYX)
                .select(pl.struct(ZYX).alias("pos"))["pos"]
                .to_list()
            )
            self.df = self.df.filter(~pl.col("pos").is_in(to_delete))

        if event.action == "removed":
            self.refresh()

    def on_cell_clicked(self, row: int, col: int) -> None:
        """Clicking a swatch assigns that type to the selected points."""
        if len(self.layer.selected_data) == 0:
            return
        to_change = (
            pl.DataFrame(
                self.layer.data[np.array(tuple(self.layer.selected_data))], schema=ZYX
            )
            .select(pl.struct(ZYX).alias("pos"))["pos"]
            .to_list()
        )
        self.df = self.df.with_columns(
            pl.when(pl.col("pos").is_in(to_change))
            .then(pl.lit(True))
            .otherwise("manual")
            .alias("manual"),
            pl.when(pl.col("pos").is_in(to_change))
            .then(pl.lit(self.names[row]))
            .otherwise("type")
            .alias("type"),
        )
        self.layer.selected_data = set()
        self.refresh()

    def export_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", "cells.csv", "CSV (*.csv)"
        )
        if path:
            self.local_df.drop("_id", "type").write_csv(path)
