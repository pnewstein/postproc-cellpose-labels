from pathlib import Path


import napari_scripts as ns

viewer = ns.get_viewer_from_file(
    Path(
        "/Volumes/DoeLab65TB/lab_member_data/Kasey Drake/Fer2/Fer2 mutant [e]/2026-10-28/NewFer2mutant[e] 555Dbx -01.czi"
    ),
    0,
)

import numpy as np
import napari
from napari.layers import Image
import polars as pl
from skimage.feature import blob_log
from skimage.morphology import opening
import napari_segment_blobs_and_things_with_membranes
import scipy.ndimage as ndi
from skimage.feature import peak_local_max
import polars as pl


def detect_blobs_log(
    viewer,
    shapes_layer,
    image_layer,
    points_name="centroids",
    threshold=0.05,
    num_sigma=5,
    overlap=0.5,
):
    """blob_log on a 3D image; first two shapes = smallest/largest cell. One radius per shape, z ignored."""
    # One radius per shape in world units: mean of y/x half-extents
    r_world = np.array(
        [
            np.ptp(
                np.asarray(shapes_layer.data_to_world(np.asarray(s)))[:, 1:], axis=0
            ).mean()
            / 2
            for s in shapes_layer.data[:2]
        ]
    )
    r_min, r_max = r_world.min(), r_world.max()

    # Isotropic in world space -> per-axis sigma in voxels via image scale (sigma = r / sqrt(3))
    scale = np.asarray(image_layer.scale, dtype=float)
    min_sigma = tuple(np.maximum(2 * r_min / scale / np.sqrt(3), 0.5))
    max_sigma = tuple(np.maximum(2 * r_max / scale / np.sqrt(3), 0.5))

    img = np.asarray(image_layer.data, dtype=np.float32)
    img = (img - img.min()) / (np.ptp(img) + 1e-12)

    blobs = blob_log(
        img,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold=threshold,
        overlap=overlap,
        exclude_border=False,
    )

    return viewer.add_points(
        blobs[:, :3],
        name=points_name,
        scale=image_layer.scale,
        translate=image_layer.translate,
        size=2 * r_min,
        face_color="transparent",
        border_color="red",
        ndim=3,
    )


def max_opening_erosion(radius: float, image_layer: Image) -> pl.DataFrame:
    """
    radius: size of noise to remove, in pixels (xy); z radius is scaled by image_layer.scale
    image_layer: 3D napari Image layer (z, y, x) to find intensity maxima in
    returns a Polars dataframe with columns "label", "z", "y", "x", "max_intensity"
    (z, y, x in voxel coordinates; label is an index column)
    """
    img = np.asarray(image_layer.data, dtype=np.float32)
    scale = np.asarray(image_layer.scale[-3:], dtype=float)

    # Ellipsoid footprint: spherical in world space, radius px in the finest-scale axis
    r = radius * scale[-1] / scale  # per-axis radius in voxels
    grid = np.ogrid[tuple(slice(-int(np.ceil(ri)), int(np.ceil(ri)) + 1) for ri in r)]
    footprint = sum((g / ri) ** 2 for g, ri in zip(grid, r)) <= 1

    opened = ndi.grey_opening(img, footprint=footprint)
    smoothed = ndi.gaussian_filter(opened, sigma=radius / 2 * scale.min() / scale)
    viewer.add_image(smoothed, scale=scale)

    peaks = peak_local_max(smoothed, footprint=footprint, exclude_border=False)

    return pl.DataFrame(
        {
            "label": np.arange(len(peaks)),
            "z": peaks[:, 0],
            "y": peaks[:, 1],
            "x": peaks[:, 2],
            "max_intensity": smoothed[tuple(peaks.T)],
        }
    ).sort("max_intensity", descending=True)


def add_points(viewer, df, thresh):
    df = df.filter(pl.col("max_intensity")>thresh)
    return viewer.add_points(
        df[["z", "y", "x"]].to_numpy(),
        name="points",
        scale=viewer.layers[0].scale,
        size=5,
        face_color="transparent",
        border_color="red",
        ndim=3,
    )
