import numpy as np
from numpy.typing import ArrayLike
import pyift.shortestpath as sp

from napari.layers import Image, Labels
from napari.qt.threading import thread_worker
from napari_plugin_engine import napari_hook_implementation
from qtpy.QtWidgets import QMessageBox
from magicgui import magic_factory


def emit_message(text: str) -> None:
    msg_box = QMessageBox()
    msg_box.setText(text)
    msg_box.exec_()


@magic_factory(call_button='Segment',
               alpha=dict(value=-0.2, min=-1, max=1, step=0.1,
                          tooltip='Sets the algorithm orientation. Negative values favors segmentation from'
                                              'brighter to darker regions. Positive favors the opposite. Zero is neutral.'),
               background_label=dict(value=1),
               transform_intensity=dict(value=True, tooltip='Indicates if segmentation algorithm is applied to the'
                                                            'original image or with contrast and gamma changes.'))
def interactive_segmentation_widget(
        image_layer: Image,
        seeds_layer: Labels,
        output_layer: Labels,
        background_label: int,
        alpha: float,
        transform_intensity: bool,
) -> None:
    if image_layer is None or seeds_layer is None or output_layer is None:
        emit_message('All layers must be selected to perform segmentation.')
        return

    image = image_layer.data
    seeds = seeds_layer.data
    if image.shape != seeds.shape:
        emit_message(f'Image and seeds shape must match. {image.shape} and {seeds.shape} found.')
        return

    def _update_label(labels: ArrayLike) -> None:
        output_layer.data = labels

    @thread_worker(connect=dict(returned=_update_label))
    def _segmentation_worker(seeds: ArrayLike, image: ArrayLike, alpha: float) -> ArrayLike:
        if transform_intensity:
            image = image.astype(float)
            np.clip(image, *image_layer.contrast_limits, image)
            np.power(image, image_layer.gamma, out=image)
        _, _, _, labels = sp.oriented_seed_competition(seeds, image=image, alpha=alpha, background_label=-1)
        labels[labels == background_label] = 0
        return labels

    _segmentation_worker(seeds, image, alpha)


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return interactive_segmentation_widget, {'name': 'Interactive Segmentation'}
