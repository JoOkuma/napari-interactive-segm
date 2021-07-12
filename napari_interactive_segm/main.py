import click
import napari

from tifffile import imsave, imread
import numpy as np

from pathlib import Path


@click.command()
@click.option('--im-dir', '-i', required=True, help='Input images directory.')
@click.option('--mk-dir', '-m', required=True, help='Output markers images directory.')
@click.option('--lb-dir', '-l', required=True, help='Output labels images directory')
def main(im_dir: str, mk_dir: str, lb_dir: str) -> None:
    im_dir = Path(im_dir)
    mk_dir = Path(mk_dir)
    lb_dir = Path(lb_dir)

    mk_dir.mkdir(exist_ok=True)
    lb_dir.mkdir(exist_ok=True)

    contrast_limits = (0, 1500)
    gamma = 1.0

    for im_path in im_dir.glob('*.tif*'):

        viewer = napari.Viewer()
        viewer.window.add_plugin_dock_widget("napari-interactive-segm")

        im = imread(im_path)
        mk = np.zeros_like(im, dtype=int)
        lb = np.zeros_like(im, dtype=int)

        im_layer = viewer.add_image(im, contrast_limits=contrast_limits, gamma=gamma)
        lb_layer = viewer.add_labels(lb)
        mk_layer = viewer.add_labels(mk)
        mk_layer.brush_size = 1

        viewer.show(block=True)

        contrast_limits = im_layer.contrast_limits
        gamma = im_layer.gamma

        imsave(lb_dir / im_path.name, lb_layer.data)
        imsave(mk_dir / im_path.name, mk_layer.data)


if __name__ == '__main__':
    main()
