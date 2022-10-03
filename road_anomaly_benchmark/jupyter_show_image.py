
# Display images in jupyter web UI.
# Encodes the image directly into HTML (base64) without going through matplotlib. 
# The compression format is customizable. A single function call can show multiple images in a grid.
# 
# `show(img_1, img_2)` will draw each image on a separate row
# `show([img_1, img_2])` will draw both images in one row
# `show([img_1, img_2], [img_3, img_4])` will draw two rows

# Specifying the format:
# 	`show(..., fmt='webp')`: image format, usually png jpeg webp
# Whether to try converting unusual shapes and datatypes to the needed RGB:
# 	`show(..., adapt=True or False)`

# 2020 Krzysztof Lis
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


from pathlib import Path
from io import BytesIO
from binascii import b2a_base64

import numpy as np
from PIL import Image as PIL_Image
from matplotlib import cm
from IPython.display import display_html

from skimage.segmentation import mark_boundaries
from scipy.ndimage.morphology import binary_dilation
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import cv2
matplotlib.use('Agg')

### Image IO

def imread(path):
    return np.asarray(PIL_Image.open(path))

IMWRITE_OPTS = dict(
    webp = dict(quality = 85),
)

def imwrite(path, data, format=None):
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)

    PIL_Image.fromarray(data).save(
        path,
        format = format,
        **IMWRITE_OPTS.get(path.suffix.lower()[1:], {}),
    )


def get_boundary_mask(arr, index=1):
    arr[arr != index] = 0
    arr[arr == index] = 255
    bd = mark_boundaries(arr, arr)
    mask = binary_dilation(np.all(bd == [1, 1, 0], axis=-1).astype(int)).astype("uint8")
    mask = cv2.dilate(mask, kernel = np.ones((5, 5), np.uint8), iterations=1)
    bd[np.all(bd == np.ones(3), axis=-1)] = np.zeros(3)
    bd[mask == 1] = [0, 1, 0]
    im = (bd * 255).astype("uint8")
    return im


def get_heat(heat_array, overlay=None, vrange=None):
    heat_array = heat_array.astype("float32")
    h, w = heat_array.shape
    dpi = 100
    fig = plt.Figure(figsize=(w / dpi, h / dpi), dpi=dpi, frameon=False)
    canvas = FigureCanvas(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    if vrange is None:
        ax.imshow(heat_array, cmap=plt.get_cmap('RdBu_r', 16), interpolation='none')
    else:
        ax.imshow(heat_array, cmap=plt.get_cmap('RdBu_r', 16), interpolation='none', vmin=vrange[0], vmax=vrange[1])
    canvas.draw()
    heat_arr = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape((h, w, 3))
    if overlay is not None:
        bd = get_boundary_mask(overlay)
        heat_arr[~np.all(bd == (0, 0, 0), axis=-1)] = bd[~np.all(bd == (0, 0, 0), axis=-1)]
    return heat_arr


### Image display

def adapt_img_data(img_data, cmap_pos=cm.get_cmap('magma'), cmap_div=cm.get_cmap('Spectral').reversed(), value_range=None):
    """
    Produce a HxWx3 uint8 image given a data array.
    If the array is 1-channel, we use matplotlib colormap to colorize it
    If the array is float, we may convert 0...1 to 0...255.
    Boolean image is shown as black vs white.

    @param img_data: data array to display
    @param cmap_pos: colormap for all-positive data
    @param cmap_div: colormap for when the array contains positive and negative data - these are drawn with different colors
    """
    num_dims = img_data.shape.__len__()

    if num_dims == 3 or num_dims == 4:
        # if img_data.shape[2] > 3:
        # 	img_data = img_data[:, :, :3]

        if img_data.dtype != np.uint8:
            if np.max(img_data) < 1.1:
                img_data = img_data * 255
            img_data = img_data.astype(np.uint8)

    elif num_dims == 2:
        if img_data.dtype == np.bool:
            img_data = img_data.astype(np.uint8)*255
            #c = 'png'

        else:
            if value_range is not None:
                vmin, vmax = value_range
            else:
                vmin, vmax = np.min(img_data), np.max(img_data)

            # vmax = np.max(img_data)
            if img_data.dtype == np.uint8 and vmax == 1:
                img_data = img_data * 255

            else:
                #vmin = np.min(img_data)

                if vmin >= 0:
                    img_data = (img_data - vmin) * (1 / (vmax - vmin))
                    img_data = cmap_pos(img_data, bytes=True)[:, :, :3]

                else:
                    vrange = max(-vmin, vmax)
                    img_data = img_data / (2 * vrange) + 0.5
                    img_data = cmap_div(img_data, bytes=True)[:, :, :3]

    return img_data


class ImageHTML:
    """
    Represents an image as a HTML <img> with the data encoded as base64
    """
    CONTENT_TMPL = """<div style="width:100%;"><img src="data:image/{fmt};base64,{data}" /></div>"""

    def __init__(self, image_data, fmt='webp', adapt=True):
        self.fmt = fmt
        image_data = adapt_img_data(image_data) if adapt else image_data
        self.data_base64 = self.encode_image(image_data, fmt)

    @staticmethod
    def encode_image(image, fmt):
        with BytesIO() as buffer:
            PIL_Image.fromarray(image).save(buffer, format=fmt)
            image_base64 = str(b2a_base64(buffer.getvalue()), 'utf8')
        return image_base64

    def _repr_html_(self):
        return self.CONTENT_TMPL.format(fmt=self.fmt, data=self.data_base64)

    def show(self):
        display_html(self)


class ImageGridHTML:
    """
    Represents an collections of images as a grid in HTML.
    Each of the positional arguments gets a separate row.
    If an argument value is a list of images, it will be drawn as columns in the row.
    """

    ROW_START = """<div style="display:flex; justify-content: space-evenly;">"""
    ROW_END = """</div>"""

    def __init__(self, *rows, fmt='webp', adapt=True):
        """
        `show(img_1, img_2)` will draw each image on a separate row
        `show([img_1, img_2])` will draw both images in one row
        `show([img_1, img_2], [img_3, img_4])` will draw two rows

        @param fmt: image format, usually png jpeg webp
        @param adapt: whether to try converting unusual shapes and datatypes to the needed RGB
        """
        self.fmt = fmt
        self.adapt = adapt
        self.rows = [self.encode_row(r) for r in rows]

    def encode_row(self, row):
        if isinstance(row, (list, tuple)):
            return [ImageHTML(img, fmt=self.fmt, adapt=self.adapt) for img in row if img is not None]
        elif row is None:
            return []
        else:
            return [ImageHTML(row, fmt=self.fmt, adapt=self.adapt)]

    def _repr_html_(self):
        fragments = []

        for row in self.rows:
            fragments.append(self.ROW_START)
            fragments += [img._repr_html_() for img in row]
            fragments.append(self.ROW_END)

        return '\n'.join(fragments)

    def show(self):
        display_html(self)

    @staticmethod
    def show_image(*images, **options):
        """
        `show(img_1, img_2)` will draw each image on a separate row
        `show([img_1, img_2])` will draw both images in one row
        `show([img_1, img_2], [img_3, img_4])` will draw two rows

        @param fmt: image format, usually png jpeg webp
        @param adapt: whether to try converting unusual shapes and datatypes to the needed RGB
        """
        ImageGridHTML(*images, **options).show()

show = ImageGridHTML.show_image
