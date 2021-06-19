# Copyright 2021 RangiLyu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2
import matplotlib as mpl
import matplotlib.figure as mplfigure
import numpy as np
import pycocotools.mask as mask_util
from matplotlib.backends.backend_agg import FigureCanvasAgg

_SMALL_OBJECT_AREA_THRESH = 1000


def overlay_bbox_cv(img, dets, class_names, score_thresh):
    all_box = []
    for label in dets:
        for bbox in dets[label]:
            score = bbox[-1]
            if score > score_thresh:
                x0, y0, x1, y1 = [int(i) for i in bbox[:4]]
                all_box.append([label, x0, y0, x1, y1, score])
    all_box.sort(key=lambda v: v[5])
    for box in all_box:
        label, x0, y0, x1, y1, score = box
        # color = self.cmap(i)[:3]
        color = (_COLORS[label] * 255).astype(np.uint8).tolist()
        text = "{}:{:.1f}%".format(class_names[label], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[label]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        txt_size = cv2.getTextSize(text, font, 0.5, 2)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        cv2.rectangle(
            img,
            (x0, y0 - txt_size[1] - 1),
            (x0 + txt_size[0] + txt_size[1], y0 - 1),
            color,
            -1,
        )
        cv2.putText(img, text, (x0, y0 - 1), font, 0.5, txt_color, thickness=1)
    return img


def rand_cmap(
    nlabels,
    type="bright",
    first_color_black=False,
    last_color_black=False,
    verbose=False,
):
    """
    Creates a random colormap to be used together with matplotlib.
    Useful for segmentation tasks
    :param nlabels: Number of labels (size of colormap)
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_black: Option to use first color as black, True or False
    :param last_color_black: Option to use last color as black, True or False
    :param verbose: Prints the number of labels and shows the colormap. True or False
    :return: colormap for matplotlib
    """
    import colorsys

    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    if type not in ("bright", "soft"):
        print('Please choose "bright" or "soft" for type')
        return

    if verbose:
        print("Number of labels: " + str(nlabels))

    # Generate color map for bright colors, based on hsv
    if type == "bright":
        randHSVcolors = [
            (
                np.random.uniform(low=0.0, high=1),
                np.random.uniform(low=0.2, high=1),
                np.random.uniform(low=0.9, high=1),
            )
            for i in range(nlabels)
        ]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(
                colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2])
            )

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        random_colormap = LinearSegmentedColormap.from_list(
            "new_map", randRGBcolors, N=nlabels
        )

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == "soft":
        low = 0.6
        high = 0.95
        randRGBcolors = [
            (
                np.random.uniform(low=low, high=high),
                np.random.uniform(low=low, high=high),
                np.random.uniform(low=low, high=high),
            )
            for i in range(nlabels)
        ]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list(
            "new_map", randRGBcolors, N=nlabels
        )

    return random_colormap


class VisImage:
    """
    Visualize detection results.

    Modified from Detectron2
    https://github.com/facebookresearch/detectron2
    """

    def __init__(self, img, scale=1.0):
        self.img = img
        self.scale = scale
        self.width, self.height = img.shape[1], img.shape[0]
        self._setup_figure(img)

    def _setup_figure(self, img):
        """
        Args:
            Same as in :meth:`__init__()`.

        Returns:
            fig (matplotlib.pyplot.figure): top level container for all the
                image plot elements.
            ax (matplotlib.pyplot.Axes): contains figure elements and sets
                the coordinate system.
        """
        fig = mplfigure.Figure(frameon=False)
        self.dpi = fig.get_dpi()
        # add a small 1e-2 to avoid precision lost due to matplotlib's truncation
        # (https://github.com/matplotlib/matplotlib/issues/15363)
        fig.set_size_inches(
            (self.width * self.scale + 1e-2) / self.dpi,
            (self.height * self.scale + 1e-2) / self.dpi,
        )
        self.canvas = FigureCanvasAgg(fig)
        # self.canvas = mpl.backends.backend_cairo.FigureCanvasCairo(fig)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.axis("off")
        ax.set_xlim(0.0, self.width)
        ax.set_ylim(self.height)

        self.fig = fig
        self.ax = ax

    def save(self, filepath):
        """
        Args:
            filepath (str): a string that contains the absolute path, including
                the file name, where the visualized image will be saved.
        """
        if filepath.lower().endswith(".jpg") or filepath.lower().endswith(".png"):
            # faster than matplotlib's imshow
            cv2.imwrite(filepath, self.get_image()[:, :, ::-1])
        else:
            # support general formats (e.g. pdf)
            self.ax.imshow(self.img, interpolation="nearest")
            self.fig.savefig(filepath)

    def get_image(self):
        """
        Returns:
            ndarray:
                the visualized image of shape (H, W, 3) (RGB) in uint8 type.
                The shape is scaled w.r.t the input image using the given
                `scale` argument.
        """
        canvas = self.canvas
        s, (width, height) = canvas.print_to_buffer()
        if (self.width, self.height) != (width, height):
            img = cv2.resize(self.img, (width, height))
        else:
            img = self.img

        # buf = io.BytesIO()  # works for cairo backend
        # canvas.print_rgba(buf)
        # width, height = self.width, self.height
        # s = buf.getvalue()

        buffer = np.frombuffer(s, dtype="uint8")

        # imshow is slow. blend manually (still quite slow)
        img_rgba = buffer.reshape(height, width, 4)
        rgb, alpha = np.split(img_rgba, [3], axis=2)

        try:
            import numexpr as ne  # fuse them with numexpr

            visualized_image = ne.evaluate(
                "img * (1 - alpha / 255.0) + rgb * (alpha / 255.0)"
            )
        except ImportError:
            alpha = alpha.astype("float32") / 255.0
            visualized_image = img * (1 - alpha) + rgb * alpha

        visualized_image = visualized_image.astype("uint8")

        return visualized_image


class Visualizer:
    def __init__(self, img, dets, class_names, socre_thresh):
        self.img = img
        self.dets = dets
        self.class_names = class_names
        self.num_classes = len(self.class_names)
        self.score_thresh = socre_thresh
        self.viz = VisImage(img=self.img)
        self._default_font_size = max(
            np.sqrt(self.viz.height * self.viz.width) // 100, 10
        )

    def mask_to_polygon(self, mask, need_binary=True):
        res = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        hierarchy = res[-1]
        if hierarchy is None:  # empty mask
            return None, None, None
        has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
        res = res[-2]
        res = [x.flatten() for x in res]
        res = [x for x in res if len(x) >= 6]

        p = mask_util.frPyObjects(res, self.viz.height, self.viz.width)
        p = mask_util.merge(p)
        bbox = mask_util.toBbox(p)
        bbox[2] += bbox[0]
        bbox[3] += bbox[1]

        return res, bbox, has_holes

    def draw_box(self, box_coord, alpha=0.5, edge_color="g", line_style="-"):
        x0, y0, x1, y1 = box_coord
        width = x1 - x0
        height = y1 - y0
        linewidth = max(self._default_font_size / 6, 1)
        self.viz.ax.add_patch(
            mpl.patches.Rectangle(
                (x0, y0),
                width,
                height,
                fill=False,
                edgecolor=edge_color,
                linewidth=linewidth * self.viz.scale,
                alpha=alpha,
                linestyle=line_style,
            )
        )
        return self.viz

    def draw_polycon(self, mask, color, edge_color, alpha=0.5):
        if edge_color is None:
            edge_color = color
        edge_color = mpl.colors.to_rgb(edge_color) + (1,)

        polygon = mpl.patches.Polygon(
            mask,
            fill=False,
            # facecolor=mpl.colors.to_rgb(color) + (alpha,),
            edgecolor=edge_color,
            linewidth=max(self._default_font_size // 15 * self.viz.scale, 1),
        )
        self.viz.ax.add_patch(polygon)
        return self.viz

    def draw_mask(self, mask, polys, color, edge_color, alpha=0.5):
        if edge_color is None:
            edge_color = color
        edge_color = mpl.colors.to_rgb(edge_color) + (1,)
        color_mask = np.ones((mask.shape[0], mask.shape[1], 3))
        for i in range(3):
            color_mask[:, :, i] = color[i]
        self.viz.ax.imshow(np.dstack((color_mask, mask * alpha)))
        for ploy in polys:
            self.draw_polycon(ploy.reshape(-1, 2), color, edge_color=None, alpha=alpha)

    def _jitter(self, color):
        """
        Randomly modifies given color to produce a slightly different color than
        the color given.

        Args:
            color (tuple[double]): a tuple of 3 elements, containing the RGB
                values of the color picked. The values in the list are in the
                 [0.0, 1.0] range.

        Returns:
            jittered_color (tuple[double]): a tuple of 3 elements, containing
                the RGB values of the color after being jittered. The values
                in the list are in the [0.0, 1.0] range.
        """
        color = mpl.colors.to_rgb(color)
        vec = np.random.rand(3)
        # better to do it in another color space
        vec = vec / np.linalg.norm(vec) * 0.5
        res = np.clip(vec + color, 0, 1)
        return tuple(res)

    def overlay_bbox(self, alpha=1.0):
        for label in self.dets:
            for bbox in self.dets[label]:
                x0, y0, x1, y1, score = bbox
                if score >= self.score_thresh:
                    # color = self.cmap(i)[:3]
                    color = _COLORS[label]
                    text = "{}:{:.1f}%".format(self.class_names[label], score * 100)
                    self.draw_box(bbox[:4], alpha=1.0, edge_color=color, line_style="-")
                    text_pos = (x0, y0)
                    instance_area = (y1 - y0) * (x1 - x0)
                    if (
                        instance_area < _SMALL_OBJECT_AREA_THRESH * self.viz.scale
                        or y1 - y0 < 40 * self.viz.scale
                    ):
                        if y1 >= self.viz.height - 5:
                            text_pos = (x1, y0)
                        else:
                            text_pos = (x0, y1)

                    height_ratio = (y1 - y0) / np.sqrt(self.viz.height * self.viz.width)
                    font_size = (
                        np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2)
                        * 0.5
                        * self._default_font_size
                    )

                    self.draw_text(
                        text,
                        text_pos,
                        color="black",
                        horizontal_alignment="left",
                        font_size=font_size,
                    )
        out = self.viz.get_image()
        return out

    def overlay_masks(self, alpha=0.5):
        ov = self.img.copy()
        im = self.img  # .astype(np.float32)
        total_ma = np.zeros([im.shape[0], im.shape[1]])
        total_contours = []
        for i, det in enumerate(self.dets[::-1]):
            score = det["score"]
            if score >= self.score_thresh:
                ma = det["mask"]
                _, ma = cv2.threshold(
                    ma, thresh=127, maxval=255, type=cv2.THRESH_BINARY
                )
                fg = (
                    im * alpha
                    + np.ones(im.shape) * (1 - alpha) * self.cmap(i)[:3] * 255
                )
                ov[ma == 255] = fg[ma == 255]
                total_ma += ma
                contours = cv2.findContours(
                    ma.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
                )[-2:]
                total_contours.append(contours)
        for cnt in total_contours:
            cv2.drawContours(ov, cnt[0], -1, (0.0, 0.0, 0.0), 1)
        ov[total_ma == 0] = im[total_ma == 0]
        return ov

    def overlay_instance(self, alpha=0.4):
        for i, det in enumerate(self.dets[::-1]):
            score = det["score"]
            if score >= self.score_thresh:
                label = det["label"]
                binary_mask = det["mask"]
                # color = self.cmap(i)[:3]
                color = _COLORS[label]
                color = self._jitter(color)
                contours, bbox, has_holes = self.mask_to_polygon(binary_mask.copy())
                if not contours:
                    continue
                self.draw_mask(
                    binary_mask, contours, color, edge_color=None, alpha=alpha
                )

                x0, y0, x1, y1 = bbox
                text = "{}:{:.1f}%".format(self.class_names[label], score * 100)
                text_pos = np.median(binary_mask.nonzero(), axis=1)[::-1]
                instance_area = (y1 - y0) * (x1 - x0)
                if (
                    instance_area < _SMALL_OBJECT_AREA_THRESH * self.viz.scale
                    or y1 - y0 < 40 * self.viz.scale
                ):
                    if y1 >= self.viz.height - 5:
                        text_pos = (x1, y0)
                    else:
                        text_pos = (x0, y1)

                height_ratio = (y1 - y0) / np.sqrt(self.viz.height * self.viz.width)
                font_size = (
                    np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2)
                    * 0.5
                    * self._default_font_size
                )

                self.draw_text(
                    text,
                    text_pos,
                    color="black",
                    horizontal_alignment="center",
                    font_size=font_size,
                )
        out = self.viz.get_image()
        return out

    def draw_text(
        self,
        text,
        position,
        *,
        font_size=None,
        color="g",
        horizontal_alignment="center",
        rotation=0
    ):
        """
        Args:
            text (str): class label
            position (tuple): a tuple of the x and y coordinates to place text on image.
            font_size (int, optional): font of the text. If not provided, a font size
                proportional to the image width is calculated and used.
            color: color of the text. Refer to `matplotlib.colors` for full list
                of formats that are accepted.
            horizontal_alignment (str): see `matplotlib.text.Text`
            rotation: rotation angle in degrees CCW

        Returns:
            output (VisImage): image object with text drawn.
        """
        if not font_size:
            font_size = self._default_font_size

        # since the text background is dark, we don't want the text to be dark
        color = np.maximum(list(mpl.colors.to_rgb(color)), 0.2)
        color[np.argmax(color)] = max(0.8, np.max(color))

        x, y = position
        self.viz.ax.text(
            x,
            y,
            text,
            size=font_size * self.viz.scale,
            family="sans-serif",
            bbox={
                "facecolor": (0.5, 0.5, 1.0),
                "alpha": 0.8,
                "pad": 0.7,
                "edgecolor": (0.8, 0.8, 1.0),
            },
            verticalalignment="top",
            horizontalalignment=horizontal_alignment,
            color=color,
            zorder=10,
            rotation=rotation,
        )
        return self.viz


_COLORS = (
    np.array(
        [
            0.000,
            0.447,
            0.741,
            0.850,
            0.325,
            0.098,
            0.929,
            0.694,
            0.125,
            0.494,
            0.184,
            0.556,
            0.466,
            0.674,
            0.188,
            0.301,
            0.745,
            0.933,
            0.635,
            0.078,
            0.184,
            0.300,
            0.300,
            0.300,
            0.600,
            0.600,
            0.600,
            1.000,
            0.000,
            0.000,
            1.000,
            0.500,
            0.000,
            0.749,
            0.749,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            1.000,
            0.667,
            0.000,
            1.000,
            0.333,
            0.333,
            0.000,
            0.333,
            0.667,
            0.000,
            0.333,
            1.000,
            0.000,
            0.667,
            0.333,
            0.000,
            0.667,
            0.667,
            0.000,
            0.667,
            1.000,
            0.000,
            1.000,
            0.333,
            0.000,
            1.000,
            0.667,
            0.000,
            1.000,
            1.000,
            0.000,
            0.000,
            0.333,
            0.500,
            0.000,
            0.667,
            0.500,
            0.000,
            1.000,
            0.500,
            0.333,
            0.000,
            0.500,
            0.333,
            0.333,
            0.500,
            0.333,
            0.667,
            0.500,
            0.333,
            1.000,
            0.500,
            0.667,
            0.000,
            0.500,
            0.667,
            0.333,
            0.500,
            0.667,
            0.667,
            0.500,
            0.667,
            1.000,
            0.500,
            1.000,
            0.000,
            0.500,
            1.000,
            0.333,
            0.500,
            1.000,
            0.667,
            0.500,
            1.000,
            1.000,
            0.500,
            0.000,
            0.333,
            1.000,
            0.000,
            0.667,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            0.000,
            1.000,
            0.333,
            0.333,
            1.000,
            0.333,
            0.667,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            0.000,
            1.000,
            0.667,
            0.333,
            1.000,
            0.667,
            0.667,
            1.000,
            0.667,
            1.000,
            1.000,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            1.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.143,
            0.143,
            0.143,
            0.286,
            0.286,
            0.286,
            0.429,
            0.429,
            0.429,
            0.571,
            0.571,
            0.571,
            0.714,
            0.714,
            0.714,
            0.857,
            0.857,
            0.857,
            0.000,
            0.447,
            0.741,
            0.314,
            0.717,
            0.741,
            0.50,
            0.5,
            0,
        ]
    )
    .astype(np.float32)
    .reshape(-1, 3)
)
