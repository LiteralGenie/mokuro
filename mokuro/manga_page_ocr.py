from pathlib import Path
from statistics import median
from typing import Generator, Literal

import cv2
import doctr
import numpy as np
import torch
from comic_ocr.lib.constants import KOREAN_ALPHABET
from comic_ocr.lib.inference_utils import calc_windows, eval_window
from comic_ocr.lib.label_utils import OcrMatch, stitch_blocks, stitch_lines
from doctr.models.predictor import OCRPredictor
from loguru import logger
from manga_ocr import MangaOcr
from PIL import Image
from scipy.signal.windows import gaussian

from comic_text_detector.inference import TextDetector
from mokuro import __version__
from mokuro.cache import cache
from mokuro.utils import imread


class InvalidImage(Exception):
    def __init__(self, message="Animation file, Corrupted file or Unsupported type"):
        super().__init__(message)


class MangaPageOcr:
    def __init__(
        self,
        det_arch: str,
        det_weights: Path | None,
        reco_arch: str,
        reco_weights: Path | None,
        max_ocr_width: int,
        force_cpu=False,
        detector_input_size=1024,
        text_height=64,
        max_ratio_vert=16,
        max_ratio_hor=8,
        anchor_window=2,
        disable_ocr=False,
    ):
        self.det_arch = det_arch
        self.det_weights = det_weights
        self.reco_arch = reco_arch
        self.reco_weights = reco_weights
        self.max_ocr_width = max_ocr_width

        self.text_height = text_height
        self.max_ratio_vert = max_ratio_vert
        self.max_ratio_hor = max_ratio_hor
        self.anchor_window = anchor_window
        self.disable_ocr = disable_ocr

        if not self.disable_ocr:
            cuda = torch.cuda.is_available()
            device = "cuda" if cuda and not force_cpu else "cpu"
            logger.info(f"Initializing text detector, using device {device}")
            self.text_detector = TextDetector(
                model_path=cache.comic_text_detector,
                input_size=detector_input_size,
                device=device,
                act="leaky",
            )
            self.predictor = _load_predictor(
                self.det_arch,
                self.det_weights,
                self.reco_arch,
                self.reco_weights,
                device,
            )

    def __call__(self, img_path):
        img = imread(img_path)
        if img is None:
            raise InvalidImage()
        H, W, *_ = img.shape
        result = {"version": __version__, "img_width": W, "img_height": H, "blocks": []}

        if self.disable_ocr:
            return result

        matches = _ocr_page(self.predictor, img_path, self.max_ocr_width)
        lines = stitch_lines(matches)
        blocks = stitch_blocks(lines)

        for blk_idx, blk in enumerate(blocks):
            # This doesn't seem to do anything
            med_line_height = median([ln.height for ln in blk.lines])
            font_size = int(med_line_height / 1.3)

            y1, x1, y2, x2 = blk.bbox
            result_blk = {
                "box": [x1, y1, x2, y2],
                "vertical": False,
                "font_size": int(font_size),
                "lines_coords": [],
                "lines": [],
            }

            for line_idx, line in enumerate(blk.lines):
                y1, x1, y2, x2 = line.bbox
                result_blk["lines_coords"].append(
                    [
                        [x1, y1],
                        [x1, y2],
                        [x2, y2],
                        [x2, y1],
                    ]
                )
                result_blk["lines"].append(line.value)

            result["blocks"].append(result_blk)

        return result

    @staticmethod
    def split_into_chunks(
        img, mask_refined, blk, line_idx, textheight, max_ratio=16, anchor_window=2
    ):
        line_crop = blk.get_transformed_region(img, line_idx, textheight)

        h, w, *_ = line_crop.shape
        ratio = w / h

        if ratio <= max_ratio:
            return [line_crop], []

        else:
            k = gaussian(textheight * 2, textheight / 8)

            line_mask = blk.get_transformed_region(mask_refined, line_idx, textheight)
            num_chunks = int(np.ceil(ratio / max_ratio))

            anchors = np.linspace(0, w, num_chunks + 1)[1:-1]

            line_density = line_mask.sum(axis=0)
            line_density = np.convolve(line_density, k, "same")
            line_density /= line_density.max()

            anchor_window *= textheight

            cut_points = []
            for anchor in anchors:
                anchor = int(anchor)

                n0 = np.clip(anchor - anchor_window // 2, 0, w)
                n1 = np.clip(anchor + anchor_window // 2, 0, w)

                p = line_density[n0:n1].argmin()
                p += n0

                cut_points.append(p)

            return np.split(line_crop, cut_points, axis=1), cut_points


def _ocr_page(
    predictor: OCRPredictor, fp_image: Path, max_ocr_width: int
) -> list[OcrMatch]:
    im = Image.open(fp_image)

    resize_mult = 1
    if im.size[0] > max_ocr_width:
        w, h = im.size

        resize_mult = max_ocr_width / w
        new_size = (int(w * resize_mult), int(h * resize_mult))

        print(f"Resizing image from {(w,h)} to {new_size}")
        im = im.resize(new_size)

    windows = calc_windows(
        im.size,
        1024,
        100,
    )

    matches: list[OcrMatch] = []
    for idx, w in enumerate(windows):
        r = eval_window(predictor, im, w, 0)

        matches.extend(r["matches"])

    if resize_mult != 1:
        matches = [_rescale(m, 1 / resize_mult) for m in matches]

    return matches


def _load_predictor(
    det_arch: str,
    det_weights: Path | None,
    reco_arch: str,
    reco_weights: Path | None,
    device: Literal["cuda"] | str,
) -> OCRPredictor:
    det_model = doctr.models.detection.__dict__[det_arch](
        pretrained=False,
        pretrained_backbone=False,
    )
    if det_weights:
        print(f"Loading detector model weights from {det_weights}")
        det_params = torch.load(
            Path(det_weights),
            map_location="cpu",
            weights_only=True,
        )
        det_model.load_state_dict(det_params)

    reco_model = doctr.models.recognition.__dict__[reco_arch](
        vocab=KOREAN_ALPHABET,
        pretrained=False,
        pretrained_backbone=False,
    )
    if reco_weights:
        print(f"Loading recognizer model weights from {reco_weights}")
        reco_params = torch.load(
            Path(reco_weights),
            map_location="cpu",
            weights_only=True,
        )
        reco_model.load_state_dict(reco_params)

    predictor = doctr.models.ocr_predictor(
        det_arch=det_model,
        reco_arch=reco_model,
    )

    if device == "cuda":
        predictor = predictor.cuda()

    return predictor


def _rescale(match: OcrMatch, k: float) -> OcrMatch:
    y1, x1, y2, x2 = match.bbox
    y1 = int(y1 * k)
    x1 = int(x1 * k)
    y2 = int(y2 * k)
    x2 = int(x2 * k)

    bbox = (y1, x1, y2, x2)

    return OcrMatch(
        bbox,
        match.confidence,
        match.value,
    )
