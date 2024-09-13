from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Sequence, Union

import fire
from loguru import logger

from mokuro import MokuroGenerator, __version__
from mokuro.legacy.overlay_generator import generate_legacy_html
from mokuro.volume import VolumeCollection


def run(
    *paths: Optional[Sequence[Union[str, Path]]],
    det_arch: str = "db_resnet50",
    det_weights: Path | None = Path(
        "./data/db_resnet50_20240815-081119_0.0807676_0.88_0.89.pt"
    ),
    reco_arch: str = "parseq",
    reco_weights: Path | None = Path(
        "./data/parseq_20240816-150838_0.450588_0.87_0.87.pt"
    ),
    max_ocr_width: int = 1200,
    parent_dir: Optional[Union[str, Path]] = None,
    force_cpu: bool = False,
    disable_confirmation: bool = False,
    disable_ocr: bool = False,
    ignore_errors: bool = False,
    no_cache: bool = False,
    unzip: bool = False,
    legacy_html: bool = True,
    as_one_file: bool = True,
    version: bool = False,
):
    """
    Process manga volumes with mokuro.

    Args:
        paths: Paths to manga volumes. Volume can be a directory, a zip file or a cbz file.
        det_arch: See link for options: https://mindee.github.io/doctr/modules/models.html#doctr-models-detection,
        det_weights: Path to detector weights from https://github.com/LiteralGenie/comic_ocr,,
        reco_arch: See link for options: https://mindee.github.io/doctr/modules/models.html#doctr-models-recognition,
        reco_weights: Path to recognizer weights from https://github.com/LiteralGenie/comic_ocr,
        max_ocr_width: Images are resized to this width (if wider) before being OCR'd. This improves results for images with large fonts (>60pt)
        parent_dir: Parent directory to scan for volumes. If provided, all volumes inside this directory will be processed.
        force_cpu: Force the use of CPU even if CUDA is available.
        disable_confirmation: Disable confirmation prompt. If False, the user will be prompted to confirm the list of volumes to be processed.
        disable_ocr: Disable OCR processing. Generate mokuro/HTML files without OCR results.
        ignore_errors: Continue processing volumes even if an error occurs.
        no_cache: Do not use cached OCR results from previous runs (_ocr directories).
        unzip: Extract volumes in zip/cbz format in their original location.
        legacy_html: Enable legacy HTML output. If True, acts as if --unzip is True.
        as_one_file: Applies only to legacy HTML. If False, generate separate CSS and JS files instead of embedding them in the HTML file.
        version: Print the version of mokuro and exit.
    """

    if version:
        print(f"{__version__}")
        return

    if disable_ocr:
        logger.info("Running with OCR disabled")

    if legacy_html:
        logger.warning(
            "Legacy HTML output is deprecated and will not be further developed. "
            "It's recommended to use .mokuro format and web reader instead. "
            "Legacy HTML will be disabled by default in the future. To explicitly enable it, run with option --legacy-html."
        )
        # legacy HTML works only with unzipped output
        unzip = True

    logger.info("Scanning paths...")

    paths_ = []
    for path in paths:
        path_normalized = Path(path).expanduser().absolute()

        try:
            path_valid = path_normalized.exists()
        except OSError:
            path_valid = False

        if path_valid:
            paths_.append(path_normalized)
        else:
            logger.error(f"Invalid path: {path_normalized}")
            return

    paths = paths_

    if parent_dir is not None:
        for p in Path(parent_dir).expanduser().absolute().iterdir():
            if (
                p not in paths
                and (p.is_dir() and p.stem != "_ocr")
                or (p.is_file() and p.suffix.lower() in {".zip", ".cbz"})
            ):
                paths.append(p)

    vc = VolumeCollection()

    for path_in in paths:
        vc.add_path_in(path_in)

    if len(vc) == 0:
        logger.error("Found no paths to process. Did you set the paths correctly?")
        return

    for title in vc.titles.values():
        title.set_uuid()

    status_counter = Counter()

    print(f"\nFound {len(vc)} volumes:\n")

    for volume in vc:
        print(volume)
        status_counter[volume.status] += 1

    msg = "\nEach of the paths above will be treated as one volume.\n"
    print(msg)

    if not disable_confirmation:
        inp = input("\nContinue? [yes/no]")
        if inp.lower() not in ("y", "yes"):
            return

    mg = MokuroGenerator(
        force_cpu=force_cpu,
        disable_ocr=disable_ocr,
        det_arch=det_arch,
        det_weights=det_weights,
        reco_arch=reco_arch,
        reco_weights=reco_weights,
        max_ocr_width=max_ocr_width,
    )

    with TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)

        # unzip == True means that zipped volumes will be unzipped in their original location
        # in that case, we don't use a temporary directory
        if unzip:
            tmp_dir = None

        num_sucessful = 0
        for i, volume in enumerate(vc):
            logger.info(f"Processing {i + 1}/{len(vc)}: {volume.path_in}")

            try:
                volume.unzip(tmp_dir)
                mg.process_volume(
                    volume, ignore_errors=ignore_errors, no_cache=no_cache
                )
                if legacy_html:
                    generate_legacy_html(
                        volume, as_one_file=as_one_file, ignore_errors=ignore_errors
                    )

            except Exception:
                logger.exception(f"Error while processing {volume.path_in}")
            else:
                num_sucessful += 1

        logger.info(f"Processed successfully: {num_sucessful}/{len(vc)}")


if __name__ == "__main__":
    fire.Fire(run)
