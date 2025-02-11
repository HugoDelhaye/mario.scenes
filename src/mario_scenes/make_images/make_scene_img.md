# Create scene images

## Description
This script extracts scenes from the manually annotated .pptx file. **NOTE : These images are only approximations of the positions used in the scene_mastersheet.csv, which lead the real clip extractions. They are generated here for illustrative purposes.**
It works by finding red-bordered shapes from slides in a PowerPoint presentation and saving them as cropped images.

## Requirements
- Python 3.x
- `python-pptx`
- `pdf2image`
- `opencv-python`
- `numpy`

Install dependencies with:

```bash
pip install python-pptx pdf2image opencv-python numpy
```

## Usage

```bash
python make_scene_img.py [pptx_file] [output_folder]
```

- `pptx_file` (optional): Path to the PowerPoint file (default: `resources/mario_scenes_manual_annotation.pptx`).
- `output_folder` (optional): Path to save cropped images (default: `resources/scenes_images`).

## Output
Cropped images are saved in the format:

```
w<world>l<level>s<scene>.jpg
```

Example:

```
w1l1s1.jpg
w1l1s2.jpg
```

## Notes
- Ensure the PowerPoint file is in `.pptx` format.
- Red shapes should have an outline color of RGB (255, 0, 0).

