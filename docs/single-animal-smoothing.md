# Single-animal smoothing

In Segmentation, expand **Single-animal Smoothing (Optional)** below
**Additional Outlier** and enable **Smooth single-animal blobs**. It is off by
default. **Smoothing level** ranges from 1 to 20 (initial value 3); increase it
to remove thicker protrusions. With IQR classification, run **Analyze** first.
The preview updates when the level changes. Save or run Processing to persist
the settings in `segmentation_gui_config.json`.

Only blue, non-crossing blobs are smoothed, including animals rescued by Result
OBB Import. The remaining contour is used for preview, screenshots, labeled
video export, and the segmentation pickle consumed by tracking and training.
Removed pixels become background; they are not exported as new outlier blobs.
Any overlapping outlier pixels are also removed so they cannot restore the
discarded protrusions.

## Algorithm and processing order

- Classify original contours using the existing area bounds and OBB matching.
  Keep sampled-frame statistics and cached source masks unchanged.
- Rasterize each selected single-animal contour in a padded local crop.
- Apply circular morphological opening (erosion followed by dilation). The
  radius is 4% of the maximum inscribed radius per level, rounded to pixels,
  with a minimum of one pixel where the blob is thick enough. This restores
  the body after erosion without fitting an ellipse or adding foreground.
- Reduce the radius if it would split the animal, introduce a hole, or remove
  more than one third of its pixels. If no positive radius is safe, keep the
  original contour. Small blobs and adjacent levels can therefore produce
  the same result, and removal of thick appendages is deliberately limited.
- Update the output contour's area, center and bounding box while retaining
  its single-animal classification. Do not classify its reduced area again.
- Subtract removed pixels from overlapping outliers too. The pickle format
  holds filled contours without hole hierarchies, so a residual outlier with
  a hole is split into filled pieces rather than filling discarded pixels.

Smoothing always starts from the original classified contours. Repeated
redraws do not progressively shrink the animals, and turning the option off
returns to the existing pipeline. Parallel Processing uses a settings snapshot
rather than reading Tk variables from worker threads.

Saved settings are `single_blob_smoothing_enabled` (boolean, default `false`)
and `single_blob_smoothing_level` (integer, 1–20, default `3`). Loading an older
configuration without these keys resets this option to its defaults.

The raw foreground-mask helper used independently by identity correction is
unchanged: it has no single-animal/outlier classification. Smoothing is applied
to the Segmentation output contours, not indiscriminately to raw foreground.
