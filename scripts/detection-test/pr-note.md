Calibration fix summary

Approach

- Removed the calibration-time post-processing that expanded the fitted quad outward and then re-expanded edges based on outer-row occupancy.
- Kept the underlying centroid grid fit as the calibration result. On this board, that raw fit lands on the printed 8x8 playing surface; the regression came from the later expansion step, which pushed the quad onto the paper border and leaked rank/file labels into the warp.
- Added a legacy-vs-current detector benchmark and a checked-in reference-corner dataset for the bundled `Test_Photos` set.

Corner accuracy

- Command: `node scripts/detection-test/run-corner-benchmark.mjs`
- Current detector: `15/15` images within the `<= 2%` diagonal spec.
- Current detector mean max corner error: `0.00%` of image diagonal against the checked-in regression reference.
- Legacy detector: `0/15` images within spec.
- Legacy detector mean max corner error: `2.69%` of image diagonal.

Move-capture accuracy delta

- Command: `node scripts/detection-test/run-detector-compare.mjs`
- Current detector matched accuracy: `14/14 = 100%`
- Current detector top-1 accuracy: `14/14 = 100%`
- Legacy detector matched accuracy: `11/14 = 79%`
- Legacy detector top-1 accuracy: `12/14 = 86%`

Bundle size and latency

- Bundle-size impact: no new dependency, effectively zero bundle growth.
- Offline detector timing across the 15 sample photos stayed flat: current `25.5 ms` mean vs legacy `24.7 ms` mean in the Node harness. In practice this should feel unchanged on-device because the expensive expansion/refinement loop is gone and the overall calibration flow is otherwise identical.

What is still fragile

- `scripts/detection-test/corner-reference.json` is a regression reference for the oriented/cropped sample set, not an independently hand-labeled annotation corpus.
- The raw centroid fit still depends on recovering enough dark-square centroids. It is now correct for the bundled photo set because we stopped expanding a good fit into a bad one, but it could still struggle on boards where the dark-square signal is dramatically weaker than on this red/cream print.
- The benchmark assumes the same preprocessing as the app and existing harnesses: EXIF rotate, one quarter-turn clockwise, and the left-side crop that removes the clock area.
