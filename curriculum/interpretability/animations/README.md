# Manim animation sources

These short, silent visual explanations are authoring assets for the blog
edition. Readers do not need Manim: notebooks embed the reviewed MP4, poster,
captions, and transcript. The source deliberately uses no LaTeX or external
images.

Authoring environment:

- Manim Community 0.20.1;
- FFmpeg 6.1.1;
- Cairo 1.18 and PangoCairo 1.52; and
- 854×480, 15 fps, low-quality preview render.

From the repository root:

    .venv/bin/manim -ql --fps 15 -r 854,480 --media_dir /tmp/chess-interp-manim curriculum/interpretability/animations/scenes.py SuperpositionGeometry

    .venv/bin/manim -ql --fps 15 -r 854,480 --media_dir /tmp/chess-interp-manim curriculum/interpretability/animations/scenes.py JohnsonLindenstraussDistances

The committed files under assets/animations are publishing artifacts. After a
render, regenerate posters with FFmpeg, update the SHA-256 manifest, rerun the
course validator, and inspect motion, captions, and the reduced-motion/static
path before publishing.
