"""Short authoring scenes for the interpretability visual essays.

Render with Manim Community 0.20.1. These scenes avoid LaTeX and external
assets so that the only authoring dependencies are Manim, Pango/Cairo, and
FFmpeg. The videos contain no speech; adjacent WEBVTT tracks describe motion.
"""

from __future__ import annotations

import numpy as np
from manim import (
    AnimationGroup,
    Arrow,
    BLUE_C,
    DOWN,
    FadeIn,
    FadeOut,
    GREEN_C,
    GrowArrow,
    LEFT,
    ORANGE,
    PURPLE_C,
    Rectangle,
    RIGHT,
    Scene,
    Text,
    UP,
    VGroup,
    WHITE,
    Write,
    YELLOW,
    config,
)


BACKGROUND = "#101722"
INK = "#eef4ff"
MUTED = "#aebbd0"
COLORS = (BLUE_C, ORANGE, GREEN_C, PURPLE_C, YELLOW)


def _label(text: str, *, size: int = 26, color=INK) -> Text:
    return Text(text, font_size=size, color=color, font="DejaVu Sans")


class SuperpositionGeometry(Scene):
    """Sparse features share a two-dimensional activation space."""

    def construct(self) -> None:
        self.camera.background_color = BACKGROUND
        title = _label("Superposition: more features than dimensions", size=34).to_edge(UP)
        subtitle = _label(
            "Non-orthogonal co-activation creates decoder cross-talk",
            size=21,
            color=MUTED,
        ).next_to(title, DOWN, buff=0.18)
        self.play(Write(title), FadeIn(subtitle))

        origin = LEFT * 2.5 + DOWN * 0.45
        x_axis = Arrow(origin + LEFT * 0.2, origin + RIGHT * 2.7, buff=0, color=MUTED)
        y_axis = Arrow(origin + DOWN * 0.2, origin + UP * 2.35, buff=0, color=MUTED)
        axis_labels = VGroup(
            _label("neuron 1", size=18, color=MUTED).next_to(x_axis, DOWN),
            _label("neuron 2", size=18, color=MUTED).next_to(y_axis, UP, buff=0.06),
        )
        self.play(GrowArrow(x_axis), GrowArrow(y_axis), FadeIn(axis_labels))

        angles = np.deg2rad([0, 55, 125, 200, 285])
        arrows = VGroup()
        labels = VGroup()
        for index, (angle, color) in enumerate(zip(angles, COLORS, strict=True), start=1):
            end = origin + 1.55 * np.asarray([np.cos(angle), np.sin(angle), 0.0])
            arrow = Arrow(origin, end, buff=0, color=color, stroke_width=6)
            label = _label(f"f{index}", size=18, color=color).next_to(
                arrow.get_end(), RIGHT if np.cos(angle) >= 0 else LEFT, buff=0.08
            )
            arrows.add(arrow)
            labels.add(label)
        self.play(
            AnimationGroup(
                *(GrowArrow(arrow) for arrow in arrows),
                lag_ratio=0.18,
            ),
            AnimationGroup(*(FadeIn(label) for label in labels), lag_ratio=0.18),
        )

        capacity = _label("5 candidate directions / 2 axes", size=25).move_to(
            RIGHT * 2.25 + UP * 1.65
        )
        capacity.scale_to_fit_width(3.6)
        self.play(Write(capacity))
        self.wait(0.4)

        active_one = arrows[0].copy().set_stroke(width=12)
        active_two = arrows[1].copy().set_stroke(width=12)
        sparse_lines = VGroup(
            _label("Sparse input:  x = d1", size=24, color=GREEN_C),
            _label("readout d1 · x = 1", size=22, color=GREEN_C),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        sparse_lines.move_to(RIGHT * 2.25 + DOWN * 0.05)
        self.play(FadeIn(active_one), FadeIn(sparse_lines))
        self.wait(0.5)

        direction_one = active_one.get_end() - origin
        direction_two = active_two.get_end() - origin
        combined_end = origin + direction_one + direction_two
        sum_arrow = Arrow(origin, combined_end, buff=0, color=WHITE, stroke_width=7)
        cross_term = float(
            np.dot(direction_one, direction_two)
            / (np.linalg.norm(direction_one) * np.linalg.norm(direction_two))
        )
        readout_lines = VGroup(
            _label("Decode feature 1 with d1 · x", size=23, color=INK),
            _label("x = d1 + d2", size=23, color=WHITE),
            _label("d1 · x = 1 + d1 · d2", size=22, color=WHITE),
            _label(f"= 1 + {cross_term:.2f} = {1.0 + cross_term:.2f}", size=25, color=ORANGE),
            _label(f"true f1 = 1  →  error +{cross_term:.2f}", size=22, color=ORANGE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.13)
        readout_lines.scale_to_fit_width(3.75).move_to(RIGHT * 2.25 + DOWN * 0.15)
        self.play(
            FadeIn(active_two),
            GrowArrow(sum_arrow),
            FadeOut(sparse_lines),
            FadeOut(capacity),
            FadeIn(readout_lines),
        )
        self.wait(0.8)

        orthogonal_arrow = Arrow(
            origin,
            origin + UP * 1.55,
            buff=0,
            color=WHITE,
            stroke_width=9,
        )
        control_lines = VGroup(
            _label("Orthogonal control", size=25, color=GREEN_C),
            _label("replace d2 by d⊥", size=22, color=WHITE),
            _label("d1 · (d1 + d⊥) = 1 + 0", size=22, color=WHITE),
            _label("readout = 1.00  →  error 0", size=23, color=GREEN_C),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        control_lines.scale_to_fit_width(3.75).move_to(RIGHT * 2.25 + DOWN * 0.05)
        self.play(
            FadeOut(active_two),
            FadeOut(sum_arrow),
            FadeOut(readout_lines),
            FadeIn(orthogonal_arrow),
            FadeIn(control_lines),
        )
        self.wait(0.7)
        conclusion = _label(
            "Interference is decoder cross-talk (d1 · d2), not vector addition alone.",
            size=21,
            color=YELLOW,
        ).to_edge(DOWN)
        self.play(Write(conclusion))
        self.wait(1.0)


class JohnsonLindenstraussDistances(Scene):
    """Distance matrices before and after one random projection."""

    def _matrix(
        self,
        distances: np.ndarray,
        center: np.ndarray,
        label: str,
        *,
        scale_maximum: float,
    ) -> VGroup:
        count = distances.shape[0]
        cells = VGroup()
        size = 0.42
        for row in range(count):
            for column in range(count):
                intensity = float(distances[row, column] / scale_maximum)
                color = BLUE_C if row != column else "#263247"
                cell = Rectangle(
                    width=size,
                    height=size,
                    stroke_width=0.7,
                    stroke_color=BACKGROUND,
                    fill_color=color,
                    fill_opacity=0.12 + 0.82 * intensity,
                )
                cell.move_to(
                    center
                    + RIGHT * (column - (count - 1) / 2) * size
                    + DOWN * (row - (count - 1) / 2) * size
                )
                cells.add(cell)
        caption = _label(label, size=22).next_to(cells, DOWN, buff=0.2)
        return VGroup(cells, caption)

    def construct(self) -> None:
        self.camera.background_color = BACKGROUND
        rng = np.random.default_rng(19)
        points = rng.normal(size=(7, 64))
        projection = rng.normal(size=(64, 12)) / np.sqrt(12)
        projected = points @ projection
        original_distance = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
        projected_distance = np.linalg.norm(
            projected[:, None] - projected[None, :], axis=-1
        )
        upper = np.triu_indices(points.shape[0], 1)
        max_distortion = float(
            np.max(
                np.abs(
                    projected_distance[upper] / original_distance[upper]
                    - 1.0
                )
            )
        )

        title = _label("Johnson–Lindenstrauss: preserve a finite geometry", size=34).to_edge(UP)
        subtitle = _label(
            "Distance matrices expose 64D geometry on one shared color scale.",
            size=21,
            color=MUTED,
        ).next_to(title, DOWN, buff=0.18)
        self.play(Write(title), FadeIn(subtitle))

        shared_maximum = float(max(original_distance.max(), projected_distance.max()))
        left_matrix = self._matrix(
            original_distance,
            LEFT * 3.2 + DOWN * 0.2,
            "before: D = 64",
            scale_maximum=shared_maximum,
        )
        right_matrix = self._matrix(
            projected_distance,
            RIGHT * 3.2 + DOWN * 0.2,
            "after: k = 12",
            scale_maximum=shared_maximum,
        )
        self.play(FadeIn(left_matrix))

        projector = Rectangle(
            width=1.5,
            height=2.2,
            fill_color=PURPLE_C,
            fill_opacity=0.25,
            stroke_color=PURPLE_C,
        )
        projector_label = _label("random\nprojection", size=20, color=PURPLE_C).move_to(projector)
        map_group = VGroup(projector, projector_label)
        map_arrow = Arrow(
            left_matrix.get_right() + RIGHT * 0.15,
            right_matrix.get_left() + LEFT * 0.15,
            color=PURPLE_C,
            buff=0.1,
        )
        self.play(FadeIn(map_group), GrowArrow(map_arrow))
        self.play(FadeIn(right_matrix))

        equation = _label("k grows like log(n) / ε²", size=28, color=YELLOW).to_edge(DOWN)
        sample = _label(
            f"one sampled map: maximum relative pair distortion = {max_distortion:.0%}",
            size=20,
            color=MUTED,
        ).next_to(equation, UP, buff=0.18)
        self.play(Write(equation), FadeIn(sample))

        warning = _label(
            "A theorem about finite-set distances — not a theory of learned features",
            size=20,
            color=ORANGE,
        ).next_to(subtitle, DOWN, buff=0.16)
        self.play(FadeIn(warning))
        self.wait(1.0)


if __name__ == "__main__":
    config.pixel_width = 854
    config.pixel_height = 480
    config.frame_rate = 15
