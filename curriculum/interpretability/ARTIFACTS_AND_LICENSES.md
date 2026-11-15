# Artifact and licensing register

## Course-created material

`pyproject.toml` declares Apache-2.0, but the repository currently has no root
license text. That absence is recorded rather than hidden. Downstream
redistributors should resolve it before publishing a standalone course package.

The superposition and Johnson–Lindenstrauss animations are created in this
repository from animations/scenes.py with Manim Community 0.20.1 and FFmpeg.
The committed MP4, PNG poster, and WebVTT caption files contain no external
image, music, voice, checkpoint, or dataset asset. A manifest binds source and
output hashes, durations, and sizes. Rasterized text comes from the local
authoring stack; standalone redistributors should still review their chosen
render environment and Manim/FFmpeg licenses.

## External and historical artifacts

| Artifact class | Bundled? | Course use | Redistribution status |
| --- | --- | --- | --- |
| Raw BT4 (`61e43e98d2c4cb747498c6bc13a5bbb3b07e985d3f5d87f0351a4db855fa0651`) / terminal Hero (`665ac91ad37e14a6c8810068700dba021e39e9c0fe145c8bf620882bb5a71692`) checkpoints | No | local, user-supplied source reruns | not granted by this course |
| Published LoRSA weights | No | local, user-supplied advanced reruns | no established redistribution license in the retained record |
| LC0 tar / converted corpus | No | Hero training, outside course runtime | not granted by this course |
| ChessBench / puzzle arrays | No new copy | existing local artifacts after checksum validation | follow upstream license and manifest |
| Derived Raw/Hero metrics | compact subset only | snapshot figures and tables | generated here; source hashes retained |
| Real Raw/Hero activation gallery (`49f6f352…a5860`, 30,830,817 bytes) | **No; local and gitignored** | selectable residual rows, policy lenses, trained-probe predictions, VJPs, and spectra | no checkpoint/bulk corpus included; do not stage, push, or publish the tensor or its rendered real-derived views until a human clears the source-model and source-position terms |
| Position FEN strings | minimal examples | board display and attribution teaching | source ID/provenance retained; no bulk puzzle copy |

The snapshot contains no tensor checkpoint, sparse/probe weight, large
prediction array, or bulk game corpus. Every source record includes path,
SHA-256, byte size, schema, role, license note, and allowed use. Source mode
refuses identity drift.

Before pushing any real-derived tensor/view to this public repository, or
publishing a standalone course, a human must verify:

1. a root license file exists and matches project metadata;
2. every source manifest names an upstream license;
3. no external weight or dataset is embedded in HTML exports;
4. screenshots, fonts, and icons have redistribution permission; and
5. animation source/output hashes and caption transcripts are current; and
6. static exports expose the same provenance badge as live notebooks; and
7. the derived activation tensor pack is either approved under the source
   model/position terms or remains local and gitignored; and
8. any rendered activation table, map, curve, or static HTML page is separately
   approved as a derived view before it is staged or pushed.
