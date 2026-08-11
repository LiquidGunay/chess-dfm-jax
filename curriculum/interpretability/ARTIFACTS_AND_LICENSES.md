# Artifact and licensing register

## Course-created material

`pyproject.toml` declares Apache-2.0, but the repository currently has no root
license text. That absence is recorded rather than hidden. Downstream
redistributors should resolve it before publishing a standalone course package.

## External and historical artifacts

| Artifact class | Bundled? | Course use | Redistribution status |
| --- | --- | --- | --- |
| Raw BT4 (`61e43e98d2c4cb747498c6bc13a5bbb3b07e985d3f5d87f0351a4db855fa0651`) / terminal Hero (`665ac91ad37e14a6c8810068700dba021e39e9c0fe145c8bf620882bb5a71692`) checkpoints | No | local, user-supplied source reruns | not granted by this course |
| Published LoRSA weights | No | local, user-supplied advanced reruns | no established redistribution license in the retained record |
| LC0 tar / converted corpus | No | Hero training, outside course runtime | not granted by this course |
| ChessBench / puzzle arrays | No new copy | existing local artifacts after checksum validation | follow upstream license and manifest |
| Derived Raw/Hero metrics | compact subset only | snapshot figures and tables | generated here; source hashes retained |
| Position FEN strings | minimal examples | board display and attribution teaching | source ID/provenance retained; no bulk puzzle copy |

The snapshot contains no tensor checkpoint, sparse/probe weight, large
prediction array, or bulk game corpus. Every source record includes path,
SHA-256, byte size, schema, role, license note, and allowed use. Source mode
refuses identity drift.

Before public standalone release, a human must verify:

1. a root license file exists and matches project metadata;
2. every source manifest names an upstream license;
3. no external weight or dataset is embedded in HTML exports;
4. screenshots, fonts, and icons have redistribution permission; and
5. static exports expose the same provenance badge as live notebooks.
