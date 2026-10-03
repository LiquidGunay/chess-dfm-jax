# DFM experiment workspace

Start with [the registry](registry.json) for status and dependencies, then
[the research program](../../docs/dfm_research_program.md) for scientific gates.
Execution instructions for the current run are in
[the Modal continuation runbook](../../docs/modal_proposal_a_continuation.md).

The experiment branch is `research/modal-experiments`. It is isolated from
the uncommitted interpretability-course work on
`research/interpretability-blog-lenses`. Historical experiment documents and
artifact paths remain valid; use this index for new work instead of moving
old files and breaking their source bindings.

Each new experiment gets a stable ID, one question, a frozen specification,
an explicit parent/baseline, and a decision. Before launch, record the data
and source hashes, seed, model/config identity, training and evaluation
budgets, checkpoint retention, and acceptance criteria. After launch, record
the Modal App/function IDs and result label here. Keep compact evidence in
`research/analysis/<experiment-id>/`; model and optimizer states stay remote.
An idea in the backlog is not a scheduled GPU run.

After a completed checkpoint and a frozen opening allocation, the Modal arena
accepts either a completed Hero result label or `--opponent-label raw-bt4`.
The latter selects the immutable native raw policy in the same Torch runtime.
Use candidate pass counts 1 and 8 as distinct evaluations; raw BT4 always uses
its single policy evaluation. Neither development mode by itself constitutes
a confirmation or promotion claim.

Use separate commits for launcher/accounting changes, experimental plans,
and compact run receipts. Keep model/objective changes in separate experiment
branches once the parent checkpoint and evaluation protocol are frozen.

## Full-history input policy

As of EV-005, native-Torch raw BT4 defaults to full LC0 history. The explicit
`--raw-bt4-plane-history-mode current_only_as_preprocessed` option is only a
historical/attribution control; its mode is bound into result and resume
identities. EV-001/003/004 raw anchors used current-only inputs and do not
establish strength against normally configured BT4.

New training defaults to `--data-format lc0_sequential`; select an audited
full-history dataset with `--data-root`. Both the local trainer and Modal
launcher reject trajectory-v3 unless `--allow-legacy-current-only` is given
for an explicitly requested historical reproduction. A first-batch audit
checks that valid future targets actually retain history. Old current-only
milestone pools cannot be used for full-history training: use sequential
validation and the history-aware arena. This changes no retained weights.

## Storage policy

Inventory all chess Volumes before a large run. Start a cleanup review when
current plus projected storage reaches 75% of the included 1,024 GiB
(768 GiB). Pause new checkpoint-heavy launches at 85% (870.4 GiB) until there
is sufficient room. These are workflow thresholds, not provider quotas.

Preserve raw input identities, selected baseline/final models, active exact
recovery states, the epoch's update-6219 continuation checkpoint, and compact
evidence. First consider obsolete compiler cache, then superseded intermediate
states and failed-attempt duplicates after verifying that no pending run or
future comparison references them. Promote selected artifacts and verify
their checksums before any deletion. Cleanup records must name exact paths,
sizes, retained replacements, and recovery locations; never delete a Volume
wholesale to make room. No cleanup is needed at the September 29 baseline of
233.53 GiB.
