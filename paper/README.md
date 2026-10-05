# Living Paper / 함께 성장하는 연구 논문

[한국어 논문](paper_ko.md) | [English paper](paper_en.md) | [연구 부록 / History](appendix/full_experiment_history.md)

The latest papers share one reviewed source: `data/sections.json`, with paired Korean/English headings and paragraphs. Tables come from common CSV files. The root `../paper_manifest.json` is canonical; `data/paper_manifest.json` is a generated mirror. `snapshots/paper_v16.5.2_ko.md` and `_en.md` preserve this release; v16.5.1 snapshots stay immutable. Public CNRPark-EXT overlays and charts now accompany the evidence tables; unavailable private CCTV images remain placeholders.

## Release growth rule / 릴리스별 성장 규칙

1. Review new runs and record their protocol, input hashes, parameters, sample sizes and evidence files. Separate retained-video, independent temporal and external spatial results.
2. Integrate meaningful new methods, results, figures, errors and limitations into their existing research sections. Update both language paragraphs together. Do not append a release diary to the main paper.
3. Update evidence CSVs, figure provenance and the root manifest. Record `research_changed`, `research_revision`, `release_kind`, and `evidence_updates`. Never promote CANDIDATE without independent temporal evidence.
4. Generate both papers with `python tools/build_paper.py`. For a new release add an empty language snapshot entry under `snapshots` and run `python tools/build_paper.py --snapshot`. Already published snapshots cannot be overwritten.
5. Run `python tools/build_paper.py --check` and tests. The release workflow repeats these checks and generates Markdown-based Korean/English PDF assets with embedded Korean fonts.

연구 내용이 없는 patch는 논문 분량을 늘릴 필요가 없다. 버전 표기와 새 스냅샷만 갱신해도 된다. 의미 있는 연구 변경은 본문에 통합하고, 상세 개발 이력과 실패 실험은 부록에 보존한다. 새 결과가 없으면 빈 수치와 미보고 상태를 유지한다.

## PDF path

Install `tools/requirements-paper.txt`, then run `python tools/build_paper_pdf.py --output-dir dist`. CI installs Nanum Korean fonts, checks text/glyphs, renders page previews, and uploads both PDFs, Markdown snapshots, release ZIP and checksums. PDF previews remain workflow artifacts for visual inspection. A failing paper validation or PDF build blocks publication.

## Frozen external baseline / 외부 기준선

Keep `data/external_v1652/` and `figures/v16_5_2/` immutable. `tools/check_external_evidence.py` verifies archive hashes and recomputes overall/camera/weather occupancy and image-count metrics. Record archived-prediction recalculation separately from full inference and selected-image replay. Future external runs store parameters/model hashes before inference, deduplicate both prepared annotations and evaluation input, report skipped images, and generate actual overlays. Never tune this baseline against target labels. Count metrics sum occupied annotated spaces per image, not all or cross-camera unique vehicles.

The v16.5.3 snapshots add actual retained-video cross-window evidence. `tools/check_window_evidence.py` recalculates frozen paired traces. Preserve `data/windows_v1653/` and `figures/v16_5_3/` and earlier snapshots. Window changes are evaluation-only; do not re-label historical DEV as independent hold-out.
