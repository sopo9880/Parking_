# Figure evidence / 그림 근거

No actual paper-ready or GT-review images are committed at v16.5.1. The Git tree and available local output directories were checked; no image evidence was recovered. Figure placeholders are descriptions, not measured outcomes.

현재 실제 이미지가 없어 본문의 그림은 설명 자리표시자이다. 향후 `output/.../paper_ready/` 또는 `gt_review/` 자료를 검토한 뒤 추가한다.

Add reviewed evidence under `figures/v<release>/`. For every image, update the root `paper_manifest.json` figure record with a relative path, source run, input/frame timestamp, SAFE/CANDIDATE variant, privacy review, and SHA-256. Use matched timestamps and unchanged GT for before/after comparisons. Update both captions together in the manifest. Do not fabricate screenshots or imply a missing result was observed. Keep technical labels and original result filenames canonical.
