"""Generate synchronized Markdown papers from paired, reviewed source sections."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def table(path, headers, columns):
    with (ROOT/path).open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    out = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---']*len(headers))+' |']
    for row in rows:
        out.append('| '+' | '.join(row.get(col, '') or '-' for col in columns)+' |')
    return '\n'.join(out)

def render(language, manifest):
    source = json.loads((ROOT/'paper/data/sections.json').read_text(encoding='utf-8'))
    out = [f"# {source['title'][language]}",
           f"Living Paper | {manifest['version']} | Research revision {manifest['research_revision']}",
           '[한국어](paper_ko.md) | [English](paper_en.md)',
           source['notice'][language]]
    for section in source['sections']:
        assert section.get('ko') and section.get('en'), section['id']
        out += [f"<!-- section:{section['id']} -->", '## '+section['heading'][language], section[language]]
        if section.get('table') == 'metrics':
            headers = ['버전/방법','TEST Exact (%)','MAE','최대 오차','근거'] if language=='ko' else ['Version / method','TEST Exact (%)','MAE','Max error','Evidence']
            out.append(table('paper/data/version_metrics.csv',headers,['version','exact_percent','mae','max_error','evidence']))
        if section.get('table') == 'failures':
            headers = ['실험','TEST Exact (%)','관찰','출처'] if language=='ko' else ['Experiment','TEST Exact (%)','Observation','Source']
            out.append(table('paper/data/failed_experiments.csv',headers,['experiment','exact_percent','observation_'+language,'source']))
        for extra in section.get('tables', []):
            out += ['### '+extra['title'][language],table(extra['path'],extra['headers'][language],extra['columns'])]
        for figure_id in section.get('figures', []):
            fig = next(f for f in manifest['figures'] if f['id']==figure_id)
            if fig['path']:
                assert (ROOT/'paper'/fig['path']).is_file(), fig['path']
                if fig.get('sha256'):
                    assert hashlib.sha256((ROOT/'paper'/fig['path']).read_bytes()).hexdigest()==fig['sha256'], 'Figure evidence was modified'
                out.append(f"![{fig['caption'][language]}]({fig['path']})\n\n{fig['caption'][language]}")
            else:
                prefix = '그림 자리표시자' if language=='ko' else 'Figure placeholder'
                out.append(f"> **{prefix} {fig['number']}. {fig['caption'][language]}**\n> {fig['missing_note'][language]}")
    return '\n\n'.join(out)+'\n'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--snapshot', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((ROOT/'paper_manifest.json').read_text(encoding='utf-8'))
    version = (ROOT/'VERSION.txt').read_text().strip()
    assert manifest['version']==version, 'Paper release version differs from application version'
    assert set(manifest['snapshots'].get(version, {}))=={'ko','en'} or not args.check, 'Both release snapshots are required'
    assert manifest['release_kind'] in ('research','presentation_patch','maintenance_patch')
    if manifest['research_changed']:
        assert manifest['evidence_updates'], 'A research release needs traceable new evidence'
    generated = {}
    for language in ('ko','en'):
        content = render(language, manifest)
        path = ROOT/f'paper/paper_{language}.md'
        generated[language] = digest(content)
        if args.check:
            assert path.read_text(encoding='utf-8')==content, f'Stale {language} paper; run tools/build_paper.py'
        else:
            path.write_text(content, encoding='utf-8')
        if args.snapshot:
            snap = ROOT/f'paper/snapshots/paper_{version}_{language}.md'
            # Relative image links must work from the deeper snapshot folder too.
            content = content.replace('](figures/', '](../figures/').replace('](appendix/', '](../appendix/').replace('](data/', '](../data/')
            content = content.replace('](paper_ko.md)', f'](paper_{version}_ko.md)').replace('](paper_en.md)', f'](paper_{version}_en.md)')
            if snap.exists():
                assert snap.read_text(encoding='utf-8')==content, 'Published snapshots are immutable'
            else:
                snap.parent.mkdir(parents=True, exist_ok=True)
                snap.write_text(content, encoding='utf-8')
            manifest['snapshots'][version][language] = {'path':str(snap.relative_to(ROOT)).replace('\\','/'), 'sha256':digest(content)}
    if args.check:
        assert generated == manifest['paper_sha256'], 'Manifest paper hashes are stale'
        for release in manifest['snapshots'].values():
            for snapshot in release.values():
                assert digest((ROOT/snapshot['path']).read_text(encoding='utf-8'))==snapshot['sha256'], 'Snapshot modified'
        mirror = json.loads((ROOT/'paper/data/paper_manifest.json').read_text(encoding='utf-8'))
        assert mirror==manifest, 'Manifest mirror differs'
        for artifact in manifest.get('external_evidence', {}).get('artifacts', []):
            assert hashlib.sha256((ROOT/artifact['path']).read_bytes()).hexdigest()==artifact['sha256'], 'Frozen external evidence was modified'
        print('Living Paper: bilingual sections, tables, figures, hashes and immutable snapshots PASS')
    else:
        manifest['paper_sha256'] = generated
        for name in ('paper_manifest.json','paper/data/paper_manifest.json'):
            (ROOT/name).write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n', encoding='utf-8')

if __name__=='__main__':
    main()
