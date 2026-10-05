"""Build and verify both PDFs from the release Markdown snapshots."""
from __future__ import annotations
import argparse
import html
import json
import os
from pathlib import Path

import fitz
from markdown_it import MarkdownIt
from PIL import Image, ImageOps, ImageDraw
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, Image as PDFImage

ROOT = Path(__file__).resolve().parents[1]

def fonts():
    candidates = [
        (os.environ.get('PAPER_FONT'), os.environ.get('PAPER_FONT_BOLD')),
        ('/usr/share/fonts/truetype/nanum/NanumGothic.ttf','/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf'),
        ('C:/Windows/Fonts/malgun.ttf','C:/Windows/Fonts/malgunbd.ttf'),
    ]
    for regular,bold in candidates:
        if regular and Path(regular).is_file():
            pdfmetrics.registerFont(TTFont('PaperKO',regular))
            pdfmetrics.registerFont(TTFont('PaperKOBold',bold if bold and Path(bold).is_file() else regular))
            break
    else:
        raise RuntimeError('Install fonts-nanum or set PAPER_FONT to a Korean TrueType font')
    for regular,bold in [('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'),('C:/Windows/Fonts/arial.ttf','C:/Windows/Fonts/arialbd.ttf')]:
        if Path(regular).is_file():
            pdfmetrics.registerFont(TTFont('PaperEN',regular))
            pdfmetrics.registerFont(TTFont('PaperENBold',bold))
            break
    else:
        raise RuntimeError('Install fonts-dejavu-core or provide standard Arial fonts')
    pdfmetrics.registerFontFamily('PaperKO',normal='PaperKO',bold='PaperKOBold',italic='PaperKO',boldItalic='PaperKOBold')
    pdfmetrics.registerFontFamily('PaperEN',normal='PaperEN',bold='PaperENBold',italic='PaperEN',boldItalic='PaperENBold')

def text_markup(text, bold=False):
    # Select fonts by actual glyph availability; fail instead of emitting tofu.
    runs=[];buffer='';previous=None
    for char in text:
        font = 'PaperENBold' if bold else 'PaperEN'
        if ord(char) not in pdfmetrics.getFont(font).face.charToGlyph:
            font = 'PaperKOBold' if bold else 'PaperKO'
        if not char.isspace() and ord(char) not in pdfmetrics.getFont(font).face.charToGlyph:
            raise ValueError(f'Missing glyph U+{ord(char):04X}: {char}')
        if font!=previous and buffer:
            runs.append(f'<font name="{previous}">{html.escape(buffer)}</font>');buffer=''
        previous=font;buffer+=char
    if buffer: runs.append(f'<font name="{previous}">{html.escape(buffer)}</font>')
    return ''.join(runs)

def inline(token, heading=False, base_path=None):
    out=[];bold=heading
    for child in token.children or []:
        if child.type=='strong_open': bold=True
        elif child.type=='strong_close': bold=heading
        elif child.type in ('text','code_inline'): out.append(text_markup(child.content,bold))
        elif child.type in ('softbreak','hardbreak'): out.append(' ')
        elif child.type=='link_open':
            href=child.attrGet('href')
            # PDF links to Markdown evidence remain usable outside the checkout.
            if not href.startswith(('https://','http://')):
                relative=(base_path/href).resolve().relative_to(ROOT).as_posix()
                href='https://github.com/sopo9880/Parking_/blob/main/'+relative
            out.append(f'<a href="{html.escape(href,quote=True)}" color="#2563a6">')
        elif child.type=='link_close': out.append('</a>')
        elif child.type=='image': out.append(text_markup(child.content))
    return ''.join(out)

def build(markdown, target, language, version):
    styles=getSampleStyleSheet()
    base=dict(fontName='PaperKO' if language=='ko' else 'PaperEN',fontSize=9.2,leading=15.2,wordWrap='CJK',spaceAfter=8,allowWidows=0,allowOrphans=0)
    body=ParagraphStyle('PaperBody',**base)
    quote=ParagraphStyle('PaperQuote',parent=body,fontSize=8.5,leading=13,textColor=colors.HexColor('#526174'),leftIndent=10,borderPadding=7,backColor=colors.HexColor('#f2f5f9'))
    headings={n:ParagraphStyle(f'PaperH{n}',parent=body,fontSize={1:19,2:13,3:10.7}[n],leading={1:27,2:20,3:17}[n],spaceBefore=12 if n>1 else 0,spaceAfter=8,keepWithNext=True,textColor=colors.HexColor('#15334d')) for n in (1,2,3)}
    cell=ParagraphStyle('Cell',parent=body,fontSize=7.1,leading=10.5,spaceAfter=0)
    tokens=MarkdownIt('commonmark').enable('table').parse(markdown.read_text(encoding='utf-8'))
    story=[];i=0;quote_depth=0;list_depth=0
    while i<len(tokens):
        token=tokens[i]
        if token.type=='blockquote_open': quote_depth+=1
        elif token.type=='blockquote_close': quote_depth-=1
        elif token.type in ('bullet_list_open','ordered_list_open'): list_depth+=1
        elif token.type in ('bullet_list_close','ordered_list_close'): list_depth-=1
        elif token.type=='heading_open':
            level=min(3,int(token.tag[1]));i+=1
            story.append(Paragraph(inline(tokens[i],True,markdown.parent),headings[level]))
        elif token.type=='paragraph_open':
            i+=1;content=tokens[i]
            children=content.children or []
            if len(children)==1 and children[0].type=='image':
                path=(markdown.parent/children[0].attrGet('src')).resolve()
                img=PDFImage(str(path));scale=min(170*mm/img.imageWidth,95*mm/img.imageHeight)
                img.drawWidth=img.imageWidth*scale;img.drawHeight=img.imageHeight*scale;story.append(img)
            else:
                story.append(Paragraph(('- ' if list_depth else '')+inline(content,base_path=markdown.parent),quote if quote_depth else body))
        elif token.type=='table_open':
            rows=[];row=[];i+=1
            while tokens[i].type!='table_close':
                t=tokens[i]
                if t.type=='tr_open': row=[]
                if t.type=='inline': row.append(Paragraph(inline(t,base_path=markdown.parent),cell))
                if t.type=='tr_close': rows.append(row)
                i+=1
            width=170*mm
            proportions=([.25,.14,.12,.12,.37] if len(rows[0])==5 else
                         [.24,.14,.32,.30] if len(rows[0])==4 else [1/len(rows[0])]*len(rows[0]))
            table=Table(rows,colWidths=[width*p for p in proportions],repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7eef6')),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#7092b1')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f7f9fc')]),('BOTTOMPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6)]))
            # Long evidence tables may span pages; repeat the header and retain headings.
            story += [table,Spacer(1,10)]
        i+=1
    # Keep the short references section together instead of leaving its title stranded.
    last_heading = next((n for n in range(len(story)-1,-1,-1)
                         if isinstance(story[n],Paragraph) and story[n].style.name=='PaperH2'),None)
    if last_heading is not None:
        story[last_heading:] = [KeepTogether(story[last_heading:])]
    def footer(canvas,doc):
        canvas.saveState();canvas.setStrokeColor(colors.HexColor('#c7d4e2'));canvas.line(20*mm,16*mm,190*mm,16*mm)
        canvas.setFont('PaperEN',7.5);canvas.setFillColor(colors.HexColor('#526174'))
        canvas.drawString(20*mm,11*mm,f'Parking Research Agent | {version} | {language.upper()} | Living Paper')
        canvas.drawRightString(190*mm,11*mm,str(doc.page));canvas.restoreState()
    doc=SimpleDocTemplate(str(target),pagesize=A4,rightMargin=20*mm,leftMargin=20*mm,topMargin=19*mm,bottomMargin=23*mm,title='Parking Research Agent Living Paper '+version+' '+language,author='Parking Research Agent project')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)

def verify_and_render(pdf,previews,language):
    previews.mkdir(parents=True,exist_ok=True)
    document=fitz.open(pdf);pages=[];texts=[]
    for i,page in enumerate(document):
        texts.append(page.get_text())
        pix=page.get_pixmap(matrix=fitz.Matrix(1.2,1.2),alpha=False)
        path=previews/f'{language}-{i+1:02}.png';pix.save(path);pages.append(path)
        for block in page.get_text('dict')['blocks']:
            for line in block.get('lines',[]):
                for span in line['spans']:
                    x0,y0,x1,y1=span['bbox']
                    assert x0>=0 and y0>=0 and x1<=page.rect.width+1 and y1<=page.rect.height+1, 'Text outside page'
    text='\n'.join(texts)
    for required in ('85.29','97.06','0.1471','0.0294','CANDIDATE','SAFE_BASELINE'):
        assert required in text, f'PDF missing {required}'
    if 'v16.5.2' in pdf.name or 'v16.5.3' in pdf.name:
        for required in ('165530','95.0770','95.4401','33.9553','1.440953'):
            assert required in text, f'PDF missing external result {required}'
    if 'v16.5.3' in pdf.name:
        for required in ('34.96','38.21','1/4'):
            assert required in text, f'PDF missing window result {required}'
    assert '\ufffd' not in text, 'Replacement glyph found'
    assert len(text)>3000 and len(pages)>1, 'Paper unexpectedly empty'
    if language=='ko': assert '초록' in text and '결론' in text, 'Korean text missing'
    for start in range(0,len(pages),4):
        sheet=Image.new('RGB',(900,1320),'#dce3ec');draw=ImageDraw.Draw(sheet)
        for j,path in enumerate(pages[start:start+4]):
            im=Image.open(path);im.thumbnail((430,620))
            x=15+(j%2)*450;y=25+(j//2)*650
            sheet.paste(im,(x,y));draw.text((x,y-17),f'{language.upper()} page {start+j+1}',fill='black')
        sheet.save(previews/f'contact-{language}-{start//4+1}.png')
    print(f'{pdf.name}: {len(pages)} pages, embedded glyph/text/page checks PASS')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',default='dist');args=parser.parse_args()
    output=ROOT/args.output_dir;output.mkdir(parents=True,exist_ok=True)
    version=(ROOT/'VERSION.txt').read_text().strip();fonts()
    manifest=json.loads((ROOT/'paper_manifest.json').read_text(encoding='utf-8'))
    for language in ('ko','en'):
        markdown=ROOT/manifest['snapshots'][version][language]['path']
        target=output/f'ParkingResearchAgent-{version}-paper-{language}.pdf'
        build(markdown,target,language,version)
        verify_and_render(target,output/'paper-previews',language)

if __name__=='__main__': main()
