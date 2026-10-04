"""Presentation-only Korean/English translations; research data stays canonical."""
from __future__ import annotations
import json
import re
from functools import lru_cache
from pathlib import Path

CATALOG = json.loads((Path(__file__).parent / 'locales' / 'ko.json').read_text(encoding='utf-8'))
_language = 'ko'


def set_language(language):
    global _language
    _language = language if language in ('ko', 'en') else 'ko'
    return _language


def get_language():
    return _language


@lru_cache(maxsize=2)
def _translator(language):
    # Longest fragments first: filenames and machine identifiers are never keys.
    pairs = sorted(CATALOG.items(), key=lambda pair: len(pair[0]), reverse=True)
    if language == 'en':
        pairs = sorted(((v, k) for k, v in pairs), key=lambda pair: len(pair[0]), reverse=True)
    # Split once against original text to avoid cascading translations.
    keys = [k for k, v in pairs if len(k) > 4]
    lookup = dict(pairs)
    pattern = '|'.join(re.escape(k) for k in keys)
    return lookup, re.compile(pattern)


_PROTECTED = re.compile(r'https?://\S+|[A-Za-z]:[\\/][^\n]+|[\w./\\-]+\.(?:csv|json|zip|txt|jpg|png|pdf|md|py)\b')


def tr(value):
    """Translate presentation text while leaving paths/filenames intact."""
    text = str(value)
    lookup, pattern = _translator(_language)
    if text in lookup:
        return lookup[text]
    def translate(part):
        return pattern.sub(lambda m: lookup[m.group()], part)
    result = []
    start = 0
    for match in _PROTECTED.finditer(text):
        result.extend((translate(text[start:match.start()]), match.group()))
        start = match.end()
    result.append(translate(text[start:]))
    return ''.join(result)


def refresh_tree(root):
    """Refresh existing display widgets in place without rebuilding input forms."""
    import tkinter as tk
    from tkinter import ttk
    def visit(widget):
        if isinstance(widget, (tk.Tk, tk.Toplevel)):
            widget.title(tr(widget.title()))
        if 'text' in widget.keys():
            widget.configure(text=tr(widget.cget('text')))
        if isinstance(widget, tk.Text) and str(widget.cget('state')) == 'disabled':
            content = widget.get('1.0', 'end-1c')
            widget.configure(state='normal')
            widget.delete('1.0', 'end')
            widget.insert('1.0', tr(content))
            widget.configure(state='disabled')
        if isinstance(widget, ttk.Treeview):
            for column in widget.cget('columns'):
                widget.heading(column, text=tr(widget.heading(column, 'text')))
        # Only label display variables; Entry/Combobox data is untouched.
        if isinstance(widget, (tk.Label, ttk.Label)) and widget.cget('textvariable'):
            name = widget.cget('textvariable')
            widget.setvar(name, tr(widget.getvar(name)))
        for child in widget.winfo_children():
            visit(child)
    visit(root)
