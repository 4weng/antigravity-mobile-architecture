"""Verification tests ensuring web/index.html obeys modern-web-guidance rules."""

from pathlib import Path

WEB_INDEX = Path(__file__).resolve().parent.parent / "web" / "index.html"


def test_web_companion_implements_modern_web_guidance_patterns() -> None:
    html = WEB_INDEX.read_text(encoding="utf-8")

    # dark-mode guide
    assert '<meta name="color-scheme" content="light dark">' in html
    assert "color-scheme: light dark;" in html
    assert "light-dark(" in html

    # light-dismiss-a-dialog guide + Safari fallback
    assert 'closedby="any"' in html
    assert "'closedBy' in HTMLDialogElement.prototype" in html or (
        '"closedBy" in HTMLDialogElement.prototype' in html
    )
    assert "dialog::backdrop" in html

    # declarative-dialog-popover-control guide + conditional polyfills
    assert 'commandfor="approval-dialog"' in html
    assert 'command="show-modal"' in html
    assert 'commandfor="tier-popover"' in html
    assert 'command="toggle-popover"' in html
    assert "popovertarget" not in html
    assert '"commandForElement" in HTMLButtonElement.prototype' in html
    assert 'Object.hasOwn(HTMLElement.prototype, "popover")' in html
    assert r"[popover]:is(:popover-open, .\:popover-open)" in html

    # responsive-table guide
    assert 'role="region"' in html
    assert "container-type: inline-size;" in html
    assert "@container (width < 600px)" in html
    assert "inset-block-start: 0;" in html
    assert "inset-inline-start: 0;" in html
