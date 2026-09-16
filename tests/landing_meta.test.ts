/**
 * The contract behind what a visitor sees before the site loads: the tab icon
 * and the share card.
 *
 * Neither is checked by anything else in the suite. A missing favicon is a blank
 * tab among forty open ones; a dead `og:image` is a grey box in every chat the
 * link gets pasted into, and both fail *outside* the repo, where no test run
 * ever looks. So the assertions here are about files and strings that have to
 * line up:
 *
 * - every page links the icon, and the icon exists in `demo/public/`, which is
 *   what the build copies to the site root;
 * - `og:title` and `og:description` are this page's own `<title>` and
 *   description, because a share card that paraphrases the page is a card that
 *   drifts from it the first time either is edited;
 * - `og:url` is the page's deployed address, and `og:image` names a file under
 *   `demo/public/`, so a card can never point at an image the deploy does not
 *   serve. `scripts/capture_shots.mjs` is what mints those images.
 *
 * The URLs are absolute on purpose, which is the opposite of the rule
 * `tests/build_base.test.ts` enforces for `src`/`href`: a scraper resolving
 * `og:image` has no document base, so relative is wrong there and correct here.
 */

import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

import { DEMO_PAGES } from '../demo/pages.js';

const ROOT = resolve(import.meta.dirname, '..');
const PUBLIC = resolve(ROOT, 'demo/public');
const SITE = 'https://flowinginthewind700.github.io/threedream/';

function pageHtml(file: string): string {
  return readFileSync(resolve(ROOT, 'demo', file), 'utf8');
}

/** The `content` of a meta tag, which the formatter is free to wrap. */
function metaContent(html: string, attr: string, value: string): string | null {
  const at = html.indexOf(`${attr}="${value}"`);
  if (at === -1) return null;
  const from = html.indexOf('content="', at);
  // A later tag's content must not satisfy an earlier tag's absence.
  if (from === -1 || html.slice(at, from).includes('>')) return null;
  const start = from + 'content="'.length;
  const end = html.indexOf('"', start);
  return end === -1 ? null : html.slice(start, end);
}

function tagText(html: string, tag: string): string | null {
  const m = html.match(new RegExp(`<${tag}>([^<]*)</${tag}>`));
  return m ? m[1].trim() : null;
}

describe('every demo page carries a favicon', () => {
  it('ships the icon from demo/public/', () => {
    const icon = resolve(PUBLIC, 'favicon.svg');
    expect(existsSync(icon), 'demo/public/favicon.svg is what the build copies to /').toBe(true);
    expect(readFileSync(icon, 'utf8').trimStart().startsWith('<svg')).toBe(true);
  });

  for (const page of DEMO_PAGES) {
    it(`${page.file} links it`, () => {
      expect(pageHtml(page.file)).toContain(
        '<link rel="icon" type="image/svg+xml" href="./favicon.svg" />',
      );
    });
  }
});

describe('every demo page carries share metadata', () => {
  for (const page of DEMO_PAGES) {
    const html = pageHtml(page.file);
    const deployed = page.file === 'index.html' ? '' : page.file;

    it(`${page.file} repeats its own title and description in the card`, () => {
      expect(metaContent(html, 'property', 'og:title')).toBe(tagText(html, 'title'));
      expect(metaContent(html, 'property', 'og:description')).toBe(
        metaContent(html, 'name', 'description'),
      );
      expect(metaContent(html, 'name', 'twitter:card')).toBe('summary_large_image');
      expect(metaContent(html, 'property', 'og:type')).toBe('website');
    });

    it(`${page.file} points the card at its own deployed address`, () => {
      expect(metaContent(html, 'property', 'og:url')).toBe(`${SITE}${deployed}`);
    });

    it(`${page.file} points the card at an image the deploy serves`, () => {
      const image = metaContent(html, 'property', 'og:image');
      expect(image, 'og:image missing').not.toBeNull();
      expect(image!.startsWith(SITE), `og:image must be absolute: ${image}`).toBe(true);
      const tail = image!.slice(SITE.length);
      expect(
        existsSync(resolve(PUBLIC, tail)),
        `${tail} is not in demo/public/, so the card would 404`,
      ).toBe(true);
    });
  }
});
