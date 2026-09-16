/**
 * The demo nav, and the page list it is built from.
 *
 * Five pages that do not link to each other are four pages nobody finds: the site
 * root is the trainer, and a visitor who landed there had no way to learn the
 * particle or soft-body pages existed short of reading the README. The nav is the
 * fix, and these tests pin the two ways it can quietly stop being one.
 *
 * 1. Drift from the build. `vite.config.ts` derives its rollup inputs from the
 *    same array the nav renders, so a page can be linked without being deployed
 *    (a 404 that reads like a typo) or deployed without being linked (invisible)
 *    only if that derivation breaks. The check loads the real config through
 *    vite's own loader rather than grepping the file, so what is asserted here is
 *    what CI builds.
 * 2. The URL logic. Which link is current is read from `location.pathname` in the
 *    browser, where the site sits under a subpath on Pages and at the root
 *    locally. `pageFromPathname` is pure, so both shapes are covered here instead
 *    of one of them being covered by luck.
 *
 * The DOM half (`demo/nav.ts`) is not exercised here for the same reason
 * `src/render/` is not: vitest runs in a Node environment. `e2e/demo.spec.ts`
 * loads all five built pages and asserts the strip is present, marks the right
 * link, and uses relative hrefs -- the last of which is only meaningful against a
 * built site served from its subpath, which is exactly what the e2e run does.
 */

import { readdirSync } from 'node:fs';
import { basename, resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

import { DEMO_PAGES, navLinks, pageFromPathname } from '../demo/pages.js';

const ROOT = resolve(import.meta.dirname, '..');
const DEMO = resolve(ROOT, 'demo');

/** Every page the build could ship: an HTML file sitting in `demo/`. */
function htmlFilesInDemo(): string[] {
  return readdirSync(DEMO)
    .filter((f) => f.endsWith('.html'))
    .sort();
}

describe('demo/pages.ts is the one page list', () => {
  it('lists every HTML file in demo/, and nothing that is not there', () => {
    // Both directions matter, and they fail differently. An HTML file in demo/
    // that is not in the list is neither built nor linked, because
    // `vite.config.ts` reads this same array: such a page works on the dev server
    // and exists nowhere else. A list entry with no file behind it is the louder
    // half -- a nav link that 404s on every deployment, including the one being
    // demoed, and nobody clicks all five links during a demo.
    expect(DEMO_PAGES.map((p) => p.file).sort()).toEqual(htmlFilesInDemo());
  });

  it('gives each page a unique, non-empty label, icon and tooltip', () => {
    const labels = DEMO_PAGES.map((p) => p.label);
    expect(new Set(labels).size, 'labels are what the nav shows').toBe(labels.length);
    expect(new Set(DEMO_PAGES.map((p) => p.file)).size).toBe(DEMO_PAGES.length);
    for (const page of DEMO_PAGES) {
      expect(page.label.trim().length, `${page.file} label`).toBeGreaterThan(0);
      expect(page.icon.trim().length, `${page.file} icon`).toBeGreaterThan(0);
      // The tooltip is the only place a page says what it measures, so an empty
      // one is a label with no explanation behind it.
      expect(page.title.trim().length, `${page.file} title`).toBeGreaterThan(0);
    }
  });

  it('puts the trainer first, because it is the site root', () => {
    expect(DEMO_PAGES[0]!.file).toBe('index.html');
  });
});

describe('the build inputs are derived from the same list', () => {
  it('vite builds exactly the pages the nav links', async () => {
    const { loadConfigFromFile } = await import('vite');
    const loaded = await loadConfigFromFile(
      { command: 'build', mode: 'production' },
      resolve(ROOT, 'vite.config.ts'),
    );
    expect(loaded, 'vite.config.ts should load').not.toBeNull();

    const rollup = loaded!.config.build?.rollupOptions as
      | { input?: Record<string, string> | string | string[] }
      | undefined;
    const input = rollup?.input;
    expect(typeof input, 'inputs should be a keyed record').toBe('object');
    const entries = Object.entries(input as Record<string, string>);

    expect(entries.map(([, file]) => basename(file)).sort()).toEqual(
      DEMO_PAGES.map((p) => p.file).sort(),
    );
    // The keys are the emitted file stems, so `index` has to stay `index` rather
    // than become `index.html`: a key of `index.html` would emit
    // `dist/index.html.html` and take the site root with it.
    expect(entries.map(([key]) => key).sort()).toEqual(
      DEMO_PAGES.map((p) => p.file.replace(/\.html$/, '')).sort(),
    );
  }, 60_000);
});

describe('pageFromPathname finds the current page under either base', () => {
  it.each([
    ['/threedream/', 'index.html'],
    ['/', 'index.html'],
    ['', 'index.html'],
    ['/threedream/index.html', 'index.html'],
    ['/index.html', 'index.html'],
    ['/threedream/soft.html', 'soft.html'],
    ['/soft.html', 'soft.html'],
    ['/threedream/particles.html', 'particles.html'],
    ['/threedream/physics-check.html', 'physics-check.html'],
    ['/threedream/shared-device.html', 'shared-device.html'],
  ])('%s is %s', (pathname, expected) => {
    expect(pageFromPathname(pathname)).toBe(expected);
  });

  it('accepts an extensionless page name', () => {
    // A dev-server habit and a plausible future route shape; matching it costs
    // one replace and saves a nav that highlights nothing.
    expect(pageFromPathname('/threedream/soft')).toBe('soft.html');
  });

  it('treats a directory URL as the index even when the directory is named like a page', () => {
    // The subpath is the repo name. Were the repo called `soft`, `/soft/` must
    // still be the trainer and not `soft.html`, which is why the trailing slash
    // is decided before the last segment is matched.
    expect(pageFromPathname('/soft/')).toBe('index.html');
  });

  it('returns null for a path that is not one of ours, rather than guessing', () => {
    // Guessing would mark a link current on a page that is not it, which is worse
    // than marking none: the strip still renders, so navigation still works.
    expect(pageFromPathname('/threedream/nope.html')).toBeNull();
    expect(pageFromPathname('/threedream')).toBeNull();
    expect(pageFromPathname('/assets/soft-Bvt8qKhY.js')).toBeNull();
  });
});

describe('navLinks renders one link per page', () => {
  it('keeps the list order and carries the label, icon and tooltip through', () => {
    const links = navLinks('soft.html');
    expect(links.map((l) => l.file)).toEqual(DEMO_PAGES.map((p) => p.file));
    expect(links.map((l) => l.label)).toEqual(DEMO_PAGES.map((p) => p.label));
    expect(links.map((l) => l.icon)).toEqual(DEMO_PAGES.map((p) => p.icon));
    expect(links.map((l) => l.title)).toEqual(DEMO_PAGES.map((p) => p.title));
  });

  it('marks exactly the current page, and nothing when the page is unknown', () => {
    for (const page of DEMO_PAGES) {
      const current = navLinks(page.file).filter((l) => l.current);
      expect(current.map((l) => l.file), `current for ${page.file}`).toEqual([page.file]);
    }
    expect(navLinks(null).filter((l) => l.current)).toEqual([]);
  });

  it('emits relative hrefs, so one bundle works at / and under /threedream/', () => {
    // The failure this exists to catch: `/particles.html` is correct on the local
    // dev server and 404s on Pages, where the site lives under a subpath. The
    // page still loads and its assets still resolve, so nothing else goes red.
    for (const link of navLinks('index.html')) {
      expect(link.href.startsWith('/'), `domain-absolute href: ${link.href}`).toBe(false);
      expect(link.href.startsWith('./'), `not explicitly relative: ${link.href}`).toBe(true);
    }
  });

  it('links the index as the directory URL, not as index.html', () => {
    // Two URLs for one page is two history entries and two things to keep
    // canonical; `./` is what Pages and the preview server both serve the root at.
    expect(navLinks(null).find((l) => l.file === 'index.html')!.href).toBe('./');
    expect(navLinks(null).find((l) => l.file === 'soft.html')!.href).toBe('./soft.html');
  });
});
