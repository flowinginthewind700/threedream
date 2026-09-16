/**
 * The nav strip every demo page carries.
 *
 * Loaded by its own `<script type="module">` in each page's HTML rather than
 * imported from the page's entry, for two reasons. Five entries calling
 * `mountDemoNav()` is five places to forget it, and the sixth page would be the
 * one nobody navigates to; a script tag in the HTML is one line next to the line
 * that already loads the page. It also keeps this module out of the entries'
 * import graph, so a nav change cannot break a page's own bootstrap order.
 *
 * Which link is current comes from `location.pathname` (see `pageFromPathname`)
 * rather than from an argument, so no page can claim to be another one.
 *
 * The links are built at runtime, which means the built HTML carries no nav
 * markup and no href for `tests/build_base.test.ts` to inspect. That is why the
 * hrefs are asserted in the browser instead, against the built and served site:
 * `e2e/demo.spec.ts` loads all five pages and requires every href to be
 * relative. A domain-absolute `/particles.html` would work under `npm run dev`
 * and 404 on Pages, and no Node-side check of this file would catch it.
 *
 * DOM-only, like `src/render/`: the pure half (the page list and the URL logic)
 * lives in `demo/pages.ts` and is what `tests/demo_nav.test.ts` exercises under
 * Node. The module-scope mount is guarded so importing this file by accident in
 * a Node context is a no-op rather than a `document is not defined`.
 */

import {
  Bot,
  Fingerprint,
  Grid3x3,
  Share2,
  Waypoints,
  createElement,
  type IconNode,
} from 'lucide';

import { navLinks, pageFromPathname } from './pages.js';

/** Name to node, the same shape each page's own icon map uses. */
const ICONS: Readonly<Record<string, IconNode>> = {
  bot: Bot,
  fingerprint: Fingerprint,
  'grid-3x3': Grid3x3,
  'share-2': Share2,
  waypoints: Waypoints,
};

/**
 * Build the nav and attach it to the page's top bar.
 *
 * Returns the element so a caller (or a spec) can hold it, and `null` when there
 * is nowhere to put it. Appended last into `.topbar` and moved by CSS `order`,
 * which is what lets it sit between the brand and the page's own headline
 * control without the HTML of five pages having to reserve a slot for it.
 *
 * Idempotent: a second call returns the existing strip instead of adding one.
 */
export function mountDemoNav(doc: Document): HTMLElement | null {
  const existing = doc.querySelector<HTMLElement>('.pagenav');
  if (existing) return existing;

  const host = doc.querySelector('.topbar') ?? doc.querySelector('.app');
  if (!host) return null;

  const nav = doc.createElement('nav');
  nav.className = 'pagenav';
  nav.setAttribute('aria-label', 'Demos');

  const current = pageFromPathname(doc.location?.pathname ?? '');
  for (const link of navLinks(current)) {
    const a = doc.createElement('a');
    a.href = link.href;
    a.title = link.title;
    a.textContent = link.label;
    // `aria-current="page"` is the machine-readable half of "you are here"; the
    // copper icon and the panel background are the human half. A link styled as
    // current but not announced as such is decoration.
    if (link.current) a.setAttribute('aria-current', 'page');

    const icon = ICONS[link.icon];
    if (icon) {
      const span = doc.createElement('span');
      span.className = 'pagenav-icon';
      span.setAttribute('aria-hidden', 'true');
      span.replaceChildren(createElement(icon));
      a.prepend(span);
    }

    nav.append(a);
  }

  host.append(nav);
  return nav;
}

if (typeof document !== 'undefined') mountDemoNav(document);
