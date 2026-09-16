/**
 * The site's page list, declared once.
 *
 * Two consumers need it and they must not drift:
 *
 * - `vite.config.ts` builds exactly these as rollup inputs, and a page missing
 *   from that list is not built, not deployed and not testable while still
 *   looking fine on the dev server.
 * - `demo/nav.ts` links exactly these, so a visitor who lands on any one page
 *   can see and reach the other four. Without that the site reads as one demo
 *   with four orphans: the entry point is the trainer, and nothing on it said
 *   the particle or soft-body pages existed.
 *
 * Deriving both from this array makes "the nav links to a page the build did not
 * emit" and "the build emits a page the nav hides" unrepresentable rather than
 * merely unlikely, and `tests/demo_nav.test.ts` pins the derivation by loading
 * the real vite config and comparing its inputs to this list.
 *
 * Pure data and pure functions on purpose: no DOM and no imports, so the config
 * file can consume it under esbuild and a Node-environment unit test can
 * exercise every branch of the URL logic without a browser.
 */

export interface DemoPage {
  /** File name under `demo/`, and the tail of the deployed URL. */
  file: string;
  /** Nav label, matching the row it has in the README's demo table. */
  label: string;
  /** lucide icon name, resolved to a node in `nav.ts`. */
  icon: string;
  /** Tooltip: what the page measures, in one line. */
  title: string;
}

/**
 * Every page the site ships, in the order the nav shows them: the trainer first
 * because it is the entry point and the only page at the site root, then the
 * determinism and shared-device checks, then the two scale layers in milestone
 * order.
 */
export const DEMO_PAGES: readonly DemoPage[] = [
  {
    file: 'index.html',
    label: 'Trainer',
    icon: 'bot',
    title: 'A policy-gradient learner training in the page, on DriveEnv and ReachEnv',
  },
  {
    file: 'physics-check.html',
    label: 'Physics check',
    icon: 'fingerprint',
    title: 'One canonical scene through builtin, wasm and a wasm replay, digests compared',
  },
  {
    file: 'shared-device.html',
    label: 'Shared device',
    icon: 'share-2',
    title: 'One GPUDevice backing three.js rendering and a raw WGSL compute pipeline',
  },
  {
    file: 'particles.html',
    label: 'Particles',
    icon: 'waypoints',
    title: '1k-100k particles: the tier you got, blit or CPU upload, draw calls, hash overflow',
  },
  {
    file: 'soft.html',
    label: 'Soft bodies',
    icon: 'grid-3x3',
    title: 'Cloth / sheets / cube / rope to 20k nodes: islands, colors, dispatches, stretch',
  },
];

/**
 * Which page a pathname is on, or `null` when it is not one of ours.
 *
 * Read from the pathname rather than passed in by each page, because five
 * hand-written "this is me" strings are five chances to mark the wrong link
 * current, and the browser already knows. A trailing slash means a directory
 * URL, which on Pages and on the preview server alike is the index; that case is
 * checked before the last segment is matched, so a repo whose name happened to
 * collide with a page name (`/soft/`) still resolves to the trainer rather than
 * to `soft.html`.
 */
export function pageFromPathname(pathname: string): string | null {
  if (pathname === '' || pathname.endsWith('/')) return 'index.html';
  const last = pathname.slice(pathname.lastIndexOf('/') + 1);
  const file = last.endsWith('.html') ? last : `${last}.html`;
  return DEMO_PAGES.some((page) => page.file === file) ? file : null;
}

export interface DemoNavLink {
  file: string;
  href: string;
  label: string;
  title: string;
  icon: string;
  current: boolean;
}

/**
 * The nav's links for the page named by `current` (`null` marks none).
 *
 * Every href is relative. The same bundle is served from the domain root locally
 * and from `https://<user>.github.io/<repo>/` on Pages, so a domain-absolute
 * `/particles.html` is correct in exactly one of the two and 404s in the other;
 * `./particles.html` resolves against the document URL and is correct in both.
 * That is the same distinction `tests/build_base.test.ts` pins for assets,
 * applied to navigation, where the failure is quieter still: the page loads, the
 * assets load, and only the link is dead.
 *
 * The index links to `./` rather than `./index.html` because the directory URL
 * is the canonical one on Pages, and two URLs for one page is two entries in
 * anyone's history.
 */
export function navLinks(current: string | null): DemoNavLink[] {
  return DEMO_PAGES.map((page) => ({
    file: page.file,
    href: page.file === 'index.html' ? './' : `./${page.file}`,
    label: page.label,
    title: page.title,
    icon: page.icon,
    current: page.file === current,
  }));
}
