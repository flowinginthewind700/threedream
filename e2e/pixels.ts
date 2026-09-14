/**
 * Pixel readback, shared by every spec that asserts a canvas is not blank.
 *
 * This is the only readback path that works on the renderers under test.
 * `render/scene.ts` creates its WebGL context with `preserveDrawingBuffer:
 * false`, so by the time any JS runs the drawing buffer has been cleared for
 * compositing: `canvas.toDataURL()`, `ctx.drawImage(canvas, ...)` and
 * `gl.readPixels()` all return exactly one flat colour even while the page is
 * visibly animating (measured: 1 distinct colour from all three, 1512 from the
 * decoded screenshot, same frame). A WebGPU canvas behaves the same way.
 * Reading the compositor's own capture is what actually contains the pixels.
 *
 * Decoding via an in-page `<img>` keeps this dependency-free: no pngjs, no
 * sharp, just the browser's own PNG decoder.
 */

import type { Page } from '@playwright/test';

/** Count distinct RGB triples in a PNG screenshot. */
export async function distinctColors(page: Page, png: Buffer): Promise<number> {
  return page.evaluate(async (b64: string) => {
    const img = new Image();
    await new Promise<void>((resolve, reject) => {
      img.onload = () => resolve();
      img.onerror = () => reject(new Error('screenshot did not decode'));
      img.src = `data:image/png;base64,${b64}`;
    });
    const off = document.createElement('canvas');
    off.width = img.naturalWidth;
    off.height = img.naturalHeight;
    const ctx = off.getContext('2d');
    if (!ctx) throw new Error('no 2d context');
    ctx.drawImage(img, 0, 0);
    const data = ctx.getImageData(0, 0, off.width, off.height).data;
    const seen = new Set<string>();
    for (let i = 0; i < data.length; i += 4) {
      seen.add(`${data[i]},${data[i + 1]},${data[i + 2]}`);
    }
    return seen.size;
  }, png.toString('base64'));
}
