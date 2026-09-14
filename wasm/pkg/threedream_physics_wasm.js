/* @ts-self-types="./threedream_physics_wasm.d.ts" */

/**
 * ABI version. `src/physics/wasm.ts` checks this on init, so a stale `wasm/pkg`
 * from an older build fails at load instead of misreading a buffer layout.
 * @returns {number}
 */
export function td_abi_version() {
    const ret = wasm.td_abi_version();
    return ret >>> 0;
}

/**
 * Allocate `count` 8-byte-aligned `f64`s and return the byte offset.
 *
 * `count == 0` returns `ERR_ALLOC` rather than a dangling-but-valid pointer:
 * a zero-sized scratch buffer is always a caller bug, and silently handing back
 * an offset that aliases the next allocation would corrupt it.
 * @param {number} count
 * @returns {number}
 */
export function td_alloc(count) {
    const ret = wasm.td_alloc(count);
    return ret >>> 0;
}

/**
 * @param {number} id
 * @param {number} handle
 * @param {number} x
 * @param {number} y
 * @param {number} z
 */
export function td_body_apply_force(id, handle, x, y, z) {
    wasm.td_body_apply_force(id, handle, x, y, z);
}

/**
 * @param {number} id
 * @param {number} handle
 * @param {number} x
 * @param {number} y
 * @param {number} z
 */
export function td_body_apply_impulse(id, handle, x, y, z) {
    wasm.td_body_apply_impulse(id, handle, x, y, z);
}

/**
 * Create a body. Returns its handle, or a negative sentinel from the `ERR_*`
 * constants above.
 *
 * `dynamic` arrives as a `bool` because the TS adapter has already collapsed
 * `kind === 'dynamic'`; `kinematic` is treated as static, exactly as
 * `BuiltinPhysics` does (its `mass` computation only special-cases `'dynamic'`).
 * @param {number} id
 * @param {number} shape_kind
 * @param {number} radius
 * @param {number} hx
 * @param {number} hy
 * @param {number} hz
 * @param {number} px
 * @param {number} py
 * @param {number} pz
 * @param {number} rx
 * @param {number} ry
 * @param {number} rz
 * @param {number} vx
 * @param {number} vy
 * @param {number} vz
 * @param {boolean} dynamic
 * @param {number} mass
 * @param {number} restitution
 * @param {number} friction
 * @param {number} group
 * @param {number} mask
 * @returns {number}
 */
export function td_body_create(id, shape_kind, radius, hx, hy, hz, px, py, pz, rx, ry, rz, vx, vy, vz, dynamic, mass, restitution, friction, group, mask) {
    const ret = wasm.td_body_create(id, shape_kind, radius, hx, hy, hz, px, py, pz, rx, ry, rz, vx, vy, vz, dynamic, mass, restitution, friction, group, mask);
    return ret;
}

/**
 * @param {number} id
 * @param {number} handle
 */
export function td_body_destroy(id, handle) {
    wasm.td_body_destroy(id, handle);
}

/**
 * Write 12 doubles for one body. Returns `1` if the handle exists, `0` if not
 * (and writes nothing).
 * @param {number} id
 * @param {number} handle
 * @param {number} out
 * @returns {number}
 */
export function td_body_get_state(id, handle, out) {
    const ret = wasm.td_body_get_state(id, handle, out);
    return ret >>> 0;
}

/**
 * `setBodyState`. `mask` is the `MASK_*` bitset: only the triples whose bit is
 * set are read, but all twelve arguments are always passed, because a varargs
 * ABI would cost more in JS marshalling than the unused registers do.
 * @param {number} id
 * @param {number} handle
 * @param {number} mask
 * @param {number} px
 * @param {number} py
 * @param {number} pz
 * @param {number} rx
 * @param {number} ry
 * @param {number} rz
 * @param {number} vx
 * @param {number} vy
 * @param {number} vz
 * @param {number} wx
 * @param {number} wy
 * @param {number} wz
 */
export function td_body_set_state(id, handle, mask, px, py, pz, rx, ry, rz, vx, vy, vz, wx, wy, wz) {
    wasm.td_body_set_state(id, handle, mask, px, py, pz, rx, ry, rz, vx, vy, vz, wx, wy, wz);
}

/**
 * Free a `td_alloc` buffer. `count` must be the value it was allocated with,
 * because `Layout` has to match exactly for `dealloc` to be sound.
 * @param {number} ptr
 * @param {number} count
 */
export function td_free(ptr, count) {
    wasm.td_free(ptr, count);
}

/**
 * Run the vectorised integration passes against scalar references inside wasm.
 *
 * This is the only test in the repo that can prove the *shipped* artifact's SIMD
 * path is bit-identical to its scalar path, because it executes in that
 * artifact. `tests/wasm_backend.test.ts` calls it and fails the build on false.
 * @returns {boolean}
 */
export function td_selftest_simd_parity() {
    const ret = wasm.td_selftest_simd_parity();
    return ret !== 0;
}

/**
 * Whether the shipped binary really carries wasm SIMD128.
 * @returns {boolean}
 */
export function td_simd_enabled() {
    const ret = wasm.td_simd_enabled();
    return ret !== 0;
}

/**
 * `PhysicsBackend.bodyCount`.
 * @param {number} id
 * @returns {number}
 */
export function td_world_body_count(id) {
    const ret = wasm.td_world_body_count(id);
    return ret >>> 0;
}

/**
 * Contacts produced by the most recent step, and destructive: calling this
 * empties the kernel's list, matching `PhysicsBackend.drainContacts`.
 * @param {number} id
 * @returns {number}
 */
export function td_world_contact_count(id) {
    const ret = wasm.td_world_contact_count(id);
    return ret >>> 0;
}

/**
 * Create a world. All five options are required; the TS adapter resolves
 * `options.x ?? default` before calling, so the defaults live in exactly one
 * place (`src/physics/types.ts`) instead of being duplicated here.
 * @param {number} gx
 * @param {number} gy
 * @param {number} gz
 * @param {number} fixed_dt
 * @param {number} solver_iterations
 * @param {number} linear_damping
 * @param {number} angular_damping
 * @returns {number}
 */
export function td_world_create(gx, gy, gz, fixed_dt, solver_iterations, linear_damping, angular_damping) {
    const ret = wasm.td_world_create(gx, gy, gz, fixed_dt, solver_iterations, linear_damping, angular_damping);
    return ret >>> 0;
}

/**
 * Drop a world and free its id. Destroying an unknown id is a no-op.
 * @param {number} id
 */
export function td_world_destroy(id) {
    wasm.td_world_destroy(id);
}

/**
 * `PhysicsBackend.dispose`. Keeps the world id valid but empties it, so an
 * adapter that disposes and reuses does not have to re-create the id.
 * @param {number} id
 */
export function td_world_dispose(id) {
    wasm.td_world_dispose(id);
}

/**
 * Drain contacts into `out` as 7 doubles each. Returns the number drained.
 *
 * Draining rather than peeking is what keeps the TS adapter's
 * `drainContacts()` semantics exact: a second call in the same frame returns an
 * empty array on both backends.
 * @param {number} id
 * @param {number} out
 * @returns {number}
 */
export function td_world_drain_contacts(id, out) {
    const ret = wasm.td_world_drain_contacts(id, out);
    return ret >>> 0;
}

/**
 * @param {number} id
 * @returns {number}
 */
export function td_world_fixed_dt(id) {
    const ret = wasm.td_world_fixed_dt(id);
    return ret;
}

/**
 * Raycast. Returns `1` and writes 8 doubles on a hit, `0` on a miss or a
 * degenerate direction (writing nothing).
 * @param {number} id
 * @param {number} ox
 * @param {number} oy
 * @param {number} oz
 * @param {number} dx
 * @param {number} dy
 * @param {number} dz
 * @param {number} max_distance
 * @param {number} out
 * @returns {number}
 */
export function td_world_raycast(id, ox, oy, oz, dx, dy, dz, max_distance, out) {
    const ret = wasm.td_world_raycast(id, ox, oy, oz, dx, dy, dz, max_distance, out);
    return ret >>> 0;
}

/**
 * Advance one fixed step. Contacts from this step stay readable until the next
 * one, which is what makes `step()` then `drainContacts()` work across a JS
 * turn boundary.
 * @param {number} id
 * @param {number} dt
 */
export function td_world_step(id, dt) {
    wasm.td_world_step(id, dt);
}

/**
 * Handles matching `td_world_write_states`, one `u32` per body, same order.
 * @param {number} id
 * @param {number} out
 * @returns {number}
 */
export function td_world_write_handles(id, out) {
    const ret = wasm.td_world_write_handles(id, out);
    return ret >>> 0;
}

/**
 * Bulk snapshot: 12 doubles per body, in ascending handle order.
 *
 * Exists so the renderer and the training loop can read a whole world in one
 * call. `td_body_get_state` is one JS->wasm transition per body; this is one
 * transition per step, which at 500 bodies is the difference between a
 * measurable and an unmeasurable overhead.
 * @param {number} id
 * @param {number} out
 * @returns {number}
 */
export function td_world_write_states(id, out) {
    const ret = wasm.td_world_write_states(id, out);
    return ret >>> 0;
}
function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbindgen_init_externref_table: function() {
            const table = wasm.__wbindgen_externrefs;
            const offset = table.grow(4);
            table.set(0, undefined);
            table.set(offset + 0, undefined);
            table.set(offset + 1, null);
            table.set(offset + 2, true);
            table.set(offset + 3, false);
        },
    };
    return {
        __proto__: null,
        "./threedream_physics_wasm_bg.js": import0,
    };
}

let wasmModule, wasmInstance, wasm;
function __wbg_finalize_init(instance, module) {
    wasmInstance = instance;
    wasm = instance.exports;
    wasmModule = module;
    wasm.__wbindgen_start();
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (!module.ok) {
            throw new Error(`failed to fetch Wasm: ${module.status} ${module.statusText} fetching '${module.url}'`);
        }

        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('threedream_physics_wasm_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
