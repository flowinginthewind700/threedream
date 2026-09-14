/* tslint:disable */
/* eslint-disable */
export const memory: WebAssembly.Memory;
export const td_abi_version: () => number;
export const td_alloc: (a: number) => number;
export const td_body_apply_force: (a: number, b: number, c: number, d: number, e: number) => void;
export const td_body_apply_impulse: (a: number, b: number, c: number, d: number, e: number) => void;
export const td_body_create: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number, r: number, s: number, t: number, u: number) => number;
export const td_body_destroy: (a: number, b: number) => void;
export const td_body_get_state: (a: number, b: number, c: number) => number;
export const td_body_set_state: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number) => void;
export const td_free: (a: number, b: number) => void;
export const td_selftest_simd_parity: () => number;
export const td_simd_enabled: () => number;
export const td_world_body_count: (a: number) => number;
export const td_world_contact_count: (a: number) => number;
export const td_world_create: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => number;
export const td_world_destroy: (a: number) => void;
export const td_world_dispose: (a: number) => void;
export const td_world_drain_contacts: (a: number, b: number) => number;
export const td_world_fixed_dt: (a: number) => number;
export const td_world_raycast: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => number;
export const td_world_step: (a: number, b: number) => void;
export const td_world_write_handles: (a: number, b: number) => number;
export const td_world_write_states: (a: number, b: number) => number;
export const __wbindgen_externrefs: WebAssembly.Table;
export const __wbindgen_start: () => void;
