/**
 * ThreeDream public API.
 *
 * Layering is enforced by import direction, and this barrel is the only entry
 * point consumers should need:
 *
 *   core    clock / ECS / events / engine facade   (no three.js, no WASM)
 *   physics backend interface + built-in solver    (no three.js, no WASM)
 *   ai      MLP, Gaussian policy, REINFORCE trainer (no three.js, no WASM)
 *   envs    learning tasks                         (physics only)
 *   render  three.js bridge                        (the only layer importing three)
 *
 * Everything except `render` and `physics/rapier` runs in bare Node, which is
 * what makes headless training and CI possible. `physics/wasm` is the exception
 * that still belongs to that group: it runs in Node too, because
 * `loadWasmKernel` reads the `.wasm` bytes through `node:fs` when the glue
 * resolves to a `file:` URL.
 *
 * For that reason `render` is deliberately *not* re-exported here: importing
 * this barrel must not pull three.js into a training script. Browser code
 * imports `ThreeRenderer` from `threedream/render` (`src/render/scene.ts`)
 * directly, and `createRapierPhysics` likewise stays a separate import so the
 * WASM only loads where it is asked for.
 */

export { Rng, createRng } from './core/rng.js';
export { FixedClock, DEFAULT_FIXED_DT } from './core/clock.js';
export type { FixedClockOptions } from './core/clock.js';
export {
  World,
  defineComponent,
  type ComponentToken,
  type EntityId,
  type System,
  type SystemContext,
  type SystemPhase,
} from './core/ecs.js';
export { EventBus } from './core/events.js';
export type { EventEmitter, Listener } from './core/events.js';
export { Engine, createEngine } from './core/engine.js';
export type {
  EngineEvents,
  EngineOptions,
  EngineStats,
  CreateEngineOptions,
} from './core/engine.js';

export {
  vec3,
  addVec3,
  subVec3,
  scaleVec3,
  dotVec3,
  crossVec3,
  lengthVec3,
  normalizeVec3,
  distanceVec3,
  axisVec3,
  quatFromEulerXYZ,
  eulerFromQuatXYZ,
} from './physics/types.js';
export type {
  Vec3,
  Quat,
  BodyShape,
  BodyKind,
  BodyDescriptor,
  BodyState,
  ContactEvent,
  RayHit,
  PhysicsBackend,
  PhysicsWorldOptions,
} from './physics/types.js';
export { BuiltinPhysics, createBuiltinPhysics } from './physics/builtin.js';
export { RapierPhysics, createRapierPhysics } from './physics/rapier.js';
export {
  WasmPhysics,
  createWasmPhysics,
  WASM_ABI_VERSION,
  loadWasmKernel,
} from './physics/wasm.js';
export type { WasmPhysicsOptions, WasmKernel } from './physics/wasm.js';
export {
  PhysicsSystem,
  RigidBodyComponent,
  type RigidBody,
  type PhysicsEvents,
  type PhysicsSystemOptions,
} from './physics/components.js';

export { Mlp } from './ai/mlp.js';
export type { MlpSpec, MlpSnapshot } from './ai/mlp.js';
export { GaussianPolicy } from './ai/policy.js';
export type { GaussianPolicyOptions, PolicySnapshot } from './ai/policy.js';
export { Trainer } from './ai/trainer.js';
export type { TrainerOptions, EpisodeResult, TrainResult } from './ai/trainer.js';

export type {
  LearningEnvironment,
  StepResult,
  RewardShape,
} from './envs/types.js';
export { DriveEnv } from './envs/drive.js';
export type { DriveEnvOptions, DriveDiagnostics } from './envs/drive.js';
export { ReachEnv } from './envs/reach.js';
export type { ReachEnvOptions, ReachDiagnostics } from './envs/reach.js';
