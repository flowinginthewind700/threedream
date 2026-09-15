/**
 * ThreeDream public API.
 *
 * Layering is enforced by import direction, and this barrel is the only entry
 * point consumers should need:
 *
 *   core    clock / ECS / events / engine facade   (no three.js, no WASM)
 *   physics backend interface + three solvers      (no three.js; WASM on demand)
 *   gpu     probing, shared device, compute, particles,
 *           soft bodies                            (no three.js, no WASM)
 *   ai      MLP, Gaussian policy, REINFORCE trainer (no three.js, no WASM)
 *   envs    learning tasks                         (physics only)
 *   render  three.js bridge                        (the only layer importing three)
 *
 * Everything except `render` runs in bare Node, which is what makes headless
 * training and CI possible. The two WASM backends are re-exported below and
 * still belong to that group, because neither touches its binary at import
 * time: `createRapierPhysics` and `loadWasmKernel` each `await import()` their
 * module from inside the factory, so a script that never asks for a WASM world
 * never instantiates one. `loadWasmKernel` additionally reads the `.wasm` bytes
 * through `node:fs` when the glue resolves to a `file:` URL, which Node's
 * `fetch` cannot.
 *
 * For that reason `render` is deliberately *not* re-exported here: importing
 * this barrel must not pull three.js into a training script. Browser code
 * imports `ThreeRenderer` from `threedream/render` (`src/render/scene.ts`)
 * directly. `ParticleView` (`src/render/particles.ts`) is the M3 half of that
 * rule: the whole `gpu/particle*` stack below runs headless -- the CPU tier is
 * the deterministic reference precisely so that replay and training never need
 * a device -- and only the view that turns its output into an `InstancedMesh`
 * lives in `render/`. `tests/barrel.test.ts` pins this by importing the barrel
 * in bare Node.
 *
 * `SoftView` (`src/render/soft.ts`) is the M4 half of the same rule, and the
 * soft stack below it is the same shape: a mesh generator, an island grouper, a
 * constraint colorer, a CPU reference solver, the WGSL generated from that
 * reference's constants, and the GPU backend -- all headless, all reachable from
 * this barrel, with only the surface-and-wireframe view in `render/`.
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

export {
  LIMIT_FLOOR,
  LIMIT_NAMES,
  OPTIONAL_FEATURES,
  OPTIONAL_FEATURE_NAMES,
  clampWorkgroupSize,
  describeCapabilities,
  gpuFrom,
  maxElements,
  probeWebGpu,
  probeWebgl2,
  selectRenderTier,
  snapshotLimits,
  unmetLimitsOf,
} from './gpu/capabilities.js';
export type {
  CanvasLike,
  FeatureLevel,
  GpuAdapterLike,
  GpuLike,
  GpuLimits,
  LimitName,
  OptionalFeature,
  ProbeWebGpuOptions,
  RenderTier,
  RenderTierDecision,
  UnavailableReason,
  WebGpuCapabilities,
  Webgl2Capabilities,
} from './gpu/capabilities.js';

export {
  REQUESTED_LIMITS,
  SharedDevice,
  SharedDeviceManager,
  acquireSharedDevice,
  deviceGpuFrom,
  gpuConstantsFrom,
  requiredLimitsFor,
  sharedDevices,
} from './gpu/device.js';
export type {
  DeviceFailure,
  DeviceFailureReason,
  DeviceGpuLike,
  GpuBufferLike,
  GpuCommandEncoderLike,
  GpuCompilationMessage,
  GpuComputePassLike,
  GpuConstants,
  GpuDeviceAdapterLike,
  GpuDeviceLike,
  GpuDeviceLostInfo,
  GpuQueueLike,
  GpuShaderModuleLike,
  SharedDeviceInfo,
  SharedDeviceOptions,
} from './gpu/device.js';

export {
  ComputeBuffer,
  ComputeContext,
  ComputeProgram,
  PingPong,
  ShaderCompilationError,
  rawBuffer,
  submitCopy,
} from './gpu/compute.js';
export type {
  ComputeBinding,
  ComputeBufferOptions,
  ComputeBufferType,
  ComputeBufferView,
  ComputeDispatch,
  ComputeProgramOptions,
  ComputeResource,
  CopyPair,
  ReadbackSource,
} from './gpu/compute.js';

export type {
  BoundsMode,
  ParticleSimOptions,
  ParticleStepStats,
  ParticleSystem,
  ResolvedParticleOptions,
} from './gpu/particleTypes.js';
export {
  DEFAULT_BOUNDS,
  DEFAULT_RADIUS,
  PARTICLE_BYTES,
  PARTICLE_STRIDE,
  ParticleField,
  boundsCenter,
  boundsInradius,
  boundsSize,
} from './gpu/particleField.js';
export type { Bounds, ParticleFieldOptions, ParticleScene, Vec3Tuple } from './gpu/particleField.js';
export { DEFAULT_BUCKET_CAPACITY, SpatialHash, hashCellCoords, nextPow2 } from './gpu/particleHash.js';
export type { SpatialHashOptions, SpatialHashStats } from './gpu/particleHash.js';
export {
  DEFAULT_GRAVITY,
  DEFAULT_PARTICLE_OPTIONS,
  assertFieldFits,
  effectiveCellSize,
  resolveParticleOptions,
} from './gpu/particleOptions.js';
export {
  CpuParticleSystem,
  POSITION_CORRECTION,
  createCpuParticleSystem,
} from './gpu/particleCpu.js';
export type { CpuParticleSystemOptions } from './gpu/particleCpu.js';
export {
  GpuParticleSystem,
  createGpuParticleSystem,
  gpuBufferBudget,
  tableSizeFor,
} from './gpu/particleGpu.js';
export type { GpuBufferBudget, GpuParticleSystemOptions } from './gpu/particleGpu.js';
export {
  INSTANCE_BYTES,
  INSTANCE_FLOATS,
  InstanceExpander,
  instanceBufferBytes,
  instanceShaderSource,
} from './gpu/particleInstances.js';
export type { InstanceExpanderOptions } from './gpu/particleInstances.js';
export {
  PARTICLE_KERNELS,
  WORKGROUP_SIZE,
  particleShaderSource,
  workgroupsFor,
} from './gpu/particleWgsl.js';
export type { ParticleKernel, WgslBinding, WgslParamMember } from './gpu/particleWgsl.js';
export {
  ParticleRunner,
  createParticleRunner,
  createParticleSystem,
  probeParticles,
} from './gpu/particles.js';
export type {
  ParticleProbe,
  ParticleProbeRequest,
  ParticleSystemHandle,
  ParticleSystemRequest,
  TierFallback,
} from './gpu/particles.js';

export type {
  ResolvedSoftOptions,
  SoftBoundsMode,
  SoftPlan,
  SoftSimOptions,
  SoftStepStats,
  SoftSystem,
} from './gpu/softTypes.js';
export {
  DEFAULT_JITTER,
  RADIUS_BOX_FRACTION,
  RADIUS_FRACTION,
  SCENE_EXTENT_FRACTION,
  SOFT_BYTES,
  SOFT_OFFSET,
  SOFT_SCENES,
  SOFT_STIFFNESS,
  SOFT_STRIDE,
  SoftMesh,
  assertConstraints,
  assertTriangles,
  defaultExtent,
  emptyConstraints,
  sizeForScene,
} from './gpu/softMesh.js';
export type {
  SoftConstraints,
  SoftMeshOptions,
  SoftMeshSpec,
  SoftScene,
  SoftSceneSize,
} from './gpu/softMesh.js';
export {
  NODE_SENTINEL,
  SOFT_WORKGROUP_SIZE,
  groupIslands,
  isSentinel,
  softWorkgroups,
} from './gpu/softIslands.js';
export type { IslandInput, SoftIslands } from './gpu/softIslands.js';
export {
  MAX_COLORS,
  colorConstraints,
  coloringIsRaceFree,
  coloringWorkgroups,
} from './gpu/softColoring.js';
export type { SoftBatch, SoftColoring } from './gpu/softColoring.js';
export {
  DEFAULT_SOFT_GRAVITY,
  DEFAULT_SOFT_OPTIONS,
  MAX_SOFT_ITERATIONS,
  SOFT_BOUNDS_MODE_BITS,
  SOFT_FIXED_DISPATCHES,
  SOFT_FLAG,
  SOFT_PARAM_WORD,
  SOFT_PARAMS_BYTES,
  SOFT_PARAMS_FLOATS,
  assertMeshFits,
  buildSoftLayout,
  resolveSoftOptions,
  writeSoftParams,
} from './gpu/softOptions.js';
export type { SoftLayout, SoftParamsFrame } from './gpu/softOptions.js';
export { CpuSoftSystem, createCpuSoftSystem } from './gpu/softCpu.js';
export type { CpuSoftSystemOptions } from './gpu/softCpu.js';
export {
  SOFT_BASELINE_STORAGE_BUFFERS,
  SOFT_BATCH_STRIDE_BYTES,
  SOFT_BATCH_U32_PER_COLOR,
  SOFT_BINDINGS,
  SOFT_EDGE_F32_PER_CONSTRAINT,
  SOFT_ENDS_U32_PER_CONSTRAINT,
  SOFT_GROUP_STATE,
  SOFT_GROUP_STATIC,
  SOFT_KERNELS,
  SOFT_KERNEL_DISPATCH,
  SOFT_ORDER_U32_PER_CONSTRAINT,
  SOFT_PRED_VECS_PER_NODE,
  SOFT_PUBLISH_FLOATS_PER_NODE,
  SOFT_SLEEP_WORDS_PER_ISLAND,
  SOFT_STATE_FLOATS_PER_NODE,
  SOFT_STATE_VECS_PER_NODE,
  SOFT_STAT_WORD,
  SOFT_STAT_WORDS,
  SOFT_STORAGE_BINDINGS,
  SOFT_WGSL_PARAMS_LAYOUT,
  softBindingsForGroup,
  softShaderSource,
  softSolveDispatch,
} from './gpu/softWgsl.js';
export type {
  SoftBindingKind,
  SoftBufferType,
  SoftKernel,
  SoftKernelDispatch,
  SoftParamKind,
  SoftWgslBinding,
  SoftWgslParamMember,
} from './gpu/softWgsl.js';
export { GpuSoftSystem, createGpuSoftSystem, softGpuBudget } from './gpu/softGpu.js';
export type { GpuSoftSystemOptions, SoftGpuBudget } from './gpu/softGpu.js';
export { SoftRunner, createSoftRunner, createSoftSystem, probeSoft } from './gpu/soft.js';
export type {
  SoftProbe,
  SoftProbeRequest,
  SoftSystemHandle,
  SoftSystemRequest,
} from './gpu/soft.js';

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
