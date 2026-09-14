# ThreeDream

A game and physics-AI kernel that runs in the browser: a deterministic ECS with a
fixed-timestep simulation loop, a pluggable physics layer, and a reinforcement
learner that trains live in the page. Rendering is three.js. The simulation core
is plain TypeScript and needs no browser and no GPU; a Rust/wasm kernel
implements the same solver bit for bit, for the runs where speed matters.

一个跑在浏览器里的游戏与物理-AI 内核：确定性 ECS + 固定步长仿真循环、可插拔的物理
层，以及在页面里实时训练的强化学习器。渲染用 three.js。仿真核心是纯 TypeScript，不
依赖浏览器和 GPU；Rust/wasm 内核逐位实现同一个求解器，用在需要速度的场合。

[![CI](https://github.com/flowinginthewind700/threedream/actions/workflows/ci.yml/badge.svg)](https://github.com/flowinginthewind700/threedream/actions/workflows/ci.yml)

![the demo, mid-training](docs/demo-drive.jpg)

Live demo: <https://flowinginthewind700.github.io/threedream/>, deployed from
`main` automatically once every gate in CI is green.

在线演示：<https://flowinginthewind700.github.io/threedream/>，`main` 上所有 CI 关卡
通过后自动部署。

## Quickstart

```bash
npm install
npm run dev       # browser demo on http://localhost:5173
npm run train     # headless training in Node, prints a progress trace
npm test          # 359 unit tests, ~24s
npm run verify    # typecheck + test + build
```

Requires Node >= 22.12. None of that needs a Rust toolchain: `wasm/pkg` is
committed as a build artifact, so the wasm backend works from a bare checkout.
Rebuilding the kernel from `rust/` is what needs Rust >= 1.85 and wasm-pack:

```bash
npm run test:rust    # 68 native Rust tests for the solver itself
npm run build:wasm   # regenerate wasm/pkg from rust/
npm run check:wasm   # the CI artifact gate: ABI, provenance, behaviour, freshness
```

以上命令都不需要 Rust 工具链：`wasm/pkg` 作为构建产物已提交，裸克隆即可使用 wasm
后端。只有从 `rust/` 重建内核时才需要（Rust >= 1.85 与 wasm-pack）：`test:rust` 跑
68 个原生 Rust 测试，`build:wasm` 重新生成 `wasm/pkg`，`check:wasm` 是 CI 的产物
关卡（ABI、来源、行为、新鲜度）。

The demo trains in the page. Press **Train**, watch the mean episode return climb
on the sparkline, then flip **Playback** from `Random` to `Learned` to see what it
learned. The viewport badge tracks how many episodes the current policy has behind
it. **Reach** is the harder of the two tasks.

Two more pages ship beside the trainer. `/physics-check.html` runs one canonical
scene through `builtin`, `wasm` and a wasm replay in your own browser and
compares the digests. `/shared-device.html` proves that a single `GPUDevice` can
back three.js rendering and a raw WGSL compute pipeline at the same time, which
is the assumption the whole GPU roadmap rests on.

演示页里直接训练。点 **Train**，看火花线上 mean episode return 上升，再把
**Playback** 从 `Random` 切到 `Learned`，就能看到它学到了什么。视口角标记录当前策略
背后有多少 episode。**Reach** 是两个任务里更难的那个。

训练页旁边还有两个页面。`/physics-check.html` 在你自己的浏览器里用 `builtin`、
`wasm` 和一次 wasm 回放跑同一个规范场景，并比对摘要。`/shared-device.html` 证明
同一个 `GPUDevice` 可以同时支撑 three.js 渲染与一条裸 WGSL compute pipeline，
这正是整条 GPU 路线图所依赖的前提。

## Why it is shaped this way

Two rules drive the design:

1. **The simulation is the only source of truth.** The renderer mirrors physics
   state into `THREE.Object3D`s each frame and never writes back. The same scene
   therefore runs headless for training and rendered for play, with identical
   results.
2. **Everything that matters is deterministic.** A seeded RNG, a fixed timestep,
   and no hidden global state mean `engine.step(n)` in Node and `engine.frame(dt)`
   in a browser produce the same simulation. That is what makes a physics-AI
   kernel testable: all 359 tests run without a GPU, and the wasm backend is
   held to the bits of the TypeScript solver it ports.

两条规则决定了整体设计：

1. **仿真是唯一事实来源。** 渲染层每帧把物理状态镜像到 `THREE.Object3D`，从不回写。
   于是同一个场景既能无头训练，也能在浏览器里渲染游玩，结果完全一致。
2. **关键路径都是确定性的。** 带种子的 RNG、固定步长、没有隐藏的全局状态，所以 Node
   里的 `engine.step(n)` 与浏览器里的 `engine.frame(dt)` 跑的是同一个仿真。这让一个
   物理-AI 内核变得可测：359 个测试全都不需要 GPU，而 wasm 后端要对齐它所移植的 TS
   求解器的每一个比特。

## Layers

```
core     clock / ECS / events / engine facade     no three.js, no WASM
physics  backend interface + three solvers        no three.js
gpu      capability probing + render tier         no three.js, no WASM
ai       MLP, Gaussian policy, policy-gradient    no three.js, no WASM
envs     learning tasks (drive, reach)            physics only
render   three.js bridge                          the only layer importing three
```

`src/index.ts` re-exports everything **except** `render`. Importing the barrel
must not pull three.js into a training script, so browser code imports
`ThreeRenderer` directly. Deliberate constraint, not an oversight. The two WASM
backends are in the barrel and still cost nothing to import: each loads its
binary through a dynamic `import()` inside its own factory, so a headless run
never instantiates a module it did not ask for.

`gpu/` is the layer that decides how good a browser's graphics stack is, and it
is a probe, not a build flag: `selectRenderTier` takes adapter results as
arguments and returns `webgpu`, `webgl2` or `cpu`. Nothing in it reads
`import.meta.env`, so the fallback cannot be baked in at compile time.

`src/index.ts` 重导出**除** `render` **之外**的全部内容。引入这个入口不能把 three.js
带进训练脚本，所以浏览器代码直接引 `ThreeRenderer`。这是刻意的约束，不是遗漏。两个
WASM 后端**在**这个入口里，引入它们依然不花代价：各自的工厂函数内部用动态
`import()` 加载二进制，所以无头运行不会实例化它没要过的模块。

`gpu/` 是判断浏览器图形栈成色的那一层，而且是探测，不是构建开关：
`selectRenderTier` 接收 adapter 探测结果作为参数，返回 `webgpu`、`webgl2` 或
`cpu`。它内部不读 `import.meta.env`，所以回退路径不可能在编译期被写死。

### Physics backends

All three implement `PhysicsBackend`, so envs and the trainer stay agnostic:

| Backend | File | Notes |
|---|---|---|
| `builtin` | `src/physics/builtin.ts` | Pure TS, deterministic, zero dependencies. Default, and the normative spec. |
| `wasm` | `src/physics/wasm.ts` | First-party Rust kernel. Bit-identical to `builtin`, 2.6x-4.9x faster in Node. |
| `rapier` | `src/physics/rapier.ts` | Third-party Rapier WASM. Same interface, different solver: ~3% off steady-state speeds, and different trajectories. |

`wasm` is the backend whose claim is unusual. Not "same interface" but "same
bits": `src/physics/reference.ts` defines one canonical scene plus a digest over
raw IEEE-754 bit patterns, and every runner has to reproduce
`REFERENCE_GOLDEN_DIGEST` — in Node, in a browser, and on a replay of the same
seed. Rounding the digest inputs would hide a one-ulp divergence, which is the
only kind of divergence a determinism gate exists to catch. The Rust side is
three crates under `rust/`: the solver, with no wasm-bindgen dependency, so
`cargo test` runs it natively; a cdylib that is nothing but the ABI; and a wgpu
skeleton for the GPU milestones. `rapier` stays as a cross-check against a
solver nobody here controls.

Swapping backends has caught real bugs: Rapier's `RigidBodyDesc.setAdditionalMass`
is lazy, folded in only at the next `world.step()`, so a first-frame impulse used
to land on a body roughly 400x too light. `tests/rapier_backend.test.ts` pins both
the impulse-on-first-step behaviour and the cross-backend agreement.

三个后端都实现 `PhysicsBackend`，因此 env 与 trainer 与之无关：

| 后端 | 文件 | 说明 |
|---|---|---|
| `builtin` | `src/physics/builtin.ts` | 纯 TS、确定性、零依赖。默认，也是规范实现。 |
| `wasm` | `src/physics/wasm.ts` | 自研 Rust 内核。与 `builtin` 逐位一致，Node 下快 2.6x-4.9x。 |
| `rapier` | `src/physics/rapier.ts` | 第三方 Rapier WASM。接口相同，求解器不同：稳态速度差约 3%，轨迹也不同。 |

`wasm` 是承诺最特殊的那个后端。不是「接口相同」，而是「比特相同」：
`src/physics/reference.ts` 定义一个规范场景，外加一个基于 IEEE-754 原始比特模式的
摘要，所有运行方都必须复现 `REFERENCE_GOLDEN_DIGEST` —— 在 Node 里、在浏览器里、
以及同一 seed 的回放里。对摘要输入做舍入会掩盖 1 ulp 的偏差，而那正是确定性关卡
唯一要抓的偏差类型。Rust 侧是 `rust/` 下的三个 crate：不依赖 wasm-bindgen 的求解器
（因此 `cargo test` 能原生跑它）、只负责 ABI 的 cdylib，以及为 GPU 里程碑准备的
wgpu 骨架。`rapier` 保留下来，作为对「这里没人能控制的求解器」的交叉验证。

换后端真抓到过 bug：Rapier 的 `RigidBodyDesc.setAdditionalMass` 是惰性的，要到下一次
`world.step()` 才生效，于是第一帧的冲量曾经打在一个轻了约 400 倍的刚体上。
`tests/rapier_backend.test.ts` 钉死了「首帧冲量」行为和两个后端的一致性。

### The learner

`src/ai/trainer.ts` is a batched policy-gradient method with GAE advantages and a
learned critic. Deliberately not PPO: no importance ratios, no replay buffer, so
the whole update is readable end to end while still being a real policy-gradient
method. Its header comment records three design decisions that were each forced by
a measured failure, because those failures are silent: entropy stays healthy,
training returns look plausible, and greedy evaluation simply never improves.

`src/ai/trainer.ts` 是带 GAE 优势估计与学习式 critic 的批量策略梯度方法。刻意不做
PPO：没有重要性采样比，也没有 replay buffer，整个更新过程可以从头读到尾，同时仍然是
一个真正的策略梯度方法。文件头注释记录了三个「各自被一次实测失败逼出来」的设计决定，
因为这类失败是无声的：熵保持健康、训练回报看着合理，但贪婪评估就是不涨。

Environments follow a gymnasium-style contract (`reset` -> `step` -> `observe`)
over typed fixed-size buffers, because allocation inside a training loop starts to
dominate cost once episodes run in the thousands.

环境遵循 gymnasium 风格的契约（`reset` -> `step` -> `observe`），使用带类型的定长缓冲
区，因为一旦 episode 上千，训练循环里的内存分配就开始主导开销。

| Env | Task | Obs | Act |
|---|---|---|---|
| `DriveEnv` | Steer a body to a randomised goal under real dynamics | 6 | 2 |
| `ReachEnv` | Push a puck into a goal through contact | 14 | 2 |

`ReachEnv`'s 14 observations are 6 planar positions (agent / puck / goal), 4 unit
direction vectors (agent->puck, puck->goal) and 4 planar velocities. Contact tasks
are second-order: knowing where the puck is without knowing which way it is already
sliding leaves the policy a step behind. Positions are normalised by arena size so
the policy transfers across scales.

`ReachEnv` 的 14 维观测是 6 个平面位置（agent / puck / goal）、4 个单位方向向量
（agent->puck、puck->goal）和 4 个平面速度。接触类任务是二阶的：只知道 puck 在哪而不
知道它正往哪滑，策略就永远慢一拍。位置按场地尺寸归一化，所以策略能跨尺度迁移。

## Headless training

```bash
npm run train
npm run train -- --episodes 4000 --seed 3 --out artifacts/policy.json
npm run train -- --help
```

Flags: `--episodes N` `--seed N` `--lr N` `--batch N` `--hidden a,b` `--every N`
`--out PATH`. The script trains `ReachEnv`; the same loop drives the browser demo,
so a policy that trains here plays there.

参数：`--episodes N` `--seed N` `--lr N` `--batch N` `--hidden a,b` `--every N`
`--out PATH`。该脚本训练 `ReachEnv`；同一个循环也在驱动浏览器演示，所以这里训练出来
的策略就能在那里玩。

## Writing your own task

```ts
import { createEngine, createBuiltinPhysics, vec3 } from './src/index.js';

const physics = createBuiltinPhysics({ gravity: vec3(0, -9.81, 0) });
const engine = createEngine({ physics });

engine.addGround();
const box = physics.createBody({
  shape: { kind: 'box', halfExtents: vec3(0.25, 0.25, 0.25) },
  position: vec3(0, 3, 0),
  mass: 1,
  restitution: 0.3,
});

engine.step(120); // two seconds at 60 Hz
console.log(physics.getBodyState(box)!.position); // -> [0, 0.246, 0]
```

Implement `LearningEnvironment` over such a world and the trainer learns it with no
changes. `src/envs/drive.ts` is the shortest useful example.

在这样的世界上实现 `LearningEnvironment`，trainer 无需任何改动就能学它。
`src/envs/drive.ts` 是最短且有用的示例。

## Testing and CI

The project is developed test-first: a spec exists for every module under `src/`,
and `tests/tdd.test.ts` fails the build the moment a new module lands without one.

| Command | What it runs | Cost |
|---|---|---|
| `npm test` | 359 unit tests, headless, no GPU needed | ~24s |
| `npm run test:coverage` | same suite under v8, floor enforced by `vitest.config.ts` | ~16s |
| `npm run test:e2e` | 19 Playwright tests over 3 specs, against a real browser | ~25s |
| `npm run test:rust` | 68 native Rust tests for the solver | ~1s warm |
| `npm run check:wasm` | 35 assertions over the committed `wasm/pkg`: ABI, provenance, behaviour | ~1s |

Coverage is a separate command, not a flag on `npm test`: instrumenting the
training inner loop costs ~10x wall time, and folding that into the fast loop
would destroy the red/green cadence the tests exist to provide. So the coverage
run skips the two convergence tests (`COVERAGE=1`, `tests/trainer.test.ts`) that
cost 180s of the 185s and contribute ~0.3% of branch coverage — `npm test` still
runs them, and `tests/tdd.test.ts` pins that asymmetry. The floor sits a few
points under what the suite actually measures (93/82 against 98/89), which is
the part that matters — a threshold set far below current reality is decoration,
while one set at it makes every refactor a fight.

`.github/workflows/ci.yml` is one file with five jobs: `verify` (typecheck +
tests + build), `coverage`, `e2e` and `rust` run in parallel as gates, and
`deploy` depends on all four. A separate `pages.yml` would deploy `main` when a gate
is red, because Actions cannot express `needs:` across workflow files — so the
dependency lives in one graph, and `tests/ci_workflow.test.ts` pins it (along with
the SHA-pinning of every action and the fact that gate jobs hold no publish
permission). On a green push to `main`, `deploy` builds with the Pages base path
and publishes `dist/`.

The `rust` gate exists because `wasm/pkg` is a checked-in build artifact, and no
other job can tell whether those bytes still match `rust/`: TypeScript checks
`src/physics/wasm.ts` against the `WasmBindings` interface it declares, not
against the binary. So the job runs the native solver tests, builds the wgpu
crate for `wasm32` (the M0 claim that the pinned version still compiles), checks
the committed artifact, then rebuilds it and requires identical bytes. wasm-pack
comes from a checksummed tarball rather than `curl | sh`: that installer is a
mutable ref on a third-party repo, and this is the tool minting the binary the
determinism claim rests on.

本项目采用测试先行：`src/` 下每个模块都有对应 spec，一旦出现没有测试的新模块，
`tests/tdd.test.ts` 会让构建失败。

覆盖率是独立命令而非 `npm test` 的参数：给训练内层循环插桩会让墙钟时间涨约 10 倍，
把它塞进快速循环会毁掉测试本该提供的红/绿节奏。因此覆盖率运行会跳过那两个收敛测试
（`COVERAGE=1`，`tests/trainer.test.ts`）—— 它们占了 185s 里的 180s，却只贡献约 0.3%
的分支覆盖率；`npm test` 照跑不误，这条不对称由 `tests/tdd.test.ts` 钉死。
下限压在实测值之下几个点（实测 98 / 89，下限 93 / 82）—— 这才是关键：远低于现状的
阈值只是装饰，而贴着现状设阈值则会让每次重构都变成搏斗。

`.github/workflows/ci.yml` 是单文件五任务：`verify`（类型检查 + 测试 + 构建）、
`coverage`、`e2e`、`rust` 四个关卡并行，`deploy` 依赖这四者。单独的 `pages.yml` 会在关卡变红时
仍然部署 `main`，因为 Actions 无法跨工作流文件表达 `needs:`，所以依赖关系收敛在一个图里，
并由 `tests/ci_workflow.test.ts` 钉死（连同每个 action 的 SHA 锁定、以及关卡任务不持有发布
权限这一事实）。`main` 上一次绿色 push 后，`deploy` 用 Pages base 路径构建并发布 `dist/`。

`rust` 关卡存在的原因是：`wasm/pkg` 是提交进仓库的构建产物，而没有别的任务能判断
这些字节是否还与 `rust/` 一致 —— TypeScript 检查的是 `src/physics/wasm.ts` 与它
自己声明的 `WasmBindings` 接口，不是那个二进制。所以这个任务跑原生求解器测试、为
`wasm32` 构建 wgpu crate（即 M0 那条「锁定版本仍能编译」的断言）、校验已提交的
产物，然后重建并要求字节完全一致。wasm-pack 从带校验和的 tarball 安装，而不是
`curl | sh`：安装脚本是第三方仓库上的可变 ref，而它正是铸造整个确定性承诺所依赖
的那个二进制的工具。

## Unreal Engine reference (opt-in)

Current feasibility study: [Rust + wasm + WebGPU + three.js](docs/feasibility-rust-wasm-webgpu.md).
Development plan: [ThreeDream roadmap](docs/development-plan.md).

`thirdparty/UnrealEngine` is a git submodule pointing at Epic's **private**
repository, kept as an architecture reference to read rather than to build
against. The repo stores only the gitlink; no Unreal source is committed.

Cloning this project leaves that path empty. Fetching it requires a GitHub account
that has accepted Epic's EULA and been granted access:

```bash
bash scripts/clone_unreal_reference.sh          # branch: release (default)
bash scripts/clone_unreal_reference.sh 5.5      # or another branch
```

It is a multi-GB shallow clone and the script retries transient network errors.
Nothing in `src/`, `tests/` or the demo depends on it, so a plain
`npm install && npm test` works without it.

`thirdparty/UnrealEngine` 是一个指向 Epic **私有**仓库的 git submodule，留作供阅读的
架构参考，不参与构建。仓库里只存 gitlink，**不提交任何 Unreal 源码**。

当前可行性调研：[Rust + wasm + WebGPU + three.js](docs/feasibility-rust-wasm-webgpu.md)。
开发计划：[ThreeDream 路线图](docs/development-plan.md)。

克隆本项目后该目录是空的。拉取它需要一个已接受 Epic EULA 并获得访问权限的 GitHub
账号（命令同上）。这是一次数 GB 的浅克隆，脚本会对瞬时网络错误重试。`src/`、`tests/`
和演示都不依赖它，所以直接 `npm install && npm test` 就能跑通。

## Layout

```
src/core/      clock.ts ecs.ts engine.ts events.ts rng.ts
src/physics/   types.ts builtin.ts wasm.ts rapier.ts reference.ts components.ts
src/gpu/       capabilities.ts
src/ai/        mlp.ts policy.ts trainer.ts baseline.ts
src/envs/      types.ts drive.ts reach.ts
src/render/    scene.ts
rust/          physics (solver) / physics-wasm (ABI) / gpu (wgpu skeleton)
wasm/pkg/      committed wasm-pack output, rebuilt by `npm run build:wasm`
scripts/       train_headless.ts check_wasm_artifact.mjs bench_*.mjs
               audit_unreal_reference.mjs clone_unreal_reference.sh
docs/          feasibility study, development plan, demo assets
demo/          index (trainer), physics-check, shared-device: .html + .ts
tests/         18 files, 359 tests
e2e/           demo (WebGL), wasm physics, shared device (WebGPU)
.github/       ci.yml: four gates (verify / coverage / e2e / rust) then Pages deploy
thirdparty/    UnrealEngine (submodule, opt-in)
```

## License

MIT. See [LICENSE](LICENSE).

MIT，见 [LICENSE](LICENSE)。
