# ThreeDream

A game and physics-AI kernel that runs in the browser: a deterministic ECS with a
fixed-timestep simulation loop, a pluggable physics layer, and a reinforcement
learner that trains live in the page. Rendering is three.js. The simulation core
is plain TypeScript and needs no browser and no GPU; a Rust/wasm kernel
implements the same solver bit for bit, for the runs where speed matters.
On top of that sits a WebGPU scale layer: a particle system at up to six compute
dispatches a step, and a soft-body solver that colors the constraint graph so no
two edges in one dispatch share a node. Both run on a device three.js is already
using, and both blit their output straight into a buffer the renderer allocated,
so a frame costs a few draw calls and no readback. On an iGPU, as the median of
20 timed chunks a step: 100,000 particles at 20-26 ms/step, a 10,000-node cloth
at 2.3-3.2 and 20,000 nodes at 3.9-6.3.

一个跑在浏览器里的游戏与物理-AI 内核：确定性 ECS + 固定步长仿真循环、可插拔的物理
层，以及在页面里实时训练的强化学习器。渲染用 three.js。仿真核心是纯 TypeScript，不
依赖浏览器和 GPU；Rust/wasm 内核逐位实现同一个求解器，用在需要速度的场合。
在这之上是一层 WebGPU 规模层：粒子系统每步最多六次 compute dispatch；软体求解器给
约束图着色，于是同一批 dispatch 里不会有两条边共用一个节点。两者都跑在 three.js
已经在用的设备上，也都把输出直接 blit 进渲染器自己分配好的 buffer，所以一帧只有几个
draw call、没有回读。iGPU 上，每步取 20 个计时 chunk 的中位数：10 万粒子 20–26
ms/步，1 万节点布料 2.3–3.2 ms/步，2 万节点 3.9–6.3 ms/步。

[![CI](https://github.com/flowinginthewind700/threedream/actions/workflows/ci.yml/badge.svg)](https://github.com/flowinginthewind700/threedream/actions/workflows/ci.yml)

![the demo, mid-training](docs/demo-drive.jpg)

## Demos

Five pages, deployed from `main` automatically once every gate in CI is green.
Each one exists to measure a claim rather than to illustrate it: the numbers in
the sidebars are read off the running engine.

| Page | Live | What it measures |
|---|---|---|
| Trainer | [threedream/](https://flowinginthewind700.github.io/threedream/) | A policy-gradient learner training in the page, on `DriveEnv` and `ReachEnv`. |
| Physics check | [physics-check.html](https://flowinginthewind700.github.io/threedream/physics-check.html) | One canonical scene through `builtin`, `wasm` and a wasm replay, digests compared in your own browser. |
| Shared device | [shared-device.html](https://flowinginthewind700.github.io/threedream/shared-device.html) | A single `GPUDevice` backing three.js rendering and a raw WGSL compute pipeline at the same time. |
| Particles | [particles.html](https://flowinginthewind700.github.io/threedream/particles.html) | 1k-100k particles: which tier the browser gave you, blit or CPU upload, draw calls, hash overflow. |
| Soft bodies | [soft.html](https://flowinginthewind700.github.io/threedream/soft.html) | Cloth / sheets / cube / rope up to 20k nodes: islands, color batches, dispatches a step, the race-free flag, max stretch. |

Every page carries the same nav strip, generated from the same page list the
build reads (`demo/pages.ts`), so arriving on any one of them shows the other
four with the page you are on marked current. The screenshots here and the share
cards are real captures of these pages, produced by `scripts/capture_shots.mjs`
against a build served at the Pages subpath; each capture is gated on the page's
own report first, so a shot that silently fell back to a worse tier fails the run
rather than shipping under a caption it does not earn.

![the particle page at 100,000 particles on the WebGPU tier](docs/demo-particles.jpg)

100,000 particles on the tier the browser actually granted. The badges are the
live report: `gpu-blit` means the frame was copied on the device and never
crossed the bus.

![the soft-body page with 20,000 nodes in four sheets](docs/demo-soft.jpg)

20,000 soft-body nodes in four sheets, solved in coloured constraint batches on
the same `GPUDevice` three.js renders with. The sidebar is the plan the two graph
passes produced, next to the backend's own `raceFree` flag.

Locally, `npm run dev` serves the same five at `http://localhost:5173/` and
`http://localhost:5173/{page}.html`.

## 演示页

一共五个页面，`main` 上所有 CI 关卡通过后自动部署。每个页面存在的理由都是把一条结论
量出来，而不是画个示意：侧栏里的数字都是从正在运行的引擎里读出来的。

| 页面 | 线上 | 量的是什么 |
|---|---|---|
| 训练页 | [threedream/](https://flowinginthewind700.github.io/threedream/) | 在页面里实时训练的策略梯度学习器，任务是 `DriveEnv` 与 `ReachEnv`。 |
| 确定性检查 | [physics-check.html](https://flowinginthewind700.github.io/threedream/physics-check.html) | 同一个规范场景跑 `builtin`、`wasm` 与一次 wasm 回放，在你自己的浏览器里比对摘要。 |
| 共享设备 | [shared-device.html](https://flowinginthewind700.github.io/threedream/shared-device.html) | 同一个 `GPUDevice` 同时支撑 three.js 渲染与一条裸 WGSL compute pipeline。 |
| 粒子 | [particles.html](https://flowinginthewind700.github.io/threedream/particles.html) | 1k–100k 粒子：浏览器实际给了哪个档位、走 blit 还是 CPU 上传、draw call 数、哈希溢出数。 |
| 软体 | [soft.html](https://flowinginthewind700.github.io/threedream/soft.html) | 布料 / 多片布 / 立方体 / 绳，最多 2 万节点：island 数、着色批次数、每步 dispatch 数、无竞争标志、最大拉伸。 |

每个页面都带同一条导航条，由构建读取的同一份页面清单（`demo/pages.ts`）生成，所以落在
任意一页都能看到其余四页，当前页会被标出来。这里与分享卡片用的截图都是这些页面的真实
抓取：`scripts/capture_shots.mjs` 在以 Pages 子路径提供的构建上拍摄，每张都先过页面自己
报告的那一关，于是真回退到更差档位的截图会让脚本失败，而不是顶着一句它配不上的说明发出去。

![10 万粒子、WebGPU 档位的粒子页](docs/demo-particles.jpg)

10 万粒子，跑在浏览器实际给出的档位上。角标读的是实时报告：`gpu-blit` 意味着这一帧在设备
上拷贝完成，没有过总线。

![2 万节点、四片布的软体页](docs/demo-soft.jpg)

2 万软体节点分成四片布，用着色的约束批次在 three.js 渲染所用的同一块 `GPUDevice` 上求解。
侧栏就是那两趟图分析产出的计划，旁边是后端自己的 `raceFree` 标志。

本地 `npm run dev` 提供同样五个页面：`http://localhost:5173/` 与
`http://localhost:5173/{page}.html`。

## Quickstart

```bash
npm install
npm run dev       # the five browser demos on http://localhost:5173
npm run train     # headless training in Node, prints a progress trace
npm test          # 1360 unit tests, ~21s
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

The two scale pages are the ones with pickers. `/particles.html` takes a tier
(`auto`, `webgpu`, `webgl2`, `cpu`), a count and a seed, and reports the tier it
actually landed on, whether the frame went through the GPU blit or a CPU upload,
the draw-call count, and how many particles failed to fit an already-full hash
bucket this step. `/soft.html` takes the same tier picker plus a scene (cloth,
sheets, cube, rope), a node count up to 20k, solver iterations and stiffness, and
reports the plan the two graph passes produced — islands, color batches, node
workgroups, dispatches a step — beside the backend's own `raceFree` flag and the
worst constraint stretch on screen. Iterations buy *reach*, not stiffness: a
correction travels about one row of the mesh per sweep, so a 10k cloth hung from
its top edge keeps stretching near the pinned row until it has had enough of
them, and the page says so in as many words.

演示页里直接训练。点 **Train**，看火花线上 mean episode return 上升，再把
**Playback** 从 `Random` 切到 `Learned`，就能看到它学到了什么。视口角标记录当前策略
背后有多少 episode。**Reach** 是两个任务里更难的那个。

两个规模页面是可以动手选的那两个。`/particles.html` 选档位（`auto`、`webgpu`、
`webgl2`、`cpu`）、粒子数与随机种子，报告它实际落在哪个档位、这一帧走的是 GPU blit
还是 CPU 上传、draw call 数，以及这一步有多少粒子没能挤进一个已经满了的哈希桶。
`/soft.html` 除了同一个档位选择器，还选场景（布料、多片布、立方体、绳）、最多 2 万个
节点、求解迭代次数与刚度，并把两趟图分析的产物 —— island 数、着色批次数、节点
workgroup 数、每步 dispatch 数 —— 连同后端自己的 `raceFree` 标志和屏幕上最坏的约束
拉伸一起报出来。迭代次数买的是**传播距离**，不是刚度：一次扫描只把修正推进大约一行，
所以从顶边挂下来的 1 万节点布料，在拿到足够多次扫描之前会一直在固定行附近拉伸，页面
上原话写着这件事。

## Why it is shaped this way

Two rules drive the design:

1. **The simulation is the only source of truth.** The renderer mirrors physics
   state into `THREE.Object3D`s each frame and never writes back. The same scene
   therefore runs headless for training and rendered for play, with identical
   results.
2. **Everything that matters is deterministic.** A seeded RNG, a fixed timestep,
   and no hidden global state mean `engine.step(n)` in Node and `engine.frame(dt)`
   in a browser produce the same simulation. That is what makes a physics-AI
   kernel testable: all 1360 unit tests run without a GPU, and the wasm backend is
   held to the bits of the TypeScript solver it ports. The two GPU layers are held
   to a CPU reference the same way, and neither claims determinism for itself:
   `deterministic` is false on both backends, because atomics promise no order, so
   training and replay always run on the reference tier.

两条规则决定了整体设计：

1. **仿真是唯一事实来源。** 渲染层每帧把物理状态镜像到 `THREE.Object3D`，从不回写。
   于是同一个场景既能无头训练，也能在浏览器里渲染游玩，结果完全一致。
2. **关键路径都是确定性的。** 带种子的 RNG、固定步长、没有隐藏的全局状态，所以 Node
   里的 `engine.step(n)` 与浏览器里的 `engine.frame(dt)` 跑的是同一个仿真。这让一个
   物理-AI 内核变得可测：1360 个单元测试全都不需要 GPU，而 wasm 后端要对齐它所移植的
   TS 求解器的每一个比特。两个 GPU 层用同样的方式对齐一份 CPU 参照，而且都不替自己
   声称确定性：两个后端的 `deterministic` 都是 false，因为原子操作不承诺顺序，所以
   训练与回放永远走参照档。

## Layers

```
core     clock / ECS / events / engine facade     no three.js, no WASM
physics  backend interface + three solvers        no three.js
gpu      probe, shared device, scale layers       no three.js, no WASM
ai       MLP, Gaussian policy, policy-gradient    no three.js, no WASM
envs     learning tasks (drive, reach)            physics only
render   three.js bridge + particle/soft views    the only layer importing three
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
The same layer owns the shared `GPUDevice` (`device.ts`, refcounted, with a
recovery path for device loss), the wrapper that lets external WGSL bind
three.js's own buffers (`compute.ts`), and the two scale layers — particles and
soft bodies. Each of those has a CPU reference implementation beside the GPU
one, so the WGSL can be checked against numbers rather than against a
screenshot.

`src/index.ts` 重导出**除** `render` **之外**的全部内容。引入这个入口不能把 three.js
带进训练脚本，所以浏览器代码直接引 `ThreeRenderer`。这是刻意的约束，不是遗漏。两个
WASM 后端**在**这个入口里，引入它们依然不花代价：各自的工厂函数内部用动态
`import()` 加载二进制，所以无头运行不会实例化它没要过的模块。

`gpu/` 是判断浏览器图形栈成色的那一层，而且是探测，不是构建开关：
`selectRenderTier` 接收 adapter 探测结果作为参数，返回 `webgpu`、`webgl2` 或
`cpu`。它内部不读 `import.meta.env`，所以回退路径不可能在编译期被写死。
这一层同时持有共享的 `GPUDevice`（`device.ts`，引用计数，带设备丢失后的恢复路径）、
让外部 WGSL 直接绑定 three.js 自己那些 buffer 的封装（`compute.ts`），以及那两层规模层
—— 粒子与软体。两者都在 GPU 实现旁边放着一份 CPU 参照实现，所以 WGSL 要对齐的是一组
数字，而不是一张截图。

### The scale layers

Neither GPU layer is a fast path with a CPU stub. `particleCpu.ts` sits beside
`particleGpu.ts` and `softCpu.ts` beside `softGpu.ts`, each pair sharing one
layout module and one set of constants, and the soft pair its uniform packer
too, so the WGSL is asserted against numbers. `deterministic` is false on both
GPU backends: atomics promise no order and a driver may contract `a * b + c`
into an fma, so training and replay stay on the reference tier. What the
soft-body layer can claim instead is `raceFree`, and getting there took two
graph passes.

两个 GPU 层都不是「快路径配一个 CPU 桩」。`particleCpu.ts` 与 `particleGpu.ts` 并排，
`softCpu.ts` 与 `softGpu.ts` 并排，每对读的是同一个 layout 模块与同一套常量（软体那对
还共用同一个 uniform 打包器），于是 WGSL 要对齐的是一组数字。两个 GPU 后端的
`deterministic` 都是 false：原子操作不承诺顺序，驱动也可能把 `a * b + c` 收缩成 fma，
所以训练与回放留在参照档。软体层能声称的是 `raceFree`，而为了它多出了两趟图计算。

`softIslands.ts` is a union-find that also emits the node order the kernels
dispatch against: each island's nodes consecutively, padded out to a multiple of
the 64-lane workgroup, so "is this island asleep" costs one load per workgroup
rather than one per node. `softColoring.ts` then colors the constraint graph
first-fit in ascending edge order — one u32 mask per node, `MAX_COLORS = 32`,
and a graph that would need more is refused rather than silently mis-colored. A
color is a set of constraints that share no node, so every write inside one
batch lands on a distinct node and the whole solve is race-free without a single
atomic. Both passes run identically on the two tiers, and `tests/soft_gpu.test.ts`
asserts the resulting `SoftPlan` field for field — that equality, not the naming,
is what "island grouping and constraint coloring pass a determinism check"
means.

`softIslands.ts` 是 union-find，同时产出 kernel 实际 dispatch 用的节点顺序：每个
island 的节点连续排布，并补齐到 64 lane workgroup 的倍数，于是「这个 island 睡着了
吗」是每个 workgroup 一次 load，而不是每个节点一次。`softColoring.ts` 再按边的升序做
first-fit 着色 —— 一个节点一个 u32 掩码，上限 `MAX_COLORS = 32`，需要更多颜色的图会被
拒绝，而不是被悄悄错着色。一个 color 是一组互不共享节点的约束，所以同一批里的每次写入
都落在不同节点上，整个求解无竞争，而且不用一个原子操作。两趟在两档上的算法完全相同，
`tests/soft_gpu.test.ts` 逐字段断言产出的 `SoftPlan` —— 「island 分组与约束着色通过
确定性对照」说的是这个相等，不是命名。

A step is `5 + iterations * colors` dispatches: 69 for a cloth at 8 iterations
and 8 colors, a number that follows the graph's maximum degree and not the node
count. `publish` writes into a buffer three.js already allocated, and
`render/soft.ts` blits it straight into the mesh's position attribute — a plain
`BufferAttribute(itemSize=3)`, neither a storage attribute nor
`DynamicDrawUsage`, because the first would break the byte-for-byte
correspondence and the second would overwrite it with a stale CPU array every
frame. 10k nodes cost 1.40 MiB and 2.3-3.2 ms/step, 20k cost 2.81 MiB and
3.9-6.3, and 1k costs 0.7-1.6: the ladder is about 0.7 ms of fixed cost — 69
dispatch submissions and a queue flush, which the node count does not move —
plus roughly 0.16 us a node, so the bottom rung is nearly all submission and
the top one is mostly solve. Each number is the median of the 20 timed chunks a
160-step rung produces, printed with its p95 beside it, because the mean over
the four chunks a 30-step rung used to produce was a number one contended chunk
could triple: the same rung on the same machine read 2.97 and then 10.07
ms/step four minutes apart, in a run whose 20k rung came out *faster* than its
10k one. Those are the numbers `scripts/bench_gpu_soft.mjs` exists to print,
and they are a floor in device terms — headless Chromium hands
`requestAdapter()` whichever GPU the driver stack prefers, which on a laptop
with no Vulkan ICD for the discrete card is the integrated one — and not a
floor from one run to the next, which is what the p95 column is there to show.

一步是 `5 + iterations * colors` 个 dispatch：8 次迭代、8 个 color 的布料是 69 个，
这个数字跟着图的最大度数走，不跟节点数走。`publish` 写进 three.js 已经分配好的那块
buffer，`render/soft.ts` 把它直接 blit 进 mesh 的 position attribute —— 普通的
`BufferAttribute(itemSize=3)`，既不是 storage attribute 也不是 `DynamicDrawUsage`，
因为前者会破坏逐字节对应，后者会每帧用陈旧的 CPU 数组把刚 blit 进去的位置盖掉。
1 万节点 1.40 MiB、2.3–3.2 ms/步，2 万节点 2.81 MiB、3.9–6.3 ms/步，1k 是
0.7–1.6 ms/步：整条阶梯约等于 0.7 ms 的固定开销（69 次 dispatch 提交加一次队列
flush，节点数推不动它）加上每节点约 0.16 us，所以底部那一档几乎全是提交，顶部那一档
才主要是求解。每个数字都是 160 步一档产出的 20 个计时 chunk 的中位数，旁边同时打印
p95 —— 因为以前 30 步一档只有四个 chunk，而四个样本的均值是一个被抢占的 chunk 就能
翻三倍的数字：同一档在同一台机器上相隔四分钟分别读到 2.97 与 10.07 ms/步，那一轮里
20k 还比 10k「更快」。这些正是 `scripts/bench_gpu_soft.mjs` 要打印的数字，它们在设备
意义上是下限（无头 Chromium 会把 `requestAdapter()` 交给驱动栈偏好的那块 GPU，在一台
独显没有 Vulkan ICD 的笔记本上，那就是集显），但在轮次与轮次之间不是 —— 而那正是 p95
那一列要显示的东西。

The tier below that one has a number too, because "the fallback works" and "the
fallback is usable at scale" are two different claims and only the second one is
an acceptance criterion. `scripts/bench_cpu_soft.ts` runs the same ladder, the
same scene, seed and iteration count through `src/gpu/softCpu.ts`, which is the
shipped fallback rather than a restatement of it, and the same code the `webgl2`
and `cpu` tiers both simulate on. It reads 1.5 / 7.9 / 15.9 / 32.1 ms/step at
1k / 5k / 10k / 20k nodes: about 1.6 us a node with no fixed cost worth naming,
roughly ten times the device tier's marginal 0.16 us. So 10k nodes cost a whole
60 Hz frame before anything is drawn, and the fallback's stated target is ~2k
nodes at 60 Hz (about 3.2 ms/step, which leaves the rest of the frame for
drawing and for the engine) rather than the 20k the device tier holds. Its p95
sits within a percent of its p50, and printing both is the point: with no driver
submission and no queue flush inside the timed region, that spread is the
machine's own noise floor, so whatever is wider than it in the GPU table above
belongs to the device path.

下面那一档同样有数字，因为「回退能用」与「回退在规模上可用」是两条不同的结论，而只有
第二条是验收项。`scripts/bench_cpu_soft.ts` 用同一条阶梯、同一个场景、同一个 seed 与
同样的迭代次数跑 `src/gpu/softCpu.ts`：那是实际交付的回退路径，不是它的复述，也正是
`webgl2` 与 `cpu` 两档共用的那份仿真代码。它在 1k / 5k / 10k / 20k 节点上读到
1.5 / 7.9 / 15.9 / 32.1 ms/步，每节点约 1.6 us，固定开销小到不值得命名，大约是设备档
每节点 0.16 us 的十倍。于是 1 万节点在还没开始画之前就已经吃掉一整个 60 Hz 帧，回退档
给出的明确目标是 60 Hz 下约 2k 节点（约 3.2 ms/步，把这一帧剩下的预算留给绘制与引擎
其余部分），而不是设备档站得住的 2 万。它的 p95 与 p50 相差在百分之一以内，而把两个都
打出来正是重点：计时区间里没有驱动提交也没有队列 flush，这个离散度就是机器本身的噪声
底，所以上面那张 GPU 表里比它更宽的部分，属于设备路径。

The most expensive lesson in the layer is one the unit tests could not catch.
The color selector originally read `global_invocation_id.z`, on the theory that
a dispatch dimension is a free channel into the kernel. It is not: dispatch
dimensions are concurrent rather than sequential, so `(x, 1, color + 1)` runs
every color at once and re-creates exactly the race the coloring exists to
remove, while the last color never solves at all. The shader compiled, the
dispatches succeeded, the mesh moved, and the unit tests drive the CPU
reference, so no tolerance ever complained. What caught it was a CPU/GPU
comparison on a real device, which is why `e2e/soft_gpu.spec.ts` exists. The
selector now hangs off a binding: each color owns one 256-byte slot of
`batchBuf` and sees only its own 8 bytes, so the kernel reads `batchBuf[0u]` and
cannot index a neighbour's slot even if the shader text is wrong.

这一层最贵的一课，单元测试抓不到。color 选择器原本读 `global_invocation_id.z`，理由是
「dispatch 维度是通往 kernel 的一条免费通道」。它不是：dispatch 的维度是并发的，不是
顺序的，于是 `(x, 1, color + 1)` 会同时跑所有 color，把着色本该消除的那个竞争原样重建
出来，而最后一个 color 根本不会被解。shader 编译通过、dispatch 成功、网格在动，单元
测试驱动的是 CPU 参照，所以没有任何容差会报警。抓到它的是真设备上的一次 CPU/GPU 对照
—— 这正是 `e2e/soft_gpu.spec.ts` 存在的理由。现在选择器挂在 binding 上：每个 color
独占 `batchBuf` 里一个 256 字节的槽位，只看得见自己那 8 字节，于是 kernel 读
`batchBuf[0u]`，即使 shader 文本写错也索引不到邻居的槽位。

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
| `npm test` | 1360 unit tests, headless, no GPU needed | ~21s |
| `npm run test:coverage` | same suite under v8, floor enforced by `vitest.config.ts` | ~21s |
| `npm run test:e2e` | 50 Playwright tests over 7 specs, two projects: SwiftShader WebGL2 and ANGLE/Vulkan WebGPU | ~50s |
| `npm run test:rust` | 68 native Rust tests for the solver | ~1s warm |
| `npm run check:wasm` | 35 assertions over the committed `wasm/pkg`: ABI, provenance, behaviour | ~1s |
| `node scripts/bench_gpu_particles.mjs` | the M3 ladder at 1k/10k/50k/100k particles, 160 steps a rung: per-step cost as p50/p95/mean over 20 chunk samples, draw calls, blit size | ~6s |
| `node scripts/bench_gpu_soft.mjs` | the M4 ladder at 1k/5k/10k/20k nodes, 160 steps a rung: the same distribution, plus dispatches, colors and stretch | ~3s |
| `npx tsx scripts/bench_cpu_soft.ts` | the same M4 ladder on the fallback tier: `softCpu.ts`, single-threaded, no GPU, p50/p95 a step at 1k-20k nodes and a fitted us/node | ~10s |
| `node scripts/capture_shots.mjs` | the screenshots in this file and the share cards: 8 captures of the built pages served at the Pages subpath, each gated on the page's own tier report | ~40s |

The two e2e projects exist because the pages need two different GPUs. `demo`,
`physics-check`, `particles` and `soft` only need *a* GL context, and headless
Chromium has none, so they run on SwiftShader: that the render layer works
without hardware is the property worth testing. `shared-device` and the GPU
halves of `particles` and `soft` need a real `GPUDevice`, which means ANGLE's
Vulkan backend with the WebGPU service enabled. Same browser, different flags,
and a flag set that works for one silently downgrades the other. The six tests
that assume *no* adapter skip themselves on a machine that has one, rather than
reporting a tier the machine did not produce.

Coverage is a separate command, not a flag on `npm test`: instrumenting the
training inner loop costs ~10x wall time, and folding that into the fast loop
would destroy the red/green cadence the tests exist to provide. So the coverage
run skips the two convergence tests (`COVERAGE=1`, `tests/trainer.test.ts`) that
cost 180s of the 185s and contribute ~0.3% of branch coverage, plus the one
soft-body scale spec (`tests/soft_cpu.test.ts`) whose assertion is a wall clock
that instrumented code cannot meet — `npm test` still runs all three, and
`tests/tdd.test.ts` pins that asymmetry. The floor sits a few points under what
the suite actually measures (93/82/93/94 against 98.7/93.5/98.9/99.1), which is
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

Byte-exactness needs one more thing pinned, and it is not in `Cargo.toml`:
wasm-pack fetches the `wasm-bindgen` *CLI* itself, and a prebuilt release asset
stamps the artifact's `producers` section with the tag's git hash where a
`cargo install` fallback stamps the bare version. Same CLI version, identical
behaviour, 12 different bytes, which the freshness check cannot tell apart from
real drift. `scripts/check_wasm_artifact.mjs` therefore pins the stamp
(`WASM_BINDGEN_CLI`), so a source-built CLI fails provenance with the reason
printed instead of turning up as an unexplained hash mismatch.

本项目采用测试先行：`src/` 下每个模块都有对应 spec，一旦出现没有测试的新模块，
`tests/tdd.test.ts` 会让构建失败。

覆盖率是独立命令而非 `npm test` 的参数：给训练内层循环插桩会让墙钟时间涨约 10 倍，
把它塞进快速循环会毁掉测试本该提供的红/绿节奏。因此覆盖率运行会跳过那两个收敛测试
（`COVERAGE=1`，`tests/trainer.test.ts`）—— 它们占了 185s 里的 180s，却只贡献约 0.3%
的分支覆盖率 —— 外加软体那一条规模 spec（`tests/soft_cpu.test.ts`），它断言的是一个
插桩后的代码无法满足的墙钟；`npm test` 三条照跑，这条不对称由 `tests/tdd.test.ts` 钉死。
下限压在实测值之下几个点（实测 98.7 / 93.5 / 98.9 / 99.1，下限 93 / 82 / 93 / 94）——
这才是关键：远低于现状的阈值只是装饰，而贴着现状设阈值则会让每次重构都变成搏斗。

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

要让「字节完全一致」成立，还有样东西必须钉住，而它不在 `Cargo.toml` 里：wasm-pack 会
自己去取 `wasm-bindgen` *CLI*，预编译的 release 产物在 `producers` 段里打上 tag 的 git
哈希，`cargo install` 兜底则只打版本号。CLI 版本相同、行为完全相同，字节却差 12 个 ——
新鲜度检查分不出这与真正的漂移。所以 `scripts/check_wasm_artifact.mjs` 把这个印记也钉住
（`WASM_BINDGEN_CLI`）：源码编译出来的 CLI 会在 provenance 关卡失败并打印原因，而不是
表现为一次无法解释的哈希不匹配。

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
src/core/      clock.ts digest.ts ecs.ts engine.ts events.ts rng.ts
src/physics/   types.ts builtin.ts wasm.ts rapier.ts reference.ts components.ts
src/gpu/       capabilities.ts device.ts compute.ts particle*.ts soft*.ts
src/ai/        mlp.ts policy.ts trainer.ts baseline.ts
src/envs/      types.ts drive.ts reach.ts
src/render/    scene.ts particles.ts soft.ts
rust/          physics (solver) / physics-wasm (ABI) / gpu (wgpu skeleton)
wasm/pkg/      committed wasm-pack output, rebuilt by `npm run build:wasm`
scripts/       train_headless.ts check_wasm_artifact.mjs bench_*.mjs
               capture_shots.mjs audit_unreal_reference.mjs
               clone_unreal_reference.sh
docs/          feasibility study, development plan, demo assets
demo/          index (trainer), physics-check, shared-device, particles, soft:
               each one a .html + .ts pair, plus pages.ts and nav.ts (the shared
               nav) and public/ (favicon, share cards)
tests/         42 files, 1360 tests
e2e/           demo, wasm physics, particles, soft (WebGL); shared device and
               the GPU halves of particles and soft (WebGPU)
.github/       ci.yml: four gates (verify / coverage / e2e / rust) then Pages deploy
thirdparty/    UnrealEngine (submodule, opt-in)
```

## License

MIT. See [LICENSE](LICENSE).

MIT，见 [LICENSE](LICENSE)。
