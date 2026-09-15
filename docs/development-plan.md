# ThreeDream development plan

日期：2026-09-14。  
状态：主动迭代中。  
关联调研：[Rust + wasm + WebGPU + three.js](./feasibility-rust-wasm-webgpu.md)。

## 目标

做一个在浏览器里能跑、能训练、能回放、能扩展到大规模仿真的 three.js 游戏与物理-AI
内核。它不是 Unreal 的移植，而是以 Unreal 的架构为参考，重新实现一层浏览器原生的
**确定性 CPU 内核 + GPU 规模层**。

核心承诺：

1. **确定性**：同一 seed、同一输入、同一版本，在 Node、浏览器、回放中结果一致。
2. **可插拔**：物理、渲染、AI 都通过稳定接口替换，不把调用方绑死在一个后端上。
3. **规模**：CPU 层保证正确与可复现，GPU 层承担粒子、软体、布料、群体等大规模仿真。
4. **浏览器原生**：three.js 做渲染，WebGPU 做计算主线，WebGL2 做兼容回退。

## 非目标

以下事情刻意不做，避免把内核做成大而不可控的通用引擎：

- 不移植 Unreal 源码，不提交 Epic 代码，不做“兼容 UE 资产”的承诺。
- 不做完整编辑器、资源市场、网络同步或跨进程多人游戏。
- 不把 GPU 层伪装成确定性后端；训练和回放继续走 CPU 确定性路径。
- 不追求在所有浏览器上启用全部特性；能力探测与回退是产品的一部分。

## 当前基线

仓库已经有一个能跑、能测、能部署的内核：

- 882 个单元测试 + 68 个原生 Rust 测试 + 30 个浏览器测试（`chromium` 与
  `chromium-webgpu` 两个 project），`npm run verify` 全绿。
- GitHub Actions 五任务四关卡：类型检查与构建、覆盖率、浏览器渲染、Rust 内核与
  wasm 产物校验。
- Pages 自动部署，线上 demo 可用：训练页、`physics-check`、`shared-device`、
  `particles`。
- 确定性 ECS、固定步长引擎、事件总线。
- `PhysicsBackend` 抽象，已有 `builtin`、`wasm`（自研 Rust 内核）与 `rapier` 三个实现。
- `src/gpu/`：运行时能力探测与渲染档位选择（webgpu / webgl2 / cpu）、引用计数的
  共享 `GPUDevice` 管理器、外部 WGSL 的 compute 封装。
- 粒子层：CPU 参照实现、WebGPU 每步最多六个 dispatch、设备上的实例展开，一次 draw
  call
  喂 three.js；100k 粒子 29.5 ms/步（iGPU，ANGLE/Vulkan，无回读）。
- 带策略梯度的强化学习器，可在页面内训练。
- three.js 渲染桥，渲染层不回写仿真状态。

可行性调研已经证明：

- Unreal 源码必须 clean-room 重写，不能直接提取。
- three.js 与外部 WGSL 可以共享同一个 `GPUDevice` 并直接绑定 buffer，零拷贝可行。
- Rust → wasm 的真实收益约 2.3–2.4x，可作为确定性 CPU 层的性能基础。
- GPU 在大 N 场景有明确规模优势，但共享内存 iGPU 的绝对值不能当规格。
- GitHub Pages 不提供 COOP/COEP，`SharedArrayBuffer` 不可用，因此 wasm 多线程在这条
  部署路径上不可行。

## 目标架构

```text
                    +----------------------+
                    |  Game / AI / Env API |
                    +----------+-----------+
                               |
                    +----------v-----------+
                    |  Engine facade       |
                    |  ECS + fixed clock   |
                    +----------+-----------+
                               |
              +----------------+----------------+
              |                                 |
   +----------v-----------+          +----------v-----------+
   | CPU deterministic    |          | GPU scale backend    |
   | Rust wasm + SIMD128  |          | WebGPU / WGSL        |
   | rigid body/training  |          | particles/soft body  |
   +----------+-----------+          +----------+-----------+
              |                                 |
              +----------------+----------------+
                               |
                    +----------v-----------+
                    |  three.js renderer   |
                    |  shared GPUDevice    |
                    |  WebGL2 fallback     |
                    +----------------------+
```

分层约束：

- `core/` 不依赖 three.js、WebGPU 或 WASM。
- `physics/` 只暴露 `PhysicsBackend`，不暴露后端细节。
- `render/` 是唯一允许直接依赖 three.js 的层。
- GPU 后端如实声明 `deterministic: false`。
- WebGL2 回退必须是运行时能力探测的结果，不是编译期分支。

## 里程碑

### M0：把计划变成可执行基线

状态：已完成（2026-09-15）。

时间：1 周。

任务：

1. 建立本计划文档，并在 README 中链接。
2. 建 Rust workspace 骨架，锁 wgpu v25，目标 `wasm32-unknown-unknown`。
3. 增加 WebGPU 能力探测：adapter、feature set、limits、兼容模式。
4. 把 `bench_shared_device.mjs` 的三条断言纳入浏览器测试。
5. CI 增加 Rust 构建与 wasm 产物校验。

验收：

- `npm run verify` 全绿。
- Rust workspace 能构建出 wasm。
- 浏览器测试能证明共享 `GPUDevice`、共享 buffer、外部 WGSL 绑定三条事实。

落地证据：

- `rust/` 三个 crate：`physics`（求解器，不依赖 wasm-bindgen）、`physics-wasm`
  （cdylib，只有 ABI）、`gpu`（wgpu 骨架）。wgpu 锁 `=25.0.2`，工具链锁 `1.98.1`，
  `tests/rust_workspace.test.ts` 钉住这些锁定与 release profile。
- `npm run build:wasm:gpu` 证明锁定的 wgpu 能为 `wasm32-unknown-unknown` 编译，
  CI 的 `rust` 关卡每次都跑它。
- `src/gpu/capabilities.ts` 做运行时探测（adapter、feature set、limits，以及
  `core` 与 `compatibility` 两个 feature level）；`selectRenderTier` 只接收探测
  结果，所以回退不可能是编译期分支。Node 侧由 `tests/gpu_capabilities.test.ts`
  用 stub adapter 覆盖。
- `bench_shared_device.mjs` 的三条断言现在是页面加测试：`demo/shared-device.html`
  与 `e2e/shared_device.spec.ts`，后者在 CI 的 `chromium-webgpu` project 下用
  ANGLE/Vulkan 拿真实 `GPUDevice`。

### M1：确定性 Rust wasm 物理内核

状态：已完成（2026-09-15）。

时间：2–3 周。

任务：

1. 将 `builtin` 物理求解器的核心路径移植为 Rust。
2. 使用 `wasm-bindgen` 暴露稳定 ABI，不泄漏 Rust 类型。
3. 开启 SIMD128，保持固定步长与确定性约束。
4. 在 TS 侧新增 `WasmPhysicsBackend`，继续实现 `PhysicsBackend`。
5. 建立与 `builtin` 的位级一致性测试和性能回归基准。
6. 保留现有 TS 后端作为规范实现与回退。

验收：

- 同一场景、同一 seed，`builtin` 与 wasm 后端状态逐位一致或误差在明确的舍入契约内。
- 关键基准不低于 TS 版本 2x。
- Node、浏览器、回放三处结果一致。
- 所有现有环境与训练任务不改调用方即可切换后端。

落地证据：

- `rust/crates/physics` 是 `src/physics/builtin.ts` 的移植，不依赖 wasm-bindgen，
  所以 `npm run test:rust`（68 个测试）能在原生侧检查求解器自身的不变量。
- ABI 只传数字：world id、body handle，以及调用方用 `td_alloc` 分配的 f64
  缓冲区。没有字符串、结构体或 `Result`；错误以负数哨兵返回，并在 TS 侧翻译成
  与 `BuiltinPhysics` 完全相同的 `Error` / `RangeError`。
- `WASM_ABI_VERSION`（TS）与 `ABI_VERSION`（Rust）由 `tests/rust_workspace.test.ts`
  双向钉住；`scripts/check_wasm_artifact.mjs` 再把 `WasmBindings` 与二进制导出表
  双向比对，并检查产物确实带 `simd128`、由锁定工具链构建、不含泄漏的本地路径。
- 位级一致性走摘要而不是走两两比较：`src/physics/reference.ts` 定义规范场景与
  基于 IEEE-754 原始比特的 `REFERENCE_GOLDEN_DIGEST`，Node
  （`tests/wasm_backend.test.ts`）、浏览器（`demo/physics-check.html` 加
  `e2e/wasm_physics.spec.ts`）与同一 seed 的回放三处都必须复现它。
- 性能门槛设在 2x，实测 400 体场景 2.58x、60 体场景 4.85x；两个基准场景同时被
  检查逐位一致，避免「测的不是跑的那份计算」。
- 调用方零改动：`ReachEnv` / `DriveEnv` 只认 `PhysicsBackend`，`physics-check`
  页面可实时切换后端跑同一场景。`builtin.ts` 保留为规范实现与回退。

### M2：共享 GPUDevice 的 WebGPU 桥

状态：已完成（2026-09-15）。

时间：1–2 周。

任务：

1. 把 `requestDevice()` 与 `WebGPURenderer` 统一为单例设备管理器。
2. 对外暴露 buffer 句柄，而不是复制数据。
3. 增加外部 WGSL compute pipeline 的通用封装。
4. 建立能力降级矩阵：WebGPU、WebGL2、纯 CPU。
5. 加入错误恢复与设备丢失处理。

验收：

- 渲染与计算共享同一个 `GPUDevice`。
- three.js 管理的 buffer 可被外部 WGSL 直接读写。
- 断言脚本纳入 CI。
- 设备不可用时自动回退，不出现白屏或静默失败。

落地证据：

- `src/gpu/device.ts`：`SharedDeviceManager` 引用计数地持有唯一一个 `GPUDevice`，
  `acquireSharedDevice` 是唯一入口；device lost 之后 manager 记下 `failure` 并允许
  下一次 acquire 重建设备，`tests/gpu_device.test.ts` 用一个会丢设备的 stub 覆盖
  这条路径。
- `src/gpu/compute.ts`：`ComputeBuffer` / `ComputeProgram` / `ComputeContext` /
  `PingPong` / `submitCopy`，即「外部 WGSL compute pipeline 的通用封装」；编译失败
  抛 `ShaderCompilationError` 并带上浏览器给出的消息，不是静默降级。
- 零拷贝的 buffer 句柄：`renderer.backend.get(attribute).buffer` 直接拿到 three.js
  自己创建的 `GPUBuffer`，`src/render/particles.ts` 用它做 blit 目标。
- 降级矩阵：`selectRenderTier` 只吃探测结果，`probeParticles` 在其上给出粒子层的
  档位与理由；WebGPU -> WebGL2 -> CPU 全部是运行时判断。
- CI：`e2e/shared_device.spec.ts` 在 `chromium-webgpu` project 下用 ANGLE/Vulkan 拿
  真实设备，把那三条断言每次 push 都跑一遍。

### M3：GPU 粒子层

状态：已完成（2026-09-15）。

时间：2 周。

任务：

1. 实现 GPU 粒子状态容器与 ping-pong buffer。
2. 实现 n-body、重力、碰撞反弹、边界约束等基础 kernel。
3. 增加空间哈希 broadphase 与原子计数。
4. 用 three.js `InstancedMesh` 渲染，不回写仿真状态。
5. 提供 CPU 与 WebGL2 回退路径。
6. 增加视觉测试与性能基准。

验收：

- 在目标硬件上稳定运行 50,000–100,000 粒子。
- WebGPU 不可用时自动降级到 WebGL2 或 CPU。
- 回放与训练不依赖 GPU 层。
- 渲染帧率与仿真步长解耦。

落地证据：

- 状态容器与 ping-pong：`src/gpu/particleGpu.ts` 用 `PingPong` 持有粒子 SoA，一步是
  `nbody? -> hash_clear -> hash_scatter -> collide? -> integrate -> publish`，与
  `CpuParticleSystem.step` 逐 pass 对齐，步内不回读。没有前缀和：broadphase 用固定
  容量的桶加 `atomicAdd`，溢出计进 `statsBuf` 而不是被悄悄丢掉。
- kernel：`src/gpu/particleWgsl.ts` 从 CPU 常量生成 WGSL，所以 n-body、重力、碰撞
  反弹、边界约束只有一份数值定义；`tests/particle_wgsl.test.ts` 钉住 params uniform
  的偏移与成员顺序、六个入口点的顺序、以及 workgroup 恒为 64 且不用 subgroup，
  96 字节这个总数由 `tests/particle_gpu.test.ts` 的 buffer 预算断言钉住。
- broadphase：`src/gpu/particleHash.ts` 是空间哈希（原子计数 + 桶容量），CPU 与
  WGSL 共用同一套 cell 数学，`hashOverflow` 是报告字段而不是断言。
- 渲染不回写：`src/gpu/particleInstances.ts` 在设备上把粒子展开成 mat4，
  `src/render/particles.ts` 把它 blit 进 three.js 的 `instanceMatrix`。关键是那个
  attribute 必须是 `StorageInstancedBufferAttribute`：普通的
  `InstancedBufferAttribute` 会被 `createInstanceMatrixNode` 包进
  `InstancedInterleavedBuffer`，`backend.get()` 拿不到底层 `GPUBuffer`，于是整条
  GPU 路径静默退化成 CPU 上传。`tests/render_particles.test.ts` 覆盖这条退化与
  「拿不到 buffer 就大声失败」的守卫。
- 规模：`scripts/bench_gpu_particles.mjs` 是阶梯基准（1k/10k/50k/100k），实测
  1.65 / 2.59 / 11.84 / 29.55 ms/步，每档 3 个 draw call、escaped 0；per-particle
  成本在 0.24–0.30 us 之间，是 O(n) 而不是撞墙。
- 规模（实机 rAF 路径）：脚本化基准证明的是 kernel 扛得住，不是访客看到的那条路
  扛得住。`demo/particles.html?tier=webgpu&strict=1&count=100000&collisions=1`
  在真设备的实时循环下实测 22.9 fps、`frameMode=gpu-blit`、每帧 blit 6,400,000
  字节（100k × mat4）、3 个 draw call、`gpuError=null`、escaped 0。同一档下
  `hashOverflow` 是 416：416 个粒子这一步没能插进已满的桶，于是漏掉这一步的接触
  检测。它是页面上的报告字段而不是失败——桶满了这件事必须看得见，不能被悄悄丢掉。
- 回退：`demo/particles.html` 支持 `tier=auto|webgpu|webgl2|cpu` 与 `strict=1`；
  `e2e/particles.spec.ts` 在没有 WebGPU 的 `chromium` project 下跑 WebGL2 与 CPU 档，
  并断言 CPU 档同一 seed 两次跑出同一个 digest（格式 `hex:count`）。
  `e2e/particles_gpu.spec.ts` 在真设备下断言 blit 路径。GPU 与 CPU 的对照是带容差
  的，不是逐位相等：碰撞顺序与哈希插入都用原子操作，两个重叠粒子谁先被推开是硬件
  没承诺过的，所以比对的是动能与速度包络的相对误差、接触数、escaped 与
  hashOverflow。确定性住在 CPU 档里。
- 帧率与步长解耦：`ParticleRunner.frame(dt)` 按固定步长切片，`demo/particles.ts`
  里另有一条 scripted 路径专门给基准与 e2e 用。
- 训练与回放不碰 GPU 层：粒子层只从 `src/index.ts` 导出纯 TypeScript 部分，
  `render/` 依旧不在 barrel 里。

### M4：GPU 规模物理层

时间：3–4 周。

任务：

1. 实现 island 分组与 workgroup 映射。
2. 实现约束图着色分批，避免数据竞争。
3. 增加软体、布料、质点弹簧等大规模仿真。
4. 保持 workgroup size 为 64，不依赖 subgroup。
5. 加入确定性 CPU 参照实现，用于验证物理语义。
6. 增加性能预算与内存预算。

验收：

- GPU 层能稳定处理 10,000 级别的软体或布料粒子。
- island 并行与约束着色通过确定性对照测试。
- CPU 层不因 GPU 层引入而变得不可测。
- WebGL2 回退路径可用，性能目标明确降级。

### M5：AI 与游戏层整合

时间：2–3 周。

任务：

1. 优化 ECS 查询，为批量观测与动作传输准备 SoA 布局。
2. 将 GPU 规模层接入环境与奖励信号。
3. 保持训练循环完全 headless。
4. 增加策略推理路径，优先 CPU/wasm，后续再评估 GPU 推理。
5. 建立训练、回放、可视化的统一 API。

验收：

- 训练仍可在 Node 中无浏览器运行。
- 同一策略在训练与浏览器演示中表现一致。
- GPU 规模层可用于环境观测，但不破坏确定性回放。
- AI 层不引入对 three.js 或 WebGPU 的直接依赖。

### M6：生产化硬化

时间：1–2 周。

任务：

1. 建立浏览器兼容矩阵：Chrome、Edge、Firefox、Safari、Firefox Android。
2. 完善 WebGL2 回退与错误提示。
3. 建立性能、内存、包体积预算。
4. 增加设备丢失、上下文丢失、tab 切换、移动端降级测试。
5. 完善文档、示例和版本化发布。

验收：

- CI 覆盖至少 Chrome 与一个回退浏览器。
- 主路径与回退路径都有可运行的 demo。
- 包体积与首屏时间有明确预算且被测试守住。
- 发布流程可重复，不依赖手工步骤。

## 执行顺序的理由

1. **先锁确定性**：如果 CPU 层不确定，后面的训练、回放和测试都会失去地基。
2. **再打通 GPU 桥**：共享 `GPUDevice` 是整个 WebGPU 层的架构前提。
3. **先做粒子**：粒子没有 island 与约束图，是最短的 GPU 规模验证路径。
4. **再做物理规模层**：等 GPU 桥和粒子层稳定后，才引入刚体/软体的复杂依赖。
5. **最后整合 AI**：AI 需要稳定的环境契约，不能在物理接口还在变化时提前固化。

## 风险与对策

| 风险 | 影响 | 对策 |
|---|---|---|
| WebGPU 覆盖不足 | 部分用户无法使用 GPU 层 | WebGL2 与 CPU 回退必须在同一版本落地 |
| three.js 版本升级破坏共享设备 | 计算与渲染无法互通 | 锁版本，升级时先跑共享设备断言 |
| wgpu API 漂移 | Rust 侧编译与运行不稳定 | 锁 v25，升级作为独立任务评估 |
| iGPU 性能波动 | 基准不可重复 | 使用中位数与多轮分布，不把单机值当规格 |
| GPU 确定性缺失 | 训练与回放不可复现 | GPU 层声明 `deterministic: false`，训练走 CPU |
| Pages 无 SAB | wasm 多线程不可用 | 先接受单核约束；若必须多核，再迁移托管 |
| 内核范围膨胀 | 变成不可维护的大杂烩 | 每个里程碑都有明确非目标与验收门 |

## 决策门

- M1 未达到确定性验收前，不开始 M4。
- M2 未通过共享 `GPUDevice` 断言前，不接入任何 GPU 物理。
- M3 未达到粒子规模目标前，不承诺更复杂的软体/布体指标。
- WebGL2 回退缺失时，不发布只支持 WebGPU 的版本。
- 任何提交都不允许引入 Epic 源码或许可污染。

## 立即行动

M0 到 M3 已落地，证据见各里程碑下的「落地证据」。接下来按 M4 推进：

1. 给 `npm run train` 加 `--backend wasm`：这是 M2 遗留项，env 早就接受 `backend`
   选项，缺的只是脚本入口；补上之后无头训练才能用上这个内核。
2. island 分组与 workgroup 映射：M4 的第一块，先做一个确定性的 CPU 参照分组器，
   再把它搬到设备上，顺序与粒子层一致。
3. 约束图着色分批，避免同一批 dispatch 里的数据竞争；workgroup size 固定 64，
   不依赖 subgroup。
4. 软体/布料/质点弹簧的最小 kernel，规模目标 10,000 级。
5. 性能与内存预算：`scripts/bench_gpu_particles.mjs` 的阶梯形式可以直接复用，
   M4 需要一份等价的基准与一条 CI 里跑得动的门槛。
