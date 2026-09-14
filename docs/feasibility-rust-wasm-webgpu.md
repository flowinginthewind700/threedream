# Rust + wasm + WebGPU + three.js: feasibility findings

调研日期 2026-09-14。所有数字都在本机实测，脚本随仓库提交，可复现。

本报告引用 Unreal 源码树的每一个计数，都由 `node scripts/audit_unreal_reference.mjs`
重新测量并输出成同一张表——**改报告前先跑它，不要手抄**。第一版草稿正是因为
手抄了上一轮不同口径（不同正则、不同文件过滤）的数字而错了大半：一个"计数"只有
配上产生它的 pattern 与文件集才有意义。该脚本同时把三条决定架构的结论变成断言
（全部 Epic 版权 / 零 OSI 许可头 / Chaos 无 GPU 路径），不满足则退出非零。

## 结论先行

方向成立，但有三处必须改写。

1. **"提取 Unreal 内核" 不成立**，法律上和工程上都不成立。改为 **精读 → clean-room
   重写**：submodule 只作阅读参考，仓库里不出现一行 Epic 源码。
2. **GPU 物理是"规模主线"**。小 N 时它甚至比 CPU 慢；大 N 时中位吞吐已明显超过
   CPU（N=8192 约 3.4x，N=16384 约 8.3x），但共享内存 iGPU 的绝对值极不稳定，
   不能当规格引用。刚体接触求解的确定性内核仍留在 CPU，GPU 承担
   *并行规模*（粒子、软体、群体、大规模视觉表现）。
3. **wasm 多线程在 GitHub Pages 上不可能**。Pages 不发 COOP/COEP，`SharedArrayBuffer`
   为 `undefined`。要么接受单核，要么换托管。

已经验证成立的，是整套架构最关键的一环：**计算与渲染可以共用一个 `GPUDevice`，
并且外部裸 WGSL 可以直接绑定 three.js 自己管理的 buffer，零拷贝**。

## 1. 法律边界：816 个文件，0 个开源许可头

Chaos（`Engine/Source/Runtime/Experimental/Chaos`）实测：

| 项目 | 数量 |
|------|------|
| 全部文件 | 847 |
| 代码文件（`*.h` / `*.cpp` / `*.inl`） | 816 |
| 代码文件总行数 | 294,203 |
| 带 `Copyright Epic Games` 的代码文件 | **816 / 816（100%）** |
| `Copyright Epic Games, Inc. All Rights Reserved` 全串出现次数 | 840（含全部 847 文件） |
| 带 SPDX / MIT / Apache / BSD 许可头 | **0** |

口径说明：上表的"文件数"一律指 **816 个代码文件**（过滤掉 31 个 `.Build.cs` /
`.uplugin` / 文档）；"出现次数"是跨全部 847 个文件的匹配总数。两个口径不能混用，
混用正是第一版草稿里 `847 处命中` 与 `199 个文件` 都不可复现的原因。

`LICENSE.md` 全文只有一句：受 Unreal Engine EULA 约束（unrealengine.com/eula）。
EULA 授权的是 *使用引擎*，不是 *把引擎源码再分发进一个 MIT 仓库*。

推论：**"提取"这个动词必须换掉**。可行的形态是：

- submodule 只做 *阅读* 参考（已经是这样，见 `.gitmodules` 与
  `scripts/clone_unreal_reference.sh`）；仓库里只存 gitlink，不存源码。
- 内核代码 clean-room 重写：读设计、读算法结构、读边界条件，然后自己写。
- 想借鉴的是 *思路*，具体是三样：**XPBD 约束求解、island 分组管理、约束图着色
  分批**。对应源码在 `Public/Chaos/PBDConstraintGraph.h`、`Public/Chaos/Island/`
  （6 个头文件：`IslandGraph` / `IslandGroup` / `IslandGroupManager` /
  `IslandManager` / `SolverIsland`）、`Public/Chaos/GJK.h` / `EPA.h` /
  `EPAVectorized.h`。

## 2. 移植成本：逐行移植不成立

Chaos 不是一个能"摘出来编译"的模块：

- **Core 一个模块就 674,371 行**（`Engine/Source/Runtime/Core`，同口径代码文件），
  是 Chaos 自身 294,203 行的 2.3 倍；加上 CoreUObject 366,143 行、GeometryCore
  135,469 行、AutoRTFM 14,932 行。271 个 Chaos 代码文件直接 `#include` Core 系
  头文件（`CoreMinimal` / `CoreTypes` / `Containers` / `HAL` / `Templates` / `Misc`），
  其中 84 个直接包含 `CoreMinimal.h`。
- 377 个代码文件使用 `PLATFORM_*` / `UE_*` 宏——**接近半数文件带着平台抽象层**。
- 108 个文件直接使用线程原语（`ParallelFor` / TaskGraph / `TQueue` / `FRWLock` /
  `TAtomic` / `FPlatformAtomics`）——**wasm 单线程环境下没有对应物**。
- 模板高度密集：292 个文件含 `template<`，共 1,334 处 `template<`，其中
  `template<typename T>` 427 处（float/double 双份泛型）。
- `Chaos.Build.cs` 的公开依赖是 AutoRTFM / Core / CoreUObject / ChaosCore /
  IntelISPC / TraceLog / Voronoi / GeometryCore / ChaosVDRuntime /
  TraceBasedDebuggers / NNE，私有依赖 Eigen / MeshDescription；69 个代码文件引用
  ISPC，字面量 `ISPC` 出现 742 次，**而 ISPC 不能编译到 wasm**。
- Epic 自己的 `COMPILE_WITHOUT_UNREAL_SUPPORT` 独立构建开关只有 22 处，且
  `Chaos.Build.cs:36` 明写 `PublicDefinitions.Add("COMPILE_WITHOUT_UNREAL_SUPPORT=0")`
  ——**默认关闭**。也就是说这不是一个被维护着的独立构建目标。

结论：能移植的只有 *窄而高价值的子集*，且必须重写而非搬运。

## 3. 决定性反证：Chaos 完全没有 GPU 路径

在 816 个文件里搜索 `FRDGBuffer` / `RDG_BUFFER` / `AddPass` /
`FComputeShaderUtils` / `ENQUEUE_RENDER_COMMAND` / `RenderCore` / `RHI.h`：

**命中文件数 = 0。**

同时：38 个文件使用 `VectorRegister` / SIMD，108 个文件使用线程原语
（`ParallelFor` / TaskGraph / `TQueue` / `FRWLock`）。

这说明两件事：

- UE 的刚体物理是 **纯 CPU + SIMD + 多线程** 的架构。它的并行来自 *island 分组
  后在 CPU 核间并行*，不是来自 GPU。
- 所以"GPU 物理"对我们 **不是追赶 UE，而是 UE 没走的路**。这既是机会也是警告：
  机会在于 GPU 规模物理是 UE 在浏览器里做不到的；警告在于 Chaos 的成熟度来自
  十几年 CPU 迭代，GPU 侧没有可抄的答案。

顺带一个体量参照：MassEntity（UE 的大规模 ECS/AI，`Engine/Source/Runtime/MassEntity`）
102 个代码文件 / 41,684 行；连同 `Runtime/Mass` + `MassGameplay` + `MassAI` 共
600 个代码文件（634 个全部文件）。
这是"游戏 AI"那一半要参考的对象，比 Chaos 小一个数量级，架构密度更高
（archetype + fragment + query + processor，和我们现有的 ECS 是同一族思路）。

## 4. Rust → wasm：2.4x 是真的，1.43x 的 SIMD 数字是假的

实测对照（`/tmp/wasmbench/bench4.mjs`，语义严格一致的 Gauss-Seidel 约束求解，
16.384M 次求解，N=4096）：

| 实现 | 时间 | 加速比 |
|------|------|--------|
| Rust wasm（opt-level 3, lto, codegen-units 1） | 45.14 ms | **2.43x** |
| JS（Float32Array，紧凑循环） | 109.79 ms | 1x |

SoA integrate（8.192M body-steps，访存受限）：Rust wasm 20.84 ms vs JS 48.01 ms =
**2.30x**。探针 wasm 二进制仅 24KB。跨边界调用开销约 9ns/次。

⚠️ **一个已作废的数字**：早期基准报告 wasm-simd128 再快 1.43x。复核源码
（`/tmp/wasmprobe_a/src/lib.rs`）后发现 `solve_simd` 用 `v128_load(i)` 与
`v128_load(i+1)` 读入后按 *加载值* 写回，是 **块状 Jacobi**；而 `solve_scalar`
是顺序依赖的 **Gauss-Seidel**。两者语义不同，比较无效。更早那个 "5.83x /
20.43ms" 是同一类错误（Jacobi vs Gauss-Seidel）。

**教训写进仓库**：`scripts/bench_gpu_compute.mjs` 与
`scripts/bench_cpu_nbody.mjs` 的文件头都注明「两侧必须语义一致，否则比值无意义」，
并把 softening / dt / maxSpeed / seed 定为同一组常量。

所以：**Rust wasm 相对 JS 的真实收益是 2.3–2.4x，不是 5–6x。**

## 5. GPU vs CPU：性能随规模上升，但共享内存 iGPU 极不稳定

同一负载（全对 O(N²) n-body），同一台机器（i7-10700 + GTX 1080 Ti + Intel UHD 630）：

CPU 单线程 JS（`scripts/bench_cpu_nbody.mjs`，三次运行；前两次同批，第三次隔约
10 分钟重跑）：

| N | per-step ms | G pair-int/s |
|---|---|---|
| 1024 | 4.60 / 4.79 / 4.48 | 0.228 / 0.219 / 0.234 |
| 2048 | 18.90 / 18.72 / 17.42 | 0.222 / 0.224 / 0.241 |
| 4096 | 77.23 / 71.38 / 69.45 | 0.217 / 0.235 / 0.242 |
| 8192 | 289.24 / 282.18 / 273.47 | 0.232 / 0.238 / 0.245 |

CPU 吞吐在四个数量级上落在 **0.217–0.245 G pair-int/s**（±6%，含跨批次），
且**与 N 无关**——这正是 O(N²) 全对算法该有的形状，说明测的是吞吐而不是启动开销。
**绝对值不要当规格引用**（它随机器温度/调度漂 ±6%）；要引用的是两个结论：
CPU 侧平坦且低方差，GPU 侧不平坦且高方差（见下）。

GPU WebGPU（`scripts/bench_gpu_compute.mjs`）：同一脚本、同一机器、**10 次独立
进程复跑**，每个 N 先 3 次 warmup，再计时。`G pair-int/s` 的
`min / median / max` 如下，最后一列以 CPU 三次运行的中位值为 1：

| N | min | median | max | median / CPU |
|---|---:|---:|---:|---:|
| 1024 | 0.06 | 0.09 | 0.11 | 0.40x |
| 2048 | 0.13 | 0.16 | 0.20 | 0.71x |
| 4096 | 0.24 | 0.30 | 0.37 | 1.28x |
| 8192 | 0.52 | 0.84 | 1.30 | 3.53x |
| 16384 | 1.37 | 2.04 | 2.88 | 8.57x |

结论比“GPU 快/慢”更具体：

- **小 N 不占优**：1024/2048 时 GPU 中位仍比 CPU 慢，dispatch 和共享内存延迟
  吃掉了并行收益。
- **大 N 开始拉开**：8192 约 3.5x，16384 约 8.6x。全对 O(N²) 是极端可并行的
  形状，正好代表粒子/软体/群体这类规模问题。
- **方差远大于 CPU**：N=1024 的 min/max 有约 1.8 倍，N=8192 有约 2.5 倍；
  另一个短窗口甚至测到 5.2 G/s，说明共享内存 iGPU 的时钟/功耗状态能造成
  **10 倍级漂移**。绝对值不能当规格引用。

根因已定位，不是代码问题：

- `nvidia-smi` 全程显示 1080 Ti **0% 利用率 / 139 MHz / 8.59 W**——它根本没参与。
- 该卡没有任何 Vulkan 或 GL 用户态库（无 `nvidia_icd*.json`、无
  `libGLX_nvidia.so`、无 `libnvidia-glcore.so`），只装了 NVML 与 OpenCL。
  `navigator.gpu.requestAdapter()` 因此只能返回 Intel gen-9（UHD 630）。
- i915 还要与桌面合成器争抢共享内存带宽，这是大幅漂移的重要来源之一。
- adapter `features` 为 **空集**：`subgroup=false`、`timestamp-query=false`；
  compatibility 模式下 `maxComputeInvocationsPerWorkgroup=128`（default 模式 256）。

⚠️ **这组数字是本机证据，不是生产规格。** CI 环境同样跑不出代表值：GitHub
Actions 的 SwiftShader 是软件光栅化，`--use-angle=vulkan` 下 adapter 会出现但为
软件回退。任何"GPU 比 CPU 快 N 倍"的结论都必须在有独显 Vulkan ICD 的机器上重测。

**架构结论不变，但理由要更准确**，有三条：

1. GPU 把并行规模从“CPU 核数”变成“workgroup 数”，这是 CPU 无法替代的轴。
2. 性能优势随 N 增长，而 CPU 全对吞吐几乎与 N 无关；这意味着 GPU 适合
   大规模仿真，不适合小规模刚体求解。
3. Pages 没有 SAB，CPU 只有一核预算；把大规模负载移到 GPU 后，CPU 那一核
   才能留给确定性求解器、AI 推理和主循环。

所以 GPU 层的定位是 **大规模仿真与吞吐**，不是替代确定性刚体求解器。

## 6. 已验证成立的架构基石：单 GPUDevice 共享 + 零拷贝

`scripts/bench_shared_device.mjs` 在真实浏览器里断言三件事，全部通过
（three.js r186，退出码 0，脚本会把任一断言失败变成非零退出）：

```text
adapter: vendor=intel arch=gen-9
three REVISION: 186
device: subgroup=false timestampQuery=false maxWorkgroupInvocations=128 maxStorageBinding=128MB
backend is WebGPU: true
CLAIM1 same device object: true
CLAIM2 TSL computeAsync: OK
CLAIM2 three-managed GPUBuffer reachable: true
CLAIM2 usage STORAGE=true VERTEX=true COPY_SRC=true
CLAIM2 readback p[0] = [1.6000, 0.0000, 1.6000]
CLAIM3 raw pipeline bound to THREE's buffer: OK
CLAIM3 20x raw-compute + three.render interleaved: OK (N.NN ms/frame, compute+render+present)
CLAIM3 points carrying raw-kernel sentinel (w=7.5): 8192/8192
CLAIM3 sample = [0.9307, 1.0386, 1.2591]
canvas PNG bytes (render produced pixels): 6635
```

上面只有 `ms/frame` 是随机器变的（同一台机器两次运行实测 12.31 与 2.60），
它只说明"20 帧交错没有卡死"，**不是性能指标，不要引用**。其余各行是确定性输出：
哨兵值、8192/8192 命中、usage 组合、`three REVISION`、adapter 特征都应逐字一致。

三条断言的含义：

1. **`CLAIM1`** 我们自己 `requestDevice()`，把它传给
   `new THREE.WebGPURenderer({ device })`，`renderer.backend.device === device`
   为真。源码依据：`WebGPUBackend.js:213`（`parameters.device === undefined`
   才自建）与 `:296`（`this.device = device`）。
2. **`CLAIM2`** TSL compute（`storage()` + `compute()` + `Fn`）写入
   `StorageBufferAttribute` 后，three.js 为它创建的 `GPUBuffer` 可以经
   `renderer.backend.get(attr).buffer` 拿到，且 usage 同时含
   `STORAGE | VERTEX | COPY_SRC`——**同一块内存既能被计算写、又能被顶点阶段读**。
3. **`CLAIM3`** 一个 **完全外部的裸 WGSL pipeline** 把 three.js 那块 buffer 当作
   `@group(0) @binding(0)` 绑上，写入哨兵值 `w=7.5`，与 `renderer.render()` 在同一
   device/queue 上交错 20 帧无冲突；随后 8192/8192 个点全部读回哨兵值。

**这就是 Rust/wgpu 计算层需要的桥**：native 侧不必拥有 device，也不必复制数据，
只要拿到 `GPUBuffer` 句柄就能驱动它。

两条容易踩的实现细节（都已踩到并修好）：

- `StorageBufferAttribute` **没有** `setBuffer()`。它只是 `BufferAttribute` 的子类
  （`src/renderers/common/StorageBufferAttribute.js`），GPU buffer 由 three.js 从
  CPU typed array 自建。方向是"我们读它的句柄"，不是"我们塞一个给它"。
- `MAP_READ` 只能与 `COPY_DST` 组合。写成 `COPY_SRC | MAP_READ` 会得到一个
  **创建即 invalid** 的 buffer，错误却要等到 `mapAsync` 才报
  `Invalid Buffer due to previous error`，离真正的原因很远。
- `renderer.getArrayBufferAsync(attr)` 在 `target === null` 时返回裸
  `ArrayBuffer`，需要自己包 `Float32Array`。
- three.tsl.js 用裸标识符 `import ... from 'three/webgpu'`，浏览器里必须有
  importmap 才能解析。

## 7. 部署硬约束：Pages 上没有 COOP/COEP

实测响应头（`curl -sI https://flowinginthewind700.github.io/threedream/`）：

```text
server: GitHub.com
content-type: text/html; charset=utf-8
cache-control: max-age=600
```

`Cross-Origin-Opener-Policy` / `Cross-Origin-Embedder-Policy` 命中数 **0**。
浏览器内验证 `typeof SharedArrayBuffer === 'undefined'`。

后果：`wasm32-unknown-unknown` + `+atomics` 编出来的 rayon / `wasm-bindgen-rayon`
在 Pages 上会直接构造失败。**wasm 多线程在 Pages 上不可能**。

三条路，任选其一：

| 方案 | 代价 |
|------|------|
| 接受单核 wasm | 零成本，损失多核；但 GPU 层正好补这一块 |
| 迁到 Cloudflare Pages / Vercel 并自设响应头 | 换托管，保留 Pages 的 CI 语义需重配 |
| 用 OffscreenCanvas + Worker 做 *渲染* 并行 | 不能给物理加核（SAB 才是关键） |

WebGPU 覆盖率（MDN BCD）：Chrome 144 / Chrome Android 121 / Edge 镜像 /
Firefox 141 / Safari 26；**Firefox Android 不支持**；caniuse 加权约 85.7%。
所以 **WebGL2 回退路径必须存在**，不能假设人人有 WebGPU。

## 8. 确定性契约不能被 GPU 破坏

仓库现有契约（`tests/physics.test.ts:418` 钉死 `world.deterministic === true`；
`src/physics/builtin.ts:69`；`tests/rapier_backend.test.ts` 钉死 rapier 跨平台
*非* 确定）：

- 训练与回放依赖 **位级可复现**。GPU 的浮点归约顺序由驱动决定，跨设备不可复现。
- 因此 **CPU 确定性后端必须保留**，`PhysicsBackend` 接口
  （`src/physics/types.ts:146`）已经带 `readonly deterministic: boolean`，
  GPU 后端应当如实声明 `deterministic: false`。
- 训练/回放/无 WebGPU 回退 → CPU；大规模表现层 → GPU。两者不是替换关系。

## 9. 更先进的计划

### 双层物理，接口已经在那里

```text
            +--------------------------------------+
            |  PhysicsBackend (types.ts:146)       |
            |  name / deterministic / fixedDt      |
            +--------------------------------------+
               |                      |
     builtin (TS, det=true)    gpu (WebGPU, det=false)
     rapier  (wasm, det=false)      ^
                                    |
                          粒子 / 软体 / 布料 / 群体
                          island -> workgroup 映射
```

- **CPU 层**：Rust → wasm + SIMD128。承载刚体接触求解、训练、回放、WebGL2 回退。
  相对 JS 有 2.3–2.4x 的实测收益，且这是 *唯一* 能提供位级确定性的层。
- **GPU 层**：承载规模。关键设计是 **用 island 分组映射到 workgroup**，以此替代被
  SAB 封死的 CPU 多线程——island 之间本就无耦合，天然是并行的单位。
- 两层共用一个 `GPUDevice`，已由 `bench_shared_device.mjs` 验证。

### GPU 侧的硬约束必须提前设计进去

本机 adapter `features` 为空集，意味着 **不能依赖 subgroup**。
`maxComputeInvocationsPerWorkgroup` 在 compatibility 模式只有 **128**（default 256）。
`maxStorageBufferBindingSize` = 128MB（约 4.19M 个 32B 刚体）。

所以：broadphase 用 **均匀网格 / 空间哈希 + 原子计数**（可无 subgroup），
不要写成依赖 subgroup ballot/broadcast 的形式；workgroup size 取 64，
在 128 与 256 两档下都合法。

### wgpu 版本必须锁死

wgpu 25.0.2 可编 wasm32：**raw 1.04MB / gzip 0.21MB**（cdylib + lto）。
但 v25 → v30 之间 API 漂移严重：`InstanceDescriptor::default()`、
`push_constant_ranges` → `immediate_size`、`request_adapter` 的返回类型全变过。
而且 **wgpu 25 没有 `from_web`**；interop（`create_texture_from_webgpu_handle`、
`Texture::as_webgpu`）要到较新版本（v30.0.1）才有。

结论：要么锁 v25 并接受没有官方 interop（自己经 JS 传句柄，即本报告已验证的路径），
要么升到 v30 用官方 interop 但承担 API 迁移。**当前推荐前者**——第 6 节已经证明
不需要 wgpu 的 interop 也能零拷贝共享。

### 借鉴 UE 的什么

| 来源 | 借鉴 |
|------|------|
| Chaos XPBD | 约束求解的 compliance / 迭代结构 |
| Chaos `Island/` | island 分组 → **workgroup 映射**（GPU 并行的关键） |
| Chaos `PBDConstraintGraph.h` | 约束图着色分批，无着色即有数据竞争 |
| Chaos `GJK.h` / `EPAVectorized.h` | 窄相算法结构；vectorized 版是 SIMD 思路参考 |
| MassEntity | archetype + fragment + query + processor，与我们现有 ECS 同族 |

**不借鉴代码本身。**

### 建议的推进顺序

1. **锁 wgpu 版本 + 建 Rust wasm crate 骨架**，把现有 TS builtin 求解器移植为
   Rust，用 `PhysicsBackend` 的同一组测试钉住语义一致（含确定性位级复现）。
2. **GPU 层先做粒子**，不做刚体。粒子无接触图、无 island、无着色问题，
   是验证 workgroup 映射与 buffer 共享的最短路径。
3. **WebGPU → WebGL2 回退**必须与 GPU 层同时落地（14.3% 用户没有 WebGPU）。
4. **托管决策**：若要多核，此时迁出 Pages；否则明确记录"单核是设计约束"。
5. 之后才是刚体的 island → workgroup 映射与约束着色。

## 复现

```bash
node scripts/audit_unreal_reference.mjs  # 第 1-3 节全部 UE 计数 + 三条法律/GPU 断言
node scripts/bench_cpu_nbody.mjs       # CPU 基线，单线程 JS
node scripts/bench_gpu_compute.mjs     # WebGPU 吞吐 + device limits
node scripts/bench_shared_device.mjs   # 三条架构断言，失败则退出非零
```

第一个脚本需要 `thirdparty/UnrealEngine`（`bash scripts/clone_unreal_reference.sh`，
要 EULA 权限）；找不到就打印一行跳过并退出 0，所以 CI 与普通 checkout 不受影响
（CI 故意不递归 submodule，见 `tests/ci_workflow.test.ts`）。它把三条结论变成断言：
816 个代码文件 **全部** 带 Epic 版权、**零** 个 OSI 许可头、Chaos **无** GPU/RHI/RDG
路径——任一不成立即退出 1。`--json` 输出机器可读结果，`--ue <path>` 指定源码树。

后两个脚本需要支持 WebGPU 的浏览器。headless Chromium 需要：

```text
--headless=new --no-sandbox --ignore-gpu-blocklist
--enable-features=Vulkan,DefaultANGLEVulkan,WebGPUService --use-angle=vulkan
```

可用 `CHROME_PATH` 指定可执行文件，`BENCH_PORT` 指定端口（默认 8890 / 8891）。

## 本机环境（解释所有数字的前提）

| 项 | 值 |
|----|----|
| CPU | Intel i7-10700 @ 2.90GHz，16 逻辑核 |
| Node | v24.20.0 |
| Rust | 1.98.1（wasm32-unknown-unknown 已装） |
| three.js | 0.186.0 |
| GPU（实测被选中） | Intel gen-9（UHD 630），`features` 空集 |
| GPU（未被使用） | GTX 1080 Ti 11GB——**无 Vulkan/GL 用户态库**，仅 NVML + OpenCL |
| UE 参考树 | 5.8.2（`16d75d84`，shallow，EULA 门控） |
