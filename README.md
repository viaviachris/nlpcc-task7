# nlpcc-task7
# Track1 检索实验结果总结
## 任务与指标

Track1 是隐式概念检索任务：给定查询或样例文本，从固定的 ELSST 概念池中检索并排序相关概念。本实验主要关注以下指标：

- `MRR`：平均倒数排名，衡量第一个相关结果出现得是否靠前。
- `NDCG@10`：Top 10 排序质量，也是训练阶段主要使用的模型选择指标。
- `Recall@10`：Top 10 中召回相关概念的能力。
- `Recall@5`：更严格的前排召回能力。

## 基线结果

本实验先记录了两个 Track1 LoRA 微调基线：

| 模型 | 方法 | Recall@5 |
|---|---|---:|
| Qwen3-0.6B Embedding | LoRA 微调 | 0.4490 |
| e5-small | LoRA 微调 | 0.3420 |

这两个结果作为早期基线，用于衡量后续 checkpoint 选择、评估修复、融合与检索增强方法带来的提升。

## 实验方法演进

### 1. 单模型 Best 对比

第一阶段先比较不同 embedding 底座在单模型检索下的最佳结果。为了避免把同一训练方案下的不同 checkpoint 误写成不同方法，主表中只保留每个模型系列的最佳结果。

| 模型 | 方法说明 | MRR | NDCG@10 | Recall@10 | Recall@5 |
|---|---|---:|---:|---:|---:|
| E5-small (Best) | Encoder-only embedding，修复评估后选择最佳 checkpoint | 0.5768 | 0.4357 | 0.5074 | 0.4100 |
| E5-base (Best) | 更大规模的 E5 encoder-only embedding，选择最佳 checkpoint | 0.5590 | 0.4273 | 0.5067 | 0.4300 |
| Qwen3-0.6B (Best) | Qwen3-Embedding-0.6B LoRA 微调，选择最佳 checkpoint | 0.6959 | 0.5769 | 0.6612 | 0.5572 |

结论：Qwen3-0.6B 作为生成式架构的 embedding 模型，在复杂长文本到受控词表概念的映射上显著优于传统 encoder-only 的 E5 系列。E5-small 与 E5-base 的差异并没有带来稳定提升，说明该任务的瓶颈不只是模型参数量，也与预训练目标、输入格式和概念映射能力有关。

### 2. E5 多模型 RRF 融合

第二阶段将多个 E5 检索结果进行 Reciprocal Rank Fusion，简称 RRF。融合公式为：

```text
score(concept) += weight_i / (rrf_k + rank_i)
```

代表性 E5 融合结果如下：

| 版本 | 方法说明 | MRR | NDCG@10 | Recall@10 | Recall@5 |
|---|---|---:|---:|---:|---:|
| 2-way RRF, k=10 | 两路 E5 融合，偏向 NDCG | 0.5986 | 0.4574 | 0.5300 | 0.4436 |
| 3-way RRF, k=30 | 三路 E5 融合，偏向 Recall | 0.5911 | 0.4537 | 0.5345 | 0.4509 |
| 3-way RRF, k=60 | 三路 E5 融合，较大的 RRF 平滑参数 | 0.5892 | 0.4529 | 0.5350 | 0.4445 |
| Weighted E5 recall | 加权 E5 融合 | 0.5894 | 0.4525 | 0.5336 | 0.4475 |
| 4-way RRF, k=60 | 四路 E5 融合 | 0.5976 | 0.4554 | 0.5299 | 0.4456 |

结论：RRF 能明显提升 E5 结果，尤其是 MRR 和 Recall@10；但 E5 融合后的上限仍大致在 `NDCG@10 = 0.455-0.457`。

### 3. Qwen3-Embedding-0.6B 微调

第三阶段切换到 `Qwen/Qwen3-Embedding-0.6B`。这一阶段是性能提升的关键拐点。主要方法变化如下：

- 底座模型：`Qwen3Model`，hidden size 1024。
- 查询侧加入 instruction prompt。
- `max_seq_length = 1536`。
- `learning_rate = 8e-5`，cosine scheduler，`weight_decay = 0.01`。
- `per_device_train_batch_size = 64`，`per_device_eval_batch_size = 128`。
- 启用 `gradient_checkpointing = true`。
- LoRA target modules：`q_proj`、`k_proj`、`v_proj`、`o_proj`、`gate_proj`、`up_proj`、`down_proj`。
- 使用验证集 `cosine-NDCG@10` 选择最佳 checkpoint。

该方法的最佳 checkpoint 为 `checkpoint-2500`，约为 `0.895` epoch。继续训练到后续 checkpoint 后指标略有下降，因此后续单模型与融合实验均以最佳 checkpoint 为主。

结论：Qwen3 带来了最主要的性能跃升，说明生成式架构 embedding 对长文本隐式概念检索更适配。

### 4. Qwen3 v1 Prompt 版本

`track1_qwen3_embedding_v1` 是另一组 Qwen3-Embedding 训练，主要改动是查询和概念文本格式。`checkpoint-2500` 的验证集 IR evaluator 结果如下：

| 版本 | Step | Epoch | MRR@10 | NDCG@10 | Recall@10 | Recall@5 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3 embedding v1 | 2500 | 0.8948 | 0.6893 | 0.5756 | 0.6631 | 0.5582 |

该版本与前面正确加载的 Qwen3 `ckpt2500` 非常接近，Recall@5 略高，但整体 NDCG@10 和 MRR 没有明显超过原版本。

### 5. Qwen3 与 E5 加权融合

第四阶段将 Qwen3 与 E5 的 ranking 输出进行加权 RRF 融合。思路是以 Qwen3 作为主检索器，同时利用 E5 提供互补召回信号。

代表性融合结果如下：

| 版本 | 方法说明 | MRR | NDCG@10 | Recall@10 | Recall@5 |
|---|---|---:|---:|---:|---:|
| Qwen3 checkpoints RRF | 融合 Qwen3 ckpt2500/5000/5588 | 0.6992 | 0.5764 | 0.6563 | 0.5563 |
| Qwen3 ckpts + E5 weighted | Qwen3 多 checkpoint 加 E5 信号 | 0.6983 | 0.5814 | 0.6664 | 0.5607 |
| Qwen3 + E5 weighted NDCG | Qwen3 与 E5 加权 RRF | 0.7051 | 0.5858 | 0.6709 | 0.5664 |

结论：E5 单模型弱于 Qwen3，但作为互补 ranker 仍然有效。Qwen3+E5 加权融合相比 Qwen3 单模型继续提升了 NDCG@10、Recall@10 和 Recall@5。

### 6. Qwen3 + E5 + BM25 三路召回融合

第五阶段加入 BM25 字面匹配召回。BM25 单路效果较弱，但它的错误模式与 dense embedding 不同，可以补充关键词和概念词面匹配信号。

| 版本 | 方法说明 | MRR | NDCG@10 | Recall@10 | Recall@5 |
|---|---|---:|---:|---:|---:|
| BM25 单路 | 基于 concept term + definition 建立 BM25 索引 | 0.1606 | 0.1079 | 0.1550 | 0.1054 |
| Qwen3 + E5 + BM25 RRF | 三路融合，`rrf_k=10`，权重为 `1.5/0.5/0.5` | 0.7153 | 0.5973 | 0.6823 | 0.5744 |

结论：BM25 单路指标很低，但加入融合后产生正收益。当前最佳系统由 Qwen3、E5 和 BM25 三路 RRF 得到，说明字面匹配信号对 ELSST 概念检索有一定互补价值。

### 7. 通用 Cross-Encoder Reranker 消融

在当前最佳的 Qwen3+E5+BM25 RRF top100 候选基础上，尝试使用 `bge-reranker-base` 对 top50 进行重排序。结果如下：

| 版本 | 方法说明 | MRR | NDCG@10 | Recall@10 | Recall@5 |
|---|---|---:|---:|---:|---:|
| bge-reranker-base top50 | 通用 reranker 直接重排 RRF top50 | 0.1626 | 0.1092 | 0.1839 | 0.0926 |

结论：通用 reranker 直接重排严重破坏原排序，说明普通 query-passage reranker 与“长文本到 ELSST 受控词表概念”的任务并不匹配。后续如果使用 reranker，应优先进行任务内微调，而不是直接 zero-shot 硬重排。

## 最佳结果

最佳单模型：

| 模型 | 方法 | MRR | NDCG@10 | Recall@10 | Recall@5 |
|---|---|---:|---:|---:|---:|
| Qwen3-Embedding-0.6B | LoRA 微调，选择 ckpt2500 | 0.6959 | 0.5769 | 0.6612 | 0.5572 |

最佳整体系统：

| 系统 | 方法 | MRR | NDCG@10 | Recall@10 | Recall@5 |
|---|---|---:|---:|---:|---:|
| Qwen3 + E5 + BM25 | 三路 RRF 融合，`rrf_k=10`，权重 `1.5/0.5/0.5` | 0.7153 | 0.5973 | 0.6823 | 0.5744 |

相较于早期 Track1 LoRA 基线，最佳整体系统的 Recall@5 提升如下：

### 实验结果对比

#### 1. 基线与最佳系统召回率对比
| 对比对象 | 基线 Recall@5 | 最佳系统 Recall@5 |
| :--- | :---: | :---: |
| Qwen3-0.6B Embedding LoRA 微调基线 | 0.4490 | 0.5744 |
| e5-small LoRA 微调基线 | 0.3420 | 0.5744 |

#### 2. E5 单模型与最佳整体系统全指标对比
相较于最佳修复版 E5 单模型，最佳整体系统提升如下：

| 指标 | E5 单模型 | 最佳系统 | 提升幅度 |
| :--- | :---: | :---: | :---: |
| **MRR** | 0.5768 | 0.7153 | +24.01% |
| **NDCG@10** | 0.4357 | 0.5973 | +37.09% |
| **Recall@10** | 0.5074 | 0.6823 | +34.47% |
| **Recall@5** | 0.4100 | 0.5744 | +40.10% |

## 实验总结与消融分析

在“长文本到受控词表”的映射任务中，生成式 Qwen3 凭借强大的复杂概念建模能力显著优于传统 E5 模型；同时，BM25 的字面硬匹配虽单路表现极差，但能有效补充语义模型遗漏的词面线索，使三路融合达到最优；然而，直接套用通用重排模型（如 bge-reranker）会严重破坏排序分布导致指标暴跌（NDCG@10 从 0.59 降至 0.10），这证明该任务必须依赖领域专属的重排微调。

## 后续实验

1. 训练 ELSST 专用 reranker：使用 `train.jsonl` 中的 positive concepts 和 hard negatives 构造 `(query, concept)` 二分类或 pairwise ranking 数据，对 `bge-reranker-base`、`Qwen3-Reranker-0.6B` 等模型进行任务内微调。
2. Hard Negative Mining：用当前最佳 Qwen3+E5+BM25 系统在训练集上检索 top100，将高排名错误概念作为新的 hard negatives，重新微调 embedding 或 reranker。
3. 尝试更强 embedding 底座：优先考虑 `Qwen3-Embedding-4B` 或 `Qwen3-Embedding-8B`，其次考虑 `bge-m3`、`gte-large` 等模型作为新的召回源。
4. Concept Expansion：为每个 ELSST concept 生成 pseudo queries、同义表达或典型场景描述，扩展 concept 表示，缓解单一 `term + definition` 表达不足的问题。
5. Multi-vector Retrieval：将 concept 的 term、definition、pseudo queries、examples 分别编码为多个向量，检索时取 max 或加权聚合，提高概念覆盖能力。
6. Learning-to-Rank 融合：在获得更强单路模型后，用 Qwen3/E5/BM25/reranker 的 rank 和 score 作为特征，训练轻量排序模型替代手工 RRF 权重。
