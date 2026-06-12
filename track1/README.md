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
| E5-small  | Encoder-only embedding，修复评估后选择最佳 checkpoint | 0.5768 | 0.4357 | 0.5074 | 0.4100 |
| E5-base  | 更大规模的 E5 encoder-only embedding，选择最佳 checkpoint | 0.5590 | 0.4273 | 0.5067 | 0.4300 |
| Qwen3-0.6B  | Qwen3-Embedding-0.6B LoRA 微调，选择最佳 checkpoint | 0.6959 | 0.5769 | 0.6612 | 0.5572 |
| Qwen3-8B  | Qwen3-Embedding-8B LoRA 微调，选择最佳 checkpoint | 0.8857 | 0.7760 | 0.8330 |  0.7251 |

# Qwen3-Embedding-8B 版本 对比

这份说明对比了基于 Qwen3-Embedding-8B 的两个 Track1 密集检索版本。

## 方法

两个版本使用的是同一套检索训练框架：

- bi-encoder / SentenceTransformer 训练
- `MultipleNegativesRankingLoss`
- 基于 ELSST 概念文本的 label embedding
- 带 hard negatives 的监督式对比训练

差异主要体现在 prompt 设计和训练数据上。

### v1

- 使用原始 Track1 数据集训练
- 采用 `qwen3_embedding_v1` prompt 风格
- 概念文本按普通 controlled-vocabulary entry 方式组织
- 学习率为 `2e-5`
- 负样本来自数据集中已有的 hard negatives

### v1.1 label-mined
- 使用由 v1 checkpoint 挖掘得到的新训练数据
- 采用 `qwen3_embedding_v2` prompt 风格
- query prompt 更强调隐式/显式语义支持
- concept prompt 更强调概念匹配，而不只看字面重合



### v2
- hard negatives 加入了模型挖掘出的更难样本
- 在 v1.1 的 hard-negative 检索训练基础上，引入 label-mined 训练数据
- 采用 `qwen3_embedding_v2` prompt 风格，并在 query 中显式加入 `document_type`
- 将 ELSST concept 的 `term + definition` 构造成 label text，再编码为 label embedding
- 使用 LoRA 微调，目标模块包括 `q_proj`、`k_proj`、`v_proj`、`o_proj`、`gate_proj`、`up_proj`、`down_proj`
- 使用 `MultipleNegativesRankingLoss` 做监督式对比学习
- 每个 positive 使用 mined hard negatives 构造训练样本


## 指标对比

最终 checkpoint 的验证集结果如下：

| 版本 |  checkpoint | MRR | NDCG@10 | Recall@5 | Recall@10 |
|---|---|---:|---:|---:|---:|
| v1 | final checkpoint | 0.8905 | 0.7788 | 0.7250 | 0.8340 |
| v1.1 label-mined |  `checkpoint-1000` | 0.8858 | 0.7760 | 0.7251 | 0.8331 |
| v2 | final checkpoint | 0.9247 | 0.8051 | 0.7453 | 0.8473 |


## 总结

v2 是更强的检索方案。

它的提升主要来自两点：

1. 更适合隐式概念匹配的 prompt
2. 来自 v1.1 模型挖掘出的更难负样本

还有一版本在跑：1.数据基础上加入了 curriculum hard negatives，也就是负例按 easy / medium / hard 分层组织，让模型逐步学会区分更难的相似概念。
加入 label prototype 辅助目标：额外把概念的 label / definition / semantic view 做成 prototype embedding，训练时让 query 不仅靠近正例 concept，也靠近对应 label prototype。
评估/提交阶段支持 top-k rerank：评估时可以对 top-50 做轻量 rerank



# Qwen3-Embedding-8B Track1 方法演进与指标对比

这份说明在原有 `Qwen3-Embedding-8B v1 / v1.1 / v2` 对比基础上，继续补充 2026-06-05 以来查到的后续版本。日期只用于防止漏查版本，正文按“版本/方法演进”组织。

指标筛选原则：

- 主表只放 Track1 验证集 ranking 口径的结果，即基于 `val_retrieval_ranking.jsonl` 和验证集 gold labels 计算 `MRR`、`NDCG@10`、`Recall@5`、`Recall@10`。
- 只来自 SentenceTransformers 训练期 evaluator 的指标不放入主表。
- v1.1 label-mined 是原 README 中已有的官方 scorer 复算节点，保留为方法演进的一部分。

## 方法

这些版本整体延续同一条 Track1 dense label retrieval 路线：

- bi-encoder / SentenceTransformer 训练
- `MultipleNegativesRankingLoss`
- 基于 ELSST 概念文本的 label embedding
- 带 hard negatives 的监督式对比训练

差异主要体现在 prompt 设计、训练数据、hard-negative mining、ranking 融合、rerank 和 label text 构造。

### v1

- 使用原始 Track1 数据集训练
- 采用 `qwen3_embedding_v1` prompt 风格
- 概念文本按普通 controlled-vocabulary entry 方式组织
- 学习率为 `2e-5`
- 负样本来自数据集中已有的 hard negatives
- 后续用 `checkpoint-2235` 按 Track1 验证集 ranking 口径复算

### v1.1 label-mined

- 在 v1 的 hard-negative 检索训练基础上，引入 label-mined 训练数据
- 采用 `qwen3_embedding_v2` prompt 风格，并在 query 中显式加入 `document_type`
- 将 ELSST concept 的 `term + definition` 构造成 label text，再编码为 label embedding
- 使用 LoRA 微调，目标模块包括 `q_proj`、`k_proj`、`v_proj`、`o_proj`、`gate_proj`、`up_proj`、`down_proj`
- 使用 `MultipleNegativesRankingLoss` 做监督式对比学习
- 每个 positive 使用 mined hard negatives 构造训练样本
- 原 README 记录的 `checkpoint-1000` 已用官方 scorer 复算，因此保留在主表

### v2

- 使用由 v1 checkpoint 挖掘得到的新训练数据：`track1_mined_qwen3_8b_v1_ckpt2235`
- 采用 `qwen3_embedding_v2` prompt 风格
- query prompt 更强调隐式/显式语义支持
- concept prompt 更强调概念匹配，而不只看字面重合
- 学习率为 `1.5e-5`
- hard negatives 加入了模型挖掘出的更难样本
- `checkpoint-2235` 是后续 prompt ensemble、rerank、query expansion、label text enhancement 的主要基座

### v3

- 使用由 v2 checkpoint 第二轮挖掘得到的数据：`track1_mined_qwen3_8b_v2_ckpt2235`
- 仍采用 `qwen3_embedding_v2` prompt 风格
- 学习率调整为 `1e-5`
- 目标是验证第二轮 mined hard negatives 是否继续带来收益

### v2 prompt variants

- 固定模型为 `outputs/track1_qwen3_embedding_8b_v2/checkpoint-2235`
- 只替换推理时的 query/concept prompt
- 评估过 `qwen3_embedding_v3`、`qwen3_embedding_v4`、`qwen3_embedding_v5`
- 这些单独 prompt 版本主要用于观察 prompt 表述变化对 ranking 的影响

### prompt ensemble

- 融合 v2、v3、v4、v5 prompt 产生的 ranking
- 使用 weighted RRF
- 主实验权重为 `5 1 1 1`
- `train-top100` 版本额外补齐 train split 的 top100 ranking，用于训练后续 pairwise reranker

### rerank / RRF fusion

- `v2 rerank v4 top100`：以 v2 dense top100 为候选，用 `BAAI/bge-reranker-base` 训练/打分，再与 dense rank 做 weighted RRF
- `v2+v1 RRF`：融合 v2 与 v1 dense ranking，权重 `5 1`
- `v2+v1+e5+bm25 RRF`：融合 v2、v1、E5、BM25 ranking，权重 `5 1 0.2 0.02`
- `pairwise reranker ensemble top100`：先融合 v2/v3/v4/v5 prompt 的 train/val/test top100，再训练 pairwise reranker

### query expansion / label text enhancement

- `LLM query expansion v2`：对 query 追加 LLM 生成的 ELSST concept explanation，再用 v2 checkpoint 检索；当前只有 val-only 指标
- `label text enhanced v5`：重建 concept text，加入 broader/narrower/related/keywords 等 label 上下文，再用 v2 checkpoint 检索
- `label text enhanced v5 ablation`：围绕 concept text 增强强度做验证集消融，比较去掉 related labels、保留 broader/narrower、调整 keywords 与 narrower 数量后的效果
- `prompt ensemble + label text enhanced v5 RRF`：将原 prompt ensemble ranking 与 label text enhanced v5 ranking 做 weighted RRF 融合，用于验证 concept 侧增强是否与 prompt ensemble 互补

## 指标对比

主表按方法演进排序，不以日期命名。除 v1.1 为原 README 保留的官方 scorer 复算结果外，其余新增结果均来自对应输出目录的 `metrics.json`。

| 版本 / 方法 | checkpoint / 基座 | MRR | NDCG@10 | Recall@5 | Recall@10 |
|---|---|---:|---:|---:|---:|
| v1  | `checkpoint-2235` | 0.888264 | 0.776956 | 0.728153 | 0.833245 |
| v1.1 label-mined | `checkpoint-1000` | 0.885800 | 0.776000 | 0.725100 | 0.833100 |
| v2  | `checkpoint-2235` | 0.928444 | 0.813088 | 0.756504 | 0.857297 |
| v3  | `checkpoint-2235` | 0.911195 | 0.793051 | 0.732981 | 0.841799 |
| v2 prompt v3 | v2 `checkpoint-2235` | 0.925754 | 0.800034 | 0.740079 | 0.844158 |
| v2 prompt v4 | v2 `checkpoint-2235` | 0.924092 | 0.801513 | 0.741711 | 0.842659 |
| v2 prompt v5 | v2 `checkpoint-2235` | 0.924531 | 0.795586 | 0.740388 | 0.837743 |
| v2/v3/v4/v5 prompt ensemble | v2 `checkpoint-2235` | 0.929524 | 0.814456 | 0.754167 | 0.859061 |
| prompt ensemble train-top100 | v2 `checkpoint-2235` | 0.929636 | 0.813195 | 0.754167 | 0.856459 |
| v2 rerank v4 top100 | v2 dense top100 + bge-reranker-base | 0.928360 | 0.812846 | 0.755996 | 0.856724 |
| v2+v1 RRF | v2 + v1 | 0.926282 | 0.812241 | 0.755974 | 0.855930 |
| v2+v1+e5+bm25 RRF | v2 + v1 + E5 + BM25 | 0.926116 | 0.812450 | 0.758025 | 0.856085 |
| pairwise reranker ensemble top100 | prompt ensemble top100 + bge-reranker-base | 0.929626 | 0.813232 | 0.754167 | 0.856592 |
| LLM query expansion v2 | v2 `checkpoint-2235` | 0.918166 | 0.805194 | 0.750595 | 0.848765 |
| label text enhanced v5 | v2 `checkpoint-2235` | 0.932767 | 0.814471 | 0.756041 | 0.854475 |
| label text v5 ablation: no related + kw10 | v2 `checkpoint-2235` | 0.932988 | 0.814964 | 0.756922 | 0.854806 |
| label text v5 ablation: no related + kw0 | v2 `checkpoint-2235` | 0.921555 | 0.808190 | 0.754497 | 0.852315 |
| label text v5 ablation: narrower3 + no related + kw6 | v2 `checkpoint-2235` | 0.930180 | 0.814068 | 0.758532 | 0.851675 |
| label text v5 ablation: related2 + kw6 | v2 `checkpoint-2235` | 0.929298 | 0.812492 | 0.756834 | 0.850419 |
| prompt ensemble + label text enhanced v5 RRF | prompt ensemble + label text v5, RRF `1 2` | 0.932675 | 0.816444 | 0.757275 | 0.857914 |

## 结论

从方法演进看，v2 是 v1 之后最关键的提升节点：它用 v1 checkpoint 挖掘出的 hard negatives 和更适合隐式概念匹配的 `qwen3_embedding_v2` prompt，把 MRR 提升到 `0.928444`。v3 的第二轮 hard-negative mining 没有继续提升。单独 prompt variants 也没有超过 v2，但 v2/v3/v4/v5 的 prompt ensemble 带来了更高的 `NDCG@10` 和 `Recall@10`。LLM query expansion 和 reranker 没有形成稳定收益，说明当前瓶颈更偏向 ELSST concept 侧的细粒度标签表达，而不是简单扩大 query 或后置重排。

label text enhancement 是目前最有效的新方向。原始 v5 已经提高 MRR；进一步消融表明 `related labels` 主要引入噪声，`no related + kw10` 是单独 label text 版本中最强的配置，MRR 达到 `0.932988`，NDCG@10 达到 `0.814964`。不过它的 Recall@10 仍低于 prompt ensemble，说明增强后的 concept text 更擅长把已召回的正确概念排到前面，但会牺牲一部分 top10 覆盖。

当前 NDCG@10 最高的是 `prompt ensemble + label text enhanced v5 RRF`，达到 `0.816444`；当前 Recall@10 最高的仍是 `v2/v3/v4/v5 prompt ensemble`，为 `0.859061`。v1.1 label-mined 保留为早期 mined label retrieval 的关键中间版本：它证明了 label text + mined hard negatives + supervised contrastive learning 这条路线可行，后续 v2/v3 和 label text enhancement 都是在这条路线上继续改 prompt、改 hard negatives 或改 concept text 表达。
现在已经进入 ELSST 细粒度标签边界瓶颈。模型大多数时候能找到相关语义区域，但很难稳定区分正确 concept、近义 concept、上位词、下位词、related concept 谁应该进 top10、谁应该排更前。




