Paper汇总

NotebookLM https://notebooklm.google.com/notebook/10880084-fea7-4fbd-9356-68715f5e0c49

##  1. ABSA (Aspect-Based Sentiment Analysis)



ABSA 任务基于纯文本，旨在识别文本中的方面术语和其对应的情感倾向 [1]。

### 1.1 Aspect-Oriented Sentiment Classification (方面级情感分类, ASC/AOSC)



| 年份 | 论文/方法 (来源会议/期刊)                                    | 作者 (部分)                 | 核心方法/机制                                                | 来源        |
| ---- | ------------------------------------------------------------ | --------------------------- | ------------------------------------------------------------ | ----------- |
| 2014 | Adaptive Recursive Neural Network (ARRNN)                    | Dong et al.                 | 采用递归神经网络将情感从词语传播到方面 [2, 3]。              | [2, 3]      |
| 2015 | Target-dependent Twitter sentiment classification with rich automatic features | D.-T. Vo and Y. Zhang       | 基于丰富的自动特征进行目标依赖的 Twitter 情感分类 [4]。      | [4]         |
| 2016 | Attention-based LSTM (ATAE-LSTM)                             | Y. Wang et al.              | 采用基于注意力的 LSTM 模型，用于方面级情感分类 [4-8]。       | [4-8]       |
| 2016 | Aspect level sentiment classification with deep memory network (DM) | D. Tang, B. Qin, and T. Liu | 采用深度记忆网络进行方面级情感分类 [4, 9, 10]。              | [4, 9, 10]  |
| 2016 | Gated neural networks for targeted sentiment analysis        | M. Zhang et al.             | 采用门控神经网络进行目标导向情感分析 [11]。                  | [11]        |
| 2017 | Interactive attention networks (IAN)                         | D. Ma et al.                | 提出交互式注意力网络，用于方面级情感分类 [12-15]。           | [12-15]     |
| 2017 | Recurrent attention network on memory (RAN/RAM)              | P. Chen et al.              | 提出基于记忆的循环注意力网络，用于方面情感分析 [15-17]。     | [15-17]     |
| 2018 | Aspect based sentiment analysis with gated convolutional networks | W. Xue and T. Li            | 采用门控卷积网络进行方面级情感分析 [11, 18]。                | [11, 18]    |
| 2019 | Utilizing BERT for aspect-based sentiment analysis via constructing auxiliary sentence | C. Sun et al.               | 通过构建辅助句子将 ABSA 转化为句子对分类任务 [10, 19]。      | [10, 19]    |
| 2020 | Relational graph attention network (R-GAT)                   | K. Wang et al.              | 采用关系图注意力网络编码句法依存树 [7, 20, 21]。             | [7, 20, 21] |
| 2021 | Aspect-based sentiment analysis with type-aware GCNs and layer ensemble | Y. Tian et al.              | 采用类型感知图卷积网络和层集成进行 ABSA [20, 22]。           | [20, 22]    |
| 2021 | Dual graph convolutional networks (DualGCN)                  | R. Li et al.                | 提出双图卷积网络，考虑句法结构和语义关联的互补性 [3, 23-26]。 | [3, 23-26]  |
| 2022 | SSEGCN: Syntactic and Semantic Enhanced Graph Convolutional Network (NAACL) | Z. Zhang, Z. Zhou, Y. Wang  | 提出句法和语义增强的图卷积网络，用于学习方面相关的语义关联和全局语义 [1, 27-30]。 | [1, 27-30]  |
| 2022 | Aspect-based sentiment analysis via affective knowledge enhanced GCNs | B. Liang et al.             | 采用情感知识增强的 GCN 进行方面级情感分析 [3, 31-37]。       | [3, 31-37]  |



### 1.8 Aspect-Sentiment Pair Extraction (方面-情感对提取, ASPE/JASA)



| 年份 | 论文/方法 (来源会议/期刊)                                    | 作者 (部分)                  | 核心方法/机制                                                | 来源                        |
| ---- | ------------------------------------------------------------ | ---------------------------- | ------------------------------------------------------------ | --------------------------- |
| 2015 | Neural networks for open domain targeted sentiment           | M. Zhang, Y. Zhang, D.-T. Vo | 针对开放域目标情感的神经网络方法 [38-40]。                   | [38-40]                     |
| 2019 | SPAN (ACL)                                                   | M. Hu et al.                 | 提出基于 Span 的提取-分类框架，直接从句子中提取观点目标并分类 [38-46]。 | [38-46]                     |
| 2020 | D-GCN (COLING)                                               | G. Chen, Y. Tian, Y. Song    | 提出定向图卷积网络，通过序列标注范式建模词语间的依存关系，进行联合方面提取和情感分析 [16, 40, 41, 44, 45, 47-52]。 | [16, 40, 41, 44, 45, 47-52] |
| 2020 | Relation-aware collaborative learning for unified ABSA (ACL) | Z. Chen and T. Qian          | 关系感知协同学习，用于统一的 ABSA [53-55]。                  | [53-55]                     |
| 2021 | A unified generative framework for ABSA (BART) (ACL)         | H. Yan et al.                | 统一的生成式框架，将方面提取和情感分类转化为序列生成 [31, 39, 40, 44, 45, 56-62]。 | [31, 39, 40, 44, 45, 56-62] |
| 2023 | Span-level aspect-based sentiment analysis via table filling (ACL) | M. Zhang et al.              | 通过表格填充实现 Span 级方面的提取和情感分析 [63]。          | [63]                        |



### 1.12 Aspect-Opinion-Sentiment Triple Extraction (方面-观点-情感三元组提取, AOSTE)



| 年份 | 论文/方法 (来源会议/期刊)                                | 作者 (部分)    | 核心方法/机制                                       | 来源     |
| ---- | -------------------------------------------------------- | -------------- | --------------------------------------------------- | -------- |
| 2022 | Enhanced Multi-Channel Graph Convolutional Network (ACL) | H. Chen et al. | 增强的多通道 GCN，用于方面情感三元组提取 [23, 64]。 | [23, 64] |



### 1.13 Aspect-Category-Opinion-Sentiment Quadruple Extraction (方面-类别-观点-情感四元组提取, ACOSQE)



| 年份 | 论文/方法 (来源会议/期刊)                                    | 作者 (部分)   | 核心方法/机制                                 | 来源     |
| ---- | ------------------------------------------------------------ | ------------- | --------------------------------------------- | -------- |
| 2021 | Aspect-Category-Opinion-Sentiment Quadruple Extraction with Implicit Aspects and Opinions (ACL-IJCNLP) | H. Cai et al. | 处理包含隐式方面和观点的四元组提取 [65, 66]。 | [65, 66] |



## 2. Cross-Domain ABSA (跨域 ABSA)



### 2.1 Cross-Domain Aspect-Oriented Sentiment Classification (跨域方面级情感分类)



| 年份 | 论文/方法 (来源会议/期刊)                                    | 作者 (部分)    | 核心方法/机制                                | 来源 |
| ---- | ------------------------------------------------------------ | -------------- | -------------------------------------------- | ---- |
| 2019 | Neural attentive network for cross-domain aspect-level sentiment classification | M. Yang et al. | 针对跨域方面级情感分类的神经注意力网络 [5]。 | [5]  |



## 3. Multi-Modal ABSA (MABSA)



MABSA 是面向图像-文本对的细粒度情感分析任务，通常分解为 MATE、MASC 和 JMASA 三个子任务 [67-69]。



### 3.1 Multi-Modal Aspect Extraction (& Multi-Modal Named Entity Recognition, MATE/MM-NER)



| 年份 | 论文/方法 (来源会议/期刊)                                    | 作者 (部分)  | 核心方法/机制                                                | 来源                     |
| ---- | ------------------------------------------------------------ | ------------ | ------------------------------------------------------------ | ------------------------ |
| 2020 | UMT (ACL)                                                    | J. Yu et al. | 统一多模态 Transformer，用于通过实体 Span 检测改善多模态 NER [41, 45, 70-75]。 | [41, 45, 70-75]          |
| 2020 | Multimodal aspect extraction with region-aware alignment network (RAN) (NLPCC) | H. Wu et al. | 提出区域感知对齐网络，用于多模态方面提取 [41, 45, 59, 60, 76-79]。 | [41, 45, 59, 60, 76-79]  |
| 2020 | OS-CGA (ACM MM)                                              | Z. Wu et al. | 嵌入视觉引导对象的多模态表示，用于实体预测（MATE任务基线） [41, 45, 60, 70, 73, 80]。 | [41, 45, 60, 70, 73, 80] |



### 3.2 Multi-Modal Aspect-Oriented Sentiment Classification (MASC/ATMSC)



| 年份 | 论文/方法 (来源会议/期刊)                                    | 作者 (部分)        | 核心方法/机制                                                | 来源                                    |
| ---- | ------------------------------------------------------------ | ------------------ | ------------------------------------------------------------ | --------------------------------------- |
| 2019 | Entity-sensitive attention and fusion network (ESAFN) (IEEE/ACM TASLP) | J. Yu et al.       | 实体敏感的注意力与融合网络，用于实体级多模态情感分类 [41, 72, 74, 81-87]。 | [41, 72, 74, 81-87]                     |
| 2019 | Multi-interactive memory network (MIMN) (AAAI)               | N. Xu et al.       | 多交互记忆网络，用于方面级多模态情感分析 [18, 19, 58, 82, 87-96]。 | [18, 19, 58, 82, 87-96]                 |
| 2019 | TomBERT (IJCAI)                                              | J. Yu and J. Jiang | 基于 BERT 的目标导向多模态情感分类方法，采用 **目标-图像匹配层** 获取目标敏感视觉信息 [41, 44, 62, 74, 81-84, 86, 93, 97-103]。 | [41, 44, 62, 74, 81-84, 86, 93, 97-103] |
| 2021 | CapTrBERT (ACM MM)                                           | Z. Khan and Y. Fu  | 利用图像到文本的转换，将图像描述作为辅助句子进行情感分类 [41, 44, 81-84, 93, 104-107]。 | [41, 44, 81-84, 93, 104-107]            |
| 2022 | Targeted multimodal sentiment classification based on coarse-to-fine grained image-target matching (ITM) (IJCAI) | J. Yu et al.       | 基于粗粒度到细粒度的图像-目标匹配，进行目标导向的多模态情感分类 [72, 74, 108-112]。 | [72, 74, 108-112]                       |
| 2022 | Face-Sensitive Image-to-Emotional-Text Cross-modal Translation (FITE) (EMNLP) | H. Yang et al.     | 首次明确利用 **面部表情** 等视觉情感线索，通过门控机制实现细粒度匹配融合 [61, 70, 74, 85, 109, 113-119]。 | [61, 70, 74, 85, 109, 113-119]          |
| 2022 | KEF-TomBERT / Learning from adjective-noun pairs (COLING)    | F. Zhao et al.     | 知识增强框架，从图像中提取 **形容词-名词对 (ANPs)** 以增强视觉贡献 [79, 81, 102, 109, 110, 112, 120-125]。 | [79, 81, 102, 109, 110, 112, 120-125]   |
| 2023 | Image-to-Text Conversion and Aspect-Oriented Filtration (IGD+AAD) (IEEE T Affec Comput) | Q. Wang et al.     | 采用 **图像到文本转换 (IGD)** 避免模态对齐难度，并通过 **方面感知降噪 (AAD)** 模块过滤隐式 token 噪声 [79, 80, 126-129]。 | [79, 80, 126-129]                       |
| 2024 | Multi-grained Fusion Network with Self-Distillation (MGFN-SD) (KBS) | J. Yang et al.     | 通过细粒度融合和粗粒度语义增强模块，并使用 **自蒸馏** 机制提升特征表示质量 [129-131]。 | [129-131]                               |
| 2025 | Image Description and Aspect-Aware Denoising (IGD+AAD) (ICMR) | J. Sun and X. Li   | 采用图像描述模块将视觉模态转换为文本模态，并使用方面感知降噪模块过滤噪声 [132-134]。 | [132-134]                               |
| 2025 | Aspect-aware Semantic Feature Enhancement Network (ASFEN) (JSC) | B. Zeng et al.     | 转换图像为文本化的视觉语义信息（含情感线索），并构建句法依存树和多层掩码进行句法语义增强 [135]。 | [135]                                   |



### 3.3 Multi-Modal Aspect-Sentiment Pair Extraction (MASPE/JMASA/End-to-End MABSA



| 年份 | 论文/方法 (来源会议/期刊)                                    | 作者 (部分)       | 核心方法/机制                                                | 来源                                                         |
| ---- | ------------------------------------------------------------ | ----------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 2020 | UMT-collapse (ACL)                                           | J. Yu et al.      | 采用统一多模态 Transformer，通过**崩溃标签 (collapse)** 方式解决 JMASA 任务（基线） [28, 41, 70, 73, 83, 136]。 | [28, 41, 70, 73, 83, 136]                                    |
| 2020 | OSCGA-collapse (ACM MM)                                      | Z. Wu et al.      | 嵌入视觉引导对象的模型，通过**崩溃标签**方式解决 JMASA 任务（基线） [41, 60, 70, 73, 80, 83, 136, 137]。 | [41, 60, 70, 73, 80, 83, 136, 137]                           |
| 2021 | JML (Joint Multimodal Learning) (EMNLP)                      | X. Ju et al.      | **首次定义 JMASA 任务**，引入 **辅助跨模态关系检测机制 (ACRD)**，实现方面-情感对的联合学习 [41, 42, 46, 49, 50, 70, 74, 89, 91, 105, 106, 136, 138-155]。 | [41, 42, 46, 49, 50, 70, 74, 89, 91, 105, 106, 136, 138-155] |
| 2021 | RpBERT-collapse (AAAI)                                       | L. Sun et al.     | 基于文本-图像关系传播机制，通过**崩溃标签**方式应用于 JMASA 任务（基线） [41, 83, 156, 157]。 | [41, 83, 156, 157]                                           |
| 2022 | Cross-Modal Multitask Transformer (CMMT) (IPM)               | L. Yang et al.    | **跨模态多任务 Transformer**，设计文本引导的跨模态交互模块和动态门控机制，实现端到端 MABSA [41, 57, 61, 74, 75, 85, 102, 115, 148, 149, 151, 158-160]。 | [41, 57, 61, 74, 75, 85, 102, 115, 148, 149, 151, 158-160]   |
| 2022 | Vision-Language Pre-training for MABSA (VLP-MABSA) (ACL)     | Y. Ling et al.    | 基于 **BART 架构** 的统一编码器-解码器，设计任务特定的 VLP 任务（如 AOE, AOG）来改善 MABSA 三个子任务 [41, 56, 70, 82, 107, 139, 147-149, 154, 158, 161-169]。 | [41, 56, 70, 82, 107, 139, 147-149, 154, 158, 161-169]       |
| 2022 | Dual-Encoder Transformers with Cross-Modal Alignment (DTCA) (AACL) | Z. Yu et al.      | 双编码器 Transformer 架构，通过跨模态对齐机制进行特征融合 [167, 170-174]。 | [167, 170-174]                                               |
| 2022 | Hierarchical Interactive Multimodal Transformer (HIMT) (IEEE T Affec Comput) | J. Yu et al.      | 采用层次交互机制推断情感极性（涉及 ATMSC 和 ACMSC） [74, 97-99, 102, 109, 154, 175-178]。 | [74, 97-99, 102, 109, 154, 175-178]                          |
| 2023 | Aspect-oriented Method (AoM) (ACL Findings)                  | R. Zhou et al.    | 提出 **方面感知注意力模块 (A3M)** 和 **GCN**，用于减轻视觉和文本噪声，检测方面相关语义和情感信息 [41, 75, 83, 147-149, 151, 171, 172, 179-187]。 | [41, 75, 83, 147-149, 151, 171, 172, 179-187]                |
| 2023 | Cross-modal Fine-grained Alignment and Fusion Network (CoolNet) (IPM) | L. Xiao et al.    | 引入图通用卷积模块和 Vision GNN，动态对齐文本句法和视觉对象特征 [61, 92, 109, 148, 188]。 | [61, 92, 109, 148, 188]                                      |
| 2023 | Dual-Perspective Fusion Network (DPFN) (IEEE TMM)            | D. Wang et al.    | 结合 **全局语义提取** （整体情感倾向）和 **局部句法增强** 进行情感预测 [189]。 | [189]                                                        |
| 2023 | Multi-grained Multi-curriculum Denoising Framework (M2DF) (EMNLP) | F. Zhao et al.    | 采用**课程学习**方法，无需设置阈值即可通过调整训练数据顺序实现降噪 [70, 76, 158, 186, 187, 190-194]。 | [70, 76, 158, 186, 187, 190-194]                             |
| 2023 | Few-shot Joint Multimodal Aspect-Sentiment Analysis Based on Generative Multimodal Prompt (ACL Findings) | X. Yang et al.    | 在少样本场景下，提出 **生成式多模态提示 (GMP)** 模型，通过预测方面数量和生成方面导向提示解决 JMASA 和 MATE 挑战 [131, 195-198]。 | [131, 195-198]                                               |
| 2024 | Aspect-guided Multi-view Interactions and Fusion Network (AMIFN) (Neurocomputing) | J. Yang et al.    | 采用方面引导、图像门和基于 GCN 的多知识交互，用于 MABSA [118, 187, 188, 199, 200]。 | [118, 187, 188, 199, 200]                                    |
| 2024 | Multi-model Co-guided Progressive Learning (MCPL) (KBS)      | J. Zhang et al.   | 提出 **多模型协同引导渐进式学习策略**，利用任务相关性（MATE -> MASC -> JMASA）生成伪标签并进行渐进式学习 [101, 129, 185]。 | [101, 129, 185]                                              |
| 2024 | Target-oriented Multi-grained Fusion Network (TMFN) (LREC-COLING) | D. Wang et al.    | 提出 **目标导向的多粒度融合网络**，为 MATE 和 MASC 子任务融合不同粒度的图像信息 [129, 201, 202]。 | [129, 201, 202]                                              |
| 2024 | Vanessa: Visual Connotation and Aesthetic Attributes Understanding Network (EMNLP Findings) | L. Xiao et al.    | 首次探索图像的**视觉内涵和美学属性**，采用多美学属性聚合模块和自监督对比学习增强对齐 [129, 191, 203]。 | [129, 191, 203]                                              |
| 2024 | Atlantis: Aesthetic-oriented Multiple Granularities Fusion Network (IF) | L. Xiao et al.    | 提出 **美学导向的多粒度融合网络**，用于解决 JMASA 任务中图像和文本的多粒度融合问题 [59, 61, 92, 103, 119, 129, 148, 149, 173, 197, 204]。 | [59, 61, 92, 103, 119, 129, 148, 149, 173, 197, 204]         |
| 2024 | Aspects are Anchors: Aspect-driven Alignment and Refinement (ADAR) (ACM MM) | Z. Chen et al.    | 采用 **最优传输 (OT)** 和 **自适应过滤箱** 的两阶段粗-细粒度对齐框架，解决嘈杂对应问题 (NCP) [129, 151, 187, 205]。 | [129, 151, 187, 205]                                         |
| 2024 | PURE: Personality-Coupled Multi-Task Learning Framework (IEEE TKDE) | P. Zhang et al.   | **首次引入用户潜在个性特征**，提出个性耦合多任务学习框架 [129, 206]。 | [129, 206]                                                   |
| 2025 | VLHA: Vision and Language Hierarchical Alignment (Pat Rec)   | W. Zou et al.     | 采用结构对齐模块（动态对齐视觉场景图与文本依存图）和语义对齐模块，实现 **层次化对齐** [158, 207, 208]。 | [158, 207, 208]                                              |
| 2025 | AESAL: Aspect Enhancement and Syntactic Adaptive Learning (IJCAI) | L. Zhu et al.     | 提出方面增强预训练和句法自适应学习机制，通过多通道自适应 GCN 融合模态相关性 [74, 209, 210]。 | [74, 209, 210]                                               |
| 2025 | COnditional Relation based Sentiment Analysis (CORSA) (COLING) | X. Liu et al.     | 设计 **条件关系检测器 (CRD)**，解决**图像不包含方面提及对象**的条件不满足问题 [153, 182, 211-214]。 | [153, 182, 211-214]                                          |
| 2025 | Dual-Aware Enhanced Alignment Network (DaNet) (ACL Findings) | A. Zhu et al.     | 引入多模态降噪编码器、双感知对齐模块和 **LLM 引导的隐式方面观点生成 (IAOG) 预训练**，处理隐式方面和噪声 [214, 215]。 | [214, 215]                                                   |
| 2025 | Descriptions Enhanced Question-Answering Framework (DEQA)    | Z. Han et al.     | 利用 **GPT-4 生成图像描述**，将 MABSA 建模为多轮问答任务，并采用多专家集成决策 [216]。 | [216]                                                        |
| 2025 | Dynamic Multi-Relation Cross-Fusion Network (DMR-XNet) (ICMR) | F. Zhou and Z. Li | 提出动态超图多关系捕获方法，学习多对多的多模态关联，并设计双流交叉激发层 [217]。 | [217]                                                        |
| 2025 | Dual-Branch Sentiment Enhancement Modeling (DSEM) (ICMR)     | X. Ji et al.      | 文本分支利用句法和词性结构，视觉分支使用 Gumbel-Softmax 识别和聚合情感相关补丁 [218]。 | [218]                                                        |
| 2025 | Aspect Enhancement and Text Simplification (AETS)            | L. Zhu et al.     | 针对多方面多情感场景，提出 **文本简化模块**，基于句法依存结构重构文本，以消除冗余信息和复杂句法结构干扰 [219]。 | [219]                                                        |
| 2025 | Tri-Encoder with Caption Prompt (TECP) (JSC)                 | Y. Cai et al.     | 提出**三编码器多任务学习框架**，设计图像字幕编码器和依存关系权重注意力网络，以充分利用潜在视觉语义信息 [220]。 | [220]                                                        |



### 3.4 Multi-Modal Aspect/Entity-Category-Sentiment Triple Extraction (MECSTE)



| 年份 | 论文/方法 (来源会议/期刊)                                    | 作者 (部分)  | 核心方法/机制                                            | 来源  |
| ---- | ------------------------------------------------------------ | ------------ | -------------------------------------------------------- | ----- |
| 2022 | Hierarchical Interactive Multimodal Transformer (HIMT) (IEEE T Affec Comput) | J. Yu et al. | 处理 **方面-类别** 级多模态情感分类 (ACMSC) 任务 [221]。 | [221] |





**2025–2026 年**“**MABSA（Multimodal Aspect-Based Sentiment Analysis）/ABMSA/多模态方面级情感**（含 joint MABSA 与 MASC 等子任务）”相关论文。

> 说明：不同论文会用 **MABSA / ABMSA / MASC / multimodal aspect sentiment** 等不同叫法；另外有的论文 **DOI 是 2025，但期刊卷期是 2026**（算 2026）。因此我按“**期刊标注年份/卷期年份**”来分组。

------

## 2026 年（目前能检索到的）

1. **Text-Dominant Speech-Enhanced for Multimodal Aspect-Based Sentiment Analysis network** — *Information Fusion*, Vol.126, 103543（标注出版日期 2026-02-01，DOI: 10.1016/j.inffus.2025.103543） 
2. **Semantic-Guided Multi-Grained Cross-Modal Alignment and Fusion Network for Multimodal Aspect-Based Sentiment Analysis** — *Information Fusion*, Vol.127, 103878（标注出版日期 2026-03-01，DOI: 10.1016/j.inffus.2025.103878） 
3. **Enhancing multi-modal aspect-based sentiment classification via emotional semantic-aware cross-modal relation inference** — *Information Processing & Management*, 63(2): 104427 (2026) 

------

## 2025 年（期刊论文）

1. **A vision and language hierarchical alignment for multimodal aspect-based sentiment analysis** — *Pattern Recognition*（2025） 
2. **Multi-axis fusion with optimal transport learning for multimodal aspect-based sentiment analysis** — *Expert Systems with Applications*（2025） 
3. **Dual perspective hierarchical alignment network for joint multimodal aspect-based sentiment analysis** — *Neurocomputing*（Pub Date 标注 2025-12-18，DOI: 10.1016/j.neucom.2025.132466） 
4. **Multimodal aspect-based sentiment analysis based on a dual syntactic graph network and joint contrastive learning** — *Knowledge and Information Systems*（2025） 
5. **Dual-layer contrastive learning for aspect-aligned multimodal sentiment analysis** — *Applied Intelligence*（2025） 
6. **Multimodal Aspect-Based Sentiment Analysis with External Knowledge and Multi-Granularity image-text features (EKMG)** — *Neural Processing Letters*（2025） 
7. **Multimodal aspect-based sentiment analysis enhanced with semantic–syntactic graph balancing and dynamic image selection**（SGBIS）—（DOI: 10.1016/j.knosys.2025.115108；ScienceDirect 页面可见摘要信息） 
8. **A prompt-based dual-layer cross-modal distillation learning method for aspect-based sentiment analysis** — *Multimedia Systems*（2025，DOI: 10.1007/s00530-025-01963-7） 
9. **Aspect-based multimodal sentiment analysis via employing visual-to-emotional-caption translation network using visual-caption pairs** — *Language Resources and Evaluation*（2025） 
10. **Exploring Cognitive and Aesthetic Causality for Multimodal Aspect-Based Sentiment Analysis**（Chimera 方向，DBLP 收录为 2025 期刊条目） 

------

## 2025 年（会议 / 论文集）

1. **Multimodal Aspect-Based Sentiment Analysis under Conditional Relation (CORSA)** — *COLING 2025* 
2. **Aspect-Image Counterfactual Integration for Multimodal Aspect Sentiment Classification (DPCI)** — *EMNLP 2025* 
3. **Multimodal Aspect-Based Sentiment Analysis via Aspect-Guided Pseudo-text Generation** — *ICONIP 2025*（Springer CCIS） 

------

## 2025 年（arXiv 预印本 / 技术报告类）

1. **Chimera: Cognitive and Aesthetic Causality for Multimodal Aspect-Based Sentiment Analysis** — arXiv（2025 版本） 
2. **AdaptiSent: Adapting Vision-Language Models for Multimodal Aspect-Based Sentiment Analysis** — arXiv 
3. **GateMABSA: Multimodal Aspect-Based Sentiment Analysis Enhanced with Gated Selective Fusion** — arXiv 
4. **Enhanced Multimodal Aspect-Based Sentiment Analysis by LLM-Generated Rationales (LRSA)** — arXiv 
5. **CLAMP: Enhancing MABSA with CLIP and LLM Collaboration** — arXiv 