[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-b.svg)](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/index.html)

[视频](https://www.bilibili.com/video/BV1rQK8zQEo2/?vd_source=2a13aee779bc6301268e18d749a04db4)

[基于统计的强化学习控制--源起](https://zhuanlan.zhihu.com/p/1924906978565654310)

[基于统计的强化学习控制--开篇](https://zhuanlan.zhihu.com/p/1925295102420579064)


# Citation
Please use the following bibtex if you find this repo helpful and would like to cite:

```bibtex
@misc{HUMANOID,
  author = {liangjun},
  title = {基于多粒度统计的机器人运动控制学习研究},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/zengliangjun/HUMANOID}},
}
```

```bibtex
@misc{CARTPOLE,
  author = {liangjun},
  title = {基于统计的强化学习控制},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/zengliangjun/SRL_CARTPOLE}},
}
```

```
@article{mittal2023orbit,
   author={Mittal, Mayank and Yu, Calvin and Yu, Qinxi and Liu, Jingzhou and Rudin, Nikita and Hoeller, David and Yuan, Jia Lin and Singh, Ritvik and Guo, Yunrong and Mazhar, Hammad and Mandlekar, Ajay and Babich, Buck and State, Gavriel and Hutter, Marco and Garg, Animesh},
   journal={IEEE Robotics and Automation Letters},
   title={Orbit: A Unified Simulation Framework for Interactive Robot Learning Environments},
   year={2023},
   volume={8},
   number={6},
   pages={3740-3747},
   doi={10.1109/LRA.2023.3270034}
}
```



# 基于多粒度统计的机器人运动控制学习研究
**作者**：  曾良军<sub>1</sub>，陈小波，费越<sub>1</sub>，陈宏力<sub>2</sub>

1:复旦大学义乌研究院人工智能与多媒体实验室
2:江西应用科技学院


本文提出了一种基于统计特征的强化学习控制框架，通过实时计算运动状态的统计特性(均值、方差等)来增强传统强化学习算法。该方法采用双通道架构，分别将统计特征用于观测构建和奖励计算。本研究为强化学习控制提供了新的特征工程思路；相比传统方法具有以下优势：
1. 策略性能优越：在复杂地形的训练时间和适应能力均表现得更好（无需额外传感器）。
2. 系统鲁棒性强：统计对于噪声的天然过滤与规律提取能力。
3. 系统具有良好的自适应能力，无需目前很方法中设定相位和步频信息。
4. 本方法可自由扩展任一现有或将实现架构；并能提供助力。

### 贡献
1. 提出增量式多粒度统计特征体系
2. 设计双通道统计RL架构
3) 实现运动状态自适应的基于统计的奖励机制

## 1. 引言

### 1.1 研究背景与范式创新
当前强化学习在机器人控制中存在三大挑战：
1. 奖励稀疏性导致训练效率低下
2. 状态表征能力不足限制策略性能
3. 系统适应性差难以应对动态环境

目前止还没有基于历史状态用于奖励塑形的方面的实践；在己查阅资源中应该是在这个方向上第一个进行相关设计和实践。

### 1.2 "统计特征双通道"新范式

```mermaid
graph TB
    subgraph 原始状态
        A[原始观测]
    end
    subgraph 统计状态
        A --> B[Step级统计]
        A --> C[Episode级统计]
    end
    subgraph 决策层
        B --> D[策略网络]
        C --> E[价值网络]
        C --> D
        B --> E
    end
```

## 2. 方法论

### 2.1 统计特征计算与状态表征
本方法创新性地将统计特征同时用于奖励计算和状态表征：

1. **双重功能设计**：
   - 奖励计算：作为评估指标
   - 状态表征：作为观测信号
   ```mermaid
   graph LR
     A[原始数据] --> B[统计特征]
     B --> C[奖励计算]
     B --> D[状态观测]
   ```

2. 统计计算：
采用改进的Welford算法实现实时统计：

   - **均值更新**：
   $$
   \Delta = x_t - \mu_{t-1} \\
   \mu_t = \mu_{t-1} + \Delta/t
   $$

   - **方差更新**：
   $$
   \sigma_t^2 = \frac{(t-1)\sigma_{t-1}^2 + \Delta(x_t-\mu_t)}{t}
   $$


### 2.2 奖励函数设计
基于统计特征，本方法构建了四种奖励计算策略：

- **均值自比较**：
  使用公式
  $$
  R=\frac{1}{C}\sum\exp(-\|\mu_i-\mu_j\|/\sigma)
  $$
  计算不同关节间的均值差异，适用于评估对称性运动；

- **均值零比较**：
  $$
  R=\frac{1}{N}\sum\exp(-\|\mu_i\|/\sigma)
  $$
  对比关节均值与零的偏差，适用于静止或平衡状态；

- **方差自比较**：
  使用公式
  $$
  R_{var} = \exp\Big(-\Big(\frac{\|\sigma\|-\sigma_{target}}{\sigma_{target}}\Big)^2\Big)
  $$
  对运动平滑性进行评估，其中 \( \sigma \) 为实际统计方差，\( \sigma_{target} \) 为预设目标方差；

- **均值与方差组合**：
  综合上述两种奖励，得到总奖励值
  $$
  R = \frac{r_{mean}+ \lambda\, r_{var}}{1+\lambda}
  $$

  在该公式中，\( \ lambda \) 为权重系数，通过调节可适应不同任务需求。总体设计确保奖励信号能同时反映局部运动波动与全局运动趋势。


## 3. 结论与范式意义
本文提出的"统计特征双通道"框架开创了强化学习新范式：

1. **范式创新**：
   - 统一了奖励计算和状态表征的统计基础
   - 建立了从原始数据到策略优化的双通道架构
   - 实现了感知-决策的闭环优化

2. 框架价值：

1. **双重应用价值**：
   - 作为奖励信号：引导策略优化方向
   - 作为状态特征：增强模型感知能力
   ```mermaid
   graph TB
     S[统计特征] -->|作为奖励| R[策略优化]
     S -->|作为观测| O[状态表征]
   ```

2. 框架优势：

1. **性能优势**：
   - 训练阶段：收敛速度更快
   - 系统开销：

2. **技术特色**：
   - "统计特征双通道"新范式

3. **应用价值**：
   - 可应用于四足机器人、双足机器人等控制系统

## 参考文献
1. Sutton R., Barto A. Reinforcement Learning: An Introduction
2. Silver D., et al. Mastering the game of Go without human knowledge
3. Peng X.B., et al. DeepMimic: Example-Guided Deep Reinforcement Learning
