# Gradient-Balanced Adaptive Retention for Federated Continual Learning under Non-IID Data

## Abstract

Federated continual learning (FCL) requires distributed models to acquire knowledge from sequentially arriving tasks
while preserving previously learned capabilities under heterogeneous client data. This setting combines statistical
heterogeneity across clients with temporal interference across tasks, making catastrophic forgetting a central
challenge. In this work, we propose Gradient-Balanced Replay+LwF, an adaptive retention mechanism that uses the relative
magnitudes of current-task and replay-memory gradients to dynamically control the strength of knowledge distillation.
The method maps a gradient-balance score to a bounded retention coefficient and integrates it with replay and Learning
without Forgetting during local federated training. We evaluate the approach on a five-task class-incremental CIFAR-10
protocol with five non-IID clients. Across five matched random seeds, Gradient-Balanced FCL reduces average forgetting
from 4.76 ± 0.76 percentage points for fixed Replay+LwF to 1.77 ± 0.34 percentage points, corresponding to a 62.89%
relative reduction. This improvement is consistent across all five tested seeds but is accompanied by a decrease in
final average accuracy from 46.94 ± 3.40% to 44.76 ± 3.50%, revealing a stability-plasticity trade-off. Seed-42
experiments across Dirichlet concentration parameters of 0.1, 0.5, and 1.0 also show lower forgetting than the fixed
baseline at each tested level of heterogeneity. The results suggest that gradient magnitude provides a useful signal for
adaptive retention in the investigated FCL setting, while highlighting the need for broader evaluation and mechanisms
that preserve retention gains with a smaller cost to plasticity.

**Keywords:** Federated continual learning, federated learning, continual learning, catastrophic forgetting, non-IID
data, knowledge distillation, experience replay, gradient-based adaptation.

## 1. Introduction

Federated learning (FL) enables multiple distributed clients to collaboratively train a shared model while keeping their
local data decentralized [1]. This paradigm is particularly attractive in settings where data cannot be freely
transferred because of privacy, communication, or organizational constraints. However, conventional federated learning
commonly assumes that the learning objective remains relatively stable throughout training. In many real-world
environments, clients instead encounter evolving data distributions and sequential learning tasks, requiring models to
continually acquire new knowledge over time.

Federated continual learning (FCL) combines the distributed nature of federated learning with sequential learning from
evolving local data streams [6]. This setting introduces interacting challenges including client heterogeneity, model
stability, communication constraints, and catastrophic forgetting [10]. Related federated class-incremental settings
further demonstrate that both local and global models can suffer catastrophic forgetting as new classes arrive over
time [7]. As clients learn new tasks, model updates that improve performance on newly observed data may interfere with
representations acquired from previous tasks. At the same time, statistical heterogeneity across clients can cause local
optimization directions to differ substantially, making the preservation of previously learned knowledge more difficult
after federated aggregation.

Several continual-learning mechanisms can be incorporated into federated learning to mitigate forgetting. Experience
replay retains a limited memory of previously observed examples and reintroduces them during subsequent training.
Knowledge-distillation approaches such as Learning without Forgetting (LwF) [2] encourage the current model to preserve
behavior learned by an earlier model. Gradient-projection approaches such as GEM [3] and A-GEM [4] provide another
strategy by constraining or modifying interfering optimization directions. Nevertheless, the appropriate strength of
knowledge retention is unlikely to remain constant throughout federated continual training. The degree of interference
can vary across tasks, communication rounds, and heterogeneous client distributions.

This work investigates whether gradient information can be used to adapt the strength of retention dynamically in
federated continual learning. We first examine standard FedAvg-based FCL, replay, LwF, and their combination under a
class-incremental CIFAR-10 protocol with non-IID client partitions. Diagnostic experiments are then used to investigate
relationships between gradient behavior and subsequent forgetting. These observations motivate a Gradient-Balanced
Replay+LwF mechanism in which the retention coefficient is adjusted dynamically according to the relative gradient
magnitudes associated with retaining previous knowledge and learning the current task.

The proposed approach is evaluated against fixed Replay+LwF across five random seeds and is additionally examined under
multiple levels of Dirichlet client heterogeneity. An adapted Fed-A-GEM gradient-projection method [5] is also evaluated
under the same experimental protocol to provide an additional gradient-based continual-learning baseline. The
experiments reveal a clear stability-plasticity trade-off. Gradient-Balanced FCL reduces average forgetting from 4.76
percentage points with fixed Replay+LwF to 1.77 percentage points, corresponding to a 62.89% relative reduction in
forgetting across five seeds. This improvement in retention is accompanied by a 2.18 percentage-point reduction in final
average accuracy. Furthermore, lower forgetting is observed for the Gradient-Balanced method at each of the three tested
Dirichlet concentration values (\(\alpha=0.1,0.5,1.0\)) in the heterogeneity analysis.

The main contributions of this work are as follows:

1. We develop a gradient-balanced adaptive retention mechanism for federated continual learning that dynamically adjusts
   the strength of knowledge retention rather than using a fixed distillation coefficient.
2. We conduct diagnostic analyses of gradient behavior and catastrophic forgetting, providing empirical motivation for
   gradient-dependent retention in the investigated FCL setting.
3. We evaluate the proposed approach against FedAvg-FCL, replay, LwF, fixed Replay+LwF, and an adapted Fed-A-GEM
   gradient-projection baseline under a common class-incremental federated protocol.
4. We perform a five-seed comparison showing that Gradient-Balanced FCL reduces average forgetting relative to fixed
   Replay+LwF for all five tested seeds, with a 62.89% reduction in mean forgetting, while explicitly quantifying the
   associated reduction in final average accuracy.
5. We examine sensitivity to client heterogeneity at Dirichlet \(\alpha\) values of 0.1, 0.5, and 1.0 and analyze the
   learned adaptive retention weights across tasks, communication rounds, and clients.

## 2. Related Work

### 2.1 Federated Learning

Federated learning enables multiple clients to collaboratively optimize a shared model while keeping training data
decentralized. FedAvg, introduced by McMahan et al. [1], established a widely used federated optimization framework in
which clients perform local optimization and a central server periodically aggregates their model updates. Although
FedAvg was designed to operate with decentralized and potentially non-IID data, conventional federated learning
primarily addresses collaborative optimization rather than sequential acquisition and retention of knowledge across
changing tasks.

Federated continual learning extends this setting by introducing temporal non-stationarity. Clients may encounter new
classes, tasks, or distributions over time, requiring the global model to acquire new knowledge without catastrophically
overwriting information learned previously. Consequently, FCL combines challenges arising from statistical heterogeneity
across clients and temporal interference across tasks [6], [7], [10].

### 2.2 Continual Learning and Catastrophic Forgetting

Continual learning studies models that acquire knowledge sequentially while attempting to preserve performance on
previously encountered tasks. A central difficulty is catastrophic forgetting, in which learning new tasks can
substantially degrade performance on previously learned tasks [8].

Experience replay is a widely used strategy for mitigating this problem. A limited memory of previously observed
examples is retained and mixed with current-task data during subsequent optimization. Replay provides the learner with
direct access to samples representing earlier knowledge, but its effectiveness depends on factors such as memory
capacity, sampling strategy, and the interaction between old and new examples.

Knowledge-distillation methods provide another approach to knowledge retention. Learning without Forgetting (LwF),
introduced by Li and Hoiem [2], preserves previously learned behavior by encouraging the updated model to reproduce
responses generated by an earlier model while learning new information. This allows retention to be expressed as a
functional constraint rather than solely as a parameter-level regularizer.

In our experimental setting, replay and LwF individually provide limited protection against forgetting, whereas their
combination forms a substantially stronger baseline. This motivates our use of Replay+LwF as the foundation on which
adaptive retention is investigated.

### 2.3 Gradient-Based Continual Learning

Gradient-based continual-learning approaches attempt to reduce interference by explicitly modifying optimization
directions. Gradient Episodic Memory (GEM), proposed by Lopez-Paz and Ranzato [3], uses episodic memories from previous
tasks and constrains parameter updates so that they do not increase losses on stored past-task examples.

Averaged Gradient Episodic Memory (A-GEM), introduced by Chaudhry et al. [4], provides a more computationally efficient
approximation. Rather than enforcing separate constraints for multiple previous tasks, A-GEM constructs a reference
gradient from memory and projects the current gradient when it conflicts with this reference direction.

These methods demonstrate that gradient geometry contains useful information about interference between old and new
knowledge. Our work differs in that gradient information is used primarily to adapt the strength of a retention
objective rather than solely to project the optimization direction.

### 2.4 Gradient Projection in Federated Continual Learning

Gradient-projection mechanisms have also been extended to federated continual learning. Fed-A-GEM [5] adapts the A-GEM
principle to the federated setting using buffered samples and aggregated buffer-gradient information to alleviate
catastrophic forgetting across clients.

In this work, we implement an adapted Fed-A-GEM baseline under the same class-incremental protocol, model architecture,
client configuration, memory budget, and federated-round structure used for our other methods. Because this
implementation adapts the gradient-projection mechanism to the experimental framework of this repository rather than
reproducing every aspect of the original Fed-A-GEM experimental system, we refer to it explicitly as Adapted Fed-A-GEM.

Our proposed Gradient-Balanced approach takes a different direction. Instead of using gradient conflict exclusively as a
projection constraint, we investigate whether gradient information can determine how strongly previously learned
knowledge should be retained at different stages of federated continual training. The resulting method dynamically
adjusts the Replay+LwF retention coefficient according to gradient-balance information.

### 2.5 Federated Continual and Class-Incremental Learning

Several approaches have been developed specifically for continual or class-incremental learning in federated
environments. FedWeIT [6] addresses federated continual learning through parameter decomposition and weighted
inter-client transfer, allowing clients to selectively exploit task-specific knowledge learned by other participants.
This illustrates that knowledge preservation in FCL involves not only temporal interference within individual clients
but also transfer and aggregation of potentially heterogeneous knowledge across clients.

Global-Local Forgetting Compensation (GLFC) [7] considers federated class-incremental learning, where new classes arrive
over time and both local and global models are susceptible to catastrophic forgetting. GLFC introduces mechanisms that
compensate for forgetting from both local and global perspectives, highlighting the additional difficulty created when
class-incremental learning is combined with decentralized training.

TARGET [9] investigates federated class-continual learning from an exemplar-free perspective. Instead of retaining real
examples from previous tasks, it transfers knowledge from a previously trained global model and uses generated synthetic
data to approximate the global data distribution. This provides a contrasting approach to the replay-based setting
considered in the present work, where clients explicitly maintain bounded memories of previously observed examples.

Recent FCL literature also emphasizes that catastrophic forgetting cannot be considered independently of
federated-system constraints. A recent survey identifies client heterogeneity, model stability, communication overhead,
and privacy preservation as central challenges in FCL [10]. Resource-constrained evaluation further shows that storage
and computational budgets can substantially affect the practical performance of existing FCL techniques [11]. These
considerations motivate evaluating not only retention performance but also the additional computational requirements
introduced by adaptive mechanisms.

Recent work has additionally questioned whether preservation of all historical knowledge should always be treated as
desirable under strongly heterogeneous task distributions. Accurate Forgetting [12], for example, considers situations
in which previous knowledge may be biased, unrelated, or antagonistic and investigates selective use of historical
information. This perspective further illustrates that the appropriate balance between retention and adaptation can
depend on the learning environment.

The present work focuses specifically on a replay-and-distillation setting and asks whether the distillation strength
itself can be controlled using the relative magnitudes of gradients associated with current-task and replay-memory data.
Unlike projection-based methods, the proposed controller does not modify gradient direction. Instead, it uses gradient
magnitude as a local signal for dynamically adjusting a bounded knowledge-distillation coefficient.

## 3. Problem Formulation

### 3.1 Federated Continual Learning Setting

Consider a federated learning system consisting of \(K\) clients coordinated by a central server. Training proceeds over
a sequence of \(T\) continual-learning tasks,

\[
\mathcal{T}_1,\mathcal{T}_2,\ldots,\mathcal{T}_T.
\]

For task \(t\), client \(k\) receives a local dataset

\[
\mathcal{D}_{k}^{t}
=
\{(x_i,y_i)\}_{i=1}^{n_k^t},
\]

where \(n_k^t\) denotes the number of training examples available to client \(k\) for task \(t\).

The local data distributions are generally non-identically distributed across clients. In the experiments considered in
this work, client heterogeneity is generated using Dirichlet partitioning. Smaller values of the Dirichlet concentration
parameter \(\alpha\) produce stronger statistical heterogeneity among clients.

The objective is to learn a sequence of global models while maintaining performance on previously encountered tasks.
After learning task \(t\), the global model should therefore perform well not only on \(\mathcal{T}_t\), but also on
previously learned tasks.

This requirement creates a stability-plasticity problem: the model must remain sufficiently plastic to learn new tasks
while remaining sufficiently stable to preserve previously acquired knowledge.

### 3.2 Federated Optimization

Let

\[
\theta_t^r
\]

denote the global model parameters during communication round \(r\) of task \(t\).

At the beginning of a communication round, the server distributes the current global model to participating clients.
Each client performs local optimization using its task-specific data and returns updated model parameters to the server.

For client \(k\), the standard current-task objective can be written as

\[
\mathcal{L}_{\mathrm{new}}^{k,t}(\theta)
=
\frac{1}{|\mathcal{D}_{k}^{t}|}
\sum_{(x,y)\in\mathcal{D}_{k}^{t}}
\ell(f_{\theta}(x),y),
\]

where \(f_{\theta}\) denotes the model and \(\ell\) is the classification loss.

Following local optimization, the server aggregates client models using weighted averaging:

\[
\theta_t^{r+1}
=
\sum_{k=1}^{K}
\frac{n_k^t}
{\sum_{j=1}^{K}n_j^t}
\theta_{k,t}^{r+1}.
\]

In a continual-learning setting, minimizing only the current-task objective can produce parameter updates that interfere
with representations required by previous tasks. Consequently, additional retention mechanisms are required.

### 3.3 Replay and Knowledge-Distillation Retention

Each client maintains a bounded replay memory

\[
\mathcal{M}_k
\]

containing examples from previously observed tasks.

Replay introduces a memory-based objective

\[
\mathcal{L}_{\mathrm{replay}}^{k,t},
\]

computed using samples drawn from \(\mathcal{M}_k\).

In addition, Learning without Forgetting uses a previous model as a teacher. Let

\[
f_{\theta_{\mathrm{old}}}
\]

denote the frozen model representing knowledge available before learning the current task. A distillation loss

\[
\mathcal{L}_{\mathrm{LwF}}^{k,t}
\]

encourages the current model to preserve the output behavior of the previous model.

A fixed Replay+LwF objective can therefore be expressed conceptually as

\[
\mathcal{L}_{\mathrm{fixed}}
=
\mathcal{L}_{\mathrm{new}}

+

\mathcal{L}_{\mathrm{replay}}

+

\lambda\mathcal{L}_{\mathrm{LwF}},
\]

where \(\lambda\) controls the strength of knowledge retention.

The central motivation of this work is that a single fixed value of \(\lambda\) may not be appropriate throughout the
complete federated continual-learning process.

### 3.4 Continual-Learning Evaluation

Let

\[
A_{t,j}
\]

denote the test accuracy on task \(j\) after the model has completed training through task \(t\), where \(j\leq t\).

The resulting lower-triangular accuracy matrix is

\[
A=
\begin{bmatrix}
A_{1,1} & & & \\
A_{2,1} & A_{2,2} & & \\
\vdots & \vdots & \ddots & \\
A_{T,1} & A_{T,2} & \cdots & A_{T,T}
\end{bmatrix}.
\]

Final average accuracy is defined as

\[
\mathrm{ACC}
=
\frac{1}{T}
\sum_{j=1}^{T}A_{T,j}.
\]

For each task \(j<T\), forgetting is measured as the non-negative reduction from its best observed accuracy during
continual training to its final accuracy:

\[
F_j
=
\max
\left(
0,
\max_{t\in\{j,\ldots,T\}}A_{t,j}

-

A_{T,j}
\right).
\]

Average forgetting is then

\[
\mathrm{F}
=
\frac{1}{T-1}
\sum_{j=1}^{T-1}F_j.
\]

Lower values of \(\mathrm{F}\) indicate stronger retention of previously learned tasks, while higher final average
accuracy indicates stronger overall performance after the complete task sequence. These metrics are considered jointly
because reducing forgetting can come at the cost of reduced plasticity on newer tasks.

## 4. Gradient-Balanced Adaptive Retention

### 4.1 Motivation

The fixed Replay+LwF baseline applies the same distillation coefficient throughout continual training. However, the
relative optimization pressure associated with current-task learning and previously learned knowledge can change as
training progresses.

We introduce Gradient-Balanced adaptive retention to vary the distillation strength according to gradient information
measured before local optimization. The method estimates the relative gradient magnitudes associated with old-memory
data and current-task data and maps their balance to a bounded retention coefficient.

### 4.2 Old- and New-Knowledge Gradient Norms

For each client, let

\[
g_{\mathrm{old}}
=
\nabla_{\theta}\mathcal{L}_{\mathrm{old}}(\theta)
\]

denote the gradient computed using examples stored in the client's replay memory, and let

\[
g_{\mathrm{new}}
=
\nabla_{\theta}\mathcal{L}_{\mathrm{new}}(\theta)
\]

denote the gradient computed using current-task data.

Before local optimization begins, the client evaluates both gradients using the current global model and computes their
Euclidean norms:

\[
G_{\mathrm{old}}=\|g_{\mathrm{old}}\|_2,
\]

\[
G_{\mathrm{new}}=\|g_{\mathrm{new}}\|_2.
\]

In the implementation, the gradient norms are calculated over the corresponding data loaders using sample-weighted
cross-entropy gradients. Gradients are accumulated over available samples and normalized by the total number of samples
before the global parameter-gradient norm is computed.

### 4.3 Gradient-Balance Score

The relative contribution of the current-task gradient is summarized using

\[
b
=
\frac{G_{\mathrm{new}}}
{G_{\mathrm{old}}+G_{\mathrm{new}}+\epsilon},
\]

where \(\epsilon>0\) is a small numerical-stability constant.

Because both gradient norms are non-negative,

\[
0\leq b\leq1.
\]

When the current-task gradient is small relative to the old-memory gradient, the balance score approaches zero.
Conversely, when the current-task gradient dominates, the score approaches one.

The implementation additionally records the diagnostic gradient-magnitude ratio

\[
\rho
=
\frac{G_{\mathrm{old}}}
{G_{\mathrm{new}}+\epsilon}.
\]

The ratio \(\rho\) is used for analysis and diagnostics, whereas the adaptive retention coefficient itself is determined
from \(b\).

### 4.4 Adaptive Retention Weight

The balance score is linearly mapped to a bounded retention interval:

\[
\lambda_{\mathrm{GB}}
=
\lambda_{\min}

+

(\lambda_{\max}-\lambda_{\min})b.
\]

Consequently,

\[
\lambda_{\min}
\leq
\lambda_{\mathrm{GB}}
\leq
\lambda_{\max}.
\]

The main experiments use

\[
\lambda_{\min}=0.5,
\qquad
\lambda_{\max}=1.5.
\]

A larger current-task gradient therefore produces a larger balance score and a larger retention coefficient. The
mechanism responds to stronger current-task optimization pressure by increasing the strength of the distillation
constraint intended to preserve previous model behavior.

### 4.5 Local Training Objective

Current-task data and stored replay examples are combined during local training. Let

\[
\mathcal{L}_{\mathrm{CE}}
\]

denote the cross-entropy classification loss evaluated on the combined training batches.

For temperature \(\tau\), the teacher distribution is

\[
p_{\mathrm{teacher}}
=
\operatorname{softmax}
\left(
\frac{z_{\mathrm{teacher}}}{\tau}
\right),
\]

and the student log-probabilities are

\[
\log p_{\mathrm{student}}
=
\operatorname{logsoftmax}
\left(
\frac{z_{\mathrm{student}}}{\tau}
\right).
\]

The LwF distillation loss is

\[
\mathcal{L}_{\mathrm{KD}}
=
\tau^2
D_{\mathrm{KL}}
\left(
p_{\mathrm{teacher}}
\|
p_{\mathrm{student}}
\right).
\]

The Gradient-Balanced local objective is then

\[
\mathcal{L}_{\mathrm{GB}}
=
\mathcal{L}_{\mathrm{CE}}

+

\lambda_{\mathrm{GB}}\mathcal{L}_{\mathrm{KD}}.
\]

Thus, the structure of Replay+LwF is retained, but the fixed distillation coefficient is replaced by a
gradient-dependent coefficient calculated before local optimization.

For the first continual task, no previous replay memory or teacher model is available. In this case, the retention
coefficient is set to zero and training reduces to standard classification learning.

### 4.6 Client-Level Procedure

For each client and communication round after previous knowledge becomes available, Gradient-Balanced training proceeds
as follows:

1. Receive the current global model from the server.
2. Construct a loader from the client's stored replay memory.
3. Compute \(G_{\mathrm{old}}\) using replay-memory data.
4. Compute \(G_{\mathrm{new}}\) using current-task data.
5. Calculate the gradient-balance score \(b\).
6. Map \(b\) to the bounded retention coefficient \(\lambda_{\mathrm{GB}}\).
7. Train the local model using replay and the adaptive LwF objective.
8. Return the locally updated model to the server.
9. Aggregate client models using federated averaging.

The adaptive coefficient is calculated once before local optimization for the corresponding client update and is then
used throughout that local training call.

### 4.7 Computational Considerations

Compared with fixed Replay+LwF, the proposed method requires additional gradient calculations before local optimization.
Specifically, gradient norms are estimated separately from replay-memory data and current-task data.

The adaptive controller itself is lightweight: after the two gradient norms have been obtained, computing the balance
score and retention coefficient requires only scalar arithmetic. The primary additional cost therefore arises from the
diagnostic gradient passes rather than from the weight-mapping operation.

## 5. Experimental Setup

### 5.1 Dataset and Continual-Learning Protocol

Experiments are conducted on CIFAR-10 using a class-incremental federated continual-learning protocol. The ten classes
are divided into five sequential two-class tasks:

\[
\mathcal{T}_1=\{0,1\},
\]

\[
\mathcal{T}_2=\{2,3\},
\]

\[
\mathcal{T}_3=\{4,5\},
\]

\[
\mathcal{T}_4=\{6,7\},
\]

\[
\mathcal{T}_5=\{8,9\}.
\]

After completing each task, the global model is evaluated separately on the test subsets corresponding to all tasks
encountered up to that point.

### 5.2 Federated Non-IID Partitioning

The federated system contains five clients. For every continual task, training examples belonging to each of its two
classes are partitioned independently across clients using a Dirichlet distribution.

For a class \(c\), client proportions are sampled as

\[
(p_1,\ldots,p_K)
\sim
\operatorname{Dirichlet}(\alpha,\ldots,\alpha),
\]

where \(K=5\).

The primary five-seed comparison uses

\[
\alpha=0.5.
\]

Additional heterogeneity experiments use

\[
\alpha\in\{0.1,0.5,1.0\}
\]

at Seed 42.

For every task, partitioning is repeated when necessary until every client receives at least one training example.

### 5.3 Model Architecture

All compared methods use the same convolutional neural network.

The network contains two convolutional layers,

\[
3\rightarrow32\rightarrow64,
\]

using \(3\times3\) kernels and padding of one pixel. Each convolution is followed by ReLU activation and \(2\times2\)
max pooling.

For a \(32\times32\) CIFAR-10 input, the representation is flattened after the two pooling operations and passed through

\[
64\times8\times8
\rightarrow128\rightarrow10.
\]

The hidden fully connected layer uses ReLU activation.

### 5.4 Training Configuration

| Hyperparameter                   | Value |
|:---------------------------------|------:|
| Number of clients                |     5 |
| Number of continual tasks        |     5 |
| Classes per task                 |     2 |
| Batch size                       |    64 |
| Federated rounds per task        |     3 |
| Local epochs per round           |     1 |
| Learning rate                    | 0.001 |
| Replay-memory capacity           |   500 |
| LwF temperature                  |   2.0 |
| Fixed LwF coefficient            |   1.0 |
| Gradient-Balanced minimum weight |   0.5 |
| Gradient-Balanced maximum weight |   1.5 |
| Primary Dirichlet alpha          |   0.5 |

For Gradient-Balanced FCL, the adaptive retention coefficient is recalculated separately for each client update before
local optimization and held fixed throughout that local training call.

After each continual task, each client updates its replay memory using data from the task just learned. The teacher
model used for subsequent LwF distillation is then updated from the aggregated global model.

### 5.5 Compared Methods

The study considers:

1. **FedAvg-FCL:** federated continual training without an explicit continual-learning retention mechanism.
2. **Replay:** training with a bounded memory containing examples from previous tasks.
3. **LwF:** knowledge-distillation-based retention using a frozen previous model.
4. **Fixed Replay+LwF:** replay combined with LwF using \(\lambda=1.0\).
5. **Adapted Fed-A-GEM:** an adaptation of the Fed-A-GEM gradient-projection mechanism [5] to the common protocol.
6. **Gradient-Balanced Replay+LwF:** the proposed method.

The initial ablation comparison among FedAvg-FCL, Replay, LwF, and Replay+LwF is reported over three seeds.

The primary fixed-versus-Gradient-Balanced comparison uses

\[
\{42,123,2026,777,1001\}.
\]

Adapted Fed-A-GEM is also evaluated across these five seeds after selecting its learning rate using a Seed-42 sweep.

### 5.6 Adapted Fed-A-GEM Configuration

A learning-rate sweep is performed at Seed 42 using

\[
\eta\in\{0.001,0.01,0.1\}.
\]

Among these settings,

\[
\eta=0.001
\]

is selected for the five-seed evaluation.

Because the implementation adapts the Fed-A-GEM gradient-projection mechanism to the fixed protocol used in this study
rather than reproducing the complete original experimental system, results are explicitly labeled **Adapted Fed-A-GEM**.

### 5.7 Evaluation Metrics

Final average accuracy is calculated after Task 5 as

\[
\mathrm{ACC}
=
\frac{1}{T}
\sum_{j=1}^{T}A_{T,j}.
\]

For each previous task,

\[
F_j
=
\max
\left(
0,
\max_{t\geq j}A_{t,j}-A_{T,j}
\right).
\]

Average forgetting is

\[
\mathrm{F}
=
\frac{1}{T-1}
\sum_{j=1}^{T-1}F_j.
\]

Accuracy is reported as a percentage, while forgetting is reported in percentage points (pp).

For multi-seed experiments, results are summarized using mean and standard deviation.

### 5.8 Statistical Analysis

The five-seed comparison between fixed Replay+LwF and Gradient-Balanced FCL is treated as paired because both methods
use the same seeds.

For final average accuracy and average forgetting, we report the paired mean difference, a 95% confidence interval,
Cohen's \(d_z\), and an exact two-sided sign-flip test.

Because only five paired seeds are available, inferential power is limited. Statistical results are therefore
interpreted together with effect sizes, confidence intervals, and consistency across seeds.

### 5.9 Reproducibility

The implementation records numerical results, adaptive-weight diagnostics, and model checkpoints for evaluated
configurations.

For Gradient-Balanced FCL, diagnostic output records:

- old-memory gradient norm,
- current-task gradient norm,
- gradient-magnitude ratio,
- balance score, and
- adaptive retention weight.

The repository additionally maintains an experiment manifest linking experimental configurations to their corresponding
result and model artifacts.

At the current reproducibility checkpoint, the manifest contains 28 recorded experiments, with no missing result or
model artifacts in the local experimental record.

## 6. Results and Analysis

### 6.1 Ablation Study

We first examine the contribution of replay and knowledge distillation.

**Table 1. Ablation study over three seeds.**

| Method     | Final Average Accuracy | Average Forgetting |
|:-----------|-----------------------:|-------------------:|
| FedAvg-FCL |          16.42 ± 1.04% |    68.75 ± 2.84 pp |
| Replay     |          19.33 ± 4.72% |    65.55 ± 2.79 pp |
| LwF        |          19.89 ± 1.84% |    43.66 ± 5.93 pp |
| Replay+LwF |      **46.33 ± 4.08%** | **5.23 ± 0.56 pp** |

FedAvg-FCL exhibits severe catastrophic forgetting, with average forgetting of 68.75 pp and final average accuracy of
16.42%. Adding replay alone produces only a modest improvement.

LwF provides substantially stronger retention than replay alone, reducing average forgetting to 43.66 pp. Nevertheless,
final average accuracy remains below 20%.

Combining Replay and LwF produces a substantially stronger baseline. Final average accuracy increases to 46.33%, while
average forgetting decreases to 5.23 pp. Consequently, fixed Replay+LwF is used as the principal baseline for evaluating
the proposed adaptive mechanism.

### 6.2 Five-Seed Main Comparison

**Table 2. Main five-seed comparison.**

| Method            | Final Average Accuracy | Average Forgetting |
|:------------------|-----------------------:|-------------------:|
| Adapted Fed-A-GEM |          13.15 ± 0.45% |    60.00 ± 4.46 pp |
| Fixed Replay+LwF  |      **46.94 ± 3.40%** |     4.76 ± 0.76 pp |
| Gradient-Balanced |          44.76 ± 3.50% | **1.77 ± 0.34 pp** |

Gradient-Balanced FCL reduces average forgetting from 4.76 pp to 1.77 pp. The absolute reduction is approximately

\[
4.76-1.77=3.00
\]

percentage points, corresponding to

\[
\frac{4.76-1.77}{4.76}\times100
=
62.89\%.
\]

The retention improvement is accompanied by a reduction in final average accuracy from 46.94% to 44.76%, an average
difference of 2.18 pp.

The result therefore indicates a shift in the stability-plasticity trade-off toward stronger retention rather than
uniform dominance over fixed Replay+LwF.

### 6.3 Consistency Across Seeds

**Table 3. Per-seed comparison of fixed Replay+LwF and Gradient-Balanced FCL.**

| Seed | Fixed Accuracy | GB Accuracy | Fixed Forgetting | GB Forgetting |
|-----:|---------------:|------------:|-----------------:|--------------:|
|   42 |         46.57% |      45.77% |          4.80 pp |       2.01 pp |
|  123 |         42.14% |      40.22% |          5.03 pp |       1.85 pp |
| 2026 |         50.28% |      46.60% |          5.86 pp |       1.68 pp |
|  777 |         50.12% |      48.97% |          3.90 pp |       2.07 pp |
| 1001 |         45.60% |      42.26% |          4.23 pp |       1.23 pp |

Gradient-Balanced FCL produces lower average forgetting for all five tested seeds. At the same time, it produces lower
final average accuracy for each seed. This consistency reinforces the interpretation that the mechanism systematically
favors retention rather than simply producing an overall performance improvement.

### 6.4 Paired Statistical Analysis

For final average accuracy, the paired difference defined as Gradient-Balanced minus Fixed is

\[
\Delta_{\mathrm{ACC}}
=
-2.18\text{ pp},
\]

with a reported 95% confidence interval of

\[
[-3.78,-0.58]\text{ pp}
\]

and

\[
d_z=-1.692.
\]

For average forgetting, the Gradient-Balanced-minus-Fixed paired difference is approximately

\[
-3.00\text{ pp},
\]

with a reported 95% confidence interval of

\[
[-4.05,-1.94]\text{ pp}
\]

and

\[
d_z=-3.538.
\]

The exact two-sided sign-flip test gives

\[
p=0.0625
\]

for both metrics.

With only five paired observations, inferential statistical power is limited. We therefore emphasize effect magnitude
and consistency across seeds rather than interpreting the results through a conventional significance threshold alone.

### 6.5 Sensitivity to Client Heterogeneity

**Table 4. Heterogeneity analysis at Seed 42.**

| Dirichlet \(\alpha\) | Fixed Accuracy | GB Accuracy | Accuracy Difference | Fixed Forgetting | GB Forgetting | Forgetting Reduction |
|---------------------:|---------------:|------------:|--------------------:|-----------------:|--------------:|---------------------:|
|                  0.1 |         32.19% |      28.28% |            -3.91 pp |          7.56 pp |       3.99 pp |             +3.58 pp |
|                  0.5 |         46.57% |      45.77% |            -0.80 pp |          4.80 pp |       2.01 pp |             +2.79 pp |
|                  1.0 |         48.29% |      44.46% |            -3.83 pp |          5.15 pp |       1.47 pp |             +3.68 pp |

At all three tested values of \(\alpha\), Gradient-Balanced FCL produces lower forgetting than fixed Replay+LwF.

The relative forgetting reductions are 47.27%, 58.07%, and 71.36% for \(\alpha=0.1\), \(0.5\), and \(1.0\),
respectively.

The heterogeneity study uses only Seed 42 and should therefore be interpreted as a sensitivity analysis rather than a
multi-seed robustness result.

### 6.6 Adaptive Retention Behavior

Across the five primary Gradient-Balanced runs, 300 adaptive-weight observations are analyzed. The weights have

\[
\text{mean}=1.2963,
\]

\[
\text{standard deviation}=0.0951,
\]

and range from

\[
1.0619
\]

to

\[
1.4903.
\]

These values demonstrate that the controller does not collapse to the fixed Replay+LwF coefficient of 1.0.

**Table 5. Adaptive retention weights by continual task.**

| Task | Mean Weight | Standard Deviation | Minimum | Maximum |
|-----:|------------:|-------------------:|--------:|--------:|
|    2 |      1.3892 |             0.0701 |  1.2384 |  1.4903 |
|    3 |      1.2129 |             0.0807 |  1.0619 |  1.4456 |
|    4 |      1.2528 |             0.0571 |  1.1240 |  1.3461 |
|    5 |      1.3302 |             0.0548 |  1.2166 |  1.4155 |

**Table 6. Adaptive retention weights by communication round.**

| Round | Mean Weight | Standard Deviation |
|------:|------------:|-------------------:|
|     1 |      1.3594 |             0.0729 |
|     2 |      1.2735 |             0.0863 |
|     3 |      1.2560 |             0.0916 |

The average coefficient is highest during the first communication round and decreases in subsequent rounds, showing that
the controller responds to changes in relative gradient magnitude during training.

### 6.7 Adapted Fed-A-GEM Baseline

Among the Seed-42 learning rates of 0.001, 0.01, and 0.1, a learning rate of 0.001 is selected for the five-seed
evaluation.

**Table 7. Adapted Fed-A-GEM five-seed results.**

|          Seed | Final Average Accuracy |  Average Forgetting |
|--------------:|-----------------------:|--------------------:|
|            42 |                 13.52% |            59.29 pp |
|           123 |                 13.71% |            56.06 pp |
|          2026 |                 12.92% |            56.58 pp |
|           777 |                 12.64% |            67.15 pp |
|          1001 |                 12.98% |            60.93 pp |
| **Mean ± SD** |      **13.15 ± 0.45%** | **60.00 ± 4.46 pp** |

Under the protocol used in this work, Adapted Fed-A-GEM continues to exhibit severe catastrophic forgetting.

This result should not be interpreted as a general statement about the original Fed-A-GEM method [5]. The implementation
used here adapts its gradient-projection principle to the architecture, memory budget, task construction, client
configuration, and three-round-per-task protocol of this study.

### 6.8 Gradient Diagnostics

Across 60 update-conflict observations, the measured relationship with forgetting was

\[
r_{\mathrm{Pearson}}=0.4228
\]

and

\[
r_{\mathrm{Spearman}}=0.4175.
\]

Among the investigated gradient signals, the old-to-new gradient magnitude ratio exhibited the strongest reported
relationship with forgetting:

\[
r_{\mathrm{Pearson}}=-0.4731,
\]

\[
r_{\mathrm{Spearman}}=-0.4421.
\]

The mean absolute correlation was approximately 0.4576.

These moderate empirical associations motivate the use of gradient information as an adaptive signal but do not imply
that gradient magnitude alone predicts catastrophic forgetting.

### 6.9 Overall Interpretation

The experiments reveal three main patterns.

First, combining replay and knowledge distillation is critical under the investigated protocol. Neither mechanism alone
approaches the retention achieved by their combination.

Second, dynamically adjusting the distillation coefficient according to gradient balance produces substantially lower
forgetting than using a fixed coefficient of 1.0. The reduction occurs for all five primary seeds and for all three
tested heterogeneity levels in the Seed-42 analysis.

Third, stronger retention is not free. Gradient-Balanced FCL sacrifices approximately 2.18 pp of final average accuracy
while reducing mean forgetting by approximately 3.00 pp. The proposed method should therefore be understood as an
adaptive mechanism for shifting the stability-plasticity balance toward retention rather than as a method that
simultaneously maximizes both retention and final accuracy.

## 7. Discussion and Limitations

### 7.1 Why Adaptive Retention Reduces Forgetting

The experimental results suggest that the amount of retention pressure required during federated continual learning is
not constant throughout training.

The proposed controller measures gradient magnitudes associated with current-task and replay-memory data before local
optimization. When the current-task gradient becomes large relative to the old-memory gradient, the balance score
increases and the method assigns a stronger distillation coefficient.

This behavior provides a possible explanation for the observed reduction in catastrophic forgetting. Strong current-task
gradients can produce larger parameter changes during local optimization. Increasing the distillation constraint under
these conditions can counteract some resulting functional drift from the previous model.

Across the primary experiments, retention weights range from 1.0619 to 1.4903 rather than remaining near the fixed
coefficient of 1.0. The mean weight also varies across tasks and communication rounds.

These observations demonstrate that adaptation occurs, although they do not establish that the particular
gradient-balance rule used here is optimal.

### 7.2 Stability-Plasticity Trade-off

The central empirical finding is a clear stability-plasticity trade-off.

Compared with fixed Replay+LwF, Gradient-Balanced FCL reduces average forgetting from 4.76 pp to 1.77 pp, corresponding
to a 62.89% relative reduction. However, final average accuracy decreases from 46.94% to 44.76%.

The observed adaptive weights are greater than 1.0 throughout the analyzed primary experiments, meaning that the
mechanism generally applies stronger distillation pressure than the fixed baseline. Stronger preservation of teacher
behavior can protect previous-task performance but can also restrict adaptation to newly encountered classes.

The proposed method therefore should not be characterized as uniformly superior to fixed Replay+LwF. Rather, it provides
a mechanism for dynamically shifting optimization toward stability when the gradient signal indicates stronger
current-task pressure.

This distinction is important in practical continual-learning systems. The preferred operating point depends on the
relative cost of forgetting old knowledge versus under-adapting to new information.

Recent work on heterogeneous FCL further argues that forgetting is not necessarily uniformly harmful when knowledge
inherited from previous clients or tasks is biased, unrelated, or antagonistic [12]. This broader perspective reinforces
the importance of treating retention as a quantity that should be controlled rather than maximized unconditionally. The
present work addresses a different setting and does not attempt selective forgetting; nevertheless, its
adaptive-retention formulation is consistent with the general principle that the appropriate amount of preservation can
depend on the learning state.

### 7.3 Interpretation of the Gradient Signal

The proposed controller uses gradient magnitude rather than gradient direction.

The adaptive coefficient depends on the relative magnitudes of gradients computed from current-task and replay-memory
data. It does not directly measure whether the two gradients point in conflicting directions.

This distinguishes Gradient-Balanced FCL from projection-based approaches such as GEM, A-GEM, and Fed-A-GEM [3]–[5],
which explicitly use gradient geometry to modify optimization directions.

The diagnostic experiments indicate moderate associations between gradient-related quantities and subsequent forgetting.
These results motivate gradient information as a signal, but also indicate that gradient magnitude is only a partial
description of continual-learning interference.

Future adaptive controllers could combine magnitude information with directional measures such as cosine similarity,
update conflict, prediction shift, or other indicators of representational change.

### 7.4 Interpretation of Adapted Fed-A-GEM

Adapted Fed-A-GEM performs substantially worse than Replay+LwF under the experimental protocol used in this study.

This finding must be interpreted cautiously. The implementation adapts the Fed-A-GEM gradient-projection mechanism [5]
to a common experimental framework rather than reproducing the complete original training system.

The current protocol provides only three communication rounds per continual task and one local epoch per round.
Projection-based optimization may be sensitive to learning rate, memory composition, gradient estimates, local
optimization length, and communication rounds.

A Seed-42 learning-rate sweep was performed, but this does not constitute exhaustive hyperparameter optimization. The
result therefore demonstrates that the adapted implementation struggles under the present protocol; it does not
establish that Fed-A-GEM is generally inferior.

### 7.5 Effect of Client Heterogeneity

Gradient-Balanced FCL produces lower forgetting than fixed Replay+LwF at all three tested values,

\[
\alpha\in\{0.1,0.5,1.0\}.
\]

The strongest heterogeneity condition, \(\alpha=0.1\), substantially reduces final average accuracy for both methods.

However, only one seed is evaluated across all three heterogeneity conditions. These experiments should therefore be
interpreted as a sensitivity analysis rather than conclusive evidence of robustness to arbitrary non-IID distributions.

### 7.6 Limitations

Several limitations constrain the conclusions that can be drawn from the current study.

First, evaluation is limited to CIFAR-10. Larger and more diverse datasets are required to determine whether the
observed behavior generalizes.

Second, primary experiments use five clients and a relatively small convolutional neural network. Effectiveness at
substantially larger federated scale remains unknown.

Third, the protocol uses only three federated communication rounds per task and one local epoch per round.

Fourth, the primary comparison uses five random seeds. Although Gradient-Balanced FCL reduces forgetting for all five,
this remains a small sample.

Fifth, the multi-level heterogeneity analysis is restricted to Seed 42.

Sixth, the adaptive retention interval,

\[
[0.5,1.5],
\]

is fixed. Sensitivity to these bounds has not been systematically investigated.

Seventh, the controller uses only gradient magnitudes. Gradient direction, parameter-specific interference,
representation drift, and other indicators are not incorporated into the final controller.

Eighth, calculating the adaptive coefficient requires additional gradient passes over current-task and replay-memory
data before local training. Although the scalar controller itself is inexpensive and requires no additional
communication of raw training examples, these gradient computations introduce additional local computational cost.
Resource-constrained FCL research has shown that computational and storage limitations can materially affect the
practical behavior of continual-learning techniques on edge devices [11]. The current study does not provide a detailed
runtime, energy, or computational-overhead analysis, so the deployment cost of the proposed controller remains to be
quantified.

Finally, the experiments demonstrate a reduction in forgetting but also a decrease in final average accuracy. Achieving
a better adaptive balance between retention and acquisition remains an open problem.

### 7.7 Future Work

Future experiments should evaluate the method on larger continual-learning datasets, additional neural architectures,
and larger federated client populations. Multi-seed heterogeneity experiments would provide stronger evidence regarding
robustness under different non-IID conditions.

A broader comparison with federated continual-learning methods would also strengthen the empirical evaluation. In
particular, future work could investigate additional replay-based, regularization-based, distillation-based, and
gradient-based FCL baselines under a common protocol.

The adaptive controller could combine gradient magnitude with directional conflict, prediction shift, task uncertainty,
or other signals. Retention bounds could additionally be learned or adjusted dynamically.

Another important direction is to optimize the stability-plasticity trade-off explicitly, preserving the observed
retention advantage while reducing the associated accuracy cost.

Finally, evaluating computational overhead and communication efficiency would help determine whether Gradient-Balanced
retention remains practical in larger federated deployments.

## 8. Conclusion

This work investigated adaptive knowledge retention in federated continual learning under heterogeneous client data.
Starting from a class-incremental federated setting, we examined replay, Learning without Forgetting, their combination,
and gradient-based mechanisms for mitigating catastrophic forgetting.

The ablation experiments showed that replay and LwF individually provide limited protection under the investigated
protocol, whereas combining them produces a substantially stronger continual-learning baseline. Building on this
observation, we proposed Gradient-Balanced Replay+LwF, which replaces the fixed distillation coefficient with a bounded
adaptive coefficient derived from the relative magnitudes of current-task and replay-memory gradients.

Across five matched random seeds, Gradient-Balanced FCL reduced average forgetting from 4.76 ± 0.76 pp for fixed
Replay+LwF to 1.77 ± 0.34 pp. This corresponds to a 62.89% relative reduction in mean forgetting, and lower forgetting
was observed for all five tested seeds. The improvement in retention was accompanied by a reduction in final average
accuracy from 46.94 ± 3.40% to 44.76 ± 3.50%, revealing a clear stability-plasticity trade-off.

The adaptive mechanism produced retention coefficients ranging from 1.0619 to 1.4903 across the primary experiments,
demonstrating that it responds to changes in measured gradient balance rather than collapsing to the fixed baseline
coefficient. Seed-42 experiments at Dirichlet concentration parameters of 0.1, 0.5, and 1.0 additionally produced lower
forgetting than fixed Replay+LwF at each tested heterogeneity level.

These findings provide evidence that gradient magnitude can serve as a useful signal for dynamically controlling
retention strength in the investigated federated continual-learning setting. At the same time, the current results do
not establish universal superiority: evaluation is limited in dataset scale, client population, number of seeds,
communication rounds, and baseline coverage, and stronger retention is obtained at some cost to final average accuracy.

Future work should therefore investigate larger datasets and federated systems, broader FCL baselines, multi-seed
heterogeneity experiments, computational overhead, and adaptive controllers that incorporate both gradient magnitude and
directional interference.

## References

[1] H. B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. Agüera y Arcas, “Communication-Efficient Learning of Deep
Networks from Decentralized Data,” in Proceedings of the 20th International Conference on Artificial Intelligence and
Statistics (AISTATS), PMLR, vol. 54, pp. 1273–1282, 2017.

[2] Z. Li and D. Hoiem, “Learning without Forgetting,” in Proceedings of the European Conference on Computer Vision (
ECCV), pp. 614–629, 2016.

[3] D. Lopez-Paz and M. Ranzato, “Gradient Episodic Memory for Continual Learning,” in Advances in Neural Information
Processing Systems (NeurIPS), vol. 30, pp. 6467–6476, 2017.

[4] A. Chaudhry, M. Ranzato, M. Rohrbach, and M. Elhoseiny, “Efficient Lifelong Learning with A-GEM,” in International
Conference on Learning Representations (ICLR), 2019.

[5] S. Dai, J.-y. Sohn, Y. Chen, S. M. I. Alam, R. Balakrishnan, S. Banerjee, N. Himayat, and K. Lee, “Buffer-based
Gradient Projection for Continual Federated Learning,” Transactions on Machine Learning Research (TMLR), 2025.

[6] J. Yoon, W. Jeong, G. Lee, E. Yang, and S. J. Hwang, “Federated Continual Learning with Weighted Inter-client
Transfer,” in Proceedings of the 38th International Conference on Machine Learning (ICML), PMLR, vol. 139, pp.
12073–12086, 2021.

[7] J. Dong, L. Wang, Z. Fang, G. Sun, S. Xu, X. Wang, and Q. Zhu, “Federated Class-Incremental Learning,” in
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10164–10173, 2022.

[8] R. Kemker, M. McClure, A. Abitino, T. L. Hayes, and C. Kanan, “Measuring Catastrophic Forgetting in Neural
Networks,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 32, no. 1, pp. 3390–3398, 2018.

[9] J. Zhang, C. Chen, W. Zhuang, and L. Lyu, “TARGET: Federated Class-Continual Learning via Exemplar-Free
Distillation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 4782–4793, 2023.

[10] P. Hamedi, R. Razavi-Far, and E. Hallaji, “Federated Continual Learning: Concepts, Challenges, and Solutions,”
Neurocomputing, vol. 651, Art. no. 130844, 2025.

[11] Y. Li, Y. Wang, J. Dong, H. Wang, Y. Qi, R. Zhang, and R. Li, “Resource-Constrained Federated Continual Learning:
What Does Matter?,” in Advances in Neural Information Processing Systems (NeurIPS), 2025.

[12] A. Wuerkaixi, S. Cui, J. Zhang, K. Yan, B. Han, G. Niu, L. Fang, C. Zhang, and M. Sugiyama, “Accurate Forgetting
for Heterogeneous Federated Continual Learning,” arXiv preprint arXiv:2502.14205, 2025.