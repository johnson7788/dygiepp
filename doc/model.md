# 模型

我们在此包括一些注意事项和常见的建模问题。

## Table of Contents
- [Debugging](#debugging)
- [Batching and batch size](#batching-and-batch-size)
- [Multi-dataset training](#multi-dataset-training)
- [Hyperparameter optimization](#hyperparameter-optimization)
- [Reproducibility and nondeterminism](#reproducibility-and-nondeterminism)

## Debugging

通过运行`allennlp train`或`allennlp predict`进行调试并不是最理想的，因为模型仅初始化就需要10多秒。为了加快调试循环，有一个脚本[debug_forward_pass.py](.../scripts/debug/debug_forward_pass.py)将为你运行一个前向传递，而不做所有的初始化逻辑，也不加载BERT嵌入。有关使用信息，请参见该脚本。

### 调试过程中遇到的常见问题

- 梯度或参数中的 "nan"。这可能是由于tokenizer不能识别的字符造成的。更多细节见[本问题](https://github.com/allenai/allennlp/issues/4612)。在做任何更广泛的调试之前，检查你的输入文件是否有奇怪的unicode字符。


## 批次和批次大小

AllenNLP有一个数据结构来表示[Instance](https://guide.allennlp.org/reading-data#1)，它将其定义为 "机器学习中预测的原子单位"。例如，在情感分类中，一个 "实例 "通常是一个单一的句子。

`实例'对DyGIE++来说略显尴尬，因为三个任务（命名实体标签、关系抽取、事件提取）是在句子内的，使句子成为`实例'的自然单位。然而，核心推理解决是跨句子的，使得*文档*成为 "实例 "的自然单位。

我们所做的选择是将一个 "实例 "建模为一个*文档*。默认情况下，我们使用1的批次大小，这意味着在训练过程中，每个minibatch都是一个*文档*。我们做出这样的选择是因为它在概念上是最简单的，但是在某些情况下它并不是最佳选择。我们描述了这些情况并提供了一些解决方案。这些解决方案大多涉及到在建模代码之外进行数据预处理；这使得（已经有些混乱的）建模代码尽可能的简单。


--------------------
- **问题**。如果你不做核心参考解析，那么拥有长度相差很大的句子的mini-batch是很浪费的。相反，你应该从不同的文件中创建长度相近的句子的mini-batch。
- **解决方案**。我们的解决方案如下。
  - 将数据集 "Collate"成包含类似长度的句子的 "伪文件"。追踪每个句子来自哪个原始文件。用户可以编写自己的脚本，或者使用[collate.py](.../scripts/data/shared/collate.py)来完成这一工作。
  - 运行训练/预测/其他什么。
  - 对于预测，使用[uncollate.py](.../scripts/data/shared/uncollate.py)对预测进行 "un-collate "以恢复原始文档。
- **细节**。用户要以充分利用GPU内存的方式来整理句子。collate "脚本有两个选项来帮助控制这个。
  - `max_spans_per_doc`。一般来说，DyGIE++的GPU使用量与文档中的跨度数量成正比，而跨度是句子长度的。因此，`collate.py`将`max_spans_per_doc`作为输入。我们计算每个文档的跨度数量为`n_sentences * (longest_sentence_length ** 2)`。我们发现，设置`max_spans_per_doc=50000'可以有效地利用GPU的批次。然而，我们还没有对此进行详尽的探索，我们欢迎反馈和公关。
  - `max_sentences_per_doc`（每份文件最大句数）。
    - 如果你正在训练一个模型，你可能想避免创建包含数百个短句的伪文件--即使它们适合在GPU中。每个批次的句子数量变化很大，似乎在训练过程中会出现一些奇怪的事情，尽管这只是传闻。为了避免这种情况，请将`max_sentences_per_doc`设置为某个合理的值。默认的16似乎是安全的，尽管更大的也可以。
    - 如果你使用一个现有的模型进行预测。只需将其设置为一个大的数字，以最好地利用你的GPU。
  
--------------------

- **问题**。你的文件太长了，无法装入内存。你会得到类似 "RuntimeError: CUDA没有内存了。试图分配1.97 GiB (GPU 0; 10.92 GiB总容量; 7.63 GiB已经分配; 1.46 GiB空闲; 1.12 GiB缓存)`。
- **解决方案（数据集*没有*核心参考标注）**。如果你的数据集中没有核心推理标注，你可以使用[collate.py]（.../scripts/data/shared/collate.py），如上所述。如果由于某种原因你不想这样做，你可以使用[normalize.py](.../scripts/data/shared/normalize.py)来分割长的文档，而不需要对不同文档的句子进行打乱。
- **解决方案（数据集*有*核心参考标注）**。如果你有核心参考标注，你就不能创建由不同文档的句子组成的mini-batch。[normalize.py](.../scripts/data/shared/normalize.py)应该能够分割带有coref标注的长文档，但我还没有实现这个功能（我欢迎完成这个功能的PR）。所以，不幸的是，你必须自己写脚本来分割长文档。

--------------------

- **问题**。你正在做核心参考解析，但你的数据集中的文档很短；使用1的批次量会浪费GPU内存。
- **解决方案**。我们正在编写一个数据加载器，可以为你处理这个问题。

## 多数据集训练

DyGIE能够为4项任务进行多任务学习。
- 命名实体识别(NER)
- 关系抽取
- 事件提取
- 核心推理

在某些情况下，在多个不同的数据集上训练不同的任务是可取的--例如，在OntoNotes上训练的核心参考文献解析模型可以用来改善ACE数据集上的NER预测。训练共享相同基础跨度表示的多个命名实体识别模型甚至可能是有用的。

用DyGIE进行多数据集训练的方法如下。在[data](data.md)输入模型的每一行都必须有一个`dataset`字段。我们做了以下假设。

- 不同 "数据集 "的NER、关系和事件标签命名空间是_disjoint_。为每个数据集训练一个单独的模型。
- 不同 "数据集 "的核心参考标签是共享的。在所有数据集上训练一个单一的核心解析模型。

作为一个具体的例子：假设你决定对一个科学信息抽取模型进行训练。

- SciERC (NER, relation, coref)
- GENIA (NER, coref)

该模型将创建以下标签命名空间：

- `scierc__ner`
- `genia__ner`
- `scierc__relation`
- `coref`

命名空间的命名方式是 `[dataset]__[task]`.

将创建一个单独的模块来为每个命名空间进行预测。所有模块将共享相同的跨度表示，但将有不同的特定任务和特定命名空间的权重。该模型将为每个命名空间计算不同的性能指标，例如

- `scierc__ner_precision`
- `scierc__ner_recall`
- `scierc__ner_f1`
- `genia__ner_precision`
- `genia__ner_recall`
- `genia__ner_f1`
- etc.

对于每个任务，它还将通过命名空间的平均值来计算平均性能。

- `MEAN__ner_precision`
- `MEAN__ner_recall`
- `MEAN__ner_f1`

当进行预测时，要预测的输入的 "dataset "字段必须与模型训练的一个输入数据集的 "dataset "字段相匹配。


## 超参数优化
用[Optuna](https://optuna.org)调整超参数，见例子`./scripts/tuning/train_optuna.py`。这个是在一个PR中贡献的，没有得到 "官方支持"，但看起来非常有用。

Requirements:
- `sqlite3`用于存储试验。
如果没有安装`sqlite3`，你可以使用内存存储：在`optuna.create_study`中把`storage`改为`None`（不推荐）。
  

Usage:
- 将Jsonnet配置文件放在`./training_config/`目录下，见例子`./training_config/ace05_event_optuna.jsonnet`。
  - 用Jsonnet方法mask超参数的值，调用`std.extVar('{param_name}')`与`std.parseInt`用于整数或`std.parseJson`用于浮点和其他类型。
  - 用`+:`覆盖嵌套的默认模板值，见[config.md]（`./doc/config.md`）。
- 编辑 `objective`-函数 in `./scripts/tuning/optuna_train.py`:
  - Add [trial suggestions with `suggest` functions](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html).
  - 将 "metrics "参数从 "executor "改为相关的优化目标。
- `dygiepp` 环境激活, run: `python optuna_train.py <CONFIG_NAME>`
- 最佳配置dump到 `./training_config/best_<CONFIG_NAME>.json`.

For more details see [Optuna blog](https://medium.com/optuna/hyperparameter-optimization-for-allennlp-using-optuna-54b4bfecd78b).


## 可重复性和非决定性
一些用户观察到，即使在[配置模板](.../training_config/template.libsonnet)中设置了相关的随机种子，跨多个DyGIE训练的结果也是不可复制的。这是[PyTorch](https://pytorch.org/docs/stable/notes/randomness.html)的一个潜在问题。特别是，Torch的`index_select()`函数是不确定的。因此，AllenNLP的`batched_index_select()`也是如此，DyGIE在很多地方都使用了这个函数。我希望能有一个PR来解决这个问题，虽然这并不明显，但可以做到这一点。感谢[@khuangaf](https://github.com/khuangaf)指出了这一点。
