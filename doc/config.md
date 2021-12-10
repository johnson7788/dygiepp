# 模型配置

DyGIE的配置过程依赖于[AllenNLP](https://guide.allennlp.org/using-config-files)的基于`jsonnet`的配置系统。关于AllenNLP配置过程的更多信息，请看AllenNLP的[指南](https://guide.allennlp.org)。

DyGIE在此基础上增加了一层复杂性。它将配置的因数纳入。

- 所有DyGIE模型都有的组件。这些被定义在[template.libsonnet](.../training_config/template.libsonnet)。
- 特定于在特定数据集上训练的单个模型的组件。这些包含在[training config](../training_config)目录下的`jsonnet`文件中。它们使用jsonnet的继承机制来扩展`template.libsonnet`中定义的基类。 关于jsonnet继承的更多信息，请参见[jsonnet教程](https://jsonnet.org/learning/tutorial.html)

## Table of Contents
- [Required settings](#required-settings)
- [Optional settings](#optional-settings)
- [Changing arbitrary parts of the template](#changing-arbitrary-parts-of-the-template)
- [A full example](#a-full-example)


## Required settings

[template.libsonnet](../training_config/template.libsonnet)文件留下了三个未设置的变量。这些必须由继承对象来设置。关于如何工作的例子，见 [scierc_lightweight.jsonnet](../training_config/scierc_lightweight.jsonnet)。

- `data_paths`: 一个带有训练集、验证集和测试集路径的字典。
- `loss_weights`: 由于DyGIE有一个多任务目标，因此根据用户确定的损失权重，将各个损失结合起来。
- `target_task`: 在每个epoch之后，AllenNLP训练器会评估开发集的性能，并保存取得最高性能的模型状态。由于DyGIE是多任务的，用户必须指定使用哪个任务作为评估目标。这些选项是 [`ner`, `rel`, `coref`, and `events`].

注意，如果你在`training_config`目录外创建自己的配置，你需要修改这一行
```jsonnet
local template = import "template.libsonnet";
```
以使其指向模板文件。


## 可选设置

用户也可以指定：
- `bert_model`: 一个预训练过的BERT模型的名称，可用 [HuggingFace Transformers](https://huggingface.co/transformers/). The default is `bert-base-cased`.
- `max_span_width`: 模型所列举的最大跨度长度。在实践中，8个长度表现良好。
- `cuda_device`: 默认情况下，训练是在CPU上进行的。要在GPU上进行训练，请指定一个设备。


## 并行训练

TODO

## 改变模板的其它部分

TODO 注意，默认情况下coref prop是关闭的；需要在这里打开它。

jsonnet对象继承模型允许你使用`+:`符号来修改基础对象的任何（也许是深度嵌套的）字段；关于这一点，请看jsonnet文档的更多细节。例如，如果你想改变优化器的批次大小和学习率，你可以这样做。

```jsonnet
template.DyGIE {
  ...
  data_loader +: {
    batch_size: 5
  },
  trainer +: {
    optimizer +: {
      lr: 5e-4
    }
  }
}
```
你还可以向基类添加额外的字段。例如，如果你想用现有的单词表来训练一个模型，你可以添加

```jsonnet
template.DyGIE {
  ...
  vocabulary: {
    type: "from_files",
    directory: [path_to_vocab_files]
  }
}
```

## 移除预训练的BERT嵌入（例如在调试期间）。

在相关的`.jsonnet`文件中添加这些行。

```jsonnet
dataset_reader +: {
  token_indexers: {
    tokens: {
      type: "single_id"
    }
  }
},
model :+ {
  embedder: {
    token_embedders: {
      tokens: {
        type: "embedding",
        embedding_dim: 100,
      }
    }
  }
}
```

## 完整示例

```jsonnet
local template = import "template.libsonnet";

template.DyGIE {
  // Required "hidden" fields.
  data_paths: {
    train: "data/scierc/processed_data/json/train.json",
    validation: "data/scierc/processed_data/json/dev.json",
    test: "data/scierc/processed_data/json/test.json",
  },
  loss_weights: {
    ner: 1.0,
    relation: 1.0,
    coref: 0.0,
    events: 0.0
  },
  target_task: "rel",

  // Optional "hidden" fields
  bert_model: "allenai/scibert_scivocab_cased",
  cuda_device: 0,
  max_span_width: 10,

  // Modify the data loader and trainer.
  data_loader +: {
    batch_size: 5
  },
  trainer +: {
    optimizer +: {
      lr: 5e-4
    }
  },

  // Specify an external vocabulary
  vocabulary: {
    type: "from_files",
    directory: "vocab"
  },
}
```
