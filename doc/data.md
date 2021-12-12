# Data

我们在此提供了每个数据集的数据预处理的细节。

## Table of contents

- [Data format](#data-format)
- [Format for predictions](#format-for-predictions)
- [Code for data manipulation](#code-for-data-manipulation)
- [Formatting a new dataset](#formatting-a-new-dataset)
- [Preprocessing details for existing datasets](#preprocesing-details-for-existing-datasets)

## 数据格式

在预处理之后，所有的数据集都将被格式化为[SciERC dataset](http://nlp.cs.washington.edu/sciIE/)。下载数据后，你可以看一下`data/scierc/normalized_data/json/train.json`作为例子。数据集中的每一行都是一个文件的JSON表示（技术上讲，文件应该被赋予`.jsonl`扩展名，因为每一行都是一个JSON对象，抱歉造成了混淆）。

### 必须填写的字段

- `doc_key`: 文件的唯一字符串标识符。
- `dataset`: 该文件来自的数据集的一个字符串标识符。关于这个字段的更多信息，请参见关于 [multi-datset training](model.md)
- `sentences`: 文档中的句子，写成一个嵌套的token列表。比如说。
  ```json
  [
    ["Seattle", "is", "a", "rainy", "city", "."],
    ["Jenny", "Durkan", "is", "the", "city's", "mayor", "."],
    ["She", "was", "elected", "in", "2017", "."]
  ]
  ```
  Empty strings (`""`) are not allowed as entries in sentences; the reader will raise an error if it encounters these.

### 可选的标注字段

- `weight`: 训练时，将文档的损失乘以这个权重（一个浮点数）。在结合不同大小的数据集时，或者在结合弱标注数据和glod标注时，这很有用。

- `ner`: 文件中的命名实体，写成一个嵌套的列表--每句有一个子列表。每个列表条目的形式是`[start_tok, end_tok, label]`。`start_tok`和`end_tok`标注的索引是关于_文档_的，而不是关于句子的。例如，上面的句子中的实体可能是:
  ```json
  [
    [[0, 0, "City"]],
    [[6, 7, "Person"], [9, 10, "City"]],
    [[13, 13, "Person"], [17, 17, "Year"]]
  ]
  ```
  这些实体类型只是一个例子，它们并不反映一个实际数据集的实体模式。
- `relations`: 文件中的关系，也是每个句子一个子列表。每个列表条目的形式是 `[start_tok_1, end_tok_1, start_tok_2, end_tok_2, label]`.
   ```json
   [
     [],
     [[6, 7, 9, 10, "Mayor-Of"]],
     [[13, 13, 17, 17, "Elected-In"]]
   ]
   ```
- `clusters`: 核心推理。这是一个嵌套的列表，但这里的每个子列表都给出了核心推理簇中每个提及的跨度。cluster可以跨越句子的边界。例如，本例中的第一个集群是关于西雅图的，第二个是关于市长的。
  ```json
  [
    [
      [0, 0], [9, 10]
    ],
    [
      [6, 7], [13, 13]
    ]
  ]
  ```

SciERC数据集没有任何事件数据。要看一个事件数据的例子，运行[README](README.md)中描述的ACE事件预处理步，看看`data/ace-event/processed-data`中的一个文件。你会看到以下的附加字段。
- `events`: 文件中的事件，每个句子有一个子列表。一个有`N`个参数的事件将被写成一个列表，其形式为 `[[trigger_tok, event_type], [start_tok_arg1, end_tok_arg1, arg1_type], [start_tok_arg2, end_tok_arg2, arg2_type], ..., [start_tok_argN, end_tok_argN, argN_type]]`. 请注意，在ACE中，事件触发器只能是一个token。比如说。
  ```json
  [
    [],
    [],
    [
      [
        [15, "Peronnel.Election"],
        [13, 13, "Person"],
        [17, 17, "Date"]
      ]
    ]
  ]
  ```

- `event_clusters`: 事件核心推理群组。其结构与`clusters`相同，但每个群组对应于一个事件，而不是一个实体。每个跨度对应于触发器的跨度。虽然事件触发器在ACE中只能是一个token，但为了与`clusters`保持一致，我们保留了结尾的token。注意：事件集群是由一个贡献者添加的，并非 "官方支持"。
  ```json
  [
    [
      [517, 517], [711, 711], [723, 723]
    ],
    [
      [603, 603], [741, 741]
    ]
  ]
  ```
也可能有一个`sentence_start'字段，表示每个句子的开始相对于文件的标记索引。这可以被忽略。


### 用户定义的句子元数据

你可以定义与每个句子相关的额外元数据，这些元数据将被模型忽略；这些元数据字段应以`_`为前缀。例如，如果你想明确地跟踪文档中每个句子的索引，你可以在你的输入文档中添加一个字段

```python
{
  "doc_key": "some_document",
  "dataset": "some_dataset",
  "weight": 0.5,
  "sentences": [["One", "sentence"], ["Another", "sentence"]],
  "_sentence_index": [0, 1]   # User-added metadata field.
}
```

## 预测的格式

当模型预测被保存到文件时，它们的格式如上所述，但有以下变化。

- 字段名有 "predicted "的前缀。例如，`predicted_ner`，`predicted_relations`，等等。
- 每个预测都有两个附加条目，指定了预测标签的对数分数和软性最大概率。比如说。
  - 一个单一的预测关系预测的形式是 `[start_tok_1, end_tok_1, start_tok_2, end_tok_2, predicted_label, label_logit, label_softmax]`.
  - 一个单一的预测事件的形式是 `[[trigger_tok, predited_event_type, event_type_logit, event_type_softmax], [start_tok_arg1, end_tok_arg1, predicted_arg1_type, arg1_type_logit, arg1_type_softmax], ...]`.
  - TODO: 这一点还没有在coreference中实现。


## 数据操作的代码

模块[document.py](./dygie/data/dataset_readers/document.py)包含加载、保存、操作和可视化DyGIE格式数据的类和方法。参见[document.ipynb](.../notebooks/document.ipynb)，了解使用实例。

## 格式化一个新的数据集

如果你想使用预训练好的DyGIE++模型在一个新的数据集上进行预测，你的新数据集中的`dataset`字段必须与原始模型训练的`dataset`相匹配；这表明模型应该使用哪个标签命名空间进行预测。参见[可用的预训练模型]一节（.../README.md#pretrained-models），了解每个模型的数据集名称。关于标签命名空间的更多信息，见[多数据集训练](model.md/#multi-dataset-training)一节。

### 无标签数据

如果你的无标签数据是以`.txt`文件的目录形式存储的（每个文件一个文件），你可以运行`python scripts/data/new-dataset/format_new_dataset.py [input-directory] [output-file]`将文件格式化为`jsonl`文件，每个文件有一行。如果你的数据集是科学文本，添加`--使用scispacy`标志，让[SciSpacy](https://allenai.github.io/scispacy/)进行tokenization处理。

如果你的数据不是以这种形式出现的，你可以按照脚本中的这个基本配方。

-  Use [Spacy](https://spacy.io) (or [SciSpacy](https://allenai.github.io/scispacy/) for scientific text) to 将每个文件分成句子，然后再分成token。
- 将所有文件收集到一个`jsonl`文件中，每个文件一行，对每个文件的`doc_key's使用一些适当的方案。

### 有标签数据

许多有标签的数据集每个文件有两个文件。
1. 一个包含文档文本的文件（通常是一个`.txt`文件）。
2. 一个包含源文件中每个实体提及的字符索引的文件，以及表明参与关系和事件的实体提及的标注（通常是一个`.ann`文件）。

将这样的数据集转化为DyGIE格式需要对文本进行tokenizing，将字符级的命名实体标注与token的文本对齐，并将关系和事件提及的内容映射为token。这可能会有噪音，因为tokenizer产生的token边界可能并不总是与命名实体的字符索引一致。处理这个问题的最简单方法是直接扔掉不匹配的实体。一般的过程在`scripts/data/chemprot/02_chemprot_to_input.py`中实现。

如果你在预处理数据集的过程中遇到困难，请发布一个问题。或者，如果你想出了一个很好的、通用的标注数据预处理脚本，请提交一个PR

#### 转换标注为brat的数据
The script [brat_to_input.py](https://github.com/dwadden/dygiepp/tree/master/scripts/new-dataset/brat_to_input.py) 是一个通用的预处理脚本，适用于用[brat rapid annotation tool](https://brat.nlplab.org/)标注的数据。这个脚本对文本进行tokenization和对齐，其中字符索引被转换为文档级的token索引，关系和事件被映射到这些token上。这个脚本的输出是一个文件，其中包含每个文档的一个json格式的dict，以及输入到DyGIE所需的字段。不能使用文本的spacy tokenization来对齐的实体会被抛出并发出警告，而且只有在存在`coref`和`event`数据类型的情况下才会包含这些字段。你可以这样使用该脚本。
```
python brat_to_input.py \
  path/to/my/data/ \
  output_file.jsonl \
  scierc \
  --use-scispacy \
  --coref \
```  
`--use-scispacy`标志表示scispacy将被用作tokenizer。--coref标志表示是否将brat的等价关系类型（在`.ann`文件中的类型为`*`）作为核心推理cluster。这是目前包括核心推理的唯一方法。注意：这段代码是由一个贡献者添加的，并不是 "官方支持"。如果你遇到问题，请创建一个问题并标记@serenalotreck。

### 处理长句子

大多数基于transformer的编码器有512个token的限制。长于此数的句子会导致错误。不幸的是，你不能只是检查你的每个 "句子 "字段是否最多只有512个token。这些token被转换为BERT的字节对编码，一个 "单词token "可能被分割成多个 "BERT token"。我们提供了一个脚本`scripts/data/shared/check_sentence_length.py`，你可以在一个输入文件上运行它。它将识别那些字节对编码超过你所使用的编码器限制的句子。

如果你的句子太长，你有两个选择。

1. 分割长句，并重新运行`check_sentence_length.py`以检查它们是否足够短。

2. 修改配置，使编码器将长句分割成小块，分别进行编码，然后再合并在一起。我还没有试过这个方法（欢迎把这个方法用于PR）。要用这种方式训练模型，应该可以这样修改训练配置。

```jsonnet
dataset_reader +: {
  token_indexers +: {
    bert +: {
      max_length: 512
    }
  }
},
model +: {
  embedder +: {
    token_embedders +: {
      bert +: {
        max_length: 512
      }
    }
  }
}
```
我不确定这是否能与 "allennlp predict "一起工作；这些选项可能需要用 "overrides "标记来设置 "predict "命令。

欲了解更多信息，请查看AllenNLP的`max_length`参数。 [PretrainedTransformerMismatchedIndexer](https://docs.allennlp.org/master/api/data/token_indexers/pretrained_transformer_mismatched_indexer/). 注意token索引器和token嵌入器必须被赋予相同的`max_length'。

如果你使用的是BERT以外的模型，而你不知道要设置的正确的`max_length'，你可以这样得到它。

```python
from transformers import AutoConfig

config = AutoConfig.from_pretrained([model_name])
print(config.max_position_embeddings)
```


## 现有数据集的预处理细节

许多信息提取数据集的每个文件都有两个文件，其形式大致如下。
1. 文件的文本（通常是一个`.txt`文件）。
2. 一个包含从字符索引到NER提及的映射，以及从NER提及的对/图到关系提及/事件提及的映射的文件（通常是一个`.ann`文件，但有时是`.xml`或其他文件）。
  
为了预处理这类数据，你通常需要。
- 使用[Spacy](https://spacy.io)或类似的方法对文档文本进行句法化和tokenize。
- 将字符级的NER提及的内容与tokenize的文档对齐。通常，有一些情况下，token的边界与标注实体提及的字符索引不一致。
- 将关系/事件提及的内容映射到token对齐的NER提及的内容。

这个过程通常是混乱的。作为一个例子，见`./scripts/data/chemprot/02_chemprot_to_input.py`。如果有值得注意的特殊情况，请随时创建一个问题。

### SciERC

[SciERC dataset](http://nlp.cs.washington.edu/sciIE/) 包含计算机科学研究论文摘要的实体、关系和核心参考标注。

这个数据不需要进行预处理。运行脚本`./scripts/data/get_scierc.sh`，数据将被下载并放在`./data/scierc/processed_data`中。


### GENIA

The [GENIA dataset](https://orbit.nlm.nih.gov/browse-repository/dataset/human-annotated/83-genia-corpus/visit) 包含生物医学研究论文摘要的实体、关系和事件标注。实体可以是嵌套的，也可以是重叠的。我们的预处理代码将实体和核心参考链接转换为我们的JSON格式。事件刨除了复杂的层次结构，留待以后的工作。

To download the GENIA data and preprocess it into the form used in our paper, run the script `./scripts/data/get_genia.sh`. The final `json` versions of the data will be placed in `./data/genia/processed-data`. We use the `json-coref-ident-only` version. The script will take roughly 10 minutes to run.

In GENIA, coreference annotations are labeled one of `IDENT, NONE, RELAT, PRON, APPOS, OTHER, PART-WHOLE, WHOLE-PART`. In the processed data folder, `json-coref-all` has all coreference annotations. `json-coref-ident-only` uses only `IDENT` coreferences. We use the `ident-only` version in our experiments. `json-ner` has only the named entity annotations.

We followed the preprocessing and train / dev / test split from the [SUTD NLP group's](https://gitlab.com/sutd_nlp/overlapping_mentions/tree/master/data/GENIA) work on overlapping entity mention detection for GENIA. We added some additional scripts to convert their named entity data to our JSON format, and to merge in the GENIA coreference data. Some documents were named slightly differently in the entity and coreference data, and we did our best to stitch the annotations back together.

We encountered off-by-one errors stitching together the coref and ner annotations for 10 training documents, and excluded these. They are listed in `./scripts/data/genia/exclude.txt`. If for some reason you want to include these documents anyhow, pass the `--keep-excluded` flag as detailed in a comment at the end of  `./scripts/data/get_genia.sh`.


### ACE Relation

The [ACE 2005](https://catalog.ldc.upenn.edu/LDC2006T06) dataset contains entity, relation, and event annotations for an assortment of newswire and online text. Our preprocessing code is based on the code from the [LSTM-ER repo](https://github.com/tticoin/LSTM-ER), and uses the train / dev / test split described in [Miwa and Bansal (2016)](https://www.semanticscholar.org/paper/End-to-End-Relation-Extraction-using-LSTMs-on-and-Miwa-Bansal/3899f87a2031f3434f89beb68c11a1ca6428328a).


### ACE Event

We start off with the same data as for ACE Relation, but use different splits and preprocessing. For ACE Event, we use the standard split for event extraction used in [Yang and Mitchell (2016)](https://www.semanticscholar.org/paper/Joint-Extraction-of-Events-and-Entities-within-a-Yang-Mitchell/c558e2b5dcab8d89f957f3045a9bbd43fd6a28ed). Unfortunately, there are a number of different ways that the ACE data can be preprocessed. We follow the conventions of [Zhang et al. (2019)](https://www.semanticscholar.org/paper/Joint-Entity-and-Event-Extraction-with-Generative-Zhang-Ji/ea00a63c2acd145839eb6f6bbc01a5cfb4930d43), which claimed SOTA at the time our paper was submitted.

不幸的是，不同的论文使用了不同的约定，因此我们的结果可能无法直接比较。然而，我们在脚本`./scripts/data/ace-event/parse_ace_event.py`中加入了一些标志，以使研究人员能够做出不同的预处理选择。可用的标志是。

- **use_span_extent**: By default, when defining entity mentions, we use the `head` of the mention, rather than its `extent`, as in this example:
  ```xml
  <entity_mention ID="AFP_ENG_20030330.0211-E3-1" TYPE="NOM" LDCTYPE="NOM" LDCATR="FALSE">
    <extent>
      <charseq START="134" END="170">Some 2,500 mainly university students</charseq>
    </extent>
    <head>
      <charseq START="163" END="170">students</charseq>
    </head>
  </entity_mention>
  ```
  Running `parse_ace_event.py` with the flag `--use_span_extent` will use `extent`s rather than `head`s.

- **include_times_and_values**: By default, `timex2` and `value` mentions are *not* treated as entity mentions, and are ignored. For instance, this annotation would be ignored:
  ```xml
  <timex2 ID="AFP_ENG_20030327.0022-T1" VAL="2003-03-27">
    <timex2_mention ID="AFP_ENG_20030327.0022-T1-1">
      ...
    </timex2_mention>
  </timex2>
  ```
  So would this one:
  ```xml
  <value ID="AFP_ENG_20030330.0211-V1" TYPE="Numeric" SUBTYPE="Percent">
    <value_mention ID="AFP_ENG_20030330.0211-V1-1">
      ...
    </value_mention>
  </value>
  ```
  To include these mentions as entity mentions, use the flag `--include_times_and_values`. Note that all values are given entity type `VALUE`. Some work has assigned entity types using the `TYPE` of the value - for instance `"Numeric"` in the example above. We welcome a pull request to add this feature.

- **include_pronouns**: By default, pronouns (entities with `TYPE="PRO"`) are also *ignored*. For instance, this annotation  would be ignored:
  ```xml
  <entity_mention ID="AFP_ENG_20030330.0211-E3-2" TYPE="PRO" LDCTYPE="WHQ" LDCATR="FALSE">
    ...
  </entity_mention>
  ```
  To include pronouns as entity mentions, use the flag `--include_pronouns`.

- **include_entity_coreference**: Added by PR, not "officially supported". Include entity coreference clusters.

- **include_event_coreference**: Added by PR, not "officially supported". Include event coreference clusters.


### WLPC

TODO
