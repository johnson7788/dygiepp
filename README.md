# DyGIE++

参考论文 [Entity, Relation, and Event Extraction with Contextualized Span Representations](https://www.semanticscholar.org/paper/Entity%2C-Relation%2C-and-Event-Extraction-with-Span-Wadden-Wennberg/fac2368c2ec81ef82fd168d49a0def2f8d1ec7d8).

## Table of Contents
- [Updates](#updates)
- [Project status](#project-status)
- [Issues](#issues)
- [Dependencies](#dependencies)
- [Model training](#training-a-model)
- [Model evaluation](#evaluating-a-model)
- [Pretrained models](#pretrained-models)
- [Making predictions on existing datasets](#making-predictions-on-existing-datasets)
- [Working with new datasets](#working-with-new-datasets)
- [Contact](#contact)

参见`doc`文件夹中的文档，有更多细节 on the [data](doc/data.md), [model implementation and debugging](doc/model.md), and [model configuration](doc/config.md).


## Updates

**April 2021**: 我们增加了MECHANIC数据集的数据和模型，该数据集在NAACL 2021年的论文中提出。 [Extracting a Knowledge Base of Mechanisms from COVID-19 Papers](https://www.semanticscholar.org/paper/c4ce6aca9aed41d57d588674484932e0c2cd3547).

- [Download the dataset](https://ai2-s2-mechanic.s3-us-west-2.amazonaws.com/data/data.zip)
- [Download the "coarse" model](https://ai2-s2-mechanic.s3-us-west-2.amazonaws.com/models/mechanic-coarse.tar.gz)
- [Download the "granular" model](https://ai2-s2-mechanic.s3-us-west-2.amazonaws.com/models/mechanic-granular.tar.gz)

你也可以通过运行`bash scripts/data/get_mechanic.sh`来获得数据，这将把数据放在`data/mechanic`中。

在将模型移到`pretrained`文件夹后，你可以像这样进行预测。

```bash
allennlp predict \
  pretrained/mechanic-coarse.tar.gz \
  data/mechanic/coarse/test.json \
  --predictor dygie \
  --include-package dygie \
  --use-dataset-reader \
  --output-file predictions/covid-coarse.jsonl \
  --cuda-device 0 \
  --silent
```


## 项目状态

这个分支曾经被命名为 "allennlp-v1"，现在它已经成为新的 "master"。它与新版本的AllenNLP兼容，模型配置过程也被简化。我建议今后所有的工作都使用这个分支。如果你因为某些原因需要旧版本的代码，它在[hnlp-2019](https://github.com/dwadden/dygiepp/tree/emnlp-2019)这个分支上。

不幸的是，我现在没有带宽来增加额外的特征。但如果你有问题，请创建一个新问题。
- 重现README中报告的结果。
- 在一个新的数据集上使用预训练好的模型进行预测。
- 在一个新的数据集上训练你自己的模型。

See [below](#issues) for guidelines on creating an issue.

有很多方法可以改进这段代码，我绝对欢迎 pull requests。如果你有兴趣，请参阅[contribution.md](doc/contributions.md)，了解ideas的列表。

### 提交模型

如果你有一个在新数据集上训练的DyGIE模型，请随时上传[这里](https://docs.google.com/forms/d/e/1FAIpQLSdwws7zVAqF15-kBqkKBupymWe0ASkXhODH8yomYkRDy5DvCw/viewform?usp=sf_link)，我将把它加入预训练模型的集合中。

## Issues

如果你无法运行该代码，请随时创建一个问题。请做以下工作。

- 确认你已经完全按照下面的[Dependencies](#dependencies)部分设置了一个Conda环境。只有当你在这个环境中运行代码时，我才能提供支持。
- 请指明你用来下载预训练模型或下载/预处理数据的任何命令。请将代码放在代码块中，例如。
  ```bash
  # 下载预训练过的模型。

  bash scripts/pretrained/get_dygiepp_pretrained.sh
  ```
- 例如，分享你运行的导致该问题的命令。
  ```
  allennlp evaluate \
  pretrained/scierc.tar.gz \
  data/scierc/normalized_data/json/test.json \
  --cuda-device 2 \
  --include-package dygie
  ```
- 包括你得到的完整错误信息。


## 依赖

克隆这个版本库并在你的系统上导航到版本库的根目录。然后执行。

```
conda create --name dygiepp python=3.7
pip install -r requirements.txt
conda develop .   # Adds DyGIE to your PYTHONPATH
```

这个库依赖于[AllenNLP](https://allennlp.org)，并使用AllenNLP shell[命令](https://docs.allennlp.org/master/#package-overview)来启动训练、评估和测试。

If you run into an issue installing `jsonnet`, [this issue](https://github.com/allenai/allennlp/issues/2779) may prove helpful.

### Docker build
Pytorch + CUDA + CUDNN基础镜像中提供了一个 "Dockerfile"，用于安装全栈式GPU。
它将创建用于建模的`dygiepp'和用于ACE05-Event预处理的`ace-event-preprocess'的conda环境。

默认情况下，构建会下载所有任务的数据集和依赖性。
这需要很长的时间，并产生一个大的image，所以你会想在Docker文件中注释掉不需要的数据集/任务。

- Comment out unneeded task sections in `Dockerfile`.
- Build container: `docker build --tag dygiepp:dev <dygiepp-repo-dirpath>`
- Run the container interactively, mount this project dir to /dygiepp/: `docker run --gpus all -it --ipc=host -v <dygiepp-repo-dirpath>:/dygiepp/ --name dygiepp dygiep:dev`

**注意**。这个Docker文件是在一个贡献者的PR中添加的。我还没有测试过它，所以它不是 "官方支持"。不过，我们欢迎更多的PR。

## 训练模型

*关于核心参考解析的警告*。核心参考代码在只有一个token的句子上会失效。如果你的数据集中有这些句子，要么把它们去掉，要么停用模型的核心推理部分。

我们依靠[Allennlp train]（https://docs.allennlp.org/master/api/commands/train/）来处理模型训练。`train`命令接受一个配置文件作为参数，并根据配置初始化一个模型，并将训练后的模型序列化。关于DyGIE配置过程的更多细节可以在[doc/config.md](doc/config.md)找到。

要训练一个模型，在命令行输入`bash scripts/train.sh [config_name]`，其中`config_name`是`training_config`目录下的一个文件的名字。例如，要使用`scierc.jsonnet`配置来训练一个模型，你要输入

```bash
bash scripts/train.sh scierc
```
产生的模型将放在`models/scierc`中。关于如何修改训练配置的更多信息（例如，改变用于训练的GPU），请参阅[config.md]（doc/config.md）。

关于准备特定训练数据集的信息在下面。关于如何创建有效利用GPU资源的训练批次的更多信息，见[model.md](doc/model.md)。
超参数优化搜索是使用[Optuna](https://optuna.readthedocs.io)实现的，见[model.md](doc/model.md)。

### SciERC数据集

在SciERC数据集上训练一个用于命名实体识别、关系抽取和核心推理的模型。

- **下载数据**. 在该版本的top文件夹中，输入`bash ./scripts/data/get_scierc.sh`。这将把scierc数据集下载到`./data/scierc`文件夹中。
- **训练模型**. Enter `bash scripts/train.sh scierc`.
- 要训练一个 "轻量级 "的模型，不做核心推理传播，并使用1的上下文宽度，可以用`bash scripts/train.sh scierc_lightweight`代替。关于为什么要这样做的更多信息，请参见[预测]（#making-predictions）一节。

### GENIA数据集

步骤类似于SciERC

- **Download the data**. From the top-level folder for this repo, enter `bash ./scripts/data/get_genia.sh`.
- **Train the model**. Enter `bash scripts/train genia`.
- 与SciERC一样，我们也提供了一个 "轻量级 "版本，上下文宽度为1，没有核心推理传播。


### ChemProt
[ChemProt](https://biocreative.bioinformatics.udel.edu/news/corpora/chemprot-corpus-biocreative-vi/)语料库包含药物/蛋白质相互作用的实体和关系标注。ChemProt的预处理需要一个单独的环境。

```shell
conda deactivate
conda create --name chemprot-preprocess python=3.7
conda activate chemprot-preprocess
pip install -r scripts/data/chemprot/requirements.txt
```

- **Get the data**.
  - Run `bash ./scripts/data/get_chemprot.sh`. 这将下载数据并将其处理成DyGIE输入格式。
    - NOTE: 这是一个快速和dirty的脚本，跳过那些字符偏移量与SciSpacy产生的tokenization不完全一致的实体。我们因此失去了数据集中大约10%的命名实体和20%的关系。
  - Switch back to your DyGIE environment.
  - Collate the data:
    ```
    mkdir -p data/chemprot/collated_data

    python scripts/data/shared/collate.py \
      data/chemprot/processed_data \
      data/chemprot/collated_data \
      --train_name=training \
      --dev_name=development
   - For a quick spot-check to see how much of the data was lost, you can run:
    ```
    python scripts/data/chemprot/03_spot_check.py
    ```   ```
- **Train the model**. Enter `bash scripts/train chemprot`.


### ACE05 (ACE for entities and relations)

#### Creating the dataset

For more information on ACE relation and event preprocessing, see [doc/data.md](doc/data.md) and [this issue](https://github.com/dwadden/dygiepp/issues/11).

We use preprocessing code adapted from the [DyGIE repo](https://github.com/luanyi/DyGIE), which is in turn adapted from the [LSTM-ER repo](https://github.com/tticoin/LSTM-ER). The following software is required:
- Java, to run CoreNLP.
- Perl.
- zsh. If this isn't available on your system, you can create a conda environment and install [zsh](https://anaconda.org/conda-forge/zsh).

First, we need to download Stanford CoreNLP:
```
bash scripts/data/ace05/get_corenlp.sh
```
Then, run the driver script to preprocess the data:
```
bash scripts/data/get_ace05.sh [path-to-ACE-data]
```

The results will go in `./data/ace05/collated-data`. The intermediate files will go in `./data/ace05/raw-data`.

#### Training a model

Enter `bash scripts/train ace05_relation`. A model trained this way will not reproduce the numbers in the paper. We're in the process of debugging and will update.

### ACE05事件抽取

#### Creating the dataset
我写的预处理代码在最新版本的Spacy中出现故障。因此，不幸的是，我们需要创建一个单独的virtualenv，使用旧版本的Spacy，并使用它来进行预处理。

```shell
conda deactivate
conda create --name ace-event-preprocess python=3.7
conda activate ace-event-preprocess
pip install -r scripts/data/ace-event/requirements.txt
python -m spacy download en_core_web_sm
```
Then, collect the relevant files from the ACE data distribution with
```
bash ./scripts/data/ace-event/collect_ace_event.sh [path-to-ACE-data].
```
The results will go in `./data/ace-event/raw-data`.

Now, run the script
```
python ./scripts/data/ace-event/parse_ace_event.py [output-name] [optional-flags]
```
You can see the available flags by calling `parse_ace_event.py -h`. For detailed descriptions, see [data.md](doc/data.md). The results will go in `./data/ace-event/processed-data/[output-name]`. We require an output name because you may want to preprocess the ACE data multiple times using different flags. For default preprocessing settings, you could do:
```
python ./scripts/data/ace-event/parse_ace_event.py default-settings
```
Now `conda deactivate` the `ace-event-preprocess` environment and re-activate your modeling environment.

Finally, collate the version of the dataset you just created. For instance, continuing the example above,
```
mkdir -p data/ace-event/collated-data/default-settings/json

python scripts/data/shared/collate.py \
  data/ace-event/processed-data/default-settings/json \
  data/ace-event/collated-data/default-settings/json \
  --file_extension json
```

#### 训练事件抽取模型

To train on the data preprocessed with default settings, enter `bash scripts/train.sh ace05_event`. A model trained in this fashion will reproduce (within 0.1 F1 or so) the results in Table 4 of the paper. To train on a different version, modify `training_config/ace05_event.jsonnet` to point to the appropriate files.

To reproduce the results in Table 1 requires training an ensemble model of 4 trigger detectors. The basic process is as follows:

- Merge the ACE event train + dev data, then create 4 new train / dev splits.
- Train a separate trigger detection model on each split. To do this, modify `training_config/ace05_event.jsonnet` by setting
  ```jsonnet
  model +: {
    modules +: {
      events +: {
        loss_weights: {
          trigger: 1.0,
          arguments: 0.5
        }
      }
    }
  }
  ```
- Make trigger predictions using a majority vote of the 4 ensemble models.
- Use these predicted triggers when making event argument predictions based on the event argument scores output by the model saved at `models/ace05_event`.

If you need more details, email me.


### MECHANIC

You can get the dataset by running `bash scripts/data/get_mechanic.sh`. For detailed training instructions, see the [DyGIE-COFIE](https://github.com/AidaAmini/DyGIE-COFIE) repo.


## Evaluating a model

要检查你的一个模型或一个预训练的模型的性能，你可以使用`allennlp evaluate`命令。

请注意，`allennlp`命令只有在以下情况下才能发现这个软件包中的代码。
- 你从这个项目的根目录`dygiepp`下运行这些命令，或者。
- 你从这个项目的根目录下运行`conda develop .`，将代码添加到你的Python路径。

否则，你会得到一个错误 `ModuleNotFoundError: No module named 'dygie'`.

一般来说，你可以对这样的模式做评估。
```shell
allennlp evaluate \
  [model-file] \
  [data-path] \
  --cuda-device [cuda-device] \
  --include-package dygie \
  --output-file [output-file] # Optional; if not given, prints metrics to console.
```
For example, to evaluate the [pretrained SciERC model](#pretrained-models), you could do
```shell
allennlp evaluate \
  pretrained/scierc.tar.gz \
  data/scierc/normalized_data/json/test.json \
  --cuda-device 2 \
  --include-package dygie
```

为了评估你在SciERC数据上训练的模型，你可以这样做
```shell
allennlp evaluate \
  models/scierc/model.tar.gz \
  data/scierc/normalized_data/json/test.json \
  --cuda-device 2  \
  --include-package dygie \
  --output-file models/scierc/metrics_test.json
```

## 预训练模型

有许多模型可供下载。它们是以它们所训练的数据集命名的。"轻量级 "模型是在数据集上训练的模型，这些数据集上有核心推理标注，但我们没有使用它们。之所以说是 "轻量级"，是因为核心词解析的成本很高，因为它需要预测跨度之间的跨句关系。

如果你想使用这些预训练的模型之一在新的数据集上进行预测，你需要在新的数据集中为实例设置`dataset`字段，以匹配模型训练的`dataset`名称。例如，要使用预训练的SciERC模型进行预测，将新实例中的`dataset`字段设置为`scierc`。关于`dataset`字段的更多信息，请参阅[data.md](doc/data.md)。

要下载所有可用的模型，运行`scripts/pretrained/get_dygiepp_pretrained.sh`。或者，点击下面的链接，只下载一个模型。

### 可用训练好的模型

下面是可用模型的链接，后面是模型训练的 "数据集 "的名称。

- [SciERC](https://s3-us-west-2.amazonaws.com/ai2-s2-research/dygiepp/master/scierc.tar.gz): `scierc`
- [SciERC lightweight](https://s3-us-west-2.amazonaws.com/ai2-s2-research/dygiepp/master/scierc-lightweight.tar.gz): `scierc`
- [GENIA](https://s3-us-west-2.amazonaws.com/ai2-s2-research/dygiepp/master/genia.tar.gz): `genia`
- [GENIA lightweight](https://ai2-s2-research.s3-us-west-2.amazonaws.com/dygiepp/master/genia-lightweight.tar.gz): `genia`
- [ChemProt (lightweight only)](https://ai2-s2-research.s3-us-west-2.amazonaws.com/dygiepp/master/chemprot.tar.gz): `chemprot`
- [ACE05 relation](https://ai2-s2-research.s3-us-west-2.amazonaws.com/dygiepp/master/ace05-relation.tar.gz): `ace05`
- [ACE05 event](https://ai2-s2-research.s3-us-west-2.amazonaws.com/dygiepp/master/ace05-event.tar.gz): `ace-event`
- [MECHANIC "coarse"](https://ai2-s2-mechanic.s3-us-west-2.amazonaws.com/models/mechanic-coarse.tar.gz) `None`
- [MECHANIC "granular"](https://ai2-s2-mechanic.s3-us-west-2.amazonaws.com/models/mechanic-granular.tar.gz) `covid-event`

### 每个预训练模型的性能

- SciERC
  ```
  "_scierc__ner_f1": 0.6846741045214326,
  "_scierc__relation_f1": 0.46236559139784944
  ```

- SciERC lightweight
  ```
  "_scierc__ner_f1": 0.6717245404143566,
  "_scierc__relation_f1": 0.4670588235294118
  ```

- GENIA
  ```
  "_genia__ner_f1": 0.7713070807912737
  ```

- GENIA lightweight
  And the lightweight version:
  ```
  "_genia__ner_f1": 0.7690401296349251
  ```

- ChemProt
  ```
  "_chemprot__ner_f1": 0.9059113300492612,
  "_chemprot__relation_f1": 0.5404867256637169
  ```
  Note that we're doing span-level evaluation using predicted entities. We're also evaluating on all ChemProt relation classes, while the official task only evaluates on a subset (see [Liu et al.](https://www.semanticscholar.org/paper/Attention-based-Neural-Networks-for-Chemical-Liu-Shen/a6261b278d1c2155e8eab7ac12d924fc2207bd04) for details). Thus, our relation extraction performance is lower than, for instance, [Verga et al.](https://www.semanticscholar.org/paper/Simultaneously-Self-Attending-to-All-Mentions-for-Verga-Strubell/48f786f66eb846012ceee822598a335d0388f034), where they use gold entities as inputs for relation prediction.

- ACE05-Relation
  ```
  "_ace05__ner_f1": 0.8634611855386309,
  "_ace05__relation_f1": 0.6484907497565725,
  ```

- ACE05-Event
  ```
  "_ace-event__ner_f1": 0.8927209418006965,
  "_ace-event_trig_class_f1": 0.6998813760379595,
  "_ace-event_arg_class_f1": 0.5,
  "_ace-event__relation_f1": 0.5514950166112956
  ```

## 对现有数据集进行预测
要进行预测，你可以使用`allennlp predict`。例如，要用预训练的scierc模型进行预测，你可以这样做。

```bash
allennlp predict pretrained/scierc.tar.gz \
    data/scierc/normalized_data/json/test.json \
    --predictor dygie \
    --include-package dygie \
    --use-dataset-reader \
    --output-file predictions/scierc-test.jsonl \
    --cuda-device 0 \
    --silent
```
预测结果包括预测标签，以及对数和softmax分数。更多信息见，[docs/data.md](docs/data.md)。

**Caveat**。为预测核心参考簇而训练的模型需要一次性对整个文档进行预测。这可能会导致内存问题。为了解决这个问题，有两个选择。

- 使用一个不做核心推理传播的模型进行预测。这些模型每次预测一个句子，应该不会遇到内存问题。使用 "轻量级 "模型来避免这种情况。要训练你自己的无核心推理模型，在相关的训练配置中把[coref loss weight](https://github.com/dwadden/dygiepp/blob/master/training_config/scierc_working_example.jsonnet#L50)设置为0。
- 将文档分割成小块（5个句子应该是安全的），使用带有coref prop的模型进行预测，然后将它们拼接起来。

See the [docs](https://allenai.github.io/allennlp-docs/api/commands/predict/) for more prediction options.

### 关系抽取评估指标

Following [Li and Ji (2014)](https://www.semanticscholar.org/paper/Incremental-Joint-Extraction-of-Entity-Mentions-and-Li-Ji/ab3f1a4480c1ef8409d1685889600f7efb76af24), 
如果 "其关系类型是正确的，并且两个实体提及的参数的头部偏移量都是正确的"，我们认为预测的关系是正确的。

特别是，我们不要求实体提及参数的类型是正确的，就像一些工作（例如[Zhang等人（2017）]（https://www.semanticscholar.org/paper/End-to-End-Neural-Relation-Extraction-with-Global-Zhang-Zhang/ee13e1a3c1d5f5f319b0bf62f04974165f7b0a37））所做的那样。我们欢迎实现这种替代性评估指标的PR。如果你对此感兴趣，请开一个问题。


## 使用新的数据集

参考 [Formatting a new dataset](doc/data.md#formatting-a-new-dataset).

### 在新的数据集上预测

对一个新的、无标签样本集进行预测。

1. 下载与你的文本领域最接近的[预训练模型](#pretrained-models)。
2. 确保你的新数据集的`dataset`字段与预训练模型的标签命名空间相匹配。参见[这里](doc/model.md#multi-dataset-training)了解更多关于标签命名空间的信息。要查看预训练模型的可用标签命名空间，请使用[print_label_namespaces.py]（scripts/debug/print_label_namespaces.py）。
3. 以与[现有数据集]相同的方式进行预测（#making-predictions-on-existing-datasets）。

```
allennlp predict pretrained/[name-of-pretrained-model].tar.gz \
    [input-path] \
    --predictor dygie \
    --include-package dygie \
    --use-dataset-reader \
    --output-file [output-path] \
    --cuda-device [cuda-device]
```

有几个小技巧可以使事情顺利进行。

1. 如果你要对一个大的数据集进行预测，你可能想lazy地加载它，而不是在预测前把整个数据集加载进来。为了达到这个目的，在上述命令中加入以下标志。
```
  --overrides "{'dataset_reader' +: {'lazy': true}}"
```
2. 如果模型在一个给定的预测中耗尽了GPU内存，它将警告你并继续下一个例子，而不是完全停止。这比另一种方法更不令人厌烦。预测失败的例子仍将被写入指定的`jsonl'输出，但它们将有一个额外的字段`{"_FAILED_PREDICTION": true}`，表明模型在这个例子上耗尽了内存。
3. 要预测的数据集中的`dataset`字段必须与模型训练的`dataset`之一相匹配；否则，模型将不知道哪些标签要应用于预测的数据。


### 在一个新的（标注的）数据集上训练一个模型

按照[训练一个模型](#training-a-model)中描述的过程，但要适当调整输入和输出文件的路径。

# Contact

For questions or problems with the code, create a GitHub issue (preferred) or email `dwadden@cs.washington.edu`.
