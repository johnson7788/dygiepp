import logging
from typing import Dict, List, Optional, Union
import copy

import torch
import torch.nn.functional as F
from overrides import overrides

from allennlp.data import Vocabulary
from allennlp.common.params import Params
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, FeedForward, TimeDistributed
from allennlp.modules.span_extractors import EndpointSpanExtractor
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator

# Import submodules.
from dygie.models.coref import CorefResolver
from dygie.models.ner import NERTagger
from dygie.models.relation import RelationExtractor
from dygie.models.events import EventExtractor
from dygie.data.dataset_readers import document

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("dygie")
class DyGIE(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 modules,  # TODO(dwadden) Add type.
                 feature_size: int,
                 max_span_width: int,
                 target_task: str,
                 feedforward_params: Dict[str, Union[int, float]],
                 loss_weights: Dict[str, float],
                 initializer: InitializerApplicator = InitializerApplicator(),
                 module_initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 display_metrics: List[str] = None) -> None:
        super(DyGIE, self).__init__(vocab, regularizer)
        """
        TODO(dwadden) document me.

        Parameters
        ----------
        vocab : ``Vocabulary``  从预训练模型加载的vocab
        text_field_embedder : ``TextFieldEmbedder``
            Used to embed the ``text`` ``TextField`` we get as input to the model.
        context_layer : ``Seq2SeqEncoder``
            这一层包含了文件中每个词的上下文信息。
        feature_size: ``int``
            所有嵌入特征的嵌入尺寸，如距离或跨度宽度。
        submodule_params: ``TODO(dwadden)``
            一个嵌套的字典，指定要传递给子模块初始化的参数。
        max_span_width: ``int``
            候选跨度的最大宽度。
        target_task: ``str``:
            用于作出早期停止决定的任务。
        initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
            用于初始化模型参数。
        module_initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
            用来初始化各个模块。
        regularizer : ``RegularizerApplicator``, optional (default=``None``)
            如果提供，将用于计算训练期间的正则化惩罚。
        display_metrics: ``List[str]``.  在模型训练期间应该打印出来的指标列表
            .
        """

        ####################

        # 创建一个 span 提取器.
        self._endpoint_span_extractor = EndpointSpanExtractor(
            embedder.get_output_dim(),
            combination="x,y",
            num_width_embeddings=max_span_width,
            span_width_embedding_dim=feature_size,
            bucket_widths=False)
        """
        将跨度表示为其端点的嵌入的组合。此外，跨度的宽度可以被嵌入并拼接到最终的组合上。
        假设`x = span_start_embeddings`和`y = span_end_embeddings`，支持以下类型的表示。
       `x`, `y`, `x*y`, `x+y`, `x-y`, `x/y`, 其中每个二进制运算都是按元素进行的。 你可以列出你想要的组合，以逗号分隔。
        例如，你可以给`x,y,x*y`作为这个类的`组合`参数。
        然后计算出的相似性函数将是`[x; y; x*y]`，然后可以选择与span的宽度的嵌入式表示相拼接。
        注册为`SpanExtractor`，名称为 "endpoint"。
        """
        ####################

        # Set parameters.
        self._embedder = embedder
        # 配置文件中的损失的权重 scierc.jsonnet
        self._loss_weights = loss_weights
        # 最大span宽度
        self._max_span_width = max_span_width
        # 获取任务显示的metric
        self._display_metrics = self._get_display_metrics(target_task)
        token_emb_dim = self._embedder.get_output_dim()
        span_emb_dim = self._endpoint_span_extractor.get_output_dim()
        ####################
        # 创建子moduel
        modules = Params(modules)

        # 前馈网络函数
        def make_feedforward(input_dim):
            return FeedForward(input_dim=input_dim,
                               num_layers=feedforward_params["num_layers"],
                               hidden_dims=feedforward_params["hidden_dims"],
                               activations=torch.nn.ReLU(),
                               dropout=feedforward_params["dropout"])

        # 实体识别模块， span_emb_dim： 1556， feature_size：20
        self._ner = NERTagger.from_params(vocab=vocab,
                                          make_feedforward=make_feedforward,
                                          span_emb_dim=span_emb_dim,
                                          feature_size=feature_size,
                                          params=modules.pop("ner"))
        # 指代关系模块
        self._coref = CorefResolver.from_params(vocab=vocab,
                                                make_feedforward=make_feedforward,
                                                span_emb_dim=span_emb_dim,
                                                feature_size=feature_size,
                                                params=modules.pop("coref"))
        # 关系抽取模块
        self._relation = RelationExtractor.from_params(vocab=vocab,
                                                       make_feedforward=make_feedforward,
                                                       span_emb_dim=span_emb_dim,
                                                       feature_size=feature_size,
                                                       params=modules.pop("relation"))
        # 事件抽取模块
        self._events = EventExtractor.from_params(vocab=vocab,
                                                  make_feedforward=make_feedforward,
                                                  token_emb_dim=token_emb_dim,
                                                  span_emb_dim=span_emb_dim,
                                                  feature_size=feature_size,
                                                  params=modules.pop("events"))

        ####################
        # 初始化文本嵌入器和所有子模块
        for module in [self._ner, self._coref, self._relation, self._events]:
            # 根据regex匹配，对模块的参数应用初始化。 任何没有明确匹配的参数都不会被初始化，而是使用模块代码中的任何默认初始化。
            module_initializer(module)
        initializer(self)

    @staticmethod
    def _get_display_metrics(target_task):
        """
        target 是用于做出早期停止决定的任务的名称。显示与该任务相关的指标。
        """
        lookup = {
            "ner": [f"MEAN__{name}" for name in
                    ["ner_precision", "ner_recall", "ner_f1"]],
            "relation": [f"MEAN__{name}" for name in
                         ["relation_precision", "relation_recall", "relation_f1"]],
            "coref": ["coref_precision", "coref_recall", "coref_f1", "coref_mention_recall"],
            "events": [f"MEAN__{name}" for name in
                       ["trig_class_f1", "arg_class_f1"]]}
        if target_task not in lookup:
            raise ValueError(f"Invalied value {target_task} has been given as the target task.")
        return lookup[target_task]

    @staticmethod
    def _debatch(x):
        # TODO(dwadden) Get rid of this when I find a better way to do it.
        return x if x is None else x.squeeze(0)

    @overrides
    def forward(self,
                text,
                spans,
                metadata,
                ner_labels=None,
                coref_labels=None,
                relation_labels=None,
                trigger_labels=None,
                argument_labels=None):
        """
        :params text: {"bert": {"token_ids", "mask", "type_ids", "wordpiece_mask",}}
        token_ids: shape: [1,4,47], 【文档数，句子数，单词数】
        """
        # 在AllenNLP中，AdjacencyFields是以浮点数形式传递的。这就解决了这个问题。
        # relation_labels类型转换，shape: [1,4,268,268]，
        if relation_labels is not None:
            relation_labels = relation_labels.long()
        if argument_labels is not None:
            argument_labels = argument_labels.long()
        # 目前还不支持多文档的mini batch。目前，摆脱了输入张量中的额外维度。一旦模型运行，将回到这个问题上。
        if len(metadata) > 1:
            raise NotImplementedError("尚不支持多文档mini-batch处理。")
        # metadata是一篇文档中的内容
        metadata = metadata[0]
        spans = self._debatch(spans)  #_debatch函数去掉文档的维度1， (n_sents, max_n_spans, 2) eg： 【4，268，2】
        ner_labels = self._debatch(ner_labels)  # (n_sents, max_n_spans) 【4，268】
        coref_labels = self._debatch(coref_labels)  #  (n_sents, max_n_spans) 【4，268】
        relation_labels = self._debatch(relation_labels)  # (n_sents, max_n_spans, max_n_spans)  【4，268， 268】
        trigger_labels = self._debatch(trigger_labels)  # TODO(dwadden)
        argument_labels = self._debatch(argument_labels)  # TODO(dwadden)

        # 使用BERT进行编码，然后进行debatch
        # 由于数据是分批的，我们使用`num_wrapping_dims=1`来debatch文档维度。
        # (1, n_sents, max_sententence_length, embedding_dim) ，1 代表文档数量
        # n_sents：句子数量， max_sententence_length：最大句子长度，embedding_dim：嵌入维度

        # TODO处理输入长度超过512的情况。
        text_embeddings = self._embedder(text, num_wrapping_dims=1)
        # (n_sents, max_n_wordpieces, embedding_dim)， 形状 【4，37，768】
        text_embeddings = self._debatch(text_embeddings)

        # (n_sents, max_sentence_length)
        """
        AllenNLP中的util的 get_text_field_mask: 函数
        接收由`TextField`产生的张量字典，并返回一个mask，其中0是填充的token，否则是1。`padding_id`指定了填充token的id。    
        我们也处理被任意数量的 "ListField "包裹的 "TextField"，其中包裹的 "ListField "的数量由 "num_wrapping_dims "给出。     
        如果`num_wrapping_dims == 0`，返回的mask形状为`(batch_size, num_tokens)`。   
        如果`num_wrapping_dims > 0`，那么返回的mask有`num_wrapping_dims`的额外维度，
        所以形状将是`(batch_size, ..., num_tokens)`。    
        张量字典中可能有几个具有不同形状的条目（例如，一个用于单词ID，一个用于字符ID）。 为了得到一个token mask，我们使用字典中维度最少的张量。 在减去`num_wrapping_dims'后，如果这个张量有两个维度，我们假定它的形状是`(batch_size, ..., num_tokens)`，并使用它作为mask。 
        如果它有三个维度，我们假定它的形状是`(batch_size, ..., num_tokens, num_features)`，并在最后一个维度上求和，以产生mask。 
        最常见的是一个字符ID张量，但它也可以是每个token的特征表示，等等。     
        如果输入的`text_field_tensors`包含 "mask "键，将返回这个键而不是推理出mask。
        """
        text_mask = self._debatch(util.get_text_field_mask(text, num_wrapping_dims=1).float())
        # 文本真实的长度， eg： tensor([37, 26, 37, 17], device='cuda:0')
        sentence_lengths = text_mask.sum(dim=1).long()  # (n_sents)

        span_mask = (spans[:, :, 0] >= 0).float()  # (n_sents, max_n_spans)
        # SpanFields在被用作填充时返回 - 1。由于我们在参加由这些索引生成的跨度表示时，根据跨度宽度进行了一些比较，我们需要它们 <= 0。这只与我们在剪枝阶段后考虑的跨度数量 >= 跨度总数的边缘情况有关，因为在这种情况下，我们有可能考虑一个被mask的跨度。
        spans = F.relu(spans.float()).long()  # (n_sents, max_n_spans, 2)
        # text_embeddings： 【4，37，768】， spans 【4，268，2】
        # Shape: (batch_size, num_spans, 2 * encoding_dim + feature_size)
        span_embeddings = self._endpoint_span_extractor(text_embeddings, spans)

        # 保存损失
        output_coref = {'loss': 0}
        output_ner = {'loss': 0}
        output_relation = {'loss': 0}
        output_events = {'loss': 0}

        # 为指代关系模块修剪和计算跨度表示
        if self._loss_weights["coref"] > 0 or self._coref.coref_prop > 0:
            output_coref, coref_indices = self._coref.compute_representations(
                spans, span_mask, span_embeddings, sentence_lengths, coref_labels, metadata)

        # 传播全局信息以加强跨度嵌入
        if self._coref.coref_prop > 0:
            output_coref = self._coref.coref_propagation(output_coref)
            span_embeddings = self._coref.update_spans(
                output_coref, span_embeddings, coref_indices)

        # 对每个模块进行预测并计算损失
        if self._loss_weights['ner'] > 0:
            output_ner = self._ner(
                spans, span_mask, span_embeddings, sentence_lengths, ner_labels, metadata)

        if self._loss_weights['coref'] > 0:
            output_coref = self._coref.predict_labels(output_coref, metadata)

        if self._loss_weights['relation'] > 0:
            output_relation = self._relation(
                spans, span_mask, span_embeddings, sentence_lengths, relation_labels, metadata)

        if self._loss_weights['events'] > 0:
            # `text_embeddings`作为事件触发器的代表。
            output_events = self._events(
                text_mask, text_embeddings, spans, span_mask, span_embeddings,
                sentence_lengths, trigger_labels, argument_labels,
                ner_labels, metadata)

        # 使用`get`，因为在某些情况下，输出的dict不会有损失--例如，在做预测的时候。
        loss = (self._loss_weights['coref'] * output_coref.get("loss", 0) +
                self._loss_weights['ner'] * output_ner.get("loss", 0) +
                self._loss_weights['relation'] * output_relation.get("loss", 0) +
                self._loss_weights['events'] * output_events.get("loss", 0))

        # 将损失乘以该文档的权重乘数。
        weight = metadata.weight if metadata.weight is not None else 1.0
        loss *= torch.tensor(weight)

        output_dict = dict(coref=output_coref,
                           relation=output_relation,
                           ner=output_ner,
                           events=output_events)
        output_dict['loss'] = loss

        output_dict["metadata"] = metadata

        return output_dict

    def update_span_embeddings(self, span_embeddings, span_mask, top_span_embeddings,
                               top_span_mask, top_span_indices):
        # TODO(Ulme) Speed this up by tensorizing

        new_span_embeddings = span_embeddings.clone()
        for sample_nr in range(len(top_span_mask)):
            for top_span_nr, span_nr in enumerate(top_span_indices[sample_nr]):
                if top_span_mask[sample_nr, top_span_nr] == 0 or span_mask[sample_nr, span_nr] == 0:
                    break
                new_span_embeddings[sample_nr,
                                    span_nr] = top_span_embeddings[sample_nr, top_span_nr]
        return new_span_embeddings

    @overrides
    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]):
        """
        Converts the list of spans and predicted antecedent indices into clusters
        of spans for each element in the batch.

        Parameters
        ----------
        output_dict : ``Dict[str, torch.Tensor]``, required.
            The result of calling :func:`forward` on an instance or batch of instances.

        Returns
        -------
        The same output dictionary, but with an additional ``clusters`` key:

        clusters : ``List[List[List[Tuple[int, int]]]]``
            A nested list, representing, for each instance in the batch, the list of clusters,
            which are in turn comprised of a list of (start, end) inclusive spans into the
            original document.
        """

        doc = copy.deepcopy(output_dict["metadata"])

        if self._loss_weights["coref"] > 0:
            # TODO(dwadden) Will need to get rid of the [0] when batch training is enabled.
            decoded_coref = self._coref.make_output_human_readable(output_dict["coref"])["predicted_clusters"][0]
            sentences = doc.sentences
            sentence_starts = [sent.sentence_start for sent in sentences]
            predicted_clusters = [document.Cluster(entry, i, sentences, sentence_starts)
                                  for i, entry in enumerate(decoded_coref)]
            doc.predicted_clusters = predicted_clusters
            # TODO(dwadden) update the sentences with cluster information.

        if self._loss_weights["ner"] > 0:
            for predictions, sentence in zip(output_dict["ner"]["predictions"], doc):
                sentence.predicted_ner = predictions

        if self._loss_weights["relation"] > 0:
            for predictions, sentence in zip(output_dict["relation"]["predictions"], doc):
                sentence.predicted_relations = predictions

        if self._loss_weights["events"] > 0:
            for predictions, sentence in zip(output_dict["events"]["predictions"], doc):
                sentence.predicted_events = predictions

        return doc

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """
        Get all metrics from all modules. For the ones that shouldn't be displayed, prefix their
        keys with an underscore.
        """
        metrics_coref = self._coref.get_metrics(reset=reset)
        metrics_ner = self._ner.get_metrics(reset=reset)
        metrics_relation = self._relation.get_metrics(reset=reset)
        metrics_events = self._events.get_metrics(reset=reset)

        # Make sure that there aren't any conflicting names.
        metric_names = (list(metrics_coref.keys()) + list(metrics_ner.keys()) +
                        list(metrics_relation.keys()) + list(metrics_events.keys()))
        assert len(set(metric_names)) == len(metric_names)
        all_metrics = dict(list(metrics_coref.items()) +
                           list(metrics_ner.items()) +
                           list(metrics_relation.items()) +
                           list(metrics_events.items()))

        # If no list of desired metrics given, display them all.
        if self._display_metrics is None:
            return all_metrics
        # Otherwise only display the selected ones.
        res = {}
        for k, v in all_metrics.items():
            if k in self._display_metrics:
                res[k] = v
            else:
                new_k = "_" + k
                res[new_k] = v
        return res
