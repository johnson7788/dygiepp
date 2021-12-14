import logging
from typing import Any, Dict, List, Optional, Callable

import torch
from torch.nn import functional as F
from overrides import overrides

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TimeDistributed
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator

from dygie.training.ner_metrics import NERMetrics
from dygie.data.dataset_readers import document

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class NERTagger(Model):
    """
    DyGIE model的实体识别模块

    Parameters
    ----------
    mention_feedforward : ``FeedForward``
    这个前馈网络适用于跨度表示，然后由一个线性层进行评分。
    feature_size: ``int``
       所有嵌入特征的嵌入尺寸，如距离或跨度宽度。
    lexical_dropout: ``int``
        嵌入文本的dropout维度的概率
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        如果提供，将用于计算训练期间的正则化惩罚。
    """

    def __init__(self,
                 vocab: Vocabulary,
                 make_feedforward: Callable,
                 span_emb_dim: int,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(NERTagger, self).__init__(vocab, regularizer)
        # 获取对应的namespace： 'scierc__ner_labels'
        self._namespaces = [entry for entry in vocab.get_namespaces() if "ner_labels" in entry]

        # 类别的数量决定了最终层的输出维度， 这里是：7
        self._n_labels = {name: vocab.get_vocab_size(name) for name in self._namespaces}

        # 在计算指标时，需要有空标签来跟踪。空标签代表实体是没有类别的
        for namespace in self._namespaces:
            null_label = vocab.get_token_index("", namespace)
            assert null_label == 0, "空标签的类别索引必须是0"  # 如果不是这样，dumpy类别就不会对应于空标签。

        # 输出的dim比标签的数量少1，因为我们不给空标签打分。
        # 我们只是默认给它打0分。

        # 为我们要处理的每个数据集创建一个单独的评分器和指标。
        self._ner_scorers = torch.nn.ModuleDict()
        self._ner_metrics = {}
        for namespace in self._namespaces:  # 这里只有一个namespace， 'scierc__ner_labels'
            mention_feedforward = make_feedforward(input_dim=span_emb_dim)
            self._ner_scorers[namespace] = torch.nn.Sequential(
                TimeDistributed(mention_feedforward),
                TimeDistributed(torch.nn.Linear(
                    mention_feedforward.get_output_dim(),
                    self._n_labels[namespace] - 1)))
            """
        TimeDistributed
        给出一个形状像`(batch_size, time_steps, [rest])`的输入和一个接受像`(batch_size, [rest])`的输入的`Module`，`TimeDistributed`将输入重塑为`(batch_size * time_steps, [rest])`，应用包含的`Module`，然后将它重塑回来。
        注意，虽然上面给出的形状是以`batch_size'为先，但如果`batch_size'为后，这个`Module'也可以工作--我们总是只合并前两个维度，然后再分割。
        它也会重塑关键字参数，除非它们不是张量或者它们的名字在可选的`pass_through`迭代中被指定。
            """
            # null_label 空标签是0
            self._ner_metrics[namespace] = NERMetrics(self._n_labels[namespace], null_label)

        self._active_namespace = None
        # 创建一个交叉熵损失
        self._loss = torch.nn.CrossEntropyLoss(reduction="sum")

    @overrides
    def forward(self,  # type: ignore
                spans: torch.IntTensor,
                span_mask: torch.IntTensor,
                span_embeddings: torch.IntTensor,
                sentence_lengths: torch.Tensor,
                ner_labels: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        """
        TODO(dwadden) Write documentation.
        """

        # Shape: (Batch size, Number of Spans, Span Embedding Size)
        # span_embeddings
        if metadata.dataset == None:
            metadata.dataset = "scierc"
        self._active_namespace = f"{metadata.dataset}__ner_labels"
        scorer = self._ner_scorers[self._active_namespace]

        ner_scores = scorer(span_embeddings)
        # Give large negative scores to masked-out elements.
        mask = span_mask.unsqueeze(-1)
        ner_scores = util.replace_masked_values(ner_scores, mask.bool(), -1e20)
        # The dummy_scores are the score for the null label.
        dummy_dims = [ner_scores.size(0), ner_scores.size(1), 1]
        dummy_scores = ner_scores.new_zeros(*dummy_dims)
        ner_scores = torch.cat((dummy_scores, ner_scores), -1)

        _, predicted_ner = ner_scores.max(2)

        predictions = self.predict(ner_scores.detach().cpu(),
                                   spans.detach().cpu(),
                                   span_mask.detach().cpu(),
                                   metadata)
        output_dict = {"predictions": predictions}

        if ner_labels is not None:
            metrics = self._ner_metrics[self._active_namespace]
            metrics(predicted_ner, ner_labels, span_mask)
            ner_scores_flat = ner_scores.view(-1, self._n_labels[self._active_namespace])
            ner_labels_flat = ner_labels.view(-1)
            mask_flat = span_mask.view(-1).bool()

            loss = self._loss(ner_scores_flat[mask_flat], ner_labels_flat[mask_flat])

            output_dict["loss"] = loss

        return output_dict

    def predict(self, ner_scores, spans, span_mask, metadata):
        # TODO(dwadden) Make sure the iteration works in documents with a single sentence.
        # Zipping up and iterating iterates over the zeroth dimension of each tensor; this
        # corresponds to iterating over sentences.
        predictions = []
        zipped = zip(ner_scores, spans, span_mask, metadata)
        for ner_scores_sent, spans_sent, span_mask_sent, sentence in zipped:
            predicted_scores_raw, predicted_labels = ner_scores_sent.max(dim=1)
            softmax_scores = F.softmax(ner_scores_sent, dim=1)
            predicted_scores_softmax, _ = softmax_scores.max(dim=1)
            ix = (predicted_labels != 0) & span_mask_sent.bool()

            predictions_sent = []
            zip_pred = zip(predicted_labels[ix], predicted_scores_raw[ix],
                           predicted_scores_softmax[ix], spans_sent[ix])
            for label, label_score_raw, label_score_softmax, label_span in zip_pred:
                label_str = self.vocab.get_token_from_index(label.item(), self._active_namespace)
                span_start, span_end = label_span.tolist()
                ner = [span_start, span_end, label_str, label_score_raw.item(),
                       label_score_softmax.item()]
                prediction = document.PredictedNER(ner, sentence, sentence_offsets=True)
                predictions_sent.append(prediction)

            predictions.append(predictions_sent)

        return predictions

    # TODO(dwadden) This code is repeated elsewhere. Refactor.
    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        "Loop over the metrics for all namespaces, and return as dict."
        res = {}
        for namespace, metrics in self._ner_metrics.items():
            precision, recall, f1 = metrics.get_metric(reset)
            prefix = namespace.replace("_labels", "")
            to_update = {f"{prefix}_precision": precision,
                         f"{prefix}_recall": recall,
                         f"{prefix}_f1": f1}
            res.update(to_update)

        res_avg = {}
        for name in ["precision", "recall", "f1"]:
            values = [res[key] for key in res if name in key]
            res_avg[f"MEAN__ner_{name}"] = sum(values) / len(values) if values else 0
            res.update(res_avg)

        return res
