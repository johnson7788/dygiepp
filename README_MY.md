## 测试预测sciERC数据
allennlp predict pretrained/scierc.tar.gz data/scierc/normalized_data/json/test.json \
    --predictor dygie \
    --include-package dygie \
    --use-dataset-reader \
    --output-file data/scierc/normalized_data/json/test_predict.txt \
    --cuda-device 0

## 训练模型, 数据以sciERC为例
allennlp train training_config/scierc.jsonnet --serialization-dir models/scierc --include-package dygie

### 强制覆盖输出文件目录
allennlp train training_config/scierc.jsonnet --serialization-dir models/scierc --include-package dygie --force