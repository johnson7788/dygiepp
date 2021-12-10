## 测试预测sciERC数据
allennlp predict pretrained/scierc.tar.gz 
    data/scierc/processed_data/json/test.json \
    --predictor dygie \
    --include-package dygie \
    --use-dataset-reader \
    --output-file data/scierc/processed_data/json/test_predict.txt \
    --cuda-device 0 \