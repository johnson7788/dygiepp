# USAGE: `bash train.sh [config_name]`
#
# `config_name`是`training_config`目录下的一个`jsonnet`配置文件的名称，例如`scierc`。训练的结果将被放在`models/[config_name]`下。

config_name=$1

allennlp train "training_config/${config_name}.jsonnet" \
    --serialization-dir "models/${config_name}" \
    --include-package dygie
