import json
import os
import uuid

import sagemaker
import sagemaker.huggingface
from sagemaker.huggingface import HuggingFace
from sagemaker.inputs import FileSystemInput

sess = sagemaker.Session()
sagemaker_session_bucket = None
if sagemaker_session_bucket is None and sess is not None:
    # set to default bucket if a bucket name is not given
    sagemaker_session_bucket = sess.default_bucket()
sess = sagemaker.Session(default_bucket=sagemaker_session_bucket)

# GPT-J モデル設定
MODEL_NAME = "gpt-j"
MAX_CONTEXT_WIDTH = 20
# MAX_CONTEXT_WIDTH = 2048
NUM_LAYERS = 28
# NUM_LAYERS = 2
HIDDEN_WIDTH = 200
# HIDDEN_WIDTH = 4096 
NUM_HEADS = 10
# NUM_HEADS = 16


# よく変更する学習ジョブ設定
N_INSTANCE = 1
# N_INSTANCE = 4
MAX_STEP = 30
# MAX_STEP = 143000
TP_DEGREE = 8
ACTIVATION_CHECKPOINTING = 1
BATCH_SIZE = 8
NUM_KEPT_CHECKPOINTS = 3
CHECKPOINT_FREQ = 3
# CHECKPOINT_FREQ = 2000
INSTANCE_TYPE = "ml.p4d.24xlarge"
VOLUME_SIZE = 1024
MAX_RUN = 86400


# 秘匿しておきたい AWS 環境の設定を config.json から読み込みます
with open("config.json") as f:
    config = json.loads(f.read())
ROLE = config["aws_config"]["role"]
SUBNETS = config["fsx_config"]["subnets"]
SECURITY_GROUP_IDS = config["fsx_config"]["security_group_ids"]

FILE_SYSTEM_ID = config["fsx_config"]["file_system_id"]
TRAIN_DIRECTORY_PATH = config["fsx_config"]["train_directory_path"]
VALIDATIOM_DIRECTORY_PATH = config["fsx_config"]["validation_directory_path"]


if __name__ == "__main__":
    # 学習ジョブに渡すハイパラを格納するために準備
    hyperparameters = {}

    training_config = {
        "max_steps": MAX_STEP,
        "seed": 12345,
        "fp16": 1,
        "lr": 1.2e-4,
        "lr_decay_style": "cosine",
        "lr_decay_iters": 20000,
        "min_lr": 0.00001,
        "plateau": 0.0,
        "warmup": 0.003,
        "num_kept_checkpoints": NUM_KEPT_CHECKPOINTS,
        "checkpoint_freq": CHECKPOINT_FREQ,
        "validation_freq": 200,
        # "enable_memory_profiling": 1,
        "logging_freq": 200,
        "zipped_data": 0,
        "data_type": "c4",
        "train_batch_size": BATCH_SIZE,
        "val_batch_size": BATCH_SIZE,
        "logits_output": "logits_output",
        "weight_decay": 0.2,
        "use_adamw": 1, 
        # below flag loads model and optimizer state from checkpoint_s3_uri
        # 'load_partial': 1,
    }
    # ジョブにわたすハイパラ定義に格納
    hyperparameters.update(training_config)

    model_config = {
        "max_context_width": MAX_CONTEXT_WIDTH,
        "hidden_width": HIDDEN_WIDTH,
        "num_layers": NUM_LAYERS,
        "num_heads": NUM_HEADS,
    }
    hyperparameters.update(model_config)


    # SageMaker でのが分散学習などに必要な設定を定義
    smp_configs = {
        "ddp": True,
        "tensor_parallel_degree": TP_DEGREE,
        "pipeline_parallel_degree": 1,
        "save_final_full_model": 1,
        "manual_partition": 1,
        "skip_full_optimizer": 1,
        "shard_optimizer_state": 1,
        "activation_checkpointing": ACTIVATION_CHECKPOINTING,
        "activation_strategy": "each",
        "optimize": "speed",
        "prescaled_batch": 0,
        "active_microbatches": 1,
        "microbatches": 2,
    }
    hyperparameters.update(smp_configs)


    # SageMaker HuggingFace Estimater に分散学習を指定する distribution についての設定をします
    mpioptions = "-x NCCL_DEBUG=WARN -x SMDEBUG_LOG_LEVEL=ERROR "
    mpioptions += (
        "-x SMP_DISABLE_D2D=1 -x SMP_D2D_GPU_BUFFER_SIZE_BYTES=1 -x SMP_NCCL_THROTTLE_LIMIT=1 "
    )
    mpioptions += "-x FI_EFA_USE_DEVICE_RDMA=1 -x FI_PROVIDER=efa -x RDMAV_FORK_SAFE=1"
    mpi_config = {
        "enabled": True,
        "processes_per_host": 8,
        "custom_mpi_options": mpioptions
    }

    smp_parameters = {
        "ddp": hyperparameters["ddp"],
        "tensor_parallel_degree": hyperparameters["tensor_parallel_degree"],
        # partitions is a required param in the current SM SDK so it needs to be passed,
        # these two map to the same config
        "partitions": hyperparameters["pipeline_parallel_degree"],
        "shard_optimizer_state": hyperparameters["shard_optimizer_state"] > 0,
        "prescaled_batch": hyperparameters["prescaled_batch"] > 0,
        "fp16_params": hyperparameters["fp16"] > 0,
        "optimize": hyperparameters["optimize"],
        "auto_partition": False if hyperparameters["manual_partition"] else True,
        "default_partition": 0,
        "microbatches": hyperparameters["microbatches"],
        "active_microbatches": hyperparameters["active_microbatches"],
    }

    # distribution を HuggingFace Estimator クラスに渡すことで、分散学習が反映されます
    distribution = {
        "mpi": mpi_config,
        "smdistributed": {
            "modelparallel": {
                "enabled": True,
                "parameters": smp_parameters
                },
        }
    }



    # 学習ジョブ名の prefix を自由に自由に設定できます。
    model_name = MODEL_NAME
    max_context_width = model_config["max_context_width"] 
    n_layers = model_config["num_layers"]
    base_job_name = f'GPT6B-nl{n_layers}-AC{ACTIVATION_CHECKPOINTING}-TP{TP_DEGREE}-BS{BATCH_SIZE}-ninstance{N_INSTANCE}-megagon'
    
    # 学習中に出力されたチェックポイントの S3 での保存先を指定します。
    checkpoint_id = uuid.uuid4().hex
    checkpoint_s3_uri = "s3://ricoh-poc/output/" + checkpoint_id

    # チェックポイントの保存先を指定し、学習ジョブ内のコンテナから
    SM_CHECKPOINT_DIR = "/opt/ml/checkpoints"
    hyperparameters["checkpoint-dir"] = f"{SM_CHECKPOINT_DIR}"

    metric_definitions = [
        {"Name": "Val loss", "Regex": "Validation loss: ([0-9.]+$)"},
        {"Name": "Val ppl", "Regex": "Validation perplexity: ([0-9.]+$)"},
    ]

    huggingface_estimator = HuggingFace(
        entry_point="train_gpt_simple.py",
        source_dir=os.getcwd(),
        role=ROLE,
        metrics_definition=metric_definitions,
        instance_type=INSTANCE_TYPE,
        instance_count=N_INSTANCE,
        subnets=SUBNETS,
        security_group_ids=SECURITY_GROUP_IDS,
        volume_size=VOLUME_SIZE,
        max_run=MAX_RUN,
        transformers_version="4.17",
        pytorch_version="1.10",
        py_version="py38",
        distribution=distribution,
        hyperparameters=hyperparameters,
        checkpoint_s3_uri=checkpoint_s3_uri,
        checkpoint_local_path=SM_CHECKPOINT_DIR,
        debugger_hook_config=False,
        disable_profiler=True,
        base_job_name=base_job_name
    )

    # FSx for Lustre を使う場合は下記の　fs_input と、fit() を活用してください
    train_fs_input = FileSystemInput(file_system_id=FILE_SYSTEM_ID,
                                       file_system_type="FSxLustre",
                                        directory_path=TRAIN_DIRECTORY_PATH,
                                        file_system_access_mode='ro')
    val_fs_input = FileSystemInput(file_system_id=FILE_SYSTEM_ID,
                                        file_system_type="FSxLustre",
                                        directory_path=VALIDATIOM_DIRECTORY_PATH,
                                        file_system_access_mode='ro')
    huggingface_estimator.fit({"train": train_fs_input, "test": val_fs_input})
    
    # S3 を使う場合はこちらのコメントアウトされている部分を活用してください
    # train_dir = "s3://ricoh-poc/c4/ja_gpt_2048/train/"
    # validation_dir = "s3://ricoh-poc/c4/ja_gpt_2048/validation/"
    # huggingface_estimator.fit({"train": train_dir, "test": validation_dir})

