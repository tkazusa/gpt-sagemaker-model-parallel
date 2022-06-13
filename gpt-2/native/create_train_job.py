import json
import os

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

if __name__ == "__main__":

    hyperparameters = {}

    training_config = {
        "max_steps": 5,
        "seed": 12345,
        "fp16": 1,
        "lr": 2.0e-4,
        "lr_decay_iters": 125000,
        "min_lr": 0.00001,
        "lr-decay-style": "linear",
        "warmup": 0.01,
        "num_kept_checkpoints": 5,
        "checkpoint_freq": 200,
        "logging_freq": 1,
        "zipped_data": 0,
        "data_type": "c4/en",
        # below flag loads model and optimizer state from checkpoint_s3_uri
        # 'load_partial': 1,
    }
    hyperparameters.update(training_config)

    # Params required by SMP
    smp_configs = {
        "save_final_full_model": 0,
        "manual_partition": 1,
        "skip_full_optimizer": 1,
        "shard_optimizer_state": 1,
        "activation_checkpointing": 1,
        "activation_strategy": "each",
        "optimize": "speed",

    }
    hyperparameters.update(smp_configs)

    # SM_DATA_DIR = "/opt/ml/input/data"
    # hyperparameters["checkpoint-dir"] = f"{SM_DATA_DIR}/checkpointdir"
    # hyperparameters["model-dir"] = f"{SM_DATA_DIR}/modeldir"
    # hyperparameters["training-dir"] = f"{SM_DATA_DIR}/train"
    # hyperparameters["test-dir"] = f"{SM_DATA_DIR}/validation"


    model_config = {
        "max_context_width": 512,
        "hidden_width": 768,
        "num_layers": 12,
        "num_heads": 12,
        "tensor_parallel_degree": 4,
        "pipeline_parallel_degree": 1,
        "train_batch_size": 2,
        "val_batch_size": 4,
        "prescaled_batch": 0,
    }
    hyperparameters.update(model_config)


    metric_definitions = [
        {"Name": "train_runtime", "Regex": "train_runtime.*=\D*(.*?)$"},
        {"Name": "train_samples_per_second", "Regex": "train_samples_per_second.*=\D*(.*?)$"},
        {"Name": "epoch", "Regex": "epoch.*=\D*(.*?)$"},
        {"Name": "f1", "Regex": "f1.*=\D*(.*?)$"},
        {"Name": "exact_match", "Regex": "exact_match.*=\D*(.*?)$"},
    ]
    
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
        "ddp": True,
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
        "fp16_params": hyperparameters["fp16"] > 0,
        "optimize": hyperparameters["optimize"],
    }


    distribution = {
        "mpi": mpi_config,
        "smdistributed": {
            "modelparallel": {
                "enabled": True,
                "parameters": smp_parameters
                },
        }
    }

    # instance configurations
    instance_type = "ml.p3.16xlarge"
    instance_count = 1
    volume_size = 500

    with open("config.json") as f:
        config = json.loads(f.read())
    role = config["aws_config"]["role"]

    machine_str = instance_type.split(".")[1] + instance_type.split(".")[2][:3]
    pp_degree = hyperparameters["pipeline_parallel_degree"]
    tp_degree = hyperparameters["tensor_parallel_degree"]
    base_job_name = f'smp-{model_config}-{machine_str}-tp{tp_degree}-pp{pp_degree}-bs{hyperparameters["train_batch_size"]}'


    huggingface_estimator = HuggingFace(
        entry_point="train_gpt_simple.py",
        source_dir=os.getcwd(),
        role=role,
        metrics_definition=metric_definitions,
        instance_type=instance_type,
        instance_count=instance_count,
        # subnets=['xxxxxxxxxxxxxx'],
        # security_group_ids=['xxxxxxxxxxxxxx'],
        volume_size=volume_size,
        transformers_version="4.17",
        pytorch_version="1.10",
        py_version="py38",
        distribution=distribution,
        hyperparameters=hyperparameters,
        debugger_hook_config=False,
    )

    #train_fs_input = FileSystemInput(file_system_id=file_sytem_id,
    #                                    file_system_type="FSxLustre",
    #                                    directory_path=train_directory_path,
    #                                    file_system_access_mode='ro')
    #val_fs_input = FileSystemInput(file_system_id=file_sytem_id,
    #                                    file_system_type="FSxLustre",
    #                                    directory_path=val_directory_path,
    #                                    file_system_access_mode='ro')

    # print(huggingface_estimator.hyperparameters())
    # starting the train job with our uploaded datasets as input

    train_dir = "s3://ricoh-poc/c4/en_gpt_preprocessed_small/train/"
    validation_dir = "s3://ricoh-poc/c4/en_gpt_preprocessed_small/validation/"

    huggingface_estimator.fit({"train": train_dir, "test": validation_dir})
