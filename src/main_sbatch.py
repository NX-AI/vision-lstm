import os
import platform
import shlex
import sys
import uuid
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import yaml
import kappaconfig as kc


def get_parser():
    parser = ArgumentParser()
    # how many GPUs
    gpus_group = parser.add_mutually_exclusive_group()
    gpus_group.add_argument("--nodes", type=int)
    gpus_group.add_argument("--gpus", type=int)
    #
    parser.add_argument("--time", type=str, required=True)
    parser.add_argument("--account", type=str)
    parser.add_argument("--qos", type=str)
    parser.add_argument("--constraint", type=str)
    parser.add_argument("--preload", type=str)
    parser.add_argument("--env", type=str)
    #
    parser.add_argument("--dependency", type=int)
    # resume
    parser.add_argument("--resume_stage_id", type=str)
    parser.add_argument("--resume_checkpoint", type=str)
    return parser


def main(
        nodes,
        gpus,
        time,
        account,
        qos,
        constraint,
        preload,
        dependency,
        resume_stage_id,
        resume_checkpoint,
        env,
):
    # by default use 1 node
    if nodes is None and gpus is None:
        print(f"no --nodes and no --gpus defined -> use 1 node")
        nodes = 1
    # load template submit script
    if nodes is not None:
        with open("sbatch_template_nodes.sh") as f:
            template = f.read()
    elif gpus is not None:
        with open("sbatch_template_gpus.sh") as f:
            template = f.read()
    else:
        raise NotImplementedError

    # load config
    config = kc.DefaultResolver().resolve(kc.from_file_uri("sbatch_config.yaml"))
    # check paths exist
    chdir = Path(config["chdir"]).expanduser()
    assert chdir.exists(), f"chdir {chdir} doesn't exist"
    # default account
    account = account or config.get("default_account", None)
    # not every server has qos and qos doesnt need to be defined via CLI args
    qos = qos or config.get("default_qos")

    # get sbatch-only arguments
    parser = get_parser()
    args_to_filter = []
    # noinspection PyProtectedMember
    for action in parser._actions:
        if action.dest == "help":
            continue
        # currently only supports to filter out args with -- prefix
        assert len(action.option_strings) == 1
        assert action.option_strings[0].startswith("--")
        args_to_filter.append(action.option_strings[0])
    # filter out sbatch-only arguments
    train_args = []
    i = 0
    while i < len(sys.argv[1:]):
        arg = sys.argv[1 + i]
        if arg.startswith("--") and arg in args_to_filter:
            i += 2
        else:
            train_args.append(arg)
            i += 1
    cli_args_str = " ".join(map(shlex.quote, train_args))

    # patch template
    if preload is not None:
        assert "{preload}" in template
        config["preload"] = "true"
        config["preload_yaml"] = preload
        cli_args_str += " --datasets_were_preloaded"
    else:
        config["preload"] = "false"
        config["preload_yaml"] = "nothing"
    if resume_stage_id is not None:
        cli_args_str += f" --resume_stage_id {resume_stage_id}"
    if resume_checkpoint is not None:
        cli_args_str += f" --resume_checkpoint {resume_checkpoint}"
    if env is not None:
        env_name = config.pop("env_name")
        print(f"replacing env_name: '{env_name}' from sbatch_config with '{env}'")
        config["env_name"] = env
    patched_template = template.format(
        time=time,
        nodes=nodes,
        gpus=gpus,
        account=account,
        qos=qos,
        constraint=constraint,
        cli_args=cli_args_str,
        **config,
    )
    # wait for job to finish before start
    if dependency is not None:
        lines = patched_template.split("\n")
        lines.insert(4, f"#SBATCH --dependency=afterok:{dependency}")
        patched_template = "\n".join(lines)
    print(patched_template)

    # create a shell script
    out_path = Path("submit")
    out_path.mkdir(exist_ok=True)
    fname = f"{datetime.now():%m.%d-%H.%M.%S}-{uuid.uuid4()}.sh"
    with open(out_path / fname, "w") as f:
        f.write(patched_template)

    # execute the shell script
    if os.name != "nt":
        os.system(f"sbatch submit/{fname}")


if __name__ == "__main__":
    main(**vars(get_parser().parse_known_args()[0]))
