import os
from pathlib import Path


def main():
    # clean_logs should be started as `python clean.logs.py` not `python logs/clean_logs.py` or something like this
    assert Path(os.getcwd()) == Path(__file__).parent

    # find all log files
    uris = []
    for root, _, files in os.walk("."):
        for name in files:
            if name.endswith(".log"):
                uris.append(Path(root) / name)
    print(f"found {len(uris)} log files")
    for uri in uris:
        print(f"- {uri.as_posix()}")

    # cleanup everything that might even remotely be a security concern
    for uri in uris:
        # read all lines
        with open(uri, "r") as file:
            lines = file.readlines()

        # remove hostnames for multi-node
        for i in range(len(lines)):
            if "hostnames: " in lines[i]:
                while True:
                    if "-----" in lines[i + 1]:
                        break
                    lines.pop(i + 1)
                break
        # clean lines
        for i in range(len(lines)):
            line = lines[i]
            for mode, pattern, replacement in [
                ("replace_after", "executable: ", ".../bin/python"),
                ("replace_after", "host name: ", "..."),
                ("replace_after", "OS: ", "..."),
                ("replace_after", "OS version: ", "..."),
                ("replace_after", "hostname=", "..."),
                ("replace_after", "account_name: ", "..."),
                ("replace_after", "output_path: ", ".../save"),
                ("replace_after", "local_dataset_path: ", "/tmp"),
                ("replace_after", "tmpfs", "..."),
                ("replace_after", "slurm job id: ", "..."),
                ("replace_after", "hostnames: ", "..."),
                ("replace_after", "copied unresolved hp to ", ".../hp_unresolved.yaml"),
                ("replace_after", "dumped resolved hp to ", ".../hp_resolved.yaml"),
                ("replace_after", "extracting 1000 zips from ", "... to ..."),
                ("replace_after", "loaded model kwargs from ", lambda x: ".../" + Path(*Path(x).parts[-4:]).as_posix()[:-1]),
                ("replace_after", "loaded weights of vislstm from ", lambda x: ".../" + Path(*Path(x).parts[-4:]).as_posix()[:-1]),
                ("replace_full", "UserWarning: Grad strides", lambda x: "..." + x[x.index("/site-packages/"):-1]),
                ("replace_after", "saved vislstm to ", lambda x: ".../" + Path(*Path(x).parts[-4:]).as_posix()[:-1]),
                ("replace_after", "saved vislstm optim to ", lambda x: ".../" + Path(*Path(x).parts[-4:]).as_posix()[:-1]),
                ("replace_after", "loaded optimizer of vislstm from ", lambda x: ".../" + Path(*Path(x).parts[-4:]).as_posix()[:-1]),
                ("replace_after", "saved trainer state_dict to ", lambda x: ".../" + Path(*Path(x).parts[-4:]).as_posix()[:-1]),
                ("replace_after", "loaded trainer checkpoint ", lambda x: ".../" + Path(*Path(x).parts[-4:]).as_posix()[:-1]),
                ("replace_after", "log entries to ", lambda x: ".../" + Path(*Path(x).parts[-4:]).as_posix()[:-1]),
                ("replace_after", "function: 'forward' ", ".../site-packages/kappamodules/layers/drop_path.py:40)"),
            ]:
                if pattern in line:
                    if mode == "replace_after":
                        split_idx = line.index(pattern) + len(pattern)
                        if not isinstance(replacement, str):
                            to_remove = line[split_idx:]
                            replacement = replacement(to_remove)
                        line = line[:split_idx] + replacement
                    elif mode == "replace_full":
                        if isinstance(replacement, str):
                            line = replacement
                        else:
                            line = replacement(line)
                    else:
                        raise NotImplementedError

            # replace line if something changed
            if lines[i] != line:
                lines[i] = f"{line}\n"

        # write back to logfile
        with open(uri, "w") as file:
            file.writelines(lines)


if __name__ == "__main__":
    main()
