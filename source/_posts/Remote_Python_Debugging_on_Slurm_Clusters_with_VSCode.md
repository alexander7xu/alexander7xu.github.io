---
title: Remote Python Debugging on Slurm Clusters with VSCode
mathjax: false
date: 2026-04-06 00:10:42
categories: Coding
description: A comprehensive tutorial on remote debugging Python scripts in HPC environments using VSCode, debugpy, and Slurm.
---

## UV

[Install UV](https://docs.astral.sh/uv/#standalone-installer)

```bash
export UV_INSTALL_DIR="$HOME/CISPA-home/.uv"
curl -LsSf https://astral.sh/uv/install.sh | sh
touch ~/CISPA-home/.uv/uv.toml
```

Best practice:
- Before running `uv init` or `uv venv`, create a soft link of venv to the same storage of uv cache by `mkdir -p $PATH_TO_LARGE_STORAGE/$PROJECT_NAME/.venv && ln -s $PATH_TO_LARGE_STORAGE/$PROJECT_NAME/.venv ./`. This ensures that hard-link of uv works properly.
- If the `.venv` folder of current project has to been placed in a storage different from uv cache, use soft-link in `uv.toml` by `echo 'link-mode = "symlink"' >> ~/CISPA-home/.uv/uv.toml`.

## Environment Variables

VSCode does not automatically pass the current environment variables to a task, meaning we must set them manually. Furthermore, since `~/.bashrc` is only accessible on the login node, we need to store our environment variables in a dedicated configuration file located in a shared directory accessible by all nodes.

File `~/CISPA-home/.envrc`:
```bash
# local variables
PATH_TO_LARGE_STORAGE="$HOME/CISPA-az6/__TBD__"

# LLM packages
export HF_HOME="$PATH_TO_LARGE_STORAGE/.large/huggingface"
export TORCH_HOME="$PATH_TO_LARGE_STORAGE/.large/torch"

# uv
. "$HOME/CISPA-home/.uv/env"
export UV_CACHE_DIR="$PATH_TO_LARGE_STORAGE/.uv/cache"
export UV_CONFIG_FILE="$HOME/CISPA-home/.uv/uv.toml"
export UV_PYTHON_INSTALL_DIR="$PATH_TO_LARGE_STORAGE/.uv/python"
export UV_TOOL_DIR="$HOME/CISPA-home/.uv/tool"
```

Add the lines below at the end of file `~/.profile`

```bash
# load envrc
if [ -f "$HOME/CISPA-home/.envrc" ] ; then
    . "$HOME/CISPA-home/.envrc"
fi
```

If you are using VSCode, kill the server and reload the window:
- `Ctrl+Shift+P`, `>Remote-SSH: Kill Current VS Code Server `
- `Ctrl+Shift+P`, `>Developer: Reload Window`

## Utility Script

Note that VSCode will invasively execute a `source` command in every new task terminal when using a Python `venv`. This behavior can interfere with the first `input()` calling in the script being debugged. To prevent this, we use `read -r -t 0.1` to clear the `stdin` before starting `debugpy`.

Since the debugging task must run in the background, we need to output specific string patterns to indicate the current status of the debugging node to VSCode:
- **Start:** `>>>>>>>> MYDEBUGPY HELLO hostname:port`
- **Ready:** `>>>>>>>> DEBUGGING [script, args...]`
- **Finish:** `>>>>>>>> MYDEBUGPY BYEBYE code`

File `~/CISPA-home/mydebugpy`:
```bash
#!/bin/bash
set -e

. $HOME/CISPA-home/.envrc
[ -e ".venv/bin/activate" ] && source .venv/bin/activate
[ -e ".env" ] && set -a && . .env && set +a

LISTENING_PORT=$1
CMD=$(printf "'%s', " "${@:2}")

echo ">>>>>>>> MYDEBUGPY HELLO $(hostname):$LISTENING_PORT"
read -r -t 0.1 || true  # clear stdin

uv run python -u -c "
import debugpy ;\
import runpy ;\
import sys ;\
debugpy.listen(('0.0.0.0', $LISTENING_PORT)) ;\
sys.argv=[$CMD] ;\
print('>>>>>>>> DEBUGGING', sys.argv) ;\
print('', flush=True) ;\
debugpy.wait_for_client() ;\
runpy.run_path(sys.argv[0], run_name='__main__') ;\
"
RETV=$?

echo ""
echo ">>>>>>>> MYDEBUGPY BYEBYE $RETV"
```

## VSCode Workspace

### Node Configure

Define the configuration parameters for the target compute node in `settings.json`. This allows the variables to be shared seamlessly between `tasks.json` and `launch.json`.

File `.vscode/settings.json`:
```json
{
    "myDebugpy.host": "xe8545-a100-23",
    "myDebugpy.port": 19810,
    "myDebugpy.cpus": 16,
    "myDebugpy.gpus": 1,
}
```

### Debugging Task

In this task configuration, we request cluster resources via `srun` and launch the target Python script on the allocated compute node.

File `.vscode/tasks.json`:
```json
{
    // See https://go.microsoft.com/fwlink/?LinkId=733558
    // for the documentation about the tasks.json format
    "version": "2.0.0",
    "tasks": [
        {
            "label": "debug node",
            "isBackground": true,
            "problemMatcher": {
                "owner": "custom",
                "pattern": {
                    "regexp": "^$"
                },
                "background": {
                    "activeOnStart": true,
                    "beginsPattern": "^>>>>>>>> MYDEBUGPY HELLO .+$",
                    "endsPattern": "^>>>>>>>> DEBUGGING .+$"
                }
            },
            "type": "process",
            "command": "srun",
            "options": {
                "cwd": "${workspaceFolder}",
                "env": {
                    "PYTHONPATH": "${workspaceFolder}/src",
                }
            },
            "args": [
                "--time=08:00:00",
                "--unbuffered",
                "--partition=debug,xe8545,gpu",
                "--nodelist=${config:myDebugpy.host}",
                "--cpus-per-task=${config:myDebugpy.cpus}",
                "--gpus-per-node=${config:myDebugpy.gpus}",
                "${userHome}/CISPA-home/Utils/mydebugpy", "${config:myDebugpy.port}",
                // write the script to debug below:
                "./main.py",
                "args1",
                "args2",
                "-k", "value",
                "--key", "value",
            ]
        }
    ]
}
```

### Launch Debugger


Finally, attach the VSCode debugger to the remote `debugpy` session running on the allocated compute node.

File `.vscode/launch.json`:
```json
{
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "sbatch debug",
            "type": "debugpy",
            "request": "attach",
            "connect": {
                "host": "${config:myDebugpy.host}",
                "port": "${config:myDebugpy.port}",
            },
            "preLaunchTask": "debug node",
            "pathMappings": [
                {
                    "localRoot": "${workspaceFolder}",
                    "remoteRoot": "."
                }
            ]
        }
    ]
}
```
