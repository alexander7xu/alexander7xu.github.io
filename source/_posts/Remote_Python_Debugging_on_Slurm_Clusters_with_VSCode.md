---
title: Remote Python Debugging on Slurm Clusters with VSCode
mathjax: false
date: 2026-04-06 00:10:42
categories: Coding
description: A comprehensive tutorial on remote debugging Python scripts in HPC environments using VSCode, debugpy, and Slurm.
---

## UV

First of all, create symbolic links, so that some files could be accessed on compute nodes.

```bash
mv ~/.local ~/CISPA-home/.local/ && ln -s ~/CISPA-home/.local/ ~/
mv ~/.config ~/CISPA-home/.config/ && ln -s ~/CISPA-home/.config/ ~/
```

Use symbolink. File `~/.config/uv/uv.toml`

```toml
link-mode = "symlink"
```

## Environment Variables

VSCode does not automatically pass the current environment variables to a task, meaning we must set them manually. Furthermore, since `~/.bashrc` is only accessible on the login node, we need to store our environment variables in a dedicated configuration file located in a shared directory accessible by all nodes.

File `~/CISPA-home/.config/envrc`
```bash
# local bin
export PATH="$HOME/CISPA-home/.local/bin/:$PATH"

# uv
export UV_CACHE_DIR="$HOME/CISPA-home/.local/share/uv/cache"
export UV_PYTHON_INSTALL_DIR="$HOME/CISPA-home/.local/share/uv/python"
export UV_TOOL_DIR="$HOME/CISPA-home/.local/share/uv/tool"
```

Add this line at the end of file `~/.bashrc`

```bash
# load envrc
ENVRC_PATH=$HOME/CISPA-home/.config/envrc
[ -e "$ENVRC_PATH" ] && source $ENVRC_PATH
```

## Utility Script

Note that VSCode will invasively execute a `source` command in every new task terminal when using a Python `venv`. This behavior can interfere with the first `input()` calling in the script being debugged. To prevent this, we use `read -r -t 0.1` to clear the `stdin` before starting `debugpy`.

Since the debugging task must run in the background, we need to output specific string patterns to indicate the current status of the debugging node to VSCode:
- **Start:** `>>>>>>>> DEBUGPY HELLO hostname:port`
- **Ready:** `>>>>>>>> DEBUGGING [script, args...]`
- **Finish:** `>>>>>>>> DEBUGPY BYEBYE code`

File `~/CISPA-home/debugpy`
```bash
#!/bin/bash
source $HOME/CISPA-home/.config/envrc

LISTENING_PORT=$1
CMD=$(printf "'%s', " "${@:2}")

echo ">>>>>>>> DEBUGPY HELLO $(hostname):$LISTENING_PORT"
read -r -t 0.1    # clear stdin

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
echo ">>>>>>>> DEBUGPY BYEBYE $RETV"
```

## VSCode Workspace

### Node Configure

Define the configuration parameters for the target compute node in `settings.json`. This allows the variables to be shared seamlessly between `tasks.json` and `launch.json`.

File `.vscode/settings.json`
```json
{
    "myDebug.host": "xe8545-a100-23",
    "myDebug.port": 19810,
    "myDebug.cpus": 16,
    "myDebug.gpus": 1,
}
```

### Debugging Task

In this task configuration, we request cluster resources via `srun` and launch the target Python script on the allocated compute node.

File `.vscode/tasks.json`
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
                    "beginsPattern": "^>>>>>>>> DEBUGPY HELLO .+$",
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
                "--nodelist=${config:myDebug.host}",
                "--cpus-per-task=${config:myDebug.cpus}",
                "--gpus-per-node=${config:myDebug.gpus}",
                "${userHome}/CISPA-home/Utils/debugpy", "${config:myDebug.port}",
                // write the script to debug below:
                "${workspaceFolder}/main.py",
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

File `.vscode/launch.json`
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
                "host": "${config:myDebug.host}",
                "port": "${config:myDebug.port}",
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
