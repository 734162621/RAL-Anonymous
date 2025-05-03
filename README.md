


# RAL - Retrieval Augmented Learning



Submission to NeurIPS 2025, Submission Number: 5179 (Retrieval Augmented Learning: A Retrial-based Large Language Model Self-Supervised Learning and Autonomous Knowledge Generation)



## Quick Start Guide

### Get Dify for RAL

Download source code in https://github.com/langgenius/dify，suggest version v0.11.2.

Modify following code to release the token limitation：

（1）The first limitation in dify compose file line 249 INDEXING_MAX_SEGMENTATION_TOKENS_LENGTH. 

（2）Thee second limitation in dify source code dify.api.services.hit_testing_service.py line 165

Build up a container though docker compose file, suggest docker-desktop of version 4.35.1 (173168) in Windows10/11 system.

Then set your embedding model api (suggest GLM embedding-3) in local dify container http://127.0.0.1/apps and create empty databases and empty document.

### Test Dify

get the dify api_url in local dify container http://127.0.0.1/apps, write to  the script in llm_pysc2/lib/rag_dify.py

test Dify by running the script in llm_pysc2/lib/rag_dify.py

test each functions, especially:

```
def dify_segment_qa_create_by_text(api_base, api_key, dataset_id, **kwargs)
def dify_segment_qa_update_by_text(api_base, api_key, dataset_id, **kwargs)
def dify_retrieve(api_base, api_key, dataset_id, **kwargs)
```



### Get StarCraft II

LLM-PySC2 depends on the full StarCraft II game and only works with versions that
include the API, which is 3.16.1 and above.

#### Windows

Install of the game as normal from [Battle.net](https://battle.net). Even the
[Starter Edition](http://battle.net/sc2/en/legacy-of-the-void/) will work.
If you used the default install location LLM-PySC2 should find the latest binary.
If you changed the install location, you might need to set the `SC2PATH`
environment variable with the correct location.

LLM-PySC2 should work on MacOS and Windows systems running Python 3.8+,
but has only been thoroughly tested on Linux. We welcome suggestions and patches
for better compatibility with other systems.

### Get LLM-PySC2

download the LLM-PySC2 code from our github page [LLM-PySC2](https://github.com/NKAI-Decision-Team/LLM-PySC2).

use pip install to initialize the environment:

```shell
$ conda create --name YOUR_ENV_NAME python==3.9
$ conda activate YOUR_ENV_NAME
$ pip install -e .
```

you can use mirrors like `pip install -e . -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple` to speed up downloading.

### Get the maps

We have placed the required maps in the project folder:

```
llm_pysc2/maps/llm_pysc2
llm_pysc2/maps/llm_smac
```

You need to copy and paste these `2 folders` into the Maps folder of the StarCraft2 program. Generally, the folder path is:

```
C:\Program Files (x86)\StarCraft II\Maps
```

and finally looks like:

```
C:\Program Files (x86)\StarCraft II\Maps\llm_pysc2
C:\Program Files (x86)\StarCraft II\Maps\llm_smac
```

If you used a custom path in installation, you may need to find the Map folder to finish the step.


### Get llm api key

If you do not know how to get api_key, you can contact us to obtain a temporary gpt-3.5-turbo api_key with 2M tokens for free.

You need to write your api_key in `./llm_pysc2/agents/configs/config.ProtossAgentConfig` before test the llm:

    class ProtossAgentConfig(AgentConfig):
        def __init__(self):
            super(ProtossAgentConfig, self).__init__()
            self.race = 'protoss'
            self.model_name = 'gpt-3.5-turbo'
            self.api_base = 'YOUR_API_BASE'
            self.api_key = 'YOUR_API_KEY'
            ...

or set api_key like what we do in `./llm_pysc2/bin/experiment_llm_pysc2.py`:

    config.reset_llm(model_name, api_base, api_key)

if you do not have api_key but still want to test the environment, 
you can set `config.LLM_SIMULATION_TIME = 5` to simulate a 5-second response large model 
and continue the tutorial below.


### Test the environment

After specify your LLM api_key, api_base and model_name, you can run our experiments to test LLM and
both the llm_pysc2 tasks and llm-smac tasks:

```shell
$ python -m llm_pysc2.bin.experiment_llm_pysc2
$ python -m llm_pysc2.bin.experiment_llm_smac
```

These two script will load gpt-3.5 energized agents and use pure text observation to make decisions.
If you want to use multimodal LLMs like gpt-4v, you can set `config.ENABLE_IMAGE_RGB = True` to 
activate image observations.

Also, you can use --parallel parameter (or edit files in ./llm_pysc2/bin) to run several games at the same time:

```shell
$ python -m pysc2.bin.agent --map pvz_task4_level1 --agent_race protoss --parallel 2 --agent llm_pysc2.bin.experiment_llm_pysc2.MainAgentLLMPysc2
$ python -m pysc2.bin.agent --map pvz_task4_level1 --agent_race protoss --parallel 4 --agent llm_pysc2.bin.experiment_llm_pysc2.MainAgentLLMPysc2
```

which may significantly improve experimental efficiency.



## Experiments

run llm_pysc2/agents/bin/experiment_ral.py to test 3s_vs_nz learning process.

more script will be added in the future.
