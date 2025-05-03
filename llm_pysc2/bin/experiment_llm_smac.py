
# Copyright 2024, LLM-PySC2 Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from llm_pysc2.cfg.llm_smac import *
from llm_pysc2.agents import MainAgent, LLMAgent, RAG_LLMAgent, RAG2_LLMAgent, RAG3_LLMAgent, Reflect_LLMAgent
import os


def get_config(map_name):
  llm_smac_configs = {
    '1c3s5z': ConfigSmac_1c3s5z(),
    '2c_vs_64zg': ConfigSmac_2c(),
    '2s3z': ConfigSmac_2s3z(),
    '2s_vs_1sc': ConfigSmac_2s(),
    '3s5z': ConfigSmac_3s5z(),
    '3s5z_vs_3s6z': ConfigSmac_3s5z(),
    '3s_vs_3z': ConfigSmac_3s(),
    '3s_vs_4z': ConfigSmac_3s(),
    '3s_vs_5z': ConfigSmac_3s(),
  }
  if map_name in llm_smac_configs.keys():
    config = llm_smac_configs[map_name]
    # config.LLM_SIMULATION_TIME = 10
    config.MAX_LLM_DECISION_FREQUENCY = 2
    config.MAX_NUM_ACTIONS = 5
    for team in config.AGENTS['CombatGroupSmac']['team']:
      team['task'] = [
        {'time': None, 'pos': None, 'info': "Try to kill all enemy units before all controlled units dead."},
      ]
    return config
  else:
    raise AssertionError(f"wrong map_name: {map_name}")


map_name = '3s_vs_3z'
enable_image_rgb = False
enable_image_feature = False

class MainAgentLLMSmac(MainAgent):
  def __init__(self):
    config = get_config(map_name)
    model_name = 'gpt-3.5-turbo'
    api_base = 'https://api.xty.app/v1'
    api_key = 'sk-D1cNc5ege4oD3zpP55E0105a7cAb41D392711832CbB17669'
    # model_name = 'gpt-4o-mini'
    # api_base = 'https://api.xty.app/v1'
    # api_key = 'sk-NZYM3FNSWv3hEAFZl1XyW5jVR4FQR4ghFMKRMwd2EVAKampw'
    # model_name = 'deepseek-v3'
    # api_base = 'https://hk.xty.app/v1'
    # api_key = 'sk-APfKIMKJ3NKO9JTKQG5ZaNAABqpyDTK1ZV5wXd4quWfVdMkX'
    # model_name = 'deepseek-r1-250120'
    # api_base = 'https://ark.cn-beijing.volces.com/api/v3'
    # api_key = 'f4a09859-1ea1-412a-a197-a33530fa4617'
    # config.MAX_LLM_WAITING_TIME = 45
    # config.MAX_LLM_RUNTIME_ERROR_TIME = 120
    config.reset_llm(model_name, api_base, api_key, enable_image_rgb, enable_image_feature)
    super(MainAgentLLMSmac, self).__init__(config, RAG3_LLMAgent)  # LLMAgent Reflect_

  def step(self, obs):
    return super().step(obs)


if __name__ == "__main__":

  if not (enable_image_rgb or enable_image_feature):
    os.system(f"python -m pysc2.bin.agent --map {map_name} --agent_race protoss --parallel 5 "
              f"--agent llm_pysc2.bin.experiment_llm_smac.MainAgentLLMSmac")
  elif enable_image_rgb:
    os.system(f"python -m pysc2.bin.agent --map {map_name} --agent_race protoss --parallel 1 "
              f"--agent llm_pysc2.bin.experiment_llm_smac.MainAgentLLMSmac "
              f"--feature_screen_size 256 --feature_minimap_size 64 "
              f"--rgb_screen_size 256 --rgb_minimap_size 64 "
              f"--action_space RGB")
  elif enable_image_feature:  # parallel experiments with feature map obs do not available currently, set --parallel 1
    os.system(f"python -m pysc2.bin.agent --map {map_name} --agent_race protoss --parallel 1 "
              f"--agent llm_pysc2.bin.experiment_llm_smac.MainAgentLLMSmac "
              f"--feature_screen_size 256 --feature_minimap_size 64 "
              f"--rgb_screen_size 0 --rgb_minimap_size 0 "
              f"--render")
  else:
    print("Can not enable_image_rgb and enable_image_feature at the same time'''")
