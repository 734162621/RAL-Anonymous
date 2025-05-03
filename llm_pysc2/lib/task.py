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


import numpy as np
import math


def get_info(obs):
  # for time condition
  game_loop = obs.observation.game_loop
  game_s = int(game_loop / 22 % 60)  # SC2 runs at 22.4 game loops per second
  game_m = int(game_loop / 22 // 60)  # SC2 runs at 22.4 game loops per second
  # for position condition
  idx = np.nonzero(obs.observation['feature_minimap']['camera'])
  minimap_x, minimap_y = int(idx[:][1].mean()), int(idx[:][0].mean())
  return minimap_x, minimap_y, game_m, game_s

def get_dist(x1, y1, x2, y2):
  dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
  return dist


def task_harass(agent):
  task_dict = {}
  for team in agent.config.AGENTS[agent.name]['team']:
    if len(team['obs']) == 0:
      continue
    x, y, m, s = get_info(team['obs'][0])
    if team['name'] in ['Oracle-1', 'Adept-1', 'Phoenix-1']:
      d1 = 4 if team['name'] == 'Adept-1' else 6  # flying unit +2
      d2 = 10

      # define tasks
      task1 = "Go to minimap coordinate [52, 32] as quick as possible."
      task2 = "Kill as much as enemy **Drones** as possible until all units dead."
      task3 = "Go back to minimap coordinate [52, 32], continuing killing **Drones** until all units dead."
      # task0 -> task1, or continue task1
      if (team['task'] == '' and get_dist(x, y, 52, 32) > d1) or \
          (team['task'] == task1 and get_dist(x, y, 52, 32) > d1):
        team['task'] = task1
      # task1 -> task2, task3 -> task2, or continue task2
      if (team['task'] == task1 and get_dist(x, y, 52, 32) <= d1) or \
          (team['task'] == task3 and get_dist(x, y, 52, 32) <= d2) or \
          (team['task'] == task2 and get_dist(x, y, 52, 32) <= d2):
        team['task'] = task2
      # task2 -> task3, task3 -> task2, or continue task3
      if (team['task'] == task3 and get_dist(x, y, 52, 32) > d2) or \
          (team['task'] == task2 and get_dist(x, y, 52, 32) > d2):
        team['task'] = task3

    if team['name'] == 'AdeptPhase-1':
      # define tasks
      task1 = "Assist the team Adept-1 to sneak into enemy territory, reposition during combat, or retreat."
      team['task'] = task1  # always task1

    task_dict[team['name']] = team['task']

  return task_dict


def task_smac(agent):
  task_dict = {}
  for team in agent.config.AGENTS[agent.name]['team']:
    if len(team['obs']) == 0:
      continue
    x, y, m, s = get_info(team['obs'][0])

    if s < 1:
      team['task'] = "Hold position and attack incoming enemies."
      task_dict[team['name']] = team['task']
    else:
      team['task'] = "Kill as much as enemy units as possible and avoid losing unit."
      task_dict[team['name']] = team['task']

  return task_dict



# search task function by map name, obs.observation.map_name
FACTORY = {
  'pvz_task1_level1': task_harass,
  'pvz_task1_level2': task_harass,
  'pvz_task1_level3': task_harass,
  'pvz_task2_level1': task_harass,
  'pvz_task2_level2': task_harass,
  'pvz_task2_level3': task_harass,

  '3s_vs_3z': task_smac,
  '3s_vs_4z': task_smac,
  '3s_vs_5z': task_smac,
  '2s3z': task_smac,
  '3s5z': task_smac,
}

