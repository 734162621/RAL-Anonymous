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


from pysc2.lib import features, units, buffs


def get_unit_in_unit_dict(tag, unit_dict):
  for unit_tag in unit_dict.keys():
    if unit_tag == tag:
      return unit_dict[unit_tag]


def get_event(agent):
  # team in agent.teams, usually, one team only have one obs
  event_dict = {}
  unit_dict = {}

  for team in agent.teams:

    if len(team['obs']) == len(team['obs_last']):
      team_name = team['name']
      event_dict[team_name] = {}

      for i in range(len(team['obs'])):
        obs1 = team['obs_last'][i]
        obs2 = team['obs'][i]
        unit_ctrl_state_dict1 = {}
        unit_ally_state_dict1 = {}
        unit_enemy_state_dict1 = {}
        unit_ctrl_state_dict2 = {}
        unit_ally_state_dict2 = {}
        unit_enemy_state_dict2 = {}
        unit_ctrl_state_dict3 = {}
        unit_ally_state_dict3 = {}
        unit_enemy_state_dict3 = {}
        team_event_dict = {'ctrl': {}, 'ally': {}, 'enemy': {}}

        for unit in obs1.observation.raw_units:
          # controlled/ally/enemy units
          if unit.is_on_screen and unit.alliance in [1] and unit.tag in agent.unit_tag_list_history:
            unit_ctrl_state_dict1[unit.tag] = unit
          if unit.is_on_screen and unit.alliance in [1, 2] and unit.tag not in agent.unit_tag_list_history:
            unit_ally_state_dict1[unit.tag] = unit
          if unit.is_on_screen and unit.alliance in [4]:
            unit_enemy_state_dict1[unit.tag] = unit
            
        for unit in obs2.observation.raw_units:
          # controlled/ally/enemy units
          if unit.alliance in [1] and unit.tag in agent.unit_tag_list_history:
            unit_ctrl_state_dict2[unit.tag] = unit
          if unit.alliance in [1, 2] and unit.tag not in agent.unit_tag_list_history:
            unit_ally_state_dict2[unit.tag] = unit
          if unit.alliance in [4]:
            unit_enemy_state_dict2[unit.tag] = unit

        for unit in obs2.observation.raw_units:
          # controlled/ally/enemy units
          if unit.is_on_screen and unit.alliance in [1] and unit.tag in agent.unit_tag_list_history:
            unit_ctrl_state_dict3[unit.tag] = unit
          if unit.is_on_screen and unit.alliance in [1, 2] and unit.tag not in agent.unit_tag_list_history:
            unit_ally_state_dict3[unit.tag] = unit
          if unit.is_on_screen and unit.alliance in [4]:
            unit_enemy_state_dict3[unit.tag] = unit

        for tag in unit_ctrl_state_dict1.keys():
          unit1 = unit_ctrl_state_dict1[tag]
          unit_info = f"{hex(tag)}({str(units.get_unit_type(unit1.unit_type))})"
          if unit1.tag in unit_ctrl_state_dict2.keys():
            unit2 = get_unit_in_unit_dict(unit1.tag, unit_ctrl_state_dict2)
            delta_health = (unit2.health + unit2.shield) - (unit1.health + unit1.shield)
            if delta_health > 0:
              team_event_dict['ctrl'][unit1.tag] = f'unit {unit_info} is healing, health +{abs(delta_health)}'
            if delta_health < 0:
              team_event_dict['ctrl'][unit1.tag] = f'unit {unit_info} is attacked, health -{abs(delta_health)}'
            unit_ctrl_state_dict3[unit1.tag] = None
          if unit1.tag not in unit_ctrl_state_dict2.keys():
            team_event_dict['ctrl'][unit1.tag] = f'unit {unit_info} dead'

        for tag in unit_ally_state_dict1.keys():
          unit1 = unit_ally_state_dict1[tag]
          unit_info = f"{hex(tag)}({str(units.get_unit_type(unit1.unit_type))})"
          if unit1.tag in unit_ally_state_dict2.keys():
            unit2 = get_unit_in_unit_dict(unit1.tag, unit_ally_state_dict2)
            delta_health = (unit2.health + unit2.shield) - (unit1.health + unit1.shield)
            if delta_health > 0:
              team_event_dict['ally'][unit1.tag] = f'unit {unit_info} is healing, health +{abs(delta_health)}'
            if delta_health < 0:
              team_event_dict['ally'][unit1.tag] = f'unit {unit_info} is attacked, health -{abs(delta_health)}'
            unit_ally_state_dict3[unit1.tag] = None
          if unit1.tag not in unit_ally_state_dict2.keys():
            team_event_dict['ally'][unit1.tag] = f'unit {unit_info} dead'

        for tag in unit_enemy_state_dict1.keys():
          unit1 = unit_enemy_state_dict1[tag]
          unit_info = f"{hex(tag)}({str(units.get_unit_type(unit1.unit_type))})"
          if unit1.tag in unit_enemy_state_dict2.keys():
            unit2 = get_unit_in_unit_dict(unit1.tag, unit_enemy_state_dict2)
            delta_health = (unit2.health + unit2.shield) - (unit1.health + unit1.shield)
            if delta_health > 0:
              team_event_dict['enemy'][unit1.tag] = f'unit {unit_info} is healing, health +{abs(delta_health)}'
            if delta_health < 0:
              team_event_dict['enemy'][unit1.tag] = f'unit {unit_info} is attacked, health -{abs(delta_health)}'
            unit_enemy_state_dict3[unit1.tag] = None
          if unit1.tag not in unit_enemy_state_dict2.keys():
            team_event_dict['enemy'][unit1.tag] = f'unit {unit_info} dead'
            
        event_dict[team_name][i] = team_event_dict

        for tag in unit_ctrl_state_dict3.keys():
          unit3 = unit_ctrl_state_dict3[tag]
          if unit3 is not None:
            unit_info = f"{hex(tag)}({str(units.get_unit_type(unit3.unit_type))})"
            event_dict[team_name][i]['ctrl'][tag] = f'unit {unit_info} joint the team {team_name}'
        for tag in unit_ally_state_dict3.keys():
          unit3 = unit_ally_state_dict3[tag]
          if unit3 is not None:
            unit_info = f"{hex(tag)}({str(units.get_unit_type(unit3.unit_type))})"
            event_dict[team_name][i]['ally'][tag] = f'unit {unit_info} ally unit enter sight'
        for tag in unit_enemy_state_dict3.keys():
          unit3 = unit_enemy_state_dict3[tag]
          if unit3 is not None:
            unit_info = f"{hex(tag)}({str(units.get_unit_type(unit3.unit_type))})"
            event_dict[team_name][i]['enemy'][tag] = f'unit {unit_info} enemy unit enter sight'

  return event_dict