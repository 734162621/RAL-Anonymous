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


BASIC_COMBAT_RULES = \
"""
  1. Try to kill more and loss less. Always concentrating firepower on the most vulnerable enemy to quickly kill enemy units.
  2. Try to kill enemy as quick as possible, retreat promptly when/before enemy reinforcements arrive.
  3. When sacrificing your unit can earn much more profits, you can choose to sacrifice your unit.
  4. Use your skills well to achieve optimal tactical results. Especially when controlling support units. Be aware that some skills may have side effects or long cooldowns, use them with caution.
  5. Always remember the tactical tasks given by superior. Sometimes you have to sacrifice whole team to ensure the achievement of tactical objectives.
  6. Try to handle Handle micro operations well. Move during weapon cooling, attack while weapon ready, retreat heavily damaged unit and complete a kill as soon as possible.
  7. Attack one unit at a single time, always. Do not attack multi-unit at the same time or add multi attack actions into the queue at the same time.
  8. Suggest to sequence actions as "skills/abilities -> attack -> move".
  9. Every coordinate mentioned in analysis should be clearly marked whether it is screen coordinates or minimap coordinates.
  10. Using <Select_Unit_Move_Screen> to repositioning low health units to safer location while still ensuring attacking the nearest enemy. At the same time, use healther units to engage in forward combat and bear damage for those vulnerable units.
  11. When you use **Move** action, please ensure that the moving path is **valid(within the allowed range)**, **long enough** and **safe** for the next several seconds.
  12. Usually, the camera will focus you units that screen [12, 12] is the position of your units. If you need to relocate your units, move your units to a position away from [12, 12].
"""

#   11. When you use **Move** action, please ensure that the moving path is **long enough** and **safe** for the next several seconds.

BASIC_COMMAND_RULES = \
"""
"""

BASIC_DEVELOP_RULES = \
"""
"""

BASIC_BUILD_RULES = \
"""
"""

# BASIC_COMBAT_RULES_REFLECTION = \
# """
#   1. Whether each action of a_t1 are in a legal form that shown in the 'Valid Actions Part' of s_t1?
#   2. Whether each action of a_t1 is queued in correct sequence?
#   3. Whether the args of a_t1 are appropriate? for example, whether the attacked unit is the most important target?
#   4. Whether concentrated firepower on the most vulnerable enemy? Can a single attack kill this unit? (calculate the total damage of one hit are needed)
# whether the position of moving is appropriate in the micro-operation.
# """


class BasePrompt:

  def __init__(self, name, log_id, config):
    self.name = name
    self.config = config
    self.log_id = log_id
    self.sp = ''
    self.eip = ''
    self.eop = ''
    self.screen_img_rgb_prompt = ''
    self.screen_img_fea_prompt = ''
    self.minimap_img_rgb_prompt = ''
    self.minimap_img_fea_prompt = ''


class CombatGroupPrompt(BasePrompt):

  def __init__(self, name, log_id, config):
    super(CombatGroupPrompt, self).__init__(name, log_id, config)

    # Part 1
    self.sp = \
f"""
1.Identity
  You are a {self.config.AGENTS[self.name]['describe']}.
  Your should command your troops, complete the tactical tasks assigned by the superior. You will have several teams of units, you can command these teams to fight together or perform different tasks.

2.Rules
{BASIC_COMBAT_RULES}
  
3.Action Output
  You should make decisions according to observed information, tactic task and rules, give analysis and decisions for each team. For example, if you have 2 teams name as 'TeamName-1' and 'TeamName-2', you should output as:
  
  Analysis: 
    xxxxx
    
  Strategy:
    xxxxx
    
  Actions:
    Team TeamName-1:
      <ActionName1(...)>  # format like **ActionName1(...)** and -ActionName1(...)- are not valid, must use <>
      <ActionName2(...)>
    Team TeamName-2:
      <ActionName1(...)>
      
Note that actions must in the shape <ActionName(...)>, do not generate action like 'ActionName(...)' or **ActionName(...)**.
"""
    self.eip = \
"""
Game Info
  Time: 0:32

Team Oracle-1 Info:
  Team minimap position: [50, 32]
  Controlled Team Units:
    Unit: Oracle    Tag: 0x100200001    Pos: (67, 59)    Health: 100    Energy: 108    Weapon_cooldown: 0
  Nearby Enemy units:
    Unit: Drone    Tag: 0x101340001    Pos: (54, 40)    Health: 40
    Unit: Drone    Tag: 0x101280001    Pos: (61, 58)    Health: 40
    Unit: Drone    Tag: 0x1012c0001    Pos: (52, 70)    Health: 40
    Unit: Drone    Tag: 0x101480001    Pos: (61, 71)    Health: 18
    Unit: Drone    Tag: 0x101300001    Pos: (54, 94)    Health: 40
    Unit: Hatchery    Tag: 0x101100001    Pos: (34, 67)    Health: 1500
    Unit: Queen    Tag: 0x1000c0001    Pos: (50, 40)    Health: 175    Energy: 25
    Unit: Queen    Tag: 0x100580001    Pos: (57, 54)    Health: 175    Energy: 25

Here are some description of screen units:
  Protoss.Oracle
    A light, psionic, support and harassment ship. Can grant vision and harass light units and workers with its pulsar beam.(Cannot attack ground units before activating Pulsar Beam)
    unit abilities:
      Revelation: Always available. Active skill. Cost: 25 energy. Reveals enemy units and structures in an area, granting vision for 20 seconds. Also reveals cloaked or burrowed units or structures.
      Pulsar Beam: Always available. Active skill. Cost: 25 energy (+1.96 energy per second). Enables the Oracle to attack ground units with high damage, particularly effective against light units.
      Stasis Ward: Always available. Active skill. Cost: 50 energy. Places a cloaked stasis ward on the ground that traps enemy units in stasis for 21 seconds upon activation.
  Protoss.Observer
    A cloaking air unit that functions as a detector.
  Zerg.Drone
    Harvests resources and spawns structures. Is sacrificed when creating new structures.The drone morphs into structures and harvests minerals and vespene gas.
  Zerg.Hatchery
    Spawns larvae to be morphed into other zerg strains, generates creep and digests minerals and gas into a usable form. The queen is spawned directly from the hatchery.
  Zerg.Queen
    The queen a powerful attacking ground dwelling support unit ideal for zerg defense.

Valid Actions:
  <Attack_Unit(tag)>
  <Move_Screen(screen)>
  <Move_Minimap(minimap)>
  <Ability_OracleRevelation_Screen(screen)>
  <Ability_StasisTrap_Screen(screen)>
Arg: 
  tag: refers to a hexadecimal number, shape as 0x000000000.
  screen: refers to a screen coordinate, shape as [x, y], x and y range from 0 to 128.
  minimap: refers to a minimap coordinate, shape as [x, y], x and y range from 0 to 64.
"""
    self.eop = \
"""
Analysis: 
  We are controlling a team called Oracle-1, we have met several enemy Queens, Drones and Overlord. Our goal is killing as much Drone. 
  Consider that we still have enough health and energy, We should choose drone to attack, and leave the area quickly.
  
Strategy:
  Attack-Then-Retreat Strategy: first attack the most vulnerable drone then move to a safe place.
  
Actions:
  Team Oracle-1:
    <Attack_Unit(0x101480001)>
    <Move_Screen([67, 96])>
"""

    # Part 2
    if self.config.ENABLE_COMMUNICATION:
      self.sp += \
"""
4.Communication Output
  If there is Available Communicate Target, you should keep communicating with them by Communication functions. For example, if 'Commander' and 'CombatGroup4' in Available Communicate Target, you can output as:

  Communications:
    <MessageTo(Commander, '''xxxxxxxxxx''')>
    <MessageTo(CombatGroup4, '''xxxxxxxxxx''')>
"""
      self.eip += \
"""
Communication:
  From Commander: 
    Your task is to attack the enemy workers of an enemy base near minimap [48,32]. Intelligence shows that two enemy Queens are located on the minimap [44,32]. Try to avoid being detected by enemy Queens before arriving.

Available Communication Tragets:
  Commander: Protoss military supreme commander. Responsible for making macro decision through communication, and controls nexus for massrecall for tactical objectives.
Available Communication Functions:
  <MessageTo(AgentName, message)>
  <MessageTo(ChannelName, message)>
  <ListenTo(ChannelName)>
Args explanation:
  (1)AgentName: refers to a name mentioned in Available Communication Tragets.
  (2)ChannelName: shape as Channel-i, i refers to an integer.
  (2)message: any text wrapped between ''' and '''.
"""

      self.eop += \
"""
Communications:
    <MessageTo(Commander, '''Copy that, we have arrived enemy base, and started attack enemy workers''')>
"""

    # Part 3
    self.eip += \
f"""
Give each team no more than {self.config.MAX_NUM_ACTIONS} actions.
Now, start generating your analysis, stratagy and actions:
"""



class CommanderPrompt(BasePrompt):  # TODO: Design a prompt specifically for the supreme military commander
  def __init__(self, name, log_id, config):
    super(CombatGroupPrompt, self).__init__(name, log_id, config)
    # self.sp = ''
    # self.eip = ''
    # self.eop = ''


class DeveloperPrompt(BasePrompt):  # TODO: Design a prompt specifically for the supreme logistics commander
  def __init__(self, name, log_id, config):
    super(CombatGroupPrompt, self).__init__(name, log_id, config)
    # self.sp = ''
    # self.eip = ''
    # self.eop = ''


class ReflectionPrompt(BasePrompt):

  def __init__(self, name, log_id, config):
    super(ReflectionPrompt, self).__init__(name, log_id, config)
    self.sp = \
f"""
You are a StarCraft2 game reflector. You will receive two states s_t1 and s_t2(a game state of time t1 and t2) and a_t1 (analysis and actions of StarCraft2 frontline commander used in time t1).
You should make reflection on the decision process and each action, and output in a given format.

Here are some basic combat rules you need to follow in the reflection:
{BASIC_COMBAT_RULES}

After generating the reflection, you need to generate experiences, with universality and robustness, which means
do not provide detailed data such as unit tags and specific coordinates in the generated experiences.

Generate reflections and experiences(for future use) according to the following format:

Reflections:
  1. Reflection on Stratage(RS): 
    xxxxx(consistency with task, consistency with rules, result according to state transition)
  2. Reflection on Actions(RA): 
    (1) <ActionName1(...)>: xxxxx(action validity, consistency with stratagy, the choose of action args, action result)
    (2) <ActionName2(...)>: xxxxx(action validity, consistency with stratagy, the choose of action args, action result)
    ...
  3. Reflection on Action Queue(RAQ): 
    xxxxx(result according to state transition)

Experiences:
  1. Experience on Stratagy(ES): 
    xxxxx(Stratagy xxx(name) is a xxx(good/bad) stratagy, it is recommended to adopting xxx(name) strategy to xxx(purpose))
  2. Experience on Actions(EA): 
    xxxxx(Using xxx(ActionName) at this moment is xxx(good/bad) because xxx, it is recommended to xxx)
  3. Experience on Action Queue(EAQ): 
    xxxxx(First xxx(ActionName) then xxx(ActionName) is a xxx(good/bad) action sequence, it is recommended to xxx)
    
Try to be brief, try to limit the RS, RA, RQA, ES, EA and EQA to around two sentences.
"""

    self.eip = \
"""
{"s_t1": "Team Adept-1 Info:\n\tTeam minimap position: [25, 29]\n\tControlled Team Units:\n\t\tUnit: Adept    Tag: 0x100240001    ScreenPos: [67, 60]    Health: 140(100 %)    Weapon Ready\n\t\tUnit: Adept    Tag: 0x100500001    ScreenPos: [67, 65]    Health: 140(100 %)    Weapon Ready\n\tNearby Ally Units:\n\t\tUnit: AdeptPhaseShift    Tag: 0x101500001    ScreenPos: [95, 60]    Health: 140\n\t\tUnit: AdeptPhaseShift    Tag: 0x101540001    ScreenPos: [94, 65]    Health: 140\nTeam AdeptPhase-1 Info:\n\tTeam minimap position: [28, 29]\n\tControlled Team Units:\n\t\tUnit: AdeptPhaseShift    Tag: 0x101500001    ScreenPos: [66, 57]    Health: 140\n\t\tUnit: AdeptPhaseShift    Tag: 0x101540001    ScreenPos: [65, 62]    Health: 140\n\tNearby Ally Units:\n\t\tUnit: Adept    Tag: 0x100240001    ScreenPos: [40, 58]    Health: 140(100 %)\n\t\tUnit: Adept    Tag: 0x100500001    ScreenPos: [39, 63]    Health: 140(100 %)\nValid Actions:\n\tTeam Adept-1 Valid Actions:\n\t\t<Stop()>\n\t\t<No_Operation()>\n\t\t<Hold_Position()>\n\t\t<Move_Minimap(minimap)>\n\t\t<Move_Screen(screen)>\n\t\t<Attack_Unit(tag)>\n\tTeam AdeptPhase-1 Valid Actions:\n\t\t<Stop()>\n\t\t<No_Operation()>\n\t\t<Hold_Position()>\n\t\t<Move_Minimap(minimap)>\n\t\t<Move_Screen(screen)>\n\nTasks:\n\tTeam Adept-1' task: Go to minimap coordinate [52, 32], and try to avoid been detected or attacked before arrival.\n\tTeam AdeptPhase-1' task: Go to minimap coordinate [52, 32], and try to avoid been detected or attacked before arrival.\n\n", "a_t1": "### Analysis:- **Team Adept-1** and **Team AdeptPhase-1** both have the same task: to move to minimap coordinates [52, 32], where they should avoid detection and avoid being attacked.- **Team Adept-1** is already in transit and their previous move command to [52, 32] is likely still in progress.- **Team AdeptPhase-1** is also moving to the same location, and their previous action command to [52, 32] is also likely still being executed.**Important Considerations:**- Both teams should avoid being detected or attacked before reaching the objective.- As **AdeptPhase-1** has **Phase Shift**, it can potentially scout ahead or create distractions to help the main **Adept-1** team reach the destination without being seen. The key here is to use the **Phase Shift** ability efficiently.\n\nActions:\n\tTeam Adept-1:\n\t\t<Move_Screen([52, 32])>\n\tTeam AdeptPhase-1:\n\t\t<Move_Screen([52, 32])>", "s_t2": "Team Adept-1 Info:\n\tTeam minimap position: [24, 27]\n\tControlled Team Units:\n\t\tUnit: Adept    Tag: 0x100500001    ScreenPos: [64, 64]    Health: 140(100 %)    Weapon Ready\n\t\tUnit: Adept    Tag: 0x100240001    ScreenPos: [64, 59]    Health: 140(100 %)    Weapon Ready\n\tNearby Ally Units:\n\t\tUnit: AdeptPhaseShift    Tag: 0x101540001    ScreenPos: [86, 54]    Health: 140\n\t\tUnit: AdeptPhaseShift    Tag: 0x101500001    ScreenPos: [87, 49]    Health: 140\nTeam AdeptPhase-1 Info:\n\tTeam minimap position: [27, 26]\n\tControlled Team Units:\n\t\tUnit: AdeptPhaseShift    Tag: 0x101540001    ScreenPos: [64, 62]    Health: 140\n\t\tUnit: AdeptPhaseShift    Tag: 0x101500001    ScreenPos: [65, 57]    Health: 140\n\tNearby Ally Units:\n\t\tUnit: Adept    Tag: 0x100240001    ScreenPos: [42, 68]    Health: 140(100 %)\n\t\tUnit: Adept    Tag: 0x100500001    ScreenPos: [42, 73]    Health: 140(100 %)\n"}
"""

    self.eop = \
f"""
Reflections:
  1. Reflection on Stratage(RS): 
    xxxxx(consistency with task, consistency with rules, result according to state transition)
  2. Reflection on Actions(RA): 
    (1) <ActionName1(...)>: xxxxx(action validity, consistency with stratagy, the choose of action args, action result)
    (2) <ActionName2(...)>: xxxxx(action validity, consistency with stratagy, the choose of action args, action result)
    ...
  3. Reflection on Action Queue(RAQ): 
    xxxxx(result according to state transition)

Experiences:
  1. Experience on Stratagy(ES): 
    xxxxx(Stratagy xxx(name) is a xxx(good/bad) stratagy, it is recommended to adopting xxx(name) strategy to xxx(purpose))
  2. Experience on Actions(EA): 
    xxxxx(Using xxx(ActionName) at this moment is xxx(good/bad) because xxx, it is recommended to xxx)
  3. Experience on Action Queue(EAQ): 
    xxxxx(First xxx(ActionName) then xxx(ActionName) is a xxx(good/bad) action sequence, it is recommended to xxx)
"""

# Reflections:
#   1. <Move_Screen([52, 32])> is in the legal form <Move_Screen(screen)> of Valid Actions Part of s_t1.
#   2. Both the teams only queued only one action, it seems that there is no action queued in the sequence.
#   3. [52, 32] is the correct minimap coordinate for <Move_Minimap(minimap)>, but not a correct coordinate for <Move_Screen(screen)>.
#   4. We are not currently engaged in combat with the enemy, but we need to be mindful of concentrating our firepower to single unit during the engagement.
#
# Suggestions:
#   <Move_Screen(screen)> is a bad action.
#   It is suggest to use action <Move_Minimap(minimap)> instead of <Move_Screen(screen)> to go to the desired minimap coordinate.


PROTOSS_FACTORY = {
  'default': CombatGroupPrompt,
  'commander': CommanderPrompt,
  'developer': DeveloperPrompt,
}
TERRAN_FACTORY = {}
ZERG_FACTORY = {}

FACTORY = {
  'protoss': PROTOSS_FACTORY,
  'terran': TERRAN_FACTORY,
  'zerg': ZERG_FACTORY,
}


if __name__ == "__main__":
  from llm_pysc2.agents.configs.config import ProtossAgentConfig
  config = ProtossAgentConfig()
  prompt = CombatGroupPrompt('CombatGroup1', log_id=0, config=config)

  print("--" * 25 + "System Prompt" + "--" * 25)
  print(prompt.sp)
  print("--" * 25 + "Example Input Prompt" + "--" * 25)
  print(prompt.eip)
  print("--" * 25 + "Example Output Prompt" + "--" * 25)
  print(prompt.eop)