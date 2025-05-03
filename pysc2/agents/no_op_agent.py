# Copyright 2021 Google Inc. All Rights Reserved.
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
"""A no-op agent for starcraft."""

from pysc2.agents import base_agent

from s2clientprotocol import sc2api_pb2 as sc_pb

import time
class NoOpAgent(base_agent.BaseAgent):
  """A no-op agent for starcraft."""
  def __init__(self):
    super(NoOpAgent, self).__init__()
    self.t00 = 0
    self.t11 = 0

  def step(self, obs):
    super(NoOpAgent, self).step(obs)

    from pysc2.lib.actions import FUNCTIONS as F
    # import pysc2.lib.actions as a
    actions = F.Attack_unit('now', int(0x100000001), int(0x1000c0001))
    # actions = a.FunctionCall(574, ('now', int(0x100000001), int(0x1000c0001)))
    time.sleep(0.05)

    return actions # sc_pb.Action()
