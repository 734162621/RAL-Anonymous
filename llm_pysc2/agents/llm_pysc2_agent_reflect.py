# Copyright 2024, RAL authors. All Rights Reserved.
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

from llm_pysc2.agents import LLMAgent
from llm_pysc2.cfg import AgentConfig, ProtossAgentConfig
from llm_pysc2.lib import llm_client, llm_prompt, utils
# from llm_pysc2.lib.rag import rag_dify
# from llm_pysc2.lib.llm_prompt import CombatGroupPrompt

from loguru import logger

import threading
import random
import pprint
import json
import time
import copy
import re


update_rate = 0.5
Nh1, Nv1, Ne1 = 5, 5, 5
Th_s, Th_sh = 0.99, 0.97


# region Thread Func
def thread_decision(self, text_o, base64_image, decision_type, enable_action_rectification):
  try:
    if self.config.LLM_SIMULATION_TIME > 0:
      logger.warning(f"[ID {self.log_id}] LLM SIMULATION MODE, no remote llm involved")
      time.sleep(self.config.LLM_SIMULATION_TIME)  # simulate llm response, for debug
      if self.name not in self.config.AGENTS_ALWAYS_DISABLE and self.enable:
        with open(self.log_dir_path + f"/{self.name}/a_inp.txt", "r") as f:
          self.raw_text_a = f.read()  # simulate llm response by reading text in a_inp.txt
    else:
      logger.success(f"[ID {self.log_id}] LLMAgent {self.name}: LLM Decision Started, Decision Type: {decision_type}")
      prompt = json.dumps({"state": text_o, "actions": 'currently no action, just make decisions'})
      self.raw_text_a = self.client_a1.query(prompt)
      strategy, actions = separate_text(self.raw_text_a, "Actions:")
      print(f"Decision Result: raw_text_a=\n{self.raw_text_a}" + "\n" + "--" * 25)
      if enable_action_rectification:
        # TODO: a=LLM_a(s,a)
        logger.success(f"[ID {self.log_id}] LLMAgent {self.name}: Action Rectification Started")
        prompt = json.dumps({"state": text_o, "actions": actions})
        self.raw_text_a = self.client_a2.query(prompt)
        print(f"\nAction Rectification: raw_text_a2=\n{self.raw_text_a}" + "\n" + "--" * 25)
        logger.success(f"[ID {self.log_id}] LLMAgent {self.name}: Action Rectification Finished")
  except:
    self.raw_text_a = ''
    logger.error(f"[ID {self.log_id}] LLMAgent {self.name}: Thread decision dead")


def thread_learn_h1(self, hypothesis_records, segment_id):
  logger.success(f"[ID {self.log_id}] LLMAgent {self.name}: Thread learn-h start")
  try:
    self.create_hypothesis1(hypothesis_records, segment_id)
    logger.success(f"[ID {self.log_id}] LLMAgent {self.name}: Thread learn-h finished")
  except:
    logger.error(f"[ID {self.log_id}] LLMAgent {self.name}: Thread learn-h dead")


def thread_learn_v1(self, segment_id):
  logger.success(f"[ID {self.log_id}] LLMAgent {self.name}: Thread learn-v start")
  try:
    self.create_validation1(self.last_h_record1, segment_id)
    logger.success(f"[ID {self.log_id}] LLMAgent {self.name}: Thread learn-v finished")
  except:
    logger.error(f"[ID {self.log_id}] LLMAgent {self.name}: Thread learn-v dead")


def thread_learn_e1(self, validation_records, segment_id1, segment_id2):
  logger.success(f"[ID {self.log_id}] LLMAgent {self.name}: Thread learn-e start")
  try:
    self.create_experience1(self.last_h_record1, validation_records, segment_id1, segment_id2)
    logger.success(f"[ID {self.log_id}] LLMAgent {self.name}: Thread learn-e finished")
  except:
    logger.error(f"[ID {self.log_id}] LLMAgent {self.name}: Thread learn-e dead")


# endregion


class Reflect_LLMAgent(LLMAgent):

  def __init__(self, name: str, log_id: int, start_time: str, config: "AgentConfig"=ProtossAgentConfig()):
    super(Reflect_LLMAgent, self).__init__(name, log_id, start_time, config)

    self.rag_api_base = 'http://127.0.0.1/v1'
    self.rag_api_key = 'dataset-cS2LCXGmve8rawYw6QWIJClh'

    self.client_a1 = llm_client.FACTORY[self.model_name](name, log_id, config)
    self.client_a1.set_prompt(ActionRectificationPrompt(name, log_id, config))
    self.client_a2 = llm_client.FACTORY[self.model_name](name, log_id, config)
    self.client_a2.set_prompt(ActionRectificationPrompt(name, log_id, config))
    self.client = self.client_a1  # store data in cost.txt

    self.history_state = {}
    self.history_action = {}
    self.history_hypothesis1_records = {}
    self.history_validation1_records = {}
    self.history_experience11_records = {}
    self.history_hypothesis1_record = {}

    self.rag_modules = {
      'h1': {"prompt": HypothesisPrompt1(name, log_id, config), "database_id": None, "document_id": None},
      'v1': {"prompt": ValidationPrompt1(name, log_id, config), "database_id": None, "document_id": None},
      'e11': {"prompt": ExperiencePrompt1(name, log_id, config), "database_id": None, "document_id": None},
      'e12': {"prompt": ExperiencePrompt1(name, log_id, config), "database_id": None, "document_id": None},
    }
    self.rag_module_names = {'H': 'h1', 'V': 'v1', 'E1': 'e11', 'E2': 'e12'}
    self.last_h_record1 = None

    self.rag_modules['h1']['client'] = llm_client.FACTORY[self.model_name](self.name, self.log_id, self.config)
    self.rag_modules['h1']['client'].system_prompt = self.rag_modules['h1']['prompt'].sp
    self.rag_modules['h1']['client'].example_i_prompt = self.rag_modules['h1']['prompt'].eip
    self.rag_modules['h1']['client'].example_o_prompt = self.rag_modules['h1']['prompt'].eop

    self.raw_text_a = ''

  # def _auto_detect_rag_id(self, obs):
  #
  #   map_name = str(obs.observation.map_name)
  #   skip_detect = True
  #   for key in self.rag_modules.keys():
  #     if self.rag_modules[key]['database_id'] is None:
  #       skip_detect = False
  #
  #   if not skip_detect:
  #     database_info_list = rag_dify.dify_dataset_list(self.rag_api_base, self.rag_api_key).json()['data']
  #     for database_info in database_info_list:
  #       database_name, database_id = database_info['name'], database_info['id']
  #       if database_name in self.rag_module_names.keys():
  #         key = self.rag_module_names[database_name]
  #         document_name, document_id = None, None
  #         document_info_list = rag_dify.dify_document_list(self.rag_api_base, self.rag_api_key, database_id).json()['data']
  #         for document_info in document_info_list:
  #           if map_name in document_info['name']:
  #             document_name, document_id = document_info['name'], document_info['id']
  #             break
  #         if document_name is not None:
  #           self.rag_modules[key]['database_id'] = database_id
  #           self.rag_modules[key]['document_id'] = document_id
  #           self.rag_modules[key]['client'] = llm_client.FACTORY[self.model_name](self.name, self.log_id, self.config)
  #           self.rag_modules[key]['client'].system_prompt = self.rag_modules[key]['prompt'].sp
  #           self.rag_modules[key]['client'].example_i_prompt = self.rag_modules[key]['prompt'].eip
  #           self.rag_modules[key]['client'].example_o_prompt = self.rag_modules[key]['prompt'].eop
  #   for key in self.rag_modules.keys():
  #     if self.rag_modules[key]['database_id'] is None:
  #       pprint.pprint(self.rag_modules)
  #       return False
  #   return True

  # TODO: Main API Func, receive obs and get actions
  def query(self,
            obs,
            epsilon=0,
            enable_learning=True,
            enable_retrieval=True,
            enable_action_rectification=True,
            ) -> None:

    if self.config.LLM_SIMULATION_TIME > 0:
      text_o, base64_image = self._before_query(obs)
      self.raw_text_a = self.get_text_a(text_o, base64_image=base64_image)  # query the llm and get response
      if self.name not in self.config.AGENTS_ALWAYS_DISABLE and self.enable:
        utils.write_to_file(json.dumps({self.main_loop_step: text_o}), self.log_dir_path + f"/{self.name}/o.txt")
      self._after_query(self.raw_text_a)
      return

    text_o, base64_image = self._before_query(obs)
    # self._auto_detect_rag_id(obs)

    # 反思上一步
    text_hypothesis = ''
    if len(self.translator_o.states) >= 2 and len(self.translator_a.actions) >= 2:
      text_hypothesis = self.create_hypothesis1(segment_id=None)

    print(text_hypothesis + "\n" + "--" * 25)
    text_hypothesis = '\n\t\t' + json.dumps({f"hypothesis": text_hypothesis})
    text_h = f"Hypothesis:" \
             f"\n\tHere is a retrieved hypothesis on possible strategy of current state:{text_hypothesis}" \
             f"\n\tFollow the hypothesis to make decisions.\n\n"
    text_o = self.translator_o.text_obs + text_h + self.translator_o.text_task + self.translator_o.final_prompt

    # Decision, Multi-thread mode (generate actions)
    if self.name not in self.config.AGENTS_ALWAYS_DISABLE and self.enable:
      utils.write_to_file(json.dumps({self.main_loop_step: text_o}), self.log_dir_path + f"/{self.name}/o.txt")
    thread_a = threading.Thread(target=thread_decision, args=(self, text_o, base64_image, 'reflect', enable_action_rectification))

    thread_a.start()
    thread_a.join()

    self._after_query(self.raw_text_a)

    if self.name not in self.config.AGENTS_ALWAYS_DISABLE and self.enable:
      for key in self.rag_modules.keys():
        if 'client' in self.rag_modules[key].keys():
          c = self.rag_modules[key]['client']
          if c is not None:
            path = self.log_dir_path + f"/{self.name}/cost_{key}.txt"
            client_cost = f"time={c.query_time:.2f}, ave_time={c.ave_query_time:.2f}, " \
                          f"token_in={c.query_token_in}, ave_token_in={c.ave_query_token_in:.2f}, " \
                          f"token_out={c.query_token_out}, ave_token_out = {c.ave_query_token_out:.2f}"
            utils.write_to_file(json.dumps({self.main_loop_step: client_cost}), path)
      c = self.client_a1
      path = self.log_dir_path + f"/{self.name}/cost_a1.txt"
      client_cost = f"time={c.query_time:.2f}, ave_time={c.ave_query_time:.2f}, " \
                    f"token_in={c.query_token_in}, ave_token_in={c.ave_query_token_in:.2f}, " \
                    f"token_out={c.query_token_out}, ave_token_out = {c.ave_query_token_out:.2f}"
      utils.write_to_file(json.dumps({self.main_loop_step: client_cost}), path)
      c = self.client_a2
      path = self.log_dir_path + f"/{self.name}/cost_a2.txt"
      client_cost = f"time={c.query_time:.2f}, ave_time={c.ave_query_time:.2f}, " \
                    f"token_in={c.query_token_in}, ave_token_in={c.ave_query_token_in:.2f}, " \
                    f"token_out={c.query_token_out}, ave_token_out = {c.ave_query_token_out:.2f}"
      utils.write_to_file(json.dumps({self.main_loop_step: client_cost}), path)


  # def retrieval(self, database_id, query, top_k, threshold) -> (list, str):
  #   # TODO: answer = RAG(Database;query|answer)
  #   try:
  #     scores, records = [], []
  #     kwargs = {'query': query, 'top_k': top_k}
  #     response = rag_dify.dify_retrieve(self.rag_api_base, self.rag_api_key, database_id, **kwargs).json()
  #     if 'records' in response.keys() and len(response['records']) > 0:
  #       for record in response['records']:
  #         score = record['score']
  #         if score is not None:
  #           scores.append(score)
  #         if score is not None and score > threshold:
  #           records.append(record)
  #     records_text = ''
  #     for record in records:
  #       records_text += '\n' + json.dumps(record)
  #     if 'status' in response.keys() and response['status'] != 200:
  #       pprint.pprint(self.rag_modules)
  #       print(kwargs['query'], response)
  #       print(self.rag_api_base, self.rag_api_key, database_id)
  #   except Exception as e:
  #     records, scores, records_text = [], [], ''
  #     logger.error(f"[ID {self.log_id}] LLMAgent {self.name}: error in retrieval: {str(e)}")
  #   return records, scores, records_text

  # def retrieval_hypothesis1(self, top_k=Nh1, threshold=Th_s) -> list:
  #   # TODO: RAG(H;s|h)
  #   logger.success(f"[ID {self.log_id}] LLMAgent {self.name}: Start RAG(H;s|h)")
  #   state_t = self.translator_o.states[-1]
  #   rag_query = state_t['units_info'] + state_t['valid_actions'] + state_t['task_info']
  #   records, scores, records_text = self.retrieval(self.rag_modules['h1']['database_id'], rag_query, top_k, threshold)
  #   utils.write_to_file('--' * 25 + records_text, self.log_dir_path + f"/{self.name}/retrieved_h.txt")
  #   logger.success(f"[ID {self.log_id}] LLMAgent {self.name}: Finished RAG(H;s|h), valid_records={len(records)}, scores={scores}")
  #   return records
  #
  # def retrieval_validation1(self, text_s_t1, hypothesis_record, top_k=Nv1, threshold=Th_sh) -> list:
  #   # TODO: RAG(V;(s,h)|v)
  #   logger.success(f"[ID {self.log_id}] LLMAgent {self.name}: Start RAG(V;(s,h)|v)")
  #   text_analysis, text_hypothesis = separate_text(hypothesis_record['segment']['answer'], 'Hypothesis:')
  #   text_hypothesis = clear_text(text_hypothesis)
  #   rag_query = text_s_t1 + text_hypothesis
  #   records, scores, records_text = self.retrieval(self.rag_modules['v1']['database_id'], rag_query, top_k, threshold)
  #   utils.write_to_file('--' * 25+ records_text, self.log_dir_path + f"/{self.name}/retrieved_v.txt")
  #   logger.success(f"[ID {self.log_id}] LLMAgent {self.name}: Finished RAG(V;(s,h)|v), valid_records={len(records)}, scores={scores}")
  #   return records
  #
  # def retrieval_experience11(self, top_k=Ne1, threshold=Th_s) -> list:
  #   # # TODO: RAG(E1;s|e)
  #   logger.success(f"[ID {self.log_id}] LLMAgent {self.name}: Start RAG(E1;s|e)")
  #   state_t = self.translator_o.states[-1]
  #   rag_query = state_t['units_info'] + state_t['valid_actions'] + state_t['task_info']
  #   records, scores, records_text = self.retrieval(self.rag_modules['e11']['database_id'], rag_query, top_k, threshold)
  #   utils.write_to_file('--' * 25 + records_text, self.log_dir_path + f"/{self.name}/retrieved_e1.txt")
  #   logger.success(f"[ID {self.log_id}] LLMAgent {self.name}: Finished RAG(E1;s|e), valid_records={len(records)}, scores={scores}")
  #   return records
  #
  # def retrieval_experience12(self, text_s_t1, hypothesis_record, top_k=Ne1, threshold=Th_sh) -> list:
  #   # TODO: RAG(E2;(s,h)|e)
  #   logger.success(f"[ID {self.log_id}] LLMAgent {self.name}: Start RAG(E2;(s,h)|e)")
  #   text_analysis, text_hypothesis = separate_text(hypothesis_record['segment']['answer'], 'Hypothesis:')
  #   text_hypothesis = clear_text(text_hypothesis)
  #   rag_query = text_s_t1 + text_hypothesis
  #   records, scores, records_text = self.retrieval(self.rag_modules['e12']['database_id'], rag_query, top_k, threshold)
  #   utils.write_to_file('--' * 25+ records_text, self.log_dir_path + f"/{self.name}/retrieved_e2.txt")
  #   logger.success(f"[ID {self.log_id}] LLMAgent {self.name}: Finished RAG(E2;(s,h)|e), valid_records={len(records)}, scores={scores}")
  #   return records
  #
  # def create_segment(self, database_id, document_id, rag_query, rag_answer, segment_id=None):
  #   if segment_id is None:
  #     kwargs = {'document_id': document_id, 'text_q': rag_query, 'text_a': rag_answer}
  #     response = rag_dify.dify_segment_qa_create_by_text(self.rag_api_base, self.rag_api_key, database_id, **kwargs)
  #   else:
  #     kwargs = {'document_id': document_id, 'text_q': rag_query, 'text_a': rag_answer, 'segment_id': segment_id}
  #     response = rag_dify.dify_segment_qa_update_by_text(self.rag_api_base, self.rag_api_key, database_id, **kwargs)
  #   return response

  def create_hypothesis1(self, segment_id=None):
    logger.success(f"[ID {self.log_id}] LLMAgent {self.name}: Start Create Hypothesis")
    # TODO: a=LLM(s), s'=ENV(s,a) h=LLM_h(s,[h],a,s'), store(H;s|h)
    # TODO: 未来可以更新这一部分，使用e来生成更先进的实验性策略

    # s_t1 = self.translator_o.states[-1]
    # text_s_t1 = s_t1['units_info'] + s_t1['valid_actions'] + s_t1['task_info']

    s_t1 = self.translator_o.states[-2]
    s_t2 = self.translator_o.states[-1]
    a_t1 = self.translator_a.actions[-2]
    text_s_t1 = s_t1['units_info'] + s_t1['valid_actions'] + s_t1['task_info']
    text_s_t2 = s_t2['units_info']
    _, text_strategy = separate_text(a_t1['analysis'], 'Strategy:')
    text_a_t1 = text_strategy + a_t1['actions']
    text_k_t1 = s_t1['knowledge_info']
    text_event = s_t2['event_info']

    # prompt = json.dumps({"s_t1": text_s_t1, "existing_hypothesis": text_h})
    prompt = json.dumps({"s_t1": text_s_t1, "a_t1": text_a_t1, "s_t2": text_s_t2, "event_info": text_event})
    logger.success(f"[ID {self.log_id}] LLMAgent {self.name}: Start: h=LLM_h(s,[h],a,s')")

    output = self.rag_modules['h1']['client'].query(prompt)
    utils.write_to_file(prompt, self.log_dir_path + f"/{self.name}/query_h.txt")
    utils.write_to_file("--" * 25 + '\n' + output, self.log_dir_path + f"/{self.name}/output_h.txt")
    logger.success(f"[ID {self.log_id}] LLMAgent {self.name}: Finished: h=LLM_h(s,[h],a,s')")

    analysis, hypothesis = separate_text(output, 'Hypothesis:')
    hypothesis = clear_text(hypothesis)
    rag_query, rag_answer = text_s_t1, hypothesis
    # response = self.create_segment(self.rag_modules['h1']['database_id'], self.rag_modules['h1']['document_id'], rag_query, rag_answer, segment_id)
    print(hypothesis)
    utils.write_to_file("--" * 25 + '\n' + rag_query + rag_answer, self.log_dir_path + f"/{self.name}/store_h.txt")
    logger.success(f"[ID {self.log_id}] LLMAgent {self.name}: Finished: store(H;s|h), {'create' if segment_id is None else f'update {segment_id}'}")

    return hypothesis


  # def create_validation1(self, hypothesis_record, segment_id=None, enable_long_term=False, force_long_term=True):
  #   logger.success(f"[ID {self.log_id}] LLMAgent {self.name}: Start Create Validation")
  #   # # TODO: needed a=LLM(s,h), s'=ENV(s,a); v=LLM_v(s,h,a,s'), store(V;(s+h)|v)
  #   # TODO: 未来可以更新这一部分，旧的v在学习过程中删除掉
  #
  #   long_term_validation = False
  #   t, n, m = self.main_loop_step - 1, 5, 4
  #   if (t-n) in self.history_state and (t-n) in self.history_action and (t-n+1) in self.history_state:
  #     long_term_validation = True
  #   if force_long_term and not long_term_validation:
  #     logger.warning(f"[ID {self.log_id}] LLMAgent {self.name}: force_long_term validation but data not prepared")
  #     segment_id = None
  #     return
  #   if not enable_long_term:
  #     long_term_validation = False
  #
  #   if long_term_validation and self.history_hypothesis1_record[t-n] is not None:
  #     s_t1 = self.history_state[t-n]
  #     s_t2 = self.history_state[t-n+1]
  #     a_t1 = self.history_action[t-n]
  #     text_analysis, text_hypothesis = separate_text(self.history_hypothesis1_record[t-n]['segment']['answer'], 'Hypothesis:')
  #   else:
  #     s_t1 = self.translator_o.states[-2]
  #     s_t2 = self.translator_o.states[-1]
  #     a_t1 = self.translator_a.actions[-2]
  #     text_analysis, text_hypothesis = separate_text(hypothesis_record['segment']['answer'], 'Hypothesis:')
  #
  #   text_s_t1 = s_t1['units_info'] + s_t1['valid_actions'] + s_t1['task_info']
  #   text_s_t2 = s_t2['units_info']
  #   _, text_strategy = separate_text(a_t1['analysis'], 'Strategy:')
  #   text_a_t1 = text_strategy + a_t1['actions']
  #   text_k_t1 = s_t1['knowledge_info']
  #   text_event = s_t2['event_info']
  #
  #   # text_analysis, text_hypothesis = separate_text(hypothesis_record['segment']['answer'], 'Hypothesis:')
  #   text_hypothesis = clear_text(text_hypothesis)
  #   text_hypothesis = json.dumps({f"hypothesis": text_hypothesis})
  #   text_h = f"Here is a hypothesis on possible strategy of current state:\n\t{text_hypothesis}\n" \
  #            f"Validate the hypothesis according to actual result."
  #
  #   if long_term_validation:
  #     event = events.get_events(self, t, n=n, i_list=[-m])[str(-m)]
  #     print(f"event1={event}")
  #     event_before = json.dumps(event)
  #     event = events.get_events(self, t, n=n+1, i_list=[m])[str(m)]
  #     print(f"event2={event}")
  #     event_after = json.dumps(event)
  #     prompt = json.dumps({"s_t1": text_s_t1, "a_t1": text_a_t1, "s_t2": text_s_t2, "current_hypothesis": text_h, "current_event_info": text_event, "long_horizon_event_before": event_before, "long_horizon_event_after": event_after})
  #   else:
  #     prompt = json.dumps({"s_t1": text_s_t1, "a_t1": text_a_t1, "s_t2": text_s_t2, "current_hypothesis": text_h, "current_event_info": text_event})
  #
  #   output = self.rag_modules['v1']['client'].query(prompt)
  #   utils.write_to_file(prompt, self.log_dir_path + f"/{self.name}/query_v.txt")
  #   utils.write_to_file("--" * 25 + '\n' + output, self.log_dir_path + f"/{self.name}/output_v.txt")
  #   logger.success(f"[ID {self.log_id}] LLMAgent {self.name}: Finished: v=LLM_v(s,h,a,s')")
  #
  #   analysis, validation = separate_text(output, 'Validation:')
  #   rag_query, rag_answer = text_s_t1 + text_hypothesis, validation
  #   response = self.create_segment(self.rag_modules['v1']['database_id'], self.rag_modules['v1']['document_id'], rag_query, rag_answer, segment_id)
  #   print(text_hypothesis)
  #   print(validation)
  #   utils.write_to_file("--" * 25 + '\n' + rag_query + rag_answer, self.log_dir_path + f"/{self.name}/store_v.txt")
  #   logger.success(f"[ID {self.log_id}] LLMAgent {self.name}: Finished: store(V;(s,h)|v), {'create' if segment_id is None else f'update {segment_id}'}")
  #
  #
  # def create_experience1(self, hypothesis_record, validation_records, segment_id1=None, segment_id2=None):
  #   logger.success(f"[ID {self.log_id}] LLMAgent {self.name}: Start Create Experience")
  #   # TODO: [v]=RAG(V,(s+h)), e=LLM_e(h,[v]) if len([v]) > n, store(E;s|e) if e
  #
  #   s_t1 = self.translator_o.states[-2]
  #   text_s_t1 = s_t1['units_info'] + s_t1['valid_actions'] + s_t1['task_info']
  #
  #   text_analysis, text_hypothesis = separate_text(hypothesis_record['segment']['answer'], 'Hypothesis:')
  #   text_hypothesis = clear_text(text_hypothesis)
  #   text_hypothesis = json.dumps({f"hypothesis": text_hypothesis})
  #   text_h = f"Here is a hypothesis on possible strategy of current state:\n{text_hypothesis}\n" \
  #            f"Generate 'Experience' on the hypothetical strategy according to 'Validation'."
  #
  #   text_v = f""
  #   for i in range(len(validation_records)):
  #     record = validation_records[i]
  #     text_analysis, text_validation = separate_text(record['segment']['answer'], 'Validation:')
  #     text_validation = json.dumps({f"validations-{i}": text_validation})
  #     text_v += '\n\t' + text_validation
  #   text_v = f"Here are the validations on the hypothetical strategy:{text_v}\n" \
  #            f"Generate 'Experience' on the hypothetical strategy according to these validations"
  #
  #   prompt = json.dumps({"hypothesis": text_h, "validations": text_v})
  #   output = self.rag_modules['e11']['client'].query(prompt)
  #   utils.write_to_file(prompt, self.log_dir_path + f"/{self.name}/query_e.txt")
  #   utils.write_to_file("--" * 25 + '\n' + output, self.log_dir_path + f"/{self.name}/output_e.txt")
  #   logger.success(f"[ID {self.log_id}] LLMAgent {self.name}: Finished: e=LLM_e(s,h,[v])")
  #
  #   analysis, experience = separate_text(output, 'Experience:')
  #   rag_query, rag_answer = text_s_t1, experience
  #   response = self.create_segment(self.rag_modules['e11']['database_id'], self.rag_modules['e11']['document_id'], rag_query, rag_answer, segment_id1)
  #   utils.write_to_file("--" * 25 + '\n' + rag_query + rag_answer, self.log_dir_path + f"/{self.name}/store_e1.txt")
  #   logger.success(f"[ID {self.log_id}] LLMAgent {self.name}: Finished: store(E1;s|e), {'create' if segment_id1 is None else f'update {segment_id1}'}")
  #
  #   rag_query, rag_answer = text_s_t1 + text_hypothesis, experience
  #   response = self.create_segment(self.rag_modules['e12']['database_id'], self.rag_modules['e12']['document_id'], rag_query, rag_answer, segment_id2)
  #   print(text_hypothesis)
  #   print(experience)
  #   utils.write_to_file("--" * 25 + '\n' + rag_query + rag_answer, self.log_dir_path + f"/{self.name}/store_e2.txt")
  #   logger.success(f"[ID {self.log_id}] LLMAgent {self.name}: Finished: store(E2;(s,h)|e), {'create' if segment_id1 is None else f'update {segment_id2}'}")


class ActionRectificationPrompt(llm_prompt.BasePrompt):
  def __init__(self, name, log_id, config):
    super(ActionRectificationPrompt, self).__init__(name, log_id, config)
    output_format = \
"""
Analysis:
  1. xxxxx
  2. xxxxx
    (1)
    (2)
  3. xxxxx
    (1)
    (2)
    (3)
    (4)
    (5)
    (6)
  4. xxxxx
    (1)
    (2)
    (3)
  5. xxxxx
    (1)
    (2)
    (3)
    (4)
  6. xxxxx
    (1)
    (2)
    (3)
    (4)

Actions:
  Team TeamName-1:
    <ActionName1(...)>  # format like **ActionName1(...)** and -ActionName1(...)- are not valid, must use <>
    <ActionName2(...)>
  Team TeamName-2:
    <ActionName1(...)>
    
Remember to concentrate **all teams‘ fire** at only one unit(Not only one team concentrate on attacking one unit, all teams need to concentrate on attacking one unit)
"""
    self.sp = \
f"""
You are a StarCraft2 frontline commander. You will receive 'valid actions', 'strategy' and 'actions' of current state.
You should correct the actions if there are issues with these actions. Finally, You should output actions in a given format(Regardless of whether the action has been modified or not).

Here are the aspects you need to analyse:
1. (Action Validity) Which actions do not in the listed in the 'Valid Actions' part of s_t1? Can other legal actions be used instead to achieve similar effects?
2. (Attack Analysis) If there is any action use unit tag, determine that 
   (1) (Target Analyse) which unit is the one with highest DPS, among which, which is the most vulnerable?
   (2) (Target Choose) which unit is the best the target?
3. (Move Analysis) If there is any move actions, determine that 
   (1) (Move Necessity Check) do we really need to move? (move/staying away from the battlefield will definitely lead to faster deaths of nearby allys. if there are allys nearby, do not move easily.)
   (2) (Coordinate System) if move is necessary, whether correctly selected **Screen** or **Minimap**? 
   (3) (Enemy direction) if move is necessary, where are the enemy/ally units and how long is the distance? (The positive direction of the x-axis/y-axis is to the right/up, use left/right/up/down and their combinations to describe direction).
   (4) (Moving direction) if move is necessary, where shall we go? close to enemy to share the damage from allys or away from enemy? (use right/left/up/down and their combinations to describe direction, and finally generate a coordinate)
   (5) (Moving distance) if move is necessary, whether the target position is **valid**(within map range) and **can reach enemy**(enemy in attack distance after moving)? if go away from enemy, whether **safe** and **long enough**(when at least 2 Manhattan Distance)?
   (6) (Micro operation) if move is necessary, whether all units need to move? can we use <Select_Unit_Move_xxx(...)> to optimize moving operations?
4. (Skill/Ability Analysis) If there is any skill/ability actions, determine that 
   (1) (Skill Necessity Check) whether the skill/ability at the optimal time for release? do we really need to use the skill right now?
   (2) (Optimal Release Position) whether correctly selected **Screen** or **Minimap** (if the skill need a target position)?
   (3) (Side Effect) what side effect of releasing the skill/ability? (cooldown time or other side effects)
5. (Fire Concentrating) 
   (1) (Current Target) whether **all teams** are concentrating their firepower **on the same target** (all <Attack_Unit(tag)> actions use the same tag)? 
   (2) (Re-choose Target) concentrating **all teams' firepower to only one unit** is **always necessary**, which unit should all team attack together?
   (3) (Correct Actions) which action dispersed the fire power to different units? how should it be modified?
   (4) (Next Target) if the target is nearly dead(health low than one salvo damage), queuing a next attack action is necessary (give 2 <Attack_Unit(tag)> action for each team), which unit should we attack next?
6. (Emergency evacuation)
   (1) (Heavily Damaged Unit) How much health does **each unit** currently have? which unit's health is lower than **33%**?
   (2) (Enemy Direction) if and only if unit health lower than **33%**, then, which direction is the enemy? (use left/right/up/down to describe direction)
   (3) (Retreat Direction) if and only if unit health lower than **33%** then retreat the unit, if so, which direction should the unit retreat?
   (4) (Retreat Action) consider that unit usually at screen center [12, 12] and retreat distance should more than 3 Manhattan Distance, what should the move action be like? write it(format: <Action_Name(args)>), and use it for emergency evacuation


After generating the 'Analysis', you need to generate 'Strategy' and 'Actions' according to analysis.
In the 'Actions' part, make sure each team have no more than {self.config.MAX_NUM_ACTIONS} actions.
Generate 'Analysis', 'Strategy' and 'Actions' strictly follow format:

{output_format}
"""
    self.eip = """xxxxx"""
    self.eop = f"""{output_format}"""

# 6. (Emergency evacuation)
#    (1) (Heavily Damaged Unit) How much health does **each unit** currently have? which unit's health is lower than **33%**?
#    (2) (Enemy Direction) if and only if unit health lower than **33%**, then, which direction is the enemy? (use left/right/up/down to describe direction)
#    (3) (Retreat Direction) if and only if unit health lower than **33%** then retreat the unit, if so, which direction should the unit retreat?
#    (4) (Retreat Action) consider that unit usually at screen center [12, 12] and retreat distance should more than 3 Manhattan Distance, what should the move action be like? write it(format: <Action_Name(args)>), and use it for emergency evacuation



class HypothesisPrompt1(llm_prompt.BasePrompt):
  def __init__(self, name, log_id, config):
    super(HypothesisPrompt1, self).__init__(name, log_id, config)
    output_format = \
"""
Analysis:
  1. xxxxx
  2. xxxxx
  3. xxxxx
  4. xxxxx
  5. xxxxx
Hypothesis:
  Hypothetical Strategy name: xxxxx (such as **Hold Position Attack**(concentrate fire on xxx first), **Hit and Run**(concentrate fire on xxx first), **Retreat**, **Wait for Healing**)
    Use xxxxx(actions) to xxxxx(purpose, to complete the task given in s_t1).
    Possible benefit: xxxxx (kill a unit/ dealt damage/ positive impact on the task/ other benefit)
    Possible cost: xxxxx (lose a unit/ taken damage/ negative impact on the task/ other cost)
"""
    self.sp = \
f"""
You are a StarCraft2 game analyser. You will receive two states s_t1 and s_t2(a game state of time t1 and t2) and a_t1 (analysis and actions of StarCraft2 frontline commander used in time t1).
You should analyse the actual effect according to the transition from s_t1 to s_t2 and event(if given). 
Then, generating hypothesis on possible better strategy(just one hypothesis), and output in a given format.

Here are some basic combat rules you need to follow in the analysis:
{llm_prompt.BASIC_COMBAT_RULES}

Here are the aspects you need to reflect:
  1. (Action Validity) Whether each action of a_t1 are in a legal form that listed in the 'Valid Actions' part of s_t1? 
  2. (Action Sequence) Whether each action of a_t1 is queued in correct sequence? 
  3. (Skills/Abilities) Is it currently **possible to use skills** and **should they be used**? What are the possible side effects of using these skills? 
  4. (Attack) Whether firepower concentrated on the most **vulnerable/valuable** enemy? What are the possible side effects of attacking these units? 
  5. (Move) Whether the position of moving is **valid**, **long enough** and **safe** for next several seconds? What are the possible side effects of moving to the position?

After generating the 'Analysis', you need to generate 'Hypothesis' with universality and robustness, which means
do not provide any detailed data (such as unit tags, unit health, specific coordinates) in the 'Hypothesis'.

Also, you will be provided with 'existing_hypothesis'. Do not generate similar hypothesis. You should propose diverse hypothetical strategies to achieve the task.
Also, Pay Attention to 'Valid Actions' and do not provide strategy that requires invalid actions in s_t1.

Generate 'Analysis' and 'Hypothesis' according to the following format:

{output_format}
"""
    self.eip = """xxxxx"""
    self.eop = f"""{output_format}"""


class ValidationPrompt1(llm_prompt.BasePrompt):
  def __init__(self, name, log_id, config):
    super(ValidationPrompt1, self).__init__(name, log_id, config)
    output_format = \
"""
Analysis:
  1. xxxxx
  2. xxxxx
  3. xxxxx
  4. xxxxx
Validation:
  Consider xxxxx.
  This is a xxxxx (excellent/good/bad/terrible) hypothesis.
"""
    self.sp = \
f"""
You are a StarCraft2 game analyser. You will receive two states s_t1 and s_t2(a game state of time t1 and t2) and a_t1 (analysis and actions of StarCraft2 frontline commander used in time t1).
Note that a_t1 was proposed according a strategy hypothesis, which means, it is testing the hypothesis in the game state s_t1.
You should analyse the actual effect according to the transition from s_t1 to s_t2 and events(if given).
Then, generating 'Validation' of the given hypothesis and output in a given format.

Here are some basic combat rules you need to follow in the analysis:
{llm_prompt.BASIC_COMBAT_RULES}

Here are the aspects you need to analyse:
  1. (Action Validity) Whether the strategy is available (main actions of the strategy in the 'Valid Action' part)
  2. (Strategy-Task Correlation) Whether the executed actions a_t1 helps to achieve the task of s_t1?
  3. (Benefits) According to current event and state transition from s_t1 to s_t2, what is the benefits of this hypothetical strategy? (positive impact on the task? killed enemy unit? dealt damage? or any other benefits)
  4. (Costs) According to current event and state transition from s_t1 to s_t2, what is the cost of this hypothetical strategy? (negative impact on the task? how many unit dead? how many health lost? gained debuff? or any other cost)

After generating the 'Analysis', you need to generate 'Validation', clearly point out the actual benefits and 
costs of this strategy(No need for pointing out potential benefits or costs).

Please carefully and accurately analyse costs and benefits and compare whether the benefits outweigh the losses. 
Do not analyze too optimistically, you should conduct rigorous numerical analysis.

Generate 'Analysis' and 'Validation' according to the following format:

{output_format}
"""
    self.eip = """xxxxx"""
    self.eop = f"""{output_format}"""


class ExperiencePrompt1(llm_prompt.BasePrompt):
  def __init__(self, name, log_id, config):
    super(ExperiencePrompt1, self).__init__(name, log_id, config)
    output_format = \
"""
Analysis:
  1. xxxxx
  2. xxxxx
Experience:
  xxxxx(content of the hypothesis strategy) is a xxxxx(hypothesis good/bad) hypothesis. 
  Its advantages include xxxx. Its drawbacks include xxxx. Extra attention should be paid on xxxxx when implementing this strategy.
"""
    self.sp = \
f"""
You are a StarCraft2 game analyser. You will receive strategy hypothesis and several validation data on the hypothesis.
Generating 'Experience' on the hypothetical strategy according to validations.

Here are some basic combat rules you need to follow in the analysis:
{llm_prompt.BASIC_COMBAT_RULES}

Here are the aspects you need to analyse:
  1. How many times the hypothesis achieved **good** results? What can be learned from them?
  2. How many times the hypothesis lead to **bad** results? What can be learned from them?

After generating the 'Analysis', you need to generate 'Experience', clearly point out the actual benefits and 
costs of this strategy(No need for pointing out potential benefits or costs).

Generate 'Analysis' and 'Experience' according to the following format:

{output_format}
"""
    self.eip = """xxxxx"""
    self.eop = f"""{output_format}"""


class ValidationPrompt2(llm_prompt.BasePrompt):
  def __init__(self, name, log_id, config):
    super(ValidationPrompt2, self).__init__(name, log_id, config)


class HypothesisPrompt2(llm_prompt.BasePrompt):
  def __init__(self, name, log_id, config):
    super(HypothesisPrompt2, self).__init__(name, log_id, config)


class ExperiencePrompt2(llm_prompt.BasePrompt):
  def __init__(self, name, log_id, config):
    super(ExperiencePrompt2, self).__init__(name, log_id, config)


def record_in_records(record, records):
  if record is None:
    return False
  for record_ in records:
    if record['segment']['id'] == record_['segment']['id']:
      return True
  return False

def separate_text(text, second_part_text) -> (str, str):
  text_1 = ''
  text_2 = ''
  if not isinstance(text, str):
    print(f"text in separate_text()={text}")
  for line in text.splitlines():
    if second_part_text in line or text_2 != '':
      text_2 += line + '\n'
    else:
      text_1 += line + '\n'
  return text_1, text_2

def clear_text(text) -> str:
  text = re.sub(r'0x\w+', 'tag', text)
  text = re.sub(r'\[-?\d+\.?\d*e?-?\d*?, -?\d+\.?\d*e?-?\d*?\]', '[x, y]', text)
  text = re.sub(r'\[-?\d+\.?\d*e?-?\d*? -?\d+\.?\d*e?-?\d*?\]', '[x, y]', text)
  text = re.sub(r'(\d+) HP', '', text)
  text = re.sub(r'(\d+)% HP', '', text)
  text = re.sub(r'(\d+) health', '', text)
  text = re.sub(r'(\d+)% health', '', text)
  return text
