# !/usr/bin/env python3
# -*- coding:utf-8 -*-

from agentuniverse.agent.agent import Agent
from agentuniverse.agent.agent_manager import AgentManager
# @Time    : 2024/5/8 11:41
# @Author  : wangchongshi
# @Email   : wangchongshi.wcs@antgroup.com
# @FileName: peer_chat_bot.py
from agentuniverse.base.agentuniverse import AgentUniverse

AgentUniverse().start(config_path="../../config/config.toml")


def chat(question: str):
    """Peer agents example.

    The peer agents in agentUniverse become a chatbot and can ask questions to get the answer.
    """
    instance: Agent = AgentManager().get_instance_obj("sale_peer_agent")
    instance.run(input=question)


if __name__ == "__main__":
    question = """半导体板块的走势会受到哪些因素的影响？"""
    chat(question)
