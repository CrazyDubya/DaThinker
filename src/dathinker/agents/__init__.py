"""Thinking agents module."""

from .base import BaseAgent, AgentResponse
from .socratic import SocraticAgent
from .devils_advocate import DevilsAdvocateAgent
from .clarifier import ClarifierAgent
from .synthesizer import SynthesizerAgent
from .perspective import PerspectiveExpanderAgent

__all__ = [
    "BaseAgent",
    "AgentResponse",
    "SocraticAgent",
    "DevilsAdvocateAgent",
    "ClarifierAgent",
    "SynthesizerAgent",
    "PerspectiveExpanderAgent",
]
