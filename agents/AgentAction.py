from __future__ import annotations

from dataclasses import dataclass
from typing import Union, Dict


# Custom AgentAction that saves the number of custom parsings done to be used during eval
@dataclass
class AgentAction:
    """A full description of an action for an ActionAgent to execute."""

    tool: str
    """The name of the Tool to execute."""
    tool_input: Union[str, Dict]
    """The input to pass in to the Tool."""
    log: str
    """Additional information to log about the action."""
    custom_parsings: int
    """Number of custom parsings performed on the tool input."""
