"""Model registry. To add an architecture: subclass PromptSegModel in a new module, decorate it with
@register("<name>"), import the module below, and set `model.name: <name>` in the config."""
from .base import PromptSegModel
from .registry import MODELS, build_model, register
from . import clipseg  # noqa: F401  (registers "clipseg")
