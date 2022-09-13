import json
import os.path as osp
from typing import Union


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # template_name = "alpaca"
            template_name = "alpaca_short"
        file_name = osp.join("dataloader", "templates", f"{template_name}.json")
        if not osp.exists(file_name):
      