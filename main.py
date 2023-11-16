import traceback
import warnings
from typing import TYPE_CHECKING, Optional, Tuple, Union

import numpy as np
import torch
import tqdm

from .audio import (
    FRAMES_PER_SECOND,
