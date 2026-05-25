import io
import re
import base64
import numpy as np
from PIL import Image
from functools import lru_cache
from bs4 import BeautifulSoup, Tag

import collections
collections.Callable = collections.abc.Callable

from autoe2e.utils.singleton import Singleton, AbstractSingleton
from autoe2e.utils.hash import hash_string
from autoe2e.utils.queue import Queue
from autoe2e.utils.logger import logger


KEEP_ATTRIBUTES = [
    'href',
    'src',
    'alt',
    'action',
    'name',
    'type',
    'for',
    'id',
    'class',
    'placeholder',
    'value',
    'alt',
    # input attributes
    'min',
    'max',
    'maxlength',
    'multiple',
    'pattern',
    'required',
    'readonly',
    'disabled',
    'step',
    # data attributes
    'data-testid',
    'data-formid',
    'data-submitid'
]


def png_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
    
    image = Image.open(io.BytesIO(image_data))
    resized_image = image.resize((512, 512))
    
    buffer = io.BytesIO()
    resized_image.save(buffer, format="PNG")
    base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    return base64_image


def extract_response_content(text):
    """Extracts content enclosed within a <Response> tag.

    Args:
        text: The string to parse.

    Returns:
        The content inside the <Response> tag, or None if not found.
    """
    if not text:
        return None
    
    # Handle both <Response> and <Response>: formats with potential whitespace and newlines
    match = re.search(r"<Response>:?\s*(.*?)\s*(?:</Response>|$)", text, re.DOTALL)
    if match:
        content = match.group(1).strip()
        return content if content else None
    return None


def log_user_messages(user_messages):
    """Logs the user messages in a human-readable format.

    Args:
        user_messages: The user messages to log.
    """
    for message in filter(lambda x: x['type'] == 'text', user_messages):
        logger.info(message["text"])


def clean_children_html(element_html):
    element = BeautifulSoup(element_html, 'html.parser')
    
    for child in element.descendants:
        if isinstance(child, Tag):
            for attr in list(child.attrs):
                if attr not in KEEP_ATTRIBUTES:
                    del child[attr]
    
    return str(element)


def geometric_score(rank, p=0.5, R=10):
    """
    Convert rank to score using geometric distribution.
    
    Formula: rank_score(r) = (r - 1) * log(1-p) + log(p)
    
    With p=0.5, R=10:
    - Rank 1: -0.693
    - Rank 2: -1.386
    - Rank 3: -2.079
    - Penalty (not in top-R): -7.62
    
    Ablation Support:
    - A3.1: Uniform scoring (equal weight for top-R)
    - A3.2: Linear scoring (linear decay by rank)
    - A3.3: Binary scoring (1 if in top-R, 0 otherwise)
    - A3.4: Different p values
    - A3.5: Different R values
    
    Note: When in ablation mode, the scoring function component is used
    instead of default parameters.
    
    Args:
        rank: Feature rank (1-indexed, 1 is best). None for penalty score.
        p: Probability parameter (default 0.5 per paper)
        R: Maximum candidates to consider (default 10 per paper)
    
    Returns:
        Log probability score
    """
    # Check for ablation mode (imported here to avoid circular imports)
    try:
        from autoe2e.ablation_integration import is_ablation_mode, get_scoring_function
        if is_ablation_mode():
            sf = get_scoring_function()
            if sf:
                return sf.score(rank)
    except ImportError:
        pass  # ablation_integration not available, use default
    
    # Default geometric scoring
    if rank is not None and rank >= 1:
        # Paper formula: (r-1) * log(1-p) + log(p)
        return (rank - 1) * np.log(1 - p) + np.log(p)
    # Penalty for features not in top-R: R * log(1-p) + log(p)
    return R * np.log(1 - p) + np.log(p)


__all__ = [
    'Singleton',
    'AbstractSingleton',
    'hash_string',
    'Queue',
    'logger',
    'png_to_base64',
    'extract_response_content',
    'log_user_messages',
    'clean_children_html',
    'geometric_score'
]