"""
Inference Utilities Module

Handles inference tasks for the AutoE2E system. Backend is provided by
``llm_api_call`` (Ollama by default, OpenAI for the GPT-4o reference run).
Includes robust parsing helpers to tolerate the smaller local models'
output variability.

Key Functions:
- extract_state_context: Extracts context from application state using vision
- extract_action_functionalities: Infers functionalities from UI actions
- insert_functionalities: Stores functionalities with embeddings in MongoDB
- update_functionality_score: Updates scoring for feature ranking
- mark_final_functionalities: Marks features as final/testable
- is_action_critical: Determines if an action is critical (e.g., delete, submit)
- create_form_filling_values: Generates test values for form fields

Parsing Helpers:
- strip_code_fences: Remove markdown code fences from text
- extract_first_json_array: Find first [...] block in text
- extract_first_json_object: Find first {...} block in text
- get_llm_text: Extract and clean LLM response text
- parse_feature_list: Parse features from various formats
- parse_bool_list: Parse boolean arrays with length enforcement
- parse_bool: Parse single boolean values
"""

import os
import re
import json
import ast

from dotenv import load_dotenv
from bson.objectid import ObjectId

from autoe2e.utils import logger

from autoe2e.browser.utils import save_screenshot
from autoe2e.crawler.crawl_context import CrawlContext
from autoe2e.crawler.state import State, StateIdEvaluator
from autoe2e.crawler.action import Action

from autoe2e.llm_api_call import (
    sonnet_chain,
    haiku_chain,
    embeddings
)
from autoe2e.prompts import (
    CONTEXT_EXTRACTION_SYSTEM_PROMPT,
    FUNCTIONALITY_EXTRACTION_SYSTEM_PROMPT,
    SIMILARITY_SYSTEM_PROMPT,
    CRITICAL_ACTION_SYSTEM_PROMPT,
    FORM_VALUE_SYSTEM_PROMPT,
    FINALITY_SYSTEM_PROMPT,
    create_context_user_messages,
    create_functionality_user_messages,
    create_similarity_user_messages,
    create_simple_user_messages,
    create_finality_user_messages
)
from autoe2e.utils import (
    png_to_base64,
    extract_response_content,
    geometric_score
)
from autoe2e.mongo_utils import (
    action_func_db,
    func_db
)


load_dotenv()


# =============================================================================
# Parsing Helper Functions
# =============================================================================

def strip_code_fences(text: str) -> str:
    """
    Remove markdown code fences from text.
    Handles ```json, ```python, ``` and similar patterns.
    
    Args:
        text: Input text possibly containing code fences
        
    Returns:
        Text with code fences removed
    """
    if not text:
        return ""
    
    # Remove code fences with language specifiers (```json, ```python, etc.)
    text = re.sub(r'```\w*\s*\n?', '', text)
    # Remove closing code fences
    text = re.sub(r'\n?```', '', text)
    
    return text.strip()


def extract_first_json_array(text: str) -> str | None:
    """
    Find and extract the first JSON array [...] from text.
    Handles nested brackets correctly.
    
    Args:
        text: Input text containing a JSON array
        
    Returns:
        The extracted JSON array string, or None if not found
    """
    if not text:
        return None
    
    # Find the first '[' character
    start = text.find('[')
    if start == -1:
        return None
    
    # Track bracket depth to find matching ']'
    depth = 0
    in_string = False
    escape_next = False
    
    for i in range(start, len(text)):
        char = text[i]
        
        if escape_next:
            escape_next = False
            continue
        
        if char == '\\':
            escape_next = True
            continue
        
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        
        if in_string:
            continue
        
        if char == '[':
            depth += 1
        elif char == ']':
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    
    return None


def extract_first_json_object(text: str) -> str | None:
    """
    Find and extract the first JSON object {...} from text.
    Handles nested braces correctly.
    
    Args:
        text: Input text containing a JSON object
        
    Returns:
        The extracted JSON object string, or None if not found
    """
    if not text:
        return None
    
    # Find the first '{' character
    start = text.find('{')
    if start == -1:
        return None
    
    # Track brace depth to find matching '}'
    depth = 0
    in_string = False
    escape_next = False
    
    for i in range(start, len(text)):
        char = text[i]
        
        if escape_next:
            escape_next = False
            continue
        
        if char == '\\':
            escape_next = True
            continue
        
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        
        if in_string:
            continue
        
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    
    return None


def get_llm_text(res) -> str:
    """
    Extract and clean LLM response text.
    Handles <Response> tags, code fences, and raw responses.
    
    Args:
        res: Raw LLM response (string or object)
        
    Returns:
        Cleaned text string
    """
    # First try extract_response_content for <Response> tags
    content = extract_response_content(res)
    
    # If no <Response> tags found, use str(res)
    if content is None:
        content = str(res) if res is not None else ""
    
    # Remove <Reasoning> section if present (LLM may include it by mistake)
    reasoning_pattern = r'<Reasoning>.*?(?:</Reasoning>|(?=<Response>))'
    content = re.sub(reasoning_pattern, '', content, flags=re.DOTALL)
    
    # Strip code fences and whitespace
    content = strip_code_fences(content)
    
    return content.strip()


def clean_feature_text(feature: str) -> str:
    """
    Clean feature text from verbose LLM formatting.
    
    Handles patterns like:
    - "1. **Feature Name:** description" -> "Feature Name"
    - "**Feature Name:** description" -> "Feature Name"
    - "Feature Name: description with explanation" -> "Feature Name"
    """
    if not feature:
        return feature
    
    # Remove leading numbers like "1. ", "2. "
    feature = re.sub(r'^\d+\.\s*', '', feature)
    
    # Extract text from **bold** format: "**Feature Name:** description" -> "Feature Name"
    bold_match = re.match(r'\*\*([^*]+)\*\*:?\s*.*', feature)
    if bold_match:
        return bold_match.group(1).strip()
    
    # If there's a colon followed by a long explanation, take only before colon
    # But only if what follows looks like a sentence (has spaces and is long)
    if ':' in feature:
        parts = feature.split(':', 1)
        # If the part after colon looks like an explanation (>30 chars), keep only before
        if len(parts) > 1 and len(parts[1].strip()) > 30:
            return parts[0].strip()
    
    return feature.strip()


def parse_feature_list(text: str) -> list[str]:
    """
    Parse a list of features from various LLM output formats.
    
    Accepts:
    - List of dicts with "feature" key: [{"feature": "..."}, ...]
    - List of strings: ["feature1", "feature2"]
    - Single dict with "feature": {"feature": "..."}
    - Malformed JSON with extractable "feature": "..." patterns
    
    Args:
        text: LLM response text
        
    Returns:
        List of feature strings, empty list on failure
    """
    if not text:
        return []
    
    # Clean the text
    text = strip_code_fences(text)
    
    # Try to extract JSON array first
    json_array = extract_first_json_array(text)
    if json_array:
        try:
            parsed = json.loads(json_array)
            if isinstance(parsed, list):
                features = []
                for item in parsed:
                    if isinstance(item, dict) and 'feature' in item:
                        features.append(clean_feature_text(str(item['feature'])))
                    elif isinstance(item, str):
                        features.append(clean_feature_text(item))
                if features:
                    return features
        except json.JSONDecodeError:
            pass
    
    # Try to extract JSON object (single feature)
    json_obj = extract_first_json_object(text)
    if json_obj:
        try:
            parsed = json.loads(json_obj)
            if isinstance(parsed, dict) and 'feature' in parsed:
                return [clean_feature_text(str(parsed['feature']))]
        except json.JSONDecodeError:
            pass
    
    # Try direct JSON parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            features = []
            for item in parsed:
                if isinstance(item, dict) and 'feature' in item:
                    features.append(clean_feature_text(str(item['feature'])))
                elif isinstance(item, str):
                    features.append(clean_feature_text(item))
            if features:
                return features
        elif isinstance(parsed, dict) and 'feature' in parsed:
            return [clean_feature_text(str(parsed['feature']))]
    except json.JSONDecodeError:
        pass
    
    # Fallback: regex extraction of "feature": "..." patterns
    feature_pattern = r'"feature"\s*:\s*"([^"]+)"'
    matches = re.findall(feature_pattern, text)
    if matches:
        return [clean_feature_text(f) for f in matches]
    
    # Alternative pattern with single quotes
    feature_pattern_single = r"'feature'\s*:\s*'([^']+)'"
    matches = re.findall(feature_pattern_single, text)
    if matches:
        return [clean_feature_text(f) for f in matches]
    
    logger.info(f"Failed to parse feature list from: {text[:200]}...")
    return []


def parse_bool_list(text: str, expected_len: int) -> list[bool]:
    """
    Parse a boolean array from LLM output.
    
    Accepts:
    - JSON array: [true, false, true]
    - Python list: [True, False, True]
    
    Args:
        text: LLM response text
        expected_len: Expected length of the boolean array
        
    Returns:
        List of booleans with exactly expected_len elements (padded with False or trimmed)
    """
    if not text or expected_len <= 0:
        return [False] * expected_len
    
    # Clean the text
    text = strip_code_fences(text)
    
    result = None
    
    # Try to extract JSON array first
    json_array = extract_first_json_array(text)
    if json_array:
        try:
            parsed = json.loads(json_array)
            if isinstance(parsed, list):
                result = [bool(x) for x in parsed]
        except json.JSONDecodeError:
            pass
    
    # If JSON parsing failed, try ast.literal_eval for Python format
    if result is None:
        json_array = extract_first_json_array(text) or text.strip()
        try:
            parsed = ast.literal_eval(json_array)
            if isinstance(parsed, list):
                result = [bool(x) for x in parsed]
        except (ValueError, SyntaxError):
            pass
    
    # If still no result, try direct parsing
    if result is None:
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                result = [bool(x) for x in parsed]
        except json.JSONDecodeError:
            pass
    
    if result is None:
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                result = [bool(x) for x in parsed]
        except (ValueError, SyntaxError):
            pass
    
    # If parsing failed completely, return default
    if result is None:
        logger.info(f"Failed to parse bool list from: {text[:200]}...")
        return [False] * expected_len
    
    # Ensure exactly expected_len elements
    if len(result) < expected_len:
        result.extend([False] * (expected_len - len(result)))
    elif len(result) > expected_len:
        result = result[:expected_len]
    
    return result


def parse_bool(text: str, default: bool = False) -> bool:
    """
    Parse a single boolean value from LLM output.
    
    Accepts:
    - JSON boolean: true/false
    - Python boolean: True/False
    - String representations: "true", "false", "True", "False"
    
    Args:
        text: LLM response text
        default: Default value if parsing fails
        
    Returns:
        Parsed boolean value or default
    """
    if not text:
        return default
    
    # Clean the text
    text = strip_code_fences(text).strip().lower()
    
    # Direct string check
    if text in ('true', 'yes', '1'):
        return True
    if text in ('false', 'no', '0'):
        return False
    
    # Try JSON parsing
    try:
        parsed = json.loads(text)
        if isinstance(parsed, bool):
            return parsed
        if isinstance(parsed, dict):
            # Handle {"boolean": true} or {"result": true} formats
            for key in ('boolean', 'result', 'value', 'is_critical', 'critical'):
                if key in parsed:
                    return bool(parsed[key])
    except json.JSONDecodeError:
        pass
    
    # Try ast.literal_eval for Python format
    try:
        parsed = ast.literal_eval(text.strip())
        return bool(parsed)
    except (ValueError, SyntaxError):
        pass
    
    # Try to find true/false anywhere in the text
    if 'true' in text:
        return True
    if 'false' in text:
        return False
    
    logger.info(f"Failed to parse bool from: {text[:100]}...")
    return default


def parse_json_object(text: str) -> dict:
    """
    Parse a JSON object from LLM output with robust handling.
    
    Args:
        text: LLM response text
        
    Returns:
        Parsed dict or empty dict on failure
    """
    if not text:
        return {}
    
    # Clean the text
    text = strip_code_fences(text)
    
    # Try to extract JSON object
    json_obj = extract_first_json_object(text)
    if json_obj:
        try:
            parsed = json.loads(json_obj)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    
    # Try direct JSON parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    
    logger.info(f"Failed to parse JSON object from: {text[:200]}...")
    return {}


# =============================================================================
# State Context Extraction
# =============================================================================

def extract_state_context(
    crawl_context: CrawlContext,
    state: State,
    prev_state: State | None = None,
    prev_action: Action | None = None
) -> str:
    """
    Extract context from application state using vision model.
    
    Ablation Support:
    - A1.1: When include_screenshot=False, uses placeholder image
    - A1.2: When include_previous_state=False, always "None. This is the first state."
    - A1.3: When include_previous_action=False, always "None. This is the first state."
    - A1.4: Minimal context (all above combined)
    """
    # Import ablation integration (lazy to avoid circular imports)
    from autoe2e.ablation_integration import (
        is_ablation_mode, 
        get_context_extractor,
        should_include_screenshot,
        should_include_previous_state,
        should_include_previous_action
    )
    
    state_id = state.get_id(StateIdEvaluator.BY_ACTIONS)
    
    screenshot_path = f'{crawl_context.config.temp_dir}/screenshot_{state_id}.png'
    save_screenshot(crawl_context.driver, screenshot_path)
    logger.info(f'Saved screenshot to {screenshot_path}')
    
    # Determine what context to include based on ablation config
    include_screenshot = should_include_screenshot()
    include_prev_state = should_include_previous_state()
    include_prev_action = should_include_previous_action()
    
    if is_ablation_mode():
        logger.info(f'[ABLATION] Context: screenshot={include_screenshot}, prev_state={include_prev_state}, prev_action={include_prev_action}')
    
    # Prepare context inputs (respecting ablation settings)
    if include_prev_state and prev_state is not None:
        previous_state_context = prev_state.get_context()
    else:
        previous_state_context = "None. This is the first state."
    
    if include_prev_action and prev_action is not None:
        previous_action_html = prev_action.element.outerHTML
    else:
        previous_action_html = "None. This is the first state."
    
    # Get image (real screenshot or placeholder based on ablation config)
    if include_screenshot:
        image_base64 = png_to_base64(screenshot_path)
    else:
        # Use placeholder image when screenshot is disabled (A1.1)
        ctx_extractor = get_context_extractor()
        if ctx_extractor:
            image_base64 = ctx_extractor.get_placeholder_image()
        else:
            # 1x1 transparent PNG as fallback
            image_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    
    try:
        context_text = sonnet_chain(
            CONTEXT_EXTRACTION_SYSTEM_PROMPT,
            create_context_user_messages(
                {
                    "description": "None",
                    "previous_state": previous_state_context,
                    "previous_action": previous_action_html,
                },
                image_base64
            )
        )
        return get_llm_text(context_text)
    except Exception as e:
        logger.error(f"Failed to extract state context: {e}")
        return "Context extraction failed"


# =============================================================================
# Action Functionalities Extraction
# =============================================================================

def extract_action_functionalities(
    state: State,
    action: Action,
    prev_action: Action | None = None
) -> list[str]:
    """
    Extract functionalities for an action. Single LLM call, no fallback.
    
    Returns:
        List of feature strings, empty list on failure
    """
    try:
        res = sonnet_chain(
            FUNCTIONALITY_EXTRACTION_SYSTEM_PROMPT,
            create_functionality_user_messages(
                state.context,
                action.element.outerHTML,
                prev_action.element.outerHTML if prev_action is not None else None
            )
        )
    except Exception as e:
        logger.error(f"LLM call failed for functionality extraction: {e}")
        return []
    
    # Use robust parsing
    text = get_llm_text(res)
    features = parse_feature_list(text)
    
    if not features:
        logger.info("No features extracted from LLM response")
    
    return features


# =============================================================================
# Similarity Search and Database Functions
# =============================================================================

def query_similar_functionalities(embedding):
    """Query MongoDB for similar functionalities using vector search."""
    query = [
        {
            '$vectorSearch': {
                'index': 'vector_index', 
                'path': 'embedding', 
                'queryVector': embedding, 
                'limit': 50,
                'numCandidates': 200,
                'filter': {
                    'app': os.getenv("APP_NAME")  # Filter DURING vector search
                }
            }
        },
        {
            '$match': {
                'app': os.getenv("APP_NAME"),
            }
        },
        {
            '$limit': 5
        }
    ]

    similar_funcs = list(func_db.aggregate(query))
    return similar_funcs


def get_exact_match_indices(text, similar_funcs):
    """Return indices of vector-search hits whose text exactly matches `text`."""
    return [
        i for i in range(len(similar_funcs))
        if similar_funcs[i].get('text', '') == text
    ]


def map_similar_func_to_exact_match(func_info):
    """
    Map a functionality to existing similar ones (exact match + LLM similarity)
    or mark it as new. This follows the original AUTOE2E paper aggregation:
    exact text match first, then a single LLM equivalence call.
    """
    rank, text, embedding, similar_funcs = func_info

    # PRIORITY 1: Exact text matches in vector search results
    exact_match_indices = get_exact_match_indices(text, similar_funcs)

    if len(exact_match_indices) > 0:
        match = {
            'match': True,
            'match_index': exact_match_indices,
            'combined_text': text,
            'match_source': 'exact_text_match'
        }
    else:
        match = {'match': False}

    # PRIORITY 2: LLM semantic similarity (only if no exact match found)
    if len(similar_funcs) != 0 and not match.get('match'):
        try:
            res = sonnet_chain(
                SIMILARITY_SYSTEM_PROMPT,
                create_similarity_user_messages(
                    text,
                    '\n'.join(map(lambda x: x['text'], similar_funcs))
                )
            )
            llm_text = get_llm_text(res)
            parsed_match = parse_json_object(llm_text)

            if parsed_match:
                match = parsed_match
                match['match_source'] = 'llm_similarity'
        except Exception as e:
            logger.error(f"Similarity matching failed: {e}")

    # Convert match_index to match_id (actual MongoDB _ids)
    if match.get('match'):
        match_index = match.get('match_index')
        if isinstance(match_index, int):
            match['match_id'] = [similar_funcs[match_index]['_id']]
        elif isinstance(match_index, list) and match_index:
            match['match_id'] = [
                similar_funcs[m]['_id'] for m in match_index if m < len(similar_funcs)
            ]

    match['rank'] = rank
    match['text'] = text
    match['embedding'] = embedding

    return match


def no_match_insert(match):
    """Insert a new functionality that did not match any existing one."""
    app_name = os.getenv("APP_NAME")
    text = match['text']

    res = func_db.insert_one({
        "app": app_name,
        "text": text,
        "embedding": match['embedding'],
        "score": geometric_score(match['rank'] + 1),  # 1-indexed: rank 0 -> rank 1
        "final": False,
        "executable": True
    })
    return res.inserted_id


def match_update(match):
    """Update an existing functionality with new match information."""
    combined_text = match.get('combined_text', match['text'])

    try:
        embedding = embeddings.embed_query(combined_text)
    except Exception as e:
        error_msg = str(e).lower()
        if "not found" in error_msg or "model" in error_msg:
            raise RuntimeError(
                f"Embedding model missing or unavailable. "
                f"Please run: ollama pull nomic-embed-text\n"
                f"Original error: {e}"
            )
        raise

    app_name = os.getenv("APP_NAME")
    primary_id = match['match_id'][0]

    func_db.update_one(
        filter={
            'app': app_name,
            '_id': primary_id
        },
        update={
            '$set': {
                'text': combined_text,
                'embedding': embedding
            }
        },
        upsert=False
    )

    if len(match['match_id']) > 1:
        # Remove duplicate documents
        func_db.delete_many({
            'app': app_name,
            '_id': {'$in': match['match_id'][1:]}
        })

        # Update action-function pointers to point to the primary match
        action_func_db.update_many(
            filter={
                'app': app_name,
                'func_pointer': {'$in': list(map(str, match['match_id'][1:]))}
            },
            update={
                '$set': {
                    'func_pointer': str(primary_id)
                }
            },
            upsert=False
        )

    return primary_id


def update_databases_with_match(match):
    """Update or insert functionality based on match status."""
    if match.get('match'):
        return match_update(match)
    return no_match_insert(match)


def insert_functionalities(functionalities: list[str]):
    """
    Insert functionalities into the database with embeddings.

    Pipeline (matches the original AUTOE2E paper):
      1. Embed the candidate functionality texts.
      2. Vector-search the existing collection for nearest neighbours.
      3. For each candidate: exact text match -> merge; otherwise a single LLM
         equivalence call decides whether to merge or insert.

    Raises RuntimeError if the embedding model is unavailable.
    """
    if not functionalities:
        return []

    try:
        embed_vectors = embeddings.embed_documents(functionalities)
    except Exception as e:
        raise RuntimeError(
            f"Embedding generation failed. "
            f"Please ensure the embedding model is available:\n"
            f"  ollama pull nomic-embed-text\n\n"
            f"Original error: {e}"
        )

    similar_funcs = map(query_similar_functionalities, embed_vectors)

    matches = map(
        map_similar_func_to_exact_match,
        zip(
            range(len(functionalities)),
            functionalities,
            embed_vectors,
            similar_funcs
        )
    )

    insertion_ids = list(map(update_databases_with_match, matches))

    return insertion_ids


def insert_action_functionality(
    func_ids: list,
    state_id: str,
    state_url: str,
    prev_state_id: str,
    action_id: str,
    prev_action_id: str,
    action_test_id: str,
    action_depth: int,
    action_type: str = "SINGLE",
    form_fields_chain: list = None  # NEW: list of form field test_ids like ["t15", "t16"]
):
    """Insert action-functionality mappings into the database."""
    documents = [
        {
            "app": os.getenv("APP_NAME"),
            "url": state_url,
            "state": state_id,
            "prev_state": prev_state_id,
            "action": action_id,
            "prev_action": prev_action_id,
            "test_id": action_test_id,
            "depth": action_depth,
            "type": action_type,
            "rank_score": geometric_score(i + 1),  # 1-indexed: rank 1 = first item
            "func_pointer": str(func_ids[i]),
            "final": False,
            "should_execute": True,
            "form_fields": form_fields_chain  # NEW: store form fields if any
        } for i in range(len(func_ids))
    ]

    action_func_db.insert_many(documents)


def insert_form_action_functionality(
    func_ids: list,
    state_id: str,
    state_url: str,
    prev_state_id: str,
    form_action_id: str,
    prev_action_id: str,
    form_test_id: str,
    submit_test_id: str,
    form_field_test_ids: list,  # List of field test_ids like ["t15-owner-add-first-name", ...]
    action_depth: int,
    action_type: str = "FORM"
):
    """
    Insert action-functionality mappings for form actions.
    This creates a chain that includes form field interactions.
    
    The test_id will be formatted to include form fields for proper evaluation:
    - Form fields are extracted from test_ids (e.g., "t15" from "t15-owner-add-first-name")
    - Submit button is extracted (e.g., "c31" from "c31-owner-add-submit")
    """
    # Extract prefixes from form field test_ids (e.g., "t15" from "t15-owner-add-first-name")
    form_field_prefixes = []
    for field_id in form_field_test_ids:
        if field_id:
            match = re.match(r'^([cts]\d+)', field_id)
            if match:
                form_field_prefixes.append(match.group(1))
    
    # Extract submit button prefix
    submit_prefix = None
    if submit_test_id:
        match = re.match(r'^([cts]\d+)', submit_test_id)
        if match:
            submit_prefix = match.group(1)
    
    documents = [
        {
            "app": os.getenv("APP_NAME"),
            "url": state_url,
            "state": state_id,
            "prev_state": prev_state_id,
            "action": form_action_id,
            "prev_action": prev_action_id,
            "test_id": submit_test_id,  # Use submit button as the action's test_id
            "form_test_id": form_test_id,  # Store the form's test_id
            "form_fields": form_field_prefixes,  # Store form field prefixes
            "submit_prefix": submit_prefix,  # Store submit button prefix
            "depth": action_depth,
            "type": action_type,
            "rank_score": geometric_score(i + 1),  # 1-indexed: rank 1 = first item
            "func_pointer": str(func_ids[i]),
            "final": True,  # Form submissions are typically final actions
            "should_execute": True
        } for i in range(len(func_ids))
    ]

    action_func_db.insert_many(documents)


# =============================================================================
# Functionality Score Updates
# =============================================================================

def update_functionality_score(prev_state, prev_action, curr_state, curr_action):
    """
    Update functionality scores based on action sequence.

    Score Update Rule (from AUTOE2E paper):
        score(F) += rank(F | A_i, A_{i-1}) - rank(F | A_{i-1})

    Ablation Support (when ABLATION_MODE is on):
      - A5.1: Final only (use only last action's score, no accumulation)
      - A5.2: Simple sum (sum pair scores without differential)
      - A5.3: Maximum (take highest pair score)
    """
    from autoe2e.ablation_integration import (
        ABLATION_MODE,
        get_score_accumulator
    )

    app_name = os.getenv("APP_NAME")

    curr_action_funcs = list(action_func_db.find({
        'app': app_name,
        'state': curr_state.get_id(StateIdEvaluator.BY_ACTIONS),
        'action': curr_action.get_id(),
        'type': 'DOUBLE'
    }))

    prev_action_funcs = list(action_func_db.find({
        'app': app_name,
        'state': prev_state.get_id(StateIdEvaluator.BY_ACTIONS),
        'action': prev_action.get_id(),
        'type': 'SINGLE'
    }))

    func_score_updates = {}

    accumulator = get_score_accumulator() if ABLATION_MODE else None
    if accumulator:
        logger.info(f"[ABLATION] Score accumulation: {accumulator.get_description()}")

    for curr_func in curr_action_funcs:
        corresponding_func_in_prev = list(filter(
            lambda x: x['func_pointer'] == curr_func['func_pointer'],
            prev_action_funcs
        ))
        prev_score = (
            geometric_score(None) if len(corresponding_func_in_prev) == 0
            else corresponding_func_in_prev[0]['rank_score']
        )

        if accumulator:
            diff = accumulator.compute_update(curr_func['rank_score'], prev_score)
        else:
            # Default: differential accumulation (paper formula)
            diff = curr_func['rank_score'] - prev_score

        func_score_updates[curr_func['func_pointer']] = diff

    for _id, diff in func_score_updates.items():
        func_db.update_one(
            filter={
                'app': app_name,
                '_id': ObjectId(_id),
                'final': False
            },
            update={
                '$inc': {
                    'score': diff
                }
            },
            upsert=False
        )


# =============================================================================
# Finality Marking Functions
# =============================================================================

def mark_final_functionalities(curr_state, curr_action):
    """
    Mark functionalities as final/testable based on context analysis.
    Uses LLM to determine if actions represent final testable behavior.
    """
    curr_action_funcs = list(action_func_db.find({
        'app': os.getenv("APP_NAME"),
        'state': curr_state.get_id(StateIdEvaluator.BY_ACTIONS),
        'action': curr_action.get_id(),
    }))
    retreived_funcs = list(func_db.find({
        'app': { '$eq': os.getenv("APP_NAME") },
        '_id': {
            '$in': list(map(lambda x: ObjectId(x['func_pointer']), curr_action_funcs))
        }
    }))

    if len(retreived_funcs) == 0:
        return
    
    # LLM-based finality detection
    try:
        res = sonnet_chain(
            FINALITY_SYSTEM_PROMPT,
            create_finality_user_messages(
                curr_state.context,
                curr_action.element.outerHTML,
                '\n'.join(map(lambda x: x['text'], retreived_funcs))
            )
        )
    except Exception as e:
        logger.error(f"LLM call failed for finality marking: {e}")
        return
    
    # Use robust parsing
    text = get_llm_text(res)
    finality = parse_bool_list(text, len(retreived_funcs))
    
    for i in range(len(finality)):
        if finality[i]:
            func_db.update_one(
                filter={
                    'app': os.getenv("APP_NAME"),
                    '_id': retreived_funcs[i]['_id']
                },
                update={
                    '$set': {
                        'final': True,
                    }
                },
                upsert=False
            )


# =============================================================================
# Critical Action Detection
# =============================================================================

def is_dropdown_toggle(action: Action) -> bool:
    """
    Detect if an action is a dropdown toggle that just reveals/hides menu items.
    These actions don't lead to new meaningful states.
    """
    element_html = action.get_element().outerHTML.lower()
    
    # Common dropdown patterns
    dropdown_indicators = [
        'dropdown-toggle',
        'data-toggle="dropdown"',
        'data-bs-toggle="dropdown"',
        'aria-haspopup="true"',
        'aria-expanded=',
        'class="caret"',
        '<span class="caret">',
    ]
    
    for indicator in dropdown_indicators:
        if indicator in element_html:
            return True
    
    # Check for common dropdown toggle patterns in class names
    if 'dropdown' in element_html and ('toggle' in element_html or 'trigger' in element_html):
        return True
    
    return False


def is_action_critical(action: Action) -> bool:
    """
    Determine if an action is critical (e.g., delete, submit).
    Single LLM call, returns False on failure.
    """
    element_html = action.get_element().outerHTML

    try:
        res = haiku_chain(
            CRITICAL_ACTION_SYSTEM_PROMPT,
            create_simple_user_messages(element_html)
        )
    except Exception as e:
        logger.error(f"LLM call failed for critical action check: {e}")
        return False
    
    # Use robust parsing
    text = get_llm_text(res)
    return parse_bool(text, default=False)


# =============================================================================
# Form Value Generation
# =============================================================================

def _fix_form_values(values: dict, element_html: str) -> dict:
    """
    Post-process form values to fix common LLM errors.
    
    Smaller models often return boolean `true` for all fields instead of
    actual text values. This function detects and fixes such cases by
    generating appropriate default values based on field type.
    """
    if not values:
        return values
    
    # Lowercase html for matching
    html_lower = element_html.lower()
    
    fixed_values = {}
    for key, value in values.items():
        key_lower = key.lower()
        
        # If value is a boolean but field is likely a text input, generate appropriate value
        if isinstance(value, bool):
            # Check if this is a checkbox/radio by looking at the HTML context
            # Look for type="checkbox" or type="radio" near this field's key
            is_checkbox_or_radio = False
            
            # Simple heuristic: check if there's a checkbox/radio input with this id/name
            if f'id="{key}"' in element_html or f"id='{key}'" in element_html:
                # Check surrounding context for type
                idx = element_html.find(f'id="{key}"')
                if idx == -1:
                    idx = element_html.find(f"id='{key}'")
                if idx != -1:
                    context = element_html[max(0, idx-100):idx+100].lower()
                    if 'type="checkbox"' in context or 'type="radio"' in context:
                        is_checkbox_or_radio = True
            
            # If it's truly a checkbox/radio, keep the boolean
            if is_checkbox_or_radio:
                fixed_values[key] = value
                continue
            
            # Otherwise, generate an appropriate text value based on field name
            if 'email' in key_lower:
                fixed_values[key] = 'test@example.com'
            elif 'password' in key_lower or 'pwd' in key_lower:
                fixed_values[key] = 'TestPassword123!'
            elif 'name' in key_lower and 'user' in key_lower:
                fixed_values[key] = 'testuser'
            elif 'first' in key_lower and 'name' in key_lower:
                fixed_values[key] = 'John'
            elif 'last' in key_lower and 'name' in key_lower:
                fixed_values[key] = 'Doe'
            elif 'name' in key_lower:
                fixed_values[key] = 'Test User'
            elif 'phone' in key_lower or 'tel' in key_lower:
                fixed_values[key] = '555-123-4567'
            elif 'address' in key_lower:
                fixed_values[key] = '123 Test Street'
            elif 'city' in key_lower:
                fixed_values[key] = 'Test City'
            elif 'zip' in key_lower or 'postal' in key_lower:
                fixed_values[key] = '12345'
            elif 'country' in key_lower:
                fixed_values[key] = 'United States'
            elif 'date' in key_lower:
                fixed_values[key] = '2024-01-15'
            elif 'url' in key_lower or 'website' in key_lower:
                fixed_values[key] = 'https://example.com'
            elif 'description' in key_lower or 'comment' in key_lower or 'message' in key_lower:
                fixed_values[key] = 'This is a test description.'
            elif 'title' in key_lower:
                fixed_values[key] = 'Test Title'
            elif 'company' in key_lower or 'organization' in key_lower:
                fixed_values[key] = 'Test Company'
            elif 'search' in key_lower or 'query' in key_lower:
                fixed_values[key] = 'test'
            elif 'amount' in key_lower or 'price' in key_lower or 'number' in key_lower:
                fixed_values[key] = '100'
            else:
                # Generic fallback
                fixed_values[key] = 'TestValue123'
            
            logger.warn(f"Fixed boolean value for '{key}': {value} -> '{fixed_values[key]}'")
        else:
            # Keep non-boolean values as-is
            fixed_values[key] = value
    
    return fixed_values


def create_form_filling_values(action: Action):
    """
    Generate test values for form fields.
    Single LLM call, returns empty dict on failure.
    Includes post-processing to fix common LLM errors (e.g., boolean values for text fields).
    """
    element_html = action.get_element().outerHTML

    try:
        res = sonnet_chain(
            FORM_VALUE_SYSTEM_PROMPT,
            create_simple_user_messages(element_html)
        )
    except Exception as e:
        logger.error(f"LLM call failed for form value generation: {e}")
        return {}
    
    # Use robust parsing
    text = get_llm_text(res)
    values = parse_json_object(text)
    
    # Post-process to fix common errors (e.g., boolean values for text fields)
    return _fix_form_values(values, element_html)
