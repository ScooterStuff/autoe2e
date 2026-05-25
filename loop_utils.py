import os
import re
import random

from dotenv import load_dotenv

from autoe2e.utils import logger

from autoe2e.crawler.crawl_context import CrawlContext
from autoe2e.crawler.state import State, StateIdEvaluator
from autoe2e.crawler.action import Action, CandidateActionExtractor

# Get max states limit from environment
from autoe2e.manual_ndd import (
    VISIT_ONCE,
    NEVER_VISIT
)
from autoe2e.infer_utils import (
    extract_state_context,
    extract_action_functionalities,
    insert_functionalities,
    insert_action_functionality,
    update_functionality_score,
    mark_final_functionalities,
    is_action_critical,
    is_dropdown_toggle,
    create_form_filling_values
)
from autoe2e.mongo_utils import (
    action_func_db,
    func_db
)


load_dotenv()


def get_next_action(crawl_context: CrawlContext):
    # final: all the actions in a chain have been found
    # executable: there is at least one action connected to the functionality that is possible to execute
    # actions might not be executable because they've already been executed once. No redundant action execution.
    highest_funcs = list(func_db.find({
        'app': os.getenv("APP_NAME"),
        'final': False,
        'executable': True
    }).sort({ 'score': -1 }).limit(1))

    # if no function is returned it means all the functionalities have been explored and finalized.
    if len(highest_funcs) == 0:
        return None, None, None

    # randomly choose from one of the highest scorings functionalities.
    highest_func = random.choice(highest_funcs)
    
    logger.info(f'Exploring feature: {highest_func["text"]}')

    connected_actions = list(action_func_db.find({
        'app': os.getenv("APP_NAME"),
        'func_pointer': str(highest_func['_id']),
        'should_execute': True
    }))

    # if no actions are connected (meaning that their should_execute is false) the feature is not executable anymore
    if len(connected_actions) == 0:
        func_db.update_one(
            filter={
                'app': os.getenv("APP_NAME"),
                '_id': highest_func['_id']
            },
            update={
                '$set': {
                    'executable': False
                }
            },
            upsert=False
        )
        return get_next_action(crawl_context)

    # select a random action that has the highest depth, because this is likely closer to finalizing
    max_depth = list(action_func_db.find({
        'app': os.getenv("APP_NAME"),
        'func_pointer': str(highest_func['_id']),
        'should_execute': True
    }).sort({ 'depth': -1 }).limit(1))[0]['depth']
    max_depth_actions = list(action_func_db.find({
        'app': os.getenv("APP_NAME"),
        'func_pointer': str(highest_func['_id']),
        'should_execute': True,
        'depth': max_depth
    }))
    selected_action = random.choice(max_depth_actions)

    state_id, action_id = selected_action['state'], selected_action['action']

    state = crawl_context.state_machine.state_graph.get_state(state_id)
    action = list(filter(lambda x: x.get_id() == action_id, state.get_actions()))[0]

    return str(highest_func['_id']), state, action


def flag_action_to_stop_execution(state: State, action: Action, feature_id: str | None = None):
    if feature_id is None:
        action_func_db.update_many(
            filter={
                'app': os.getenv("APP_NAME"),
                'state': state.get_id(StateIdEvaluator.BY_ACTIONS),
                'action': action.get_id()
            },
            update={
                '$set': {
                    'should_execute': False
                }
            },
            upsert=False
        )
    else:
        action_func_db.update_many(
            filter={
                'app': os.getenv("APP_NAME"),
                'state': state.get_id(StateIdEvaluator.BY_ACTIONS),
                'action': action.get_id(),
                'func_pointer': feature_id
            },
            update={
                '$set': {
                    'should_execute': False
                }
            },
            upsert=False
        )


def is_state_in_graph(crawl_context: CrawlContext, state: State, triggered_by_action: Action = None) -> bool:
    """
    Check if a state already exists in the graph.
    Uses multiple strategies to detect duplicates:
    1. Exact action hash match
    2. URL-based match with core action comparison (merges new actions)
    3. State equality (DOM or action IDs)
    
    When a semantically similar state is found (same URL, similar actions),
    we MERGE the new actions into the existing state AND re-queue it
    so the new actions get explored.
    
    EXCEPTION: If the state was reached via a dropdown toggle, we DON'T merge
    because we need to preserve the action chain (dropdown → item).
    """
    # Exact match by action hash
    if state.get_id(StateIdEvaluator.BY_ACTIONS) in crawl_context.state_machine.state_graph.states:
        return True
    
    # Check by State equality (compares DOM or action IDs)
    if state in crawl_context.state_machine.state_graph.states.values():
        return True
    
    # If the state was reached by clicking a dropdown toggle, DON'T merge
    # This preserves the action chain: dropdown_toggle → dropdown_item
    if triggered_by_action and is_dropdown_toggle(triggered_by_action):
        logger.info(f'State reached via dropdown toggle, not merging to preserve chain')
        return False
    
    # URL-based deduplication: if same URL with similar core actions, 
    # merge new actions into existing state and RE-QUEUE it
    for existing_state in crawl_context.state_machine.state_graph.states.values():
        if existing_state.url == state.url:
            if _are_states_semantically_similar(existing_state, state):
                # MERGE: Add any new actions from the new state to the existing state
                existing_action_ids = {a.get_id() for a in existing_state.get_actions()}
                new_actions_to_add = []
                
                for action in state.get_actions():
                    if action.get_id() not in existing_action_ids:
                        new_actions_to_add.append(action)
                        action.set_parent_state_id(existing_state.get_id())
                
                if new_actions_to_add:
                    existing_state.extend_actions(new_actions_to_add)
                    # RE-QUEUE the existing state so the new actions get processed
                    # The old actions already have should_execute=False so they won't run again
                    crawl_context.crawl_queue.enqueue(existing_state)
                    logger.info(f'Merged {len(new_actions_to_add)} new actions into existing state at {state.url} and re-queued')
                else:
                    logger.info(f'State at {state.url} is semantically similar, no new actions to merge')
                
                return True
    
    return False


def _get_core_action_ids(state: State) -> set:
    """
    Get action IDs that represent core page functionality.
    Filters out actions that are likely from dropdown menus or transient UI elements.
    """
    core_actions = set()
    for action in state.get_actions():
        action_id = action.get_id()
        # Include actions that are not nested deep (dropdown items often have deeper XPaths)
        # Also include form actions which are always meaningful
        if action.get_type().get_value() == 'form':
            core_actions.add(action_id)
        elif action_id.count('/') <= 8:  # Not too deeply nested
            core_actions.add(action_id)
    return core_actions


def _are_states_semantically_similar(state1: State, state2: State, threshold: float = 0.7) -> bool:
    """
    Check if two states are semantically similar based on their core actions.
    Two states are similar if they share a high percentage of core actions.
    
    IMPORTANT: States with different form elements are NOT similar - 
    a form page is a different state from a list page, even if URL is same.
    """
    core1 = _get_core_action_ids(state1)
    core2 = _get_core_action_ids(state2)
    
    if not core1 or not core2:
        return False
    
    # Check if one state has form inputs that the other doesn't
    # This indicates a real page transition (list → form), not just UI toggle
    form_inputs1 = _get_form_input_actions(state1)
    form_inputs2 = _get_form_input_actions(state2)
    
    # If one has form inputs and the other doesn't, they're different states
    if form_inputs1 and not form_inputs2:
        logger.info(f'States differ: state2 has no form inputs, state1 has {len(form_inputs1)}')
        return False
    if form_inputs2 and not form_inputs1:
        logger.info(f'States differ: state1 has no form inputs, state2 has {len(form_inputs2)}')
        return False
    
    # If both have form inputs but they're different, they're different states
    if form_inputs1 and form_inputs2:
        if form_inputs1 != form_inputs2:
            logger.info(f'States differ: different form inputs')
            return False
    
    # Calculate Jaccard similarity
    intersection = len(core1 & core2)
    union = len(core1 | core2)
    
    if union == 0:
        return False
    
    similarity = intersection / union
    return similarity >= threshold


def _get_form_input_actions(state: State) -> set:
    """
    Get action IDs that are form-related (inputs, forms, textareas, selects).
    These indicate a form page which is distinct from list/nav pages.
    """
    form_inputs = set()
    for action in state.get_actions():
        element_html = action.get_element().outerHTML.lower()
        
        # Check for form input elements
        if any(tag in element_html for tag in ['<input', '<textarea', '<select', '<form']):
            # Exclude search inputs and filters which are common across pages
            if 'search' not in element_html and 'filter' not in element_html:
                form_inputs.add(action.get_id())
        
        # Check for form action type
        if action.get_type().get_value() == 'form':
            form_inputs.add(action.get_id())
    
    return form_inputs


def explore_connected_states(crawl_context: CrawlContext, state: State):
    crawl_context.state_machine.set_current_state(state)
    
    logger.info(f'state: {state.get_id(StateIdEvaluator.BY_ACTIONS)}')
    
    actions: list[Action] = state.get_actions()

    for action in actions:        
        is_critical = is_action_critical(action)

        if is_critical:
            action.set_should_execute(False)
            flag_action_to_stop_execution(state, action)
            continue

        logger.info(f"Executing action {action.element.outerHTML}")
        
        if action.get_type().get_value() == 'form' and not action.has_params():
            values = create_form_filling_values(action)
            action.set_params(values)

        crawl_context.load_state(crawl_context.state_machine.get_current_state())
        action.execute(crawl_context.driver)

        new_actions: list[Action] = CandidateActionExtractor.extract_candidate_actions(crawl_context.driver)
        new_state: State = crawl_context.create_state_from_driver(new_actions)

        if is_state_in_graph(crawl_context, new_state):
            action.set_should_execute(False)
            flag_action_to_stop_execution(state, action)
            continue
        
        # Check state limit BEFORE adding new state
        current_total_states = len(crawl_context.state_machine.state_graph.states)
        if MAX_TOTAL_STATES > 0 and current_total_states >= MAX_TOTAL_STATES:
            logger.info(f'STATE LIMIT REACHED ({MAX_TOTAL_STATES}): Skipping new state addition')
            action.set_should_execute(False)
            flag_action_to_stop_execution(state, action)
            continue
        
        logger.info(f'Adding state: {new_state.get_id(StateIdEvaluator.BY_ACTIONS)}')
        crawl_context.state_machine.add_state_from_current_state(new_state, action)


def extract_state_action_features(crawl_context: CrawlContext, state: State):
    crawl_context.state_machine.set_current_state(state)
    crawl_context.load_state(crawl_context.state_machine.get_current_state())
    
    logger.info('Extracting state context using LLM')
    
    state_context = extract_state_context(
        crawl_context,
        state,
        state.crawl_path.get_state(-1) if len(state.crawl_path) > 0 else None,
        state.crawl_path.get_action(-1) if len(state.crawl_path) > 0 else None,
    )
    state.set_context(state_context)

    actions: list[Action] = list(filter(lambda a: a.get_should_execute(), state.get_actions()))

    for action in actions:
        logger.info(f'Extracting action scenarios: {action.element.outerHTML}')

        functionalities = extract_action_functionalities(state, action)
        if len(functionalities) != 0:
            functionality_ids = insert_functionalities(functionalities)
            insert_action_functionality(
                func_ids=functionality_ids,
                state_id=state.get_id(StateIdEvaluator.BY_ACTIONS),
                state_url=state.url,
                prev_state_id=state.crawl_path.get_state(-1).get_id(StateIdEvaluator.BY_ACTIONS) if len(state.crawl_path) > 0 else None,
                action_id=action.get_id(),
                prev_action_id=state.crawl_path.get_action(-1).get_id() if len(state.crawl_path) > 0 else None,
                action_test_id=action.element.test_id,
                action_depth=len(state.crawl_path),
                action_type="SINGLE"
            )
    
        if len(state.crawl_path) > 0:
            logger.info('Extracting double action scenarios')
            functionalities = extract_action_functionalities(state, action, state.crawl_path.get_action(-1))
            if len(functionalities) != 0:
                functionality_ids = insert_functionalities(functionalities)
                insert_action_functionality(
                    func_ids=functionality_ids,
                    state_id=state.get_id(StateIdEvaluator.BY_ACTIONS),
                    state_url=state.url,
                    prev_state_id=state.crawl_path.get_state(-1).get_id(StateIdEvaluator.BY_ACTIONS) if len(state.crawl_path) > 0 else None,
                    action_id=action.get_id(),
                    prev_action_id=state.crawl_path.get_action(-1).get_id() if len(state.crawl_path) > 0 else None,
                    action_test_id=action.element.test_id,
                    action_depth=len(state.crawl_path),
                    action_type="DOUBLE"
                )
            
            logger.info('Updating action scores')

            update_functionality_score(
                state.crawl_path.get_state(-1),
                state.crawl_path.get_action(-1),
                state,
                action
            )

            logger.info('Action scores updated')
        
        logger.info('Marking final functionalities')
    
        mark_final_functionalities(state, action)

        logger.info('Final actions marked')


def is_match(text, pattern):
    match = re.search(pattern, text)
    return match is not None


def is_visit_forbidden(state: State, visit_counter: dict[str, int]) -> bool:
    url = state.url

    for never in NEVER_VISIT:
        if is_match(url, never):
            return visit_counter, True

    for once in VISIT_ONCE:
        if is_match(url, once) and once in visit_counter:
            return visit_counter, True
        elif is_match(url, once):
            visit_counter[once] = 1

    return visit_counter, False
