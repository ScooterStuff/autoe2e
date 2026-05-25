import os
import sys
import time
import json
import argparse
from datetime import datetime, timedelta

# -----------------------------------------------------------------------------
# Pre-parse --experiment-config so its values reach os.environ *before*
# autoe2e.llm_api_call is imported (the chat backend is built at import time).
# -----------------------------------------------------------------------------
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument('--experiment-config', type=str, default=None)
_pre_args, _ = _pre_parser.parse_known_args()
if _pre_args.experiment_config:
    # Lazy import so we don't trigger heavy autoe2e package initialisation yet.
    from autoe2e.init_utils import load_experiment_yaml
    load_experiment_yaml(_pre_args.experiment_config)

from autoe2e.utils import *
from autoe2e.init_utils import *
from autoe2e.infer_utils import *
from autoe2e.loop_utils import *
from autoe2e.mongo_utils import *
from autoe2e.manual_ndd import *
from autoe2e.llm_api_call import llm_stats, OLLAMA_MODEL, OLLAMA_EMBEDDING_MODEL, LLM_PROVIDER
from autoe2e.ablation_integration import (
    init_ablation_components, 
    ABLATION_MODE,
    get_ablation_id,
    get_prompt_manager
)


# =============================================================================
# Argument Parsing (for ablation study integration)
# =============================================================================
parser = argparse.ArgumentParser(description='AutoE2E Crawler')
parser.add_argument('--config', type=str, default=None,
                    help='Path to application configuration file')
parser.add_argument('--experiment-config', type=str, default=None,
                    help='Path to unified experiment YAML (configs/experiment.yaml)')
parser.add_argument('--ablation-config', type=str, default=None,
                    help='Path to ablation configuration JSON file (enables ablation mode)')
parser.add_argument('--output-dir', type=str, default=None,
                    help='Output directory for results')
parser.add_argument('--headless', action='store_true',
                    help='Run browser in headless mode')
args, _ = parser.parse_known_args()

# Initialize ablation mode if config provided
if args.ablation_config:
    init_ablation_components(args.ablation_config)

APP_NAME = os.getenv('APP_NAME', 'PETCLINIC')
MAX_RUNTIME_MINUTES = int(os.getenv('MAX_RUNTIME_MINUTES', '0'))  # Default 60 minutes, 0 = no limit
MAX_TOTAL_STATES = int(os.getenv('MAX_TOTAL_STATES', '0'))  # Default 30 states, 0 = no limit
MAX_ACTIONS_PER_STATE = int(os.getenv('MAX_ACTIONS_PER_STATE', '0'))  # Limit actions per state to prevent explosion

# =============================================================================
# Startup Banner - Show configured limits
# =============================================================================
print("=" * 70)
print(f"AUTOE2E CRAWLER - {APP_NAME}")
print("=" * 70)
print(f"  Time Limit:      {MAX_RUNTIME_MINUTES} minutes" if MAX_RUNTIME_MINUTES > 0 else "  Time Limit:      No limit")
print(f"  State Limit:     {MAX_TOTAL_STATES} states" if MAX_TOTAL_STATES > 0 else "  State Limit:     No limit")
print(f"  Actions/State:   {MAX_ACTIONS_PER_STATE}" if MAX_ACTIONS_PER_STATE > 0 else "  Actions/State:   No limit")
print("=" * 70)


action_func_db.delete_many({ 'app': APP_NAME })
func_db.delete_many({ 'app': APP_NAME })


crawl_context: CrawlContext = CrawlContext()
crawl_context = crawl_context.set_temp_var('config_path', f'./configs/{APP_NAME}.json')

config: dict = read_config(config_path=crawl_context.temp_vars.get('config_path', None))
config_obj: Config = Config.from_dict(config)

if config_obj.base_url is None:
    raise ValueError('base_url is required in config')

crawl_context = crawl_context.set_config(config_obj)

driver = initialize_driver(config_obj)
crawl_context = crawl_context.set_driver(driver)

crawl_context = initialize_variables(crawl_context)


LOOP_COUNTER = 0

# =============================================================================
# Progress Tracking Variables
# =============================================================================
start_time = datetime.now()
states_processed = 0
actions_processed = 0
state_times = []  # Track time per state for estimation
crawl_error = None  # Track if crawl crashed
processed_state_ids = set()  # Track states we've already fully processed


try:
    while len(crawl_context.crawl_queue) > 0:
        # Check time limit
        if MAX_RUNTIME_MINUTES > 0:
            elapsed_minutes = (datetime.now() - start_time).total_seconds() / 60
            if elapsed_minutes >= MAX_RUNTIME_MINUTES:
                logger.info("=" * 70)
                logger.info(f"TIME LIMIT REACHED: {MAX_RUNTIME_MINUTES} minutes")
                logger.info("Stopping crawl gracefully...")
                logger.info("=" * 70)
                break
        
        # Check total state limit
        total_states_discovered = len(crawl_context.state_machine.state_graph.states)
        if MAX_TOTAL_STATES > 0 and total_states_discovered >= MAX_TOTAL_STATES:
            logger.info("=" * 70)
            logger.info(f"TOTAL STATE LIMIT REACHED: {MAX_TOTAL_STATES} states discovered")
            logger.info("Stopping crawl gracefully...")
            logger.info("=" * 70)
            break
        
        state_start_time = time.time()
        
        state: State = crawl_context.crawl_queue.dequeue()
        state_id = state.get_id(StateIdEvaluator.BY_ACTIONS)
        
        # Skip states we've already fully processed (prevents re-visiting via re-queuing)
        if state_id in processed_state_ids:
            logger.info(f"Skipping already processed state: {state_id}")
            continue
        
        states_processed += 1
        
        # Progress logging
        queue_size = len(crawl_context.crawl_queue)
        elapsed = datetime.now() - start_time
        
        # Estimate remaining time based on average state processing time
        if state_times:
            avg_time_per_state = sum(state_times) / len(state_times)
            estimated_remaining_secs = avg_time_per_state * queue_size
            estimated_remaining = str(timedelta(seconds=int(estimated_remaining_secs)))
        else:
            estimated_remaining = "calculating..."
        
        logger.info("=" * 70)
        logger.info(f"PROGRESS: States processed: {states_processed} | Queue: {queue_size} | Discovered: {total_states_discovered}")
        logger.info(f"TIMING: Elapsed: {elapsed} | Est. remaining: {estimated_remaining}")
        logger.info(f"LLM STATS: Calls: {llm_stats.total_calls} | Tokens: ~{llm_stats.total_input_tokens + llm_stats.total_output_tokens}")
        logger.info("=" * 70)
        
        logger.info(f"Visiting state {state.get_id(StateIdEvaluator.BY_ACTIONS)}")
        crawl_context.state_machine.set_current_state(state)
        
        current_state: State = crawl_context.state_machine.get_current_state()
        current_actions: list[Action] = current_state.get_actions()
        
        # Limit actions per state to prevent BFS explosion
        if MAX_ACTIONS_PER_STATE > 0 and len(current_actions) > MAX_ACTIONS_PER_STATE:
            logger.info(f"Limiting actions from {len(current_actions)} to {MAX_ACTIONS_PER_STATE}")
            current_actions = current_actions[:MAX_ACTIONS_PER_STATE]
        
        logger.info(f"Processing {len(current_actions)} actions in current state")

        crawl_context.load_state(current_state)

        logger.info('Extracting state context using LLM')

        state_context = extract_state_context(
            crawl_context,
            current_state,
            current_state.crawl_path.get_state(-1) if len(current_state.crawl_path) > 0 else None,
            current_state.crawl_path.get_action(-1) if len(current_state.crawl_path) > 0 else None,
        )
        current_state.set_context(state_context)

        for action in current_actions:
            # Check time limit inside action loop for more responsive timeout
            if MAX_RUNTIME_MINUTES > 0:
                elapsed_minutes = (datetime.now() - start_time).total_seconds() / 60
                if elapsed_minutes >= MAX_RUNTIME_MINUTES:
                    logger.info("=" * 70)
                    logger.info(f"TIME LIMIT REACHED: {MAX_RUNTIME_MINUTES} minutes (during action processing)")
                    logger.info("Stopping crawl gracefully...")
                    logger.info("=" * 70)
                    raise TimeoutError("Time limit reached")
            
            LOOP_COUNTER += 1
            
            logger.info(f'Executing action {action.element.outerHTML}')
            
            try:
                # This variable is true unless an action leads to near-duplicate state.
                # Using this variable we can extract the functionality if even if the action is critical and we wouldn't execute it.
                should_extract_func = True
                form_field_test_ids = []  # Track form fields for this action
                submit_button_test_id = None  # Track submit button

                is_critical = is_action_critical(action)
                
                # Skip recording if this is a repeated navigation action (e.g., clicking c5 when already on c5)
                # This reduces noise from clicking the same navbar item twice
                is_repeated_nav = False
                if len(current_state.crawl_path) > 0:
                    prev_action = current_state.crawl_path.get_action(-1)
                    prev_test_id = prev_action.element.test_id if prev_action and prev_action.element else None
                    curr_test_id = action.element.test_id if action.element else None
                    if prev_test_id and curr_test_id and prev_test_id == curr_test_id:
                        is_repeated_nav = True
                        logger.info(f"Skipping repeated action: {curr_test_id}")
                
                if not is_critical:
                    if action.get_type().get_value() == 'form':
                        values = create_form_filling_values(action)
                        action.set_params(values)
                        logger.info(f'Form values generated: {values}')
                    
                    action.execute(crawl_context.driver)
                    
                    # After executing a form action, capture the filled fields
                    if action.get_type().get_value() == 'form':
                        form_field_test_ids = action.get_filled_field_test_ids()
                        submit_button_test_id = action.get_submit_button_test_id()
                        logger.info(f'Form fields filled: {form_field_test_ids}')
                        logger.info(f'Submit button: {submit_button_test_id}')

                    new_actions = []

                    for i in range(10):
                        try:
                            new_actions: list[Action] = CandidateActionExtractor.extract_candidate_actions(crawl_context.driver)
                            break
                        except:
                            time.sleep(0.1)

                    if len(new_actions) == 0:
                        logger.warn("No new actions found after executing action, skipping...")
                        # Even if no new actions, still track form interactions
                        if action.get_type().get_value() == 'form' and form_field_test_ids:
                            should_extract_func = True
                        else:
                            continue
                    
                    new_state: State = crawl_context.create_state_from_driver(new_actions)
                    
                    if not is_state_in_graph(crawl_context, new_state, triggered_by_action=action):
                        # Check state limit before adding new state
                        current_total_states = len(crawl_context.state_machine.state_graph.states)
                        if MAX_TOTAL_STATES > 0 and current_total_states >= MAX_TOTAL_STATES:
                            logger.info(f"State limit reached ({MAX_TOTAL_STATES}), not adding new state")
                        else:
                            print('Adding state', new_state.get_id(StateIdEvaluator.BY_ACTIONS))
                            crawl_context.crawl_queue.enqueue(new_state)
                            crawl_context.state_machine.add_state_from_current_state(new_state, action)
                    else:
                        # State is duplicate, but we STILL need to record the action that led here
                        # This is critical for maintaining the action chain in the database
                        # Without this, child actions will have broken prev_action references
                        should_extract_func = True  # Always record the action for chain integrity
                        logger.info(f"State is duplicate, but recording action for chain integrity")
                
                # Don't record repeated navigation actions (e.g., c5 -> c5) as they add noise
                if is_repeated_nav:
                    should_extract_func = False

                if should_extract_func:
                    logger.info(f'Extracting action scenarios: {action.element.outerHTML}')
                    
                    # Determine which prompt types to use based on ablation config
                    # A2.1: single_action only, A2.2: action_pair only, A2.3: merged
                    has_prev = len(current_state.crawl_path) > 0
                    prompt_types = get_prompt_manager().get_prompt_types(has_prev) if ABLATION_MODE and get_prompt_manager() else (["single", "double"] if has_prev else ["single"])
                    
                    if ABLATION_MODE:
                        logger.info(f'[ABLATION] Prompt types: {prompt_types}')
                    
                    # Extract SINGLE action functionalities (if enabled by prompting strategy)
                    if "single" in prompt_types:
                        functionalities = extract_action_functionalities(current_state, action)
                        if len(functionalities) != 0:
                            functionality_ids = insert_functionalities(functionalities)
                            
                            # For form actions, use the special form insertion
                            if action.get_type().get_value() == 'form' and form_field_test_ids:
                                insert_form_action_functionality(
                                    func_ids=functionality_ids,
                                    state_id=state.get_id(StateIdEvaluator.BY_ACTIONS),
                                    state_url=state.url,
                                    prev_state_id=state.crawl_path.get_state(-1).get_id(StateIdEvaluator.BY_ACTIONS) if len(state.crawl_path) > 0 else None,
                                    form_action_id=action.get_id(),
                                    prev_action_id=state.crawl_path.get_action(-1).get_id() if len(state.crawl_path) > 0 else None,
                                    form_test_id=action.element.test_id,
                                    submit_test_id=submit_button_test_id,
                                    form_field_test_ids=form_field_test_ids,
                                    action_depth=len(state.crawl_path),
                                    action_type="FORM"
                                )
                            else:
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
                    
                    # Extract DOUBLE action functionalities (if enabled and has previous action)
                    if "double" in prompt_types and len(current_state.crawl_path) > 0:
                        logger.info('Extracting double action scenarios')
                        functionalities = extract_action_functionalities(current_state, action, current_state.crawl_path.get_action(-1))
                        if len(functionalities) != 0:
                            functionality_ids = insert_functionalities(functionalities)
                            
                            # For form actions, use the special form insertion
                            if action.get_type().get_value() == 'form' and form_field_test_ids:
                                insert_form_action_functionality(
                                    func_ids=functionality_ids,
                                    state_id=state.get_id(StateIdEvaluator.BY_ACTIONS),
                                    state_url=state.url,
                                    prev_state_id=state.crawl_path.get_state(-1).get_id(StateIdEvaluator.BY_ACTIONS) if len(state.crawl_path) > 0 else None,
                                    form_action_id=action.get_id(),
                                    prev_action_id=state.crawl_path.get_action(-1).get_id() if len(state.crawl_path) > 0 else None,
                                    form_test_id=action.element.test_id,
                                    submit_test_id=submit_button_test_id,
                                    form_field_test_ids=form_field_test_ids,
                                    action_depth=len(state.crawl_path),
                                    action_type="FORM_DOUBLE"
                                )
                            else:
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
                        current_state.crawl_path.get_state(-1),
                        current_state.crawl_path.get_action(-1),
                        current_state,
                        action
                    )

                    logger.info('Action scores updated')

                logger.info('Marking final functionalities')

                mark_final_functionalities(current_state, action)

                logger.info('Final actions marked')
            
                crawl_context.load_state(crawl_context.state_machine.get_current_state())

            except Exception as action_error:
                from selenium.common.exceptions import InvalidSessionIdException, WebDriverException
                
                # Check if this is a fatal session error (browser died)
                error_str = str(action_error).lower()
                is_session_dead = (
                    isinstance(action_error, InvalidSessionIdException) or
                    'invalid session id' in error_str or
                    'session deleted' in error_str or
                    'browser session died' in error_str or
                    'browser session is dead' in error_str
                )
                
                if is_session_dead:
                    logger.error("=" * 70)
                    logger.error("FATAL: Browser session has died!")
                    logger.error(f"Error: {action_error}")
                    logger.error("Cannot continue crawling without a valid browser session.")
                    logger.error("=" * 70)
                    # Re-raise to trigger clean shutdown with progress saving
                    raise RuntimeError(f"Browser session died: {action_error}")
                
                logger.warn(f"Action failed: {action_error}")
                logger.warn(f"Skipping action and continuing with next...")
                # Try to recover by reloading current state
                try:
                    crawl_context.load_state(crawl_context.state_machine.get_current_state())
                except Exception as reload_error:
                    # Check if reload also failed due to dead session
                    reload_error_str = str(reload_error).lower()
                    if 'invalid session id' in reload_error_str or 'browser session' in reload_error_str:
                        logger.error("Browser session is completely dead, stopping crawl.")
                        raise RuntimeError(f"Browser session died during recovery: {reload_error}")
                    pass  # If reload fails for other reasons, just continue

            logger.info("")
        
        # Track state processing time for estimation
        state_end_time = time.time()
        state_times.append(state_end_time - state_start_time)
        
        # Mark this state as fully processed (prevents re-processing if re-queued)
        processed_state_ids.add(state_id)
        
        # Keep only last 20 times for rolling average
        if len(state_times) > 20:
            state_times = state_times[-20:]

except TimeoutError as te:
    # Clean timeout - not an error, just reached limit
    logger.info("=" * 70)
    logger.info(f"TIMEOUT: {te}")
    logger.info("Crawl stopped due to time limit (this is expected behavior)")
    logger.info("=" * 70)

except Exception as e:
    crawl_error = str(e)
    logger.error("=" * 70)
    logger.error(f"CRAWL ERROR: {e}")
    logger.error("Saving progress before exit...")
    logger.error("=" * 70)
    import traceback
    traceback.print_exc()

# =============================================================================
# Final Summary
# =============================================================================
end_time = datetime.now()
total_duration = end_time - start_time

logger.info("=" * 70)
logger.info("CRAWL COMPLETE")
logger.info(f"Total states: {len(crawl_context.state_machine.state_graph.states)}")
logger.info(f"Total actions processed: {LOOP_COUNTER}")
logger.info(f"Total time: {total_duration}")
logger.info(f"LLM calls: {llm_stats.total_calls}")
logger.info(f"Estimated tokens: {llm_stats.total_input_tokens + llm_stats.total_output_tokens}")
logger.info("=" * 70)

crawl_context.driver.quit()


states_converted = {}

for state_id, state_obj in crawl_context.state_machine.state_graph.states.items():
    states_converted[state_id] = {
        'url': state_obj.url,
        'context': state_obj.context,
        'actions': [{
                'type': a.action_type.get_value(),
                'id': a.element.get_id(),
                'outerHTML': clean_children_html(a.element.outerHTML),
                'testId': a.element.test_id
            } for a in state_obj.get_actions()
        ],
        'prev_state': state_obj.crawl_path.get_state(-1).get_id() if len(state.crawl_path) > 0 else None,
        'prev_action': state_obj.crawl_path.get_action(-1).get_id() if len(state.crawl_path) > 0 else None,
    }


adj_list_converted = {}

for state_id, neighbor_list in crawl_context.state_machine.state_graph.adjacency_list.items():
    adj_list_converted[state_id] = {}

    for action_obj, n_state_id in neighbor_list.items():
        adj_list_converted[state_id][action_obj.get_id()] = n_state_id


# Ensure the report directory exists
os.makedirs('./report', exist_ok=True)

json.dump(
    {
        'nodes': states_converted,
        'edges': adj_list_converted
    },
    open(f'./report/{APP_NAME}.json', 'w+')
)

# =============================================================================
# Save Run Summary JSON
# =============================================================================
final_states = len(crawl_context.state_machine.state_graph.states)
stopped_by_state_limit = MAX_TOTAL_STATES > 0 and final_states >= MAX_TOTAL_STATES

run_summary = {
    'app_name': APP_NAME,
    'run_timestamp': start_time.isoformat(),
    'completion_timestamp': end_time.isoformat(),
    'duration': {
        'total_seconds': total_duration.total_seconds(),
        'formatted': str(total_duration)
    },
    'crawl_stats': {
        'total_states': final_states,
        'total_actions_processed': LOOP_COUNTER,
        'states_processed': states_processed,
        'remaining_queue_size': len(crawl_context.crawl_queue),
        'completed': len(crawl_context.crawl_queue) == 0 and crawl_error is None and not stopped_by_state_limit,
        'stopped_by_time_limit': MAX_RUNTIME_MINUTES > 0 and total_duration.total_seconds() / 60 >= MAX_RUNTIME_MINUTES,
        'stopped_by_state_limit': stopped_by_state_limit,
        'error': crawl_error,
    },
    'llm_stats': llm_stats.get_summary(),
    'model_config': {
        'provider': LLM_PROVIDER,
        'chat_model': os.getenv('OPENAI_MODEL', 'gpt-4o') if LLM_PROVIDER == 'openai' else OLLAMA_MODEL,
        'embedding_model': OLLAMA_EMBEDDING_MODEL,
    },
    'config': {
        'max_runtime_minutes': MAX_RUNTIME_MINUTES if MAX_RUNTIME_MINUTES > 0 else None,
        'max_total_states': MAX_TOTAL_STATES if MAX_TOTAL_STATES > 0 else None,
        'max_actions_per_state': MAX_ACTIONS_PER_STATE if MAX_ACTIONS_PER_STATE > 0 else None,
    }
}

summary_filename = f'./report/{APP_NAME}_run_summary_{start_time.strftime("%Y%m%d_%H%M%S")}.json'
with open(summary_filename, 'w') as f:
    json.dump(run_summary, f, indent=2)

logger.info(f"Run summary saved to: {summary_filename}")
logger.info(f"To evaluate, run: python evaluate_autoe2e.py {APP_NAME}")
