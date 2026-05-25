import json
import os
import time
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from autoe2e.utils import logger

from autoe2e.crawler.config import Config
from autoe2e.browser import get_driver_container
from autoe2e.crawler.crawl_context import CrawlContext
from autoe2e.crawler.state import State
from autoe2e.crawler.action import Action, CandidateActionExtractor


def read_config(config_path: str) -> dict:
    logger.info(f'Reading config from {config_path}')
    
    with open(config_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def load_experiment_yaml(yaml_path: str) -> dict:
    """
    Load a unified experiment YAML and export its values into ``os.environ``.

    Variables already present in the real environment are preserved (so a
    ``.env`` file or shell export still wins). This keeps the rest of the
    codebase unchanged: it can keep reading ``os.getenv("OLLAMA_MODEL")``
    etc. while the YAML acts as a single human-friendly source of truth.

    Returns the parsed YAML dict (also useful for the ablation_config path).
    """
    import yaml  # imported lazily so the dep is optional for non-YAML runs

    logger.info(f'Loading experiment YAML from {yaml_path}')
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}

    def _set(name: str, value):
        if value is None:
            return
        if name in os.environ:
            return
        os.environ[name] = str(value)

    _set('APP_NAME', data.get('app_name'))

    llm = data.get('llm', {}) or {}
    _set('LLM_PROVIDER', llm.get('provider'))

    ollama = llm.get('ollama', {}) or {}
    _set('OLLAMA_BASE_URL', ollama.get('base_url'))
    _set('OLLAMA_MODEL', ollama.get('model'))
    _set('OLLAMA_EMBEDDING_MODEL', ollama.get('embedding_model'))
    _set('OLLAMA_TEMPERATURE', ollama.get('temperature'))
    _set('OLLAMA_NUM_PREDICT', ollama.get('num_predict'))
    _set('OLLAMA_TIMEOUT', ollama.get('timeout'))

    openai_cfg = llm.get('openai', {}) or {}
    _set('OPENAI_MODEL', openai_cfg.get('model'))
    _set('OPENAI_TEMPERATURE', openai_cfg.get('temperature'))
    _set('OPENAI_EMBEDDING_MODEL', openai_cfg.get('embedding_model'))

    embedding = data.get('embedding', {}) or {}
    _set('EMBEDDING_PROVIDER', embedding.get('provider'))

    budget = data.get('budget', {}) or {}
    _set('MAX_RUNTIME_MINUTES', budget.get('max_runtime_minutes'))
    _set('MAX_TOTAL_STATES', budget.get('max_total_states'))
    _set('MAX_ACTIONS_PER_STATE', budget.get('max_actions_per_state'))

    return data


def initialize_driver(config: Config) -> WebDriver:
    logger.info('Initializing driver')
    return get_driver_container(config).get_driver()

def initialize_variables(crawl_context: CrawlContext) -> CrawlContext:
    logger.info('Initializing classes with initial state')
    
    crawl_context.driver.get(crawl_context.config.base_url)

    crawl_context.crawl_queue.reset()
    crawl_context.state_machine.reset()

    actions: list[Action] = CandidateActionExtractor.extract_candidate_actions(crawl_context.driver)
    
    initial_state: State = crawl_context.create_state_from_driver(actions)
    
    crawl_context.crawl_queue.enqueue(initial_state)
    
    return crawl_context
