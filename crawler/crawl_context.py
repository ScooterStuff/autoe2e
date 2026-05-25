from typing import Any, Self

from selenium.webdriver.remote.webdriver import WebDriver

from autoe2e.crawler.config import Config
from autoe2e.crawler.state import State, StateMachine
from autoe2e.crawler.action import Action
from autoe2e.utils import Queue


class CrawlContext:
    def __init__(self):
        # change to config later
        self.config: Config | None = None
        self.driver: WebDriver | None = None
        self.crawl_queue: Queue[State] = Queue()
        self.state_machine: StateMachine = StateMachine()
        self.temp_vars: dict = {}
    
    
    def set_config(self, config: Config) -> Self:
        if self.config is not None:
            raise ValueError('config is already set')
        self.config = config
        return self
    
    
    def set_driver(self, driver: WebDriver) -> Self:
        if self.driver is not None:
            raise ValueError('driver is already set')
        self.driver = driver
        return self

    
    def load_state(self, state: State) -> None:
        import time
        self.driver.get(self.config.base_url)
        time.sleep(0.5)  # Wait for initial page load
        
        actions = state.crawl_path.get_actions()
        failed_actions = 0
        
        for i, action in enumerate(actions):
            try:
                action.execute(self.driver)
                time.sleep(0.2)  # Small delay between actions
            except Exception as e:
                failed_actions += 1
                print(f"Warning: Action {i+1}/{len(actions)} failed during state replay: {e}")
                
                # If we failed more than half the actions, this state is probably unreachable
                if failed_actions > len(actions) / 2:
                    print(f"Too many failed actions ({failed_actions}), state may be unreachable")
                    raise RuntimeError(f"State unreachable - {failed_actions} of {len(actions)} actions failed")
                
                # Try to recover by going back to base URL and continuing
                print(f"Attempting to continue...")
                try:
                    # Check if we're still on a valid page
                    current_url = self.driver.current_url
                    if not current_url or 'about:blank' in current_url:
                        self.driver.get(self.config.base_url)
                        time.sleep(0.3)
                except:
                    self.driver.get(self.config.base_url)
                    time.sleep(0.3)
    
    
    def create_state_from_driver(self, actions: list[Action]) -> State:
        state: State = State(
            url=self.driver.current_url,
            dom=self.driver.page_source,
            actions=actions
        )
        for action in state.get_actions():
            action.set_parent_state_id(state.get_id())
        return state
    
    
    def set_temp_var(self, key: str, value: Any) -> Self:
        self.temp_vars[key] = value
        return self
    
    
    def get_temp_var(self, key: str) -> Any:
        return self.temp_vars.get(key, None)
    
    
    def reset_temp_var(self, key: str) -> Self:
        self.temp_vars.pop(key, None)
        return self
    
    
    def clear_temp_vars(self) -> Self:
        self.temp_vars = {}
        return self
