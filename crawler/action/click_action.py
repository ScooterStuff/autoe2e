from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By

from autoe2e.crawler.action.action import Action, ActionType
from autoe2e.crawler.action.element import Element
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException, ElementClickInterceptedException, ElementNotInteractableException
import time


class ClickActionType(ActionType):
    def __init__(self):
        super().__init__('click')


class ClickAction(Action):
    def __init__(self, element: Element):
        super().__init__(element, action_type=ClickActionType())
    
    
    def _try_open_parent_dropdown(self, driver: WebDriver) -> bool:
        """
        If this element is inside a dropdown menu, try to open the parent dropdown first.
        Returns True if a dropdown was opened, False otherwise.
        """
        try:
            element = self.element.get(driver)
            
            # Check if element is displayed - if yes, no need to open dropdown
            if element.is_displayed():
                return False
            
            # Look for parent dropdown toggle
            # Common patterns: parent <li> with class 'dropdown', parent has 'dropdown-toggle'
            xpath = self.element.get_id()
            
            # Try to find dropdown toggle in parent hierarchy
            # For Bootstrap-style dropdowns: //li[contains(@class,'dropdown')]/a[contains(@class,'dropdown-toggle')]
            parent_li = None
            try:
                # Navigate up to find parent li with dropdown
                element = driver.find_element(By.XPATH, xpath)
                parent = element.find_element(By.XPATH, "./ancestor::li[contains(@class,'dropdown')]")
                if parent:
                    parent_li = parent
            except:
                pass
            
            if parent_li:
                try:
                    # Find the dropdown toggle within this li
                    toggle = parent_li.find_element(By.CSS_SELECTOR, ".dropdown-toggle, [data-toggle='dropdown'], [data-bs-toggle='dropdown']")
                    if toggle:
                        ActionChains(driver).move_to_element(toggle).click(toggle).perform()
                        time.sleep(0.3)  # Wait for dropdown to open
                        print(f"Opened parent dropdown to access element")
                        return True
                except:
                    pass
            
            # Alternative: try clicking parent element with 'dropdown-toggle' class
            try:
                # Get parent elements and check for dropdown toggle
                parent_xpath = "/".join(xpath.split("/")[:-1])  # Go up one level
                if parent_xpath:
                    parent_elements = driver.find_elements(By.XPATH, f"{parent_xpath}/preceding-sibling::a[contains(@class,'dropdown')]")
                    if not parent_elements:
                        parent_elements = driver.find_elements(By.XPATH, f"{parent_xpath}/../a[contains(@class,'dropdown')]")
                    for pe in parent_elements:
                        if 'dropdown' in pe.get_attribute('class').lower():
                            ActionChains(driver).move_to_element(pe).click(pe).perform()
                            time.sleep(0.3)
                            print(f"Opened dropdown via sibling toggle")
                            return True
            except:
                pass
                
            return False
        except:
            return False
    
    
    def execute(self, driver: WebDriver) -> None:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Wait for element to be clickable on first attempt
                element = self.element.get(driver, wait_for_clickable=True)
                
                # Check if element is visible/interactable
                if not element.is_displayed():
                    # Try to open parent dropdown first
                    self._try_open_parent_dropdown(driver)
                    element = self.element.get(driver, wait_for_clickable=True)  # Re-fetch element
                
                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)
                time.sleep(0.3)  # Brief pause after scroll
                
                # Try JavaScript click as a fallback for Angular apps
                try:
                    ActionChains(driver).move_to_element(element).click(element).perform()
                except (ElementClickInterceptedException, ElementNotInteractableException):
                    # Fallback to JavaScript click
                    driver.execute_script("arguments[0].click();", element)
                
                return  # Success, exit the method
                
            except ElementNotInteractableException as e:
                # Element exists but can't be clicked - likely in closed dropdown
                if attempt == 0:
                    print(f"Element not interactable, trying to open parent dropdown...")
                    if self._try_open_parent_dropdown(driver):
                        continue  # Retry after opening dropdown
                
                # Try JavaScript click as last resort
                if attempt == max_retries - 2:
                    try:
                        element = self.element.get(driver)
                        driver.execute_script("arguments[0].click();", element)
                        return  # Success
                    except:
                        pass
                
                if attempt < max_retries - 1:
                    print(f"Click attempt {attempt + 1} failed: ElementNotInteractable, retrying...")
                    time.sleep(0.5)
                else:
                    print(f"Click failed after {max_retries} attempts - element not interactable")
                    print(f"URL: {driver.current_url}")
                    print(f"Element ID: {self.element.get_id()}")
                    raise RuntimeError(f"Click action failed on element {self.element.get_id()}: {e}") from e
                
            except (TimeoutException, StaleElementReferenceException, ElementClickInterceptedException) as e:
                # On TimeoutException, try JS click as fallback
                if isinstance(e, TimeoutException) and attempt < max_retries - 1:
                    try:
                        # Try finding element without waiting for clickable
                        element = driver.find_element(By.XPATH, self.element.get_id())
                        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)
                        time.sleep(0.2)
                        driver.execute_script("arguments[0].click();", element)
                        return  # Success
                    except:
                        pass
                
                if attempt < max_retries - 1:
                    print(f"Click attempt {attempt + 1} failed: {type(e).__name__}, retrying...")
                    time.sleep(0.5)
                else:
                    print(f"Click failed after {max_retries} attempts")
                    print(f"URL: {driver.current_url}")
                    print(f"Element ID: {self.element.get_id()}")
                    raise RuntimeError(f"Click action failed on element {self.element.get_id()}: {e}") from e
            except Exception as e:
                print(f"Unexpected error during click: {type(e).__name__}")
                print(f"URL: {driver.current_url}")
                print(f"Element ID: {self.element.get_id()}")
                raise RuntimeError(f"Click action failed on element {self.element.get_id()}: {e}") from e
