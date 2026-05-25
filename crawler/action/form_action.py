from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.keys import Keys

import time

from autoe2e.crawler.action.action import Action, ActionType
from autoe2e.crawler.action.element import Element
from autoe2e.browser.utils import get_element_xpath

from autoe2e.manual_ndd import FORBIDDEN_ACTIONS


class FormActionType(ActionType):
    def __init__(self):
        super().__init__('form')


def find_input_element(container: WebElement, key: str) -> WebElement:
    """
    Find an input element using multiple fallback strategies.
    Tries different locators in order of specificity.
    """
    # List of XPath strategies to try in order
    selectors = [
        f".//*[@data-testid='{key}']",           # data-testid (preferred)
        f".//*[@name='{key}']",                   # name attribute
        f".//*[@formcontrolname='{key}']",        # Angular formControlName
        f".//*[@placeholder='{key}']",            # placeholder text
        f".//*[@id='{key}']",                     # id attribute
        f".//input[@type='{key}']",               # type (for unique types like 'password')
        # Case-insensitive variations
        f".//*[translate(@placeholder, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz')='{key.lower()}']",
        f".//*[translate(@name, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz')='{key.lower()}']",
        # Partial matches (contains)
        f".//*[contains(@data-testid, '{key}')]",
        f".//*[contains(@name, '{key}')]",
        f".//*[contains(@placeholder, '{key}')]",
        f".//*[contains(@formcontrolname, '{key}')]",
    ]
    
    for selector in selectors:
        try:
            element = container.find_element(By.XPATH, selector)
            return element
        except:
            continue
    
    raise Exception(f"Could not find input element with key: {key}")


def find_submit_button(driver: WebDriver, container: WebElement, form_id: str = None) -> WebElement:
    """
    Find a submit button using multiple fallback strategies.
    """
    selectors = []
    
    # If we have a form_id, try data-submitid first
    if form_id:
        selectors.append(f".//*[@data-submitid='{form_id}']")
    
    # Common submit button patterns
    selectors.extend([
        ".//button[@type='submit']",                    # Standard submit button
        ".//input[@type='submit']",                     # Submit input
        ".//button[contains(@class, 'submit')]",        # Class contains 'submit'
        ".//button[contains(@class, 'btn-primary')]",   # Primary button (Bootstrap)
        ".//button[contains(@class, 'btn-success')]",   # Success button
        ".//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'submit')]",
        ".//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'sign in')]",
        ".//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'login')]",
        ".//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'register')]",
        ".//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'sign up')]",
        ".//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'create')]",
        ".//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'save')]",
        ".//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'add')]",
        ".//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'update')]",
        ".//button[not(@type) or @type='button']",      # Any button without type or type=button
        ".//*[@role='button']",                         # Role=button elements
    ])
    
    # First try within the container
    for selector in selectors:
        try:
            element = container.find_element(By.XPATH, selector)
            return element
        except:
            continue
    
    # If not found in container, search the whole page for submit buttons
    page_selectors = [s.replace("./", "//") for s in selectors]
    for selector in page_selectors:
        try:
            element = driver.find_element(By.XPATH, selector)
            return element
        except:
            continue
    
    raise Exception("Could not find submit button")


class FormAction(Action):
    def __init__(self, element: Element):
        super().__init__(element, action_type=FormActionType())
        self.params = None
        self.filled_field_test_ids = []  # Track which fields were filled
        self.submit_button_test_id = None  # Track the submit button test_id
    
    
    def set_params(self, params: dict[str, str | int | float | bool]) -> None:
        self.params = params
    
    
    def has_params(self) -> bool:
        return self.params is not None
    
    
    def get_filled_field_test_ids(self) -> list:
        """Return list of test_ids for fields that were filled."""
        return self.filled_field_test_ids
    
    
    def get_submit_button_test_id(self) -> str:
        """Return the test_id of the submit button."""
        return self.submit_button_test_id
    
    
    def execute(self, driver: WebDriver) -> None:
        if self.params is None:
            raise ValueError('Parameters are not set for the form action.')
        
        print(f"  Starting form execution with params: {self.params}")
        
        try:
            form_element = self.element.get(driver)
            print(f"  Found form element")
        except Exception as e:
            print(f"  Failed to get form element: {e}")
            raise
        
        driver.execute_script("arguments[0].scrollIntoView(true);", form_element)
        time.sleep(0.2)
        
        element: WebElement = form_element
        self.filled_field_test_ids = []  # Reset before execution
        
        try:
            form_id = element.get_attribute('data-formid')
            print(f"  Form ID: {form_id}")
            
            for param_key, param_value in self.params.items():
                try:
                    print(f"  Looking for input: {param_key}")
                    input_element = find_input_element(element, param_key)
                    print(f"  Found input element for {param_key}")
                    
                    # Check if this is a select element
                    tag_name = input_element.tag_name.lower()
                    role = input_element.get_attribute('role') or ''
                    
                    # Handle Angular Material mat-select (role="combobox")
                    if tag_name == 'mat-select' or role == 'combobox':
                        print(f"  Detected mat-select/combobox for {param_key}")
                        time.sleep(0.2)
                        
                        # Click to open the dropdown
                        try:
                            driver.execute_script("arguments[0].click();", input_element)
                        except:
                            input_element.click()
                        time.sleep(0.3)
                        
                        # Handle multi-select vs single select
                        is_multiple = input_element.get_attribute('multiple') is not None
                        values_to_select = param_value if isinstance(param_value, list) else [param_value]
                        
                        for val in values_to_select:
                            val_str = str(val).lower().strip()
                            
                            # Find and click the mat-option with matching text
                            # mat-options are rendered in an overlay panel
                            options_xpath = "//mat-option"
                            try:
                                options = driver.find_elements(By.XPATH, options_xpath)
                                option_found = False
                                
                                for option in options:
                                    option_text = option.text.strip().lower()
                                    if option_text == val_str or val_str in option_text:
                                        # Check if already selected (for multi-select)
                                        is_selected = 'mat-selected' in (option.get_attribute('class') or '')
                                        if not is_selected:
                                            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", option)
                                            time.sleep(0.1)
                                            driver.execute_script("arguments[0].click();", option)
                                            print(f"  Selected option: {option.text.strip()}")
                                        else:
                                            print(f"  Option already selected: {option.text.strip()}")
                                        option_found = True
                                        time.sleep(0.1)
                                        break
                                
                                if not option_found:
                                    print(f"  Could not find option matching '{val}' in mat-select")
                            except Exception as opt_err:
                                print(f"  Error selecting mat-option: {opt_err}")
                        
                        # Close dropdown by clicking outside or pressing Escape
                        if not is_multiple:
                            time.sleep(0.1)  # Single select auto-closes
                        else:
                            # For multi-select, click the backdrop or press Escape to close
                            try:
                                backdrop = driver.find_element(By.CSS_SELECTOR, ".cdk-overlay-backdrop")
                                backdrop.click()
                            except:
                                try:
                                    input_element.send_keys(Keys.ESCAPE)
                                except:
                                    pass
                            time.sleep(0.2)
                    
                    elif tag_name == 'select':
                        # Handle select elements with Select class
                        select = Select(input_element)
                        options = select.options
                        
                        if not options:
                            print(f"  Select {param_key} has no options, skipping...")
                            continue
                        
                        # Try to find matching option by value or visible text
                        selected = False
                        param_str = str(param_value)
                        
                        # Skip invalid values like '[object Object]'
                        if param_str in ['[object Object]', 'undefined', 'null', '']:
                            # Select first non-empty option if available
                            for opt in options:
                                opt_value = opt.get_attribute('value')
                                opt_text = opt.text.strip()
                                if opt_value and opt_value not in ['', 'undefined', 'null']:
                                    try:
                                        select.select_by_value(opt_value)
                                        print(f"  Selected first valid option: {opt_text}")
                                        selected = True
                                        break
                                    except:
                                        pass
                        else:
                            # Try to match by value first
                            for opt in options:
                                opt_value = opt.get_attribute('value')
                                if opt_value == param_str:
                                    try:
                                        select.select_by_value(param_str)
                                        selected = True
                                        break
                                    except:
                                        pass
                            
                            # Try to match by visible text
                            if not selected:
                                for opt in options:
                                    if opt.text.strip().lower() == param_str.lower():
                                        try:
                                            select.select_by_visible_text(opt.text.strip())
                                            selected = True
                                            break
                                        except:
                                            pass
                            
                            # Try partial match
                            if not selected:
                                for opt in options:
                                    if param_str.lower() in opt.text.strip().lower():
                                        try:
                                            select.select_by_visible_text(opt.text.strip())
                                            selected = True
                                            break
                                        except:
                                            pass
                            
                            # Last resort: select first non-empty option
                            if not selected and len(options) > 0:
                                for opt in options:
                                    opt_value = opt.get_attribute('value')
                                    if opt_value and opt_value not in ['', 'undefined', 'null']:
                                        try:
                                            select.select_by_value(opt_value)
                                            print(f"  Could not match '{param_str}', selected first valid option")
                                            selected = True
                                            break
                                        except:
                                            pass
                        
                        # Trigger change event for Angular
                        driver.execute_script("""
                            var el = arguments[0];
                            el.dispatchEvent(new Event('change', { bubbles: true }));
                        """, input_element)
                        
                    else:
                        # Handle regular input/textarea elements
                        input_type = (input_element.get_attribute('type') or '').lower()
                        
                        # Check if this is a date/datetime input or mat-datepicker
                        is_date_input = input_type in ['date', 'datetime', 'datetime-local', 'time', 'month', 'week']
                        has_datepicker = input_element.get_attribute('matDatepicker') is not None
                        
                        if is_date_input or has_datepicker:
                            # Handle date inputs via JavaScript to avoid triggering calendar popup
                            # Format: YYYY-MM-DD for date, YYYY-MM-DDTHH:MM for datetime-local
                            date_value = str(param_value)
                            print(f"  Setting date input via JS: {date_value}")
                            driver.execute_script("""
                                var el = arguments[0];
                                var value = arguments[1];
                                // Set value directly via JavaScript
                                el.value = value;
                                // Trigger all necessary events for Angular/React
                                el.dispatchEvent(new Event('input', { bubbles: true }));
                                el.dispatchEvent(new Event('change', { bubbles: true }));
                                el.dispatchEvent(new Event('blur', { bubbles: true }));
                                // For Angular Material datepicker
                                if (el.hasAttribute('matDatepicker') || el.hasAttribute('matInput')) {
                                    // Try to update the form control value
                                    var ngModel = el.getAttribute('ng-reflect-model');
                                    if (ngModel) {
                                        el.dispatchEvent(new CustomEvent('ngModelChange', { detail: value, bubbles: true }));
                                    }
                                }
                            """, input_element, date_value)
                        else:
                            # Regular text input - clear and type
                            input_element.clear()
                            input_element.send_keys(str(param_value))
                            
                            # Trigger Angular change detection events
                            driver.execute_script("""
                                var el = arguments[0];
                                el.dispatchEvent(new Event('input', { bubbles: true }));
                                el.dispatchEvent(new Event('change', { bubbles: true }));
                                el.dispatchEvent(new Event('blur', { bubbles: true }));
                            """, input_element)
                    
                    # Track this field as filled
                    self.filled_field_test_ids.append(param_key)
                    print(f"  Filled field: {param_key} = {param_value}")
                except Exception as field_error:
                    print(f"  Could not fill field {param_key}: {field_error}")
            
            submit_button = find_submit_button(driver, element, form_id)
            
            # Get submit button test_id (try multiple attributes)
            self.submit_button_test_id = (
                submit_button.get_attribute('data-testid') or
                submit_button.get_attribute('data-submitid') or
                submit_button.get_attribute('id') or
                submit_button.get_attribute('name') or
                submit_button.text.strip()[:20]  # Use button text as fallback
            )
            
            pair = (driver.current_url, get_element_xpath(driver, submit_button))
            
            if pair not in FORBIDDEN_ACTIONS:
                print(f"  Clicking submit button: {self.submit_button_test_id}")
                
                # Wait a moment for Angular to process form validation
                time.sleep(0.3)
                
                # Check if button is disabled and try to enable it
                if submit_button.get_attribute('disabled') is not None:
                    print("  Submit button is disabled, attempting to enable...")
                    driver.execute_script("arguments[0].removeAttribute('disabled');", submit_button)
                    time.sleep(0.1)
                
                # Scroll submit button into center of viewport to avoid navbar/footer obstruction
                driver.execute_script("""
                    arguments[0].scrollIntoView({block: 'center', inline: 'center'});
                """, submit_button)
                time.sleep(0.2)
                
                # Try clicking with JavaScript directly to avoid click interception issues
                # This is more reliable than normal click when elements like navbars overlap
                try:
                    # First try JavaScript click (more reliable for overlapped elements)
                    driver.execute_script("arguments[0].click();", submit_button)
                except Exception as js_click_error:
                    print(f"  JS click failed: {js_click_error}, trying normal click...")
                    try:
                        submit_button.click()
                    except Exception as click_error:
                        print(f"  Normal click also failed: {click_error}")
                        # Last resort: focus and submit the form directly
                        driver.execute_script("""
                            var btn = arguments[0];
                            var form = btn.closest('form');
                            if (form) {
                                form.submit();
                            } else {
                                btn.click();
                            }
                        """, submit_button)
        except Exception as e:
            print(e)
            print("Form action failed, skipping...")
            # Auto-skip instead of waiting for user input
