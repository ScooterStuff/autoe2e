from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from autoe2e.browser.utils import get_element_xpath
from autoe2e.crawler.action.identification import Identification, How



class Element:
    def __init__(self, driver: WebDriver, element: WebElement):
        self._id: Identification | None = None
        self.outerHTML = element.get_attribute("outerHTML")
        # Use fallback chain for test_id
        self.test_id = (
            element.get_attribute("data-testid") or 
            element.get_attribute("data-formid") or
            element.get_attribute("name") or
            element.get_attribute("id") or
            element.get_attribute("formcontrolname") or
            element.get_attribute("placeholder")
        )
        self.set_identification(driver, element)
    
    
    def set_identification(self, driver: WebDriver, element: WebElement) -> None:
        # if element.get_attribute("data-testid"):
        #     self._id = Identification(element.get_attribute("data-testid"), How.BY_TEST_ID)
        if element.get_attribute("id"):
            self._id = Identification(element.get_attribute("id"), How.BY_ID)
        else:
            self._id = Identification(get_element_xpath(driver, element), How.BY_XPATH)
    
    
    def get_id(self):
        return self._id.get_value()
    
    
    def get(self, driver: WebDriver, wait_for_clickable: bool = False) -> WebElement:
        if self._id is None:
            raise ValueError("Element not identified")

        # if self._id.get_how() == How.BY_TEST_ID:
        #     return driver.find_element(By.XPATH, f"//*[@data-testid='{self._id.get_value()}']")
        wait = WebDriverWait(driver, 15)  # Increased timeout for Angular apps
        
        # Choose appropriate expected condition
        condition = EC.element_to_be_clickable if wait_for_clickable else EC.presence_of_element_located
        
        if self._id.get_how() == How.BY_ID:
            return wait.until(condition((By.ID, self._id.get_value())))
            # return driver.find_element(By.ID, self._id.get_value())
        
        return wait.until(condition((By.XPATH, self._id.get_value())))
        
        # return driver.find_element(By.XPATH, self._id.get_value())