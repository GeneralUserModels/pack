from typing import Optional, Dict, Any
import sys
import ctypes

try:
    import comtypes
    import comtypes.client
except ImportError:
    # This file should only be imported on Windows, but we handle the import
    # failure gracefully to avoid breaking environments where it's missing.
    pass

from .accessibility import AccessibilityHandlerBase
from napsack.record.models.event import InputEvent


class AccessibilityHandlerWindows(AccessibilityHandlerBase):
    ROLE_KEY = 'ControlType'
    
    # Mapping UIA properties to our internal keys
    # UIA Property IDs can be found in UIAutomationClient
    UNIVERSAL_ATTRS = [
        'Name',
        'ControlType',
        'AutomationId',
        'IsEnabled',
        'HasKeyboardFocus',
        'IsPassword',
        'ClassName',
    ]
    
    # Specific properties to look for based on control type
    ROLE_SPECIFIC = {
        'UIA_ButtonControlTypeId': ['Name', 'HelpText'],
        'UIA_EditControlTypeId': ['Name', 'Value'],
        'UIA_CheckBoxControlTypeId': ['Name', 'ToggleState'],
        'UIA_RadioButtonControlTypeId': ['Name', 'SelectionItemIsSelected'],
        'UIA_ComboBoxControlTypeId': ['Name', 'Value'],
        'UIA_ListControlTypeId': ['Name'],
        'UIA_ListItemControlTypeId': ['Name', 'SelectionItemIsSelected'],
        'UIA_HyperlinkControlTypeId': ['Name', 'Value'],
        'UIA_SliderControlTypeId': ['Name', 'RangeValueValue'],
        'UIA_SpinnerControlTypeId': ['Name', 'RangeValueValue'],
        'UIA_TabItemControlTypeId': ['Name', 'SelectionItemIsSelected'],
    }

    USEFUL_FIELDS = ['Name', 'Value', 'HelpText', 'RangeValueValue']
    GENERIC_ROLES = {
        'UIA_PaneControlTypeId', 'UIA_GroupControlTypeId', 'UIA_TextControlTypeId',
        'UIA_WindowControlTypeId', 'UIA_DocumentControlTypeId', 'UIA_CustomControlTypeId'
    }
    INTERACTIVE_ROLES = {
        'UIA_ButtonControlTypeId', 'UIA_EditControlTypeId', 'UIA_CheckBoxControlTypeId',
        'UIA_RadioButtonControlTypeId', 'UIA_ComboBoxControlTypeId', 'UIA_ListItemControlTypeId',
        'UIA_HyperlinkControlTypeId', 'UIA_SliderControlTypeId', 'UIA_TabItemControlTypeId',
        'UIA_MenuItemControlTypeId'
    }

    def __init__(self):
        super().__init__()
        # Pre-generate the module if it's not already there
        try:
            from comtypes.gen import UIAutomationClient
        except ImportError:
            comtypes.client.GetModule('UIAutomationCore.dll')
            from comtypes.gen import UIAutomationClient
        self.UIAutomationClient = UIAutomationClient
        self._automation = None
        self._initialize_automation()

    def _initialize_automation(self):
        try:
            # We initialize automation once, but CoInitialize is needed per thread
            self._automation = comtypes.client.CreateObject(
                self.UIAutomationClient.CUIAutomation,
                interface=self.UIAutomationClient.IUIAutomation
            )
        except:
            self._automation = None

    def __call__(self, input_event: InputEvent) -> Dict[str, Any]:
        try:
            ctypes.windll.ole32.CoInitialize(None)
            try:
                return super().__call__(input_event)
            finally:
                ctypes.windll.ole32.CoUninitialize()
        except:
            return {}

    def _get_element_at_position(self, x: int, y: int) -> Optional[Any]:
        if not self._automation:
            self._initialize_automation()
        if not self._automation:
            return None
        
        try:
            point = ctypes.wintypes.POINT(x, y)
            element = self._automation.ElementFromPoint(point)
            return element
        except:
            return None
    
    def _get_focused_element(self) -> Optional[Any]:
        if not self._automation:
            self._initialize_automation()
        if not self._automation:
            return None
            
        try:
            element = self._automation.GetFocusedElement()
            return element
        except:
            return None
    
    def _extract_element_info(self, element) -> Optional[Dict[str, Any]]:
        if not element:
            return None
        
        info = {}
        
        # Extract universal attributes
        for attr in self.UNIVERSAL_ATTRS:
            try:
                prop_id = getattr(self.UIAutomationClient, f"UIA_{attr}PropertyId")
                value = element.GetCurrentPropertyValue(prop_id)
                if value is not None:
                    # Special handling for ControlType to get its name
                    if attr == 'ControlType':
                        value = self._get_control_type_name(value)
                    info[attr] = self._clean_value(value)
            except:
                pass
        
        # Extract role-specific attributes
        try:
            control_type_id = element.CurrentControlType
            role_name = self._get_control_type_name(control_type_id)
            if role_name in self.ROLE_SPECIFIC:
                for attr in self.ROLE_SPECIFIC[role_name]:
                    if attr not in info:
                        try:
                            prop_id = getattr(self.UIAutomationClient, f"UIA_{attr}PropertyId")
                            value = element.GetCurrentPropertyValue(prop_id)
                            if value is not None:
                                info[attr] = self._clean_value(value)
                        except:
                            pass
        except:
            pass

        # Extract parent info
        try:
            walker = self._automation.ControlViewWalker
            parent = walker.GetParentElement(element)
            if parent:
                parent_info = {}
                try:
                    parent_role = self._get_control_type_name(parent.CurrentControlType)
                    parent_name = parent.CurrentName
                    if parent_role:
                        parent_info['ControlType'] = parent_role
                    if parent_name:
                        parent_info['Name'] = parent_name
                except:
                    pass
                if parent_info:
                    info['_parent'] = parent_info
        except:
            pass
        
        return info if info else None

    def _get_control_type_name(self, control_type_id: int) -> str:
        """Helper to get the UIA_... name from the integer ID."""
        for attr in dir(self.UIAutomationClient):
            if attr.startswith('UIA_') and attr.endswith('ControlTypeId'):
                if getattr(self.UIAutomationClient, attr) == control_type_id:
                    return attr
        return str(control_type_id)

    def _has_useful_info(self, ax_data: Dict[str, Any]) -> bool:
        if not ax_data:
            return False
        
        for field in self.USEFUL_FIELDS:
            value = ax_data.get(field)
            if value and str(value).strip():
                return True
        
        role = ax_data.get(self.ROLE_KEY, '')
        
        if role in self.GENERIC_ROLES:
            return False
        
        if role in self.INTERACTIVE_ROLES:
            return True
        
        parent = ax_data.get('_parent', {})
        if parent:
            for field in self.USEFUL_FIELDS:
                value = parent.get(field)
                if value and str(value).strip():
                    return True
        
        return False
