from typing import Optional, Dict, Any
import sys

try:
    from ApplicationServices import (
        AXUIElementCreateSystemWide,
        AXUIElementCopyElementAtPosition,
        AXUIElementCopyAttributeValue,
    )
except ImportError:
    # This file should only be imported on macOS, but we handle the import
    # failure gracefully to avoid breaking environments where it's missing.
    pass

from .accessibility import AccessibilityHandlerBase


class AccessibilityHandlerMac(AccessibilityHandlerBase):
    ROLE_KEY = 'AXRole'
    UNIVERSAL_ATTRS = [
        'AXRole',
        'AXRoleDescription',
        'AXTitle',
        'AXDescription',
        'AXIdentifier',
        'AXDOMIdentifier',
        'AXEnabled',
        'AXFocused',
    ]
    
    ROLE_SPECIFIC = {
        'AXButton': ['AXTitle', 'AXDescription'],
        'AXCheckBox': ['AXTitle', 'AXValue'],
        'AXRadioButton': ['AXTitle', 'AXValue'],
        'AXTextField': ['AXTitle', 'AXValue', 'AXPlaceholderValue'],
        'AXTextArea': ['AXTitle', 'AXValue', 'AXSelectedText'],
        'AXStaticText': ['AXValue'],
        'AXLink': ['AXTitle', 'AXURL', 'AXVisited'],
        'AXImage': ['AXTitle', 'AXDescription', 'AXURL'],
        'AXMenuItem': ['AXTitle', 'AXEnabled'],
        'AXPopUpButton': ['AXTitle', 'AXValue'],
        'AXComboBox': ['AXTitle', 'AXValue'],
        'AXSlider': ['AXTitle', 'AXValue', 'AXMinValue', 'AXMaxValue'],
        'AXTab': ['AXTitle', 'AXValue'],
    }

    USEFUL_FIELDS = ['AXTitle', 'AXDescription', 'AXValue', 'AXPlaceholderValue', 'AXURL', 'AXLabel']
    GENERIC_ROLES = {'AXImage', 'AXGroup', 'AXStaticText', 'AXScrollArea', 'AXUnknown', 'AXCell'}
    INTERACTIVE_ROLES = {
        'AXButton', 'AXTextField', 'AXTextArea', 'AXCheckBox', 'AXRadioButton',
        'AXLink', 'AXMenuItem', 'AXPopUpButton', 'AXComboBox', 'AXTab', 'AXSlider'
    }

    def _get_element_at_position(self, x: int, y: int) -> Optional[Any]:
        try:
            system_wide = AXUIElementCreateSystemWide()
            error_code, element = AXUIElementCopyElementAtPosition(system_wide, x, y, None)
            
            if error_code == 0 and element:
                return element
            return None
        except:
            return None
    
    def _get_focused_element(self) -> Optional[Any]:
        try:
            system_wide = AXUIElementCreateSystemWide()
            error_code, element = AXUIElementCopyAttributeValue(
                system_wide, 'AXFocusedUIElement', None
            )
            
            if error_code == 0 and element:
                return element
            return None
        except:
            return None
    
    def _extract_element_info(self, element) -> Optional[Dict[str, Any]]:
        if not element:
            return None
        
        info = {}
        
        for attr in self.UNIVERSAL_ATTRS:
            try:
                error_code, value = AXUIElementCopyAttributeValue(element, attr, None)
                if error_code == 0 and value:
                    info[attr] = self._clean_value(value)
            except:
                pass
        
        role = info.get('AXRole')
        if role and role in self.ROLE_SPECIFIC:
            for attr in self.ROLE_SPECIFIC[role]:
                if attr not in info:
                    try:
                        error_code, value = AXUIElementCopyAttributeValue(element, attr, None)
                        if error_code == 0 and value:
                            info[attr] = self._clean_value(value)
                    except:
                        pass
        
        try:
            error_code, parent = AXUIElementCopyAttributeValue(element, 'AXParent', None)
            if error_code == 0 and parent:
                parent_info = {}
                for attr in ['AXRole', 'AXTitle']:
                    try:
                        error_code, value = AXUIElementCopyAttributeValue(parent, attr, None)
                        if error_code == 0 and value:
                            parent_info[attr] = self._clean_value(value)
                    except:
                        pass
                if parent_info:
                    info['_parent'] = parent_info
        except:
            pass
        
        return info if info else None
