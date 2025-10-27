from typing import Optional, Dict, Any
from ApplicationServices import (
    AXUIElementCreateSystemWide,
    AXUIElementCopyElementAtPosition,
    AXUIElementCopyAttributeValue,
)
from AppKit import NSWorkspace
from record.models.event import InputEvent, EventType


class AccessibilityHandler:
    
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
    
    def __init__(self):
        self._workspace = NSWorkspace.sharedWorkspace()
        self._move_counter = 0
    
    def __call__(self, input_event: InputEvent) -> Dict[str, Any]:
        if input_event.event_type == EventType.MOUSE_MOVE:
            return self._handle_mouse_move(input_event)
        elif input_event.event_type == EventType.MOUSE_DOWN:
            return self._handle_mouse_down(input_event)
        elif input_event.event_type == EventType.MOUSE_UP:
            return self._handle_mouse_up(input_event)
        elif input_event.event_type == EventType.MOUSE_SCROLL:
            return self._handle_mouse_scroll(input_event)
        elif input_event.event_type == EventType.KEY_PRESS:
            return self._handle_key_press(input_event)
        elif input_event.event_type == EventType.KEY_RELEASE:
            return self._handle_key_release(input_event)
        return {}
    
    def _handle_mouse_move(self, input_event: InputEvent) -> Dict[str, Any]:
        self._move_counter += 1
        if self._move_counter % 10 != 0:
            return {}
        
        x, y = input_event.cursor_position
        element = self._get_element_at_position(x, y)
        if element:
            ax_info = self._extract_element_info(element)
            if ax_info:
                return {'accessibility': ax_info}
        return {}
    
    def _handle_mouse_down(self, input_event: InputEvent) -> Dict[str, Any]:
        x, y = input_event.cursor_position
        result = {}
        
        element = self._get_element_at_position(x, y)
        if element:
            ax_info = self._extract_element_info(element)
            if ax_info:
                result['accessibility'] = ax_info
        
        app_info = self._get_active_app_info()
        if app_info:
            result['active_app'] = app_info
        
        return result
    
    def _handle_mouse_up(self, input_event: InputEvent) -> Dict[str, Any]:
        return {}
    
    def _handle_mouse_scroll(self, input_event: InputEvent) -> Dict[str, Any]:
        x, y = input_event.cursor_position
        element = self._get_element_at_position(x, y)
        if element:
            ax_info = self._extract_element_info(element)
            if ax_info:
                return {'accessibility': ax_info}
        return {}
    
    def _handle_key_press(self, input_event: InputEvent) -> Dict[str, Any]:
        result = {}
        
        focused_element = self._get_focused_element()
        if focused_element:
            ax_info = self._extract_element_info(focused_element)
            if ax_info:
                result['focused_element'] = ax_info
        
        app_info = self._get_active_app_info()
        if app_info:
            result['active_app'] = app_info
        
        return result
    
    def _handle_key_release(self, input_event: InputEvent) -> Dict[str, Any]:
        return {}
    
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
    
    def _get_active_app_info(self) -> Optional[Dict[str, str]]:
        try:
            active_app = self._workspace.frontmostApplication()
            return {
                'app_name': active_app.localizedName(),
                'bundle_id': active_app.bundleIdentifier(),
            }
        except:
            return None
    
    @staticmethod
    def _clean_value(value):
        if value is None:
            return None
        
        if isinstance(value, (str, int, float, bool)):
            return value
        
        if isinstance(value, (list, tuple)):
            return [AccessibilityHandler._clean_value(v) for v in value]
        
        if isinstance(value, dict):
            return {k: AccessibilityHandler._clean_value(v) for k, v in value.items()}
        
        try:
            return str(value)
        except:
            return None
