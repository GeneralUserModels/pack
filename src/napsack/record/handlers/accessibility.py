import sys
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Set
from napsack.record.models.event import InputEvent, EventType


class AccessibilityHandlerBase(ABC):
    ROLE_KEY: str = "Role"
    UNIVERSAL_ATTRS: List[str] = []
    ROLE_SPECIFIC: Dict[str, List[str]] = {}
    USEFUL_FIELDS: List[str] = []
    GENERIC_ROLES: Set[str] = set()
    INTERACTIVE_ROLES: Set[str] = set()
    
    def __init__(self):
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
            if ax_info and self._has_useful_info(ax_info):
                return {'accessibility': ax_info}
        return {}
    
    def _handle_mouse_down(self, input_event: InputEvent) -> Dict[str, Any]:
        x, y = input_event.cursor_position
        
        element = self._get_element_at_position(x, y)
        if element:
            ax_info = self._extract_element_info(element)
            if ax_info and self._has_useful_info(ax_info):
                return {'accessibility': ax_info}
        
        return {}
    
    def _handle_mouse_up(self, input_event: InputEvent) -> Dict[str, Any]:
        return {}
    
    def _handle_mouse_scroll(self, input_event: InputEvent) -> Dict[str, Any]:
        x, y = input_event.cursor_position
        element = self._get_element_at_position(x, y)
        if element:
            ax_info = self._extract_element_info(element)
            if ax_info and self._has_useful_info(ax_info):
                return {'accessibility': ax_info}
        return {}
    
    def _handle_key_press(self, input_event: InputEvent) -> Dict[str, Any]:
        focused_element = self._get_focused_element()
        if focused_element:
            ax_info = self._extract_element_info(focused_element)
            if ax_info and self._has_useful_info(ax_info):
                return {'focused_element': ax_info}
        
        return {}
    
    def _handle_key_release(self, input_event: InputEvent) -> Dict[str, Any]:
        return {}
    
    @abstractmethod
    def _get_element_at_position(self, x: int, y: int) -> Optional[Any]:
        pass
    
    @abstractmethod
    def _get_focused_element(self) -> Optional[Any]:
        pass
    
    @abstractmethod
    def _extract_element_info(self, element) -> Optional[Dict[str, Any]]:
        pass
    
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
    
    @staticmethod
    def _clean_value(value):
        if value is None:
            return None
        
        if isinstance(value, (str, int, float, bool)):
            return value
        
        if isinstance(value, (list, tuple)):
            return [AccessibilityHandlerBase._clean_value(v) for v in value]
        
        if isinstance(value, dict):
            return {k: AccessibilityHandlerBase._clean_value(v) for k, v in value.items()}
        
        try:
            return str(value)
        except:
            return None


if sys.platform == 'darwin':
    from ._accessibility_mac import AccessibilityHandlerMac as AccessibilityHandler
elif sys.platform == 'win32':
    from ._accessibility_windows import AccessibilityHandlerWindows as AccessibilityHandler
else:
    from ._accessibility_windows import AccessibilityHandlerWindows as AccessibilityHandler
