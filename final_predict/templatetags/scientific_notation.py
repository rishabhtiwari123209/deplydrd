# templatetags/scientific_notation.py

from django import template

register = template.Library()

@register.filter
def scientific_notation(value):
    try:
        return f"{value:.2e}"
    except (ValueError, TypeError):
        return value
