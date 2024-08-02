from django import template

register = template.Library()

@register.filter
def remove_media_prefix(value):
    return value.replace('/media', '', 1)