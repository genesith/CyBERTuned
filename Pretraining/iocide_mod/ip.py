"""
IP address extraction
"""
from copy import deepcopy
from ipaddress import IPv4Address, IPv6Address
from logging import getLogger

from .template import (
	DEFANG_CHARACTER_REGEX,
	Template,
)


logger = getLogger(name=__name__)


class DefangedIPv4Address(IPv4Address):
	"""
	Provide the defanged value as the `str` representation if present
	"""
	def __init__(self, *args, defanged, **kwargs):
		
		super().__init__(*args, **kwargs)
		self.defanged = defanged

	def __str__(self):
		return self.defanged


class DefangedIPv6Address(IPv6Address):
	"""
	Provide the defanged value as the `str` representation if present
	"""
	def __init__(self, *args, defanged, **kwargs):
		super().__init__(*args, **kwargs)
		self.defanged = defanged

	def __str__(self):
		return self.defanged


IPV4_TEMPLATE = Template(
	format=(
		r'(?<!\w|\.)'
		r'{segment}(?:{separator}{segment}){{3}}'
		r'(?!\w|\.)'
	)
)
IPV4_TEMPLATE['segment'] = r'[12]?\d{{1,2}}(?<!\b0\d{{1,2}})'

IPV4_TEMPLATE['separator'] = Template.from_defang_pattern(
	pattern=r'[\.\,]|dot', normalised='.', allow_unbalanced=True)
IPV4_TEMPLATE['separator'].normalised = '.'


logger.debug('IPV4_TEMPLATE: %s', IPV4_TEMPLATE)



_IPV6_FORMATS = [
	r'{segment}(?:{separator}{segment}){{7}}',
	r'{separator}(?:{separator}{segment}){{1,6}}',
]

for prefix in range(1, 6):
	suffix = 6 - prefix
	_IPV6_FORMATS.append(
		fr'(?:{{segment}}{{separator}}){{{{1,{prefix}}}}}'
		fr'(?:{{separator}}{{segment}}){{{{1,{suffix}}}}}'
	)

_IPV6_FORMATS.append(r'(?:{segment}{separator}){{1,6}}{separator}')

# atomic
IPV6_FORMAT = r'(?>{})'.format(r'|'.join(_IPV6_FORMATS))

IPV6_TEMPLATE = Template(format=IPV6_FORMAT)
IPV6_TEMPLATE['segment'] = r'[0-9a-fA-F]{{1,4}}'

IPV6_TEMPLATE['separator'] = Template(':', normalised=':')

logger.debug('IPV6_TEMPLATE: %s', IPV6_TEMPLATE)


PATTERN_TEMPLATE = Template(r'(?:(?P<ipv4>{ipv4})|(?P<ipv6>{ipv6}))')
PATTERN_TEMPLATE['ipv4'] = deepcopy(IPV4_TEMPLATE)
PATTERN_TEMPLATE['ipv6'] = deepcopy(IPV6_TEMPLATE)


logger.debug('PATTERN_TEMPLATE: %s', PATTERN_TEMPLATE)


PARSED_TYPES = {
	'ipv4': (IPv4Address, DefangedIPv4Address),
	'ipv6': (IPv6Address, DefangedIPv6Address),
}


def parse_match(match, refang=False):
	"""
	Create an IP address from a regular expression match
	"""
	defanged_string = match.group(0)
	key, = (k for k in PARSED_TYPES if match[k])
	separator = PATTERN_TEMPLATE[key]['separator']
	separated = separator.regex.sub(separator.normalised, match.group(0))
	refanged_string = DEFANG_CHARACTER_REGEX.sub('', separated)
	refanged_type, defanged_type = PARSED_TYPES[key]
	span =(match.start(), match.end())
	if refang or (refanged_string == defanged_string):
		return (refanged_type(refanged_string), span)
	# return (defanged_type(refanged_string, defanged=match.group()), span)
	return (match.group(), span)


def extract(text, refang=False):
	"""
	Extract IP address values from text
	"""
	for match in PATTERN_TEMPLATE.regex.finditer(text):
		logger.debug('Found match: %r', match)
		try:
			yield parse_match(match, refang=refang)
		except:
			print("skip this ip")


def replace(text,replacement):
	return PATTERN_TEMPLATE.regex.sub(replacement, text)

def refang(defanged_ip):
	"""
	Create an IP address from a defanged text value
	"""
	match = PATTERN_TEMPLATE.regex.fullmatch(defanged_ip)
	return parse_match(match, refang=True)
