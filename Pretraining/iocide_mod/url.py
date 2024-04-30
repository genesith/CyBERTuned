"""
URL detection
"""
from copy import deepcopy
from logging import getLogger
import re
from string import ascii_letters, digits, hexdigits
from urllib.parse import ParseResult

from . import hostname
from . import ip
from .template import DEFANG_CHARACTER_REGEX, Template


logger = getLogger(name=__name__)


URI_GEN_DELIMS = r':/?#[]@'
URI_SUB_DELIMS = r"!$&'()*+,;="
URI_UNRESERVED = digits + ascii_letters + r'-._~'

URI_UNENCODED_PCHAR = URI_UNRESERVED + URI_SUB_DELIMS + ':@'

URI_PERCENT_ENCODED = f'%[{hexdigits}]{{{{2}}}}'

MAX_REPEAT = 128


PERIOD_TEMPLATE = Template.from_defang_pattern(
	pattern=r'\.|dot', normalised='.')
AT_TEMPLATE = Template.from_defang_pattern(pattern='@|at', normalised='@')

DEFANGED_PATH_PERIOD = Template.from_defang_pattern(
	pattern=r'\.|dot',
	normalised='.',
	openers='[{<',
	match_unwrapped=False,
	allow_unbalanced=True,
)


class ParsedUrl(ParseResult):
	"""
	Disallow params component specification in respect of range of path values
	allowed by RFC 3986
	"""
	def __new__(cls, scheme, netloc, path, params, query, fragment):
		return super().__new__(
			cls,
			scheme=scheme,
			netloc=netloc,
			path=path,
			params='',
			query=query,
			fragment=fragment,
			
		)

	def __str__(self):
		return self.get_string(self)
		

	@classmethod
	def get_string(self, url):
		"""
		Provide string value construction for URL values with handling for
		specific edge cases
		"""
		url_string = url.geturl()
		# If there's a netloc but no scheme, default result leads with
		# authority slashes
		if url.netloc and not url.scheme:
			url_string = url_string.lstrip('/')

		# Default inserts a path slash if there are following components
		if not url.path:
			url_string = url_string.replace(
				'{}/'.format(url.netloc), url.netloc, 1)

		return url_string


PATTERN_TEMPLATE= Template(
	r'(?<!\w)'
	r'(?>(?P<scheme>{scheme}){colon_slashes})?'
	r'(?P<netloc>{netloc})'
	r'(?P<path>{path})'
	r'(?>\?(?P<query>{query}))?'
	r'(?>#(?P<fragment>{fragment}))?'
	r'(?!\w)'
)

# atomic
PATTERN_TEMPLATE['scheme'] = (
	r'[A-Za-z](?>[A-Za-z0-9\+\-]|{period}){{0,{max_repeat}}}')
PATTERN_TEMPLATE['scheme']['period'] = PERIOD_TEMPLATE
PATTERN_TEMPLATE['scheme']['max_repeat'] = MAX_REPEAT

#atomic
PATTERN_TEMPLATE['colon_slashes'] = r'(?>\://|\[:\]//|:\\\\|:?__)'

PATTERN_TEMPLATE['netloc'] = (
	r'(?:(?P<userinfo>{userinfo}){at})?{host}(?:{port_colon}(?P<port>{port}))?')


PATTERN_TEMPLATE['netloc']['userinfo'] = (
	'(?P<username>{character}{{1,{max_repeat}}})'
	r'(?:\:(?P<password>{character}{{0,{max_repeat}}}))?'
)
PATTERN_TEMPLATE['netloc']['userinfo']['character'] = (
	r'(?>'
		'{period}'
		f'|[{re.escape(URI_UNRESERVED+URI_SUB_DELIMS+":")}]'
		f'|{URI_PERCENT_ENCODED}'
	r')'
)

PATTERN_TEMPLATE['netloc']['userinfo']['character']['period'] = (
	Template.from_defang_pattern(
		pattern=r'\.|dot', normalised='.', openers='[{<')
)

PATTERN_TEMPLATE['netloc']['userinfo']['max_repeat'] = MAX_REPEAT

PATTERN_TEMPLATE['netloc']['at'] = Template.from_defang_pattern(
	pattern='@|at', normalised='@', allow_unbalanced=False)

# atomic
PATTERN_TEMPLATE['netloc']['host'] = (
	r'(?>(?P<hostname>{hostname})|\[?(?P<ipv6>{ipv6})\]?|(?P<ipv4>{ipv4}))')
PATTERN_TEMPLATE['netloc']['host']['hostname'] = deepcopy(
	hostname.PATTERN_TEMPLATE)
PATTERN_TEMPLATE['netloc']['host']['ipv4'] = deepcopy(ip.IPV4_TEMPLATE)
PATTERN_TEMPLATE['netloc']['host']['ipv6'] = deepcopy(ip.IPV6_TEMPLATE)

PATTERN_TEMPLATE['netloc']['port_colon'] = Template(format=':', normalised=':')
PATTERN_TEMPLATE['netloc']['port'] = r'[1-6]?\d{{1,4}}(?<!\b0{{2,}})'

PATTERN_TEMPLATE['path'] = (
	r'(?:/{character}{{0,{max_repeat}}}){{0,{max_repeat}}}')
PATTERN_TEMPLATE['path']['character'] = (
	f'(?>{{period}}|[{re.escape(URI_UNENCODED_PCHAR)}]|{URI_PERCENT_ENCODED})')
PATTERN_TEMPLATE['path']['character']['period'] = DEFANGED_PATH_PERIOD
PATTERN_TEMPLATE['path']['max_repeat'] = MAX_REPEAT

PATTERN_TEMPLATE['query'] = (
	'(?>'
		'{period}'
		f'|[{re.escape(URI_UNENCODED_PCHAR+"/?")}]'
		f'|{URI_PERCENT_ENCODED}'
	'){{0,{max_repeat}}}'
)
PATTERN_TEMPLATE['query']['max_repeat'] = MAX_REPEAT ** 2
PATTERN_TEMPLATE['query']['period'] = DEFANGED_PATH_PERIOD

PATTERN_TEMPLATE['fragment'] = deepcopy(PATTERN_TEMPLATE['query'])


URL_MATCH_COMPONENTS = ['scheme', 'netloc', 'path', 'params', 'query', 'fragment']


SCHEME_MAP = {'hxxp': 'http', 'hxxps': 'https'}


refang_map = {
	'hostname': hostname.refang,
	'ipv4': lambda i: str(ip.refang(i)),
	'ipv6': lambda i: '[{}]'.format(ip.refang(i))
}


def refang(defanged_url):
	"""
	Create a URL from a defanged text value
	"""
	match = PATTERN_TEMPLATE.regex.fullmatch(defanged_url)
	return parse_match(match, refang=True)


def extract(text, refang=False):
	"""
	Extract URL values from text
	"""
	for match in PATTERN_TEMPLATE.regex.finditer(text):
		logger.debug('Found match: %r', match)
		yield parse_match(match, refang=refang)


def replace(text,replacement):
	
	return PATTERN_TEMPLATE.regex.sub(replacement, text)


def parse_match(match, refang=False):
	"""
	Create a URL from a regular expression match
	"""
	logger.debug('Parsing URL match: %r', match)
	groups = match.groupdict(default='')
	groups['params'] = ''
	components = {k: groups[k] for k in URL_MATCH_COMPONENTS}
	span =(match.start(), match.end())
	
 

	if not refang:
		return (ParsedUrl(**components), span)

	raw_scheme = components.pop('scheme')
	stripped_scheme = DEFANG_CHARACTER_REGEX.sub('', raw_scheme)
	if stripped_scheme:
		lowercase_scheme = stripped_scheme.lower()
		scheme = SCHEME_MAP.get(lowercase_scheme, lowercase_scheme)
	elif match['username'] and not match['password']:
		scheme = 'mailto'
	else:
		scheme = 'http'

	(host_key, refang_host), = (
		(k, v) for k, v in refang_map.items() if match[k])

	host_component = match[host_key]
	netloc = refang_host(host_component)

	userinfo = match['userinfo']
	if userinfo:
		userinfo_template = PATTERN_TEMPLATE['netloc']['userinfo']
		period_normalised_userinfo = (
			userinfo_template['character']['period'].normalise(userinfo))
		userinfo = AT_TEMPLATE.normalise(period_normalised_userinfo)
		netloc = f'{userinfo}@{netloc}'

	if match['port']:
		port = DEFANG_CHARACTER_REGEX.sub('', match['port'])
		netloc += f':{port}'

	path = components.pop('path')
	if path:
		path = PERIOD_TEMPLATE.normalise(path)
	else:
		path = ''

	refanged_components = {
		'scheme': scheme,
		'netloc': netloc,
		'path': path,
		'query': components['query'],
		'fragment': components['fragment'],
	}

	logger.debug('refanged components: %r', refanged_components)

	return ParsedUrl(**refanged_components)
