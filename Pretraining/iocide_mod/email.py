"""
Email address detection
"""
from copy import deepcopy
from email.headerregistry import Address
from logging import getLogger
from string import ascii_letters, digits, printable

import regex

from . import hostname, ip, url
from .template import Template


logger = getLogger(name=__name__)


class ParsedAddress(Address):
	"""
	Provide the defanged value as the `str` representation if present
	"""
	def __init__(self, username, domain, span, *, defanged=None, **kwargs):
		try:
			self.span = span
			super().__init__(username=username, domain=domain, **kwargs)
		except ValueError:
			self._username = username
			self._domain = domain
			self._display_name = kwargs.get('display_name', '')

		self.defanged = defanged

	def __str__(self):
		if self.defanged:
			return self.defanged

		return f'{self.username}@{self.domain}'

	def __hash__(self):
		return hash((self.username, self.domain))


unquoted_local_alphabet = digits + ascii_letters + r"!#$%&'*+-/?^_`{|}~"
quoted_alphabet = printable.replace('"', '').replace('\\', '')

PATTERN_TEMPLATE = Template(
	format=(
		r'(?<!\w)(?P<local>(?>(?P<unquoted>{unquoted})|(?P<quoted>{quoted})))'
		r'{at}'
		r'(?P<domain>{domain})'
	),
)

PATTERN_TEMPLATE['unquoted'] = (
	r'{valid}{{1,{max_repeat}}}'
	r'(?:{dot}{valid}{{1,{max_repeat}}}){{0,{max_repeat}}}'
)
PATTERN_TEMPLATE['unquoted']['valid'] = (
	f'[{regex.escape(unquoted_local_alphabet)}]')
PATTERN_TEMPLATE['unquoted']['dot'] = Template(
	r'(?>{symbol}|{text})', normalised='.')
PATTERN_TEMPLATE ['unquoted']['dot']['symbol'] = Template.from_defang_pattern(
	pattern=r'[\.\,]', normalised=None, openers='([<')
PATTERN_TEMPLATE ['unquoted']['dot']['text'] = Template.from_defang_pattern(
	pattern=r'dot',
	normalised=None,
	openers='([<',
	match_unwrapped=False,
	allow_unbalanced=False,
)


PATTERN_TEMPLATE['unquoted']['max_repeat'] = 64

PATTERN_TEMPLATE['quoted'] = r'"(?>{dot}|{character}){{1,64}}"'
PATTERN_TEMPLATE['quoted']['character'] = r'[{}]'.format(
	regex.escape(quoted_alphabet.replace('.', '')))
PATTERN_TEMPLATE['quoted']['dot'] = deepcopy(
	PATTERN_TEMPLATE['unquoted']['dot'])
		
PATTERN_TEMPLATE['at'] = deepcopy(url.PATTERN_TEMPLATE['netloc']['at'])

PATTERN_TEMPLATE['domain'] = r'(?>(?P<hostname>{hostname})|\[(?P<ip>{ip})\])'
PATTERN_TEMPLATE['domain']['hostname'] = deepcopy(hostname.PATTERN_TEMPLATE)
PATTERN_TEMPLATE['domain']['ip'] = deepcopy(ip.PATTERN_TEMPLATE)


DOMAIN_REFANG_MAP = {
	'ip': lambda d: f'[{ip.refang(d)}]',
	'hostname': hostname.refang,
}


def parse_match(match, refang=False):
	"""
	Create an email address from a regular expression match
	"""
	logger.debug('Parsing email match: %r', match)
	unquoted_local = match['unquoted']
	if unquoted_local:
		local = PATTERN_TEMPLATE['unquoted']['dot'].normalise(unquoted_local)
	else:
		local = match['quoted']
		logger.debug('Quoted local: %r', local)

	(domain_key, defanged_domain), = (
		(k, match[k]) for k in ['hostname', 'ip'] if match[k])

	domain = DOMAIN_REFANG_MAP[domain_key](defanged_domain)
	span =(match.start(), match.end())
	if refang:
		return (ParsedAddress(username=local, domain=domain, span=span), span)
	
	return (ParsedAddress(
		username=local, domain=domain, defanged=match.group(0), span=span),span)


def extract(text, refang=False):
	"""
	Extract email address values from text
	"""
	for match in PATTERN_TEMPLATE.regex.finditer(text):
		yield parse_match(match, refang=refang)

def replace(text,replacement):
	
	return PATTERN_TEMPLATE.regex.sub(replacement, text)


def refang(defanged_email):
	"""
	Create a refanged email address from a defanged text value
	"""
	match = PATTERN_TEMPLATE.regex.fullmatch(defanged_email)
	return parse_match(match, refang=True)

