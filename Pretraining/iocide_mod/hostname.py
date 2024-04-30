"""
Hostname value detetction
"""
from logging import getLogger
from pathlib import Path

from .template import Template


logger = getLogger(name=__name__)


TLD_PATH = Path(__file__).parent.joinpath('data', 'top_level_domains.txt')
with TLD_PATH.open() as tld_file:
	TLD_LIST = [x.strip() for x in tld_file.readlines()]


PATTERN_TEMPLATE = Template(
	format=(
		r'(?<!\w|\.)'
		r'(?:{label}{separator}){{1,30}}{tld}'
		r'(?!\w|\.)'
	),
)

PATTERN_TEMPLATE ['label'] = r'[a-zA-Z0-9][a-zA-Z0-9\-]{{0,62}}'

PATTERN_TEMPLATE ['separator'] = Template(
	r'(?>{symbol}|{text})', normalised='.')
PATTERN_TEMPLATE ['separator']['symbol'] = Template.from_defang_pattern(
	pattern=r'[\.\,]', normalised=None)
PATTERN_TEMPLATE ['separator']['text'] = Template.from_defang_pattern(
	pattern=r'dot',
	normalised=None,
	match_unwrapped=False,
	allow_unbalanced=False,
)

#atomic
PATTERN_TEMPLATE ['tld'] = r'(?>{})'.format(
	'|'.join(rf'{t}\b' for t in TLD_LIST))


logger.debug('PATTERN_TEMPLATE: %s', PATTERN_TEMPLATE)


def extract(text, refang=False):
	"""
	Extract hostname values from text
	"""
	for match in PATTERN_TEMPLATE.regex.finditer(text):
		yield parse_match(match, refang=refang)


def parse_match(match, refang=False):
	"""
	Create a hostname string from a regular expression match
	"""
	defanged_hostname = match.group(0)
	if not refang:
		return defanged_hostname

	return PATTERN_TEMPLATE['separator'].normalise(defanged_hostname)


def refang(defanged_hostname):
	"""
	Create a refanged hostname string from a defanged string value
	"""
	match = PATTERN_TEMPLATE.regex.fullmatch(defanged_hostname)
	return parse_match(match, refang=True)
