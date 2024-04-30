"""
Hash value detection
"""
from logging import getLogger
from string import hexdigits

from .template import Template


logger = getLogger(name=__name__)


MD5_SIZE = 32
SHA1_SIZE = 40
SHA2_SIZES = [224, 256, 384, 512]

HEX = fr'[{hexdigits}]'


PATTERN_TEMPLATE = Template(
	format=r'(?>(?P<md5>{md5})|(?P<sha1>{sha1})|{sha2})')  # atomic


PATTERN_TEMPLATE['md5'] = fr'\b{HEX}{{{{{MD5_SIZE}}}}}\b'
PATTERN_TEMPLATE['sha1'] = fr'\b{HEX}{{{{{SHA1_SIZE}}}}}\b'

PATTERN_TEMPLATE['sha2'] = r'\b(?>{})'.format(
	'|'.join(
		fr'(?P<sha{size}>{HEX}{{{{{int(size/4)}}}}})\b'
		for size in reversed(SHA2_SIZES)
	)
)


def extract(text):
	"""
	Extract hash values from text
	"""
	for match in PATTERN_TEMPLATE.regex.finditer(text):
		logger.debug('Found Hash value: %r', match)
		group_name, = (k for k, v in match.groupdict().items() if v)
		logger.debug('Found %s: %r', group_name, match)
		yield match.group(0)
