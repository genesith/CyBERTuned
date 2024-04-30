"""
Detect and decode data encoded according to the RFC 3548 specification
(base64, base32, base16)
"""
import base64
from functools import partial
from logging import getLogger
import math
from string import ascii_lowercase, ascii_uppercase, digits, hexdigits

import regex

from . import hashes



logger = getLogger(name=__name__)


PAD = '='

MIN_TEXT_LENGTH = 0x10
MAX_TEXT_LENGTH = 0x1000

# items ordered in decreasing exclusivity of pattern:
# base32 has double the alphabet of base16 but 1/4 the rate of acceptable
# lengths
SCHEMES = {
	'base32': {
		'alphabet': ascii_uppercase + '234567',
		'decode': base64.b32decode,
	},
	'base16': {
		'alphabet': digits + 'ABCDEF',
		'decode': base64.b16decode,
	},
	'base64': {
		'alphabet': ascii_uppercase + ascii_lowercase + digits + '+/',
		'decode': partial(base64.b64decode, validate=True),
	},
}


ALT_SCHEMES = {
	**SCHEMES,
	'base64_url_filename': {
		'alphabet': ascii_uppercase + ascii_lowercase + digits + '-_',
		'decode': partial(base64.b64decode, validate=True, altchars='-_'),
	},
}


def get_pattern(
		alphabet,
		min_unencoded_length=MIN_TEXT_LENGTH,
		max_unencoded_length=MAX_TEXT_LENGTH,
		pad=PAD,
):
	"""
	Create regular expression pattern string for RFC-3548-encoded data
	according to the specified requirements
	"""
	CHAR_BITS = 8
	unique_element_count = len(set(alphabet))
	character_bits = math.log2(unique_element_count)
	if character_bits.is_integer():
		character_bits = int(character_bits)
	else:
		raise ValueError(
			f'Invalid encoding alphabet {alphabet!r}:'
			f' number of unique elements {unique_element_count} is not a power'
			' of 2'
		)

	logger.debug('character bits: %r', character_bits)

	quantum_bits = _lowest_common_multiple(character_bits, CHAR_BITS)
	quantum_characters = quantum_bits // character_bits

	logger.debug('quantum character length: %r', quantum_characters)

	min_bits = min_unencoded_length * CHAR_BITS
	min_quanta = min_bits // quantum_bits
	logger.debug('Minimum whole quanta: %r', min_quanta)

	max_bits = max_unencoded_length * CHAR_BITS
	max_quanta = max_bits // quantum_bits
	logger.debug('Maximum whole quanta: %r', max_quanta)

	escaped_alphabet = regex.escape(alphabet)
	logger.debug('Escaped alphabet: %r', escaped_alphabet)

	unpadded = (
		'(?>'
			f'[{escaped_alphabet}]{{{quantum_characters}}}'
		f'){{{min_quanta},{max_quanta}}}'
	)
	logger.debug('Unpadded segment pattern: %r', unpadded)

	padding_options = []
	min_whole_quanta_bits = min_quanta * quantum_bits
	max_whole_quanta_bits = max_quanta * quantum_bits
	logger.debug(
		'Whole quanta bit range: %r-%r',
		min_whole_quanta_bits,
		max_whole_quanta_bits,
	)
	for partial_bits in range(CHAR_BITS, quantum_bits, CHAR_BITS):
		logger.debug('Testing %r-bit partial quanta', partial_bits)
		max_bit_count = max_whole_quanta_bits + partial_bits
		if max_bit_count < min_bits:
			logger.debug(
				'Maximum whole quanta plus partial bits lower than minimum %r',
				min_bits,
			)
			continue

		min_bit_count = min_whole_quanta_bits + partial_bits
		if min_bit_count > max_bits:
			logger.debug(
				'Minimum whole quanta plus partial bits exceeds maximum %r',
				max_bits,
			)
			break

		logger.debug('%r-bit partial quanta is in range', partial_bits)

		padded_bits = quantum_bits - partial_bits
		padding_length = math.floor(padded_bits / character_bits)
		partial_characters = quantum_characters - padding_length
		padding_options.append(
			f'[{escaped_alphabet}]{{{partial_characters}}}'
			f'{pad}{{{padding_length}}}'
		)

	if padding_options:
		pattern = unpadded + '(?>{})?'.format(
			'|'.join(reversed(padding_options)))
	else:
		logger.debug('No padding for specified alphabet')
		pattern = unpadded

	logger.debug('Encoding pattern for alphabet %r: %r', alphabet, pattern)
	return pattern


def _lowest_common_multiple(integer_1, integer_2):
	"""
	Get the lowest common multiple of two integers
	"""
	inputs = []
	for input in [integer_1, integer_2]:
		if isinstance(input, float) and not input.is_integer():
			raise ValueError(f'float input {input} is not an integer value')
		
		inputs.append(int(input))

	smaller, larger = sorted(inputs)
	product = smaller * larger
	multiple = larger
	while multiple < product:
		if multiple % smaller == 0:
			logger.debug(
				'Lowest common multiple of %r and %r is %r',
				integer_1,
				integer_2,
				multiple,
			)
			return multiple

		multiple += larger

	logger.debug(
		'Lowest common multiple of %r and %r is %r (the product)',
		integer_1,
		integer_2,
		product,
	)
	return product


def _get_merged_regex(schemes_data, set_scheme_patterns=False, **kwargs):
	"""
	Get the regular expression matching data encoded according to the specified
	schemes
	"""
	scheme_subpatterns = []
	merged_alphabets = set()
	for scheme, data in schemes_data.items():
		scheme_alphabet = data['alphabet']
		scheme_pattern = get_pattern(alphabet=scheme_alphabet, **kwargs)
		scheme_subpatterns.append(f'(?P<{scheme}>{scheme_pattern})')
		merged_alphabets.update(scheme_alphabet)
		if set_scheme_patterns:
			data['pattern'] = scheme_pattern

	valid_character = r'[{}]'.format(
		regex.escape(''.join(sorted(merged_alphabets)) + PAD))

	choice = '|'.join(fr'{p}(?!{valid_character})' for p in scheme_subpatterns)

	merged_pattern = fr'(?<!{valid_character})(?>{choice})'

	return regex.compile(merged_pattern)


REGEX = _get_merged_regex(schemes_data=SCHEMES, set_scheme_patterns=True)

INCLUDE_ALT_REGEX = _get_merged_regex(
	schemes_data=ALT_SCHEMES, set_scheme_patterns=True)


def parse_match(match, refang=False):
	"""
	Create an embedded binary representation from a regular expression match

	If refang is True, return the binary string
	"""
	blob_text = match.group(0)
	if not refang:
		return blob_text

	scheme, = (k for k, v in match.groupdict().items() if v)
	decode_function = ALT_SCHEMES[scheme]['decode']
	return decode_function(blob_text)


def extract(text, refang=False, include_alt_schemes=False, **pattern_kwargs):
	"""
	Extract RFC-3548-encoded binary blobs from text
	"""
	expression = _get_expression(
		include_alt_schemes=include_alt_schemes, **pattern_kwargs)
	for match in expression.finditer(text):
		match_text = match.group(0)
		if hashes.PATTERN_TEMPLATE.regex.fullmatch(match_text):
			continue

		yield parse_match(match, refang=refang)


def refang(blob_text, include_alt_schemes=False, **pattern_kwargs):
	"""
	Create a binary string from the encoded data blob
	"""
	expression = _get_expression(
		include_alt_schemes=include_alt_schemes, **pattern_kwargs)
	match = expression.fullmatch(blob_text)
	return parse_match(match, refang=True)


def _get_expression(include_alt_schemes=False, **pattern_kwargs):
	"""
	Get the regular expression for the specified parameters
	"""
	if not pattern_kwargs:
		return INCLUDE_ALT_REGEX if include_alt_schemes else REGEX

	return _get_merged_regex(schemes_data=ALT_SCHEMES, **pattern_kwargs)
