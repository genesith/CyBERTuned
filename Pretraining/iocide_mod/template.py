"""
Top-down composition of regular expression patterns
"""
from copy import deepcopy
from functools import lru_cache
import itertools
from logging import getLogger
from urllib.parse import quote

import regex


logger = getLogger(name=__name__)


DEFANG_WRAPPERS = {
	'(': ')',
	'[': ']',
	'{': '}',
	'<': '>',
}

ZERO_WIDTH = chr(0x200b)

DEFANG_CHARACTERS = (
	''.join(o+c for o, c in DEFANG_WRAPPERS.items()) + ZERO_WIDTH)

DEFANG_CHARACTER_REGEX = regex.compile(fr'[{regex.escape(DEFANG_CHARACTERS)}]')


class DefaultFormatMap(dict):
	"""
	Reinsert the format string for the specified key if a value is not present
	"""
	def __getitem__(self, key):
		try:
			return super().__getitem__(key)
		except KeyError:
			return f'{{{key}}}'


class Template:
	"""
	Allows top-down composition of strings by mapping format keys to
	recursively-nested subcomponents
	"""
	def __init__(self, format, normalised=None, data=None):
		self.format_string = str(format)
		data = {} if data is None else data
		self.data = DefaultFormatMap(data)
		self.normalised = normalised
		self.globals = {}

	def __str__(self):
		escaped = self.expand_format(component_map=self.data)
		return escaped.replace(r'{{', r'{').replace(r'}}', r'}')

	def __repr__(self):
		return f'{self.__class__}({self.format_string})'

	def __setitem__(self, key, value):
		if not isinstance(value, self.__class__):
			value = self.__class__(format=value)

		self.data[key] = value

	def __getitem__(self, key):
		return self.data[key]

	def __deepcopy__(self, memo=None):
		format_string = self.format_string
		data = deepcopy(self.data)
		return self.__class__(
			format=format_string, normalised=self.normalised, data=data)

	@property
	def regex(self):
		"""
		A regular expression using the formatted template as a pattern
		"""
		return compile_regex(str(self))

	def expand_format(self, component_map):
		"""
		Map the format string values using the specified component map,
		preserving occurrences of `{{` and `}}`
		"""
		format = self.format_string.replace(r'{{', r'{{{{')
		format = format.replace(r'}}', r'}}}}')
		return format.format_map(component_map)

	def normalise(self, text):
		"""
		Replace text that matches this pattern from the input text with the
		specified normalised representation
		"""
		if self.normalised is None:
			raise ValueError('Missing a normalisation value')

		return self.regex.sub(self.normalised, text)

	@classmethod
	def from_defang_pattern(
			cls,
			pattern,
			normalised,
			openers=None,
			match_unwrapped=True,
			allow_unbalanced=True,
			include_encoded=True,
	):
		"""
		Create a template that matches the specified pattern enclosed in
		various wrapping schemes according to specified requirements
		"""
		if openers is None:
			openers = DEFANG_WRAPPERS.keys()

		components = (
			(o, pattern, c) for o, c in wrapper_patterns(
				openers=openers, include_encoded=include_encoded)
		)

		if match_unwrapped and allow_unbalanced:
			components = ((f'{o}?', p, f'{c}?') for o, p, c in components)
		elif allow_unbalanced:
			components = itertools.chain(
				((f'{o}', p, f'{c}?') for o, p, c in components),
				((f'{o}?', p, f'{c}') for o, p, c in components),
			)
		elif match_unwrapped:
			components = itertools.chain(components, [('', pattern, '')])

		#atomic
		template_format = '(?>{})'.format(
			'|'.join(f'{o}(?>{p}){c}' for o, p, c in components))

		logger.debug('Refanging template format: %r', template_format)
		return cls(format=template_format, normalised=normalised)


def wrapper_patterns(openers, include_encoded):
	"""
	Get opening and closing wrapper patterns
	"""
	for o in openers:
		yield (
			regex.escape(o).replace('{', '{{'),
			regex.escape(DEFANG_WRAPPERS[o]).replace('}', '}}'),
		)
		if not include_encoded:
			continue

		# match against lowercase and uppercase versions of encoded wrappers.
		# Not using (?i)...(?-i) because don't want to unset flags in nested
		# pattern.
		for transform in str.lower, str.upper:
			yield (
				transform(regex.escape(quote(o)).replace('{', '{{')),
				transform(
					regex.escape(quote(DEFANG_WRAPPERS[o])).replace('}', '}}')),
			)


@lru_cache()
def compile_regex(pattern):
	"""
	Cache regex compilation results to avoid recomputing values for each
	regex property access
	"""
	return regex.compile(pattern)
